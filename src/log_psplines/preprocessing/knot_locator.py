import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import medfilt, savgol_filter

from log_psplines.data.spectral_utils import psd_to_cholesky_components

_KNOT_TOL = 1e-12


def _dedup_sorted_with_tol(
    knots: np.ndarray, *, tol: float = _KNOT_TOL
) -> np.ndarray:
    """Deduplicate a sorted knot array using a distance tolerance."""
    if knots.size == 0:
        return knots
    uniq = [float(knots[0])]
    for value in knots[1:]:
        if float(value) - float(uniq[-1]) > tol:
            uniq.append(float(value))
            continue
        if abs(float(value)) <= tol:
            uniq[-1] = 0.0
        elif abs(float(value) - 1.0) <= tol:
            uniq[-1] = 1.0
    return np.asarray(uniq, dtype=np.float64)


def _enforce_exact_knot_count(
    knots: np.ndarray, *, target_count: int
) -> np.ndarray:
    """Ensure knots has exactly target_count entries while preserving endpoints."""
    if target_count < 2:
        raise ValueError("target_count must be >= 2")

    knots = np.asarray(knots, dtype=np.float64)
    if knots.size == 0:
        return np.linspace(0.0, 1.0, target_count)
    if knots[0] != 0.0 or knots[-1] != 1.0:
        raise ValueError(
            "knots must include endpoints before count enforcement"
        )

    while knots.size > target_count:
        # Drop the least informative interior knot in the tightest local region.
        gaps = np.diff(knots)
        interior_scores = np.minimum(gaps[:-1], gaps[1:])
        drop_idx = 1 + int(np.argmin(interior_scores))
        knots = np.delete(knots, drop_idx)

    while knots.size < target_count:
        # Add a new knot at the midpoint of the widest interval.
        gaps = np.diff(knots)
        insert_left = int(np.argmax(gaps))
        new_knot = 0.5 * (knots[insert_left] + knots[insert_left + 1])
        knots = np.insert(knots, insert_left + 1, new_knot)

    return knots


def init_knots(
    n_knots: int,
    freqs: np.ndarray,
    power: np.ndarray,
    guide_power: np.ndarray | None = None,
    method: str = "density",
    knots: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Core knot allocation implementation from raw frequency/power arrays."""
    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError(f"freqs must be 1-D, got shape {freqs.shape}")
    if power.ndim != 1:
        raise ValueError(f"power must be 1-D, got shape {power.shape}")
    if freqs.shape[0] != power.shape[0]:
        raise ValueError(
            "freqs and power must have the same length, "
            f"got {freqs.shape[0]} and {power.shape[0]}"
        )
    if freqs.shape[0] == 0:
        raise ValueError("freqs/power must be non-empty.")

    min_freq, max_freq = float(freqs[0]), float(freqs[-1])

    if n_knots == 2:
        return np.array([0.0, 1.0])

    if knots is not None:
        knots = np.array(knots)
    else:
        if method == "uniform":
            knots = np.linspace(min_freq, max_freq, n_knots)

        elif method == "log":
            min_freq_log = max(min_freq, 1e-10)
            knots = np.logspace(
                np.log10(min_freq_log), np.log10(max_freq), n_knots
            )

        elif method == "density":
            knots = _quantile_based_knots(
                n_knots,
                freqs,
                power,
                guide_power=guide_power,
                guide_strength=float(kwargs.get("guide_strength", 1.0)),
            )

        else:
            raise ValueError(f"Unknown knot placement method: {method}")

    # Normalize to [0, 1] and ensure proper ordering
    original_knots = knots.copy()
    knots = np.array(knots, dtype=np.float64)
    knots = np.sort(knots)
    knots = (knots - min_freq) / (max_freq - min_freq)
    knots = np.clip(knots, 0.0, 1.0)
    knots[np.abs(knots) <= _KNOT_TOL] = 0.0
    knots[np.abs(knots - 1.0) <= _KNOT_TOL] = 1.0
    # print if we have some nanas
    if np.isnan(knots).any():
        missing_knots = original_knots[np.isnan(knots)]
        warnings.warn(
            f"Some knots are NaN after normalization. "
            f"Missing knots: {missing_knots}",
            stacklevel=2,
        )
        knots = knots[~np.isnan(knots)]

    # ensure we have knots at ends 0 and 1
    knots = np.concatenate([[0.0], knots, [1.0]])
    knots = np.sort(knots)
    knots = _dedup_sorted_with_tol(knots)
    if knots.size == 0 or knots[0] != 0.0:
        knots = np.concatenate([[0.0], knots])
    else:
        knots[0] = 0.0
    if knots[-1] != 1.0:
        knots = np.concatenate([knots, [1.0]])
    else:
        knots[-1] = 1.0

    # Density-based knots are expected to honor the requested count exactly.
    if method == "density":
        knots = _enforce_exact_knot_count(knots, target_count=int(n_knots))

    return knots


def _adaptive_denoise(signal: np.ndarray) -> np.ndarray:
    """Denoise a 1-D signal for gradient-based knot placement.

    Two-stage, parameter-free pipeline:

    1. **Median filter** (wide, ~5 % of N) — aggressively removes heavy-tailed
       periodogram noise spikes without shifting peaks or transitions.
    2. **Savitzky-Golay filter** (narrower, ~2 % of N, cubic) — smooths the
       residual while preserving the shape, location, and height of genuine
       features.

    The asymmetric widths are intentional: the median pass must be wide enough
    to suppress noise in noisy regions, while the savgol pass stays tight to
    avoid blurring real spectral features.
    """
    n = signal.size
    if n < 5:
        return signal.copy()

    # Stage 1: wide median filter — kills heavy-tailed outlier noise.
    med_win = max(5, n // 20)  # ~5 % of N
    med_win = med_win if med_win % 2 == 1 else med_win + 1
    denoised = medfilt(signal, kernel_size=med_win)

    # Stage 2: tighter Savitzky-Golay — preserves peak shape.
    sg_win = max(5, n // 50)  # ~2 % of N
    sg_win = sg_win if sg_win % 2 == 1 else sg_win + 1
    polyorder = min(3, sg_win - 1)
    denoised = savgol_filter(
        denoised, window_length=sg_win, polyorder=polyorder
    )

    return denoised


def denoise_score(
    signal: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Denoise a score signal using the same pipeline as knot placement.

    Args:
        signal: 1-D score array (may be signed).
        freqs: Corresponding frequency array (unused, kept for API
            compatibility with the preprocessing plot).

    Returns:
        Denoised signal on the original frequency grid.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 5:
        return signal.copy()
    return _adaptive_denoise(signal)


def _quantile_based_knots(
    n_knots: int,
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    guide_power: np.ndarray | None = None,
    guide_strength: float = 1.0,
) -> np.ndarray:
    """Place knots at equal quantiles of a gradient-based spectral feature score.

    Knot density is proportional to the absolute gradient of the denoised
    score signal, plus a small uniform floor so that flat regions still
    receive some knots. When ``guide_power`` is provided, its gradient is
    added as an auxiliary guide signal so known analytical features can pull
    knots toward them without subtracting from or flattening the empirical
    score. All processing is in linear frequency — the space where the
    B-spline basis is evaluated.
    """
    power = np.asarray(power, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)

    n = power.size
    if n < 3:
        return np.linspace(float(freqs[0]), float(freqs[-1]), n_knots)

    smooth = _adaptive_denoise(power)

    gradient = np.abs(np.gradient(smooth, freqs))
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    if guide_power is not None:
        guide = np.asarray(guide_power, dtype=np.float64)
        if guide.shape != power.shape:
            raise ValueError(
                "guide_power must match periodogram power shape, "
                f"got {guide.shape} vs {power.shape}"
            )
        guide_smooth = _adaptive_denoise(guide)
        guide_gradient = np.abs(np.gradient(guide_smooth, freqs))
        guide_gradient = np.nan_to_num(
            guide_gradient, nan=0.0, posinf=0.0, neginf=0.0
        )
        guide_scale = float(np.mean(guide_gradient))
        grad_scale = float(np.mean(gradient))
        if guide_scale > 0.0:
            if grad_scale > 0.0:
                guide_gradient = guide_gradient * (grad_scale / guide_scale)
            gradient = gradient + float(guide_strength) * guide_gradient

    # Uniform floor so featureless regions still get some knots.
    signal_scale = float(np.mean(np.abs(smooth)))
    floor = 0.01 * signal_scale if signal_scale > 0.0 else 1.0
    z = gradient + floor
    z = z / z.sum()

    cdf = np.cumsum(z)
    cdf = np.insert(cdf, 0, 0.0)
    freqs_ext = np.insert(freqs, 0, freqs[0])

    quantiles = np.linspace(0, 1, n_knots)
    knots = np.interp(quantiles, cdf, freqs_ext)

    return knots


def multivar_psd_knot_scores(
    Y_np: np.ndarray,
    Nb: int,
    p: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Compute per-component knot scores from an empirical PSD matrix.

    Scores are the model-native Cholesky components of the empirical PSD
    (``Y_np / Nb``), returned as signed (not absolute) values so that
    downstream gradient-based knot placement sees genuine shape transitions
    rather than artificial kinks at zero crossings:

    - diagonal scores: ``log_delta_sq`` (signed, real)
    - off-diagonal real scores: ``real(theta)`` (signed)
    - off-diagonal imaginary scores: ``imag(theta)`` (signed)

    Args:
        Y_np: (N, p, p) complex Wishart matrix (sum of outer products).
        Nb: Number of blocks used to form Y_np.
        p: Number of channels.

    Returns:
        diagonal_scores: List of p arrays of shape (N,), one per channel.
        offdiag_re_scores: List of arrays of shape (N,), one per
            theta_re_{j,l} component in lower-triangular order.
        offdiag_im_scores: List of arrays of shape (N,), one per
            theta_im_{j,l} component in lower-triangular order.
    """
    log_delta_sq, theta = psd_to_cholesky_components(Y_np / max(int(Nb), 1))
    diagonal_scores = [log_delta_sq[:, i].copy() for i in range(p)]
    offdiag_re_scores = [
        np.real(theta[:, i, j]).copy() for i in range(1, p) for j in range(i)
    ]
    offdiag_im_scores = [
        np.imag(theta[:, i, j]).copy() for i in range(1, p) for j in range(i)
    ]

    return diagonal_scores, offdiag_re_scores, offdiag_im_scores


@dataclass(frozen=True)
class Component:
    """A smooth pilot component with coordinate axes in array order."""

    values: np.ndarray
    coordinates: Mapping[str, np.ndarray]


def variation_profiles(
    component: Component, aggregation: str = "rms"
) -> dict[str, np.ndarray]:
    """Marginal absolute derivatives in each supplied coordinate.

    The pilot values and coordinates must be finite. For a surface,
    derivatives are aggregated over the other axis using RMS by default.
    """
    values = np.asarray(component.values, dtype=float)
    if (
        values.ndim != len(component.coordinates)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "component needs finite values and one coordinate per axis"
        )
    if aggregation not in {"rms", "median", "q90"}:
        raise ValueError("aggregation must be 'rms', 'median', or 'q90'")

    profiles: dict[str, np.ndarray] = {}
    for axis, (name, coordinate) in enumerate(component.coordinates.items()):
        x = np.asarray(coordinate, dtype=float)
        if (
            x.ndim != 1
            or x.size != values.shape[axis]
            or x.size < 3
            or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError(
                f"{name} must be finite, increasing, and match its axis"
            )
        derivative = np.abs(np.gradient(values, x, axis=axis, edge_order=2))
        other_axes = tuple(i for i in range(values.ndim) if i != axis)
        if other_axes:
            if aggregation == "rms":
                derivative = np.sqrt(
                    np.mean(derivative * derivative, axis=other_axes)
                )
            elif aggregation == "median":
                derivative = np.median(derivative, axis=other_axes)
            else:
                derivative = np.quantile(derivative, 0.9, axis=other_axes)
        profiles[name] = np.asarray(derivative, dtype=float)
    return profiles


def quantile_knots(
    coordinate: np.ndarray,
    variation: np.ndarray,
    count: int,
    *,
    min_spacing: float,
    floor_fraction: float = 0.1,
    power: float = 1.0,
) -> np.ndarray:
    """Interior knots at quantiles of component variation plus a uniform floor.

    ``min_spacing`` is in coordinate units and includes the end intervals.
    The caller supplies a smooth training-only pilot and a spacing tied to
    the analysis resolution.
    """
    x = np.asarray(coordinate, dtype=float)
    variation = np.asarray(variation, dtype=float)
    if (
        x.ndim != 1
        or x.size < 2
        or variation.shape != x.shape
        or not np.isfinite(x).all()
        or not np.isfinite(variation).all()
        or np.any(np.diff(x) <= 0)
        or np.any(variation < 0)
    ):
        raise ValueError(
            "invalid coordinates or nonnegative variation profile"
        )
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count < 0
        or not np.isfinite(min_spacing)
        or min_spacing <= 0
        or not np.isfinite(floor_fraction)
        or not 0 < floor_fraction <= 1
        or not np.isfinite(power)
        or power <= 0
    ):
        raise ValueError("invalid knot count, spacing, floor, or power")

    count = int(count)
    span = float(x[-1] - x[0])
    if (count + 1) * min_spacing > span * (1 + 1e-12):
        raise ValueError("requested knots exceed the analysis spacing limit")
    if count == 0:
        return np.empty(0, dtype=float)

    scaled_variation = variation**power
    mass = float(trapezoid(scaled_variation, x))
    density = (
        np.ones_like(x)
        if mass <= 0
        else floor_fraction
        + (1 - floor_fraction) * scaled_variation * span / mass
    )
    cdf = cumulative_trapezoid(density, x, initial=0)
    knots = np.interp(np.arange(1, count + 1) / (count + 1), cdf / cdf[-1], x)
    for i in range(count):
        knots[i] = max(knots[i], (knots[i - 1] if i else x[0]) + min_spacing)
    for i in range(count - 1, -1, -1):
        knots[i] = min(
            knots[i], (knots[i + 1] if i + 1 < count else x[-1]) - min_spacing
        )
    return knots


def allocate_components(
    components: Mapping[str, Component],
    counts: Mapping[str, Mapping[str, int]],
    spacings: Mapping[str, Mapping[str, float]],
    *,
    aggregation: str = "rms",
    floor_fraction: float = 0.1,
    power: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Allocate independent knot arrays for each named pilot component."""
    if set(components) != set(counts) or set(components) != set(spacings):
        raise ValueError(
            "components, counts, and spacings must have the same keys"
        )
    allocated: dict[str, dict[str, np.ndarray]] = {}
    for name, component in components.items():
        profiles = variation_profiles(component, aggregation=aggregation)
        if set(profiles) != set(counts[name]) or set(profiles) != set(
            spacings[name]
        ):
            raise ValueError(
                f"counts and spacings for {name} must cover its axes"
            )
        allocated[name] = {
            axis: quantile_knots(
                coordinate,
                profiles[axis],
                counts[name][axis],
                min_spacing=spacings[name][axis],
                floor_fraction=floor_fraction,
                power=power,
            )
            for axis, coordinate in component.coordinates.items()
        }
    return allocated
