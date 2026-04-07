import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt, savgol_filter

from ...datatypes import Periodogram
from ...datatypes.multivar_utils import psd_to_cholesky_components
from .lvk_knot_allocator import LvkKnotAllocator

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
    periodogram: Periodogram,
    parametric_model: np.ndarray | None = None,
    method: str = "density",
    knots: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Select knots using various placement strategies.

    Parameters
    ----------
    n_knots : int
        Total number of knots to select
    periodogram : Periodogram
        Periodogram object with freqs and power
    parametric_model : jnp.ndarray, optional
        Parametric model to subtract from power before knot placement
    method : str, default="density"
        Knot placement method:
        - "uniform": Uniformly spaced knots
        - "log": Logarithmically spaced knots
        - "density": Quantile-based placement using periodogram (Patricio's method)
        - "lvk": LVK-specific method

    Returns
    -------
    np.ndarray
        Array of knot locations normalized to [0, 1]
    """
    return _init_knots_from_arrays(
        n_knots=n_knots,
        freqs=np.asarray(periodogram.freqs),
        power=np.asarray(periodogram.power),
        parametric_model=parametric_model,
        method=method,
        knots=knots,
        **kwargs,
    )


def _init_knots_from_arrays(
    n_knots: int,
    freqs: np.ndarray,
    power: np.ndarray,
    parametric_model: np.ndarray | None = None,
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
            periodogram = Periodogram(freqs=freqs, power=power)
            knots = _quantile_based_knots(
                n_knots, periodogram, parametric_model
            )

        elif method == "lvk":
            knot_alloc = LvkKnotAllocator(
                freqs=freqs,
                psd=power,
                fmin=min_freq,
                fmax=max_freq,
                **kwargs,
            )
            knots = knot_alloc.knots_hz

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
            f"Missing knots: {missing_knots}"
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
    periodogram: Periodogram,
    parametric_model: np.ndarray | None = None,
) -> np.ndarray:
    """Place knots at equal quantiles of a gradient-based spectral feature score.

    Knot density is proportional to the absolute gradient of the denoised
    score signal, plus a small uniform floor so that flat regions still
    receive some knots.  All processing is in linear frequency — the space
    where the B-spline basis is evaluated.
    """
    power = np.asarray(periodogram.power, dtype=np.float64)
    freqs = np.asarray(periodogram.freqs, dtype=np.float64)

    if parametric_model is not None:
        power = power - parametric_model
        power = power + np.abs(np.min(power))

    n = power.size
    if n < 3:
        return np.linspace(float(freqs[0]), float(freqs[-1]), n_knots)

    smooth = _adaptive_denoise(power)

    gradient = np.abs(np.gradient(smooth, freqs))
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

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
