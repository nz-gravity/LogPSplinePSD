import warnings

import numpy as np

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


def _quantile_based_knots(
    n_knots: int,
    periodogram: Periodogram,
    parametric_model: np.ndarray | None = None,
) -> np.ndarray:
    """
    Implement Patricio's quantile-based knot placement method.

    The procedure follows these steps:
    1. Take square root of periodogram values
    2. Standardize the values
    3. Take absolute values and normalize to create a PMF
    4. Interpolate to get a continuous CDF
    5. Place knots at equally spaced quantiles of this CDF
    """
    # Step 1: Square root transformation
    x = np.sqrt(periodogram.power)

    # Optionally subtract parametric model
    if parametric_model is not None:
        # Subtract from power, then take square root
        power_adjusted = periodogram.power - parametric_model
        # Ensure positivity
        power_adjusted = power_adjusted + np.abs(np.min(power_adjusted))
        x = np.sqrt(power_adjusted)

    # Step 2: Standardize
    x_mean = np.mean(x)
    x_std = np.std(x)
    if not np.isfinite(x_std) or x_std <= 0.0:
        y = np.zeros_like(x)
    else:
        y = (x - x_mean) / x_std

    # Step 3: Absolute values and normalize to create PMF
    z = np.abs(y)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    total = np.sum(z)
    if total <= 0:
        z = np.ones_like(z) / z.size
    else:
        z = z / total

    # Step 4: Create cumulative distribution function
    cdf_values = np.cumsum(z)
    cdf_values = np.insert(cdf_values, 0, 0.0)
    freqs = np.insert(periodogram.freqs, 0, periodogram.freqs[0])

    # Step 5: Place knots at equally spaced quantiles
    # We want n_knots total, including endpoints
    quantiles = np.linspace(0, 1, n_knots)

    # Interpolate to find frequencies corresponding to these quantiles
    knots = np.interp(quantiles, cdf_values, freqs)

    return knots


def multivar_psd_knot_scores(
    Y_np: np.ndarray,
    Nb: int,
    p: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compute per-component knot placement scores from an empirical PSD matrix.

    Cholesky-decomposes the empirical PSD (Y_np / Nb) to extract the
    model-native components (LogDelta, ReTheta, ImTheta) and returns their
    absolute values as scores for quantile-based knot placement.

    Args:
        Y_np: (N, p, p) complex Wishart matrix (sum of outer products).
        Nb: Number of blocks used to form Y_np.
        p: Number of channels.

    Returns:
        diagonal_scores: List of p arrays of shape (N,), one per channel.
            Score for channel i is |log(delta_i^2)| from the Cholesky diagonal.
        offdiag_score: (N,) array — mean of |theta_ij| over all lower-triangle
            pairs, used as a shared score for Re/Im off-diagonal components.
    """
    log_delta_sq, theta = psd_to_cholesky_components(Y_np / max(int(Nb), 1))

    diagonal_scores = [np.abs(log_delta_sq[:, i]) for i in range(p)]

    if p > 1:
        pair_scores = [
            np.abs(theta[:, i, j]) for i in range(1, p) for j in range(i)
        ]
        offdiag_score = np.mean(np.vstack(pair_scores), axis=0)
    else:
        offdiag_score = np.zeros(Y_np.shape[0], dtype=np.float64)

    return diagonal_scores, offdiag_score
