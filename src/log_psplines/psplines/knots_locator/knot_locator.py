import warnings

import numpy as np

from ...datatypes import Periodogram
from .lvk_knot_allocator import LvkKnotAllocator


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
    unique_knots, counts = np.unique(knots, return_counts=True)

    return unique_knots


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
