"""
Base plotting utilities for shared functionality across plotting modules.
"""

import copy
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ..logger import logger

# Color constants used across plotting modules
COLORS = {
    "data": "#d3d3d3",  # lightgray
    "model": "#ff7f0e",  # tab:orange
    "knots": "#d62728",  # tab:red
    "true": "#000000",  # black
    "empirical": "#404040",  # dark gray
    "ci_fill": "#1f77b4",  # tab:blue
    "coherence": "#1f77b4",  # tab:blue
    "real": "#2ca02c",  # tab:green
    "imag": "#ff7f0e",  # tab:orange
}


@dataclass
class PlotConfig:
    """Configuration for plotting parameters."""

    figsize: tuple = (12, 8)
    dpi: int = 150
    fontsize: int = 11
    labelsize: int = 12
    titlesize: int = 12
    linewidth: float = 1.2
    markersize: float = 4.5
    alpha: float = 0.7


def interior_frequency_slice(n_freq: int) -> slice:
    """Return a slice that drops first/last frequency bins when possible."""
    return slice(1, -1) if n_freq > 3 else slice(None)


def _quantiles_from_standard_psd_dataset(
    psd_ds,
) -> Dict[str, np.ndarray | None]:
    """Compute fixed 5/50/95 quantiles from normalized PSD draw datasets."""
    percentiles = np.asarray([5.0, 50.0, 95.0], dtype=float)
    posterior_psd = np.asarray(psd_ds["spectral_density"].values).reshape(
        -1,
        psd_ds["spectral_density"].shape[2],
        psd_ds["spectral_density"].shape[3],
        psd_ds["spectral_density"].shape[4],
    )
    posterior_psd = np.moveaxis(posterior_psd, -1, 1)
    real = np.percentile(posterior_psd.real, percentiles, axis=0)
    imag = np.percentile(posterior_psd.imag, percentiles, axis=0)

    coherence = None
    if "coherence" in psd_ds:
        coherence_samples = np.asarray(psd_ds["coherence"].values).reshape(
            -1,
            psd_ds["coherence"].shape[2],
            psd_ds["coherence"].shape[3],
            psd_ds["coherence"].shape[4],
        )
        coherence_samples = np.moveaxis(coherence_samples, -1, 1)
        coherence = np.percentile(coherence_samples, percentiles, axis=0)

    return {
        "percentile": percentiles,
        "freq": np.asarray(psd_ds.coords["frequency"].values, dtype=float),
        "spectral_density": np.asarray(real + 1j * imag, dtype=np.complex128),
        "coherence": (
            np.asarray(coherence, dtype=np.float64)
            if coherence is not None
            else None
        ),
    }


def extract_plotting_data(
    idata, weights_key: int | None = None
) -> Dict[str, Any]:
    """
    Extract common plotting data from inference data object.

    Args:
        idata: ArviZ InferenceData object
        weights_key: Key for weights in posterior (optional)

    Returns:
        Dictionary containing extracted data
    """
    from ..arviz_utils import (
        get_multivar_prior_psd_quantiles,
        get_periodogram,
        get_psd_dataset,
        get_spline_model,
        get_weights,
    )

    data: Dict[str, Any] = {}

    # Extract core data
    try:
        data["periodogram"] = get_periodogram(idata)
    except KeyError:
        data["periodogram"] = None

    try:
        data["spline_model"] = get_spline_model(idata)
    except KeyError:
        data["spline_model"] = None

    try:
        if isinstance(weights_key, int):
            data["weights"] = get_weights(idata, weights_key)
        else:
            data["weights"] = get_weights(idata)
    except (KeyError, AttributeError):
        data["weights"] = None

    attrs = idata.attrs or {}
    try:
        psd_ds = get_psd_dataset(idata, source="best")
    except (KeyError, TypeError, ValueError, StopIteration):
        psd_ds = None

    if psd_ds is not None:
        quantiles = _quantiles_from_standard_psd_dataset(psd_ds)
        data["frequencies"] = np.asarray(quantiles["freq"], dtype=float)
        n_channels = int(psd_ds["spectral_density"].shape[2])
        if n_channels > 1:
            data["posterior_psd_matrix_quantiles"] = {
                "percentile": np.asarray(quantiles["percentile"], dtype=float),
                "spectral_density": np.asarray(
                    quantiles["spectral_density"], dtype=np.complex128
                ),
                "coherence": (
                    np.asarray(quantiles["coherence"], dtype=np.float64)
                    if quantiles["coherence"] is not None
                    else None
                ),
            }
        else:
            psd_q = np.asarray(
                quantiles["spectral_density"], dtype=np.complex128
            )
            data["posterior_psd_quantiles"] = {
                "percentile": np.asarray(quantiles["percentile"], dtype=float),
                "values": np.asarray(psd_q.real[:, :, 0, 0], dtype=np.float64),
            }

    try:
        vi_psd_ds = get_psd_dataset(idata, source="vi")
    except (KeyError, TypeError, ValueError, StopIteration):
        vi_psd_ds = None
    if vi_psd_ds is not None:
        vi_quantiles = _quantiles_from_standard_psd_dataset(vi_psd_ds)
        n_channels = int(vi_psd_ds["spectral_density"].shape[2])
        if n_channels > 1:
            data["vi_psd_matrix_quantiles"] = {
                "percentile": np.asarray(
                    vi_quantiles["percentile"], dtype=float
                ),
                "spectral_density": np.asarray(
                    vi_quantiles["spectral_density"], dtype=np.complex128
                ),
                "coherence": (
                    np.asarray(vi_quantiles["coherence"], dtype=np.float64)
                    if vi_quantiles["coherence"] is not None
                    else None
                ),
            }
        else:
            vi_psd_q = np.asarray(
                vi_quantiles["spectral_density"], dtype=np.complex128
            )
            data["vi_psd_quantiles"] = {
                "percentile": np.asarray(
                    vi_quantiles["percentile"], dtype=float
                ),
                "values": np.asarray(
                    vi_psd_q.real[:, :, 0, 0], dtype=np.float64
                ),
            }

    if attrs.get("tau") is not None and attrs.get("design_psd") is not None:
        prior_quantiles = get_multivar_prior_psd_quantiles(idata)
        data["prior_psd_matrix_quantiles"] = {
            "percentile": np.asarray(
                prior_quantiles["percentile"], dtype=float
            ),
            "spectral_density": np.asarray(
                prior_quantiles["spectral_density"], dtype=np.complex128
            ),
            "coherence": None,
        }

    # Extract true PSD if available
    if "true_psd" in attrs:
        data["true_psd"] = attrs["true_psd"]

    # Extract frequencies if available
    if "frequencies" in attrs:
        data["frequencies"] = attrs["frequencies"]

    return data


def compute_confidence_intervals(
    samples: np.ndarray,
    quantiles: Tuple[float, float, float] = (16, 50, 84),
    method: str = "percentile",
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence intervals from posterior samples.

    Args:
        samples: Array of posterior samples
        quantiles: Tuple of quantiles to compute (low, median, high)
        method: Method for CI computation ('percentile' or 'uniform')
        alpha: Significance level for uniform CI

    Returns:
        Tuple of (lower_bound, median, upper_bound)
    """
    if method == "percentile":
        ci = np.asarray(
            jnp.percentile(samples, q=jnp.array(quantiles), axis=0)
        )
        return ci[0], ci[1], ci[2]
    elif method == "uniform":
        return _compute_uniform_ci(samples, alpha)
    else:
        raise ValueError(f"Unknown CI method: {method}")


def _compute_uniform_ci(samples: np.ndarray, alpha: float = 0.1):
    """
    Compute uniform (simultaneous) confidence intervals.

    Args:
        samples: Shape (num_samples, num_points) array of function samples
        alpha: Significance level

    Returns:
        Tuple of (lower_bound, median, upper_bound)
    """
    num_samples, num_points = samples.shape

    # Compute pointwise median and standard deviation
    median = jnp.median(samples, axis=0)
    std = jnp.std(samples, axis=0)

    # Compute the max deviation over all samples
    deviations = (samples - median[None, :]) / std[None, :]
    max_deviation = jnp.max(jnp.abs(deviations), axis=1)

    # Compute the scaling factor using the distribution of max deviations
    k_alpha = jnp.percentile(max_deviation, 100 * (1 - alpha))

    # Compute uniform confidence bands
    lower_bound = median - k_alpha * std
    upper_bound = median + k_alpha * std

    return lower_bound, median, upper_bound


def compute_coherence_ci(
    psd_samples: np.ndarray,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute coherence confidence intervals from multivariate PSD samples.

    Args:
        psd_samples: Shape (n_samples, N, p, p)

    Returns:
        Dictionary mapping (i,j) channel pairs to (q05, q50, q95) tuples
    """
    ci_dict = {}
    n_samples, N, p, _ = psd_samples.shape

    for i in range(p):
        for j in range(p):
            if i > j:  # Only compute for upper triangle
                coh = np.abs(psd_samples[:, :, i, j]) ** 2 / (
                    np.abs(psd_samples[:, :, i, i])
                    * np.abs(psd_samples[:, :, j, j])
                )
                q05 = np.percentile(coh, 5, axis=0)
                q50 = np.percentile(coh, 50, axis=0)
                q95 = np.percentile(coh, 95, axis=0)
                ci_dict[(i, j)] = (q05, q50, q95)

    return ci_dict


def compute_cross_spectra_ci(psd_samples: np.ndarray) -> Tuple[Dict, Dict]:
    """
    Compute real and imaginary parts of cross-spectra.

    Args:
        psd_samples: Shape (n_samples, N, p, p)

    Returns:
        Tuple of (real_ci_dict, imag_ci_dict)
    """
    real_dict = {}
    imag_dict = {}
    n_samples, N, p, _ = psd_samples.shape

    for i in range(p):
        for j in range(p):
            if i != j:
                # Real part
                re_q05 = np.percentile(psd_samples[:, :, i, j].real, 5, axis=0)
                re_q50 = np.percentile(
                    psd_samples[:, :, i, j].real, 50, axis=0
                )
                re_q95 = np.percentile(
                    psd_samples[:, :, i, j].real, 95, axis=0
                )
                real_dict[(i, j)] = (re_q05, re_q50, re_q95)

                # Imaginary part
                im_q05 = np.percentile(psd_samples[:, :, i, j].imag, 5, axis=0)
                im_q50 = np.percentile(
                    psd_samples[:, :, i, j].imag, 50, axis=0
                )
                im_q95 = np.percentile(
                    psd_samples[:, :, i, j].imag, 95, axis=0
                )
                imag_dict[(i, j)] = (im_q05, im_q50, im_q95)

    return real_dict, imag_dict


def setup_plot_style(config: Optional[PlotConfig] = None) -> PlotConfig:
    """Setup consistent matplotlib styling for plots."""
    if config is None:
        config = PlotConfig()

    plt.rcParams.update(
        {
            "font.size": config.fontsize,
            "axes.labelsize": config.labelsize,
            "axes.titlesize": config.titlesize,
            "xtick.labelsize": config.fontsize - 1,
            "ytick.labelsize": config.fontsize - 1,
            "legend.fontsize": config.fontsize - 1,
            "axes.linewidth": config.linewidth,
            "xtick.major.width": config.linewidth - 0.1,
            "ytick.major.width": config.linewidth - 0.1,
            "figure.dpi": config.dpi,
            "savefig.dpi": config.dpi * 2,
        }
    )

    return config


def validate_plotting_data(data: Dict[str, Any], required_keys: list) -> bool:
    """Validate that required data is available for plotting."""
    missing_keys = [
        key for key in required_keys if key not in data or data[key] is None
    ]
    if missing_keys:
        logger.warning(f"Missing required data for plotting: {missing_keys}")
        return False
    return True


def subsample_weights(
    weights: np.ndarray, max_samples: int = 500
) -> np.ndarray:
    """Subsample weights array if it's too large for efficient computation."""
    if weights.shape[0] > max_samples:
        idx = np.random.choice(
            weights.shape[0], size=max_samples, replace=False
        )
        return weights[idx]
    return weights
