"""
Base plotting utilities for shared functionality across plotting modules.
"""

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


def safe_plot(filename: str, dpi: int = 150):
    """Decorator for safe plotting with error handling."""

    def decorator(plot_func: Callable):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"--- plotting: {os.path.basename(filename)}")
                result = plot_func(*args, **kwargs)
                plt.savefig(filename, dpi=dpi, bbox_inches="tight")
                plt.close()
                return True
            except Exception as e:
                logger.warning(
                    f"Failed to create {os.path.basename(filename)}: {e}"
                )
                plt.close("all")
                return False

        return wrapper

    return decorator


def extract_plotting_data(idata, weights_key: str = None) -> Dict[str, Any]:
    """
    Extract common plotting data from inference data object.

    Args:
        idata: ArviZ InferenceData object
        weights_key: Key for weights in posterior (optional)

    Returns:
        Dictionary containing extracted data
    """
    from ..arviz_utils import (
        get_periodogram,
        get_spline_model,
        get_weights,
    )

    data = {}

    # Extract core data - handle univariate, multivariate, and VI cases
    try:
        data["periodogram"] = get_periodogram(idata)
    except KeyError:
        # For multivariate or VI data, periodogram might not be available
        # or might be stored differently
        data["periodogram"] = None

    # Extract spline model - handle VI case where 'knots' might not exist
    try:
        data["spline_model"] = get_spline_model(idata)
    except KeyError:
        # For VI data, spline model might be stored differently
        data["spline_model"] = None

    # Extract weights - handle different data structures
    try:
        data["weights"] = get_weights(idata, weights_key)
    except (KeyError, AttributeError):
        # For VI data, weights might be stored differently
        data["weights"] = None

    # Extract posterior samples if available
    if hasattr(idata, "posterior_psd"):
        if "psd" in idata.posterior_psd:
            data["posterior_psd"] = idata.posterior_psd.psd.values
        if "psd_matrix" in idata.posterior_psd:
            data["posterior_psd_matrix"] = (
                idata.posterior_psd.psd_matrix.values
            )

    # Extract true PSD if available
    if hasattr(idata, "attrs") and "true_psd" in idata.attrs:
        data["true_psd"] = idata.attrs["true_psd"]

    # Extract frequencies if available
    if hasattr(idata, "attrs") and "frequencies" in idata.attrs:
        data["frequencies"] = idata.attrs["frequencies"]

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
        return jnp.percentile(samples, q=jnp.array(quantiles), axis=0)
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
        psd_samples: Shape (n_samples, n_freq, n_channels, n_channels)

    Returns:
        Dictionary mapping (i,j) channel pairs to (q05, q50, q95) tuples
    """
    ci_dict = {}
    n_samples, n_freq, n_channels, _ = psd_samples.shape

    for i in range(n_channels):
        for j in range(n_channels):
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
        psd_samples: Shape (n_samples, n_freq, n_channels, n_channels)

    Returns:
        Tuple of (real_ci_dict, imag_ci_dict)
    """
    real_dict = {}
    imag_dict = {}
    n_samples, n_freq, n_channels, _ = psd_samples.shape

    for i in range(n_channels):
        for j in range(n_channels):
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
