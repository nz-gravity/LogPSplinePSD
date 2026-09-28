"""
Base plotting utilities for shared functionality across plotting modules.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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


def compute_confidence_intervals(
    samples: np.ndarray,
    quantiles: tuple[float, float, float] = (16, 50, 84),
    method: str = "percentile",
    alpha: float = 0.1,
) -> tuple[
    np.ndarray | jnp.ndarray,
    np.ndarray | jnp.ndarray,
    np.ndarray | jnp.ndarray,
]:
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


def setup_plot_style(config: PlotConfig | None = None) -> PlotConfig:
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
