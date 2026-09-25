"""
Base plotting utilities for shared functionality across plotting modules.
"""

from dataclasses import dataclass
from typing import Any

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


def _quantiles_from_standard_psd_dataset(
    psd_ds,
) -> dict[str, np.ndarray | None]:
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


def _as_matrix_quantiles(
    quantiles: dict[str, np.ndarray | None],
) -> dict[str, np.ndarray | None]:
    """Convert univariate quantiles into the matrix-quantile shape."""
    spectral_density = np.asarray(
        quantiles["spectral_density"], dtype=np.complex128
    )
    return {
        "percentile": np.asarray(quantiles["percentile"], dtype=float),
        "spectral_density": spectral_density,
        "coherence": (
            np.asarray(quantiles["coherence"], dtype=np.float64)
            if quantiles["coherence"] is not None
            else None
        ),
    }


def extract_plotting_data(
    result, weights_key: int | None = None
) -> dict[str, Any]:
    """Extract plotting inputs from a PSDResult."""
    data: dict[str, Any] = {}

    posterior = result.posterior
    if "weights" in posterior:
        weights = posterior["weights"].values
    else:
        name = next(
            (
                str(key)
                for key in posterior.data_vars
                if str(key).startswith("weights_delta_")
            ),
            None,
        )
        weights = None if name is None else posterior[name].values
    if weights is not None:
        weights = np.asarray(weights).reshape(-1, *np.asarray(weights).shape[2:])
        if isinstance(weights_key, int):
            weights = weights[::weights_key]
    data["weights"] = weights

    q = result.quantiles((5.0, 50.0, 95.0))
    if "time" not in q.dims:
        q = q.transpose(
            "percentile", "frequency", "channel", "channel_aux"
        )
        coh = np.asarray(result.coherence)
        coh = coh.reshape(-1, *coh.shape[2:])
        coh_q = np.percentile(coh, [5.0, 50.0, 95.0], axis=0)
        data["frequencies"] = result.frequency
        data["posterior_psd_matrix_quantiles"] = {
            "percentile": np.asarray([5.0, 50.0, 95.0]),
            "spectral_density": np.asarray(q.values),
            "coherence": coh_q,
        }

    if result.vi_spectrum is not None and result.time is None:
        values = np.asarray(result.vi_spectrum)
        flat = values.reshape(-1, *values.shape[2:])
        vi_q = np.percentile(flat.real, [5, 50, 95], axis=0) + 1j * np.percentile(
            flat.imag, [5, 50, 95], axis=0
        )
        vi_coh = np.asarray(
            __import__("log_psplines.models.matrix", fromlist=["SpectralMatrix"])
            .SpectralMatrix.coherence(values)
        ).reshape(-1, *values.shape[2:])
        data["vi_psd_matrix_quantiles"] = {
            "percentile": np.asarray([5.0, 50.0, 95.0]),
            "spectral_density": vi_q,
            "coherence": np.percentile(vi_coh, [5, 50, 95], axis=0),
        }

    if "true_psd" in result.metadata:
        data["true_psd"] = result.metadata["true_psd"]
    return data

def compute_confidence_intervals(
    samples: np.ndarray,
    quantiles: tuple[float, float, float] = (16, 50, 84),
    method: str = "percentile",
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
