"""
Base class for multivariate PSD samplers.
"""

import tempfile
from typing import Any, Dict, Tuple

import arviz as az
import jax.numpy as jnp
import morphZ
import numpy as np

from ...datatypes import MultivarFFT
from ...psplines.multivariate import MultivariateLogPSplines
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def whittle_likelihood(
        data: MultivarFFT,
        log_delta_sq: jnp.ndarray,  # (n_freq, n_dim)
        theta_re: jnp.ndarray,  # (n_freq, n_theta)
        theta_im: jnp.ndarray  # (n_freq, n_theta)
) -> jnp.ndarray:
    """Multivariate Whittle likelihood for Cholesky PSD parameterization."""
    sum_log_det = -jnp.sum(log_delta_sq)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)

    if data.Z_re.shape[2] > 0:
        Z_theta_re = jnp.einsum('fij,fj->fi', data.Z_re, theta_re) - jnp.einsum('fij,fj->fi', data.Z_im, theta_im)
        Z_theta_im = jnp.einsum('fij,fj->fi', data.Z_re, theta_im) + jnp.einsum('fij,fj->fi', data.Z_im, theta_re)
        u_re = data.y_re - Z_theta_re
        u_im = data.y_im - Z_theta_im
    else:
        u_re = data.y_re
        u_im = data.y_im

    numerator = u_re ** 2 + u_im ** 2
    internal = numerator * exp_neg_log_delta
    tmp2 = -jnp.sum(internal)
    return sum_log_det + tmp2


class MultivarBaseSampler(BaseSampler):
    """
    Base class for multivariate PSD samplers.

    Handles multi-channel FFT data with MultivariateLogPSplines models.
    """

    def __init__(
            self,
            fft_data: MultivarFFT,
            spline_model: MultivariateLogPSplines,
            config: SamplerConfig
    ):
        # Type hints for clarity
        self.fft_data: MultivarFFT = fft_data
        self.spline_model: MultivariateLogPSplines = spline_model

        super().__init__(fft_data, spline_model, config)

    def _setup_data(self) -> None:
        """Setup multivariate-specific data attributes."""
        self.n_freq = self.fft_data.n_freq
        self.n_channels = self.fft_data.n_dim
        self.n_theta = self.spline_model.n_theta

        # Get all bases and penalties for NumPyro model
        self.all_bases, self.all_penalties = self.spline_model.get_all_bases_and_penalties()

        # FFT data arrays
        self.y_re = jnp.array(self.fft_data.y_re)
        self.y_im = jnp.array(self.fft_data.y_im)
        self.Z_re = jnp.array(self.fft_data.Z_re)
        self.Z_im = jnp.array(self.fft_data.Z_im)
        self.freq = jnp.array(self.fft_data.freq)

    @property
    def data_type(self) -> str:
        return "multivariate"

    def _create_inference_data(
            self,
            samples: Dict[str, jnp.ndarray],
            sample_stats: Dict[str, Any],
            lnz: float,
            lnz_err: float
    ) -> az.InferenceData:
        """Create InferenceData for multivariate case."""
        # TODO: Implement multivariate results_to_arviz equivalent
        # For now, create a basic InferenceData
        return az.from_dict(
            posterior=samples,
            sample_stats=sample_stats,
            attrs=dict(
                device=str(self.device),
                runtime=self.runtime,
                lnz=lnz,
                lnz_err=lnz_err,
                sampler_type=self.sampler_type,
                data_type=self.data_type,
                n_channels=self.n_channels,
                n_freq=self.n_freq,
            )
        )

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save multivariate-specific plots."""
        # TODO: Implement multivariate plotting
        # For now, just create a basic plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Multivariate Results\nChannels: {self.n_channels}",
                ha='center', va='center')
        fig.savefig(f"{self.config.outdir}/multivariate_summary.png")

    def _get_lnz(
            self,
            samples: Dict[str, np.ndarray],
            sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Default implementation for multivariate LnZ computation."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        # TODO: Implement multivariate LnZ computation
        return np.nan, np.nan