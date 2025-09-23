"""
Base class for multivariate PSD samplers.
"""

import tempfile
from typing import Any, Dict, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import morphZ
import numpy as np
from ...datatypes import MultivarFFT
from ...plotting import plot_psd_matrix, compute_empirical_psd
from ...psplines.multivar_psplines import MultivariateLogPSplines
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def batch_spline_eval(basis: jnp.ndarray, weights_batch: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled batch spline evaluation over multiple weight vectors.

    Args:
        basis: Basis matrix (n_freq, n_basis)
        weights_batch: Batch of weight vectors (n_samples, n_basis)

    Returns:
        Batch of spline evaluations (n_samples, n_freq)
    """
    return jnp.sum(basis[None, :, :] * weights_batch[:, None, :], axis=-1)


@jax.jit
def whittle_likelihood(
        y_re: jnp.ndarray,  # (n_freq, n_dim)
        y_im: jnp.ndarray,  # (n_freq, n_dim)
        Z_re: jnp.ndarray,  # (n_freq, n_dim, n_theta)
        Z_im: jnp.ndarray,  # (n_freq, n_dim, n_theta)
        log_delta_sq: jnp.ndarray,  # (n_freq, n_dim)
        theta_re: jnp.ndarray,  # (n_freq, n_theta)
        theta_im: jnp.ndarray  # (n_freq, n_theta)
) -> jnp.ndarray:
    """Multivariate Whittle likelihood for Cholesky PSD parameterization."""
    sum_log_det = -jnp.sum(log_delta_sq)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)

    if Z_re.shape[2] > 0:
        Z_theta_re = jnp.einsum('fij,fj->fi', Z_re, theta_re) - jnp.einsum('fij,fj->fi', Z_im, theta_im)
        Z_theta_im = jnp.einsum('fij,fj->fi', Z_re, theta_im) + jnp.einsum('fij,fj->fi', Z_im, theta_re)
        u_re = y_re - Z_theta_re
        u_im = y_im - Z_theta_im
    else:
        u_re = y_re
        u_im = y_im

    numerator = u_re ** 2 + u_im ** 2
    internal = numerator * exp_neg_log_delta
    tmp2 = -jnp.sum(internal)
    return sum_log_det + tmp2


class MultivarBaseSampler(BaseSampler):
    """
    Base class for multivariate PSD samplers.

    Handles multi-channel FFT data with MultivariateLogPSplines models using
    Cholesky parameterization for cross-spectral density estimation.
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

        # FFT data arrays for JAX operations
        self.y_re = jnp.array(self.fft_data.y_re)
        self.y_im = jnp.array(self.fft_data.y_im)
        self.Z_re = jnp.array(self.fft_data.Z_re)
        self.Z_im = jnp.array(self.fft_data.Z_im)
        self.freq = jnp.array(self.fft_data.freq)

        # Create MultivarFFT object for JAX functions
        self.data_jax = MultivarFFT(
            y_re=self.y_re,
            y_im=self.y_im,
            Z_re=self.Z_re,
            Z_im=self.Z_im,
            freq=self.freq,
            n_freq=self.n_freq,
            n_dim=self.n_channels
        )

    @property
    def data_type(self) -> str:
        return "multivariate"

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save multivariate-specific plots."""
        try:
            # Create empirical PSD matrix for comparison
            empirical_psd = compute_empirical_psd(
                fft_data_re=np.array(self.fft_data.y_re),
                fft_data_im=np.array(self.fft_data.y_im),
                n_channels=self.n_channels
            )

            plot_psd_matrix(
                idata=idata,
                n_channels=self.n_channels,
                freq=np.array(self.freq),
                empirical_psd=empirical_psd,
                outdir=self.config.outdir
            )
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not create plots: {e}")

    def _get_lnz(
            self,
            samples: Dict[str, np.ndarray],
            sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Multivariate LnZ computation using morphZ."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        try:
            # Flatten all samples into a single array
            all_samples = []
            param_sizes = []
            param_names = []

            for name, sample_array in samples.items():
                if name.startswith(('weights_', 'delta_', 'phi_')):
                    flat_samples = sample_array.reshape(sample_array.shape[0], -1)
                    all_samples.append(flat_samples)
                    param_sizes.append(flat_samples.shape[1])
                    param_names.append(name)

            if not all_samples:
                return np.nan, np.nan

            posterior_samples = np.concatenate(all_samples, axis=1)
            lposterior = sample_stats.get("log_likelihood", sample_stats.get("lp", None))

            if lposterior is None:
                return np.nan, np.nan

            def log_posterior_fn(params):
                """Reconstruct log posterior from flattened parameters."""
                # This is a simplified version - you'd need to implement the full
                # multivariate log posterior computation here
                return 0.0  # Placeholder

            # Compute evidence using morphZ
            lnz_res = morphZ.evidence(
                posterior_samples,
                lposterior,
                log_posterior_fn,
                morph_type='pair',
                kde_bw="cv_iso",
                output_path=tempfile.gettempdir()
            )[0]

            return float(lnz_res.lnz), float(lnz_res.uncertainty)

        except Exception as e:
            if self.config.verbose:
                print(f"Warning: LnZ computation failed: {e}")
            return np.nan, np.nan
