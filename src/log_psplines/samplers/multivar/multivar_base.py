"""
Base class for multivariate PSD samplers.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import morphZ
import numpy as np

from ...datatypes import MultivarFFT
from ...logger import logger
from ...plotting import (
    plot_psd_matrix,
    plot_vi_elbo,
    plot_vi_initial_psd_matrix,
)
from ...psplines.multivar_psplines import MultivariateLogPSplines
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def batch_spline_eval(
    basis: jnp.ndarray, weights_batch: jnp.ndarray
) -> jnp.ndarray:
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
    theta_im: jnp.ndarray,  # (n_freq, n_theta)
) -> jnp.ndarray:
    """Multivariate Whittle likelihood for Cholesky PSD parameterization."""
    sum_log_det = -jnp.sum(log_delta_sq)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)

    if Z_re.shape[2] > 0:
        Z_theta_re = jnp.einsum("fij,fj->fi", Z_re, theta_re) - jnp.einsum(
            "fij,fj->fi", Z_im, theta_im
        )
        Z_theta_im = jnp.einsum("fij,fj->fi", Z_re, theta_im) + jnp.einsum(
            "fij,fj->fi", Z_im, theta_re
        )
        u_re = y_re - Z_theta_re
        u_im = y_im - Z_theta_im
    else:
        u_re = y_re
        u_im = y_im

    numerator = u_re**2 + u_im**2
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
        config: SamplerConfig,
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
        all_bases, all_penalties = (
            self.spline_model.get_all_bases_and_penalties()
        )
        self.all_bases = tuple(
            jnp.asarray(basis, dtype=jnp.float32) for basis in all_bases
        )
        self.all_penalties = all_penalties

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
            n_dim=self.n_channels,
        )

    @property
    def data_type(self) -> str:
        return "multivariate"

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save multivariate-specific plots."""
        try:
            # Create empirical PSD matrix for comparison
            empirical_psd = MultivarFFT.get_empirical_psd(
                np.array(self.fft_data.y_re, dtype=np.float64),
                np.array(self.fft_data.y_im, dtype=np.float64),
            )
            plot_psd_matrix(
                idata=idata,
                n_channels=self.n_channels,
                freq=np.array(self.freq),
                empirical_psd=empirical_psd,
                outdir=self.config.outdir,
            )

            import os

            vi_diag = getattr(self, "_vi_diagnostics", None)
            if vi_diag and self.config.outdir is not None:
                outdir = self.config.outdir
                diagnostics_dir = os.path.join(outdir, "diagnostics")
                losses = vi_diag.get("losses")
                if losses is not None:
                    losses_arr = np.asarray(losses)
                    if losses_arr.ndim > 1:
                        losses_arr = losses_arr.mean(axis=0)
                    if losses_arr.size:
                        plot_vi_elbo(
                            losses=losses_arr,
                            guide_name=vi_diag.get("guide", "vi"),
                            outfile=os.path.join(
                                diagnostics_dir, "vi_elbo_trace.png"
                            ),
                        )

                vi_psd = vi_diag.get("psd_matrix")
                if vi_psd is not None:
                    plot_vi_initial_psd_matrix(
                        outfile=os.path.join(
                            diagnostics_dir, "vi_initial_psd_matrix.png"
                        ),
                        freq=np.array(self.freq),
                        vi_psd=vi_psd,
                        empirical_psd=empirical_psd,
                        true_psd=vi_diag.get("true_psd"),
                        psd_quantiles=vi_diag.get("psd_quantiles"),
                        coherence_quantiles=vi_diag.get("coherence_quantiles"),
                    )
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"Could not create plots: {e}")

    def _get_lnz(
        self, samples: Dict[str, np.ndarray], sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Multivariate LnZ computation using morphZ."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        # Temporarily disabled until multivariate morphZ path is stabilised.
        if self.config.verbose:
            logger.warning(
                "LnZ computation is not yet supported for multivariate samplers; returning NaN."
            )
        return np.nan, np.nan

        # --- Previous implementation (kept for future reference) ---
        # try:
        #     parameter_items = [
        #         (name, np.asarray(array, dtype=np.float64))
        #         for name, array in samples.items()
        #         if name.startswith(("weights_", "phi_", "delta_"))
        #     ]
        #
        #     if not parameter_items:
        #         return np.nan, np.nan
        #
        #     n_draws = parameter_items[0][1].shape[0]
        #
        #     flat_blocks = []
        #     layout = []  # (name, shape)
        #
        #     for name, array in parameter_items:
        #         if array.shape[0] != n_draws:
        #             raise ValueError(
        #                 f"Sample array {name} has inconsistent draw dimension."
        #             )
        #         shape = array.shape[1:]
        #         layout.append((name, shape))
        #         flat_blocks.append(array.reshape(n_draws, -1))
        #
        #     posterior_samples = np.concatenate(flat_blocks, axis=1)
        #
        #     lp = sample_stats.get("lp")
        #     if lp is None:
        #         raise ValueError("Sample stats missing 'lp' for LnZ computation.")
        #
        #     lp = np.asarray(lp, dtype=np.float64)
        #     if lp.ndim > 1:
        #         lp = lp.reshape(n_draws, -1)[:, 0]
        #     else:
        #         lp = lp.reshape(-1)
        #
        #     def log_posterior_fn(sample_vec: np.ndarray) -> float:
        #         offset = 0
        #         params: Dict[str, jnp.ndarray] = {}
        #         for name, shape in layout:
        #             size = int(np.prod(shape)) if shape else 1
        #             segment = sample_vec[offset : offset + size]
        #             offset += size
        #             if shape:
        #                 params[name] = jnp.asarray(segment.reshape(shape))
        #             else:
        #                 params[name] = jnp.asarray(segment.item())
        #         return self._compute_log_posterior(params)
        #
        #     lnz_res = morphZ.evidence(
        #         posterior_samples,
        #         lp,
        #         log_posterior_fn,
        #         morph_type="pair",
        #         kde_bw="scott",
        #         output_path=tempfile.gettempdir(),
        #     )[0]
        #
        #     return float(lnz_res.lnz), float(lnz_res.uncertainty)
        # except Exception as e:
        #     if self.config.verbose:
        #         print(f"Warning: LnZ computation failed: {e}")
        #     return np.nan, np.nan

    def _compute_log_posterior(self, params: Dict[str, jnp.ndarray]) -> float:
        """Compute log posterior for LnZ calculation (implemented by subclasses)."""
        raise NotImplementedError
