"""
Base class for multivariate PSD samplers.
"""

import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import morphZ
import numpy as np

from ...datatypes import MultivarFFT
from ...datatypes.multivar import (
    EmpiricalPSD,
    _get_coherence,
    _interp_complex_matrix,
)
from ...logger import logger
from ...plotting import (
    plot_psd_matrix,
    save_vi_diagnostics_multivariate,
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
        self.y_re = jnp.array(self.fft_data.y_re, dtype=jnp.float32)
        self.y_im = jnp.array(self.fft_data.y_im, dtype=jnp.float32)
        self.freq = jnp.array(self.fft_data.freq, dtype=jnp.float32)
        self.u_re = jnp.array(self.fft_data.u_re, dtype=jnp.float32)
        self.u_im = jnp.array(self.fft_data.u_im, dtype=jnp.float32)
        self.nu = int(self.fft_data.nu)

        # Optional frequency weights (used to scale the log-det term)
        if self.config.freq_weights is not None:
            fw = jnp.asarray(self.config.freq_weights, dtype=jnp.float32)
            if fw.shape[0] != self.n_freq:
                raise ValueError(
                    "Frequency weights length must match number of frequencies"
                )
            self.freq_weights = fw
        else:
            self.freq_weights = jnp.ones((self.n_freq,), dtype=jnp.float32)

        if self.config.verbose:
            logger.info(
                f"Frequency bins used for inference (N): {self.n_freq}"
            )
            basis_shapes = ", ".join(
                [f"{tuple(b.shape)}" for b in self.all_bases]
            )
            logger.info(f"B-spline basis shapes: {basis_shapes}")
            if self.config.freq_weights is not None:
                total = float(jnp.sum(self.freq_weights))
                logger.info(
                    f"Applied coarse-grain weights; total effective count = {total:.1f}"
                )

    @property
    def data_type(self) -> str:
        return "multivariate"

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save multivariate-specific plots."""
        try:
            # Create empirical PSD matrix for comparison
            empirical_psd = self._compute_empirical_psd()
            plot_psd_matrix(
                idata=idata,
                freq=np.array(self.freq),
                empirical_psd=empirical_psd,
                outdir=self.config.outdir,
            )

            self._save_vi_diagnostics(empirical_psd=empirical_psd)
        except Exception as e:
            if self.config.verbose:
                logger.warning(
                    f"Could not create VI plots: {e}, \nFull trace:\n{traceback.format_exc()}"
                )

    def _compute_empirical_psd(self) -> EmpiricalPSD:
        if (
            getattr(self.fft_data, "raw_psd", None) is not None
            and getattr(self.fft_data, "raw_freq", None) is not None
        ):
            freq = np.asarray(self.fft_data.raw_freq, dtype=np.float64)
            psd = np.asarray(self.fft_data.raw_psd, dtype=np.complex128)
            if psd.shape[0] != self.n_freq:
                psd = _interp_complex_matrix(freq, np.array(self.freq), psd)
                freq = np.array(self.freq, dtype=np.float64)
            coherence = _get_coherence(psd)
            channels = np.arange(psd.shape[1])
            return EmpiricalPSD(
                freq=freq, psd=psd, coherence=coherence, channels=channels
            )

        u_re = np.asarray(self.fft_data.u_re, dtype=np.float64)
        u_im = np.asarray(self.fft_data.u_im, dtype=np.float64)
        u_complex = u_re + 1j * u_im
        Y = np.einsum("fkc,fkd->fcd", u_complex, np.conj(u_complex))
        norm_factor = 2 * np.pi
        base_nu = float(max(int(self.fft_data.nu), 1))

        freq_weights = np.asarray(self.freq_weights, dtype=np.float64)
        if freq_weights.shape != (self.n_freq,):
            raise ValueError(
                "Frequency weights must have same length as frequency grid."
            )
        effective_nu = np.clip(freq_weights * base_nu, a_min=1e-12, a_max=None)
        S = (2.0 / effective_nu[:, None, None]) * Y / norm_factor
        S *= float(self.fft_data.scaling_factor or 1.0)
        coherence = _get_coherence(S)
        freq = np.array(self.freq, dtype=np.float64)
        channels = np.arange(S.shape[1])
        return EmpiricalPSD(
            freq=freq, psd=S, coherence=coherence, channels=channels
        )

    def _save_vi_diagnostics(
        self, *, empirical_psd: Optional[EmpiricalPSD] = None
    ) -> None:
        """Persist VI diagnostics if available."""
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if not vi_diag:
            return

        save_vi_diagnostics_multivariate(
            outdir=self.config.outdir,
            freq=np.array(self.freq),
            empirical_psd=empirical_psd,
            diagnostics=vi_diag,
        )

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
