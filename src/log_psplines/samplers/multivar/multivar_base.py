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
from xarray import DataArray, Dataset

from ...datatypes import MultivarFFT
from ...datatypes.multivar import (
    EmpiricalPSD,
    _get_coherence,
    _interp_complex_matrix,
)
from ...logger import logger
from ...plotting import (
    generate_vi_diagnostics_summary,
    plot_psd_matrix,
    save_vi_diagnostics_multivariate,
)
from ...psplines.multivar_psplines import MultivariateLogPSplines
from ...spectrum_utils import u_to_wishart_matrix, wishart_matrix_to_psd
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def batch_spline_eval(
    basis: jnp.ndarray, weights_batch: jnp.ndarray
) -> jnp.ndarray:
    """JIT-compiled batch spline evaluation over multiple weight vectors.

    Args:
        basis: Basis matrix (N, n_basis)
        weights_batch: Batch of weight vectors (n_samples, n_basis)

    Returns:
        Batch of spline evaluations (n_samples, N)
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
        self.N = self.fft_data.N
        self.p = self.fft_data.p
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
        self.Nb = int(self.fft_data.Nb)
        self.duration = float(getattr(self.fft_data, "duration", 1.0) or 1.0)
        if self.duration <= 0.0:
            raise ValueError("fft_data.duration must be positive")

        # For equal-sized coarse bins, a single Nh scalar is sufficient.
        self.Nh = float(getattr(self.fft_data, "Nh", 1.0) or 1.0)
        if self.Nh <= 0.0:
            raise ValueError("fft_data.Nh must be positive")

        if self.config.verbose:
            logger.info(f"Frequency bins used for inference (N): {self.N}")
            basis_shapes = ", ".join(
                [f"{tuple(b.shape)}" for b in self.all_bases]
            )
            logger.info(f"B-spline basis shapes: {basis_shapes}")
            total = float(self.N * self.Nh)
            logger.info(
                f"Applied coarse-grain counts; total effective count = {total:.1f}"
            )

    @property
    def data_type(self) -> str:
        return "multivariate"

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save multivariate-specific plots."""
        try:
            # Create empirical PSD matrix for comparison
            empirical_psd = self._compute_empirical_psd()

            # Extract true PSD if available
            true_psd = None
            if self.config.true_psd is not None:
                true_psd = np.asarray(self.config.true_psd)

            # If VI diagnostics are attached, overlay VI quantiles on top of
            # the posterior bands (instead of treating VI as another empirical
            # curve which is drawn behind the posterior fill).
            overlay_vi = bool(getattr(idata, "vi_posterior_psd", None))
            extra_empirical_psd = getattr(
                self.config, "extra_empirical_psd", None
            )
            extra_empirical_labels = getattr(
                self.config, "extra_empirical_labels", None
            )
            extra_empirical_styles = getattr(
                self.config, "extra_empirical_styles", None
            )

            plot_psd_matrix(
                idata=idata,
                freq=np.array(self.freq),
                empirical_psd=empirical_psd,
                extra_empirical_psd=extra_empirical_psd,
                extra_empirical_labels=extra_empirical_labels,
                extra_empirical_styles=extra_empirical_styles,
                true_psd=true_psd,
                overlay_vi=overlay_vi,
                outdir=self.config.outdir,
            )

            self._save_vi_diagnostics(
                empirical_psd=empirical_psd, log_summary=False
            )
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
            if psd.shape[0] != self.N:
                psd = _interp_complex_matrix(freq, np.array(self.freq), psd)
                freq = np.array(self.freq, dtype=np.float64)
            psd = self._rescale_psd(psd)
            coherence = _get_coherence(psd)
            channels = np.arange(psd.shape[1])
            return EmpiricalPSD(
                freq=freq, psd=psd, coherence=coherence, channels=channels
            )

        u_re = np.asarray(self.fft_data.u_re, dtype=np.float64)
        u_im = np.asarray(self.fft_data.u_im, dtype=np.float64)
        u_complex = u_re + 1j * u_im
        S = wishart_matrix_to_psd(
            u_to_wishart_matrix(u_complex),
            Nb=self.fft_data.Nb,
            duration=float(getattr(self.fft_data, "duration", 1.0) or 1.0),
            scaling_factor=float(self.fft_data.scaling_factor or 1.0),
            weights=float(self.Nh),
        )
        S = self._rescale_psd(S)
        coherence = _get_coherence(S)
        freq = np.array(self.freq, dtype=np.float64)
        channels = np.arange(S.shape[1])
        return EmpiricalPSD(
            freq=freq, psd=S, coherence=coherence, channels=channels
        )

    def _rescale_psd(self, psd: np.ndarray) -> np.ndarray:
        channel_stds = getattr(self.fft_data, "channel_stds", None)
        sf = float(getattr(self.fft_data, "scaling_factor", 1.0) or 1.0)
        if channel_stds is None:
            return psd
        scale_matrix = np.outer(channel_stds, channel_stds).astype(psd.dtype)
        if sf != 0:
            return psd * (scale_matrix / sf)
        return psd * scale_matrix

    def _extract_vi_psd_median(
        self, idata: az.InferenceData
    ) -> Optional[EmpiricalPSD]:
        """Extract VI PSD median as EmpiricalPSD for overlay plotting."""
        if not hasattr(idata, "vi_posterior_psd"):
            return None

        vi_psd_group = idata.vi_posterior_psd
        if "psd_matrix_real" not in vi_psd_group:
            return None

        try:
            real_array = np.asarray(
                vi_psd_group["psd_matrix_real"].values, dtype=np.float64
            )
            imag_array = np.asarray(
                vi_psd_group["psd_matrix_imag"].values, dtype=np.float64
            )
            percentiles = np.asarray(
                vi_psd_group["psd_matrix_real"].coords["percentile"].values
            )

            # Find the median (50th percentile)
            median_idx = np.argmin(np.abs(percentiles - 50.0))
            vi_psd_median = (
                real_array[median_idx] + 1j * imag_array[median_idx]
            )

            freq = np.asarray(
                vi_psd_group["psd_matrix_real"].coords["freq"].values,
                dtype=np.float64,
            )
            coherence = _get_coherence(vi_psd_median)
            channels = np.arange(vi_psd_median.shape[1])

            return EmpiricalPSD(
                freq=freq,
                psd=vi_psd_median,
                coherence=coherence,
                channels=channels,
            )
        except Exception as e:
            if self.config.verbose:
                logger.debug(f"Could not extract VI PSD median: {e}")
            return None

    def _save_vi_diagnostics(
        self,
        *,
        empirical_psd: Optional[EmpiricalPSD] = None,
        log_summary: bool = True,
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
        generate_vi_diagnostics_summary(
            vi_diag, outdir=self.config.outdir, log=log_summary
        )

    def _create_vi_inference_data(
        self,
        samples: Dict[str, jnp.ndarray],
        sample_stats: Dict[str, jnp.ndarray],
        diagnostics: Optional[Dict[str, Any]],
    ) -> az.InferenceData:
        """Convert VI samples to ArviZ and attach matrix diagnostics."""

        idata = self._create_inference_data(
            samples,
            sample_stats,
            lnz=np.nan,
            lnz_err=np.nan,
        )
        self._attach_vi_psd_group(idata, diagnostics)
        return idata

    def _attach_vi_psd_group(
        self, idata: az.InferenceData, diagnostics: Optional[Dict[str, Any]]
    ) -> None:
        """Add VI PSD summaries to an auxiliary InferenceData group."""

        if not diagnostics:
            return

        psd_quantiles = diagnostics.get("psd_quantiles")
        channels = np.arange(self.p)
        freq = np.asarray(self.freq, dtype=np.float32)
        coords = {
            "freq": freq,
            "channels": channels,
            "channels2": channels,
        }

        if psd_quantiles:
            percentiles = []
            real_entries = []
            imag_entries = []
            for label, percentile in [
                ("q05", 5.0),
                ("q50", 50.0),
                ("q95", 95.0),
            ]:
                real_val = psd_quantiles.get("real", {}).get(label)
                imag_val = psd_quantiles.get("imag", {}).get(label)
                if real_val is None:
                    continue
                percentiles.append(percentile)
                real_entries.append(np.asarray(real_val))
                if imag_val is not None:
                    imag_entries.append(np.asarray(imag_val))
                else:
                    imag_entries.append(np.zeros_like(real_entries[-1]))
            if not percentiles:
                return
            real_array = np.stack(real_entries, axis=0)
            imag_array = np.stack(imag_entries, axis=0)
            percentiles = np.asarray(percentiles, dtype=np.float32)
        else:
            median = diagnostics.get("psd_matrix")
            if median is None:
                return
            real_array = np.asarray(median)[None, ...]
            imag_array = np.zeros_like(real_array)
            percentiles = np.asarray([50.0], dtype=np.float32)

        data_vars = {
            "psd_matrix_real": DataArray(
                real_array,
                dims=["percentile", "freq", "channels", "channels2"],
                coords={**coords, "percentile": percentiles},
            ),
            "psd_matrix_imag": DataArray(
                imag_array,
                dims=["percentile", "freq", "channels", "channels2"],
                coords={**coords, "percentile": percentiles},
            ),
        }

        coherence_quantiles = diagnostics.get("coherence_quantiles")
        if coherence_quantiles:
            coh_entries = []
            coh_percentiles = []
            for label, percentile in [
                ("q05", 5.0),
                ("q50", 50.0),
                ("q95", 95.0),
            ]:
                value = coherence_quantiles.get(label)
                if value is None:
                    continue
                coh_entries.append(np.asarray(value))
                coh_percentiles.append(percentile)
            if coh_entries:
                coh_array = np.stack(coh_entries, axis=0)
                coh_percentiles = np.asarray(coh_percentiles, dtype=np.float32)
                data_vars["coherence"] = DataArray(
                    coh_array,
                    dims=["percentile", "freq", "channels", "channels2"],
                    coords={**coords, "percentile": coh_percentiles},
                )

        dataset = Dataset(data_vars)
        attr_keys = [
            "riae_matrix",
            "riae_per_channel",
            "riae_offdiag",
            "coherence_riae",
            "coverage",
            "ci_coverage",
            "coverage_interval",
            "coverage_level",
            "riae_matrix_errorbars",
        ]
        attrs = {
            key: diagnostics[key]
            for key in attr_keys
            if diagnostics.get(key) is not None
        }
        if attrs:
            dataset.attrs.update(attrs)
        idata.add_groups(vi_posterior_psd=dataset)

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
