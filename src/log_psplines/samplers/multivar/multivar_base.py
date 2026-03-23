from __future__ import annotations

"""
Base class for multivariate PSD samplers.
"""

import os
import tempfile
import time
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
from ...datatypes.multivar import EmpiricalPSD
from ...datatypes.multivar_utils import (
    U_to_Y,
    Y_to_S,
    _get_coherence,
    _interp_complex_matrix,
)
from ...diagnostics.plotting import generate_vi_diagnostics_summary
from ...logger import logger
from ...plotting import (
    PSDMatrixPlotSpec,
    plot_psd_matrix,
    plot_true_psd_diagnostics,
    save_vi_diagnostics_multivariate,
)
from ...psplines.multivar_psplines import MultivariateLogPSplines
from ..base_sampler import BaseSampler, SamplerConfig
from ..pspline_block import build_log_density_fn, evaluate_log_density_batch


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
        self._lnz_by_block: list[float] = []
        self._lnz_err_by_block: list[float] = []
        self._lnz_block_ids: list[str] = []

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
        self.freq = jnp.array(self.fft_data.freq, dtype=jnp.float32)
        self.u_re = jnp.array(self.fft_data.u_re, dtype=jnp.float32)
        self.u_im = jnp.array(self.fft_data.u_im, dtype=jnp.float32)
        self.Nb = int(self.fft_data.Nb)
        self.duration = float(getattr(self.fft_data, "duration", 1.0) or 1.0)
        if self.duration <= 0.0:
            raise ValueError("fft_data.duration must be positive")

        # For equal-sized coarse bins, a single Nh scalar is sufficient.
        Nh = getattr(self.fft_data, "Nh", 1)
        if isinstance(Nh, bool) or not isinstance(Nh, (int, np.integer)):
            raise TypeError("fft_data.Nh must be a positive integer")
        self.Nh = int(Nh)
        if self.Nh <= 0:
            raise ValueError("fft_data.Nh must be positive")

        # Equivalent Noise Bandwidth of the analysis window.  Defaults to 1.0
        # (rectangular window) when the attribute is absent on legacy objects.
        self.enbw = float(getattr(self.fft_data, "enbw", 1.0))
        if not np.isfinite(self.enbw) or self.enbw <= 0.0:
            raise ValueError("fft_data.enbw must be a positive finite float")

        if self.config.verbose:
            logger.info(f"Frequency bins used for inference (N): {self.N}")
            basis_shapes = ", ".join(
                [f"{tuple(b.shape)}" for b in self.all_bases]
            )
            logger.info(f"B-spline basis shapes: {basis_shapes}")
            total = self.N * self.Nh
            logger.info(
                f"Applied coarse-grain counts; total effective count = {total}"
            )
            percent_retained = 100.0 / float(self.Nh)
            percent_decimated = 100.0 - percent_retained
            logger.info(
                f"Coarse-grain retention: {percent_retained:.1f}% "
                f"(decimated {percent_decimated:.1f}%, Nh={self.Nh})."
            )

    @property
    def data_type(self) -> str:
        return "multivariate"

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save multivariate-specific plots."""
        try:
            t0 = time.perf_counter()
            logger.debug("save_plots(multivar): computing empirical PSD")
            # Create empirical PSD matrix for comparison
            empirical_psd = self._compute_empirical_psd()
            logger.debug(
                f"save_plots(multivar): empirical PSD computed in {time.perf_counter() - t0:.2f}s"
            )

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

            t0 = time.perf_counter()
            logger.debug("save_plots(multivar): plotting PSD matrix")
            spec = PSDMatrixPlotSpec(
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
            plot_psd_matrix(spec)
            logger.debug(
                f"save_plots(multivar): PSD matrix plot done in {time.perf_counter() - t0:.2f}s"
            )

            if true_psd is not None and self.config.outdir is not None:
                t0 = time.perf_counter()
                logger.debug(
                    "save_plots(multivar): plotting true-PSD diagnostics"
                )
                _outdir: str = self.config.outdir
                _diag_dir = os.path.join(_outdir, "diagnostics")
                plot_true_psd_diagnostics(
                    idata,
                    true_psd,
                    freq=np.array(self.freq),
                    outdir=_diag_dir,
                )
                logger.debug(
                    f"save_plots(multivar): true-PSD diagnostics done in {time.perf_counter() - t0:.2f}s"
                )

                # Composite accuracy.png from the three per-frequency diagnostic plots
                from ...plotting.base import (
                    composite_images_vertical as _composite,
                )

                _accuracy_sources = [
                    os.path.join(
                        _outdir,
                        "diagnostics",
                        "psd_truth_error_vs_freq.png",
                    ),
                    os.path.join(_outdir, "diagnostics", "riae_vs_freq.png"),
                    os.path.join(
                        _outdir,
                        "diagnostics",
                        "coverage_vs_freq.png",
                    ),
                ]
                _composite(
                    _accuracy_sources,
                    outfile=os.path.join(
                        _outdir, "diagnostics", "accuracy.png"
                    ),
                    dpi=150,
                    title="Accuracy Diagnostics",
                )

            t0 = time.perf_counter()
            logger.debug("save_plots(multivar): saving VI diagnostics")
            self._save_vi_diagnostics(
                empirical_psd=empirical_psd, log_summary=False
            )
            logger.debug(
                f"save_plots(multivar): VI diagnostics done in {time.perf_counter() - t0:.2f}s"
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

        S = Y_to_S(
            U_to_Y(self.fft_data.U),
            Nb=self.fft_data.Nb,
            duration=float(getattr(self.fft_data, "duration", 1.0) or 1.0),
            scaling_factor=float(self.fft_data.scaling_factor or 1.0),
            Nh=self.Nh,
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
            percentile_values: list[float] = []
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
                percentile_values.append(percentile)
                real_entries.append(np.asarray(real_val))
                if imag_val is not None:
                    imag_entries.append(np.asarray(imag_val))
                else:
                    imag_entries.append(np.zeros_like(real_entries[-1]))
            if not percentile_values:
                return
            real_array = np.stack(real_entries, axis=0)
            imag_array = np.stack(imag_entries, axis=0)
            percentiles = np.asarray(percentile_values, dtype=np.float32)
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
            coh_percentile_values: list[float] = []
            for label, percentile in [
                ("q05", 5.0),
                ("q50", 50.0),
                ("q95", 95.0),
            ]:
                value = coherence_quantiles.get(label)
                if value is None:
                    continue
                coh_entries.append(np.asarray(value))
                coh_percentile_values.append(percentile)
            if coh_entries:
                coh_array = np.stack(coh_entries, axis=0)
                coh_percentiles = np.asarray(
                    coh_percentile_values, dtype=np.float32
                )
                data_vars["coherence"] = DataArray(
                    coh_array,
                    dims=["percentile", "freq", "channels", "channels2"],
                    coords={**coords, "percentile": coh_percentiles},
                )

        dataset = Dataset(data_vars)
        attr_keys = [
            "riae_matrix",
            "l2_matrix",
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
        import xarray as _xr

        idata["vi_posterior_psd"] = _xr.DataTree(dataset=dataset)

    def _get_lnz(
        self, samples: Dict[str, Any], sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Default multivariate LnZ computation using per-channel factorisation."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        self._reset_lnz_details()
        try:
            lnz_by_block: list[float] = []
            lnz_err_by_block: list[float] = []
            block_ids: list[str] = []

            for channel_index in range(self._lnz_n_blocks()):
                lnz_j, lnz_err_j = self._compute_channel_lnz(
                    samples, channel_index
                )
                lnz_by_block.append(float(lnz_j))
                lnz_err_by_block.append(float(lnz_err_j))
                block_ids.append(f"channel_{channel_index}")
                if self.config.verbose:
                    logger.info(
                        f"LnZ channel {channel_index}: {lnz_j:.3f} +- {lnz_err_j:.3f}"
                    )

            lnz_arr = np.asarray(lnz_by_block, dtype=np.float64)
            lnz_err_arr = np.asarray(lnz_err_by_block, dtype=np.float64)
            lnz_total = float(np.sum(lnz_arr))
            lnz_err_total = float(np.sqrt(np.sum(lnz_err_arr**2)))

            self._lnz_by_block = lnz_by_block
            self._lnz_err_by_block = lnz_err_by_block
            self._lnz_block_ids = block_ids
            return lnz_total, lnz_err_total
        except NotImplementedError:
            self._reset_lnz_details()
            if self.config.verbose:
                logger.warning(
                    "LnZ computation is not implemented for this multivariate sampler; returning NaN."
                )
            return np.nan, np.nan
        except Exception as exc:
            self._reset_lnz_details()
            if self.config.verbose:
                logger.warning(
                    f"Blockwise multivariate LnZ computation failed: {exc}"
                )
            return np.nan, np.nan

    def _reset_lnz_details(self) -> None:
        self._lnz_by_block = []
        self._lnz_err_by_block = []
        self._lnz_block_ids = []

    def _lnz_n_blocks(self) -> int:
        return int(self.p)

    def _channel_model(self):
        raise NotImplementedError

    def _channel_model_kwargs(self, channel_index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _channel_parameter_names(self, channel_index: int) -> list[str]:
        raise NotImplementedError

    def _lnz_build_log_density_fn(self):
        return build_log_density_fn

    def _lnz_evaluate_log_density_batch(self):
        return evaluate_log_density_batch

    def _flatten_sample_array(self, array: np.ndarray) -> np.ndarray:
        if array.ndim == 0:
            return array.reshape(1)
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            if array.shape[0] == int(self.config.num_chains):
                return array.reshape(-1)
            return array
        return array.reshape((-1, *array.shape[2:]))

    def _extract_channel_params_batch(
        self, samples: Dict[str, Any], channel_index: int
    ) -> Tuple[Dict[str, jnp.ndarray], list[str]]:
        param_names = self._channel_parameter_names(channel_index)
        params_batch: Dict[str, jnp.ndarray] = {}
        n_batch: Optional[int] = None
        for name in param_names:
            if name not in samples:
                raise KeyError(
                    f"Missing posterior parameter '{name}' for channel {channel_index} LnZ."
                )
            array = np.asarray(samples[name], dtype=np.float64)
            flat_array = self._flatten_sample_array(array)
            if name.startswith("phi_"):
                flat_array = np.log(
                    np.maximum(flat_array, np.asarray(1e-12, dtype=np.float64))
                )
            if flat_array.ndim == 0:
                flat_array = flat_array.reshape(1)
            current_n_batch = int(flat_array.shape[0])
            if n_batch is None:
                n_batch = current_n_batch
            elif current_n_batch != n_batch:
                raise ValueError(
                    f"Inconsistent draw counts for channel {channel_index}: "
                    f"expected {n_batch}, got {current_n_batch} for '{name}'."
                )
            params_batch[name] = jnp.asarray(flat_array)
        return params_batch, param_names

    def _pack_morphz_samples(
        self, params_batch: Dict[str, jnp.ndarray], param_names: list[str]
    ) -> Tuple[np.ndarray, list[Tuple[str, Tuple[int, ...]]]]:
        flat_blocks = []
        layout: list[Tuple[str, Tuple[int, ...]]] = []
        for name in param_names:
            array = np.asarray(params_batch[name], dtype=np.float64)
            if array.ndim == 1:
                shape: Tuple[int, ...] = ()
                flat_blocks.append(array[:, None])
            else:
                shape = tuple(array.shape[1:])
                flat_blocks.append(array.reshape(array.shape[0], -1))
            layout.append((name, shape))
        return np.concatenate(flat_blocks, axis=1), layout

    def _parse_morphz_result(self, lnz_result: Any) -> Tuple[float, float]:
        if hasattr(lnz_result, "lnz"):
            lnz = float(lnz_result.lnz)
            uncertainty = getattr(lnz_result, "uncertainty", np.nan)
            return lnz, float(uncertainty)
        if isinstance(lnz_result, (tuple, list, np.ndarray)):
            if len(lnz_result) < 2:
                raise ValueError(
                    "morphZ result did not include both lnz and uncertainty."
                )
            return float(lnz_result[0]), float(lnz_result[1])
        raise TypeError(
            f"Unsupported morphZ result type: {type(lnz_result).__name__}."
        )

    def _compute_channel_lnz(
        self, samples: Dict[str, Any], channel_index: int
    ) -> Tuple[float, float]:
        params_batch, param_names = self._extract_channel_params_batch(
            samples, channel_index
        )
        logpost_fn = self._lnz_build_log_density_fn()(
            self._channel_model(),
            self._channel_model_kwargs(channel_index),
        )
        lp = self._lnz_evaluate_log_density_batch()(logpost_fn, params_batch)
        post_smp, layout = self._pack_morphz_samples(params_batch, param_names)
        if lp.ndim > 1:
            lp = lp.reshape(-1)
        if lp.shape[0] != post_smp.shape[0]:
            raise ValueError(
                f"morphZ input shape mismatch for channel {channel_index}: "
                f"{lp.shape[0]} lp values for {post_smp.shape[0]} samples."
            )

        def lp_fn(sample_vec: np.ndarray) -> float:
            offset = 0
            params: Dict[str, jnp.ndarray] = {}
            for name, shape in layout:
                size = int(np.prod(shape)) if shape else 1
                segment = sample_vec[offset : offset + size]
                offset += size
                if shape:
                    params[name] = jnp.asarray(segment.reshape(shape))
                else:
                    params[name] = jnp.asarray(segment.item())
            return float(logpost_fn(params))

        lnz_result = morphZ.evidence(
            post_smp,
            lp,
            lp_fn,
            kde_bw="scott",
            output_path=tempfile.gettempdir(),
        )[0]
        return self._parse_morphz_result(lnz_result)

    def _create_inference_data(
        self,
        samples: Dict[str, Any],
        sample_stats: Dict[str, Any],
        lnz: float,
        lnz_err: float,
    ) -> az.InferenceData:
        idata = super()._create_inference_data(
            samples, sample_stats, lnz, lnz_err
        )
        if self._lnz_by_block:
            idata.attrs["lnz_by_block"] = np.asarray(
                self._lnz_by_block, dtype=np.float64
            )
            idata.attrs["lnz_err_by_block"] = np.asarray(
                self._lnz_err_by_block, dtype=np.float64
            )
            idata.attrs["lnz_block_ids"] = list(self._lnz_block_ids)
        return idata

    def _compute_log_posterior(self, params: Dict[str, jnp.ndarray]) -> float:
        """Compute log posterior for LnZ calculation (implemented by subclasses)."""
        raise NotImplementedError
