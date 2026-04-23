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

import jax
import jax.numpy as jnp
import morphZ
import numpy as np
import xarray as xr
from xarray import Dataset

from ...arviz_utils.to_arviz import _pack_spline_model_multivar
from ...datatypes import MultivarFFT
from ...datatypes.multivar import EmpiricalPSD
from ...datatypes.multivar_utils import (
    U_to_Y,
    Y_to_S,
    _get_coherence,
    _interp_complex_matrix,
)
from ...diagnostics import (
    build_vi_summary_table,
)
from ...diagnostics._factors import vi_factor_idatas
from ...logger import logger
from ...plotting import (
    PSDMatrixPlotSpec,
    plot_psd_matrix,
    plot_vi_loss,
)
from ...psplines.multivar_psplines import MultivariateLogPSplines
from ..base_sampler import BaseSampler, SamplerConfig
from ..pspline_block import build_log_density_fn, evaluate_log_density_batch


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
        self.freq_np = np.asarray(self.freq, dtype=np.float32)
        self.channel_coords = np.arange(self.p)
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
            logger.info(
                f"B-spline basis ({len(self.all_bases)} models): {basis_shapes}"
            )
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

    def _basis_dim_coords(self) -> Dict[str, np.ndarray]:
        coords: Dict[str, np.ndarray] = {}
        for idx, basis in enumerate(self.all_bases):
            coords[f"delta_{idx}_basis_dim"] = np.arange(int(basis.shape[1]))
        for j, l in self.spline_model.theta_pairs:
            for part in ("re", "im"):
                component = f"theta_{part}_{j}_{l}"
                basis = self.spline_model.get_theta_model(part, j, l).basis
                coords[f"{component}_basis_dim"] = np.arange(
                    int(np.asarray(basis).shape[1])
                )
        return coords

    def _make_empirical_psd(
        self, freq: np.ndarray, psd: np.ndarray
    ) -> EmpiricalPSD:
        coherence = _get_coherence(psd)
        return EmpiricalPSD(
            freq=freq,
            psd=psd,
            coherence=coherence,
            channels=np.arange(psd.shape[1]),
        )

    def _arviz_coords(self) -> Dict[str, Any]:
        return {"freq": self.freq_np, **self._basis_dim_coords()}

    def _arviz_dims(
        self, samples: Optional[Dict[str, Any]] = None
    ) -> Dict[str, list[str]]:
        dims: Dict[str, list[str]] = {}
        if not samples:
            return dims
        for key in samples:
            name = str(key)
            if name.startswith("weights_"):
                component = name[8:]
                dims[name] = [f"{component}_basis_dim"]
        return dims

    def _observed_csd(self) -> np.ndarray:
        raw_psd = getattr(self.fft_data, "raw_psd", None)
        if raw_psd is not None:
            psd = np.asarray(raw_psd, dtype=np.complex128)
        else:
            psd = Y_to_S(
                U_to_Y(self.fft_data.U),
                Nb=self.fft_data.Nb,
                duration=float(getattr(self.fft_data, "duration", 1.0) or 1.0),
                scaling_factor=float(self.fft_data.scaling_factor or 1.0),
                Nh=self.Nh,
            )
        return self._rescale_psd(psd)

    def _attach_custom_groups(
        self,
        idata: xr.DataTree,
        *,
        samples: Dict[str, Any],
        sample_stats: Dict[str, Any],
    ) -> None:
        idata.attrs.update(
            {
                "p": self.fft_data.p,
                "N": self.fft_data.N,
                "n_theta": self.spline_model.n_theta,
                "frequencies": self.freq_np,
            }
        )
        idata["observed_data"] = xr.DataTree(
            dataset=Dataset(
                {
                    "periodogram": (
                        ("freq", "channels", "channels2"),
                        self._observed_csd(),
                    )
                },
                coords={
                    "freq": self.freq_np,
                    "channels": self.channel_coords,
                    "channels2": self.channel_coords,
                },
            )
        )
        idata["spline_model"] = xr.DataTree(
            dataset=_pack_spline_model_multivar(self.spline_model)
        )

    def _save_plots(self, idata: xr.DataTree) -> None:
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
            overlay_vi = bool(getattr(self, "_vi_diagnostics", None))
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
                filename="posterior_predictive.png",
            )
            plot_psd_matrix(spec)
            logger.debug(
                f"save_plots(multivar): PSD matrix plot done in {time.perf_counter() - t0:.2f}s"
            )

            t0 = time.perf_counter()
            logger.debug("save_plots(multivar): saving VI diagnostics")
            self._save_vi_diagnostics(
                idata=idata, empirical_psd=empirical_psd, log_summary=False
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
                psd = _interp_complex_matrix(freq, self.freq_np, psd)
                freq = self.freq_np.astype(np.float64)
            psd = self._rescale_psd(psd)
            return self._make_empirical_psd(freq, psd)

        return self._make_empirical_psd(
            self.freq_np.astype(np.float64), self._observed_csd()
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
        self, idata: xr.DataTree
    ) -> Optional[EmpiricalPSD]:
        """Extract VI PSD median as EmpiricalPSD for overlay plotting."""
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if not vi_diag:
            return None
        psd_quantiles = vi_diag.get("psd_quantiles")
        if not psd_quantiles:
            return None

        try:
            vi_psd_all = np.asarray(
                psd_quantiles["posterior_psd"], dtype=np.complex128
            )
            vi_psd_median = (
                vi_psd_all[1] if vi_psd_all.shape[0] >= 3 else vi_psd_all
            )
            coherence = _get_coherence(vi_psd_median)
            channels = np.arange(vi_psd_median.shape[1])

            return EmpiricalPSD(
                freq=self.freq_np.astype(np.float64),
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
        idata: Optional[xr.DataTree] = None,
        empirical_psd: Optional[EmpiricalPSD] = None,
        log_summary: bool = True,
    ) -> None:
        """Persist VI diagnostics if available."""
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if not vi_diag:
            return

        del empirical_psd
        if self.config.outdir is None:
            return

        diagnostics_dir = Path(self.config.outdir) / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        summary = build_vi_summary_table(vi_diag)
        factor_idata_map = vi_factor_idatas(idata) if idata is not None else {}
        if factor_idata_map:
            try:
                loo_summary = build_vi_summary_table(factor_idata_map)
                summary = summary.drop(
                    columns=["pareto_k_max", "pareto_k_median", "loo_warning"],
                    errors="ignore",
                ).merge(
                    loo_summary[
                        [
                            "factor",
                            "pareto_k_max",
                            "pareto_k_median",
                            "loo_warning",
                        ]
                    ],
                    on="factor",
                    how="left",
                )
            except Exception:
                logger.warning(
                    "Could not compute ArviZ VI Pareto-k summary.",
                    exc_info=True,
                )
        summary.to_csv(str(diagnostics_dir / "vi_summary.csv"), index=False)

        factor_payloads: dict[str, dict[str, np.ndarray]] = {}
        for factor in summary["factor"]:
            factor_key = str(factor)
            payload: dict[str, np.ndarray] = {}
            if "losses_per_block" in vi_diag:
                payload["losses"] = np.asarray(vi_diag["losses_per_block"])[
                    int(factor)
                ]
            factor_payloads[factor_key] = payload

        if any("losses" in payload for payload in factor_payloads.values()):
            try:
                losses_dict = {
                    factor: payload["losses"]
                    for factor, payload in factor_payloads.items()
                    if "losses" in payload
                }
                fig = plot_vi_loss(losses_dict)
                fig.savefig(
                    diagnostics_dir / "vi_elbo_factors.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                logger.warning(
                    "Could not save combined VI ELBO plot.",
                    exc_info=True,
                )

        if log_summary and not summary.empty:
            worst_row = summary.loc[
                summary["pareto_k_max"].fillna(-np.inf).idxmax()
            ]
            logger.info(
                f"VI summary: worst_factor={worst_row['factor']}, "
                f"pareto_k_max={worst_row['pareto_k_max']:.3f}, "
                f"loo_warning={bool(worst_row['loo_warning'])}"
            )

    def _attach_vi_group(
        self, idata: xr.DataTree, diagnostics: Optional[Dict[str, Any]]
    ) -> None:
        """Add VI posterior samples and diagnostics to auxiliary groups."""

        if not diagnostics:
            return

        if bool(idata.attrs.get("only_vi")):
            return

        vi_samples = diagnostics.get("vi_samples")
        if not vi_samples:
            return

        sample_vars = {}
        coords: Dict[str, Any] = {}
        for key, value in vi_samples.items():
            if not str(key).startswith("weights_"):
                continue
            array = np.asarray(value)
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2:
                continue
            component = str(key)[8:]
            basis_dim = f"{component}_basis_dim"
            coords["chain"] = np.arange(1)
            coords["draw"] = np.arange(array.shape[0])
            coords[basis_dim] = np.arange(array.shape[-1])
            sample_vars[key] = (["chain", "draw", basis_dim], array[None, ...])
        self._attach_vi_posterior_dataset(idata, sample_vars, coords)
        vi_log_likelihood = self._build_vi_log_likelihood_dataset(vi_samples)
        if vi_log_likelihood is not None:
            idata["vi_log_likelihood"] = xr.DataTree(dataset=vi_log_likelihood)
        self._attach_vi_sample_stats(
            idata,
            diagnostics,
            attr_keys=(
                "riae_matrix",
                "l2_matrix",
                "riae_per_channel",
                "riae_offdiag",
                "coherence_riae",
                "coverage",
                "ci_coverage",
                "ci_width",
                "ci_width_diag_mean",
                "coverage_interval",
                "coverage_level",
                "riae_matrix_errorbars",
                "psis_khat_max",
                "psis_khat_status",
                "psis_khat_threshold",
                "guide",
            ),
        )

    def _build_vi_log_likelihood_dataset(
        self, vi_samples: Dict[str, Any]
    ) -> Optional[Dataset]:
        """Build pointwise VI log-likelihood arrays for blocked multivariate VI."""

        if not vi_samples:
            return None

        chain_coords = np.arange(1)
        freq_coords = np.asarray(self.freq_np, dtype=np.float64)
        data_vars: Dict[str, tuple[list[str], np.ndarray]] = {}
        eta_vi = min(1.0, 1.0 / float(self.Nb * self.Nh))

        for channel_index in range(self.p):
            weights_delta_name = f"weights_delta_{channel_index}"
            weights_delta = vi_samples.get(weights_delta_name)
            if weights_delta is None:
                continue

            weights_delta = np.asarray(weights_delta, dtype=np.float64)
            if weights_delta.ndim != 2 or weights_delta.shape[0] == 0:
                continue

            n_draws = int(weights_delta.shape[0])
            delta_basis = np.asarray(
                self.all_bases[channel_index], dtype=np.float64
            )
            log_delta_sq = weights_delta @ delta_basis.T
            delta_eff_sq = np.exp(log_delta_sq)

            theta_count = channel_index
            if theta_count > 0:
                theta_re = np.zeros(
                    (n_draws, self.N, theta_count), dtype=np.float64
                )
                theta_im = np.zeros_like(theta_re)
                for theta_idx in range(theta_count):
                    basis_re = np.asarray(
                        self.spline_model.get_theta_model(
                            "re", channel_index, theta_idx
                        ).basis,
                        dtype=np.float64,
                    )
                    basis_im = np.asarray(
                        self.spline_model.get_theta_model(
                            "im", channel_index, theta_idx
                        ).basis,
                        dtype=np.float64,
                    )
                    w_re = np.asarray(
                        vi_samples[
                            f"weights_theta_re_{channel_index}_{theta_idx}"
                        ],
                        dtype=np.float64,
                    )
                    w_im = np.asarray(
                        vi_samples[
                            f"weights_theta_im_{channel_index}_{theta_idx}"
                        ],
                        dtype=np.float64,
                    )
                    theta_re[:, :, theta_idx] = w_re @ basis_re.T
                    theta_im[:, :, theta_idx] = w_im @ basis_im.T
            else:
                theta_re = np.zeros((n_draws, self.N, 0), dtype=np.float64)
                theta_im = np.zeros_like(theta_re)

            u_re_channel = np.asarray(
                self.u_re[:, channel_index, :], dtype=np.float64
            )
            u_im_channel = np.asarray(
                self.u_im[:, channel_index, :], dtype=np.float64
            )
            u_re_prev = np.asarray(
                self.u_re[:, :channel_index, :], dtype=np.float64
            )
            u_im_prev = np.asarray(
                self.u_im[:, :channel_index, :], dtype=np.float64
            )

            if theta_count > 0:
                contrib_re = np.einsum(
                    "dfl,flr->dfr", theta_re, u_re_prev
                ) - np.einsum("dfl,flr->dfr", theta_im, u_im_prev)
                contrib_im = np.einsum(
                    "dfl,flr->dfr", theta_re, u_im_prev
                ) + np.einsum("dfl,flr->dfr", theta_im, u_re_prev)
                u_re_resid = u_re_channel[None, :, :] - contrib_re
                u_im_resid = u_im_channel[None, :, :] - contrib_im
            else:
                u_re_resid = np.broadcast_to(
                    u_re_channel[None, :, :], (n_draws, *u_re_channel.shape)
                )
                u_im_resid = np.broadcast_to(
                    u_im_channel[None, :, :], (n_draws, *u_im_channel.shape)
                )

            residual_power_sum = np.sum(u_re_resid**2 + u_im_resid**2, axis=2)
            pointwise = -float(self.Nb) * float(
                self.Nh
            ) * log_delta_sq - residual_power_sum / (
                float(self.duration) * delta_eff_sq
            )
            pointwise = pointwise / float(self.enbw)
            pointwise = pointwise * float(eta_vi)

            data_vars[f"log_likelihood_block_{channel_index}"] = (
                ["chain", "draw", "freq"],
                pointwise[None, ...].astype(np.float32),
            )

        if not data_vars:
            return None

        coords = {
            "chain": chain_coords,
            "draw": np.arange(next(iter(data_vars.values()))[1].shape[1]),
            "freq": freq_coords,
        }
        return Dataset(data_vars, coords=coords)

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
        *,
        mcmc: Any | None = None,
    ) -> xr.DataTree:
        idata = super()._create_inference_data(
            samples,
            sample_stats,
            lnz,
            lnz_err,
            mcmc=mcmc,
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
