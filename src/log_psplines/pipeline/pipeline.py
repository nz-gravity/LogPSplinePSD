"""InferencePipeline and PipelineResult."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import arviz_plots as azp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..arviz_utils._datatree import save_inference_data as _save_inference_data
from ..arviz_utils.to_arviz import (
    _pack_spline_model,
    _pack_spline_model_multivar,
)
from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..diagnostics import build_nuts_summary_table, build_vi_summary_table
from ..diagnostics.plot_nuts import plot_energy
from ..logger import logger
from ..plotting import (
    PSDMatrixPlotSpec,
    plot_pdgrm,
    plot_psd_matrix,
    plot_vi_loss,
)
from .stages import NUTSStage, StageResult, VIStage


def _vi_result_to_idata(result: StageResult) -> xr.DataTree:
    """Wrap VI posterior means into a minimal xr.DataTree."""
    if not result.init_values:
        return xr.DataTree()
    ds = _init_values_to_dataset(result.init_values)
    return xr.DataTree(children={"posterior": xr.DataTree(dataset=ds)})


def _init_values_to_dataset(values: dict[str, jnp.ndarray]) -> xr.Dataset:
    """Pack VI point estimates using variable-specific trailing dimensions."""
    data_vars = {}
    for name, value in values.items():
        array = np.asarray(value)[None, None, ...]
        tail_dims = tuple(
            f"{name}_dim_{axis}" for axis in range(array.ndim - 2)
        )
        data_vars[name] = xr.DataArray(
            array,
            dims=("chain", "draw", *tail_dims),
        )
    return xr.Dataset(
        data_vars,
        coords={"chain": [0], "draw": [0]},
    )


def _losses_per_block_array(
    losses_per_block: list[jnp.ndarray] | None,
) -> np.ndarray:
    if not losses_per_block:
        return np.asarray([], dtype=float)

    arrays = [
        np.asarray(losses, dtype=float).reshape(-1)
        for losses in losses_per_block
    ]
    max_len = max((arr.size for arr in arrays), default=0)
    if max_len == 0:
        return np.asarray([], dtype=float)

    padded = np.full((len(arrays), max_len), np.nan, dtype=float)
    for idx, arr in enumerate(arrays):
        padded[idx, : arr.size] = arr
    return padded


@dataclass
class PipelineResult:
    """Outputs from InferencePipeline.run()."""

    vi_coarse: StageResult | None
    vi: StageResult | None
    idata: xr.DataTree

    @staticmethod
    def _save_placeholder_plot(path: Path, title: str, message: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_posterior_predictive(self, outdir: str) -> None:
        outfile = Path(outdir) / "posterior_predictive.png"
        try:
            fig, _ = plot_pdgrm(idata=self.idata)
            fig.savefig(outfile, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return
        except Exception as exc:
            logger.debug(f"Univariate posterior plot unavailable: {exc}")

        try:
            plot_psd_matrix(
                PSDMatrixPlotSpec(
                    idata=self.idata,
                    outdir=str(outdir),
                    filename="posterior_predictive.png",
                    save=True,
                    close=True,
                )
            )
            return
        except Exception as exc:
            logger.debug(f"Multivariate posterior plot unavailable: {exc}")

        try:
            trace_plot = azp.plot_trace_dist(
                self.idata,
                compact=True,
                backend="matplotlib",
            )
            trace_plot.savefig(
                outfile,
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")
        except Exception as exc:
            logger.warning(
                f"Could not save posterior_predictive.png: {exc}",
                exc_info=True,
            )

    @staticmethod
    def _median_numeric(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else float("nan")

    def _fallback_vi_summary(self) -> pd.DataFrame:
        losses = (
            np.asarray(self.vi.losses, dtype=float)
            if self.vi is not None and self.vi.losses is not None
            else np.asarray([], dtype=float)
        )
        return pd.DataFrame(
            [
                {
                    "factor": "0",
                    "final_elbo": float(losses[-1]) if losses.size else np.nan,
                    "pareto_k_max": np.nan,
                    "riae": np.nan,
                    "l2": np.nan,
                    "coverage": np.nan,
                }
            ]
        )

    def _save_diagnostics(
        self,
        outdir: str,
        *,
        true_psd: np.ndarray | None = None,
    ) -> None:
        diagnostics_dir = Path(outdir) / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        vi_summary: pd.DataFrame | None = None
        if self.vi is not None:
            if (
                isinstance(self.idata.attrs.get("data_type"), str)
                and self.idata.attrs["data_type"] == "multivariate"
            ):
                vi_summary = self._fallback_vi_summary()
                vi_summary.to_csv(
                    diagnostics_dir / "vi_summary.csv", index=False
                )
            else:
                try:
                    vi_summary = build_vi_summary_table(
                        self.idata,
                        true_psd=true_psd,
                    )
                    vi_summary.to_csv(
                        diagnostics_dir / "vi_summary.csv", index=False
                    )
                except Exception as exc:
                    logger.warning(
                        f"Could not save vi_summary.csv: {exc}",
                        exc_info=True,
                    )
                    vi_summary = self._fallback_vi_summary()
                    vi_summary.to_csv(
                        diagnostics_dir / "vi_summary.csv", index=False
                    )

            vi_stats = self.idata["vi_sample_stats"]
            for col in ("pareto_k_max", "riae", "l2", "coverage"):
                if col in vi_summary.columns and not vi_summary.empty:
                    vi_stats.attrs[col] = self._median_numeric(vi_summary[col])

            if self.vi.losses is not None:
                try:
                    losses_input = {
                        "losses": np.asarray(self.vi.losses, dtype=float)
                    }
                    if self.vi.losses_per_block is not None:
                        losses_input["losses_per_block"] = (
                            self.vi.losses_per_block
                        )
                    plot_vi_loss(
                        losses_input,
                        guide_name=self.vi.guide_name,
                        outfile=str(diagnostics_dir / "vi_loss.png"),
                    )
                except Exception as exc:
                    logger.warning(
                        f"Could not save vi_loss.png: {exc}",
                        exc_info=True,
                    )
                    self._save_placeholder_plot(
                        diagnostics_dir / "vi_loss.png",
                        "VI Loss",
                        "VI loss curve unavailable for this run.",
                    )

        nuts_summary: pd.DataFrame | None = None
        try:
            nuts_summary = build_nuts_summary_table(
                self.idata,
                true_psd=true_psd,
            )
        except Exception as exc:
            logger.debug(
                f"NUTS summary with truth metrics failed, "
                f"retrying without truth: {exc}"
            )
            try:
                nuts_summary = build_nuts_summary_table(self.idata)
            except Exception as err:
                logger.warning(
                    f"Could not save nuts_summary.csv: {err}",
                    exc_info=True,
                )

        if nuts_summary is not None:
            nuts_summary.to_csv(
                diagnostics_dir / "nuts_summary.csv",
                index=False,
            )
            sample_stats = self.idata.children.get("sample_stats")
            if sample_stats is not None:
                for col in (
                    "divergences",
                    "max_treedepth_hits",
                    "rhat_max",
                    "riae",
                    "l2",
                    "coverage",
                    "step_size",
                ):
                    if col in nuts_summary.columns and not nuts_summary.empty:
                        sample_stats.attrs[col] = self._median_numeric(
                            nuts_summary[col]
                        )

        if "sample_stats" in self.idata.children:
            try:
                azp.plot_trace_dist(
                    self.idata,
                    compact=True,
                    backend="matplotlib",
                ).savefig(
                    diagnostics_dir / "traces.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close("all")
            except Exception as exc:
                logger.warning(
                    f"Could not save traces.png: {exc}",
                    exc_info=True,
                )

            try:
                plot_energy(self.idata).savefig(
                    diagnostics_dir / "energy.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close("all")
            except Exception as exc:
                logger.warning(
                    f"Could not save energy.png: {exc}",
                    exc_info=True,
                )

        row: dict[str, float] = {}
        if vi_summary is not None and not vi_summary.empty:
            for col in (
                "pareto_k_max",
                "riae",
                "l2",
                "coverage",
                "final_elbo",
            ):
                if col in vi_summary.columns:
                    row[f"vi_{col}"] = self._median_numeric(vi_summary[col])

        if nuts_summary is not None and not nuts_summary.empty:
            for col in (
                "divergences",
                "max_treedepth_hits",
                "rhat_max",
                "riae",
                "l2",
                "coverage",
                "step_size",
                "ess_bulk_min",
                "ess_tail_min",
            ):
                if col in nuts_summary.columns:
                    row[f"nuts_{col}"] = self._median_numeric(
                        nuts_summary[col]
                    )

        if row:
            pd.DataFrame([row]).to_csv(
                diagnostics_dir / "diagnostics.csv",
                index=False,
            )

    def save(
        self,
        outdir: str,
        *,
        true_psd: np.ndarray | None = None,
    ) -> None:
        os.makedirs(outdir, exist_ok=True)
        self._save_posterior_predictive(outdir)
        self._save_diagnostics(outdir, true_psd=true_psd)
        _save_inference_data(
            self.idata,
            os.path.join(outdir, "inference_data.nc"),
            engine="h5netcdf",
        )
        if self.vi is not None and self.vi.losses is not None:
            np.save(
                os.path.join(outdir, "vi_losses.npy"),
                np.asarray(self.vi.losses),
            )
            losses_per_block = _losses_per_block_array(
                self.vi.losses_per_block
            )
            if losses_per_block.size:
                np.save(
                    os.path.join(outdir, "vi_losses_per_block.npy"),
                    losses_per_block,
                )
        if self.vi_coarse is not None and self.vi_coarse.losses is not None:
            np.save(
                os.path.join(outdir, "vi_coarse_losses.npy"),
                np.asarray(self.vi_coarse.losses),
            )


class InferencePipeline:
    """Sequential vi_coarse → vi → nuts inference pipeline.

    Each stage receives ``init_values`` from the previous stage so that
    downstream optimisation / sampling starts near the posterior mode.
    """

    def __init__(
        self,
        model_fn: Callable,
        full_model_kwargs: dict,
        coarse_model_kwargs: dict | None,
        data: Periodogram | MultivarFFT,
        spline_model,
        config,
        vi_stage: VIStage,
        nuts_stage: NUTSStage,
        *,
        rng_key: int | jax.Array = 42,
        verbose: bool = False,
        vi_progress_bar: bool | None = None,
        only_vi: bool = False,
        init_from_vi: bool = True,
        vi_coarse_only: bool = False,
    ) -> None:
        self.model_fn = model_fn
        self.full_model_kwargs = full_model_kwargs
        self.coarse_model_kwargs = coarse_model_kwargs
        self.data = data
        self.spline_model = spline_model
        self.config = config
        self.vi_stage = vi_stage
        self.nuts_stage = nuts_stage
        self.rng_key = rng_key
        self.verbose = verbose
        self.vi_progress_bar = (
            verbose if vi_progress_bar is None else vi_progress_bar
        )
        self.only_vi = only_vi
        self.init_from_vi = init_from_vi
        self.vi_coarse_only = vi_coarse_only

    def _observed_data_dataset(self) -> xr.Dataset:
        if isinstance(self.data, Periodogram):
            freq = np.asarray(self.data.freqs, dtype=float)
            values = np.asarray(self.data.power, dtype=float)
        else:
            freq = np.asarray(self.data.freq, dtype=float)
            if self.data.raw_psd is not None:
                values = np.real(
                    np.diagonal(
                        np.asarray(self.data.raw_psd),
                        axis1=1,
                        axis2=2,
                    )
                )
            else:
                values = np.zeros((freq.size, int(self.data.p)), dtype=float)

        return xr.Dataset(
            {
                "periodogram": xr.DataArray(
                    values,
                    dims=(
                        ("freq",) if values.ndim == 1 else ("freq", "channel")
                    ),
                    coords={"freq": freq},
                )
            }
        )

    def _vi_posterior_dataset(self, vi: StageResult) -> xr.Dataset:
        if not vi.init_values:
            return xr.Dataset()
        return _init_values_to_dataset(vi.init_values)

    def _attach_pipeline_metadata(
        self,
        idata: xr.DataTree,
        vi: StageResult | None,
    ) -> xr.DataTree:
        """Attach model/data groups needed by diagnostics and plotting."""
        if isinstance(self.data, Periodogram):
            spline_ds = _pack_spline_model(self.spline_model)
            attrs = {
                "data_type": "univariate",
                "scaling_factor": float(self.data.scaling_factor),
            }
        else:
            spline_ds = _pack_spline_model_multivar(self.spline_model)
            attrs = {
                "data_type": "multivariate",
                "scaling_factor": float(self.data.scaling_factor or 1.0),
                "channel_stds": (
                    None
                    if self.data.channel_stds is None
                    else np.asarray(self.data.channel_stds)
                ),
                "sampler": "factorized_multivar_nuts",
            }

        attrs.update(
            {
                "max_tree_depth": int(self.config.max_tree_depth),
                "posterior_psd_max_draws": int(self.config.vi_psd_max_draws),
                "vi_psd_max_draws": int(self.config.vi_psd_max_draws),
                "alpha_phi": float(self.config.alpha_phi),
                "beta_phi": float(self.config.beta_phi),
                "alpha_delta": float(self.config.alpha_delta),
                "beta_delta": float(self.config.beta_delta),
                "eta": float(self.config.eta),
                "sampling_eta": float(self.nuts_stage.eta),
            }
        )
        if self.config.target_accept_prob_by_channel is not None:
            attrs["target_accept_prob_by_channel"] = list(
                self.config.target_accept_prob_by_channel
            )
        if self.config.max_tree_depth_by_channel is not None:
            attrs["max_tree_depth_by_channel"] = list(
                self.config.max_tree_depth_by_channel
            )
        if isinstance(self.data, MultivarFFT):
            for channel_index in range(int(self.data.p)):
                attrs[f"sampling_eta_channel_{channel_index}"] = float(
                    self.nuts_stage.eta
                )
        idata.attrs.update(attrs)
        idata["observed_data"] = xr.DataTree(
            dataset=self._observed_data_dataset()
        )
        idata["spline_model"] = xr.DataTree(dataset=spline_ds)

        if vi is not None:
            idata["vi_posterior"] = xr.DataTree(
                dataset=self._vi_posterior_dataset(vi)
            )
            losses = (
                np.asarray(vi.losses, dtype=float)
                if vi.losses is not None
                else np.asarray([], dtype=float)
            )
            vi_stats = xr.Dataset(
                {
                    "losses": xr.DataArray(
                        losses,
                        dims=("draw",),
                        coords={"draw": np.arange(losses.size)},
                    )
                }
            )
            losses_per_block = _losses_per_block_array(vi.losses_per_block)
            if losses_per_block.size:
                vi_stats["losses_per_block"] = xr.DataArray(
                    losses_per_block,
                    dims=("factor", "draw_per_factor"),
                    coords={
                        "factor": np.arange(losses_per_block.shape[0]),
                        "draw_per_factor": np.arange(
                            losses_per_block.shape[1]
                        ),
                    },
                )
            idata["vi_sample_stats"] = xr.DataTree(dataset=vi_stats)
            if isinstance(self.data, MultivarFFT):
                idata["vi_log_likelihood"] = xr.DataTree(
                    dataset=xr.Dataset(
                        {
                            f"log_likelihood_block_{j}": xr.DataArray(
                                np.zeros((1, 1, int(self.data.N))),
                                dims=("chain", "draw", "freq"),
                                coords={
                                    "chain": [0],
                                    "draw": [0],
                                    "freq": np.asarray(
                                        self.data.freq, dtype=float
                                    ),
                                },
                            )
                            for j in range(int(self.data.p))
                        }
                    )
                )
        return idata

    def run(self) -> PipelineResult:
        """Execute the pipeline and return a PipelineResult."""
        rng = (
            jax.random.PRNGKey(self.rng_key)
            if isinstance(self.rng_key, int)
            else self.rng_key
        )

        vi_coarse: StageResult | None = None
        init_values: dict[str, jnp.ndarray] | None = None

        if self.coarse_model_kwargs is not None:
            rng, key = jax.random.split(rng)
            vi_coarse = self.vi_stage.run(
                self.model_fn,
                self.coarse_model_kwargs,
                init_values=None,
                rng_key=key,
                verbose=self.vi_progress_bar,
            )
            init_values = vi_coarse.init_values

        if self.vi_coarse_only:
            if vi_coarse is None:
                raise ValueError(
                    "vi_coarse_only=True requires coarse_model_kwargs. "
                    "Set coarse_grain_config_vi or auto_coarse_vi."
                )
            idata = _vi_result_to_idata(vi_coarse)
            idata = self._attach_pipeline_metadata(idata, vi_coarse)
            return PipelineResult(vi_coarse=vi_coarse, vi=None, idata=idata)

        rng, key = jax.random.split(rng)
        vi = self.vi_stage.run(
            self.model_fn,
            self.full_model_kwargs,
            init_values=init_values,
            rng_key=key,
            verbose=self.vi_progress_bar,
        )
        init_values = vi.init_values if self.init_from_vi else None

        if self.only_vi:
            idata = _vi_result_to_idata(vi)
            idata = self._attach_pipeline_metadata(idata, vi)
            return PipelineResult(vi_coarse=vi_coarse, vi=vi, idata=idata)

        logger.info(f"Spline model: {self.spline_model}")

        rng, key = jax.random.split(rng)
        idata = self.nuts_stage.run(
            self.model_fn,
            self.full_model_kwargs,
            init_values=init_values,
            rng_key=key,
            verbose=self.verbose,
        )
        idata = self._attach_pipeline_metadata(idata, vi)
        return PipelineResult(vi_coarse=vi_coarse, vi=vi, idata=idata)
