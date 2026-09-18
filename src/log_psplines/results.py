"""InferencePipeline and PSDResult."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import arviz_plots as azp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from log_psplines.arviz_utils._datatree import (
    save_inference_data as _save_inference_data,
)
from log_psplines.diagnostics import (
    build_nuts_summary_table,
    build_vi_summary_table,
)
from log_psplines.diagnostics.plot_nuts import plot_energy
from log_psplines.inference.vi import StageResult
from log_psplines.plotting import (
    PSDMatrixPlotSpec,
    plot_psd_matrix,
    plot_vi_loss,
)

from .logger import logger


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
class PSDResult:
    """Outputs from InferencePipeline.run()."""

    idata: xr.DataTree
    vi_coarse: StageResult | None = None
    vi: StageResult | None = None
    time: np.ndarray | None = None

    @property
    def posterior(self) -> xr.Dataset:
        from log_psplines.arviz_utils.from_arviz import get_sample_dataset

        return get_sample_dataset(self.idata)

    @property
    def metadata(self) -> dict:
        return dict(self.idata.attrs)

    @property
    def frequency(self) -> np.ndarray:
        from log_psplines.arviz_utils.from_arviz import (
            _get_multivar_frequency_grid,
        )

        return _get_multivar_frequency_grid(self.idata)

    @property
    def spectral_density(self) -> np.ndarray:
        """Posterior draws (chain, draw, F, C, C), in original data units.

        A future time grid would insert T before F. No TV results are produced.
        """
        from log_psplines.arviz_utils.from_arviz import get_psd_dataset

        return (
            get_psd_dataset(self.idata)
            .spectral_density.transpose(
                "chain", "draw", "frequency", "channel", "channel_aux"
            )
            .values
        )

    @property
    def psd(self) -> np.ndarray:
        """Auto spectra (chain, draw, F) for C=1, (..., F, C) otherwise."""
        diagonal = np.diagonal(self.spectral_density, axis1=-2, axis2=-1).real
        return diagonal[..., 0] if diagonal.shape[-1] == 1 else diagonal

    @property
    def coherence(self) -> np.ndarray:
        from log_psplines.models.matrix import SpectralMatrix

        return SpectralMatrix.coherence(self.spectral_density)

    def to_netcdf(self, path: str | Path) -> None:
        """Save posterior, spectral model and metadata without rendering plots."""
        _save_inference_data(self.idata, path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> "PSDResult":
        from log_psplines.arviz_utils._datatree import open_inference_data

        return cls(idata=open_inference_data(path))

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

    def _save_posterior_predictive(
        self,
        outdir: str,
        *,
        true_psd: np.ndarray | None = None,
    ) -> None:
        outfile = Path(outdir) / "posterior_predictive.png"
        overlay_vi = (
            self.vi is not None and "sample_stats" in self.idata.children
        )
        try:
            plot_psd_matrix(
                PSDMatrixPlotSpec(
                    idata=self.idata,
                    true_psd=true_psd,
                    outdir=str(outdir),
                    filename="posterior_predictive.png",
                    save=True,
                    close=True,
                    overlay_vi=overlay_vi,
                    label="NUTS 90% CI" if overlay_vi else None,
                    vi_label="VI 90% CI",
                )
            )
            return
        except Exception as exc:
            logger.debug(f"Posterior PSD plot unavailable: {exc}")

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
        self._save_posterior_predictive(outdir, true_psd=true_psd)
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
