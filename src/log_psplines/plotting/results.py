"""Figures for PSDResult; failures propagate rather than changing plot type."""

from pathlib import Path
from typing import TYPE_CHECKING

import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np

from log_psplines.diagnostics.sampling import plot_energy
from log_psplines.plotting.psd_matrix import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.plotting.vi import plot_vi_loss

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def plot_posterior_spectrum(
    result: "PSDResult",
    outdir: str | Path,
    *,
    true_psd: np.ndarray | None = None,
) -> None:
    """Save a spectrum/surface, never a substitute trace or placeholder."""
    outdir = Path(outdir)
    if result.time is not None:
        median = np.diagonal(
            result.quantiles((50.0,)).values[0], axis1=-2, axis2=-1
        ).real
        if true_psd is None and result.truth is not None:
            true_psd = np.asarray(result.truth)
        if true_psd is not None:
            true_psd = np.asarray(true_psd)
            if true_psd.ndim == 2:
                true_psd = true_psd[..., None]
        ncols = 1 if true_psd is None else 3
        fig, axes = plt.subplots(
            median.shape[-1],
            ncols,
            squeeze=False,
            figsize=(6 * ncols, 3 * median.shape[-1]),
        )
        for c in range(median.shape[-1]):
            panels = [("Posterior median", np.log(median[..., c]))]
            if true_psd is not None:
                panels = [
                    ("Truth", np.log(true_psd[..., c])),
                    *panels,
                    (
                        "log(median / truth)",
                        np.log(median[..., c] / true_psd[..., c]),
                    ),
                ]
            for ax, (title, values) in zip(axes[c], panels, strict=True):
                mesh = ax.pcolormesh(
                    result.time, result.frequency, values.T, shading="auto"
                )
                ax.set(
                    xlabel="Time",
                    ylabel="Frequency [Hz]",
                    title=f"{result.spectrum.channel.values[c]}: {title}",
                )
                fig.colorbar(mesh, ax=ax)
        fig.tight_layout()
        fig.savefig(
            outdir / "posterior_spectrum.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        return
    plot_psd_matrix(
        PSDMatrixPlotSpec(
            result=result,
            true_psd=true_psd,
            outdir=str(outdir),
            filename="posterior_spectrum.png",
            save=True,
            close=True,
        )
    )


def plot_result_diagnostics(result: "PSDResult", outdir: str | Path) -> None:
    """Render available VI and stationary NUTS diagnostic plots."""
    outdir = Path(outdir) / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    if result.vi is not None and result.vi.losses is not None:
        plot_vi_loss(
            result.vi,
            outfile=str(outdir / "vi_loss.png"),
        )
    if result.sample_stats is not None:
        if result.time is None:
            diagnostics = result.to_arviz()
            azp.plot_trace_dist(
                diagnostics, compact=True, backend="matplotlib"
            ).savefig(outdir / "traces.png", dpi=150, bbox_inches="tight")
            plt.close("all")
        plot_energy(result).savefig(
            outdir / "energy.png", dpi=150, bbox_inches="tight"
        )
        plt.close("all")
