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
        fig, ax = plt.subplots(figsize=(7, 4))
        median = np.median(result.psd, axis=(0, 1))
        mesh = ax.pcolormesh(
            result.time, result.frequency, np.log(median).T, shading="auto"
        )
        ax.set(xlabel="Rescaled time", ylabel="Frequency [Hz]")
        fig.colorbar(mesh, ax=ax, label=f"log({result.metadata['units']})")
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
