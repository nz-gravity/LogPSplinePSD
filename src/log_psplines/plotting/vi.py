"""ELBO plots for a fitted variational result."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from log_psplines.inference.vi import VIResult


def plot_vi_loss(
    vi: VIResult, outfile: str | None = None
) -> plt.Figure | None:
    """Plot each blocked channel's ELBO trace, or the shared trace."""
    traces = (
        vi.losses_per_block if vi.losses_per_block is not None else [vi.losses]
    )
    curves = [np.asarray(trace, dtype=float).reshape(-1) for trace in traces]
    curves = [curve for curve in curves if curve.size]
    if not curves:
        return None
    minimum = min(float(np.nanmin(curve)) for curve in curves)
    shift = minimum - 0.1 * abs(minimum) if minimum != 0 else -1.0
    figure, axis = plt.subplots(figsize=(8 if len(curves) > 1 else 6, 5))
    for channel, curve in enumerate(curves):
        label = f"Channel {channel}" if len(curves) > 1 else vi.guide_name
        axis.plot(np.arange(curve.size), curve - shift, label=label)
    axis.set(xlabel="VI Evaluation", ylabel="ELBO (relative)", yscale="log")
    axis.set_title("VI Convergence")
    axis.grid(True, alpha=0.3, linewidth=0.8)
    axis.legend(frameon=False)
    figure.tight_layout()
    if outfile is not None:
        figure.savefig(outfile, dpi=150)
        plt.close(figure)
        return None
    return figure
