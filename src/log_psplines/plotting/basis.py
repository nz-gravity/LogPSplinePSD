from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from log_psplines.basis import SplineBasis


def plot_basis_diagnostics(
    basis: SplineBasis, *, path: str | Path | None = None
) -> tuple[Figure, np.ndarray]:
    """Inspect the actual basis (G, K), knots, and derivative penalty (K, K).

    Pass model.frequency or model.time to inspect a fitted component's basis.
    Coordinates and operators are plotted as stored, without rebuilding them.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    axes[0].plot(basis.grid, np.asarray(basis.basis), lw=1.2)
    knots = np.unique(basis.knots)
    interior = knots[(knots > basis.grid[0]) & (knots < basis.grid[-1])]
    for positions, label, style in (
        (interior, "Interior knots", ":"),
        (np.array([basis.grid[0], basis.grid[-1]]), "Boundary knots", "--"),
    ):
        axes[0].vlines(
            positions,
            0,
            1,
            transform=axes[0].get_xaxis_transform(),
            colors="0.35",
            linestyles=style,
            lw=0.9,
            label=label,
        )
    axes[0].set(
        xlabel="Coordinate",
        ylabel="Basis value",
        xlim=(basis.grid[0], basis.grid[-1]),
        title=f"Degree {basis.degree}; derivative penalty order {basis.penalty_order}",
    )
    axes[0].legend(frameon=False)
    penalty = np.asarray(basis.penalty)
    limit = float(np.max(np.abs(penalty))) or 1.0
    mesh = axes[1].pcolormesh(penalty, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1].set(
        xlabel="Coefficient index",
        ylabel="Coefficient index",
        title="Penalty matrix",
    )
    fig.colorbar(mesh, ax=axes[1])
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig, axes
