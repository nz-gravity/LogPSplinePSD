from typing import Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure


def plot_basis(
    basis: np.ndarray, axes: np.ndarray | None = None, fname=None
) -> Tuple[Figure, np.ndarray]:
    """Plot the basis functions, and a histogram of the basis values"""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    ax = axes[0]
    for b in basis.T:
        ax.plot(b)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Basis Value")

    # min non-zero value for histogram'
    basis_vals = basis.ravel()
    min_b = np.min(basis_vals[basis_vals > 0])
    max_b = np.max(basis_vals)
    min_b = np.max([min_b, 1e-1])  # avoid log(0)

    ax = axes[1]
    ax.hist(
        basis_vals,
        bins=np.geomspace(min_b, max_b, 50),
        density=True,
        alpha=0.7,
    )
    ax.set_xlabel("Basis Value")
    ax.set_xscale("log")
    # add a textbox of the sparsity of the basis
    sparsity = np.mean(basis == 0)
    ax.text(
        0.05,
        0.95,
        f"Sparsity: {sparsity:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    plt.tight_layout()

    fig = cast(Figure, axes[0].figure)

    if fname is not None:
        plt.savefig(fname)
        plt.close(fig)

    return fig, axes


def plot_penalty(
    penalty: np.ndarray, ax: plt.Axes | None = None
) -> Tuple[Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # plot the penalty matrix (BWR white for 0)
    cmap = plt.get_cmap("bwr")
    norm = TwoSlopeNorm(vmin=np.min(penalty), vcenter=0, vmax=np.max(penalty))
    ax.pcolormesh(
        penalty,
        cmap=cmap,
        shading="auto",
        norm=norm,
    )
    ax.set_xlabel("Basis Index")
    ax.set_ylabel("Basis Index")
    # add a colorbar to fig
    plt.colorbar(ax.collections[0], ax=ax)
    fig = cast(Figure, ax.figure)
    return fig, ax


def plot_spline_basis(model, outdir: str | None = None):
    """
    Visualize B-spline basis functions and penalty matrix structure.

    Creates a three-panel plot showing:
    1. Individual B-spline basis functions
    2. Basis function overview
    3. Penalty matrix structure (sparsity pattern)

    Parameters
    ----------
    outdir : str, optional
        Directory to save the plot. If None, displays interactively

    Examples
    --------
    >>> plot_spline_basis(model,)  # Display plot
    >>> plot_spline_basis(model,outdir="./diagnostics")  # Save to file
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_basis(np.asarray(model.basis), axes=axes[:2])
    plot_penalty(np.asarray(model.penalty_matrix), ax=axes[2])
    plt.tight_layout()
    if outdir is not None:
        fig.savefig(f"{outdir}/basis_plot.png", bbox_inches="tight")
