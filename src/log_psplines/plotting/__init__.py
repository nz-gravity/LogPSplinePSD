from typing import Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from numpyro.infer import MCMC

from .pdgrm import plot_pdgrm

__all__ = ["plot_pdgrm", "plot_trace"]


def plot_trace(mcmc: MCMC, fname=None):
    inf_obj = az.from_numpyro(mcmc)
    ax = az.plot_trace(inf_obj)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
        plt.close(ax.flatten()[0].figure)


def plot_basis(basis: np.ndarray, fname=None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the basis functions, and a histogram of the basis values"""
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    ax = axes[0]
    for b in basis.T:
        ax.plot(b)
    print(basis.T.shape)
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

    if fname is not None:
        plt.savefig(fname)
        plt.close(fig)

    return fig, ax
