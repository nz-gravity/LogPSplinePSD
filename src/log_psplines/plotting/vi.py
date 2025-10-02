"""Plotting helpers for variational-inference diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .pdgrm import plot_pdgrm


def plot_vi_elbo(losses: np.ndarray, guide_name: str, outfile: str) -> None:
    """Plot the ELBO trace recorded during SVI optimisation."""
    if losses.size == 0:
        return

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    steps = np.arange(losses.size)
    ax.plot(steps, losses, color="tab:blue", lw=1.25)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("ELBO")
    ax.set_title(f"VI loss ({guide_name})")
    ax.grid(True, alpha=0.3, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_vi_initial_psd_univariate(
    *,
    outfile: str,
    periodogram,
    spline_model,
    weights: np.ndarray,
    true_psd: Optional[np.ndarray] = None,
) -> None:
    """Plot the PSD implied by the VI mean weights for the univariate model."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    fig, ax = plot_pdgrm(
        pdgrm=periodogram,
        spline_model=spline_model,
        weights=weights,
        true_psd=true_psd,
        show_knots=False,
        show_parametric=True,
        model_label="VI mean",
    )
    ax.set_title("VI initial PSD")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_vi_initial_psd_matrix(
    *,
    outfile: str,
    freq: np.ndarray,
    vi_psd: np.ndarray,
    empirical_psd: Optional[np.ndarray] = None,
    true_psd: Optional[np.ndarray] = None,
) -> None:
    """Plot diagonal auto-spectra implied by VI means for multivariate models."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    n_channels = vi_psd.shape[1]
    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(5.0, 2.6 * n_channels),
        sharex=True,
        squeeze=False,
    )

    for i in range(n_channels):
        ax = axes[i, 0]
        ax.plot(freq, vi_psd[:, i, i].real, label="VI mean", color="tab:blue")
        if empirical_psd is not None:
            ax.plot(
                freq,
                empirical_psd[:, i, i].real,
                label="Empirical",
                color="0.6",
                linestyle="--",
            )
        if true_psd is not None:
            ax.plot(
                freq,
                true_psd[:, i, i].real,
                label="True",
                color="tab:red",
                linestyle="-",
                linewidth=1.1,
            )
        ax.set_yscale("log")
        ax.set_ylabel(f"PSD[{i},{i}]")
        ax.grid(True, alpha=0.3, linewidth=0.8)
        if i == 0:
            ax.legend(frameon=False, ncol=3, fontsize="small")

    axes[-1, 0].set_xlabel("Frequency")
    fig.suptitle("VI initial PSD (diagonals)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
