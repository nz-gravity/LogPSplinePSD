"""Plotting helpers for variational-inference diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

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
    psd_quantiles: Optional[Dict[str, np.ndarray]] = None,
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

    if psd_quantiles:
        freq = np.asarray(periodogram.freqs)
        q05 = (
            np.asarray(psd_quantiles.get("q05"))
            if psd_quantiles.get("q05") is not None
            else None
        )
        q50 = (
            np.asarray(psd_quantiles.get("q50"))
            if psd_quantiles.get("q50") is not None
            else None
        )
        q95 = (
            np.asarray(psd_quantiles.get("q95"))
            if psd_quantiles.get("q95") is not None
            else None
        )
        if q05 is not None and q95 is not None:
            ax.fill_between(
                freq,
                q05,
                q95,
                alpha=0.25,
                color="tab:blue",
                label="VI 90% band",
            )
        if q50 is not None:
            ax.plot(
                freq,
                q50,
                color="tab:blue",
                linestyle="--",
                linewidth=1.0,
                label="VI median",
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
    psd_quantiles: Optional[Dict[str, np.ndarray]] = None,
    coherence_quantiles: Optional[Dict[str, np.ndarray]] = None,
    show_coherence: bool = True,
) -> None:
    """Plot diagonal auto-spectra implied by VI means for multivariate models."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    n_channels = vi_psd.shape[1]
    fig, axes = plt.subplots(
        n_channels,
        n_channels,
        figsize=(3.8 * n_channels, 3.8 * n_channels),
        squeeze=False,
    )

    q05 = psd_quantiles.get("q05") if psd_quantiles else None
    q50 = psd_quantiles.get("q50") if psd_quantiles else None
    q95 = psd_quantiles.get("q95") if psd_quantiles else None

    coh_q05 = coherence_quantiles.get("q05") if coherence_quantiles else None
    coh_q50 = coherence_quantiles.get("q50") if coherence_quantiles else None
    coh_q95 = coherence_quantiles.get("q95") if coherence_quantiles else None

    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j]

            if i == j:
                median = (
                    q50[:, i, i].real
                    if q50 is not None
                    else vi_psd[:, i, i].real
                )
                lower = q05[:, i, i].real if q05 is not None else None
                upper = q95[:, i, i].real if q95 is not None else None

                if lower is not None and upper is not None:
                    ax.fill_between(
                        freq,
                        lower,
                        upper,
                        color="tab:blue",
                        alpha=0.25,
                        label="VI 90% band",
                    )
                ax.plot(freq, median, color="tab:blue", label="VI central")

                if empirical_psd is not None:
                    ax.plot(
                        freq,
                        empirical_psd[:, i, i].real,
                        color="0.6",
                        linestyle="--",
                        label="Empirical",
                    )
                if true_psd is not None:
                    ax.plot(
                        freq,
                        true_psd[:, i, i].real,
                        color="tab:red",
                        linestyle="-",
                        linewidth=1.1,
                        label="True",
                    )
                ax.set_yscale("log")
                ax.set_ylabel(f"PSD[{i},{i}]")
                ax.grid(True, alpha=0.3, linewidth=0.8)
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize="small")

            elif show_coherence:
                if i > j:
                    if coh_q05 is not None and coh_q95 is not None:
                        ax.fill_between(
                            freq,
                            coh_q05[:, i, j],
                            coh_q95[:, i, j],
                            color="purple",
                            alpha=0.25,
                            label="VI 90% band",
                        )
                    coherence_median = (
                        coh_q50[:, i, j]
                        if coh_q50 is not None
                        else (
                            np.abs(vi_psd[:, i, j]) ** 2
                            / (
                                np.abs(vi_psd[:, i, i])
                                * np.abs(vi_psd[:, j, j])
                            )
                        )
                    )
                    ax.plot(
                        freq,
                        coherence_median,
                        color="purple",
                        label="VI coherence",
                    )

                    if empirical_psd is not None:
                        emp_coh = np.abs(empirical_psd[:, i, j]) ** 2 / (
                            np.abs(empirical_psd[:, i, i])
                            * np.abs(empirical_psd[:, j, j])
                        )
                        ax.plot(
                            freq,
                            emp_coh,
                            "k--",
                            alpha=0.3,
                            label="Empirical",
                        )

                    if true_psd is not None:
                        true_coh = np.abs(true_psd[:, i, j]) ** 2 / (
                            np.abs(true_psd[:, i, i])
                            * np.abs(true_psd[:, j, j])
                        )
                        ax.plot(
                            freq,
                            true_coh,
                            color="tab:red",
                            alpha=0.7,
                            label="True",
                        )
                    ax.set_ylim(0, 1)
                    ax.set_title(f"Coherence ({i},{j})")
                    ax.grid(True, alpha=0.3, linewidth=0.8)
                    if i == n_channels - 1:
                        ax.set_xlabel("Frequency")
                    if j == 0:
                        ax.legend(frameon=False, fontsize="small")
                else:
                    ax.axis("off")
            else:
                ax.axis("off")

    for i in range(n_channels):
        axes[i, i].set_xlabel("Frequency")

    fig.suptitle("VI initial PSD and coherence")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
