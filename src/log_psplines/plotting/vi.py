"""Plotting helpers for variational-inference diagnostics."""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .pdgrm import plot_pdgrm
from .psd_matrix import _pack_ci_dict, plot_psd_matrix


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
        model_ci=np.array(
            [
                psd_quantiles.get("q05"),
                psd_quantiles.get("q50"),
                psd_quantiles.get("q95"),
            ]
        ),
    )

    ax.set_title("VI initial PSD")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_vi_initial_psd_matrix(
    *,
    outfile: str,
    freq: np.ndarray,
    empirical_psd: Optional[np.ndarray] = None,
    true_psd: Optional[np.ndarray] = None,
    psd_quantiles: Optional[Dict[str, np.ndarray]] = None,
    coherence_quantiles: Optional[Dict[str, np.ndarray]] = None,
    show_coherence: bool = True,
) -> None:
    """Plot diagonal auto-spectra implied by VI means for multivariate models."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    ci_dict = _pack_ci_from_quantiles(
        psd_quantiles=psd_quantiles,
        coherence_quantiles=coherence_quantiles,
        show_coherence=show_coherence,
    )
    plot_psd_matrix(
        outdir=os.path.dirname(outfile),
        filename=os.path.basename(outfile),
        freq=freq,
        empirical_psd=empirical_psd,
        true_psd=true_psd,
        ci_dict=ci_dict,
        show_coherence=show_coherence,
    )


def _pack_ci_from_quantiles(
    psd_quantiles: dict | None = None,
    coherence_quantiles: dict | None = None,
    show_coherence: bool = True,
) -> dict:
    """
    Construct a ci_dict (same format as _pack_ci_dict) from precomputed quantiles.

    Args:
        psd_quantiles: dict with keys "q05", "q50", "q95", each shaped (n_freq, n_channels, n_channels)
        coherence_quantiles: dict with keys "q05", "q50", "q95", each shaped (n_freq, n_channels, n_channels)
        show_coherence: if True, populate "coh" entries; else populate "re"/"im"

    Returns:
        ci_dict in the same format as _pack_ci_dict()
    """
    ci_dict = {"psd": {}, "coh": {}, "re": {}, "im": {}}

    # Guard against missing input
    if psd_quantiles is None:
        raise ValueError("psd_quantiles must be provided.")

    q05 = psd_quantiles.get("q05")
    q50 = psd_quantiles.get("q50")
    q95 = psd_quantiles.get("q95")

    n_freq, n_channels, _ = q05.shape

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                ci_dict["psd"][(i, i)] = (
                    q05[:, i, i],
                    q50[:, i, i],
                    q95[:, i, i],
                )

            elif show_coherence and coherence_quantiles is not None and i > j:
                coh_q05 = coherence_quantiles["q05"][:, i, j]
                coh_q50 = coherence_quantiles["q50"][:, i, j]
                coh_q95 = coherence_quantiles["q95"][:, i, j]
                ci_dict["coh"][(i, j)] = (coh_q05, coh_q50, coh_q95)

            elif not show_coherence:
                # use precomputed real/imag quantiles if available
                re_q05 = np.real(q05[:, i, j])
                re_q50 = np.real(q50[:, i, j])
                re_q95 = np.real(q95[:, i, j])
                im_q05 = np.imag(q05[:, i, j])
                im_q50 = np.imag(q50[:, i, j])
                im_q95 = np.imag(q95[:, i, j])
                ci_dict["re"][(i, j)] = (re_q05, re_q50, re_q95)
                ci_dict["im"][(i, j)] = (im_q05, im_q50, im_q95)

    return ci_dict
