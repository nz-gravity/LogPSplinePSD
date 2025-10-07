"""Plotting helpers for variational-inference diagnostics."""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .base import COLORS, PlotConfig, safe_plot, setup_plot_style
from .pdgrm import plot_pdgrm
from .psd_matrix import plot_psd_matrix

# Setup consistent styling for VI plots
setup_plot_style()


def plot_vi_elbo(
    losses: np.ndarray,
    guide_name: str,
    outfile: str,
    loss_components: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Plot the ELBO trace recorded during SVI optimisation.

    Args:
        losses: Main ELBO loss values
        guide_name: Name of the VI guide
        outfile: Output file path
        loss_components: Optional dictionary of loss component names to arrays
                        (useful for multivariate VI with multiple loss terms)
    """
    if losses.size == 0:
        return

    # Use consistent styling - larger figure for multiple components
    fig_width = 8.0 if loss_components else 6.0
    config = PlotConfig(figsize=(fig_width, 5.0), fontsize=10)

    fig, ax = plt.subplots(figsize=config.figsize)
    steps = np.arange(losses.size)

    # Plot main ELBO loss
    ax.plot(
        steps,
        losses,
        color=COLORS["model"],
        lw=2,
        alpha=0.8,
        label="Total ELBO",
    )

    # Plot loss components if provided (useful for multivariate VI)
    if loss_components:
        component_colors = [
            COLORS["real"],
            COLORS["imag"],
            "purple",
            "brown",
            "pink",
        ]
        for i, (comp_name, comp_losses) in enumerate(loss_components.items()):
            if comp_losses.size == losses.size:  # Ensure same length
                color = component_colors[i % len(component_colors)]
                ax.plot(
                    steps,
                    comp_losses,
                    color=color,
                    lw=1.5,
                    alpha=0.7,
                    label=f"{comp_name}",
                )

    ax.set_xlabel("SVI Step", fontsize=config.labelsize)
    ax.set_ylabel("ELBO", fontsize=config.labelsize)
    ax.set_title(f"VI Convergence: {guide_name}", fontsize=config.titlesize)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(frameon=False, loc="best")

    # Add final statistics
    final_loss = losses[-1]
    loss_range = losses.max() - losses.min()
    stats_text = (
        f"Total ELBO:\nFinal: {final_loss:.2f}\nRange: {loss_range:.2f}"
    )

    # Add component statistics if available
    if loss_components:
        stats_text += "\n\nComponents:"
        for comp_name, comp_losses in loss_components.items():
            if comp_losses.size > 0:
                comp_final = comp_losses[-1]
                comp_range = comp_losses.max() - comp_losses.min()
                stats_text += f"\n{comp_name}: {comp_final:.2f} (range: {comp_range:.2f})"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        verticalalignment="top",
        fontfamily="monospace",
    )

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
    # Validate inputs
    if periodogram is None:
        raise ValueError("periodogram is required for VI univariate plotting")
    if weights is None:
        raise ValueError("weights are required for VI univariate plotting")

    # For VI plotting, we can work with just the quantiles if available
    # If no spline_model, we'll rely on the quantiles for the model estimate
    if spline_model is None and psd_quantiles is None:
        raise ValueError(
            "Either spline_model or psd_quantiles must be provided for VI plotting"
        )

    # Use the shared plotting function with VI-specific styling
    fig, ax = plot_pdgrm(
        pdgrm=periodogram,
        spline_model=spline_model,
        weights=weights,
        true_psd=true_psd,
        show_knots=False,
        show_parametric=bool(
            spline_model
        ),  # Only show parametric if spline_model exists
        model_label="VI Mean",
        model_color=COLORS["model"],
        data_color=COLORS["data"],
        model_ci=(
            np.array(
                [
                    psd_quantiles.get("q05") if psd_quantiles else None,
                    psd_quantiles.get("q50") if psd_quantiles else None,
                    psd_quantiles.get("q95") if psd_quantiles else None,
                ]
            )
            if psd_quantiles
            else None
        ),
    )

    # Customize title and styling for VI context
    ax.set_title(
        "Variational Inference: Initial PSD Estimate",
        fontsize=14,
        fontweight="bold",
    )

    # Add VI-specific annotations if quantiles are available
    if psd_quantiles:
        ax.text(
            0.02,
            0.98,
            "VI Posterior Quantiles (5-50-95%)",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            verticalalignment="top",
        )

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
    # Validate inputs
    if psd_quantiles is None:
        raise ValueError(
            "psd_quantiles must be provided for VI matrix plotting"
        )

    # Convert quantiles to CI format using shared utilities
    ci_dict = _pack_ci_from_quantiles(
        psd_quantiles=psd_quantiles,
        coherence_quantiles=coherence_quantiles,
        show_coherence=show_coherence,
    )

    # Use the shared plotting function with VI-specific styling
    plot_psd_matrix(
        outdir=os.path.dirname(outfile),
        filename=os.path.basename(outfile),
        freq=freq,
        empirical_psd=empirical_psd,
        true_psd=true_psd,
        ci_dict=ci_dict,
        show_coherence=show_coherence,
        dpi=150,  # Use consistent DPI
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
