"""Plotting helpers for variational-inference diagnostics."""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..datatypes.multivar import EmpiricalPSD
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

    # Normalize losses for log scale: shift to be positive while preserving relative changes
    # Find minimum across main losses and all components
    min_loss = losses.min()
    if loss_components:
        for comp_losses in loss_components.values():
            min_loss = min(min_loss, comp_losses.min())

    # Shift so minimum is slightly above zero for log scale
    shift_value = min_loss - 0.1 * np.abs(min_loss) if min_loss != 0 else -1.0
    shifted_losses = losses - shift_value

    # Plot main ELBO loss
    ax.plot(
        steps,
        shifted_losses,
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
                shifted_comp_losses = comp_losses - shift_value
                color = component_colors[i % len(component_colors)]
                ax.plot(
                    steps,
                    shifted_comp_losses,
                    color=color,
                    lw=1.5,
                    alpha=0.7,
                    label=f"{comp_name}",
                )

    ax.set_xlabel("SVI Step", fontsize=config.labelsize)
    ax.set_ylabel("ELBO (relative)", fontsize=config.labelsize)
    ax.set_yscale("log")
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
    empirical_psd: Optional[EmpiricalPSD] = None,
    true_psd: Optional[np.ndarray] = None,
    psd_quantiles: Optional[Dict[str, np.ndarray]] = None,
    coherence_quantiles: Optional[Dict[str, np.ndarray]] = None,
    show_coherence: bool = True,
    show_csd_magnitude: bool = False,
    **plot_kwargs,
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
        show_csd_magnitude=show_csd_magnitude,
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
        show_csd_magnitude=show_csd_magnitude,
        dpi=150,  # Use consistent DPI
        **plot_kwargs,
    )


def _pack_ci_from_quantiles(
    psd_quantiles: dict | None = None,
    coherence_quantiles: dict | None = None,
    show_coherence: bool = True,
    show_csd_magnitude: bool = False,
) -> dict:
    """
    Construct a ci_dict (same format as _pack_ci_dict) from precomputed quantiles.

    Args:
        psd_quantiles: dict with keys "q05", "q50", "q95", each shaped (N, p, p)
        coherence_quantiles: dict with keys "q05", "q50", "q95", each shaped (N, p, p)
        show_coherence: if True, populate "coh" entries
        show_csd_magnitude: if True, populate "mag" entries with |CSD_ij| bands

    Returns:
        ci_dict in the same format as _pack_ci_dict()
    """
    ci_dict = {"psd": {}, "coh": {}, "re": {}, "im": {}, "mag": {}}

    # Guard against missing input
    if psd_quantiles is None:
        raise ValueError("psd_quantiles must be provided.")

    real_q = psd_quantiles.get("real")
    imag_q = psd_quantiles.get("imag")
    if real_q is None:
        raise ValueError("psd_quantiles must include 'real' entries.")

    q05 = real_q.get("q05")
    q50 = real_q.get("q50")
    q95 = real_q.get("q95")

    N, p, _ = q05.shape

    for i in range(p):
        for j in range(p):
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

            elif show_csd_magnitude and imag_q is not None and i > j:
                mag_q05 = np.sqrt(
                    np.maximum(
                        q05[:, i, j] ** 2 + imag_q["q05"][:, i, j] ** 2,
                        0.0,
                    )
                )
                mag_q50 = np.sqrt(
                    np.maximum(
                        q50[:, i, j] ** 2 + imag_q["q50"][:, i, j] ** 2,
                        0.0,
                    )
                )
                mag_q95 = np.sqrt(
                    np.maximum(
                        q95[:, i, j] ** 2 + imag_q["q95"][:, i, j] ** 2,
                        0.0,
                    )
                )
                ci_dict["mag"][(i, j)] = (mag_q05, mag_q50, mag_q95)

            elif (
                not show_coherence
                and not show_csd_magnitude
                and imag_q is not None
            ):
                ci_dict["re"][(i, j)] = (
                    q05[:, i, j],
                    q50[:, i, j],
                    q95[:, i, j],
                )
                ci_dict["im"][(i, j)] = (
                    imag_q["q05"][:, i, j],
                    imag_q["q50"][:, i, j],
                    imag_q["q95"][:, i, j],
                )

    return ci_dict


def save_vi_diagnostics_univariate(
    *,
    outdir: Optional[str],
    periodogram,
    spline_model,
    diagnostics: Optional[Dict[str, np.ndarray]],
) -> None:
    """Persist VI diagnostics for univariate samplers as soon as they are available."""

    if not diagnostics or outdir is None:
        return

    diagnostics_dir = os.path.join(outdir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    losses = diagnostics.get("losses")
    if losses is not None:
        losses_arr = np.asarray(losses)
        if losses_arr.ndim > 1:
            losses_arr = losses_arr.mean(axis=0)
        if losses_arr.size:
            plot_vi_elbo(
                losses=losses_arr,
                guide_name=diagnostics.get("guide", "vi"),
                outfile=os.path.join(diagnostics_dir, "vi_elbo_trace.png"),
            )

    weights = diagnostics.get("weights")
    if weights is not None:
        plot_vi_initial_psd_univariate(
            outfile=os.path.join(diagnostics_dir, "vi_initial_psd.png"),
            periodogram=periodogram,
            spline_model=spline_model,
            weights=weights,
            true_psd=diagnostics.get("true_psd"),
            psd_quantiles=diagnostics.get("psd_quantiles"),
        )


def save_vi_diagnostics_multivariate(
    *,
    outdir: Optional[str],
    freq: np.ndarray,
    empirical_psd: Optional[np.ndarray],
    diagnostics: Optional[Dict[str, np.ndarray]],
) -> None:
    """Persist VI diagnostics for multivariate samplers right after VI initialisation."""

    if not diagnostics or outdir is None:
        return

    diagnostics_dir = os.path.join(outdir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    losses = diagnostics.get("losses")
    loss_components = diagnostics.get("losses_per_block")

    component_dict: Optional[Dict[str, np.ndarray]] = None
    component_source = None
    if loss_components is not None:
        component_source = np.asarray(loss_components)
        if component_source.ndim == 2 and component_source.shape[1] > 0:
            component_dict = {
                f"block_{idx}": component_source[idx]
                for idx in range(component_source.shape[0])
            }
        elif component_source.ndim == 3 and component_source.shape[2] > 0:
            component_dict = {}
            for block_idx in range(component_source.shape[0]):
                mean_trace = component_source[block_idx].mean(axis=0)
                component_dict[f"block_{block_idx}"] = mean_trace

    if losses is not None:
        losses_arr = np.asarray(losses)
        if losses_arr.ndim == 1 and losses_arr.size:
            plot_kwargs = dict(
                losses=losses_arr,
                guide_name=diagnostics.get("guide", "vi"),
                outfile=os.path.join(diagnostics_dir, "vi_elbo_trace.png"),
            )
            if component_dict:
                plot_kwargs["loss_components"] = component_dict
            plot_vi_elbo(**plot_kwargs)
        elif losses_arr.ndim > 1 and losses_arr.shape[1] > 0:
            mean_loss = losses_arr.mean(axis=0)
            components = {
                f"run_{idx}": losses_arr[idx]
                for idx in range(losses_arr.shape[0])
            }
            if component_dict:
                components.update(component_dict)
            plot_vi_elbo(
                losses=mean_loss,
                guide_name=diagnostics.get("guide", "vi"),
                outfile=os.path.join(diagnostics_dir, "vi_elbo_trace.png"),
                loss_components=components if components else None,
            )
    elif component_dict:
        # No aggregate losses stored; fall back to plotting components only
        mean_loss = np.mean(np.vstack(list(component_dict.values())), axis=0)
        plot_vi_elbo(
            losses=mean_loss,
            guide_name=diagnostics.get("guide", "vi"),
            outfile=os.path.join(diagnostics_dir, "vi_elbo_trace.png"),
            loss_components=component_dict,
        )

    psd_quantiles = diagnostics.get("psd_quantiles")
    if psd_quantiles is not None:
        plot_vi_initial_psd_matrix(
            outfile=os.path.join(diagnostics_dir, "vi_initial_psd_matrix.png"),
            freq=freq,
            empirical_psd=empirical_psd,
            true_psd=diagnostics.get("true_psd"),
            psd_quantiles=psd_quantiles,
            coherence_quantiles=diagnostics.get("coherence_quantiles"),
            show_coherence=True,
            show_csd_magnitude=False,
        )
