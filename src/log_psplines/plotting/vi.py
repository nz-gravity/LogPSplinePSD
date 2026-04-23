"""Plotting helpers for variational-inference diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, cast

import matplotlib.pyplot as plt
import numpy as np

from ..datatypes.multivar import EmpiricalPSD
from .base import COLORS, PlotConfig, setup_plot_style
from .pdgrm import plot_pdgrm
from .psd_matrix import PSDMatrixPlotSpec, plot_psd_matrix

# Setup consistent styling for VI plots
setup_plot_style()


def _build_vi_loss_figure(
    losses: np.ndarray,
    guide_name: str,
    loss_components: Optional[Dict[str, np.ndarray]] = None,
) -> plt.Figure | None:
    """Build a VI loss figure for reuse by save and aggregate helpers."""
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        return None

    fig_width = 8.0 if loss_components else 6.0
    config = PlotConfig(figsize=(fig_width, 5.0), fontsize=10)
    fig, ax = plt.subplots(figsize=config.figsize)
    steps = np.arange(losses.size)

    min_loss = float(np.nanmin(losses))
    if loss_components:
        for comp_losses in loss_components.values():
            comp_arr = np.asarray(comp_losses, dtype=np.float64)
            if comp_arr.size:
                min_loss = min(min_loss, float(np.nanmin(comp_arr)))

    shift_value = min_loss - 0.1 * abs(min_loss) if min_loss != 0 else -1.0
    ax.plot(
        steps,
        losses - shift_value,
        color="k",
        lw=2,
        alpha=0.9,
        label="ELBO",
    )

    if loss_components:
        component_colors = [
            COLORS["real"],
            COLORS["imag"],
            "purple",
            "brown",
            "pink",
        ]
        for i, (comp_name, comp_losses) in enumerate(loss_components.items()):
            comp_arr = np.asarray(comp_losses, dtype=np.float64)
            if comp_arr.size != losses.size:
                continue
            color = component_colors[i % len(component_colors)]
            ax.plot(
                steps,
                comp_arr - shift_value,
                color=color,
                lw=1.5,
                alpha=0.7,
                label=str(comp_name),
            )

    ax.set_xlabel("VI Evaluation", fontsize=config.labelsize)
    ax.set_ylabel("ELBO (relative)", fontsize=config.labelsize)
    ax.set_yscale("log")
    ax.set_title(f"VI Convergence: {guide_name}", fontsize=config.titlesize)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(frameon=False, loc="best")

    if loss_components:
        stats_text = "Components:"
        for comp_name, comp_losses in loss_components.items():
            comp_arr = np.asarray(comp_losses, dtype=np.float64)
            if comp_arr.size == 0:
                continue
            comp_final = comp_arr[-1]
            comp_range = comp_arr.max() - comp_arr.min()
            stats_text += (
                f"\n{comp_name}: {comp_final:.2f} "
                f"(range: {comp_range:.2f})"
            )

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
    return fig


def plot_vi_loss(
    losses: np.ndarray | Mapping[str, np.ndarray],
    guide_name: str | None = None,
    outfile: str | None = None,
    loss_components: Optional[Dict[str, np.ndarray]] = None,
) -> plt.Figure | None:
    """Plot the ELBO trace recorded during SVI optimisation.

    Args:
        losses: Main ELBO loss values (fine-grid VI), or factor -> losses.
        guide_name: Name of the VI guide for single-run plots.
        outfile: Optional output file path.
        loss_components: Optional per-block loss traces.
    """
    if isinstance(losses, Mapping):
        if not losses:
            return None
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for factor, factor_losses in losses.items():
            loss_arr = np.asarray(factor_losses, dtype=np.float64)
            if loss_arr.size == 0:
                continue
            steps = np.arange(loss_arr.size)
            min_loss = float(np.nanmin(loss_arr))
            shift_value = (
                min_loss - 0.1 * abs(min_loss) if min_loss != 0 else -1.0
            )
            ax.plot(
                steps,
                loss_arr - shift_value,
                lw=1.5,
                alpha=0.9,
                label=f"Factor {factor}",
            )
        ax.set_xlabel("VI Evaluation")
        ax.set_ylabel("ELBO (relative)")
        ax.set_yscale("log")
        ax.set_title("VI Convergence by Factor")
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        if outfile is not None:
            fig.savefig(outfile, dpi=150)
            plt.close(fig)
            return None
        return fig

    if guide_name is None:
        raise ValueError("guide_name is required for single VI loss plots")
    fig = _build_vi_loss_figure(
        losses=np.asarray(losses, dtype=np.float64),
        guide_name=guide_name,
        loss_components=loss_components,
    )
    if fig is None:
        return None
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        return None
    return fig


def plot_vi_initial_psd_univariate(
    *,
    outfile: str,
    periodogram,
    spline_model,
    weights: np.ndarray,
    true_psd: Optional[np.ndarray] = None,
    psd_quantiles: Optional[Dict[str, np.ndarray]] = None,
    coarse_vi_freq: Optional[np.ndarray] = None,
    coarse_vi_psd: Optional[np.ndarray] = None,
    coarse_vi_label: Optional[str] = None,
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
        model_label=(
            "VI Posterior Median (Fine Grid)"
            if psd_quantiles
            else "VI Mean (Fine Grid)"
        ),
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

    if coarse_vi_freq is not None and coarse_vi_psd is not None:
        ax.plot(
            coarse_vi_freq,
            coarse_vi_psd,
            color="tab:orange",
            ls="--",
            lw=1.5,
            alpha=0.8,
            label=coarse_vi_label or "Coarse-Grid VI Fit",
        )
        ax.legend(loc="best", frameon=False)

    # Customize title and styling for VI context
    ax.set_title(
        "VI Warm-Start Diagnostic",
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
    coarse_vi_freq: Optional[np.ndarray] = None,
    coarse_vi_psd: Optional[np.ndarray] = None,
    coarse_vi_label: Optional[str] = None,
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

    extra_psd: list[EmpiricalPSD] | None = None
    extra_labels: list[str] | None = None
    extra_styles: list[dict] | None = None
    if coarse_vi_freq is not None and coarse_vi_psd is not None:
        n_coarse, p, _ = coarse_vi_psd.shape
        coarse_empirical = EmpiricalPSD(
            freq=np.asarray(coarse_vi_freq),
            psd=np.asarray(coarse_vi_psd, dtype=np.complex128),
            coherence=np.zeros((n_coarse, p, p)),
        )
        extra_psd = [coarse_empirical]
        extra_labels = [coarse_vi_label or "Coarse-Grid VI Fit"]
        extra_styles = [
            {"color": "tab:orange", "ls": "--", "lw": 1.5, "alpha": 0.8}
        ]

    # Use the shared plotting function with VI-specific styling
    spec = PSDMatrixPlotSpec(
        outdir=os.path.dirname(outfile),
        filename=os.path.basename(outfile),
        freq=freq,
        label="VI Posterior Median (Fine Grid)",
        empirical_psd=empirical_psd,
        extra_empirical_psd=extra_psd,
        extra_empirical_labels=extra_labels,
        extra_empirical_styles=extra_styles,
        true_psd=true_psd,
        ci_dict=ci_dict,
        show_coherence=show_coherence,
        show_csd_magnitude=show_csd_magnitude,
        dpi=150,  # Use consistent DPI
        **plot_kwargs,
    )
    plot_psd_matrix(spec)


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
    ci_dict: dict[
        str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {
        "psd": {},
        "coh": {},
        "re": {},
        "im": {},
        "mag": {},
    }

    # Guard against missing input
    if psd_quantiles is None:
        raise ValueError("psd_quantiles must be provided.")

    posterior_psd_q = psd_quantiles.get("posterior_psd")
    if posterior_psd_q is None:
        raise ValueError("psd_quantiles must include 'posterior_psd'.")
    posterior_psd_q = np.asarray(posterior_psd_q, dtype=np.complex128)
    real_q = np.asarray(posterior_psd_q.real, dtype=np.float64)
    imag_q = np.asarray(posterior_psd_q.imag, dtype=np.float64)

    q05 = real_q[0]
    q50 = real_q[1]
    q95 = real_q[2]

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

            elif show_csd_magnitude and i > j:
                mag_q05 = np.sqrt(
                    np.maximum(
                        q05[:, i, j] ** 2 + imag_q[0, :, i, j] ** 2,
                        0.0,
                    )
                )
                mag_q50 = np.sqrt(
                    np.maximum(
                        q50[:, i, j] ** 2 + imag_q[1, :, i, j] ** 2,
                        0.0,
                    )
                )
                mag_q95 = np.sqrt(
                    np.maximum(
                        q95[:, i, j] ** 2 + imag_q[2, :, i, j] ** 2,
                        0.0,
                    )
                )
                ci_dict["mag"][(i, j)] = (mag_q05, mag_q50, mag_q95)

            elif not show_coherence and not show_csd_magnitude:
                ci_dict["re"][(i, j)] = (
                    q05[:, i, j],
                    q50[:, i, j],
                    q95[:, i, j],
                )
                ci_dict["im"][(i, j)] = (
                    imag_q[0, :, i, j],
                    imag_q[1, :, i, j],
                    imag_q[2, :, i, j],
                )

    return ci_dict
