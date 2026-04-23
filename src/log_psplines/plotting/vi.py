"""Plotting helpers for variational-inference diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np

from .base import setup_plot_style

# Setup consistent styling for VI plots
setup_plot_style()


def _as_loss_array(values: Any) -> np.ndarray:
    """Convert a loss-like object into a flat float64 array."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _nonempty_loss_array(values: Any) -> np.ndarray | None:
    """Return a flattened loss array, or None if it is empty."""
    arr = _as_loss_array(values)
    return arr if arr.size else None


def _compute_shift_value(loss_arr: np.ndarray) -> float:
    """Compute log-scale shift value for loss visualization."""
    min_loss = float(np.nanmin(loss_arr))
    return min_loss - 0.1 * abs(min_loss) if min_loss != 0 else -1.0


def _min_loss(arrays: Mapping[str, np.ndarray]) -> float | None:
    """Return the smallest finite loss value across a mapping of loss arrays."""
    min_value = float("inf")

    for arr in arrays.values():
        if arr.size == 0:
            continue
        min_value = min(min_value, float(np.nanmin(arr)))

    return None if min_value == float("inf") else min_value


def _save_or_return(fig: plt.Figure, outfile: str | None) -> plt.Figure | None:
    """Save and close a figure, or return it if no output path was provided."""
    if outfile is None:
        return fig

    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return None


def _normalize_vi_losses(
    losses: Any,
    guide_name: str | None = None,
) -> dict[str, np.ndarray] | None:
    """Normalize VI loss inputs into a factor-indexed mapping.

    The plotter treats every input as a mapping from label to loss trace. A
    single loss array is represented as a one-factor mapping.

    Handles:
    - Mapping with a ``losses_per_block`` key.
    - Mapping with a ``losses`` key.
    - Plain mapping from factor names to loss arrays.
    - Direct array-like loss input.
    """
    single_label = guide_name or "ELBO"

    if not isinstance(losses, Mapping):
        loss_arr = _nonempty_loss_array(losses)
        return {single_label: loss_arr} if loss_arr is not None else None

    losses_per_block = losses.get("losses_per_block")
    if losses_per_block is not None:
        normalized = {
            f"Factor {factor_idx}": arr
            for factor_idx, factor_losses in enumerate(
                cast(Any, losses_per_block)
            )
            if (arr := _nonempty_loss_array(factor_losses)) is not None
        }
        return normalized or None

    if "losses" in losses:
        loss_arr = _nonempty_loss_array(losses["losses"])
        return {single_label: loss_arr} if loss_arr is not None else None

    normalized = {
        str(factor): arr
        for factor, factor_losses in losses.items()
        if (arr := _nonempty_loss_array(factor_losses)) is not None
    }
    return normalized or None


def _normalize_loss_components(
    loss_components: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Normalize optional component traces into a non-empty array mapping."""
    if not loss_components:
        return {}

    return {
        str(name): arr
        for name, values in loss_components.items()
        if (arr := _nonempty_loss_array(values)) is not None
    }


def _add_component_stats_box(
    ax: plt.Axes,
    loss_components: Mapping[str, np.ndarray],
) -> None:
    """Add a small text box with final value and range for each component."""
    if not loss_components:
        return

    stats_lines = ["Components:"]
    for name, comp_arr in loss_components.items():
        comp_final = comp_arr[-1]
        comp_range = np.nanmax(comp_arr) - np.nanmin(comp_arr)
        stats_lines.append(
            f"{name}: {comp_final:.2f} (range: {comp_range:.2f})"
        )

    ax.text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        verticalalignment="top",
        fontfamily="monospace",
    )


def _plot_loss_traces(
    ax: plt.Axes,
    losses_by_factor: Mapping[str, np.ndarray],
    shift_value: float,
) -> None:
    """Plot all factor loss traces on one axis."""
    for label, loss_arr in losses_by_factor.items():
        steps = np.arange(loss_arr.size)
        ax.plot(
            steps,
            loss_arr - shift_value,
            lw=1.5,
            alpha=0.9,
            label=str(label),
        )


def _plot_component_traces(
    ax: plt.Axes,
    loss_components: Mapping[str, np.ndarray],
    shift_value: float,
    expected_size: int,
) -> None:
    """Plot component traces that have the same length as the main loss trace."""
    component_colors = ["tab:blue", "tab:orange", "purple", "brown", "pink"]
    steps = np.arange(expected_size)

    for idx, (name, comp_arr) in enumerate(loss_components.items()):
        if comp_arr.size != expected_size:
            continue

        ax.plot(
            steps,
            comp_arr - shift_value,
            color=component_colors[idx % len(component_colors)],
            lw=1.5,
            alpha=0.7,
            label=str(name),
        )


def _configure_loss_axis(ax: plt.Axes, title: str) -> None:
    """Apply common axis styling for VI loss plots."""
    ax.set_xlabel("VI Evaluation")
    ax.set_ylabel("ELBO (relative)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(frameon=False, loc="best")


def plot_vi_loss(
    losses: Any,
    guide_name: str | None = None,
    outfile: str | None = None,
    loss_components: Optional[dict[str, np.ndarray]] = None,
) -> plt.Figure | None:
    """Plot the ELBO trace recorded during SVI optimization.

    All inputs are normalized to a factor-indexed mapping before plotting. A
    single loss array is treated as a one-factor loss mapping.

    Args:
        losses: Main ELBO loss values, VI diagnostics mapping, or factor map.
        guide_name: Optional label for single-loss plots.
        outfile: Optional output file path.
        loss_components: Optional per-component loss traces. These are plotted
            only when their length matches the first main loss trace.

    Returns:
        Matplotlib Figure, or None if saved to file or no losses are available.
    """
    losses_by_factor = _normalize_vi_losses(losses, guide_name=guide_name)
    if losses_by_factor is None:
        return None

    normalized_components = _normalize_loss_components(loss_components)
    arrays_for_shift = {**losses_by_factor, **normalized_components}

    min_loss = _min_loss(arrays_for_shift)
    if min_loss is None:
        return None

    shift_value = _compute_shift_value(
        np.asarray([min_loss], dtype=np.float64)
    )

    has_components = bool(normalized_components)
    fig_width = 8.0 if has_components else 6.0
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    _plot_loss_traces(ax, losses_by_factor, shift_value)

    if has_components:
        first_loss = next(iter(losses_by_factor.values()))
        _plot_component_traces(
            ax,
            normalized_components,
            shift_value,
            expected_size=first_loss.size,
        )
        _add_component_stats_box(ax, normalized_components)

    if len(losses_by_factor) == 1:
        title = f"VI Convergence: {next(iter(losses_by_factor))}"
    else:
        title = "VI Convergence by Factor"

    _configure_loss_axis(ax, title)
    fig.tight_layout()

    return _save_or_return(fig, outfile)
