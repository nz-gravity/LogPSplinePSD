"""Small variational-inference plotting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import arviz_plots as azp
import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..arviz_utils._datatree import require_dataset as _require_dataset


def _is_arviz_loo_source(source: Any) -> bool:
    return hasattr(source, "posterior") and hasattr(source, "log_likelihood")


def _extract_losses(source: xr.DataTree | Mapping[str, Any]) -> np.ndarray:
    if isinstance(source, xr.DataTree):
        try:
            vi_sample_stats = _require_dataset(source, "vi_sample_stats")
        except (KeyError, TypeError):
            return np.array([], dtype=float)
        if "losses" not in vi_sample_stats:
            return np.array([], dtype=float)
        return np.asarray(
            vi_sample_stats["losses"].values, dtype=float
        ).reshape(-1)

    losses = source.get("losses")
    if losses is None:
        return np.array([], dtype=float)
    return np.asarray(losses, dtype=float).reshape(-1)


def _extract_pareto_k(source: xr.DataTree | Mapping[str, Any]) -> np.ndarray:
    if _is_arviz_loo_source(source):
        loo_result = azs.loo(source, pointwise=True)
        return np.asarray(loo_result.pareto_k.values, dtype=float).reshape(-1)

    for key in ("pareto_k",):
        values = source.get(key)
        if values is not None:
            return np.asarray(values, dtype=float).reshape(-1)
    return np.array([], dtype=float)


def _plot_khat_with_arviz(
    source: Any,
    *,
    figure_size: tuple[float, float],
    title: str,
) -> plt.Figure:
    loo_result = azs.loo(source, pointwise=True)
    plot = azp.plot_khat(
        loo_result,
        backend="matplotlib",
        pc_kwargs={"figure_kwargs": {"figsize": figure_size}},
    )
    fig = plot.viz["figure"].item()
    fig.suptitle(title)
    return fig


def plot_vi_elbo(
    source: xr.DataTree | Mapping[str, Any],
    *,
    factor: str | None = None,
) -> plt.Figure:
    """Plot a VI ELBO/loss history."""

    losses = _extract_losses(source)
    if losses.size == 0:
        raise ValueError("No VI ELBO/loss history found.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(losses.size), losses, color="C0", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO / loss")
    title = "VI ELBO"
    if factor is not None:
        title = f"{title} (factor {factor})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_vi_elbo_factors(
    sources: Mapping[str, xr.DataTree | Mapping[str, Any]],
) -> plt.Figure:
    """Plot one ELBO panel per factor."""

    factors = list(sources.items())
    if not factors:
        raise ValueError("No factor VI diagnostics provided.")

    fig, axes = plt.subplots(
        len(factors),
        1,
        figsize=(8, max(3, 2.5 * len(factors))),
        squeeze=False,
        sharex=False,
    )

    for row, (factor, source) in enumerate(factors):
        losses = _extract_losses(source)
        if losses.size == 0:
            raise ValueError(
                f"No VI ELBO/loss history found for factor {factor}."
            )
        ax = axes[row, 0]
        ax.plot(np.arange(losses.size), losses, color="C0", linewidth=1.5)
        ax.set_title(f"Factor {factor}")
        ax.set_ylabel("ELBO / loss")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Iteration")
    fig.suptitle("VI ELBO by factor")
    fig.tight_layout()
    return fig


def plot_vi_pareto_k(
    source: xr.DataTree | Mapping[str, Any],
    *,
    factor: str | None = None,
) -> plt.Figure:
    """Plot a Pareto-k histogram and sorted-value diagnostic."""

    if _is_arviz_loo_source(source):
        title = "VI Pareto-k diagnostics"
        if factor is not None:
            title = f"{title} (factor {factor})"
        return _plot_khat_with_arviz(source, figure_size=(8, 4), title=title)

    pareto_k = _extract_pareto_k(source)
    pareto_k = pareto_k[np.isfinite(pareto_k)]
    if pareto_k.size == 0:
        raise ValueError("No Pareto-k values found for VI diagnostics.")

    sorted_k = np.sort(pareto_k)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(sorted_k, bins=min(20, max(5, sorted_k.size)), color="C0")
    axes[0].axvline(0.7, color="C3", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Pareto k")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Pareto k histogram")

    axes[1].plot(np.arange(sorted_k.size), sorted_k, marker="o", linestyle="")
    axes[1].axhline(0.7, color="C3", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Sorted observation")
    axes[1].set_ylabel("Pareto k")
    axes[1].set_title("Sorted Pareto k")

    title = "VI Pareto-k diagnostics"
    if factor is not None:
        title = f"{title} (factor {factor})"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_vi_pareto_k_factors(
    sources: Mapping[str, xr.DataTree | Mapping[str, Any]],
) -> plt.Figure:
    """Plot Pareto-k diagnostics with one row per factor."""

    factors = list(sources.items())
    if not factors:
        raise ValueError("No factor VI diagnostics provided.")

    fig, axes = plt.subplots(
        len(factors),
        2,
        figsize=(10, max(3, 3 * len(factors))),
        squeeze=False,
    )

    for row, (factor, source) in enumerate(factors):
        pareto_k = _extract_pareto_k(source)
        pareto_k = pareto_k[np.isfinite(pareto_k)]
        if pareto_k.size == 0:
            raise ValueError(f"No Pareto-k values found for factor {factor}.")

        sorted_k = np.sort(pareto_k)
        hist_ax = axes[row, 0]
        sort_ax = axes[row, 1]

        hist_ax.hist(
            sorted_k,
            bins=min(20, max(5, sorted_k.size)),
            color="C0",
        )
        hist_ax.axvline(0.7, color="C3", linestyle="--", linewidth=1)
        hist_ax.set_title(f"Factor {factor} histogram")
        hist_ax.set_xlabel("Pareto k")
        hist_ax.set_ylabel("Count")

        sort_ax.plot(
            np.arange(sorted_k.size), sorted_k, marker="o", linestyle=""
        )
        sort_ax.axhline(0.7, color="C3", linestyle="--", linewidth=1)
        sort_ax.set_title(f"Factor {factor} sorted")
        sort_ax.set_xlabel("Sorted observation")
        sort_ax.set_ylabel("Pareto k")

    fig.suptitle("VI Pareto-k by factor")
    fig.tight_layout()
    return fig
