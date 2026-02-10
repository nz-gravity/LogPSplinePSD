import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from ..logger import logger
from ..plotting.base import PlotConfig, safe_plot, setup_plot_style
from .run_all import run_all_diagnostics

# Setup consistent styling for diagnostics plots
setup_plot_style()


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics plotting parameters."""

    figsize: tuple = (12, 8)
    dpi: int = 150
    ess_threshold: int = 400
    ess_per_chain_threshold: int = 100
    ess_tail_per_chain_threshold: int = 100
    rhat_threshold: float = 1.01
    ebfmi_threshold: float = 0.3
    tree_depth_hit_warn_frac: float = 0.1
    fontsize: int = 11
    labelsize: int = 12
    titlesize: int = 12
    # Trace plot performance guards (important for multivariate runs)
    max_trace_vars_per_group: int = 8
    max_trace_dims_per_var: int = 4
    max_trace_series_per_group: int = 12
    max_trace_draws: int = 2000
    trace_use_kde: bool = True
    trace_kde_max_points: int = 1200
    trace_max_kde_series_per_group: int = 4
    trace_max_elements_per_var: int = 500_000
    trace_max_total_elements: int = 5_000_000
    # Optional output control.
    save_acceptance: bool = False
    save_sampler_diagnostics: bool = False
    # Rank/pair plot guards for very high-dimensional posteriors.
    save_rank_plots: bool = False
    rank_max_vars: int = 6
    rank_max_dims_per_var: int = 6
    save_pair_plots: bool = False
    pair_max_vars: int = 4


def _choose_flat_indices(p: int, max_dim: int) -> np.ndarray:
    if p <= 0 or max_dim <= 0:
        return np.array([], dtype=int)
    if p <= max_dim:
        return np.arange(p, dtype=int)
    return np.unique(
        np.linspace(0, p - 1, num=max_dim, dtype=int, endpoint=True)
    )


def _thin_series(series: np.ndarray, max_points: int) -> np.ndarray:
    series = np.asarray(series)
    if max_points is None or max_points <= 0 or series.size <= max_points:
        return series
    step = int(np.ceil(series.size / max_points))
    return series[::step]


def _var_priority(name: str) -> tuple[int, str]:
    if name.startswith("delta"):
        return (0, name)
    if name.startswith("phi"):
        return (1, name)
    if name.startswith("weights"):
        return (2, name)
    return (3, name)


def _flat_dim_from_shape(shape: tuple[int, ...]) -> int:
    if len(shape) <= 2:
        return 1
    return int(np.prod(shape[2:]))


def _select_rank_plot_vars(
    idata: az.InferenceData, config: DiagnosticsConfig
) -> list[str]:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return []

    candidates: list[tuple[tuple[int, str], int, str]] = []
    for var in posterior.data_vars:
        var_name = str(var)
        try:
            flat_dim = _flat_dim_from_shape(tuple(posterior[var].shape))
        except Exception:
            continue
        if flat_dim <= 0 or flat_dim > config.rank_max_dims_per_var:
            continue
        candidates.append((_var_priority(var_name), flat_dim, var_name))

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return [name for _, _, name in candidates[: config.rank_max_vars]]


def _select_pair_plot_vars(
    idata: az.InferenceData, config: DiagnosticsConfig
) -> list[str]:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return []

    scalar_vars: list[tuple[tuple[int, str], str]] = []
    for var in posterior.data_vars:
        var_name = str(var)
        try:
            flat_dim = _flat_dim_from_shape(tuple(posterior[var].shape))
        except Exception:
            continue
        if flat_dim != 1:
            continue
        scalar_vars.append((_var_priority(var_name), var_name))

    scalar_vars.sort(key=lambda row: row[0])
    return [name for _, name in scalar_vars[: config.pair_max_vars]]


def plot_trace(
    idata: az.InferenceData,
    compact: bool = True,
    config: Optional[DiagnosticsConfig] = None,
) -> plt.Figure:
    """Lightweight trace/hist diagnostics for key parameter groups.

    Multivariate models can produce very high-dimensional parameter vectors.
    This function intentionally subsamples variables, dimensions, and draws to
    keep diagnostics fast and avoid generating enormous figures.
    """
    _ = compact  # Backwards-compatibility with older call sites.
    if config is None:
        config = DiagnosticsConfig()
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("InferenceData is missing posterior samples.")

    total_elements = 0
    try:
        total_elements = sum(
            int(posterior[var].size) for var in posterior.data_vars
        )
    except Exception:
        total_elements = 0
    if (
        config.trace_max_total_elements is not None
        and total_elements > config.trace_max_total_elements
    ):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "Trace plot skipped\n"
            f"Posterior size {total_elements:,} elements\n"
            f"Exceeds limit {config.trace_max_total_elements:,}",
            ha="center",
            va="center",
        )
        ax.set_title("Parameter Traces (skipped)")
        ax.axis("off")
        plt.tight_layout()
        return fig

    groups = {
        "delta": [
            v for v in posterior.data_vars if str(v).startswith("delta")
        ],
        "phi": [v for v in posterior.data_vars if str(v).startswith("phi")],
        "weights": [
            v for v in posterior.data_vars if str(v).startswith("weights")
        ],
    }

    fig, axes = plt.subplots(3, 2, figsize=(7, 9))

    for row, (group_name, var_names) in enumerate(groups.items()):
        ax_trace = axes[row, 0]
        ax_hist = axes[row, 1]

        ax_trace.set_title(
            f"{group_name.capitalize()} parameters", fontsize=12
        )
        ax_trace.set_ylabel(group_name, fontsize=9)
        ax_trace.set_xlabel("MCMC step", fontsize=9)
        ax_hist.set_title("Marginal", fontsize=12)
        ax_hist.set_xlabel(group_name, fontsize=9)
        ax_hist.set_yticks([])
        ax_hist.spines["left"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["top"].set_visible(False)
        ax_trace.spines["right"].set_visible(False)
        ax_trace.spines["top"].set_visible(False)

        series_list: list[tuple[str, np.ndarray]] = []
        skipped_for_size = 0
        used_kde = 0

        for var in list(var_names)[: config.max_trace_vars_per_group]:
            try:
                var_size = int(posterior[var].size)
            except Exception:
                continue
            if (
                config.trace_max_elements_per_var is not None
                and var_size > config.trace_max_elements_per_var
            ):
                skipped_for_size += 1
                continue
            try:
                values = np.asarray(posterior[var].values)
            except Exception:
                continue

            if values.ndim < 2:
                continue

            # (chain, draw, ...) -> (chain, draw, flat_dim)
            values = values.reshape((values.shape[0], values.shape[1], -1))
            idxs = _choose_flat_indices(
                int(values.shape[2]), config.max_trace_dims_per_var
            )
            if idxs.size == 0:
                continue

            for flat_idx in idxs:
                if len(series_list) >= config.max_trace_series_per_group:
                    break
                s = np.asarray(values[0, :, int(flat_idx)])
                s = s[np.isfinite(s)]
                if s.size == 0:
                    continue
                s = _thin_series(s, config.max_trace_draws)
                series_list.append((f"{var}[{int(flat_idx)}]", s))

            if len(series_list) >= config.max_trace_series_per_group:
                break

        if not series_list:
            if skipped_for_size > 0:
                msg = (
                    "Trace data skipped due to size "
                    f"(>{config.trace_max_elements_per_var:,} elements)"
                )
            else:
                msg = "No trace data available"
            ax_trace.text(
                0.5,
                0.5,
                msg,
                ha="center",
                va="center",
                transform=ax_trace.transAxes,
            )
            ax_hist.text(
                0.5,
                0.5,
                msg,
                ha="center",
                va="center",
                transform=ax_hist.transAxes,
            )
            continue

        if group_name in ("phi", "delta"):
            all_positive = all(np.all(s > 0) for _, s in series_list)
            if all_positive:
                ax_trace.set_yscale("log")
                ax_hist.set_xscale("log")

        for idx, (label, s) in enumerate(series_list):
            color = f"C{idx % 10}"
            ax_trace.plot(
                s, color=color, alpha=0.8, linewidth=1.0, label=label
            )

            if (
                group_name in ("phi", "delta")
                and ax_hist.get_xscale() == "log"
            ):
                lo = float(np.min(s))
                hi = float(np.max(s))
                if lo > 0 and hi > lo:
                    bins: int | np.ndarray = np.logspace(
                        np.log10(lo), np.log10(hi), 30
                    )
                else:
                    bins = 30
            else:
                bins = 30
            ax_hist.hist(s, bins=bins, density=True, color=color, alpha=0.25)

            if (
                config.trace_use_kde
                and used_kde < config.trace_max_kde_series_per_group
                and s.size <= config.trace_kde_max_points
            ):
                try:
                    if (
                        group_name in ("phi", "delta")
                        and ax_hist.get_xscale() == "log"
                    ):
                        log_s = np.log(s)
                        log_grid, log_pdf = az.kde(log_s)
                        grid = np.exp(log_grid)
                        pdf = log_pdf / grid
                    else:
                        grid, pdf = az.kde(s)
                    ax_hist.plot(grid, pdf, color=color, linewidth=1.2)
                    used_kde += 1
                except Exception:
                    pass

        if len(series_list) <= 6:
            ax_trace.legend(loc="best", fontsize="x-small")

    plt.suptitle("Parameter Traces (subsampled)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_diagnostics(
    idata: az.InferenceData,
    outdir: str,
    p: Optional[int] = None,
    N: Optional[int] = None,
    runtime: Optional[float] = None,
    config: Optional[DiagnosticsConfig] = None,
    *,
    summary_mode: Literal["off", "light", "full"] = "light",
    summary_position: Literal["start", "end"] = "end",
) -> None:
    """
    Create essential MCMC diagnostics in organized subdirectories.
    """
    if outdir is None:
        return

    if config is None:
        config = DiagnosticsConfig()

    # Create diagnostics subdirectory
    diag_dir = os.path.join(outdir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    logger.info("Generating MCMC diagnostics...")

    t0 = time.perf_counter()

    if summary_position == "start":
        t_summary = time.perf_counter()
        logger.info("Diagnostics step: summary text")
        generate_diagnostics_summary(idata, diag_dir, mode=summary_mode)
        logger.info(
            f"Diagnostics step: summary text done in {time.perf_counter() - t_summary:.2f}s"
        )

    t_plots = time.perf_counter()
    logger.info("Diagnostics step: plots")
    _create_diagnostic_plots(idata, diag_dir, config, p, N, runtime)
    logger.info(
        f"Diagnostics step: plots done in {time.perf_counter() - t_plots:.2f}s"
    )

    if summary_position == "end":
        t_summary = time.perf_counter()
        logger.info("Diagnostics step: summary text")
        generate_diagnostics_summary(idata, diag_dir, mode=summary_mode)
        logger.info(
            f"Diagnostics step: summary text done in {time.perf_counter() - t_summary:.2f}s"
        )

    logger.info(
        f"MCMC diagnostics finished in {time.perf_counter() - t0:.2f}s"
    )


def _create_diagnostic_plots(idata, diag_dir, config, p, N, runtime):
    """Create only the essential diagnostic plots."""
    logger.debug("Generating diagnostic plots...")

    # 1. ArviZ trace plots
    @safe_plot(f"{diag_dir}/trace_plots.png", config.dpi)
    def create_trace_plots():
        return plot_trace(idata, config=config)

    t = time.perf_counter()
    logger.info("Diagnostics plot: trace_plots.png starting")
    ok = create_trace_plots()
    logger.info(
        f"Diagnostics plot: trace_plots.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
    )

    # 2. Rank histogram diagnostics (subsampled variable list)
    t = time.perf_counter()
    logger.info("Diagnostics plot: rank_plots starting")
    _create_rank_diagnostics(idata, diag_dir, config)
    logger.info(
        f"Diagnostics plots: rank_plots done in {time.perf_counter() - t:.2f}s"
    )

    # 3. Optional pair diagnostics for low-dimensional scalar variables
    t = time.perf_counter()
    logger.info("Diagnostics plot: pair_plots starting")
    _create_pair_diagnostics(idata, diag_dir, config)
    logger.info(
        f"Diagnostics plots: pair_plots done in {time.perf_counter() - t:.2f}s"
    )

    # 4. Summary dashboard with key convergence metrics
    @safe_plot(f"{diag_dir}/summary_dashboard.png", config.dpi)
    def plot_summary():
        _plot_summary_dashboard(idata, config, p, N, runtime)

    t = time.perf_counter()
    logger.info("Diagnostics plot: summary_dashboard.png starting")
    ok = plot_summary()
    logger.info(
        f"Diagnostics plot: summary_dashboard.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
    )

    # 5. Log posterior diagnostics
    @safe_plot(f"{diag_dir}/log_posterior.png", config.dpi)
    def plot_lp():
        _plot_log_posterior(idata, config)

    t = time.perf_counter()
    logger.info("Diagnostics plot: log_posterior.png starting")
    ok = plot_lp()
    logger.info(
        f"Diagnostics plot: log_posterior.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
    )

    # 6. Energy diagnostics (E-BFMI + trace)
    @safe_plot(f"{diag_dir}/energy_diagnostics.png", config.dpi)
    def plot_energy():
        _plot_energy_diagnostics(idata, config)

    t = time.perf_counter()
    logger.info("Diagnostics plot: energy_diagnostics.png starting")
    ok = plot_energy()
    logger.info(
        f"Diagnostics plot: energy_diagnostics.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
    )

    # 7. Acceptance rate diagnostics
    if config.save_acceptance:

        @safe_plot(f"{diag_dir}/acceptance_diagnostics.png", config.dpi)
        def plot_acceptance():
            _plot_acceptance_diagnostics_blockaware(idata, config)

        t = time.perf_counter()
        logger.info("Diagnostics plot: acceptance_diagnostics.png starting")
        ok = plot_acceptance()
        logger.info(
            f"Diagnostics plot: acceptance_diagnostics.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
        )
    else:
        logger.info(
            "Diagnostics plot: acceptance_diagnostics skipped (disabled)."
        )

    # 8. Sampler-specific diagnostics
    if config.save_sampler_diagnostics:
        t = time.perf_counter()
        logger.info("Diagnostics plots: sampler-specific starting")
        _create_sampler_diagnostics(idata, diag_dir, config)
        logger.info(
            f"Diagnostics plots: sampler-specific done in {time.perf_counter() - t:.2f}s"
        )
    else:
        logger.info("Diagnostics plots: sampler-specific skipped (disabled).")

    # 9. Divergences diagnostics (for NUTS only)
    logger.info("Diagnostics plots: divergences skipped (disabled).")


def _create_rank_diagnostics(idata, diag_dir, config):
    if not config.save_rank_plots:
        logger.info("Diagnostics plot: rank_plots skipped (disabled).")
        return

    rank_vars = _select_rank_plot_vars(idata, config)
    if not rank_vars:
        logger.info(
            "Diagnostics plot: rank_plots skipped (no low-dimensional variables)."
        )
        return

    @safe_plot(f"{diag_dir}/rank_plots.png", config.dpi)
    def _plot_rank():
        az.plot_rank(idata, var_names=rank_vars)

    ok = _plot_rank()
    logger.info(
        "Diagnostics plot: rank_plots.png "
        + f"{'ok' if ok else 'failed'} for {len(rank_vars)} vars"
    )


def _create_pair_diagnostics(idata, diag_dir, config):
    if not config.save_pair_plots:
        logger.info("Diagnostics plot: pair_plots skipped (disabled).")
        return

    pair_vars = _select_pair_plot_vars(idata, config)
    if len(pair_vars) < 2:
        logger.info(
            "Diagnostics plot: pair_plots skipped (need >=2 scalar variables)."
        )
        return

    has_divergences = hasattr(idata, "sample_stats") and any(
        str(key).startswith("diverging") for key in idata.sample_stats
    )

    @safe_plot(f"{diag_dir}/pair_plot.png", config.dpi)
    def _plot_pair():
        az.plot_pair(
            idata,
            var_names=pair_vars,
            divergences=has_divergences,
            kind="scatter",
            marginals=True,
        )

    ok = _plot_pair()
    logger.info(
        "Diagnostics plot: pair_plot.png "
        + f"{'ok' if ok else 'failed'} for {len(pair_vars)} vars"
    )


def _plot_summary_dashboard(idata, config, p, N, runtime):

    # Create 2x2 layout
    fig, axes = plt.subplots(
        2, 2, figsize=(config.figsize[0] * 0.8, config.figsize[1])
    )
    ess_ax = axes[0, 0]
    meta_ax = axes[0, 1]
    param_ax = axes[1, 0]
    status_ax = axes[1, 1]

    # Get ESS values once
    ess_values = None
    try:
        ess = idata.attrs.get("ess")
        ess_values = ess[~np.isnan(ess)]
    except Exception:
        pass

    # 1. ESS Distribution
    _plot_ess_histogram(ess_ax, ess_values, config)

    # 2. Analysis Metadata
    _plot_metadata(meta_ax, idata, p, N, runtime)

    # 3. Parameter Summary
    _plot_parameter_summary(param_ax, idata)

    # 4. NUTS must-have checks
    _plot_convergence_status(status_ax, idata, ess_values, config)

    plt.tight_layout()


def _plot_ess_histogram(ax, ess_values, config):
    """Plot ESS distribution with quality thresholds."""
    if ess_values is None or len(ess_values) == 0:
        ax.text(0.5, 0.5, "ESS unavailable", ha="center", va="center")
        ax.set_title("ESS Distribution")
        return

    # Histogram
    ax.hist(ess_values, bins=30, alpha=0.7, edgecolor="black")

    # Reference lines
    thresholds = [
        (400, "red", "--", "Minimum reliable"),
        (1000, "orange", "--", "Good"),
        (np.max(ess_values), "green", ":", f"Max = {np.max(ess_values):.0f}"),
    ]

    for threshold, color, style, label in thresholds:
        ax.axvline(
            x=threshold,
            color=color,
            linestyle=style,
            linewidth=2 if threshold < np.max(ess_values) else 1,
            alpha=0.8,
            label=label,
        )

    ax.set_xlabel("ESS")
    ax.set_ylabel("Count")
    ax.set_title("ESS Distribution")
    ax.legend(loc="upper right", fontsize="x-small")
    ax.grid(True, alpha=0.3)

    # Summary stats
    pct_good = (ess_values >= config.ess_threshold).mean() * 100
    stats_text = f"Min: {ess_values.min():.0f}\nMean: {ess_values.mean():.0f}\n≥{config.ess_threshold}: {pct_good:.1f}%"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )


def _plot_metadata(ax, idata, p, N, runtime):
    """Display analysis metadata."""
    try:
        n_samples = idata.posterior.sizes.get("draw", 0)
        n_chains = idata.posterior.sizes.get("chain", 1)
        n_params = len(list(idata.posterior.data_vars))
        sampler_type = idata.attrs["sampler_type"]

        metadata_lines = [
            f"Sampler: {sampler_type}",
            f"Samples: {n_samples} × {n_chains} chains",
            f"Parameters: {n_params}",
        ]
        if p is not None:
            metadata_lines.append(f"Channels: {p}")
        if N is not None:
            metadata_lines.append(f"Frequencies: {N}")
        if runtime is not None:
            metadata_lines.append(f"Runtime: {runtime:.2f}s")

        ax.text(
            0.05,
            0.95,
            "\n".join(metadata_lines),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )
    except Exception:
        ax.text(0.5, 0.5, "Metadata unavailable", ha="center", va="center")

    ax.set_title("Analysis Summary")
    ax.axis("off")


def _plot_parameter_summary(ax, idata):
    """Display parameter count summary."""
    try:
        param_groups = _group_parameters_simple(idata)
        if param_groups:
            summary_text = "Parameter Summary:\n"
            for group_name, params in param_groups.items():
                if params:
                    summary_text += f"{group_name}: {len(params)}\n"
            ax.text(
                0.05,
                0.95,
                summary_text.strip(),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                fontfamily="monospace",
            )
    except Exception:
        ax.text(
            0.5,
            0.5,
            "Parameter summary\nunavailable",
            ha="center",
            va="center",
        )

    ax.set_title("Parameter Summary")
    ax.axis("off")


def _as_finite_scalar(value) -> Optional[float]:
    try:
        fval = float(value)
    except Exception:
        return None
    return fval if np.isfinite(fval) else None


def _compute_ebfmi_from_energy(energy_values: np.ndarray) -> Optional[float]:
    energy = _finite_1d(energy_values)
    if energy.size < 2:
        return None
    var_energy = float(np.var(energy))
    if var_energy < 1e-12:
        return None
    delta = np.diff(energy)
    return float(np.mean(delta * delta) / var_energy)


def _collect_must_have_metrics(idata, ess_values) -> dict:
    posterior = getattr(idata, "posterior", None)
    n_chains = (
        int(posterior.sizes.get("chain", 1)) if posterior is not None else 1
    )
    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)

    metrics: dict[str, object] = {"n_chains": n_chains}

    ess_bulk_min = _as_finite_scalar(attrs.get("mcmc_ess_bulk_min"))
    if ess_bulk_min is None and ess_values is not None and len(ess_values) > 0:
        ess_bulk_min = float(np.min(ess_values))
    metrics["ess_bulk_min"] = ess_bulk_min
    metrics["ess_bulk_per_chain"] = (
        ess_bulk_min / max(1, n_chains) if ess_bulk_min is not None else None
    )

    ess_tail_min = _as_finite_scalar(attrs.get("mcmc_ess_tail_min"))
    if ess_tail_min is None:
        try:
            ess_tail = attrs.get("ess_tail")
            if ess_tail is not None:
                ess_tail_vals = _finite_1d(np.asarray(ess_tail))
                if ess_tail_vals.size:
                    ess_tail_min = float(np.min(ess_tail_vals))
        except Exception:
            pass
    metrics["ess_tail_min"] = ess_tail_min
    metrics["ess_tail_per_chain"] = (
        ess_tail_min / max(1, n_chains) if ess_tail_min is not None else None
    )

    rhat_max = _as_finite_scalar(attrs.get("mcmc_rhat_max"))
    if rhat_max is None:
        try:
            rhat_attr = attrs.get("rhat")
            if rhat_attr is not None:
                rhat_vals = _finite_1d(np.asarray(rhat_attr))
                if rhat_vals.size:
                    rhat_max = float(np.max(rhat_vals))
        except Exception:
            pass
    metrics["rhat_max"] = rhat_max

    divergence_total = None
    divergence_fraction = None
    divergence_count = None
    if hasattr(idata, "sample_stats"):
        div_values = []
        for key in idata.sample_stats:
            if not str(key).startswith("diverging"):
                continue
            try:
                arr = _finite_1d(np.asarray(idata.sample_stats[key].values))
            except Exception:
                continue
            if arr.size:
                div_values.append(arr)
        if div_values:
            div = np.concatenate(div_values)
            divergence_total = float(np.sum(div))
            divergence_count = int(div.size)
            divergence_fraction = float(divergence_total / div.size)

    metrics["divergence_total"] = divergence_total
    metrics["divergence_fraction"] = divergence_fraction
    metrics["divergence_count"] = divergence_count

    max_tree_depth = _coerce_int(attrs.get("max_tree_depth"))
    tree_depth = None
    if hasattr(idata, "sample_stats") and "tree_depth" in idata.sample_stats:
        tree_depth = _finite_1d(np.asarray(idata.sample_stats["tree_depth"]))
    num_steps = None
    if hasattr(idata, "sample_stats"):
        if "num_steps" in idata.sample_stats:
            num_steps = _finite_1d(np.asarray(idata.sample_stats["num_steps"]))
        else:
            step_parts = []
            for ch in sorted(
                _get_channel_indices(idata.sample_stats, "num_steps")
            ):
                key = f"num_steps_channel_{ch}"
                try:
                    arr = _finite_1d(np.asarray(idata.sample_stats[key]))
                except Exception:
                    continue
                if arr.size:
                    step_parts.append(arr)
            if step_parts:
                num_steps = np.concatenate(step_parts)
    max_stats = _max_tree_depth_stats(num_steps, tree_depth, max_tree_depth)
    if max_stats:
        metrics["tree_hit_frac"] = float(max_stats["frac"])
        metrics["tree_message"] = _format_max_tree_depth_message(max_stats)

    ebfmi = _as_finite_scalar(attrs.get("energy_ebfmi_overall"))
    if ebfmi is None:
        try:
            candidates = [
                _as_finite_scalar(val)
                for key, val in attrs.items()
                if str(key).startswith("energy_ebfmi")
                and str(key).endswith("_overall")
            ]
            finite = [v for v in candidates if v is not None]
            if finite:
                ebfmi = float(min(finite))
        except Exception:
            pass
    if ebfmi is None and hasattr(idata, "sample_stats"):
        energy_values = []
        if "energy" in idata.sample_stats:
            energy_values.append(np.asarray(idata.sample_stats["energy"]))
        else:
            for key in idata.sample_stats:
                if str(key).startswith("energy_channel_"):
                    energy_values.append(np.asarray(idata.sample_stats[key]))
        if energy_values:
            ebfmi = _compute_ebfmi_from_energy(np.concatenate(energy_values))
    metrics["ebfmi"] = ebfmi

    return metrics


def _plot_convergence_status(ax, idata, ess_values, config):
    """Display NUTS must-have checks with pass/warn/fail statuses."""
    try:
        metrics = _collect_must_have_metrics(idata, ess_values)
        n_chains = int(metrics.get("n_chains", 1) or 1)

        lines = ["NUTS Must-Have Checks:", ""]
        fail_count = 0
        warn_count = 0

        def _append_check(label: str, state: str, detail: str) -> None:
            nonlocal fail_count, warn_count
            if state == "FAIL":
                fail_count += 1
            elif state == "WARN":
                warn_count += 1
            lines.append(f"{state:<5} {label:<12} {detail}")

        div_frac = metrics.get("divergence_fraction")
        div_total = metrics.get("divergence_total")
        div_n = metrics.get("divergence_count")
        if div_frac is None:
            _append_check("Divergences", "N/A", "Unavailable")
        elif float(div_frac) == 0.0:
            _append_check("Divergences", "PASS", "0 divergent transitions")
        else:
            detail = (
                f"{int(div_total)}/{int(div_n)} ({100.0 * float(div_frac):.2f}%)"
                if div_total is not None and div_n is not None
                else f"{100.0 * float(div_frac):.2f}%"
            )
            _append_check("Divergences", "FAIL", detail)

        tree_hit = metrics.get("tree_hit_frac")
        if tree_hit is None:
            _append_check("Tree depth", "N/A", "Unavailable")
        else:
            message = str(metrics.get("tree_message", ""))
            if float(tree_hit) == 0.0:
                _append_check("Tree depth", "PASS", message)
            elif float(tree_hit) <= config.tree_depth_hit_warn_frac:
                _append_check("Tree depth", "WARN", message)
            else:
                _append_check("Tree depth", "FAIL", message)

        ebfmi = metrics.get("ebfmi")
        if ebfmi is None:
            _append_check("E-BFMI", "N/A", "Unavailable")
        elif float(ebfmi) >= config.ebfmi_threshold:
            _append_check("E-BFMI", "PASS", f"{float(ebfmi):.3f}")
        elif float(ebfmi) >= 0.2:
            _append_check(
                "E-BFMI",
                "WARN",
                f"{float(ebfmi):.3f} (target>{config.ebfmi_threshold:.1f})",
            )
        else:
            _append_check(
                "E-BFMI",
                "FAIL",
                f"{float(ebfmi):.3f} (target>{config.ebfmi_threshold:.1f})",
            )

        rhat_max = metrics.get("rhat_max")
        if rhat_max is None:
            _append_check("R-hat", "N/A", "Unavailable")
        elif float(rhat_max) <= config.rhat_threshold:
            _append_check("R-hat", "PASS", f"max={float(rhat_max):.3f}")
        elif float(rhat_max) <= 1.05:
            _append_check("R-hat", "WARN", f"max={float(rhat_max):.3f}")
        else:
            _append_check("R-hat", "FAIL", f"max={float(rhat_max):.3f}")

        bulk_per_chain = metrics.get("ess_bulk_per_chain")
        if bulk_per_chain is None:
            _append_check("Bulk ESS", "N/A", "Unavailable")
        elif float(bulk_per_chain) >= config.ess_per_chain_threshold:
            _append_check(
                "Bulk ESS",
                "PASS",
                f"min/ch={float(bulk_per_chain):.1f} ({n_chains} chains)",
            )
        elif float(bulk_per_chain) >= 0.5 * config.ess_per_chain_threshold:
            _append_check(
                "Bulk ESS",
                "WARN",
                f"min/ch={float(bulk_per_chain):.1f} ({n_chains} chains)",
            )
        else:
            _append_check(
                "Bulk ESS",
                "FAIL",
                f"min/ch={float(bulk_per_chain):.1f} ({n_chains} chains)",
            )

        tail_per_chain = metrics.get("ess_tail_per_chain")
        if tail_per_chain is None:
            _append_check("Tail ESS", "N/A", "Unavailable")
        elif float(tail_per_chain) >= config.ess_tail_per_chain_threshold:
            _append_check(
                "Tail ESS",
                "PASS",
                f"min/ch={float(tail_per_chain):.1f} ({n_chains} chains)",
            )
        elif (
            float(tail_per_chain) >= 0.5 * config.ess_tail_per_chain_threshold
        ):
            _append_check(
                "Tail ESS",
                "WARN",
                f"min/ch={float(tail_per_chain):.1f} ({n_chains} chains)",
            )
        else:
            _append_check(
                "Tail ESS",
                "FAIL",
                f"min/ch={float(tail_per_chain):.1f} ({n_chains} chains)",
            )

        lines.append("")
        lines.append(
            "Visual checks: trace_plots.png, rank_plots.png"
            if config.save_rank_plots
            else "Visual checks: trace_plots.png (rank plots disabled)"
        )
        if config.save_pair_plots:
            lines.append(
                "Pair checks: pair_plot.png (low-dim scalar vars only)"
            )

        lines.append("")
        if fail_count > 0:
            lines.append(f"Overall: NEEDS ATTENTION ({fail_count} fail)")
            color = "red"
        elif warn_count > 0:
            lines.append(f"Overall: WATCH ({warn_count} warning)")
            color = "orange"
        else:
            lines.append("Overall: PASS")
            color = "green"

        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            fontfamily="monospace",
            color=color,
        )
    except Exception:
        ax.text(0.5, 0.5, "Status unavailable", ha="center", va="center")

    ax.set_title("Convergence Checklist")
    ax.axis("off")


def _plot_log_posterior(idata, config):
    """Log posterior trace diagnostics."""

    def _as_chain_draw(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        chain, draw = arr.shape[0], arr.shape[1]
        arr = arr.reshape(chain, draw, -1)
        return np.sum(arr, axis=-1)

    series: list[tuple[str, np.ndarray]] = []
    block_keys = sorted(
        [
            key
            for key in idata.sample_stats
            if str(key).startswith("log_likelihood_block_")
        ]
    )
    if block_keys:
        for key in block_keys:
            label = str(key).replace("log_likelihood_block_", "Block ")
            series.append(
                (f"Log Likelihood ({label})", idata.sample_stats[key].values)
            )
    elif "lp" in idata.sample_stats:
        series.append(("Log Posterior", idata.sample_stats["lp"].values))
    elif "log_likelihood" in idata.sample_stats:
        series.append(
            ("Log Likelihood", idata.sample_stats["log_likelihood"].values)
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=config.figsize)
        ax.text(
            0.5,
            0.5,
            "No log posterior\nor log likelihood\navailable",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_title("Log Posterior Trace")
        ax.axis("off")
        plt.tight_layout()
        return

    n_rows = len(series)
    height = max(config.figsize[1], 2.5 * n_rows)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(config.figsize[0], height), sharex=True
    )
    axes = np.atleast_1d(axes)

    for idx, (title_prefix, values) in enumerate(series):
        ax = axes[idx]
        values = _as_chain_draw(values)
        if values.shape[1] > config.max_trace_draws:
            step = int(np.ceil(values.shape[1] / config.max_trace_draws))
            values = values[:, ::step]

        for chain_idx in range(values.shape[0]):
            ax.plot(
                values[chain_idx],
                alpha=0.6,
                linewidth=1,
                label=f"chain {chain_idx}" if values.shape[0] > 1 else None,
            )

        ax.set_ylabel(title_prefix)
        ax.set_title(f"{title_prefix} Trace")
        if values.shape[0] > 1 or idx == 0:
            ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()

    plt.tight_layout()


def _finite_1d(values) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _resolve_target_accept(idata, default: float = 0.8) -> float:
    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)
    candidates = (
        attrs.get("target_accept_prob", None),
        attrs.get("target_accept_rate", None),
        default,
    )
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return default


def _rolling_std(series: np.ndarray, window: int = 20) -> np.ndarray:
    return np.array(
        [
            np.std(series[max(0, i - window) : i + 1])
            for i in range(series.size)
        ]
    )


def _extract_acceptance_series(
    idata, *, include_channels: bool
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], str, float, str]:
    accept_key = None
    if "accept_prob" in idata.sample_stats:
        accept_key = "accept_prob"
    elif "acceptance_rate" in idata.sample_stats:
        accept_key = "acceptance_rate"

    accept_rates = (
        _finite_1d(idata.sample_stats[accept_key].values)
        if accept_key is not None
        else np.array([])
    )
    trace_label = "overall" if accept_key is not None else "per-channel"

    channel_series: dict[int, np.ndarray] = {}
    channel_raw: dict[int, np.ndarray] = {}
    if include_channels:
        for key in idata.sample_stats:
            if isinstance(key, str) and key.startswith("accept_prob_channel_"):
                try:
                    ch = int(key.rsplit("_", 1)[-1])
                except (TypeError, ValueError):
                    continue
                raw = np.asarray(idata.sample_stats[key].values)
                series = _finite_1d(raw)
                if series.size:
                    channel_series[ch] = series
                    channel_raw[ch] = raw

    accept_hist = accept_rates
    if accept_rates.size == 0 and channel_series:
        accept_hist = np.concatenate(
            [channel_series[ch] for ch in sorted(channel_series)]
        )
        shapes = {arr.shape for arr in channel_raw.values()}
        if len(shapes) == 1:
            stacked = np.stack(
                [channel_raw[ch] for ch in sorted(channel_raw)], axis=0
            )
            mean_by_draw = np.mean(stacked, axis=0)
            accept_rates = _finite_1d(mean_by_draw)
            trace_label = "mean"

    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)
    sampler_type_attr = str(attrs.get("sampler_type", "unknown")).lower()
    sampler_type = "NUTS" if "nuts" in sampler_type_attr else "Sampler"
    target_rate = _resolve_target_accept(idata)
    return (
        accept_rates,
        accept_hist,
        channel_series,
        sampler_type,
        target_rate,
        trace_label,
    )


def _plot_acceptance_diagnostics_common(
    idata, config, *, include_channels: bool
) -> None:
    (
        accept_rates,
        accept_hist,
        channel_series,
        sampler_type,
        target_rate,
        trace_label,
    ) = _extract_acceptance_series(idata, include_channels=include_channels)
    if accept_rates.size == 0 and accept_hist.size == 0 and not channel_series:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(
            0.5,
            0.5,
            "Acceptance rate data unavailable",
            ha="center",
            va="center",
        )
        ax.set_title("Acceptance Rate Diagnostics")
        return

    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    good_range = (0.7, 0.9)
    low_range = (0.0, 0.6)
    high_range = (0.9, 1.0)
    concerning_range = (0.6, 0.7)

    axes[0, 0].axhspan(
        good_range[0],
        good_range[1],
        alpha=0.1,
        color="green",
        label=f"Good ({good_range[0]:.1f}-{good_range[1]:.1f})",
    )
    axes[0, 0].axhspan(
        low_range[0], low_range[1], alpha=0.1, color="red", label="Too low"
    )
    axes[0, 0].axhspan(
        high_range[0],
        high_range[1],
        alpha=0.1,
        color="orange",
        label="Too high",
    )
    if concerning_range[1] > concerning_range[0]:
        axes[0, 0].axhspan(
            concerning_range[0],
            concerning_range[1],
            alpha=0.1,
            color="yellow",
            label="Concerning",
        )

    if accept_rates.size:
        axes[0, 0].plot(
            accept_rates,
            alpha=0.8,
            linewidth=1,
            color="blue",
            label=trace_label,
        )

    if include_channels:
        for ch in sorted(channel_series):
            axes[0, 0].plot(
                channel_series[ch], alpha=0.6, linewidth=1, label=f"ch {ch}"
            )

    axes[0, 0].axhline(
        target_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_rate:.3g})",
    )

    if accept_rates.size:
        window_size = max(10, int(accept_rates.size // 50))
        if accept_rates.size > window_size:
            running_mean = np.convolve(
                accept_rates,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            axes[0, 0].plot(
                range(
                    window_size // 2,
                    window_size // 2 + len(running_mean),
                ),
                running_mean,
                alpha=0.9,
                linewidth=3,
                color="purple",
                label=f"Running mean (w={window_size})",
            )

    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Acceptance Rate")
    axes[0, 0].set_title(f"{sampler_type} Acceptance Rate Trace")
    axes[0, 0].legend(loc="best", fontsize="small")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(
        0.02,
        0.02,
        f"{sampler_type} aims for {target_rate:.2f}. Green: efficient sampling.",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    if accept_hist.size:
        axes[0, 1].hist(
            accept_hist,
            bins=30,
            alpha=0.7,
            density=True,
            edgecolor="black",
        )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No finite acceptance data",
            ha="center",
            va="center",
        )
    axes[0, 1].axvline(
        target_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_rate:.3g})",
    )
    axes[0, 1].set_xlabel("Acceptance Rate")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Acceptance Rate Distribution")
    axes[0, 1].legend(loc="best", fontsize="small")
    axes[0, 1].grid(True, alpha=0.3)

    if accept_rates.size > 10:
        window_std = _rolling_std(accept_rates, window=20)
        axes[1, 0].plot(window_std, alpha=0.7, color="green")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Rolling Std")
        axes[1, 0].set_title("Rolling Standard Deviation")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Acceptance variability\nanalysis unavailable",
            ha="center",
            va="center",
        )
        axes[1, 0].set_title("Acceptance Stability")

    stats_text = [
        f"Sampler: {sampler_type}",
        f"Target: {target_rate:.3f}",
    ]
    stats_series = accept_rates if accept_rates.size else accept_hist
    if stats_series.size:
        mean = float(np.mean(stats_series))
        std = float(np.std(stats_series))
        tail_size = max(1, stats_series.size // 4)
        cv = std / mean if abs(mean) > 1e-12 else np.nan
        if accept_rates.size == 0:
            stats_text.append(f"Trace: {trace_label} (no global accept_prob)")
        stats_text.extend(
            [
                f"Mean: {mean:.3f}",
                f"Std: {std:.3f}",
                (f"CV: {cv:.3f}" if np.isfinite(cv) else "CV: n/a (mean ~ 0)"),
                f"Min: {np.min(stats_series):.3f}",
                f"Max: {np.max(stats_series):.3f}",
                "",
                "Stability:",
                f"Final std: {np.std(stats_series[-tail_size:]):.3f}",
            ]
        )
    else:
        stats_text.append("No finite acceptance samples")

    if include_channels and channel_series:
        stats_text.append("")
        stats_text.append("Per-channel means:")
        for ch in sorted(channel_series):
            stats_text.append(f"  ch {ch}: {np.mean(channel_series[ch]):.3f}")

    axes[1, 1].text(
        0.05,
        0.95,
        "\n".join(stats_text),
        transform=axes[1, 1].transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )
    axes[1, 1].set_title("Acceptance Analysis")
    axes[1, 1].axis("off")

    plt.tight_layout()


def _plot_acceptance_diagnostics(idata, config):
    """Acceptance rate diagnostics for non-blocked sampler outputs."""
    _plot_acceptance_diagnostics_common(idata, config, include_channels=False)


def _plot_acceptance_diagnostics_blockaware(idata, config):
    """Acceptance diagnostics supporting blocked NUTS channel fields."""
    _plot_acceptance_diagnostics_common(idata, config, include_channels=True)


def _plot_energy_diagnostics(idata, config):
    """Energy trace diagnostics with E-BFMI summary."""
    if not hasattr(idata, "sample_stats"):
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(
            0.5,
            0.5,
            "No sample_stats available",
            ha="center",
            va="center",
        )
        ax.set_title("Energy Diagnostics")
        return

    energy_series: list[tuple[str, np.ndarray]] = []
    if "energy" in idata.sample_stats:
        energy = np.asarray(idata.sample_stats["energy"].values)
        energy_series.append(("overall", energy))
    else:
        for key in idata.sample_stats:
            if isinstance(key, str) and key.startswith("energy_channel_"):
                label = key.replace("energy_channel_", "ch ")
                energy_series.append(
                    (label, np.asarray(idata.sample_stats[key].values))
                )

    if not energy_series:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(
            0.5,
            0.5,
            "Energy data unavailable",
            ha="center",
            va="center",
        )
        ax.set_title("Energy Diagnostics")
        return

    ebfmi_lines = ["E-BFMI Summary:"]

    if len(energy_series) == 1:
        fig, axes = plt.subplots(1, 2, figsize=config.figsize)
        trace_axes = [axes[0]]
        summary_ax = axes[1]
        trace_titles = ["Energy Trace"]
    else:
        n_rows = len(energy_series)
        fig_height = max(config.figsize[1], 2.5 * n_rows)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(config.figsize[0], fig_height),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        trace_axes = [axes[idx, 0] for idx in range(n_rows)]
        summary_ax = axes[0, 1]
        for idx in range(1, n_rows):
            axes[idx, 1].axis("off")
        trace_titles = [
            f"Energy Trace ({label})" for label, _ in energy_series
        ]

    for idx, (label, energy) in enumerate(energy_series):
        energy_flat = np.asarray(energy).reshape(-1)
        energy_flat = energy_flat[np.isfinite(energy_flat)]
        if energy_flat.size == 0:
            continue
        energy_flat = _thin_series(energy_flat, config.max_trace_draws)
        trace_ax = trace_axes[idx]
        trace_ax.plot(energy_flat, alpha=0.7, linewidth=1.0, color="C0")
        trace_ax.set_xlabel("Iteration")
        trace_ax.set_ylabel("Energy")
        trace_ax.set_title(trace_titles[idx])
        trace_ax.grid(True, alpha=0.3)

        ebfmi_val = _compute_ebfmi_from_energy(energy_flat)
        if ebfmi_val is not None and np.isfinite(ebfmi_val):
            ebfmi_lines.append(f"  {label}: {ebfmi_val:.3f}")

    if len(ebfmi_lines) == 1:
        ebfmi_lines.append("  unavailable")

    summary_ax.text(
        0.05,
        0.95,
        "\n".join(ebfmi_lines),
        transform=summary_ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    summary_ax.set_title("E-BFMI")
    summary_ax.axis("off")

    plt.tight_layout()


def _get_channel_indices(sample_stats, base_key: str) -> set:
    """Return set of channel indices for the given ``base_key`` prefix."""

    prefix = f"{base_key}_channel_"
    indices = set()
    for key in sample_stats:
        if isinstance(key, str) and key.startswith(prefix):
            try:
                indices.add(int(key.replace(prefix, "")))
            except Exception:
                continue
    return indices


def _coerce_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _max_tree_depth_stats(
    num_steps: Optional[np.ndarray],
    tree_depth: Optional[np.ndarray],
    max_tree_depth: Optional[int],
) -> Optional[dict]:
    if tree_depth is not None and tree_depth.size:
        tree_depth = np.asarray(tree_depth)
        tree_depth = tree_depth[np.isfinite(tree_depth)]
        if tree_depth.size == 0:
            return None
        max_depth = (
            _coerce_int(max_tree_depth)
            if max_tree_depth is not None
            else int(np.max(tree_depth))
        )
        if max_depth is None:
            return None
        hits = int(np.sum(tree_depth >= max_depth))
        total = int(tree_depth.size)
        frac = float(hits / total) if total > 0 else 0.0
        max_steps = int(2**max_depth - 1) if max_depth is not None else None
        return {
            "frac": frac,
            "hits": hits,
            "total": total,
            "max_depth": max_depth,
            "max_steps": max_steps,
            "source": "tree_depth",
        }

    if num_steps is None or not np.size(num_steps):
        return None

    steps = np.asarray(num_steps)
    steps = steps[np.isfinite(steps)]
    if steps.size == 0:
        return None

    max_depth_from_steps = _coerce_int(max_tree_depth)
    max_steps_from_steps = (
        int(2**max_depth_from_steps - 1)
        if max_depth_from_steps is not None
        else None
    )
    if max_steps_from_steps is None:
        max_steps_from_steps = int(np.max(steps))
        max_depth_from_steps = (
            int(np.floor(np.log2(max_steps_from_steps + 1)))
            if max_steps_from_steps > 0
            else None
        )
    hits = int(np.sum(steps >= max_steps_from_steps))
    total = int(steps.size)
    frac = float(hits / total) if total > 0 else 0.0
    return {
        "frac": frac,
        "hits": hits,
        "total": total,
        "max_steps": max_steps_from_steps,
        "max_depth": max_depth_from_steps,
        "source": "num_steps",
    }


def _format_max_tree_depth_message(stats: dict) -> str:
    pct = stats["frac"] * 100
    hits = stats.get("hits")
    total = stats.get("total")
    count_msg = (
        f"{hits}/{total} ({pct:.1f}%)"
        if hits is not None and total is not None
        else f"{pct:.1f}%"
    )
    max_steps = stats.get("max_steps")
    max_depth = stats.get("max_depth")
    if max_steps is not None:
        return f"Max tree depth hits: {count_msg} (max steps {max_steps})"
    if max_depth is not None:
        return f"Max tree depth hits: {count_msg} (max depth {max_depth})"
    return f"Max tree depth hits: {count_msg}"


def _plot_nuts_diagnostics_blockaware(idata, config):
    """NUTS diagnostics supporting per‑channel (blocked) diagnostics fields.

    Overlays per‑channel series when keys like ``energy_channel_{j}`` or
    ``num_steps_channel_{j}`` are present.
    """
    # Presence of overall arrays
    has_energy = "energy" in idata.sample_stats
    has_potential = "potential_energy" in idata.sample_stats
    has_steps = "num_steps" in idata.sample_stats
    has_accept = "accept_prob" in idata.sample_stats
    has_step_size = "step_size" in idata.sample_stats
    has_tree_depth = "tree_depth" in idata.sample_stats

    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)
    max_tree_depth = _coerce_int(attrs.get("max_tree_depth"))
    tree_depth = (
        idata.sample_stats.tree_depth.values.flatten()
        if has_tree_depth
        else None
    )

    # Collect per-channel data
    def _collect(base):
        out = {}
        prefix = f"{base}_channel_"
        for key in idata.sample_stats:
            if isinstance(key, str) and key.startswith(prefix):
                try:
                    ch = int(key.replace(prefix, ""))
                    out[ch] = idata.sample_stats[key].values.flatten()
                except Exception:
                    pass
        return out

    energy_ch = _collect("energy")
    potential_ch = _collect("potential_energy")
    steps_ch = _collect("num_steps")
    accept_ch = _collect("accept_prob")
    step_size_ch = _collect("step_size")

    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    # Energy / potential
    ax = axes[0, 0]
    plotted = False
    if has_energy:
        ax.plot(
            idata.sample_stats.energy.values.flatten(),
            alpha=0.7,
            lw=1,
            label="H",
        )
        plotted = True
    if has_potential:
        ax.plot(
            idata.sample_stats.potential_energy.values.flatten(),
            alpha=0.7,
            lw=1,
            label="P",
        )
        plotted = True
    for ch in sorted(energy_ch):
        ax.plot(energy_ch[ch], alpha=0.5, lw=1, label=f"H ch {ch}")
        plotted = True
    for ch in sorted(potential_ch):
        ax.plot(potential_ch[ch], alpha=0.5, lw=1, label=f"P ch {ch}")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "Energy data\nunavailable", ha="center", va="center")
    ax.set_title("Energy Diagnostics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize="small")

    # Steps histogram
    ax = axes[0, 1]
    if has_steps:
        vals = idata.sample_stats.num_steps.values.flatten()
    else:
        vals = (
            np.concatenate(list(steps_ch.values()))
            if steps_ch
            else np.array([])
        )
    if vals.size:
        ax.hist(vals, bins=20, alpha=0.7, edgecolor="black")
        ax.set_title("Leapfrog Steps Distribution")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Trajectories")
        ax.grid(True, alpha=0.3)
        max_stats = _max_tree_depth_stats(vals, tree_depth, max_tree_depth)
        if max_stats and max_stats.get("max_steps") is not None:
            ax.axvline(
                max_stats["max_steps"],
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="max tree depth",
            )
            ax.legend(loc="best", fontsize="small")
        if max_stats:
            ax.text(
                0.02,
                0.98,
                _format_max_tree_depth_message(max_stats),
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                verticalalignment="top",
            )
    else:
        ax.text(0.5, 0.5, "Steps data\nunavailable", ha="center", va="center")

    # Acceptance (overlay per-channel)
    ax = axes[1, 0]
    plotted = False
    if has_accept:
        ax.plot(
            idata.sample_stats.accept_prob.values.flatten(),
            alpha=0.8,
            lw=1,
            label="overall",
        )
        plotted = True
    for ch in sorted(accept_ch):
        ax.plot(accept_ch[ch], alpha=0.6, lw=1, label=f"ch {ch}")
        plotted = True
    if not plotted:
        ax.text(
            0.5, 0.5, "Acceptance data\nunavailable", ha="center", va="center"
        )
    ax.set_title("Acceptance Trace")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("accept_prob")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize="small")

    # Summary text
    ax = axes[1, 1]
    lines = []
    if has_steps or steps_ch:
        lines.append("Steps summary:")
        if has_steps:
            s = idata.sample_stats.num_steps.values.flatten()
            lines.append(f"  overall μ={np.mean(s):.1f}, max={np.max(s):.0f}")
            max_stats = _max_tree_depth_stats(s, tree_depth, max_tree_depth)
            if max_stats:
                lines.append(
                    f"  overall {_format_max_tree_depth_message(max_stats)}"
                )
        for ch in sorted(steps_ch):
            s = steps_ch[ch]
            lines.append(f"  ch {ch} μ={np.mean(s):.1f}, max={np.max(s):.0f}")
            max_stats = _max_tree_depth_stats(s, None, max_tree_depth)
            if max_stats:
                lines.append(
                    f"  ch {ch} {_format_max_tree_depth_message(max_stats)}"
                )
        lines.append("")
    if has_accept or accept_ch:
        lines.append("Acceptance summary:")
        if has_accept:
            a = idata.sample_stats.accept_prob.values.flatten()
            lines.append(f"  overall μ={np.mean(a):.3f}")
        for ch in sorted(accept_ch):
            a = accept_ch[ch]
            lines.append(f"  ch {ch} μ={np.mean(a):.3f}")
        lines.append("")
    if has_step_size or step_size_ch:
        lines.append("Step size summary:")
        if has_step_size:
            ss = idata.sample_stats.step_size.values.flatten()
            if ss.size:
                lines.append(
                    f"  overall median={np.median(ss):.3g}, min={np.min(ss):.3g}"
                )
        for ch in sorted(step_size_ch):
            ss = step_size_ch[ch]
            if ss.size:
                lines.append(
                    f"  ch {ch} median={np.median(ss):.3g}, min={np.min(ss):.3g}"
                )
    if lines:
        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            family="monospace",
        )
    ax.set_title("NUTS Diagnostics Summary")
    ax.axis("off")

    plt.tight_layout()


def _plot_single_nuts_block(idata, config, channel_idx: int):
    """NUTS diagnostics for a single blocked channel."""

    def _get(key):
        full_key = f"{key}_channel_{channel_idx}"
        return (
            idata.sample_stats[full_key].values.flatten()
            if full_key in idata.sample_stats
            else None
        )

    energy = _get("energy")
    potential = _get("potential_energy")
    num_steps = _get("num_steps")
    step_size = _get("step_size")
    accept_prob = _get("accept_prob")
    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)
    max_tree_depth = _coerce_int(attrs.get("max_tree_depth"))

    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    # Energy traces
    ax = axes[0, 0]
    plotted = False
    if energy is not None:
        ax.plot(energy, alpha=0.7, lw=1, label="H")
        plotted = True
    if potential is not None:
        ax.plot(potential, alpha=0.7, lw=1, label="P")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "Energy data\nunavailable", ha="center", va="center")
    ax.set_title(f"Channel {channel_idx} Energy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize="small")

    # Acceptance trace
    ax = axes[0, 1]
    if accept_prob is not None:
        ax.axhspan(0.7, 0.9, alpha=0.1, color="green")
        ax.axhspan(0.0, 0.6, alpha=0.1, color="red")
        ax.axhspan(0.9, 1.0, alpha=0.1, color="orange")
        ax.plot(accept_prob, alpha=0.8, lw=1, color="purple")
        ax.axhline(0.8, color="red", linestyle="--", lw=1.5, label="target")
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, "Acceptance data\nunavailable", ha="center", va="center"
        )
    ax.set_title(f"Channel {channel_idx} Acceptance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("accept_prob")

    # Steps histogram
    ax = axes[1, 0]
    if num_steps is not None and num_steps.size:
        ax.hist(num_steps, bins=20, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Trajectories")
        ax.grid(True, alpha=0.3)
        max_stats = _max_tree_depth_stats(num_steps, None, max_tree_depth)
        if max_stats and max_stats.get("max_steps") is not None:
            ax.axvline(
                max_stats["max_steps"],
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="max tree depth",
            )
            ax.legend(loc="best", fontsize="small")
        if max_stats:
            ax.text(
                0.02,
                0.98,
                _format_max_tree_depth_message(max_stats),
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                verticalalignment="top",
            )
    else:
        ax.text(0.5, 0.5, "Steps data\nunavailable", ha="center", va="center")
    ax.set_title(f"Channel {channel_idx} Leapfrog Steps")

    # Summary stats
    ax = axes[1, 1]
    stats_lines = [f"Channel {channel_idx} summary:"]
    if energy is not None:
        stats_lines.append(
            f"  H μ={np.mean(energy):.2f}, σ={np.std(energy):.2f}"
        )
    if potential is not None:
        stats_lines.append(
            f"  P μ={np.mean(potential):.2f}, σ={np.std(potential):.2f}"
        )
    if num_steps is not None:
        stats_lines.append(
            f"  steps μ={np.mean(num_steps):.1f}, max={np.max(num_steps):.0f}"
        )
        max_stats = _max_tree_depth_stats(num_steps, None, max_tree_depth)
        if max_stats:
            stats_lines.append(
                f"  {_format_max_tree_depth_message(max_stats)}"
            )
    if accept_prob is not None:
        stats_lines.append(f"  accept μ={np.mean(accept_prob):.3f}")
    if step_size is not None and step_size.size:
        stats_lines.append(
            f"  step_size median={np.median(step_size):.3g}, min={np.min(step_size):.3g}"
        )

    ax.text(
        0.05,
        0.95,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top",
        family="monospace",
    )
    ax.axis("off")
    ax.set_title("Summary")

    plt.tight_layout()


def _create_sampler_diagnostics(idata, diag_dir, config):
    """Create sampler-specific diagnostics."""

    # Better sampler detection - check sampler type first
    sampler_type = (
        idata.attrs["sampler_type"].lower()
        if "sampler_type" in idata.attrs
        else "unknown"
    )

    # Check for NUTS-specific fields
    nuts_specific_fields = [
        "energy",
        "num_steps",
        "tree_depth",
        "diverging",
        "energy_error",
    ]

    has_nuts = (
        any(field in idata.sample_stats for field in nuts_specific_fields)
        or "nuts" in sampler_type
    )

    if has_nuts:

        @safe_plot(f"{diag_dir}/nuts_diagnostics.png", config.dpi)
        def plot_nuts():
            _plot_nuts_diagnostics_blockaware(idata, config)

        plot_nuts()

        # Per‑channel NUTS diagnostics for blocked samplers
        channel_indices = _get_channel_indices(
            idata.sample_stats, "accept_prob"
        )
        channel_indices |= _get_channel_indices(idata.sample_stats, "energy")
        channel_indices |= _get_channel_indices(
            idata.sample_stats, "potential_energy"
        )
        channel_indices |= _get_channel_indices(
            idata.sample_stats, "num_steps"
        )
        channel_indices |= _get_channel_indices(
            idata.sample_stats, "step_size"
        )

        for channel_idx in sorted(channel_indices):

            @safe_plot(
                f"{diag_dir}/nuts_block_{channel_idx}_diagnostics.png",
                config.dpi,
            )
            def plot_nuts_block(channel_idx=channel_idx):
                _plot_single_nuts_block(idata, config, channel_idx)

            plot_nuts_block()


def _create_divergences_diagnostics(idata, diag_dir, config):
    """Create divergences diagnostics for NUTS samplers."""
    # Check if divergences data exists
    has_divergences = "diverging" in idata.sample_stats
    has_channel_divergences = any(
        key.startswith("diverging_channel_") for key in idata.sample_stats
    )

    if not has_divergences and not has_channel_divergences:
        return  # Nothing to plot

    @safe_plot(f"{diag_dir}/divergences.png", config.dpi)
    def plot_divergences():
        _plot_divergences(idata, config)

    plot_divergences()


def _plot_divergences(idata, config):
    """Plot divergences diagnostics."""
    divergences_data: dict[str | int, np.ndarray] = {}

    if "diverging" in idata.sample_stats:
        divergences_data["main"] = (
            idata.sample_stats.diverging.values.flatten()
        )

    for key in idata.sample_stats:
        if key.startswith("diverging_channel_"):
            channel_idx = key.replace("diverging_channel_", "")
            divergences_data[int(channel_idx)] = idata.sample_stats[
                key
            ].values.flatten()

    if not divergences_data:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(
            0.5, 0.5, "No divergence data available", ha="center", va="center"
        )
        ax.set_title("Divergences Diagnostics")
        return

    fig, axes = plt.subplots(1, 2, figsize=config.figsize)
    trace_ax, summary_ax = axes

    total_divergences = 0
    total_iterations = 0

    for label, div_values in divergences_data.items():
        div = np.asarray(div_values).reshape(-1)
        div = div[np.isfinite(div)]
        if div.size == 0:
            continue
        cumulative = np.cumsum(div.astype(int))
        label_str = "overall" if label == "main" else f"ch {label}"
        trace_ax.plot(
            np.arange(cumulative.size),
            cumulative,
            linewidth=1.4,
            alpha=0.9,
            label=label_str,
        )
        total_divergences += int(np.sum(div))
        total_iterations += int(div.size)

    trace_ax.set_xlabel("Iteration")
    trace_ax.set_ylabel("Cumulative Divergences")
    trace_ax.set_title("NUTS Divergences (Cumulative)")
    trace_ax.grid(True, alpha=0.3)
    trace_ax.legend(loc="best", fontsize="small")

    summary_ax.text(
        0.05,
        0.95,
        _get_divergences_summary(divergences_data),
        transform=summary_ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )
    summary_ax.set_title("Divergences Summary")
    summary_ax.axis("off")

    overall_pct = (
        total_divergences / total_iterations * 100
        if total_iterations > 0
        else 0
    )
    fig.suptitle(f"Overall Divergences: {overall_pct:.2f}%")

    plt.tight_layout()


def _get_divergences_summary(
    divergences_data: dict[str | int, np.ndarray],
):
    """Generate text summary of divergences."""
    lines = ["Divergences Summary:", ""]

    total_divergences = 0
    total_iterations = 0

    for label, div_values in divergences_data.items():
        n_divergent = np.sum(div_values)
        pct_divergent = n_divergent / len(div_values) * 100

        if label == "main":
            lines.append(
                f"NUTS: {n_divergent}/{len(div_values)} ({pct_divergent:.2f}%)"
            )
        else:
            lines.append(
                f"Channel {label}: {n_divergent}/{len(div_values)} ({pct_divergent:.2f}%)"
            )

        total_divergences += n_divergent
        total_iterations += len(div_values)

    lines.append("")
    overall_pct = (
        total_divergences / total_iterations * 100
        if total_iterations > 0
        else 0
    )
    lines.append(
        f"Total: {total_divergences}/{total_iterations} ({overall_pct:.2f}%)"
    )

    lines.append("")
    lines.append("Interpretation:")
    if overall_pct == 0:
        lines.append("  ✓ No divergences detected")
        lines.append("    Sampling appears well-behaved")
    elif overall_pct < 0.1:
        lines.append("  ~ Few divergences")
        lines.append("    Generally good, but monitor")
    elif overall_pct < 1.0:
        lines.append("  ⚠ Some divergences detected")
        lines.append("    May indicate sampling issues")
    else:
        lines.append("  ✗ Many divergences!")
        lines.append("    Significant sampling problems")
        lines.append("    Consider model reparameterization")

    return "\n".join(lines)


def _group_parameters_simple(idata):
    """Simple parameter grouping for counting."""
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return {}
    param_groups: dict[str, list[str]] = {
        "phi": [],
        "delta": [],
        "weights": [],
        "other": [],
    }

    for param in posterior.data_vars:
        param_name = str(param)
        if param_name.startswith("phi"):
            param_groups["phi"].append(param_name)
        elif param_name.startswith("delta"):
            param_groups["delta"].append(param_name)
        elif param_name.startswith("weights"):
            param_groups["weights"].append(param_name)
        else:
            param_groups["other"].append(param_name)

    return {k: v for k, v in param_groups.items() if v}


def generate_diagnostics_summary(
    idata,
    outdir,
    *,
    mode: Literal["off", "light", "full"] = "light",
):
    """Generate a text summary of MCMC diagnostics.

    Modes:
    - ``off``: skip generation entirely.
    - ``light``: use cached attrs/sample_stats only (fast; avoids PSIS/ArviZ scans).
    - ``full``: compute full diagnostics via ``run_all_diagnostics`` (may be slow).
    """
    if mode == "off":
        return ""

    summary = []
    summary.append("=== MCMC Diagnostics Summary ===\n")

    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)
    have_cached_full = bool(attrs.get("full_diagnostics_computed", 0))

    def _as_float(value) -> Optional[float]:
        try:
            fval = float(value)
        except Exception:
            return None
        return fval if np.isfinite(fval) else None

    n_samples = idata.posterior.sizes.get("draw", 0)
    n_chains = idata.posterior.sizes.get("chain", 1)
    n_params = len(list(idata.posterior.data_vars))
    sampler_type = attrs.get("sampler_type", "Unknown")
    max_tree_depth = _coerce_int(attrs.get("max_tree_depth"))

    summary.append(f"Sampler: {sampler_type}")
    summary.append(
        f"Samples: {n_samples} per chain × {n_chains} chains = {n_samples * n_chains} total"
    )
    summary.append(f"Parameters: {n_params}")

    param_groups = _group_parameters_simple(idata)
    if param_groups:
        param_summary = ", ".join(
            [f"{k}: {len(v)}" for k, v in param_groups.items()]
        )
        summary.append(f"Parameter groups: {param_summary}")

    diag_results = {}
    mcmc_diag = {}
    if mode == "full" and not have_cached_full:
        diag_results = run_all_diagnostics(
            idata=idata,
            truth=attrs.get("true_psd"),
            psd_ref=attrs.get("true_psd"),
        )
        mcmc_diag = diag_results.get("mcmc", {}) or {}

    ess_min = None
    ess_med = None
    ess_tail_min = None
    ess_tail_med = None
    try:
        ess_attr = attrs.get("ess")
        if ess_attr is not None:
            ess_vals = np.asarray(ess_attr).reshape(-1)
            ess_vals = ess_vals[np.isfinite(ess_vals)]
            if ess_vals.size:
                ess_min = float(np.min(ess_vals))
                ess_med = float(np.median(ess_vals))
    except Exception:
        pass
    try:
        ess_tail_attr = attrs.get("ess_tail")
        if ess_tail_attr is not None:
            ess_tail_vals = np.asarray(ess_tail_attr).reshape(-1)
            ess_tail_vals = ess_tail_vals[np.isfinite(ess_tail_vals)]
            if ess_tail_vals.size:
                ess_tail_min = float(np.min(ess_tail_vals))
                ess_tail_med = float(np.median(ess_tail_vals))
    except Exception:
        pass

    rhat_max = None
    rhat_mean = None
    try:
        rhat_attr = attrs.get("rhat")
        if rhat_attr is not None:
            rhat_vals = np.asarray(rhat_attr).reshape(-1)
            rhat_vals = rhat_vals[np.isfinite(rhat_vals)]
            if rhat_vals.size:
                rhat_max = float(np.max(rhat_vals))
                rhat_mean = float(np.mean(rhat_vals))
    except Exception:
        pass

    if mode == "full":
        ess_min = mcmc_diag.get("ess_bulk_min", ess_min)
        ess_med = mcmc_diag.get("ess_bulk_median", ess_med)
        ess_tail_min = mcmc_diag.get("ess_tail_min", ess_tail_min)
        ess_tail_med = mcmc_diag.get("ess_tail_median", ess_tail_med)
        rhat_max = mcmc_diag.get("rhat_max", rhat_max)
        rhat_mean = mcmc_diag.get("rhat_mean", rhat_mean)

        # Fallback to ArviZ scans when explicitly requested.
        if ess_min is None or ess_tail_min is None or rhat_max is None:
            try:
                ess = az.ess(idata, method="bulk")
                ess_vals = np.concatenate(
                    [np.asarray(ess[v]).reshape(-1) for v in ess.data_vars]
                )
                ess_vals = ess_vals[np.isfinite(ess_vals)]
                if ess_vals.size:
                    ess_min = float(np.min(ess_vals))
                    ess_med = float(np.median(ess_vals))
            except Exception:
                pass

            try:
                ess_tail = az.ess(idata, method="tail")
                ess_tail_vals = np.concatenate(
                    [
                        np.asarray(ess_tail[v]).reshape(-1)
                        for v in ess_tail.data_vars
                    ]
                )
                ess_tail_vals = ess_tail_vals[np.isfinite(ess_tail_vals)]
                if ess_tail_vals.size:
                    ess_tail_min = float(np.min(ess_tail_vals))
                    ess_tail_med = float(np.median(ess_tail_vals))
            except Exception:
                pass

            try:
                rhat = az.rhat(idata)
                rhat_vals = np.concatenate(
                    [np.asarray(rhat[v]).reshape(-1) for v in rhat.data_vars]
                )
                rhat_vals = rhat_vals[np.isfinite(rhat_vals)]
                if rhat_vals.size:
                    rhat_max = float(np.max(rhat_vals))
                    rhat_mean = float(np.mean(rhat_vals))
            except Exception:
                pass

    cached_ess_min = _as_float(attrs.get("mcmc_ess_bulk_min"))
    cached_ess_med = _as_float(attrs.get("mcmc_ess_bulk_median"))
    cached_ess_tail_min = _as_float(attrs.get("mcmc_ess_tail_min"))
    cached_ess_tail_med = _as_float(attrs.get("mcmc_ess_tail_median"))
    cached_rhat_max = _as_float(attrs.get("mcmc_rhat_max"))
    cached_rhat_mean = _as_float(attrs.get("mcmc_rhat_mean"))

    if cached_ess_min is not None:
        ess_min = cached_ess_min
    if cached_ess_med is not None:
        ess_med = cached_ess_med
    if cached_ess_tail_min is not None:
        ess_tail_min = cached_ess_tail_min
    if cached_ess_tail_med is not None:
        ess_tail_med = cached_ess_tail_med
    if cached_rhat_max is not None:
        rhat_max = cached_rhat_max
    if cached_rhat_mean is not None:
        rhat_mean = cached_rhat_mean

    if ess_min is not None:
        summary.append(
            f"\nESS bulk: min={ess_min:.0f}"
            + (f", median={ess_med:.0f}" if ess_med is not None else "")
        )
    if ess_tail_min is not None:
        summary.append(
            f"ESS tail: min={ess_tail_min:.0f}"
            + (
                f", median={ess_tail_med:.0f}"
                if ess_tail_med is not None
                else ""
            )
        )
    if rhat_max is not None:
        summary.append(
            f"Rhat: max={rhat_max:.3f}"
            + (f", mean={rhat_mean:.3f}" if rhat_mean is not None else "")
        )

    # Acceptance rate / divergences (cheap: read directly from sample_stats).
    try:
        acc = None
        if "accept_prob" in idata.sample_stats:
            acc = float(np.mean(np.asarray(idata.sample_stats["accept_prob"])))
        elif "acceptance_rate" in idata.sample_stats:
            acc = float(
                np.mean(np.asarray(idata.sample_stats["acceptance_rate"]))
            )
        else:
            ch_vals = []
            for key in idata.sample_stats:
                if str(key).startswith("accept_prob_channel_"):
                    ch_vals.append(
                        float(np.mean(np.asarray(idata.sample_stats[key])))
                    )
            if ch_vals:
                acc = float(np.mean(ch_vals))
        if acc is not None and np.isfinite(acc):
            summary.append(f"Acceptance rate: {acc:.3f}")
    except Exception:
        pass

    try:
        div = None
        if "diverging" in idata.sample_stats:
            div = np.asarray(idata.sample_stats["diverging"]).reshape(-1)
        else:
            div_list = []
            for key in idata.sample_stats:
                if str(key).startswith("diverging_channel_"):
                    div_list.append(
                        np.asarray(idata.sample_stats[key]).reshape(-1)
                    )
            if div_list:
                div = np.concatenate(div_list)
        if div is not None:
            div = div[np.isfinite(div)]
            if div.size:
                div_count = int(np.sum(div))
                div_total = int(div.size)
                div_pct = float(np.mean(div) * 100.0)
                summary.append(
                    f"Divergences: {div_count}/{div_total} ({div_pct:.2f}%)"
                )
    except Exception:
        pass

    # PSIS is expensive (az.loo); only include when full diagnostics were run.
    khat = None
    if mode == "full":
        khat = mcmc_diag.get("psis_khat_max")
        if khat is None:
            khat = attrs.get("mcmc_psis_khat_max")
        if khat is not None:
            try:
                summary.append(f"PSIS k-hat (max): {float(khat):.3f}")
            except Exception:
                pass

    ebfmi = _as_float(attrs.get("energy_ebfmi_overall"))
    if ebfmi is None:
        ebfmi_candidates: list[float] = [
            float(v)
            for key, val in attrs.items()
            if str(key).startswith("energy_ebfmi")
            and str(key).endswith("_overall")
            for v in [_as_float(val)]
            if v is not None
        ]
        if ebfmi_candidates:
            ebfmi = float(np.min(ebfmi_candidates))
    if ebfmi is None and hasattr(idata, "sample_stats"):
        energy_values = []
        if "energy" in idata.sample_stats:
            energy_values.append(np.asarray(idata.sample_stats["energy"]))
        else:
            for key in idata.sample_stats:
                if str(key).startswith("energy_channel_"):
                    energy_values.append(np.asarray(idata.sample_stats[key]))
        if energy_values:
            ebfmi = _compute_ebfmi_from_energy(np.concatenate(energy_values))
    if ebfmi is not None:
        summary.append(f"E-BFMI: {ebfmi:.3f}")

    tree_depth = (
        idata.sample_stats.tree_depth.values.flatten()
        if "tree_depth" in idata.sample_stats
        else None
    )
    num_steps = None
    steps_by_channel = {}
    if "num_steps" in idata.sample_stats:
        num_steps = idata.sample_stats.num_steps.values.flatten()
    else:
        channel_indices = _get_channel_indices(idata.sample_stats, "num_steps")
        for ch in sorted(channel_indices):
            steps_by_channel[ch] = idata.sample_stats[
                f"num_steps_channel_{ch}"
            ].values.flatten()
        if steps_by_channel:
            num_steps = np.concatenate(list(steps_by_channel.values()))
    max_stats = _max_tree_depth_stats(num_steps, tree_depth, max_tree_depth)
    if max_stats:
        pct = max_stats["frac"] * 100
        summary.append(_format_max_tree_depth_message(max_stats))
        if steps_by_channel:
            for ch in sorted(steps_by_channel):
                ch_stats = _max_tree_depth_stats(
                    steps_by_channel[ch], None, max_tree_depth
                )
                if ch_stats:
                    summary.append(
                        f"  Channel {ch}: {_format_max_tree_depth_message(ch_stats)}"
                    )
        if pct >= 50:
            summary.append(
                "  ⚠ Max tree depth hit rate is very high; consider reparameterization or increasing max_tree_depth."
            )
            logger.warning(
                f"Max tree depth hit rate is {pct:.1f}% (very high)."
            )
        elif pct >= 10:
            summary.append(
                "  ⚠ Max tree depth hit rate is elevated; geometry may be challenging."
            )
            logger.warning(
                f"Max tree depth hit rate is {pct:.1f}% (elevated)."
            )

    step_size = None
    step_size_by_channel = {}
    if "step_size" in idata.sample_stats:
        step_size = idata.sample_stats.step_size.values.flatten()
    else:
        channel_indices = _get_channel_indices(idata.sample_stats, "step_size")
        for ch in sorted(channel_indices):
            step_size_by_channel[ch] = idata.sample_stats[
                f"step_size_channel_{ch}"
            ].values.flatten()
        if step_size_by_channel:
            step_size = np.concatenate(list(step_size_by_channel.values()))

    if step_size is not None and step_size.size:
        step_size = step_size[np.isfinite(step_size)]
        if step_size.size:
            summary.append(
                f"Step size: median={np.median(step_size):.3g}, min={np.min(step_size):.3g}"
            )
        if step_size_by_channel:
            for ch in sorted(step_size_by_channel):
                ss = step_size_by_channel[ch]
                ss = ss[np.isfinite(ss)]
                if ss.size:
                    summary.append(
                        f"  Channel {ch}: median={np.median(ss):.3g}, min={np.min(ss):.3g}"
                    )

    diverging_by_channel = {}
    if "diverging" not in idata.sample_stats:
        channel_indices = _get_channel_indices(idata.sample_stats, "diverging")
        for ch in sorted(channel_indices):
            diverging_by_channel[ch] = idata.sample_stats[
                f"diverging_channel_{ch}"
            ].values.flatten()
    if diverging_by_channel:
        for ch in sorted(diverging_by_channel):
            div = np.asarray(diverging_by_channel[ch], dtype=float)
            div = div[np.isfinite(div)]
            if div.size:
                summary.append(
                    f"  Channel {ch}: divergences={np.mean(div)*100:.2f}%"
                )

    psd_diag: dict[str, object] = (
        dict(diag_results.get("psd_compare", {})) if mode == "full" else {}
    )

    # Prefer cached PSD diagnostics from attrs (computed during ArviZ conversion).
    cached_psd_keys = [
        "riae",
        "riae_matrix",
        "coverage",
        "coherence_riae",
        "riae_diag_mean",
        "riae_diag_max",
        "riae_offdiag",
    ]
    for key in cached_psd_keys:
        if key in attrs and key not in psd_diag:
            psd_diag[key] = attrs.get(key)

    psd_lines = []
    if psd_diag:
        riae_value = _as_float(psd_diag.get("riae"))
        if riae_value is not None and np.isfinite(riae_value):
            psd_lines.append(f"  RIAE: {riae_value:.3f}")

        riae_matrix_value = _as_float(psd_diag.get("riae_matrix"))
        if riae_matrix_value is not None and np.isfinite(riae_matrix_value):
            psd_lines.append(f"  RIAE (matrix): {riae_matrix_value:.3f}")

        coverage_value = _as_float(psd_diag.get("coverage"))
        if coverage_value is not None and np.isfinite(coverage_value):
            psd_lines.append(f"  Coverage: {coverage_value*100:.1f}%")

    psd_band_lines = []
    var_med = _as_float(attrs.get("psd_bands_variance_median"))
    var_width = _as_float(attrs.get("psd_bands_variance_ci_width"))
    var_med_mv = _as_float(attrs.get("psd_bands_variance_median_mean"))
    var_width_mv = _as_float(attrs.get("psd_bands_variance_ci_width_mean"))
    if var_med is not None:
        psd_band_lines.append(f"  Variance (PSD median): {var_med:.3g}")
    if var_width is not None:
        psd_band_lines.append(f"  Variance CI width: {var_width:.3g}")
    if var_med_mv is not None:
        psd_band_lines.append(
            f"  Variance mean (PSD median): {var_med_mv:.3g}"
        )
    if var_width_mv is not None:
        psd_band_lines.append(f"  Variance CI width mean: {var_width_mv:.3g}")

    if psd_lines:
        summary.append("\nPSD accuracy diagnostics:")
        summary.extend(psd_lines)

    if psd_band_lines:
        summary.append("\nPSD band summaries:")
        summary.extend(psd_band_lines)

    # Overall assessment (best-effort)
    summary.append("\nOverall Convergence Assessment:")
    if mcmc_diag or ess_min is not None or rhat_max is not None:
        if ess_min is None or rhat_max is None:
            summary.append("  Status: UNKNOWN (missing ESS/Rhat)")
        else:
            ess_ok = ess_min >= 400
            if ess_tail_min is not None:
                ess_ok = ess_ok and ess_tail_min >= 400
            rhat_ok = rhat_max <= 1.01
            if ess_ok and rhat_ok:
                summary.append("  Status: EXCELLENT ✓")
            elif ess_ok or rhat_ok:
                summary.append("  Status: GOOD ✓")
            else:
                summary.append("  Status: NEEDS ATTENTION ⚠")
    else:
        summary.append("  Status: UNKNOWN (insufficient diagnostics)")

    # Practical NUTS guidance (kept short; ASCII-only for portability).
    sampler_lower = str(sampler_type).lower()
    if "nuts" in sampler_lower:
        summary.append("\nNUTS Energy / E-BFMI Notes:")
        summary.append(
            "  H(q,p)=U(q)+K(p); U(q) is -log posterior (up to a constant)."
        )
        summary.append(
            "  Energy mixing: if transition energy changes are much narrower than"
        )
        summary.append(
            "  marginal energy, exploration is poor even without divergences."
        )
        summary.append("  E-BFMI thresholds (rule of thumb):")
        summary.append("    >=0.3 ok; 0.2-0.3 borderline; <0.2 problematic.")
        summary.append("  Prioritize checks in this order:")
        summary.append(
            "    divergences -> max tree depth hits -> E-BFMI/energy -> Rhat/ESS"
        )
        summary.append("    -> trace/rank -> pair plots for problem blocks.")
        summary.append("  If E-BFMI is low, fixes usually target geometry:")
        summary.append(
            "    reparameterize (non-centered, log-scales), whiten weights,"
        )
        summary.append(
            "    rescale inputs/priors to O(1), or use more informative priors."
        )
        summary.append(
            "  Higher target_accept or max_tree_depth are secondary tweaks."
        )

    summary_text = "\n".join(summary)

    if outdir:
        with open(f"{outdir}/diagnostics_summary.txt", "w") as f:
            f.write(summary_text)

    logger.info(f"\n{summary_text}\n")
    return summary_text


def generate_vi_diagnostics_summary(
    diagnostics: dict, outdir: Optional[str] = None, log: bool = True
) -> str:
    """Log and optionally write a concise VI diagnostics summary."""
    if not diagnostics:
        return ""

    lines = []
    lines.append("=== VI Diagnostics Summary ===")
    lines.append("")

    guide = diagnostics.get("guide", "vi")
    lines.append(f"Guide: {guide}")

    khat_max = diagnostics.get("psis_khat_max")
    if khat_max is not None and np.isfinite(khat_max):
        status = diagnostics.get("psis_status_message") or diagnostics.get(
            "psis_khat_status", ""
        )
        status_suffix = f" ({status})" if status else ""
        lines.append(f"PSIS k-hat (max): {float(khat_max):.3f}{status_suffix}")
        threshold = diagnostics.get("psis_khat_threshold", 0.7)
        if khat_max > threshold:
            lines.append(
                f"PSIS alert: k-hat exceeds {threshold:.1f} -> posterior may be unreliable"
            )
    moment_summary = diagnostics.get("psis_moment_summary") or {}
    weight_stats = moment_summary.get("weights")
    if weight_stats:
        frac = weight_stats.get("frac_outside")
        lines.append(
            "Weight var_ratio "
            + ", ".join(
                [
                    f"min={weight_stats.get('var_ratio_min', np.nan):.2f}",
                    f"median={weight_stats.get('var_ratio_median', np.nan):.2f}",
                    f"max={weight_stats.get('var_ratio_max', np.nan):.2f}",
                    (
                        f"outside[0.7,1.3]={frac*100:.1f}%"
                        if frac is not None
                        else "outside[0.7,1.3]=n/a"
                    ),
                ]
            )
        )
    hyper_params = moment_summary.get("hyperparameters") or []
    if hyper_params:
        lines.append("PSIS moments (hyperparameters):")
        for entry in hyper_params:
            status = diagnostics.get("psis_status_message") or ""
            var_ratio = entry["var_ratio"]
            bias_pct = entry["bias_pct"]
            thresholds = moment_summary.get("thresholds", {})
            bias_thr = thresholds.get("bias_threshold", 0.05) * 100.0
            var_low = thresholds.get("var_low", 0.7)
            var_high = thresholds.get("var_high", 1.3)
            status_label = "OK"
            if abs(bias_pct) > bias_thr:
                status_label = f"⚠ bias>{bias_thr:.0f}%"
            if var_ratio < var_low:
                status_label = "⚠ under-dispersed"
            elif var_ratio > var_high:
                status_label = "⚠ over-dispersed"
            lines.append(
                f"  {entry['param']}: "
                f"μ_vi={entry['vi_mean']:.3g}, μ_psis={entry['psis_mean']:.3g}, "
                f"bias={entry['bias_pct']:.1f}%, "
                f"σ_vi={entry['vi_std']:.3g}, σ_psis={entry['psis_std']:.3g}, "
                f"var_ratio={entry['var_ratio']:.2f} {status_label}"
            )
    corr_summary = diagnostics.get("psis_correlation_summary") or {}
    for label, stats in corr_summary.items():
        if not stats:
            continue
        line = (
            f"Corr ({label}): max|r|={stats.get('max_abs', np.nan):.3f}, "
            f"median|r|={stats.get('median_abs', np.nan):.3f}"
        )
        if "mean_corr_diff" in stats:
            line += f", mean|Δ| vs ref={stats['mean_corr_diff']:.3f}"
        lines.append(line)

    # Overall quality indicator
    quality = "OK"
    if diagnostics.get("psis_flag_critical"):
        quality = "❌ NOT TRUSTWORTHY"
    elif diagnostics.get("psis_flag_warn"):
        quality = "⚠ USE WITH CAUTION"
    else:
        # Escalate if hyperparameter moments look off
        for entry in hyper_params:
            thresholds = moment_summary.get("thresholds", {})
            bias_thr = thresholds.get("bias_threshold", 0.05) * 100.0
            var_low = thresholds.get("var_low", 0.7)
            var_high = thresholds.get("var_high", 1.3)
            if (
                abs(entry["bias_pct"]) > bias_thr
                or entry["var_ratio"] < var_low
                or entry["var_ratio"] > var_high
            ):
                quality = "⚠ USE WITH CAUTION"
                break
    lines.append(f"Overall VI Quality: {quality}")

    losses = diagnostics.get("losses")
    if losses is not None:
        loss_arr = np.asarray(losses)
        if loss_arr.size:
            final_elbo = float(loss_arr.reshape(-1)[-1])
            if np.isfinite(final_elbo):
                lines.append(f"Final ELBO: {final_elbo:.3f}")
            else:
                lines.append("Final ELBO: nan")

    vi_samples = diagnostics.get("vi_samples")
    if vi_samples:
        first = next(iter(vi_samples.values()))
        n_draws = np.asarray(first).shape[0]
        lines.append(f"Posterior draws (VI): {n_draws}")

    psd_shape = None
    if "psd_matrix" in diagnostics and diagnostics["psd_matrix"] is not None:
        psd_shape = np.asarray(diagnostics["psd_matrix"]).shape
    else:
        real_q = diagnostics.get("psd_quantiles", {}).get("real") or {}
        q50 = real_q.get("q50")
        if q50 is not None:
            psd_shape = np.asarray(q50).shape
    if psd_shape is not None and len(psd_shape) >= 3:
        lines.append(
            f"PSD shape: {psd_shape[0]} freq × {psd_shape[1]} × {psd_shape[2]}"
        )

    # Accuracy metrics
    riae_matrix = diagnostics.get("riae_matrix")
    riae_err = diagnostics.get("riae_matrix_errorbars")
    if riae_matrix is not None:
        line = f"RIAE (matrix): {float(riae_matrix):.3f}"
        if riae_err and len(riae_err) >= 5:
            line += f" (5-95% [{riae_err[0]:.3f}, {riae_err[4]:.3f}])"
        lines.append(line)

    per_ch = diagnostics.get("riae_per_channel")
    if per_ch:
        formatted = ", ".join(
            f"{idx}:{val:.3f}" for idx, val in enumerate(per_ch)
        )
        lines.append(f"RIAE per channel: {formatted}")

    offdiag = diagnostics.get("riae_offdiag")
    if offdiag is not None:
        lines.append(f"RIAE off-diagonal: {float(offdiag):.3f}")

    coh_riae = diagnostics.get("coherence_riae")
    if coh_riae is not None:
        lines.append(f"Coherence RIAE: {float(coh_riae):.3f}")

    bands = diagnostics.get("riae_bands")
    if bands:
        band_str = "; ".join(
            f"[{b['start']:.2e},{b['end']:.2e}]:{b['value']:.3f}"
            for b in bands
        )
        lines.append(f"RIAE by frequency bands: {band_str}")

    coverage = diagnostics.get("coverage") or diagnostics.get("ci_coverage")
    coverage_level = diagnostics.get("coverage_level")
    if coverage is not None:
        label = (
            f"{int(round(coverage_level * 100))}% interval coverage"
            if coverage_level is not None
            else "Interval coverage"
        )
        lines.append(f"{label}: {float(coverage) * 100:.1f}%")

    summary_text = "\n".join(lines)

    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            with open(
                os.path.join(outdir, "vi_diagnostics_summary.txt"), "w"
            ) as f:
                f.write(summary_text)
        except Exception:
            logger.debug(
                "Could not write VI diagnostics summary to disk.",
                exc_info=True,
            )

    if log:
        logger.info(f"\n{summary_text}\n")
    return summary_text
