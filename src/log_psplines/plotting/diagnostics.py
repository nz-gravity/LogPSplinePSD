import os
from dataclasses import dataclass
from typing import Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from ..arviz_utils.rhat import extract_rhat_values
from ..logger import logger
from .base import PlotConfig, safe_plot, setup_plot_style

# Setup consistent styling for diagnostics plots
setup_plot_style()


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics plotting parameters."""

    figsize: tuple = (12, 8)
    dpi: int = 150
    ess_threshold: int = 400
    rhat_threshold: float = 1.01
    fontsize: int = 11
    labelsize: int = 12
    titlesize: int = 12


def plot_trace(idata: az.InferenceData, compact=True) -> plt.Figure:
    groups = {
        "delta": [
            v for v in idata.posterior.data_vars if v.startswith("delta")
        ],
        "phi": [v for v in idata.posterior.data_vars if v.startswith("phi")],
        "weights": [
            v for v in idata.posterior.data_vars if v.startswith("weights")
        ],
    }

    if compact:
        nrows = 3
    else:
        nrows = len(groups)
    fig, axes = plt.subplots(nrows, 2, figsize=(7, 3 * nrows))

    for row, (group_name, vars) in enumerate(groups.items()):

        # if vars are more than 1, and compact, then we need to repeat the axes
        if compact:
            group_axes = axes[row, :].reshape(1, 2)
            group_axes = np.repeat(group_axes, len(vars), axis=0)
        else:
            group_axes = axes[row, :]

        group_axes[0, 0].set_title(
            f"{group_name.capitalize()} Parameters", fontsize=14
        )

        for i, var in enumerate(vars):
            data = idata.posterior[
                var
            ].values  # shape is (nchain, nsamples, ndim) if ndim>1 else (nchain, nsamples)
            if data.ndim == 3:
                data = data[0].T  # shape is now (ndim, nsamples)

            ax_trace = group_axes[i, 0] if compact else group_axes[0]
            ax_hist = group_axes[i, 1] if compact else group_axes[1]
            ax_trace.set_ylabel(group_name, fontsize=8)
            ax_trace.set_xlabel("MCMC Step", fontsize=8)
            ax_hist.set_xlabel(group_name, fontsize=8)
            # place ylabel on right side of hist
            ax_hist.yaxis.set_label_position("right")
            ax_hist.set_ylabel("Density", fontsize=8, rotation=270, labelpad=0)

            # remove axes yspine for hist
            ax_hist.spines["left"].set_visible(False)
            ax_hist.spines["right"].set_visible(False)
            ax_hist.spines["top"].set_visible(False)
            ax_hist.set_yticks([])  # remove y ticks
            ax_hist.yaxis.set_ticks_position("none")

            ax_trace.spines["right"].set_visible(False)
            ax_trace.spines["top"].set_visible(False)

            color = f"C{i}"
            label = f"{var}"
            if group_name in ["phi", "delta"]:
                ax_trace.set_yscale("log")
                ax_hist.set_xscale("log")

            for p in data:
                ax_trace.plot(p, color=color, alpha=0.7, label=label)

                # if phi or delta, use log scale for hist-x, log for trace y
                if group_name in ["phi", "delta"]:
                    bins = np.logspace(
                        np.log10(np.min(p)), np.log10(np.max(p)), 30
                    )
                    logp = np.log(p)
                    log_grid, log_pdf = az.kde(logp)
                    grid = np.exp(log_grid)
                    pdf = log_pdf / grid  # change of variables
                else:
                    bins = 30
                    grid, pdf = az.kde(p)
                ax_hist.plot(grid, pdf, color=color, label=label)
                ax_hist.hist(
                    p, bins=bins, density=True, color=color, alpha=0.3
                )

                # KDE plot instead of histogram

    plt.suptitle("Parameter Traces", fontsize=16)
    plt.tight_layout()
    return fig


def plot_diagnostics(
    idata: az.InferenceData,
    outdir: str,
    n_channels: Optional[int] = None,
    n_freq: Optional[int] = None,
    runtime: Optional[float] = None,
    config: Optional[DiagnosticsConfig] = None,
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

    # Generate summary report
    generate_diagnostics_summary(idata, diag_dir)
    _create_diagnostic_plots(
        idata, diag_dir, config, n_channels, n_freq, runtime
    )


def _create_diagnostic_plots(
    idata, diag_dir, config, n_channels, n_freq, runtime
):
    """Create only the essential diagnostic plots."""
    logger.debug("Generating diagnostic plots...")

    # 1. ArviZ trace plots
    @safe_plot(f"{diag_dir}/trace_plots.png", config.dpi)
    def create_trace_plots():
        return plot_trace(idata)

    create_trace_plots()

    # 2. Summary dashboard with key convergence metrics
    @safe_plot(f"{diag_dir}/summary_dashboard.png", config.dpi)
    def plot_summary():
        _plot_summary_dashboard(idata, config, n_channels, n_freq, runtime)

    plot_summary()

    # 3. Log posterior diagnostics
    @safe_plot(f"{diag_dir}/log_posterior.png", config.dpi)
    def plot_lp():
        _plot_log_posterior(idata, config)

    plot_lp()

    # 4. Acceptance rate diagnostics
    @safe_plot(f"{diag_dir}/acceptance_diagnostics.png", config.dpi)
    def plot_acceptance():
        _plot_acceptance_diagnostics_blockaware(idata, config)

    plot_acceptance()

    # 5. Sampler-specific diagnostics
    _create_sampler_diagnostics(idata, diag_dir, config)

    # 6. Divergences diagnostics (for NUTS only)
    _create_divergences_diagnostics(idata, diag_dir, config)


def _plot_summary_dashboard(idata, config, n_channels, n_freq, runtime):

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
    _plot_metadata(meta_ax, idata, n_channels, n_freq, runtime)

    # 3. Parameter Summary
    _plot_parameter_summary(param_ax, idata)

    # 4. Convergence Status
    _plot_convergence_status(status_ax, ess_values, config)

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


def _plot_metadata(ax, idata, n_channels, n_freq, runtime):
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
        if n_channels is not None:
            metadata_lines.append(f"Channels: {n_channels}")
        if n_freq is not None:
            metadata_lines.append(f"Frequencies: {n_freq}")
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


def _plot_convergence_status(ax, ess_values, config):
    """Display convergence status based on ESS only."""
    try:
        status_lines = ["Convergence Status:"]

        if ess_values is not None and len(ess_values) > 0:
            ess_good = (ess_values >= config.ess_threshold).mean() * 100
            status_lines.append(
                f"ESS ≥ {config.ess_threshold}: {ess_good:.0f}%"
            )
            status_lines.append("")
            status_lines.append("Overall Status:")

            if ess_good >= 90:
                status_lines.append("✓ EXCELLENT")
                color = "green"
            elif ess_good >= 75:
                status_lines.append("✓ ADEQUATE")
                color = "orange"
            else:
                status_lines.append("⚠ NEEDS ATTENTION")
                color = "red"
        else:
            status_lines.append("? UNABLE TO ASSESS")
            color = "gray"

        ax.text(
            0.05,
            0.95,
            "\n".join(status_lines),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            color=color,
        )
    except Exception:
        ax.text(0.5, 0.5, "Status unavailable", ha="center", va="center")

    ax.set_title("Convergence Status")
    ax.axis("off")


def _plot_log_posterior(idata, config):
    """Log posterior diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    # Check for lp first, then log_likelihood
    if "lp" in idata.sample_stats:
        lp_values = idata.sample_stats["lp"].values.flatten()
        var_name = "lp"
        title_prefix = "Log Posterior"
    elif "log_likelihood" in idata.sample_stats:
        lp_values = idata.sample_stats["log_likelihood"].values.flatten()
        var_name = "log_likelihood"
        title_prefix = "Log Likelihood"
    else:
        # Create a fallback layout when no posterior data available
        fig, axes = plt.subplots(1, 1, figsize=config.figsize)
        axes.text(
            0.5,
            0.5,
            "No log posterior\nor log likelihood\navailable",
            ha="center",
            va="center",
            fontsize=14,
        )
        axes.set_title("Log Posterior Diagnostics")
        axes.axis("off")
        plt.tight_layout()
        return

    # Trace plot with running mean overlaid
    axes[0, 0].plot(
        lp_values, alpha=0.7, linewidth=1, color="blue", label="Trace"
    )

    # Add running mean on the same plot
    window_size = max(10, len(lp_values) // 100)
    if len(lp_values) > window_size:
        running_mean = np.convolve(
            lp_values, np.ones(window_size) / window_size, mode="valid"
        )
        axes[0, 0].plot(
            range(window_size // 2, window_size // 2 + len(running_mean)),
            running_mean,
            alpha=0.9,
            linewidth=3,
            color="red",
            label=f"Running mean (w={window_size})",
        )

    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel(title_prefix)
    axes[0, 0].set_title(f"{title_prefix} Trace with Running Mean")
    axes[0, 0].legend(loc="best", fontsize="small")
    axes[0, 0].grid(True, alpha=0.3)

    # Distribution
    axes[0, 1].hist(
        lp_values, bins=50, alpha=0.7, density=True, edgecolor="black"
    )
    axes[0, 1].axvline(
        np.mean(lp_values),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(lp_values):.1f}",
    )
    axes[0, 1].set_xlabel(title_prefix)
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title(f"{title_prefix} Distribution")
    axes[0, 1].legend(loc="best", fontsize="small")
    axes[0, 1].grid(True, alpha=0.3)

    # Step-to-step changes
    lp_diff = np.diff(lp_values)
    axes[1, 0].plot(lp_diff, alpha=0.5, linewidth=1)
    axes[1, 0].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].axhline(
        np.mean(lp_diff),
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Mean change: {np.mean(lp_diff):.1f}",
    )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel(f"{title_prefix} Difference")
    axes[1, 0].set_title("Step-to-Step Changes")
    axes[1, 0].legend(loc="best", fontsize="small")
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    stats_lines = [
        f"Mean: {np.mean(lp_values):.2f}",
        f"Std: {np.std(lp_values):.2f}",
        f"Min: {np.min(lp_values):.2f}",
        f"Max: {np.max(lp_values):.2f}",
        f"Range: {np.max(lp_values) - np.min(lp_values):.2f}",
        "",
        "Stability:",
        f"Final variation: {np.std(lp_values[-len(lp_values)//4:]):.2f}",
    ]

    axes[1, 1].text(
        0.05,
        0.95,
        "\n".join(stats_lines),
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    axes[1, 1].set_title("Posterior Statistics")
    axes[1, 1].axis("off")

    plt.tight_layout()


def _plot_acceptance_diagnostics(idata, config):
    """Acceptance rate diagnostics."""
    accept_key = None
    if "accept_prob" in idata.sample_stats:
        accept_key = "accept_prob"
    elif "acceptance_rate" in idata.sample_stats:
        accept_key = "acceptance_rate"

    if accept_key is None:
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

    accept_rates = idata.sample_stats[accept_key].values.flatten()
    target_rate = getattr(idata.attrs, "target_accept_rate", 0.44)
    sampler_type = (
        idata.attrs["sampler_type"].lower()
        if "sampler_type" in idata.attrs
        else "unknown"
    )
    sampler_type = "NUTS" if "nuts" in sampler_type else "MH"

    # Define good ranges based on sampler
    if target_rate > 0.5:  # NUTS
        good_range = (0.7, 0.9)
        low_range = (0.0, 0.6)
        high_range = (0.9, 1.0)
        concerning_range = (0.6, 0.7)
    else:  # MH
        good_range = (0.2, 0.5)
        low_range = (0.0, 0.2)
        high_range = (0.5, 1.0)
        concerning_range = (0.1, 0.2)  # MH can be lower than NUTS

    # Trace plot with color zones
    # Add background zones
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

    # Main trace plot
    axes[0, 0].plot(
        accept_rates, alpha=0.8, linewidth=1, color="blue", label="Trace"
    )
    axes[0, 0].axhline(
        target_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_rate})",
    )

    # Add running average on the same plot
    window_size = max(10, len(accept_rates) // 50)
    if len(accept_rates) > window_size:
        running_mean = np.convolve(
            accept_rates, np.ones(window_size) / window_size, mode="valid"
        )
        axes[0, 0].plot(
            range(window_size // 2, window_size // 2 + len(running_mean)),
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

    # Add interpretation text
    interpretation = f"{sampler_type} aims for {target_rate:.2f}."
    if target_rate > 0.5:
        interpretation += " Green: efficient sampling."
    else:
        interpretation += " MH adapts to find optimal rate."
    axes[0, 0].text(
        0.02,
        0.02,
        interpretation,
        transform=axes[0, 0].transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    # Distribution
    axes[0, 1].hist(
        accept_rates, bins=30, alpha=0.7, density=True, edgecolor="black"
    )
    axes[0, 1].axvline(
        target_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_rate})",
    )
    axes[0, 1].set_xlabel("Acceptance Rate")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Acceptance Rate Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Since running means are already overlaid on the main plot, use the bottom row for additional info

    # Additional acceptance analysis - evolution over time
    if len(accept_rates) > 10:
        # Show moving standard deviation or coefficient of variation
        window_std = np.array(
            [
                np.std(accept_rates[max(0, i - 20) : i + 1])
                for i in range(len(accept_rates))
            ]
        )
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

    # Summary statistics (expanded)
    stats_text = [
        f"Sampler: {sampler_type}",
        f"Target: {target_rate:.3f}",
        f"Mean: {np.mean(accept_rates):.3f}",
        f"Std: {np.std(accept_rates):.3f}",
        f"CV: {np.std(accept_rates)/np.mean(accept_rates):.3f}",
        f"Min: {np.min(accept_rates):.3f}",
        f"Max: {np.max(accept_rates):.3f}",
        "",
        "Stability:",
        f"Final std: {np.std(accept_rates[-len(accept_rates)//4:]):.3f}",
    ]

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


def _plot_acceptance_diagnostics_blockaware(idata, config):
    """Acceptance diagnostics that also handle per‑channel series from blocked NUTS.

    If keys like ``accept_prob_channel_0`` are found in ``idata.sample_stats``,
    they are overlaid on the overall trace and included in the summary.
    """
    # Detect overall series
    accept_key = None
    if "accept_prob" in idata.sample_stats:
        accept_key = "accept_prob"
    elif "acceptance_rate" in idata.sample_stats:
        accept_key = "acceptance_rate"

    # Collect per-channel series
    channel_series = {}
    for key in idata.sample_stats:
        if isinstance(key, str) and key.startswith("accept_prob_channel_"):
            try:
                ch = int(key.rsplit("_", 1)[-1])
                channel_series[ch] = idata.sample_stats[key].values.flatten()
            except Exception:
                pass

    if accept_key is None and not channel_series:
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

    # Overall or concatenated series
    if accept_key is not None:
        accept_rates = idata.sample_stats[accept_key].values.flatten()
    else:
        accept_rates = np.concatenate(list(channel_series.values()))

    sampler_type_attr = idata.attrs.get("sampler_type", "").lower()
    is_nuts = "nuts" in sampler_type_attr
    target_rate = idata.attrs.get(
        "target_accept_rate",
        idata.attrs.get("target_accept_prob", 0.8 if is_nuts else 0.44),
    )
    sampler_type = "NUTS" if is_nuts else "MH"

    # Ranges
    if is_nuts:
        good_range = (0.7, 0.9)
        low_range = (0.0, 0.6)
        high_range = (0.9, 1.0)
        concerning_range = (0.6, 0.7)
    else:
        good_range = (0.2, 0.5)
        low_range = (0.0, 0.2)
        high_range = (0.5, 1.0)
        concerning_range = (0.1, 0.2)

    # Background zones
    axes[0, 0].axhspan(good_range[0], good_range[1], alpha=0.1, color="green")
    axes[0, 0].axhspan(low_range[0], low_range[1], alpha=0.1, color="red")
    axes[0, 0].axhspan(high_range[0], high_range[1], alpha=0.1, color="orange")
    if concerning_range[1] > concerning_range[0]:
        axes[0, 0].axhspan(
            concerning_range[0], concerning_range[1], alpha=0.1, color="yellow"
        )

    # Plot traces: overall + per-channel overlays
    if accept_key is not None:
        axes[0, 0].plot(
            accept_rates, alpha=0.8, linewidth=1, color="blue", label="overall"
        )
    for ch in sorted(channel_series):
        axes[0, 0].plot(
            channel_series[ch], alpha=0.6, linewidth=1, label=f"ch {ch}"
        )
    axes[0, 0].axhline(
        target_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_rate})",
    )

    # Running mean for overall
    window_size = max(10, len(accept_rates) // 50)
    if len(accept_rates) > window_size:
        running_mean = np.convolve(
            accept_rates, np.ones(window_size) / window_size, mode="valid"
        )
        axes[0, 0].plot(
            range(window_size // 2, window_size // 2 + len(running_mean)),
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

    # Histogram and summary
    axes[0, 1].hist(
        accept_rates, bins=30, alpha=0.7, density=True, edgecolor="black"
    )
    axes[0, 1].axvline(
        target_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_rate})",
    )
    axes[0, 1].set_xlabel("Acceptance Rate")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Acceptance Rate Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if len(accept_rates) > 10:
        window_std = np.array(
            [
                np.std(accept_rates[max(0, i - 20) : i + 1])
                for i in range(len(accept_rates))
            ]
        )
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

    # Summary text
    stats_text = [
        f"Sampler: {sampler_type}",
        f"Target: {target_rate:.3f}",
        f"Mean: {np.mean(accept_rates):.3f}",
        f"Std: {np.std(accept_rates):.3f}",
        f"CV: {np.std(accept_rates)/np.mean(accept_rates):.3f}",
        f"Min: {np.min(accept_rates):.3f}",
        f"Max: {np.max(accept_rates):.3f}",
    ]
    if channel_series:
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
        va="top",
        family="monospace",
    )
    axes[1, 1].set_title("Acceptance Analysis")
    axes[1, 1].axis("off")

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
        for ch in sorted(steps_ch):
            s = steps_ch[ch]
            lines.append(f"  ch {ch} μ={np.mean(s):.1f}, max={np.max(s):.0f}")
        lines.append("")
    if has_accept or accept_ch:
        lines.append("Acceptance summary:")
        if has_accept:
            a = idata.sample_stats.accept_prob.values.flatten()
            lines.append(f"  overall μ={np.mean(a):.3f}")
        for ch in sorted(accept_ch):
            a = accept_ch[ch]
            lines.append(f"  ch {ch} μ={np.mean(a):.3f}")
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
    accept_prob = _get("accept_prob")

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
    if accept_prob is not None:
        stats_lines.append(f"  accept μ={np.mean(accept_prob):.3f}")

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

    # Check for NUTS-specific fields that MH definitely doesn't have
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

    # Check for MH-specific fields (exclude anything NUTS might have)
    has_mh = "step_size_mean" in idata.sample_stats and not has_nuts

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

        for channel_idx in sorted(channel_indices):

            @safe_plot(
                f"{diag_dir}/nuts_block_{channel_idx}_diagnostics.png",
                config.dpi,
            )
            def plot_nuts_block(channel_idx=channel_idx):
                _plot_single_nuts_block(idata, config, channel_idx)

            plot_nuts_block()
    elif has_mh:

        @safe_plot(f"{diag_dir}/mh_step_sizes.png", config.dpi)
        def plot_mh():
            _plot_mh_step_sizes(idata, config)

        plot_mh()


def _plot_nuts_diagnostics(idata, config):
    """NUTS diagnostics with enhanced information."""
    # Determine available data to decide layout
    has_energy = "energy" in idata.sample_stats
    has_potential = "potential_energy" in idata.sample_stats
    has_steps = "num_steps" in idata.sample_stats
    has_accept = "accept_prob" in idata.sample_stats
    has_divergences = "diverging" in idata.sample_stats
    has_tree_depth = "tree_depth" in idata.sample_stats
    has_energy_error = "energy_error" in idata.sample_stats

    # Create a 2x2 layout, potentially combining energy and potential on same plot
    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    # Top-left: Energy diagnostics (combine Hamiltonian and Potential if both available)
    energy_ax = axes[0, 0]

    if has_energy and has_potential:
        # Both available - plot them together on one plot
        energy = idata.sample_stats.energy.values.flatten()
        potential = idata.sample_stats.potential_energy.values.flatten()

        # Plot both energies on same axis
        energy_ax.plot(
            energy, alpha=0.7, linewidth=1, color="blue", label="Hamiltonian"
        )
        energy_ax.plot(
            potential,
            alpha=0.7,
            linewidth=1,
            color="orange",
            label="Potential",
        )

        # Add difference (which relates to kinetic energy)
        energy_diff = energy - potential
        # Create second y-axis for difference
        ax2 = energy_ax.twinx()
        ax2.plot(
            energy_diff,
            alpha=0.5,
            linewidth=1,
            color="red",
            label="H - Potential (Kinetic)",
            linestyle="--",
        )
        ax2.set_ylabel("Energy Difference", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        energy_ax.set_xlabel("Iteration")
        energy_ax.set_ylabel("Energy", color="blue")
        energy_ax.tick_params(axis="y", labelcolor="blue")
        energy_ax.set_title("Hamiltonian & Potential Energy")
        energy_ax.legend(loc="best", fontsize="small")
        energy_ax.grid(True, alpha=0.3)

        # Add statistics
        energy_ax.text(
            0.02,
            0.98,
            f"H: μ={np.mean(energy):.1f}, σ={np.std(energy):.1f}\nP: μ={np.mean(potential):.1f}, σ={np.std(potential):.1f}",
            transform=energy_ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            verticalalignment="top",
        )

    elif has_energy:
        # Only Hamiltonian energy
        energy = idata.sample_stats.energy.values.flatten()
        energy_ax.plot(energy, alpha=0.7, linewidth=1, color="blue")
        energy_ax.set_xlabel("Iteration")
        energy_ax.set_ylabel("Hamiltonian Energy")
        energy_ax.set_title("Hamiltonian Energy Trace")
        energy_ax.grid(True, alpha=0.3)

    elif has_potential:
        # Only potential energy
        potential = idata.sample_stats.potential_energy.values.flatten()
        energy_ax.plot(potential, alpha=0.7, linewidth=1, color="orange")
        energy_ax.set_xlabel("Iteration")
        energy_ax.set_ylabel("Potential Energy")
        energy_ax.set_title("Potential Energy Trace")
        energy_ax.grid(True, alpha=0.3)

    else:
        energy_ax.text(
            0.5,
            0.5,
            "Energy data\nunavailable",
            ha="center",
            va="center",
            transform=energy_ax.transAxes,
        )
        energy_ax.set_title("Energy Diagnostics")

    # Top-right: Sampling efficiency diagnostics
    if has_steps:
        steps_ax = axes[0, 1]
        num_steps = idata.sample_stats.num_steps.values.flatten()

        # Show histogram with color zones for step efficiency
        n, bins, edges = steps_ax.hist(
            num_steps, bins=20, alpha=0.7, edgecolor="black"
        )

        # Add shaded regions for different efficiency levels
        # Green: efficient (tree depth ≤5, ~32 steps)
        # Yellow: moderate (tree depth 6-8, ~64-256 steps)
        # Red: inefficient (tree depth >8, >256 steps)
        steps_ax.axvspan(
            0, 64, alpha=0.1, color="green", label="Efficient (≤64)"
        )
        steps_ax.axvspan(
            64, 256, alpha=0.1, color="yellow", label="Moderate (65-256)"
        )
        steps_ax.axvspan(
            256,
            np.max(num_steps),
            alpha=0.1,
            color="red",
            label="Inefficient (>256)",
        )

        # Add reference lines for different tree depths
        for depth in [5, 7, 10]:  # Common tree depths
            max_steps = 2**depth
            steps_ax.axvline(
                x=max_steps,
                color="gray",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
                label=f"2^{depth} ({max_steps})",
            )

        steps_ax.set_xlabel("Leapfrog Steps")
        steps_ax.set_ylabel("Trajectories")
        steps_ax.set_title("Leapfrog Steps Distribution")
        steps_ax.legend(loc="best", fontsize="small")
        steps_ax.grid(True, alpha=0.3)

        # Add efficiency statistics
        pct_inefficient = (num_steps > 256).mean() * 100
        pct_moderate = ((num_steps > 64) & (num_steps <= 256)).mean() * 100
        pct_efficient = (num_steps <= 64).mean() * 100
        steps_ax.text(
            0.02,
            0.98,
            f"Efficient: {pct_efficient:.1f}%\nModerate: {pct_moderate:.1f}%\nInefficient: {pct_inefficient:.1f}%\nMean steps: {np.mean(num_steps):.1f}",
            transform=steps_ax.transAxes,
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            verticalalignment="top",
        )

    else:
        axes[0, 1].text(
            0.5, 0.5, "Steps data\nunavailable", ha="center", va="center"
        )
        axes[0, 1].set_title("Sampling Steps")

    # Bottom-left: Acceptance and NS divergence diagnostics
    accept_ax = axes[1, 0]

    if has_accept:
        accept_prob = idata.sample_stats.accept_prob.values.flatten()

        # Plot acceptance probability with guidance zones
        accept_ax.fill_between(
            range(len(accept_prob)),
            0.7,
            0.9,
            alpha=0.1,
            color="green",
            label="Good (0.7-0.9)",
        )
        accept_ax.fill_between(
            range(len(accept_prob)),
            0,
            0.6,
            alpha=0.1,
            color="red",
            label="Too low",
        )
        accept_ax.fill_between(
            range(len(accept_prob)),
            0.9,
            1.0,
            alpha=0.1,
            color="orange",
            label="Too high",
        )

        accept_ax.plot(
            accept_prob,
            alpha=0.8,
            linewidth=1,
            color="blue",
            label="Acceptance prob",
        )
        accept_ax.axhline(
            0.8,
            color="red",
            linestyle="--",
            linewidth=2,
            label="NUTS target (0.8)",
        )
        accept_ax.set_xlabel("Iteration")
        accept_ax.set_ylabel("Acceptance Probability")
        accept_ax.set_title("NUTS Acceptance Diagnostic")
        accept_ax.legend(loc="best", fontsize="small")
        accept_ax.set_ylim(0, 1)
        accept_ax.grid(True, alpha=0.3)

    else:
        accept_ax.text(
            0.5, 0.5, "Acceptance data\nunavailable", ha="center", va="center"
        )
        accept_ax.set_title("Acceptance Diagnostic")

    # Bottom-right: Summary statistics and additional diagnostics
    summary_ax = axes[1, 1]

    # Collect available statistics
    stats_lines = []

    if has_energy:
        energy = idata.sample_stats.energy.values.flatten()
        stats_lines.append(
            f"Energy: μ={np.mean(energy):.1f}, σ={np.std(energy):.1f}"
        )

    if has_potential:
        potential = idata.sample_stats.potential_energy.values.flatten()
        stats_lines.append(
            f"Potential: μ={np.mean(potential):.1f}, σ={np.std(potential):.1f}"
        )

    if has_steps:
        num_steps = idata.sample_stats.num_steps.values.flatten()
        stats_lines.append(
            f"Steps: μ={np.mean(num_steps):.1f}, max={np.max(num_steps):.0f}"
        )
        stats_lines.append("")

    if has_tree_depth:
        tree_depth = idata.sample_stats.tree_depth.values.flatten()
        stats_lines.append(f"Tree depth: μ={np.mean(tree_depth):.1f}")
        pct_max_depth = (tree_depth >= 10).mean() * 100
        stats_lines.append(f"Max depth (≥10): {pct_max_depth:.1f}%")

    if has_divergences:
        divergences = idata.sample_stats.diverging.values.flatten()
        n_divergences = np.sum(divergences)
        pct_divergent = n_divergences / len(divergences) * 100
        stats_lines.append(
            f"Divergent: {n_divergences}/{len(divergences)} ({pct_divergent:.2f}%)"
        )

    if has_energy_error:
        energy_error = idata.sample_stats.energy_error.values.flatten()
        stats_lines.append(
            f"Energy error: |μ|={np.mean(np.abs(energy_error)):.3f}"
        )

    if not stats_lines:
        summary_ax.text(
            0.5,
            0.5,
            "No diagnostics\ndata available",
            ha="center",
            va="center",
            transform=summary_ax.transAxes,
        )
        summary_ax.set_title("NUTS Statistics")
        summary_ax.axis("off")
    else:
        summary_text = "\n".join(["NUTS Diagnostics:"] + [""] + stats_lines)
        summary_ax.text(
            0.05,
            0.95,
            summary_text,
            transform=summary_ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
        summary_ax.set_title("NUTS Summary Statistics")
        summary_ax.axis("off")

    plt.tight_layout()


def _plot_mh_step_sizes(idata, config):
    """MH step size diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    step_means = idata.sample_stats.step_size_mean.values.flatten()
    step_stds = idata.sample_stats.step_size_std.values.flatten()

    # Step size evolution
    axes[0, 0].plot(
        step_means, alpha=0.7, linewidth=1, label="Mean", color="blue"
    )
    axes[0, 0].plot(
        step_stds, alpha=0.7, linewidth=1, label="Std", color="orange"
    )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Step Size")
    axes[0, 0].set_title("Step Size Evolution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Step size distributions
    axes[0, 1].hist(step_means, bins=30, alpha=0.5, label="Mean", color="blue")
    axes[0, 1].hist(step_stds, bins=30, alpha=0.5, label="Std", color="orange")
    axes[0, 1].set_xlabel("Step Size")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Step Size Distributions")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Step size adaptation quality
    axes[1, 0].plot(step_means / step_stds, alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Mean / Std")
    axes[1, 0].set_title("Step Size Consistency")
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    summary_lines = [
        "Step Size Summary:",
        f"Final mean: {step_means[-1]:.4f}",
        f"Final std: {step_stds[-1]:.4f}",
        f"Mean of means: {np.mean(step_means):.4f}",
        f"Mean of stds: {np.mean(step_stds):.4f}",
        "",
        "Adaptation Quality:",
        f"CV of means: {np.std(step_means)/np.mean(step_means):.3f}",
        f"CV of stds: {np.std(step_stds)/np.mean(step_stds):.3f}",
    ]

    axes[1, 1].text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    axes[1, 1].set_title("Step Size Statistics")
    axes[1, 1].axis("off")

    plt.tight_layout()


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
    # Collect all divergence data
    divergences_data = {}

    # Check for main divergences (single chain NUTS)
    if "diverging" in idata.sample_stats:
        divergences_data["main"] = (
            idata.sample_stats.diverging.values.flatten()
        )

    # Check for channel-specific divergences (blocked NUTS)
    channel_divergences = {}
    for key in idata.sample_stats:
        if key.startswith("diverging_channel_"):
            channel_idx = key.replace("diverging_channel_", "")
            channel_divergences[int(channel_idx)] = idata.sample_stats[
                key
            ].values.flatten()

    if channel_divergences:
        divergences_data.update(channel_divergences)

    if not divergences_data:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(
            0.5, 0.5, "No divergence data available", ha="center", va="center"
        )
        ax.set_title("Divergences Diagnostics")
        return

    # Create subplot layout
    n_plots = len(divergences_data)
    if n_plots == 1:
        fig, axes = plt.subplots(1, 2, figsize=config.figsize)
        trace_ax, summary_ax = axes
    else:
        # Multiple plots - arrange in grid
        cols = 2
        rows = (n_plots + 1) // cols  # Ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=config.figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        # Last plot goes in summary_ax if odd number
        if n_plots % 2 == 1:
            trace_axes = axes[:-1]
            summary_ax = axes[-1]
        else:
            trace_axes = axes
            summary_ax = None

    # Plot divergences traces
    total_divergences = 0
    total_iterations = 0

    plot_idx = 0
    for label, div_values in divergences_data.items():
        if label == "main":
            title = "NUTS Divergences"
            ax = trace_axes[plot_idx] if n_plots > 1 else axes[0]
        else:
            title = f"Channel {label} Divergences"
            ax = trace_axes[plot_idx] if n_plots > 1 else axes[0]
            plot_idx += 1

        # Plot divergence indicators (where divergences occur)
        div_indices = np.where(div_values)[0]
        ax.scatter(
            div_indices,
            np.ones_like(div_indices),
            color="red",
            marker="x",
            s=50,
            linewidth=2,
            label="Divergent",
            alpha=0.8,
        )

        # Add background shading for divergent regions
        if len(div_indices) > 0:
            for idx in div_indices:
                ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.2, color="red")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Divergence Indicator")
        ax.set_title(title)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No", "Yes"])
        ax.grid(True, alpha=0.3)

        # Add statistics
        n_divergent = np.sum(div_values)
        pct_divergent = n_divergent / len(div_values) * 100
        stats_text = f"{n_divergent}/{len(div_values)} ({pct_divergent:.2f}%)"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
            verticalalignment="top",
        )

        total_divergences += n_divergent
        total_iterations += len(div_values)

        # Legend only if there are divergences
        if n_divergent > 0:
            ax.legend(loc="upper right", fontsize="small")

    # Summary plot
    if summary_ax is not None and n_plots > 1:
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
    elif n_plots == 1:
        axes[1].text(
            0.05,
            0.95,
            _get_divergences_summary(divergences_data),
            transform=axes[1].transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
        axes[1].set_title("Divergences Summary")
        axes[1].axis("off")

    # Overall title
    overall_pct = (
        total_divergences / total_iterations * 100
        if total_iterations > 0
        else 0
    )
    fig.suptitle(f"Overall Divergences: {overall_pct:.2f}%")

    plt.tight_layout()


def _get_divergences_summary(divergences_data):
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
    param_groups = {"phi": [], "delta": [], "weights": [], "other": []}

    for param in idata.posterior.data_vars:
        if param.startswith("phi"):
            param_groups["phi"].append(param)
        elif param.startswith("delta"):
            param_groups["delta"].append(param)
        elif param.startswith("weights"):
            param_groups["weights"].append(param)
        else:
            param_groups["other"].append(param)

    return {k: v for k, v in param_groups.items() if v}


def generate_diagnostics_summary(idata, outdir):
    """Generate comprehensive text summary."""
    summary = []
    summary.append("=== MCMC Diagnostics Summary ===\n")

    ess_values = None
    rhat_good = None

    # Basic info
    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)

    n_samples = idata.posterior.sizes.get("draw", 0)
    n_chains = idata.posterior.sizes.get("chain", 1)
    n_params = len(list(idata.posterior.data_vars))
    sampler_type = attrs.get("sampler_type", "Unknown")

    summary.append(f"Sampler: {sampler_type}")
    summary.append(
        f"Samples: {n_samples} per chain × {n_chains} chains = {n_samples * n_chains} total"
    )
    summary.append(f"Parameters: {n_params}")

    # Parameter breakdown
    param_groups = _group_parameters_simple(idata)
    if param_groups:
        param_summary = ", ".join(
            [f"{k}: {len(v)}" for k, v in param_groups.items()]
        )
        summary.append(f"Parameter groups: {param_summary}")

    # ESS
    try:
        ess = idata.attrs.get("ess")
        ess_values = ess[~np.isnan(ess)]

        if len(ess_values) > 0:
            summary.append(
                f"\nESS: min={ess_values.min():.0f}, mean={ess_values.mean():.0f}, max={ess_values.max():.0f}"
            )
            summary.append(f"ESS ≥ 400: {(ess_values >= 400).mean()*100:.1f}%")
    except Exception as e:
        summary.append(f"\nESS: unavailable")

    # Rhat
    try:
        if idata.posterior.sizes.get("chain", 1) > 1:
            rhat_vals = extract_rhat_values(idata)
        else:
            rhat_vals = np.array([])
        if rhat_vals.size:
            summary.append(
                f"Rhat: min={rhat_vals.min():.3f}, mean={rhat_vals.mean():.3f}, max={rhat_vals.max():.3f}"
            )
            rhat_good = (rhat_vals <= 1.01).mean() * 100
            summary.append(f"Rhat ≤ 1.01: {rhat_good:.1f}%")
        else:
            summary.append("Rhat: unavailable (needs ≥2 chains)")
    except Exception:
        summary.append("Rhat: unavailable")

    # Acceptance
    accept_key = None
    if "accept_prob" in idata.sample_stats:
        accept_key = "accept_prob"
    elif "acceptance_rate" in idata.sample_stats:
        accept_key = "acceptance_rate"

    if accept_key is not None:
        accept_rate = idata.sample_stats[accept_key].values.mean()
        target_rate = attrs.get(
            "target_accept_rate", attrs.get("target_accept_prob", 0.44)
        )
        summary.append(
            f"Acceptance rate: {accept_rate:.3f} (target: {target_rate:.3f})"
        )
    else:
        # Blocked NUTS: compute a combined mean from per‑channel keys if present
        channel_means = []
        for key in idata.sample_stats:
            if isinstance(key, str) and key.startswith("accept_prob_channel_"):
                try:
                    channel_means.append(
                        float(idata.sample_stats[key].values.mean())
                    )
                except Exception:
                    pass
        if channel_means:
            target_rate = attrs.get(
                "target_accept_rate", attrs.get("target_accept_prob", 0.8)
            )
            summary.append(
                f"Acceptance rate (per-channel mean): {np.mean(channel_means):.3f} (target: {target_rate:.3f})"
            )

    # PSD accuracy diagnostics (requires true_psd in attrs)
    has_true_psd = "true_psd" in attrs

    if has_true_psd:
        coverage_level = attrs.get("coverage_level")
        coverage_label = (
            f"{int(round(coverage_level * 100))}% interval coverage"
            if coverage_level is not None
            else "Interval coverage"
        )

        def _format_riae_line(value, errorbars, prefix="  "):
            line = f"{prefix}RIAE: {value:.3f}"
            if errorbars:
                q05, q25, median, q75, q95 = errorbars
                line += f" (median {median:.3f}, 5-95% [{q05:.3f}, {q95:.3f}])"
            summary.append(line)

        def _format_coverage_line(value, prefix="  "):
            if value is None:
                return
            summary.append(f"{prefix}{coverage_label}: {value * 100:.1f}%")

        summary.append("\nPSD accuracy diagnostics:")

        if "riae" in attrs:
            _format_riae_line(attrs["riae"], attrs.get("riae_errorbars"))
        if "coverage" in attrs:
            _format_coverage_line(attrs["coverage"])

        channel_indices = sorted(
            int(key.replace("riae_ch", ""))
            for key in attrs.keys()
            if key.startswith("riae_ch")
        )

        for idx in channel_indices:
            metrics = []
            riae_key = f"riae_ch{idx}"
            cov_key = f"coverage_ch{idx}"
            error_key = f"riae_errorbars_ch{idx}"

            if riae_key in attrs:
                riae_line = f"RIAE {attrs[riae_key]:.3f}"
                errorbars = attrs.get(error_key)
                if errorbars:
                    q05, _, median, _, q95 = errorbars
                    riae_line += (
                        f" (median {median:.3f}, 5-95% [{q05:.3f}, {q95:.3f}])"
                    )
                metrics.append(riae_line)

            if cov_key in attrs:
                metrics.append(f"{coverage_label} {attrs[cov_key] * 100:.1f}%")

            if metrics:
                summary.append(f"  Channel {idx}: " + "; ".join(metrics))

    # Overall assessment
    try:
        ess_good = (
            (ess_values >= 400).mean() * 100
            if ess_values is not None
            else None
        )
        summary.append(f"\nOverall Convergence Assessment:")
        if ess_good is None:
            summary.append("  Status: UNKNOWN (insufficient diagnostics)")
        else:
            meets_rhat = (
                rhat_good is None or rhat_good >= 90
            )  # treat missing rhat as neutral
            if ess_good >= 90 and meets_rhat:
                summary.append("  Status: EXCELLENT ✓")
            elif ess_good >= 75 and meets_rhat:
                summary.append("  Status: GOOD ✓")
            else:
                summary.append("  Status: NEEDS ATTENTION ⚠")
    except:
        pass

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
            lines.append(f"Final ELBO: {float(loss_arr.reshape(-1)[-1]):.3f}")

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
