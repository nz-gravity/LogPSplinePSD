import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from ..logger import logger
from ..plotting.base import safe_plot, setup_plot_style
from .derived_weights import (
    HDI_PROB,
    REP_WEIGHT_ESS_K,
    REP_WEIGHT_EVEN_K,
    REP_WEIGHT_VAR_K,
    build_plot_dataset,
    compute_weight_summaries,
    find_weight_vars,
    select_rep_indices,
)
from .run_all import run_all_diagnostics

# Setup consistent styling for diagnostics plots
setup_plot_style()

SAVE_ESS_RHAT_PROFILES = True


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
    save_acceptance: bool = False
    # Rank/pair plot guards for very high-dimensional posteriors.
    save_rank_plots: bool = False
    rank_max_vars: int = 6
    rank_max_dims_per_var: int = 6
    save_pair_plots: bool = False
    pair_max_vars: int = 4


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


def _build_arviz_plot_data(idata):
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return None, {}, []

    derived_scalar, derived_vector = compute_weight_summaries(
        idata, hdi_prob=HDI_PROB
    )

    rep_indices: dict[str, np.ndarray] = {}
    weight_vars = find_weight_vars(posterior)
    for name in weight_vars:
        try:
            rep_indices[name] = select_rep_indices(
                posterior[name],
                ess_k=REP_WEIGHT_ESS_K,
                var_k=REP_WEIGHT_VAR_K,
                even_k=REP_WEIGHT_EVEN_K,
            )
        except Exception:
            rep_indices[name] = np.array([], dtype=int)

    plot_ds = build_plot_dataset(idata, derived_scalar, rep_indices)
    if not plot_ds.data_vars:
        return None, derived_vector, weight_vars

    idata_plot = az.InferenceData(posterior=plot_ds)
    if hasattr(idata, "sample_stats"):
        try:
            idata_plot.add_groups(sample_stats=idata.sample_stats)
            _ensure_diverging_sample_stats(idata_plot)
        except Exception:
            pass
    return idata_plot, derived_vector, weight_vars


def _ensure_diverging_sample_stats(idata: az.InferenceData) -> None:
    if not hasattr(idata, "sample_stats"):
        return
    sample_stats = idata.sample_stats
    if "diverging" in sample_stats.data_vars:
        return
    channel_keys = [
        key
        for key in sample_stats.data_vars
        if str(key).startswith("diverging_channel_")
    ]
    if not channel_keys:
        return
    arrays = []
    for key in channel_keys:
        try:
            arrays.append(np.asarray(sample_stats[key]))
        except Exception:
            continue
    if not arrays:
        return
    try:
        stacked = np.stack(arrays, axis=0)
        combined = np.any(stacked > 0, axis=0).astype(int)
    except Exception:
        combined = (np.asarray(arrays[0]) > 0).astype(int)
    ref = sample_stats[channel_keys[0]]
    try:
        sample_stats["diverging"] = (ref.dims, combined)
    except Exception:
        sample_stats["diverging"] = combined


def _sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _create_ess_rhat_profiles(
    idata: az.InferenceData,
    diag_dir: str,
    config: DiagnosticsConfig,
    weight_vars: list[str],
) -> None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or not weight_vars:
        logger.info(
            "Diagnostics plot: ess_rhat_profiles skipped (no weights)."
        )
        return

    for name in weight_vars:
        if name not in posterior.data_vars:
            continue
        var = posterior[name]
        dims = [d for d in var.dims if d not in ("chain", "draw")]
        if not dims:
            continue
        basis_dim = dims[-1]
        try:
            ess = az.ess(var, method="bulk")
            rhat = az.rhat(var)
            ess_vals = np.asarray(ess)
            rhat_vals = np.asarray(rhat)
        except Exception:
            continue

        if ess_vals.size == 0 or rhat_vals.size == 0:
            continue

        fname = _sanitize_filename(f"ess_rhat_profile_{name}.png")

        @safe_plot(f"{diag_dir}/{fname}", config.dpi)
        def _plot():
            fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            axes[0].plot(ess_vals, color="C0", linewidth=1.5)
            axes[0].axhline(
                config.ess_threshold, color="red", linestyle="--", linewidth=1
            )
            axes[0].set_ylabel("ESS (bulk)")
            axes[0].set_title(f"{name}: ESS/R-hat profile")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(rhat_vals, color="C1", linewidth=1.5)
            axes[1].axhline(
                config.rhat_threshold, color="red", linestyle="--", linewidth=1
            )
            axes[1].set_xlabel("Basis index")
            axes[1].set_ylabel("R-hat")
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        _plot()


def _is_multivar(idata: az.InferenceData) -> bool:
    attrs = getattr(idata, "attrs", {}) or {}
    if str(attrs.get("data_type", "")).lower().startswith("multi"):
        return True
    psd = getattr(idata, "posterior_psd", None)
    return psd is not None and "psd_matrix_real" in psd


def _get_freqs(idata: az.InferenceData, model=None) -> np.ndarray:
    psd = getattr(idata, "posterior_psd", None)
    if psd is not None:
        if "psd" in psd and "freq" in psd["psd"].coords:
            return np.asarray(psd["psd"].coords["freq"])
        if (
            "psd_matrix_real" in psd
            and "freq" in psd["psd_matrix_real"].coords
        ):
            return np.asarray(psd["psd_matrix_real"].coords["freq"])
    if model is not None and hasattr(model, "basis"):
        try:
            return np.arange(np.asarray(model.basis).shape[0])
        except Exception:
            pass
    return np.array([])


def _posterior_draw_count(idata: az.InferenceData) -> int:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return 0
    chains = int(posterior.sizes.get("chain", 0) or 0)
    draws = int(posterior.sizes.get("draw", 0) or 0)
    return int(chains * draws)


def plot_diagnostics(
    idata: az.InferenceData,
    outdir: str,
    p: Optional[int] = None,
    N: Optional[int] = None,
    runtime: Optional[float] = None,
    config: Optional[DiagnosticsConfig] = None,
    model=None,
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
    _create_diagnostic_plots(idata, diag_dir, config, p, N, runtime, model)
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


def _create_diagnostic_plots(idata, diag_dir, config, p, N, runtime, model):
    """Create only the essential diagnostic plots."""
    logger.debug("Generating diagnostic plots...")

    idata_plot, vector_summaries, weight_vars = _build_arviz_plot_data(idata)

    # 1. ArviZ trace plots (lightweight subset)
    @safe_plot(f"{diag_dir}/trace_plots.png", config.dpi)
    def create_trace_plots():
        if idata_plot is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.text(
                0.5,
                0.5,
                "Trace plot skipped\n(no suitable variables)",
                ha="center",
                va="center",
            )
            ax.set_title("Parameter Traces (skipped)")
            ax.axis("off")
            plt.tight_layout()
            return fig

        has_divergences = False
        divergences_arg: str | None = None
        if hasattr(idata_plot, "sample_stats"):
            sample_stats_vars = list(idata_plot.sample_stats.data_vars)
            has_divergences = "diverging" in sample_stats_vars
            divergences_arg = "diverging" if has_divergences else None

        # Calculate number of variables to scale figure size
        num_vars = len(idata_plot.posterior.data_vars)
        figsize_height = max(4, num_vars * 1.5)
        figsize = (14, figsize_height)

        # Create trace plot with improved layout
        # Use idata_plot which has the filtered posterior + sample_stats
        axes = az.plot_trace(
            idata_plot,
            combined=False,
            compact=False,
            figsize=figsize,
            divergences=divergences_arg,
        )

        fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else plt.gcf()

        # Add Rhat values to subplot titles
        try:
            posterior = idata_plot.posterior
            for var_name in idata_plot.posterior.data_vars:
                var_data = posterior[var_name]
                rhat = az.rhat(var_data)

                # Extract values from xarray object
                if hasattr(rhat, "values"):
                    rhat_array = np.asarray(rhat.values)
                else:
                    rhat_array = np.asarray(rhat)

                # Handle both scalar and array R-hats
                if rhat_array.ndim == 0:
                    rhat_val = float(rhat_array)
                else:
                    rhat_val = float(rhat_array.mean())

                # Find axes for this variable and update title
                for ax in fig.axes:
                    title = ax.get_title()
                    # Match by variable name in the title
                    if any(part == var_name for part in title.split()):
                        if "R-hat" not in title:
                            new_title = f"{title}\nR-hat: {rhat_val:.4f}"
                            ax.set_title(new_title, fontsize=10)
                            break
        except Exception as e:
            logger.debug(f"Could not add Rhat values to trace plot: {e}")

        fig.tight_layout()
        return fig

    t = time.perf_counter()
    logger.info("Diagnostics plot: trace_plots.png starting")
    ok = create_trace_plots()
    logger.info(
        f"Diagnostics plot: trace_plots.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
    )

    # 2. Rank histogram diagnostics (subsampled variable list)
    t = time.perf_counter()
    logger.info("Diagnostics plot: rank_plots starting")
    _create_rank_diagnostics(idata_plot, diag_dir, config)
    logger.info(
        f"Diagnostics plots: rank_plots done in {time.perf_counter() - t:.2f}s"
    )

    # 3. Optional pair diagnostics for low-dimensional scalar variables
    t = time.perf_counter()
    logger.info("Diagnostics plot: pair_plots starting")
    _create_pair_diagnostics(idata_plot, diag_dir, config)
    logger.info(
        f"Diagnostics plots: pair_plots done in {time.perf_counter() - t:.2f}s"
    )

    if SAVE_ESS_RHAT_PROFILES:
        t = time.perf_counter()
        logger.info("Diagnostics plot: ess_rhat_profiles starting")
        _create_ess_rhat_profiles(idata, diag_dir, config, weight_vars)
        logger.info(
            f"Diagnostics plots: ess_rhat_profiles done in {time.perf_counter() - t:.2f}s"
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

    # 5. Acceptance rate diagnostics
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

    # 6. Sampler-specific diagnostics
    t = time.perf_counter()
    logger.info("Diagnostics plots: sampler-specific starting")
    _create_sampler_diagnostics(idata, diag_dir, config)
    logger.info(
        f"Diagnostics plots: sampler-specific done in {time.perf_counter() - t:.2f}s"
    )

    # 7. Divergences diagnostics now included in summary dashboard.


def _create_rank_diagnostics(idata, diag_dir, config):
    if not config.save_rank_plots:
        logger.info("Diagnostics plot: rank_plots skipped (disabled).")
        return
    if idata is None:
        logger.info("Diagnostics plot: rank_plots skipped (no data).")
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
    if idata is None:
        logger.info("Diagnostics plot: pair_plots skipped (no data).")
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

    # Create 2x3 layout (adds divergences overview)
    fig, axes = plt.subplots(
        2, 3, figsize=(config.figsize[0] * 1.3, config.figsize[1])
    )
    ess_ax = axes[0, 0]
    meta_ax = axes[0, 1]
    div_trace_ax = axes[0, 2]
    param_ax = axes[1, 0]
    status_ax = axes[1, 1]
    div_summary_ax = axes[1, 2]

    # Get ESS values once
    ess_values = None
    try:
        ess = idata.attrs.get("ess")
        ess_arr = np.asarray(ess) if ess is not None else None
        if ess_arr is not None:
            ess_values = ess_arr[np.isfinite(ess_arr)]
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

    # 5. Divergences overview (if available)
    _plot_divergences_panels(idata, div_trace_ax, div_summary_ax, config)

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


def _compute_ebfmi_per_chain(energy_values: np.ndarray) -> list[float]:
    energy = np.asarray(energy_values)
    if energy.ndim == 0:
        return []
    if energy.ndim == 1:
        ebfmi = _compute_ebfmi_from_energy(energy)
        return [ebfmi] if ebfmi is not None else []
    ebfmi_vals: list[float] = []
    for chain_idx in range(int(energy.shape[0])):
        chain_energy = np.asarray(energy[chain_idx]).reshape(-1)
        ebfmi = _compute_ebfmi_from_energy(chain_energy)
        if ebfmi is not None:
            ebfmi_vals.append(ebfmi)
    return ebfmi_vals


def _compute_ebfmi_metrics(
    idata, attrs: dict
) -> tuple[Optional[float], dict[int, dict[str, float]]]:
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

    ebfmi_by_channel: dict[int, dict[str, float]] = {}
    if hasattr(idata, "sample_stats"):
        if "energy" in idata.sample_stats:
            ebfmi_vals = _compute_ebfmi_per_chain(
                np.asarray(idata.sample_stats["energy"])
            )
            if ebfmi_vals and ebfmi is None:
                ebfmi = float(np.min(ebfmi_vals))
        else:
            for key in idata.sample_stats:
                if not str(key).startswith("energy_channel_"):
                    continue
                channel_str = str(key).replace("energy_channel_", "")
                try:
                    channel_idx = int(channel_str)
                except ValueError:
                    continue
                ebfmi_vals = _compute_ebfmi_per_chain(
                    np.asarray(idata.sample_stats[key])
                )
                if ebfmi_vals:
                    ebfmi_by_channel[channel_idx] = {
                        "min": float(np.min(ebfmi_vals)),
                        "median": float(np.median(ebfmi_vals)),
                    }
            if ebfmi is None and ebfmi_by_channel:
                ebfmi = float(
                    min(info["min"] for info in ebfmi_by_channel.values())
                )

    return ebfmi, ebfmi_by_channel


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

    ebfmi, ebfmi_by_channel = _compute_ebfmi_metrics(idata, attrs)
    metrics["ebfmi"] = ebfmi
    if ebfmi_by_channel:
        metrics["ebfmi_by_channel"] = ebfmi_by_channel

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
        if candidate is None:
            continue
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

    Overlays per‑channel series when keys like ``step_size_channel_{j}`` or
    ``num_steps_channel_{j}`` are present.
    """
    # Presence of overall arrays
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

    steps_ch = _collect("num_steps")
    accept_ch = _collect("accept_prob")
    step_size_ch = _collect("step_size")

    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    # Step size trace
    ax = axes[0, 0]
    plotted = False
    if has_step_size:
        ax.plot(
            idata.sample_stats.step_size.values.flatten(),
            alpha=0.7,
            lw=1,
            label="overall",
        )
        plotted = True
    for ch in sorted(step_size_ch):
        ax.plot(step_size_ch[ch], alpha=0.6, lw=1, label=f"ch {ch}")
        plotted = True
    if not plotted:
        ax.text(
            0.5, 0.5, "Step size data\nunavailable", ha="center", va="center"
        )
    ax.set_title("Step Size Trace")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("step_size")
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

    num_steps = _get("num_steps")
    step_size = _get("step_size")
    accept_prob = _get("accept_prob")
    attrs = getattr(idata, "attrs", {}) or {}
    if not hasattr(attrs, "get"):
        attrs = dict(attrs)
    max_tree_depth = _coerce_int(attrs.get("max_tree_depth"))

    fig, axes = plt.subplots(2, 2, figsize=config.figsize)

    # Step size trace
    ax = axes[0, 0]
    if step_size is not None:
        ax.plot(step_size, alpha=0.7, lw=1, label="step_size")
    else:
        ax.text(
            0.5, 0.5, "Step size data\nunavailable", ha="center", va="center"
        )
    ax.set_title(f"Channel {channel_idx} Step Size")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("step_size")
    ax.grid(True, alpha=0.3)
    if step_size is not None:
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

    if idata is None or not hasattr(idata, "sample_stats"):
        return

    # Better sampler detection - check sampler type first
    sampler_type = (
        idata.attrs["sampler_type"].lower()
        if "sampler_type" in idata.attrs
        else "unknown"
    )

    # Check for NUTS-specific fields
    nuts_specific_fields = [
        "num_steps",
        "tree_depth",
        "diverging",
        "step_size",
        "accept_prob",
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
    divergences_data = _collect_divergences_data(idata)

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


def _collect_divergences_data(idata) -> dict[str | int, np.ndarray]:
    if not hasattr(idata, "sample_stats"):
        return {}
    divergences_data: dict[str | int, np.ndarray] = {}
    sample_stats = idata.sample_stats
    if "diverging" in sample_stats:
        divergences_data["main"] = sample_stats.diverging.values.flatten()

    for key in sample_stats:
        if str(key).startswith("diverging_channel_"):
            channel_idx = str(key).replace("diverging_channel_", "")
            try:
                idx = int(channel_idx)
            except ValueError:
                continue
            divergences_data[idx] = sample_stats[key].values.flatten()
    return divergences_data


def _plot_divergences_panels(idata, trace_ax, summary_ax, config) -> None:
    divergences_data = _collect_divergences_data(idata)
    if not divergences_data:
        trace_ax.text(
            0.5,
            0.5,
            "No divergence data available",
            ha="center",
            va="center",
        )
        trace_ax.set_title("Divergences")
        trace_ax.axis("off")
        summary_ax.axis("off")
        return

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
            linewidth=1.2,
            alpha=0.9,
            label=label_str,
        )
        total_divergences += int(np.sum(div))
        total_iterations += int(div.size)

    trace_ax.set_xlabel("Iteration")
    trace_ax.set_ylabel("Cumulative")
    trace_ax.set_title("Divergences (cumulative)")
    trace_ax.grid(True, alpha=0.3)
    trace_ax.legend(loc="best", fontsize="small")

    summary_ax.text(
        0.05,
        0.95,
        _get_divergences_summary(divergences_data),
        transform=summary_ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )
    summary_ax.set_title("Divergences Summary")
    summary_ax.axis("off")


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
    has_sample_stats = hasattr(idata, "sample_stats")

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
        bulk_status = (
            "✓ excellent"
            if ess_min >= 1000
            else "✓ good" if ess_min >= 400 else "⚠ low"
        )
        summary.append(
            f"\nESS bulk: min={ess_min:.0f}"
            + (f", median={ess_med:.0f}" if ess_med is not None else "")
            + f" {bulk_status}"
        )
    if ess_tail_min is not None:
        tail_status = (
            "✓ excellent"
            if ess_tail_min >= 1000
            else "✓ good" if ess_tail_min >= 400 else "⚠ low"
        )
        summary.append(
            f"ESS tail: min={ess_tail_min:.0f}"
            + (
                f", median={ess_tail_med:.0f}"
                if ess_tail_med is not None
                else ""
            )
            + f" {tail_status}"
        )
    if rhat_max is not None:
        rhat_status = (
            "✓ excellent"
            if rhat_max <= 1.01
            else "⚠ high" if rhat_max <= 1.1 else "⚠ problematic"
        )
        summary.append(
            f"Rhat: max={rhat_max:.3f}"
            + (f", mean={rhat_mean:.3f}" if rhat_mean is not None else "")
            + f" {rhat_status}"
        )

    # Acceptance rate / divergences (cheap: read directly from sample_stats).
    if has_sample_stats:
        try:
            acc = None
            if "accept_prob" in idata.sample_stats:
                acc = float(
                    np.mean(np.asarray(idata.sample_stats["accept_prob"]))
                )
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
                    status = (
                        "✓ ok"
                        if div_pct < 1
                        else (
                            "⚠ check geometry"
                            if div_pct < 5
                            else "⚠ problematic"
                        )
                    )
                    summary.append(
                        f"Divergences: {div_count}/{div_total} ({div_pct:.2f}%) {status}"
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

    ebfmi, ebfmi_by_channel = _compute_ebfmi_metrics(idata, attrs)
    if ebfmi is not None:
        status = (
            "✓ ok"
            if ebfmi >= 0.3
            else "⚠ borderline" if ebfmi >= 0.2 else "⚠ problematic"
        )
        summary.append(f"E-BFMI: {ebfmi:.3f} {status}")
    if ebfmi_by_channel:
        for ch in sorted(ebfmi_by_channel):
            info = ebfmi_by_channel[ch]
            ch_status = (
                "✓ ok"
                if info["median"] >= 0.3
                else (
                    "⚠ borderline"
                    if info["median"] >= 0.2
                    else "⚠ problematic"
                )
            )
            summary.append(
                f"  Channel {ch}: min={info['min']:.3f}, median={info['median']:.3f} {ch_status}"
            )

    if has_sample_stats:
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
            channel_indices = _get_channel_indices(
                idata.sample_stats, "num_steps"
            )
            for ch in sorted(channel_indices):
                steps_by_channel[ch] = idata.sample_stats[
                    f"num_steps_channel_{ch}"
                ].values.flatten()
            if steps_by_channel:
                num_steps = np.concatenate(list(steps_by_channel.values()))
        max_stats = _max_tree_depth_stats(
            num_steps, tree_depth, max_tree_depth
        )
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
            channel_indices = _get_channel_indices(
                idata.sample_stats, "step_size"
            )
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
            channel_indices = _get_channel_indices(
                idata.sample_stats, "diverging"
            )
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

    max_hyper = 5
    max_blocks = 3
    min_bias_pct = 10.0
    min_var_ratio_dev = 0.2

    lines = []
    lines.append("=== VI Diagnostics Summary ===")
    lines.append("")
    moment_summary = diagnostics.get("psis_moment_summary") or {}
    weight_stats = moment_summary.get("weights")
    weight_blocks = moment_summary.get("weights_by_block") or []
    hyper_params = moment_summary.get("hyperparameters") or []
    corr_summary = diagnostics.get("psis_correlation_summary") or {}

    khat_max = diagnostics.get("psis_khat_max")
    threshold = diagnostics.get("psis_khat_threshold", 0.7)
    if khat_max is not None and np.isfinite(khat_max):
        status = diagnostics.get("psis_status_message") or diagnostics.get(
            "psis_khat_status", ""
        )
        status_suffix = f" ({status})" if status else ""
        lines.append(f"PSIS k-hat (max): {float(khat_max):.3f}{status_suffix}")
        if khat_max > threshold:
            lines.append(
                f"PSIS alert: k-hat exceeds {threshold:.1f} -> posterior may be unreliable"
            )

    # Overall quality indicator
    quality = "OK"
    if diagnostics.get("psis_flag_critical"):
        quality = "❌ NOT TRUSTWORTHY"
    elif diagnostics.get("psis_flag_warn"):
        quality = "⚠ USE WITH CAUTION"
    else:
        for entry in hyper_params:
            thresholds = moment_summary.get("thresholds", {})
            bias_thr = thresholds.get("bias_threshold", 0.05) * 100.0
            var_low = thresholds.get("var_low", 0.7)
            var_high = thresholds.get("var_high", 1.3)
            if (
                abs(entry.get("bias_pct", 0.0)) > bias_thr
                or entry.get("var_ratio", 1.0) < var_low
                or entry.get("var_ratio", 1.0) > var_high
            ):
                quality = "⚠ USE WITH CAUTION"
                break

    guide = diagnostics.get("guide", "vi")
    weight_dispersion = ""
    if weight_stats:
        median_ratio = weight_stats.get("var_ratio_median", np.nan)
        if np.isfinite(median_ratio):
            weight_dispersion = (
                "under-dispersed" if median_ratio < 1.0 else "over-dispersed"
            )
        else:
            weight_dispersion = "unknown dispersion"

    headline_parts = [f"VI Quality: {quality}"]
    if khat_max is not None and np.isfinite(khat_max):
        headline_parts.append(
            "k-hat>0.7" if khat_max > threshold else "k-hat<=0.7"
        )
    if weight_dispersion:
        headline_parts.append(f"Weights {weight_dispersion}")
    if guide:
        headline_parts.append(f"Guide={guide}")
    lines.append(" | ".join(headline_parts))

    if weight_stats:
        frac = weight_stats.get("frac_outside")
        lines.append(
            "Weights var_ratio "
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
        if (
            weight_stats.get("bias_abs_median") is not None
            or weight_stats.get("bias_abs_max") is not None
        ):
            lines.append(
                "Weights abs_bias "
                + ", ".join(
                    [
                        f"median={weight_stats.get('bias_abs_median', np.nan):.3g}",
                        f"max={weight_stats.get('bias_abs_max', np.nan):.3g}",
                    ]
                )
            )

    worst_blocks = []
    if weight_blocks:
        scored_blocks = []
        for entry in weight_blocks:
            median = entry.get("var_ratio_median", np.nan)
            frac = entry.get("frac_outside", 0.0)
            score = float(frac) + float(abs(median - 1.0))
            scored_blocks.append((score, entry))
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        worst = scored_blocks[:max_blocks]
        worst_blocks = [
            (
                entry.get("block"),
                entry.get("var_ratio_median", np.nan),
                entry.get("frac_outside", np.nan),
            )
            for _, entry in worst
        ]
    if worst_blocks:
        parts = [
            f"{block_id}(med={median:.2f},out={frac*100:.1f}%)"
            for block_id, median, frac in worst_blocks
        ]
        lines.append("Worst blocks (weights): " + ", ".join(parts))

    top_hyper = []
    if hyper_params:
        scored = []
        for entry in hyper_params:
            bias_pct = float(entry.get("bias_pct", 0.0))
            var_ratio = float(entry.get("var_ratio", 1.0))
            var_dev = abs(var_ratio - 1.0)
            if abs(bias_pct) < min_bias_pct and var_dev < min_var_ratio_dev:
                continue
            score = abs(bias_pct) + 100.0 * var_dev
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_hyper = [entry for _, entry in scored[:max_hyper]]
    if top_hyper:
        lines.append("Top hyperparameters (by severity):")
        for entry in top_hyper:
            var_ratio = float(entry.get("var_ratio", 1.0))
            bias_pct = float(entry.get("bias_pct", 0.0))
            lines.append(
                f"  {entry['param']}  "
                f"var_ratio={var_ratio:.2f}, bias={bias_pct:.1f}%"
                + (
                    f", abs={entry['bias_abs']:.3g}"
                    if entry.get("bias_abs") is not None
                    else ""
                )
            )

    corr_lines = []
    if corr_summary:
        by_label: dict[str, dict] = {}
        for label, stats in corr_summary.items():
            if not stats:
                continue
            base = str(label).split("_block_")[0]
            try:
                block_id = int(str(label).split("_block_")[1])
            except Exception:
                block_id = None
            max_abs = stats.get("max_abs", np.nan)
            current = by_label.get(base)
            if current is None or max_abs > current["max_abs"]:
                by_label[base] = {
                    "max_abs": max_abs,
                    "block": block_id,
                }
        for base, stats in by_label.items():
            block_suffix = (
                f" (block {stats['block']})"
                if stats.get("block") is not None
                else ""
            )
            corr_lines.append(
                f"{base} max|r|={stats.get('max_abs', np.nan):.3f}{block_suffix}"
            )
    if corr_lines:
        lines.append("Guide-structure corr: " + "; ".join(corr_lines))

    lines.append(
        "Next: try a richer guide (mvn/lowrank/flow) or increase VI steps/draws."
    )
    lines.append(
        "Legend: var_ratio<1 under-dispersed, >1 over-dispersed. k-hat>0.7 unreliable."
    )

    losses = diagnostics.get("losses")
    vi_samples = diagnostics.get("vi_samples")
    elbo_value = None
    if losses is not None:
        loss_arr = np.asarray(losses)
        if loss_arr.size:
            final_elbo = float(loss_arr.reshape(-1)[-1])
            if np.isfinite(final_elbo):
                elbo_value = final_elbo
    n_draws = None
    if vi_samples:
        first = next(iter(vi_samples.values()))
        n_draws = int(np.asarray(first).shape[0])
    if elbo_value is not None or n_draws is not None:
        parts = []
        if elbo_value is not None:
            parts.append(f"ELBO={elbo_value:.3f}")
        if n_draws is not None:
            parts.append(f"draws={n_draws}")
        lines.append(" | ".join(parts))

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
