from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arviz_plots import plot_pair, plot_rank, plot_trace
from arviz_stats import ess, rhat

from ..arviz_utils.from_arviz import (
    get_multivar_posterior_psd_quantiles,
    get_posterior_psd,
)
from ..logger import logger
from ..plotting.base import (
    composite_images_vertical,
    safe_plot,
    setup_plot_style,
)
from ._utils import extract_percentile
from .psd_compare import _run as _run_psd_compare
from .run_all import run_all_diagnostics

# Setup consistent styling for diagnostics plots
setup_plot_style()

# Backwards-compatibility shim for tests and older internal call sites that
# still patch/access ``diag_mod.az`` during the ArviZ split transition.
az = SimpleNamespace(
    plot_trace=plot_trace,
    plot_rank=plot_rank,
    plot_pair=plot_pair,
    ess=ess,
    rhat=rhat,
    loo=None,
)


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
    save_rank_plots: bool = True
    rank_max_vars: int = 6
    rank_max_dims_per_var: int = 6
    save_pair_plots: bool = False
    pair_max_vars: int = 4
    trace_max_plots: int = 15
    trace_plot_seed: int = 0
    random_weight_indices_per_var: int = 6
    save_ess_rhat_profiles: bool = True
    save_ess_rhat_profiles_individual: bool = False
    save_nuts_block_diagnostics_individual: bool = False


def _to_flat_finite_array(obj) -> np.ndarray:
    """Convert ArviZ/xarray outputs to a flat finite float array."""
    if obj is None:
        return np.array([], dtype=float)
    if hasattr(obj, "data_vars"):
        values: list[np.ndarray] = []
        for var in obj.data_vars.values():
            arr = np.asarray(var).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                values.append(arr)
        if not values:
            return np.array([], dtype=float)
        return np.concatenate(values)
    arr = np.asarray(obj).reshape(-1)
    return arr[np.isfinite(arr)]


def _var_priority(name: str) -> tuple[int, str]:
    if name.startswith("delta"):
        return (0, name)
    if name.startswith("phi"):
        return (1, name)
    if name.startswith("weights"):
        return (2, name)
    return (3, name)


def _find_weight_vars(posterior: xr.Dataset) -> list[str]:
    return [
        str(name)
        for name in posterior.data_vars
        if str(name).startswith("weights")
    ]


def _basis_dim(var: xr.DataArray) -> str | None:
    dims = [d for d in var.dims if d not in ("chain", "draw")]
    return str(dims[-1]) if dims else None


def _select_random_weight_indices(
    var: xr.DataArray, *, max_k: int, rng: np.random.Generator
) -> np.ndarray:
    basis_dim = _basis_dim(var)
    if basis_dim is None:
        return np.array([], dtype=int)
    size = int(var.sizes.get(basis_dim, 0))
    if size <= 0:
        return np.array([], dtype=int)
    k = min(int(max_k), size)
    if k <= 0:
        return np.array([], dtype=int)
    if k == size:
        return np.arange(size, dtype=int)
    return np.sort(rng.choice(size, size=k, replace=False)).astype(int)


def _build_plot_dataset_random_weights(
    idata,
    rep_indices: dict[str, np.ndarray],
) -> xr.Dataset:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return xr.Dataset()

    ds = xr.Dataset()
    weight_vars = set(_find_weight_vars(posterior))

    for name, var in posterior.data_vars.items():
        if "chain" not in var.dims or "draw" not in var.dims:
            continue
        if str(name) in weight_vars:
            continue
        extra_dims = [d for d in var.dims if d not in ("chain", "draw")]
        if extra_dims:
            continue
        ds[str(name)] = var

    for wname, idxs in rep_indices.items():
        if wname not in posterior.data_vars:
            continue
        var = posterior[wname]
        basis_dim = _basis_dim(var)
        if basis_dim is None:
            continue
        for idx in np.asarray(idxs, dtype=int):
            try:
                sliced = var.isel({basis_dim: int(idx)})
            except Exception:
                continue
            ds[f"{wname}__idx_{int(idx)}"] = sliced

    return ds


def _flat_dim_from_shape(shape: tuple[int, ...]) -> int:
    if len(shape) <= 2:
        return 1
    return int(np.prod(shape[2:]))


def _select_rank_plot_vars(
    idata: xr.DataTree, config: DiagnosticsConfig
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
    idata: xr.DataTree, config: DiagnosticsConfig
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


def _unravel_flat_index(
    flat_idx: int, shape: tuple[int, ...]
) -> tuple[int, ...]:
    if not shape:
        return ()
    return tuple(int(v) for v in np.unravel_index(flat_idx, shape))


def _build_trace_plot_idata(idata_plot, config: DiagnosticsConfig):
    """Build a reduced trace dataset with at most `trace_max_plots` scalar series."""
    if idata_plot is None or not hasattr(idata_plot, "posterior"):
        return idata_plot, 0, 0

    posterior = idata_plot.posterior
    if posterior is None:
        return idata_plot, 0, 0

    chain_size = int(posterior.sizes.get("chain", 0) or 0)
    draw_size = int(posterior.sizes.get("draw", 0) or 0)
    if chain_size <= 0 or draw_size <= 0:
        return idata_plot, 0, 0

    candidates: list[tuple[str, int, tuple[int, ...]]] = []
    for var_name in posterior.data_vars:
        var = posterior[var_name]
        if "chain" not in var.dims or "draw" not in var.dims:
            continue
        trailing_dims = [d for d in var.dims if d not in ("chain", "draw")]
        trailing_shape = tuple(int(var.sizes[d]) for d in trailing_dims)
        flat_dim = int(np.prod(trailing_shape)) if trailing_shape else 1
        for flat_idx in range(flat_dim):
            candidates.append((str(var_name), int(flat_idx), trailing_shape))

    total = len(candidates)
    if total == 0:
        return idata_plot, 0, 0

    max_plots = max(1, int(config.trace_max_plots))
    if total > max_plots:
        rng = np.random.default_rng(int(config.trace_plot_seed))
        selected_idx = np.sort(
            rng.choice(total, size=max_plots, replace=False)
        )
    else:
        selected_idx = np.arange(total, dtype=int)

    chain_coords = (
        posterior.coords["chain"].values
        if "chain" in posterior.coords
        else np.arange(chain_size)
    )
    draw_coords = (
        posterior.coords["draw"].values
        if "draw" in posterior.coords
        else np.arange(draw_size)
    )

    trace_vars: dict[str, xr.DataArray] = {}
    for idx in selected_idx:
        var_name, flat_idx, trailing_shape = candidates[int(idx)]
        var = posterior[var_name].transpose(
            "chain",
            "draw",
            *[
                d
                for d in posterior[var_name].dims
                if d not in ("chain", "draw")
            ],
        )
        values = np.asarray(var.values).reshape(chain_size, draw_size, -1)
        if values.shape[-1] <= int(flat_idx):
            continue
        series = values[:, :, int(flat_idx)]
        if trailing_shape:
            coords_suffix = ",".join(
                str(v)
                for v in _unravel_flat_index(int(flat_idx), trailing_shape)
            )
            label = f"{var_name}[{coords_suffix}]"
        else:
            label = str(var_name)
        trace_vars[label] = xr.DataArray(
            series,
            dims=("chain", "draw"),
            coords={"chain": chain_coords, "draw": draw_coords},
        )

    if not trace_vars:
        return idata_plot, total, 0

    trace_ds = xr.Dataset(trace_vars)
    trace_idata = xr.DataTree()
    trace_idata["posterior"] = xr.DataTree(dataset=trace_ds)
    if hasattr(idata_plot, "sample_stats"):
        try:
            sample_stats = idata_plot.sample_stats
            sample_stats_ds = (
                sample_stats.ds
                if hasattr(sample_stats, "ds")
                else sample_stats
            )
            trace_idata["sample_stats"] = xr.DataTree(dataset=sample_stats_ds)
        except Exception:
            pass
    return trace_idata, total, len(trace_vars)


def _build_arviz_plot_data(idata, config: DiagnosticsConfig):
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return None, []

    rep_indices: dict[str, np.ndarray] = {}
    weight_vars = _find_weight_vars(posterior)
    rng = np.random.default_rng(int(config.trace_plot_seed))
    for name in weight_vars:
        try:
            rep_indices[name] = _select_random_weight_indices(
                posterior[name],
                max_k=int(config.random_weight_indices_per_var),
                rng=rng,
            )
        except Exception:
            rep_indices[name] = np.array([], dtype=int)

    plot_ds = _build_plot_dataset_random_weights(idata, rep_indices)
    if not plot_ds.data_vars:
        return None, weight_vars

    idata_plot = xr.DataTree()
    idata_plot["posterior"] = xr.DataTree(dataset=plot_ds)
    if hasattr(idata, "sample_stats"):
        try:
            ss = idata.sample_stats
            ss_ds = ss.ds if hasattr(ss, "ds") else ss
            idata_plot["sample_stats"] = xr.DataTree(dataset=ss_ds)
            _ensure_diverging_sample_stats(idata_plot)
        except Exception:
            pass
    return idata_plot, weight_vars


def _ensure_diverging_sample_stats(idata: xr.DataTree) -> None:
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
    idata: xr.DataTree,
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

    profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name in weight_vars:
        if name not in posterior.data_vars:
            continue
        var = posterior[name]
        dims = [d for d in var.dims if d not in ("chain", "draw")]
        if not dims:
            continue
        try:
            ess_vals = ess(var, method="bulk")
            rhat_vals = rhat(var)
            ess_vals = _to_flat_finite_array(ess_vals)
            rhat_vals = _to_flat_finite_array(rhat_vals)
        except Exception:
            continue

        if ess_vals.size == 0 or rhat_vals.size == 0:
            continue

        profiles.append((name, ess_vals, rhat_vals))

    if not profiles:
        logger.info(
            "Diagnostics plot: ess_rhat_profiles skipped (no finite ESS/R-hat)."
        )
        return

    @safe_plot(f"{diag_dir}/ess_rhat_profiles.png", config.dpi)
    def _plot_combined():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
        ess_ax = axes[0, 0]
        rhat_ax = axes[0, 1]

        for name, ess_vals, rhat_vals in profiles:
            ess_ax.plot(ess_vals, linewidth=1.3, alpha=0.9, label=str(name))
            rhat_ax.plot(rhat_vals, linewidth=1.3, alpha=0.9, label=str(name))

        ess_ax.axhline(
            config.ess_threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"threshold={config.ess_threshold}",
        )
        ess_ax.set_ylabel("ESS (bulk)")
        ess_ax.set_title("ESS Profiles")
        ess_ax.set_xlabel("Basis index")
        ess_ax.grid(True, alpha=0.3)

        rhat_ax.axhline(
            config.rhat_threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"threshold={config.rhat_threshold:.3f}",
        )
        rhat_ax.set_ylabel("R-hat")
        rhat_ax.set_title("R-hat Profiles")
        rhat_ax.set_xlabel("Basis index")
        rhat_ax.grid(True, alpha=0.3)

        legend_kwargs = {"fontsize": "small", "framealpha": 0.9}
        if len(profiles) > 8:
            legend_kwargs["ncol"] = 2
        ess_ax.legend(loc="best", **legend_kwargs)
        rhat_ax.legend(loc="best", **legend_kwargs)
        fig.tight_layout()
        return fig

    _plot_combined()

    if not config.save_ess_rhat_profiles_individual:
        return

    for name, ess_vals, rhat_vals in profiles:
        fname = _sanitize_filename(f"ess_rhat_profile_{name}.png")

        @safe_plot(f"{diag_dir}/{fname}", config.dpi)
        def _plot_single(name=name, ess_vals=ess_vals, rhat_vals=rhat_vals):
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
            fig.tight_layout()
            return fig

        _plot_single()


def _is_multivar(idata: xr.DataTree) -> bool:
    attrs = getattr(idata, "attrs", {}) or {}
    return str(attrs.get("data_type", "")).lower().startswith("multi")


def _get_freqs(idata: xr.DataTree, model=None) -> np.ndarray:
    attrs = getattr(idata, "attrs", {}) or {}
    if str(attrs.get("data_type", "")).lower().startswith("multi"):
        quantiles = get_multivar_posterior_psd_quantiles(idata)
        return np.asarray(quantiles["freq"], dtype=float)
    try:
        freqs, _, _, _ = get_posterior_psd(idata)
        return np.asarray(freqs, dtype=float)
    except Exception:
        pass
    if model is not None and hasattr(model, "basis"):
        try:
            return np.arange(np.asarray(model.basis).shape[0])
        except Exception:
            pass
    return np.array([])


def _posterior_draw_count(idata: xr.DataTree) -> int:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return 0
    chains = int(posterior.sizes.get("chain", 0) or 0)
    draws = int(posterior.sizes.get("draw", 0) or 0)
    return int(chains * draws)


def _resolve_true_psd(
    idata: xr.DataTree, true_psd: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    if true_psd is not None:
        try:
            arr = np.asarray(true_psd)
            if arr.ndim == 3:
                return arr
        except Exception:
            pass

    attrs = getattr(idata, "attrs", {}) or {}
    if hasattr(attrs, "get"):
        attr_val = attrs.get("true_psd")
        if attr_val is not None:
            try:
                arr = np.asarray(attr_val)
                if arr.ndim == 3:
                    return arr
            except Exception:
                pass
    return None


def _entry_labels(n_channels: int) -> list[str]:
    labels: list[str] = []
    for i in range(n_channels):
        for j in range(n_channels):
            labels.append(f"S{i + 1}{j + 1}")
    return labels


def _relative_error_epsilon(values: np.ndarray) -> float:
    abs_vals = np.abs(np.asarray(values))
    finite = abs_vals[np.isfinite(abs_vals)]
    if finite.size == 0:
        return 1e-12
    ref = float(np.percentile(finite, 5.0))
    return max(1e-12, 1e-6 * ref)


def _build_multivar_truth_frequency_maps(
    idata: xr.DataTree, true_psd: Optional[np.ndarray]
) -> Optional[tuple[np.ndarray, list[str], np.ndarray, np.ndarray]]:
    if not _is_multivar(idata):
        return None
    quantiles = get_multivar_posterior_psd_quantiles(idata)
    psd_real = np.asarray(quantiles["real"], dtype=np.float64)
    psd_imag = np.asarray(quantiles["imag"], dtype=np.float64)
    freqs = np.asarray(quantiles["freq"], dtype=float)
    percentiles = np.asarray(quantiles["percentile"], dtype=float)

    truth = _resolve_true_psd(idata, true_psd)
    if truth is None:
        return None

    truth_arr = np.asarray(truth)
    if truth_arr.shape[0] != freqs.size:
        logger.info(
            f"Diagnostics plot: truth-aware PSD map skipped (truth freq bins={truth_arr.shape[0]}, posterior freq bins={freqs.size})."
        )
        return None

    n_channels = int(psd_real.shape[2])
    if truth_arr.shape[1:] != (n_channels, n_channels):
        logger.info(
            "Diagnostics plot: truth-aware PSD map skipped "
            + f"(truth matrix shape={truth_arr.shape[1:]}, posterior matrix shape={(n_channels, n_channels)})."
        )
        return None

    if percentiles.size == 0:
        percentiles = np.arange(psd_real.shape[0], dtype=float)

    q50_real = extract_percentile(psd_real, percentiles, 50.0)
    q05_real = extract_percentile(psd_real, percentiles, 5.0)
    q95_real = extract_percentile(psd_real, percentiles, 95.0)
    q50_imag = extract_percentile(psd_imag, percentiles, 50.0)
    q05_imag = extract_percentile(psd_imag, percentiles, 5.0)
    q95_imag = extract_percentile(psd_imag, percentiles, 95.0)

    estimate = q50_real + 1j * q50_imag
    q05 = q05_real + 1j * q05_imag
    q95 = q95_real + 1j * q95_imag
    truth_complex = truth_arr.astype(np.complex128, copy=False)

    eps = _relative_error_epsilon(truth_complex)
    denom = np.maximum(np.abs(truth_complex), eps)
    rel_error = np.abs(estimate - truth_complex) / denom

    lower_real = np.minimum(q05.real, q95.real)
    upper_real = np.maximum(q05.real, q95.real)
    inside_real = (truth_complex.real >= lower_real) & (
        truth_complex.real <= upper_real
    )

    lower_imag = np.minimum(q05.imag, q95.imag)
    upper_imag = np.maximum(q05.imag, q95.imag)
    inside_imag = (truth_complex.imag >= lower_imag) & (
        truth_complex.imag <= upper_imag
    )

    coverage = np.logical_and(inside_real, inside_imag).astype(float)

    labels = _entry_labels(n_channels)
    rel_error_map = np.asarray(rel_error).reshape(freqs.size, -1).T
    coverage_map = np.asarray(coverage).reshape(freqs.size, -1).T
    return freqs, labels, rel_error_map, coverage_map


def _create_truth_psd_frequency_diagnostics(
    idata: xr.DataTree,
    diag_dir: str,
    config: DiagnosticsConfig,
    true_psd: Optional[np.ndarray] = None,
) -> bool:
    maps = _build_multivar_truth_frequency_maps(idata, true_psd)
    if maps is None:
        logger.info(
            "Diagnostics plot: psd_truth_error_vs_freq skipped (requires multivariate posterior draws and true_psd)."
        )
        return False

    freqs, labels, rel_error_map, coverage_map = maps
    n_entries, n_freqs = rel_error_map.shape
    if n_entries == 0 or n_freqs == 0:
        return False

    finite_error = rel_error_map[np.isfinite(rel_error_map)]
    err_vmax = (
        float(np.percentile(finite_error, 95.0)) if finite_error.size else 1.0
    )
    if err_vmax <= 0:
        err_vmax = 1.0

    y_step = max(1, n_entries // 18)
    y_ticks = np.arange(0, n_entries, y_step, dtype=int)
    if y_ticks[-1] != n_entries - 1:
        y_ticks = np.append(y_ticks, n_entries - 1)

    x_step = max(1, n_freqs // 8)
    x_ticks = np.arange(0, n_freqs, x_step, dtype=int)
    if x_ticks[-1] != n_freqs - 1:
        x_ticks = np.append(x_ticks, n_freqs - 1)
    x_labels = [f"{freqs[idx]:.3g}" for idx in x_ticks]

    @safe_plot(f"{diag_dir}/psd_truth_error_vs_freq.png", config.dpi)
    def _plot():
        fig_height = float(np.clip(4.5 + 0.14 * n_entries, 6.0, 16.0))
        fig, axes = plt.subplots(2, 1, figsize=(12, fig_height), sharex=True)

        im_err = axes[0].imshow(
            rel_error_map,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=err_vmax,
            interpolation="nearest",
        )
        axes[0].set_title("Frequency-Resolved Relative Error")
        axes[0].set_ylabel("Spectral Entry")
        axes[0].set_yticks(y_ticks)
        axes[0].set_yticklabels([labels[idx] for idx in y_ticks], fontsize=8)
        cbar_err = fig.colorbar(im_err, ax=axes[0], pad=0.01)
        cbar_err.set_label("|S_hat - S_true| / max(|S_true|, eps)")

        im_cov = axes[1].imshow(
            coverage_map,
            origin="lower",
            aspect="auto",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[1].set_title("Frequency-Resolved 90% CI Coverage")
        axes[1].set_ylabel("Spectral Entry")
        axes[1].set_xlabel("Frequency")
        axes[1].set_yticks(y_ticks)
        axes[1].set_yticklabels([labels[idx] for idx in y_ticks], fontsize=8)
        axes[1].set_xticks(x_ticks)
        axes[1].set_xticklabels(x_labels, rotation=30, ha="right")
        cbar_cov = fig.colorbar(im_cov, ax=axes[1], pad=0.01)
        cbar_cov.set_label("Inside interval (0/1)")

        plt.tight_layout()
        return fig

    ok = _plot()
    logger.info(
        "Diagnostics plot: psd_truth_error_vs_freq.png "
        + f"{'ok' if ok else 'failed'}"
    )
    return bool(ok)


def plot_diagnostics(
    idata: xr.DataTree,
    outdir: str,
    p: Optional[int] = None,
    N: Optional[int] = None,
    runtime: Optional[float] = None,
    config: Optional[DiagnosticsConfig] = None,
    model=None,
    *,
    summary_mode: Literal["off", "light", "full"] = "light",
    summary_position: Literal["start", "end"] = "end",
    true_psd: Optional[np.ndarray] = None,
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
        logger.debug("Diagnostics step: summary text")
        generate_diagnostics_summary(idata, diag_dir, mode=summary_mode)
        logger.debug(
            f"Diagnostics step: summary text done in {time.perf_counter() - t_summary:.2f}s"
        )

    t_plots = time.perf_counter()
    logger.debug("Diagnostics step: plots")
    _create_diagnostic_plots(
        idata, diag_dir, config, p, N, runtime, model, true_psd
    )
    logger.debug(
        f"Diagnostics step: plots done in {time.perf_counter() - t_plots:.2f}s"
    )

    if summary_position == "end":
        t_summary = time.perf_counter()
        logger.debug("Diagnostics step: summary text")
        generate_diagnostics_summary(idata, diag_dir, mode=summary_mode)
        logger.debug(
            f"Diagnostics step: summary text done in {time.perf_counter() - t_summary:.2f}s"
        )

    logger.info(
        f"MCMC diagnostics finished in {time.perf_counter() - t0:.2f}s"
    )


def _create_diagnostic_plots(
    idata, diag_dir, config, p, N, runtime, model, true_psd
):
    """Create only the essential diagnostic plots."""
    logger.debug("Generating diagnostic plots...")

    idata_plot, weight_vars = _build_arviz_plot_data(idata, config)

    trace_idata, trace_total, trace_selected = _build_trace_plot_idata(
        idata_plot, config
    )
    if trace_total > trace_selected > 0:
        logger.info(
            f"Diagnostics trace plots: sampled {trace_selected}/{trace_total} parameters "
            + f"(seed={config.trace_plot_seed})"
        )

    # 1. ArviZ trace plots (lightweight subset)
    @safe_plot(f"{diag_dir}/trace_plots.png", config.dpi)
    def create_trace_plots():
        if trace_idata is None or not hasattr(trace_idata, "posterior"):
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
        if hasattr(trace_idata, "sample_stats"):
            sample_stats_vars = list(trace_idata.sample_stats.data_vars)
            has_divergences = "diverging" in sample_stats_vars
            divergences_arg = "diverging" if has_divergences else None

        # Calculate number of variables to scale figure size
        num_vars = len(trace_idata.posterior.data_vars)
        figsize_height = max(4, num_vars * 1.5)
        figsize = (14, figsize_height)

        # Create trace plot with improved layout
        # Use trace_idata which has the filtered posterior + sample_stats
        # ArviZ >= 1.0 removed combined/compact/figsize/divergences kwargs.
        # figure_kwargs passes figsize to the backend; visuals enables divergence markers.
        _trace_visuals = {"divergence": True} if has_divergences else None
        pc = plot_trace(
            trace_idata,
            figure_kwargs={"figsize": figsize},
            **({"visuals": _trace_visuals} if _trace_visuals else {}),
        )
        # ArviZ 1.0 returns a PlotCollection; earlier versions returned an axes array.
        if hasattr(pc, "viz") and "figure" in pc.viz:
            fig = pc.viz["figure"].item()
        elif hasattr(pc, "ravel"):
            fig = pc.ravel()[0].figure
        else:
            fig = plt.gcf()

        # Add Rhat values to subplot titles
        try:
            posterior = trace_idata.posterior
            for var_name in trace_idata.posterior.data_vars:
                var_data = posterior[var_name]
                rhat_result = rhat(var_data)
                rhat_array = _to_flat_finite_array(rhat_result)

                # Handle both scalar and array R-hats
                if rhat_array.size == 0:
                    continue
                rhat_val = float(np.mean(rhat_array))

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
    logger.debug("Diagnostics plot: trace_plots.png starting")
    ok = create_trace_plots()
    logger.debug(
        f"Diagnostics plot: trace_plots.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
    )

    # 2. Rank histogram diagnostics (subsampled variable list)
    t = time.perf_counter()
    logger.debug("Diagnostics plot: rank_plots starting")
    _create_rank_diagnostics(idata_plot, diag_dir, config)
    logger.debug(
        f"Diagnostics plots: rank_plots done in {time.perf_counter() - t:.2f}s"
    )

    # 3. Optional pair diagnostics for low-dimensional scalar variables
    t = time.perf_counter()
    logger.debug("Diagnostics plot: pair_plots starting")
    _create_pair_diagnostics(idata_plot, diag_dir, config)
    logger.debug(
        f"Diagnostics plots: pair_plots done in {time.perf_counter() - t:.2f}s"
    )

    if config.save_ess_rhat_profiles:
        t = time.perf_counter()
        logger.debug("Diagnostics plot: ess_rhat_profiles starting")
        _create_ess_rhat_profiles(idata, diag_dir, config, weight_vars)
        logger.debug(
            f"Diagnostics plots: ess_rhat_profiles done in {time.perf_counter() - t:.2f}s"
        )

    # 4. Summary dashboard with key convergence metrics
    @safe_plot(f"{diag_dir}/summary_dashboard.png", config.dpi)
    def plot_summary():
        _plot_summary_dashboard(idata, config, p, N, runtime)

    t = time.perf_counter()
    logger.debug("Diagnostics plot: summary_dashboard.png starting")
    ok = plot_summary()
    logger.debug(
        f"Diagnostics plot: summary_dashboard.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
    )

    # 5. Truth-aware PSD error diagnostics (multivariate only)
    t = time.perf_counter()
    logger.debug("Diagnostics plot: psd_truth_error_vs_freq.png starting")
    _create_truth_psd_frequency_diagnostics(
        idata, diag_dir, config, true_psd=true_psd
    )
    logger.debug(
        "Diagnostics plot: psd_truth_error_vs_freq.png done in "
        + f"{time.perf_counter() - t:.2f}s"
    )

    # 6. Acceptance rate diagnostics
    if config.save_acceptance:

        @safe_plot(f"{diag_dir}/acceptance_diagnostics.png", config.dpi)
        def plot_acceptance():
            _plot_acceptance_diagnostics_blockaware(idata, config)

        t = time.perf_counter()
        logger.debug("Diagnostics plot: acceptance_diagnostics.png starting")
        ok = plot_acceptance()
        logger.debug(
            f"Diagnostics plot: acceptance_diagnostics.png {'ok' if ok else 'failed'} in {time.perf_counter() - t:.2f}s"
        )
    else:
        logger.debug(
            "Diagnostics plot: acceptance_diagnostics skipped (disabled)."
        )

    # 7. Sampler-specific diagnostics
    t = time.perf_counter()
    logger.debug("Diagnostics plots: sampler-specific starting")
    _create_sampler_diagnostics(idata, diag_dir, config)
    logger.debug(
        f"Diagnostics plots: sampler-specific done in {time.perf_counter() - t:.2f}s"
    )

    # 8. Composite sampling_diagnostics.png from individual sampling plots
    t = time.perf_counter()
    _sampling_diag_sources = [
        f"{diag_dir}/summary_dashboard.png",
        f"{diag_dir}/ess_rhat_profiles.png",
        f"{diag_dir}/nuts_diagnostics.png",
        f"{diag_dir}/nuts_block_diagnostics.png",
    ]
    composite_images_vertical(
        _sampling_diag_sources,
        outfile=f"{diag_dir}/sampling_diagnostics.png",
        dpi=config.dpi,
        title="Sampling Diagnostics",
    )
    logger.debug(
        f"Diagnostics plot: sampling_diagnostics.png done in {time.perf_counter() - t:.2f}s"
    )


def _create_rank_diagnostics(idata, diag_dir, config):
    if not config.save_rank_plots:
        logger.debug("Diagnostics plot: rank_plots skipped (disabled).")
        return
    if idata is None:
        logger.debug("Diagnostics plot: rank_plots skipped (no data).")
        return

    rank_vars = _select_rank_plot_vars(idata, config)
    if not rank_vars:
        logger.debug(
            "Diagnostics plot: rank_plots skipped (no low-dimensional variables)."
        )
        return

    @safe_plot(f"{diag_dir}/rank_plots.png", config.dpi)
    def _plot_rank():
        plot_rank(idata, var_names=rank_vars)

    ok = _plot_rank()
    logger.debug(
        "Diagnostics plot: rank_plots.png "
        + f"{'ok' if ok else 'failed'} for {len(rank_vars)} vars"
    )


def _create_pair_diagnostics(idata, diag_dir, config):
    if not config.save_pair_plots:
        logger.debug("Diagnostics plot: pair_plots skipped (disabled).")
        return
    if idata is None:
        logger.debug("Diagnostics plot: pair_plots skipped (no data).")
        return

    pair_vars = _select_pair_plot_vars(idata, config)
    if len(pair_vars) < 2:
        logger.debug(
            "Diagnostics plot: pair_plots skipped (need >=2 scalar variables)."
        )
        return

    has_divergences = hasattr(idata, "sample_stats") and any(
        str(key).startswith("diverging") for key in idata.sample_stats
    )

    @safe_plot(f"{diag_dir}/pair_plot.png", config.dpi)
    def _plot_pair():
        pair_kwargs = dict(var_names=pair_vars, marginal=True)
        if has_divergences:
            pair_kwargs["aes_by_visuals"] = {"divergence": ["color"]}
        plot_pair(idata, **pair_kwargs)

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


def _plot_single_nuts_block_on_axes(
    idata, config, channel_idx: int, axes, *, title_prefix: str
):
    """Render NUTS diagnostics for one blocked channel onto 4 provided axes."""

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

    # Step size trace
    ax = axes[0]
    if step_size is not None:
        ax.plot(step_size, alpha=0.7, lw=1, label="step_size")
    else:
        ax.text(
            0.5, 0.5, "Step size data\nunavailable", ha="center", va="center"
        )
    ax.set_title(f"{title_prefix} Step Size")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("step_size")
    ax.grid(True, alpha=0.3)
    if step_size is not None:
        ax.legend(loc="best", fontsize="small")

    # Acceptance trace
    ax = axes[1]
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
    ax.set_title(f"{title_prefix} Acceptance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("accept_prob")

    # Steps histogram
    ax = axes[2]
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
    ax.set_title(f"{title_prefix} Leapfrog Steps")

    # Summary stats
    ax = axes[3]
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


def _plot_single_nuts_block(idata, config, channel_idx: int):
    """NUTS diagnostics for a single blocked channel."""
    fig, axes = plt.subplots(2, 2, figsize=config.figsize)
    _plot_single_nuts_block_on_axes(
        idata,
        config,
        channel_idx,
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        title_prefix=f"Channel {channel_idx}",
    )
    plt.tight_layout()


def _plot_nuts_blocks_grid(idata, config, channel_indices: list[int]):
    """NUTS diagnostics with one row per blocked channel."""
    n_channels = len(channel_indices)
    fig, axes = plt.subplots(
        n_channels,
        4,
        figsize=(18, max(3.0 * n_channels, 4.5)),
        squeeze=False,
    )
    for row, channel_idx in enumerate(channel_indices):
        _plot_single_nuts_block_on_axes(
            idata,
            config,
            channel_idx,
            [axes[row, 0], axes[row, 1], axes[row, 2], axes[row, 3]],
            title_prefix=f"Channel {channel_idx}",
        )
    fig.tight_layout()


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

        sorted_channel_indices = sorted(channel_indices)

        if sorted_channel_indices:

            @safe_plot(f"{diag_dir}/nuts_block_diagnostics.png", config.dpi)
            def plot_nuts_blocks_combined():
                _plot_nuts_blocks_grid(idata, config, sorted_channel_indices)

            plot_nuts_blocks_combined()

        if not config.save_nuts_block_diagnostics_individual:
            return

        for channel_idx in sorted_channel_indices:

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

    def _pick_scalar(*values) -> Optional[float]:
        for value in values:
            fval = _as_float(value)
            if fval is not None:
                return fval
        return None

    def _attrs_like(obj) -> dict:
        if obj is None:
            return {}
        merged: dict = {}
        obj_attrs = getattr(obj, "attrs", None)
        if hasattr(obj_attrs, "items"):
            merged.update(dict(obj_attrs))
        ds = getattr(obj, "ds", None)
        ds_attrs = getattr(ds, "attrs", None)
        if hasattr(ds_attrs, "items"):
            merged.update(dict(ds_attrs))
        dataset = getattr(obj, "dataset", None)
        dataset_attrs = getattr(dataset, "attrs", None)
        if hasattr(dataset_attrs, "items"):
            merged.update(dict(dataset_attrs))
        return merged

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
        if ess_min is None and ess_tail_min is None and rhat_max is None:
            try:
                ess_ds = ess(idata, method="bulk")
                ess_vals = np.concatenate(
                    [
                        np.asarray(ess_ds[v]).reshape(-1)
                        for v in ess_ds.data_vars
                    ]
                )
                ess_vals = ess_vals[np.isfinite(ess_vals)]
                if ess_vals.size:
                    ess_min = float(np.min(ess_vals))
                    ess_med = float(np.median(ess_vals))
            except Exception:
                pass

            try:
                ess_tail_ds = ess(idata, method="tail")
                ess_tail_vals = np.concatenate(
                    [
                        np.asarray(ess_tail_ds[v]).reshape(-1)
                        for v in ess_tail_ds.data_vars
                    ]
                )
                ess_tail_vals = ess_tail_vals[np.isfinite(ess_tail_vals)]
                if ess_tail_vals.size:
                    ess_tail_min = float(np.min(ess_tail_vals))
                    ess_tail_med = float(np.median(ess_tail_vals))
            except Exception:
                pass

            try:
                rhat_ds = rhat(idata)
                rhat_vals = np.concatenate(
                    [
                        np.asarray(rhat_ds[v]).reshape(-1)
                        for v in rhat_ds.data_vars
                    ]
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
        "l2_matrix",
        "coverage",
        "ci_width",
        "ci_width_diag_mean",
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

        l2_matrix_value = _as_float(psd_diag.get("l2_matrix"))
        if l2_matrix_value is not None and np.isfinite(l2_matrix_value):
            psd_lines.append(f"  L2 (matrix): {l2_matrix_value:.3f}")

        coverage_value = _as_float(psd_diag.get("coverage"))
        if coverage_value is not None and np.isfinite(coverage_value):
            psd_lines.append(f"  Coverage: {coverage_value*100:.1f}%")

        ci_width_value = _pick_scalar(
            psd_diag.get("ci_width"),
            psd_diag.get("ci_width_diag_mean"),
        )
        if ci_width_value is not None:
            psd_lines.append(f"  CI width: {ci_width_value:.3g}")

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

    vi_metrics = {}
    have_vi_psd = (
        bool(getattr(idata, "attrs", {}).get("only_vi"))
        or hasattr(idata, "vi_posterior")
        or hasattr(idata, "vi_sample_stats")
    )
    if have_vi_psd:
        vi_psd_attrs = {}
        if hasattr(idata, "vi_sample_stats"):
            vi_psd_attrs = _attrs_like(getattr(idata, "vi_sample_stats", None))
        vi_metrics["riae"] = _pick_scalar(
            attrs.get("vi_riae_vs_truth"),
            attrs.get("vi_riae"),
            vi_psd_attrs.get("riae"),
            vi_psd_attrs.get("riae_matrix"),
        )
        vi_metrics["coverage"] = _pick_scalar(
            attrs.get("vi_coverage_vs_truth"),
            attrs.get("vi_coverage"),
            vi_psd_attrs.get("coverage"),
            vi_psd_attrs.get("ci_coverage"),
        )
        vi_metrics["ci_width"] = _pick_scalar(
            attrs.get("vi_ci_width_vs_truth"),
            attrs.get("vi_ci_width"),
            attrs.get("vi_ci_width_diag_mean"),
        )
        if not any(value is not None for value in vi_metrics.values()):
            vi_psd_metrics = _run_psd_compare(
                idata=None,
                idata_vi=idata,
                truth=attrs.get("true_psd"),
                psd_ref=attrs.get("true_psd"),
            )
            vi_metrics["riae"] = _pick_scalar(
                vi_psd_metrics.get("riae"),
                vi_psd_metrics.get("riae_matrix"),
            )
            vi_metrics["coverage"] = _pick_scalar(
                vi_psd_metrics.get("coverage")
            )
            vi_metrics["ci_width"] = _pick_scalar(
                vi_psd_metrics.get("ci_width"),
                vi_psd_metrics.get("ci_width_diag_mean"),
            )

    nuts_metrics = {
        "riae": _pick_scalar(attrs.get("riae"), attrs.get("riae_matrix")),
        "coverage": _pick_scalar(attrs.get("coverage")),
        "ci_width": _pick_scalar(
            attrs.get("ci_width"),
            attrs.get("ci_width_diag_mean"),
            attrs.get("psd_compare_ci_width"),
            attrs.get("psd_compare_ci_width_diag_mean"),
        ),
    }

    if (
        have_vi_psd
        and any(value is not None for value in nuts_metrics.values())
        and any(value is not None for value in vi_metrics.values())
    ):
        summary.append("\nVI vs NUTS PSD accuracy:")
        vi_riae = vi_metrics.get("riae")
        nuts_riae = nuts_metrics.get("riae")
        if vi_riae is not None or nuts_riae is not None:
            if vi_riae is not None and nuts_riae is not None:
                summary.append(
                    f"  RIAE: VI={vi_riae:.3f} | NUTS={nuts_riae:.3f}"
                )
            else:
                summary.append("  RIAE: unavailable")

        vi_cov = vi_metrics.get("coverage")
        nuts_cov = nuts_metrics.get("coverage")
        if vi_cov is not None or nuts_cov is not None:
            if vi_cov is not None and nuts_cov is not None:
                summary.append(
                    f"  Coverage: VI={vi_cov*100:.1f}% | NUTS={nuts_cov*100:.1f}%"
                )
            else:
                summary.append("  Coverage: unavailable")

        vi_ci_width = vi_metrics.get("ci_width")
        nuts_ci_width = nuts_metrics.get("ci_width")
        if vi_ci_width is not None or nuts_ci_width is not None:
            if vi_ci_width is not None and nuts_ci_width is not None:
                summary.append(
                    f"  CI width: VI={vi_ci_width:.3g} | NUTS={nuts_ci_width:.3g}"
                )
            elif vi_ci_width is not None:
                summary.append(
                    f"  CI width: VI={vi_ci_width:.3g} | NUTS=unavailable"
                )
            else:
                summary.append(
                    f"  CI width: VI=unavailable | NUTS={nuts_ci_width:.3g}"
                )

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
            # Append VI section to the shared diagnostics_summary.txt
            diag_summary_path = os.path.join(outdir, "diagnostics_summary.txt")
            with open(diag_summary_path, "a") as f:
                f.write("\n\n" + summary_text)
            vi_summary_path = os.path.join(
                outdir, "vi_diagnostics_summary.txt"
            )
            with open(vi_summary_path, "w") as f:
                f.write(summary_text)
        except Exception:
            logger.debug(
                "Could not append VI diagnostics summary to diagnostics_summary.txt.",
                exc_info=True,
            )

    if log:
        logger.info(f"\n{summary_text}\n")
    return summary_text
