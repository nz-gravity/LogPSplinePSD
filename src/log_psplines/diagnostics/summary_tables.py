"""Per-channel diagnostics for native fitted results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz_stats as azs
import numpy as np
import pandas as pd
import xarray as xr
from arviz_base import from_dict

from log_psplines.diagnostics._utils import (
    compute_ci_coverage_multivar,
    compute_matrix_l2,
    compute_matrix_riae,
    interior_frequency_slice,
)

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def _truth_metrics_from_result(
    result: PSDResult, true_psd: Any = None
) -> dict[str, float]:
    truth = None if true_psd is None else np.asarray(true_psd)
    if truth is None:
        value = result.metadata.get("true_psd")
        truth = None if value is None else np.asarray(value)
    if truth is None or result.time is not None:
        return {}

    freqs_raw = result.frequency
    freq_idx = interior_frequency_slice(freqs_raw.size)
    freqs = freqs_raw[freq_idx]
    samples = np.asarray(result.spectral_density)
    samples = samples.reshape(-1, *samples.shape[2:])[:, freq_idx]
    q05, q50, q95 = np.percentile(samples.real, [5.0, 50.0, 95.0], axis=0)
    truth_arr = np.asarray(truth)
    if truth_arr.ndim == 1:
        truth_arr = truth_arr[:, None, None]
    truth_arr = truth_arr[freq_idx]
    return {
        "riae": float(compute_matrix_riae(q50, truth_arr, freqs)),
        "l2": float(compute_matrix_l2(q50, truth_arr, freqs)),
        "coverage": float(
            compute_ci_coverage_multivar(
                np.stack([q05, q50, q95], axis=0), truth_arr
            )
        ),
    }


def _channel_posterior(result: PSDResult, channel: int) -> xr.Dataset:
    """Select the current blocked Cholesky latent variables for one channel."""
    scalar_names = {
        f"delta_{channel}",
        f"phi_delta_{channel}",
        f"weights_delta_{channel}",
    }
    theta_prefixes = tuple(
        f"{prefix}_{channel}_"
        for prefix in (
            "delta_theta_re",
            "phi_theta_re",
            "weights_theta_re",
            "delta_theta_im",
            "phi_theta_im",
            "weights_theta_im",
        )
    )
    return result.posterior[
        [
            name
            for name in result.posterior.data_vars
            if name in scalar_names or name.startswith(theta_prefixes)
        ]
    ]


def channel_idata(result: PSDResult, channel: int) -> xr.DataTree:
    """Build the ArviZ view for one blocked stationary NUTS channel."""
    posterior = _channel_posterior(result, channel)
    if not posterior.data_vars:
        raise ValueError(f"No posterior variables for channel {channel}")
    groups: dict[str, xr.Dataset] = {"posterior": posterior}
    if result.sample_stats is not None:
        suffix = f"_channel_{channel}"
        stats = {
            name.removesuffix(suffix): var
            for name, var in result.sample_stats.data_vars.items()
            if name.endswith(suffix)
        }
        if stats:
            groups["sample_stats"] = xr.Dataset(stats)
    if result.log_likelihood is not None:
        name = f"log_likelihood_block_{channel}"
        if name in result.log_likelihood:
            groups["log_likelihood"] = xr.Dataset(
                {"log_likelihood": result.log_likelihood[name]}
            )
    attrs = dict(result.metadata)
    depths = attrs.get("max_tree_depth_by_channel")
    if depths is not None:
        attrs["max_tree_depth"] = int(depths[channel])
    idata = from_dict(groups)
    idata.attrs.update(attrs)
    return idata


def _sample_stats_array(idata: xr.DataTree, name: str) -> np.ndarray:
    if "sample_stats" not in idata.children:
        return np.array([], dtype=float)
    dataset = idata["sample_stats"].dataset
    if dataset is None or name not in dataset:
        return np.array([], dtype=float)
    return np.asarray(dataset[name].values, dtype=float).reshape(-1)


def _tree_depth_hits(idata: xr.DataTree) -> int:
    max_tree_depth = idata.attrs.get("max_tree_depth")
    if max_tree_depth is None:
        return 0
    max_tree_depth = int(max_tree_depth)
    tree_depth = _sample_stats_array(idata, "tree_depth")
    tree_depth = tree_depth[np.isfinite(tree_depth)]
    if tree_depth.size:
        return int(np.sum(tree_depth >= max_tree_depth))
    n_steps = _sample_stats_array(idata, "n_steps")
    n_steps = n_steps[np.isfinite(n_steps)]
    return int(np.sum(n_steps >= 2**max_tree_depth))


def _summary_reduction(
    summary: pd.DataFrame, column: str, reducer: str
) -> float:
    if column not in summary:
        return np.nan
    values = pd.to_numeric(summary[column], errors="coerce").to_numpy(
        dtype=float
    )
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.nan
    if reducer == "max":
        return float(np.max(finite))
    if reducer == "min":
        return float(np.min(finite))
    raise ValueError(f"Unsupported reducer: {reducer}")


def build_nuts_summary_table(
    result: PSDResult, *, true_psd: Any = None
) -> pd.DataFrame:
    """Return one NUTS diagnostics row per blocked channel."""
    if result.sample_stats is None:
        raise ValueError("NUTS diagnostics require sample_stats")
    rows: list[dict[str, Any]] = []
    truth = _truth_metrics_from_result(result, true_psd)
    n_channels = (
        1 if result.time is not None else int(result.spectrum.sizes["channel"])
    )
    for channel in range(n_channels):
        # Scalar power fits have one unsuffixed NUTS trajectory. Stationary
        # fits use one trajectory per blocked Cholesky channel.
        idata = (
            result.to_arviz()
            if result.time is not None
            else channel_idata(result, channel)
        )
        summary = azs.summary(idata)
        step_size = _sample_stats_array(idata, "step_size")
        step_size = step_size[np.isfinite(step_size)]
        row = {
            "factor": str(channel),
            "divergences": int(
                np.sum(_sample_stats_array(idata, "diverging") > 0)
            ),
            "max_treedepth_hits": _tree_depth_hits(idata),
            "step_size": float(np.median(step_size))
            if step_size.size
            else np.nan,
            "rhat_max": _summary_reduction(summary, "r_hat", "max"),
            "ess_bulk_min": _summary_reduction(summary, "ess_bulk", "min"),
            "ess_tail_min": _summary_reduction(summary, "ess_tail", "min"),
            "n_draws": int(result.posterior.sizes.get("chain", 0))
            * int(result.posterior.sizes.get("draw", 0)),
            "riae": np.nan,
            "l2": np.nan,
            "coverage": np.nan,
        }
        row.update(truth)
        rows.append(row)
    return pd.DataFrame(rows)


def build_vi_summary_table(
    result: PSDResult, *, elbo_window: int = 50, true_psd: Any = None
) -> pd.DataFrame:
    """Return ELBO diagnostics from the fitted VI result."""
    if result.vi is None:
        raise ValueError("VI diagnostics require result.vi")
    traces = result.vi.losses_per_block
    if traces is None:
        traces = [result.vi.losses]
    truth = _truth_metrics_from_result(result, true_psd)
    rows: list[dict[str, Any]] = []
    for index, trace in enumerate(traces):
        losses = np.asarray(trace, dtype=float).reshape(-1)
        window = min(max(int(elbo_window), 1), losses.size)
        row = {
            "factor": str(index),
            "final_elbo": float(losses[-1]) if losses.size else np.nan,
            "elbo_improvement_last_window": (
                float(losses[-window] - losses[-1]) if window > 1 else np.nan
            ),
            "n_draws": int(result.posterior.sizes.get("chain", 0))
            * int(result.posterior.sizes.get("draw", 0)),
        }
        row.update(truth)
        rows.append(row)
    return pd.DataFrame(rows)
