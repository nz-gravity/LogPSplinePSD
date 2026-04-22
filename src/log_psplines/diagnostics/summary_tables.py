"""Minimal per-factor diagnostics summary tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import arviz_stats as azs
import numpy as np
import pandas as pd
import xarray as xr

from ..arviz_utils._datatree import require_dataset as _require_dataset
from ..arviz_utils.from_arviz import get_psd_dataset
from ._factors import factor_idatas, vi_factor_idatas
from ._utils import (
    compute_ci_coverage_multivar,
    compute_ci_coverage_univar,
    compute_matrix_l2,
    compute_matrix_riae,
    compute_riae,
    interior_frequency_slice,
)


def _is_arviz_loo_source(source: Any) -> bool:
    return hasattr(source, "posterior") and hasattr(source, "log_likelihood")


def _resolve_truth(
    source: xr.DataTree | Mapping[str, Any], true_psd: Any = None
) -> np.ndarray | None:
    if true_psd is not None:
        return np.asarray(true_psd)
    if isinstance(source, xr.DataTree):
        attrs = getattr(source, "attrs", {}) or {}
        value = attrs.get("true_psd")
        return None if value is None else np.asarray(value)
    value = source.get("true_psd")
    return None if value is None else np.asarray(value)


def _compute_univar_l2(
    estimate: np.ndarray, truth: np.ndarray, freqs: np.ndarray
) -> float:
    numerator = float(
        np.sqrt(max(np.trapezoid((estimate - truth) ** 2, freqs), 0.0))
    )
    denominator = float(np.sqrt(max(np.trapezoid(truth**2, freqs), 0.0)))
    return numerator / denominator if denominator != 0.0 else float("nan")


def _truth_metrics_from_idata(
    idata: xr.DataTree, true_psd: Any = None
) -> dict[str, float]:
    truth = _resolve_truth(idata, true_psd)
    if truth is None:
        return {}

    psd_ds = get_psd_dataset(idata, source="best")
    freqs = np.asarray(psd_ds.coords["frequency"].values, dtype=float)
    freq_idx = interior_frequency_slice(freqs.size)
    freqs = freqs[freq_idx]
    spectral_density = np.asarray(psd_ds["spectral_density"].values)

    n_channels = int(spectral_density.shape[2])
    if n_channels == 1:
        samples = np.real(spectral_density[:, :, 0, 0, :]).reshape(
            -1, freqs.size
        )
        samples = samples[:, freq_idx]
        truth_arr = np.asarray(truth, dtype=float).reshape(-1)[freq_idx]
        q05, q50, q95 = np.percentile(samples, [5.0, 50.0, 95.0], axis=0)
        return {
            "riae": float(compute_riae(q50, truth_arr, freqs)),
            "l2": float(_compute_univar_l2(q50, truth_arr, freqs)),
            "coverage": float(
                compute_ci_coverage_univar(
                    np.stack([q05, q50, q95], axis=0), truth_arr
                )
            ),
        }

    samples = spectral_density.reshape(
        -1, n_channels, n_channels, spectral_density.shape[-1]
    )
    samples = np.moveaxis(samples[..., freq_idx], -1, 1)
    q05, q50, q95 = np.percentile(samples.real, [5.0, 50.0, 95.0], axis=0)
    truth_arr = np.asarray(truth)[freq_idx]
    return {
        "riae": float(compute_matrix_riae(q50, truth_arr, freqs)),
        "l2": float(compute_matrix_l2(q50, truth_arr, freqs)),
        "coverage": float(
            compute_ci_coverage_multivar(
                np.stack([q05, q50, q95], axis=0), truth_arr
            )
        ),
    }


def _truth_metrics_from_mapping(source: Mapping[str, Any]) -> dict[str, float]:
    metrics = {}
    for src_key, out_key in (
        ("riae", "riae"),
        ("riae_matrix", "riae"),
        ("l2", "l2"),
        ("l2_matrix", "l2"),
        ("coverage", "coverage"),
        ("ci_coverage", "coverage"),
    ):
        value = source.get(src_key)
        if value is None:
            continue
        metrics[out_key] = float(value)
    return metrics


def _shared_truth_metrics(
    source: xr.DataTree | Mapping[str, Any] | Sequence[Any],
    true_psd: Any = None,
) -> dict[str, float]:
    if isinstance(source, xr.DataTree):
        return _truth_metrics_from_idata(source, true_psd=true_psd)
    if isinstance(source, Mapping):
        return _truth_metrics_from_mapping(source)
    return {}


def _sample_stats_array(idata: xr.DataTree, name: str) -> np.ndarray:
    try:
        sample_stats = _require_dataset(idata, "sample_stats")
    except (KeyError, TypeError):
        return np.array([], dtype=float)
    if name not in sample_stats:
        return np.array([], dtype=float)
    return np.asarray(sample_stats[name].values, dtype=float).reshape(-1)


def _tree_depth_hits(idata: xr.DataTree) -> int:
    attrs = getattr(idata, "attrs", {}) or {}
    max_tree_depth = attrs.get("max_tree_depth")
    if max_tree_depth is None:
        return 0

    max_tree_depth = int(max_tree_depth)
    tree_depth = _sample_stats_array(idata, "tree_depth")
    tree_depth = tree_depth[np.isfinite(tree_depth)]
    if tree_depth.size:
        return int(np.sum(tree_depth >= max_tree_depth))

    num_steps = _sample_stats_array(idata, "num_steps")
    num_steps = num_steps[np.isfinite(num_steps)]
    if num_steps.size == 0:
        return 0
    return int(np.sum(num_steps >= 2**max_tree_depth))


def _step_size(idata: xr.DataTree) -> float:
    step_size = _sample_stats_array(idata, "step_size")
    step_size = step_size[np.isfinite(step_size)]
    return float(np.median(step_size)) if step_size.size else np.nan


def _n_draws(idata: xr.DataTree, *, group: str = "posterior") -> int:
    try:
        dataset = _require_dataset(idata, group)
    except (KeyError, TypeError):
        return 0
    n_chains = int(dataset.sizes.get("chain", 0))
    n_draws = int(dataset.sizes.get("draw", 0))
    return n_chains * n_draws


def build_nuts_summary_table(
    idata_or_factors: (
        xr.DataTree | Mapping[str, xr.DataTree] | Sequence[xr.DataTree]
    ),
    *,
    true_psd: Any = None,
) -> pd.DataFrame:
    """Return one NUTS diagnostics row per factor."""

    rows: list[dict[str, Any]] = []
    shared_truth_metrics = (
        _shared_truth_metrics(idata_or_factors, true_psd=true_psd)
        if isinstance(idata_or_factors, xr.DataTree)
        else {}
    )

    for factor, idata in factor_idatas(idata_or_factors).items():
        summary = azs.summary(idata)
        row = {
            "factor": factor,
            "divergences": int(
                np.sum(_sample_stats_array(idata, "diverging") > 0.0)
            ),
            "max_treedepth_hits": _tree_depth_hits(idata),
            "step_size": _step_size(idata),
            "rhat_max": (
                float(summary["r_hat"].max()) if "r_hat" in summary else np.nan
            ),
            "ess_bulk_min": (
                float(summary["ess_bulk"].min())
                if "ess_bulk" in summary
                else np.nan
            ),
            "ess_tail_min": (
                float(summary["ess_tail"].min())
                if "ess_tail" in summary
                else np.nan
            ),
            "n_draws": _n_draws(idata),
        }
        row.update(shared_truth_metrics)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)


def _extract_losses(source: xr.DataTree | Mapping[str, Any]) -> np.ndarray:
    if isinstance(source, xr.DataTree):
        try:
            vi_sample_stats = _require_dataset(source, "vi_sample_stats")
        except (KeyError, TypeError):
            vi_sample_stats = None
        if vi_sample_stats is None or "losses" not in vi_sample_stats:
            return np.array([], dtype=float)
        return np.asarray(
            vi_sample_stats["losses"].values, dtype=float
        ).reshape(-1)

    losses = source.get("losses")
    if losses is None:
        return np.array([], dtype=float)
    return np.asarray(losses, dtype=float).reshape(-1)


def _extract_pareto_k_from_data(
    source: Any,
) -> tuple[np.ndarray, bool]:
    if _is_arviz_loo_source(source):
        loo_result = azs.loo(source, pointwise=True)
        pareto_k = np.asarray(loo_result.pareto_k.values, dtype=float).reshape(
            -1
        )
        return pareto_k, bool(getattr(loo_result, "warning", False))

    if "pareto_k" in source and source["pareto_k"] is not None:
        pareto_k = np.asarray(source["pareto_k"], dtype=float).reshape(-1)
        warning = bool(np.any(np.isfinite(pareto_k) & (pareto_k > 0.7)))
        return pareto_k, warning

    return np.array([], dtype=float), False


def _split_vi_inputs(
    vi_or_factors: (
        xr.DataTree | Mapping[str, Any] | Sequence[Any] | Mapping[str, Any]
    ),
) -> dict[str, Any]:
    if isinstance(vi_or_factors, xr.DataTree):
        return vi_factor_idatas(vi_or_factors)

    if isinstance(vi_or_factors, Sequence):
        return {str(idx): value for idx, value in enumerate(vi_or_factors)}

    if not isinstance(vi_or_factors, Mapping):
        raise TypeError(
            "Expected a DataTree, factor mapping, sequence, or diagnostics mapping."
        )

    if any(
        isinstance(value, (xr.DataTree, Mapping))
        or _is_arviz_loo_source(value)
        for value in vi_or_factors.values()
    ):
        sample_value = next(iter(vi_or_factors.values()), None)
        if isinstance(
            sample_value, (xr.DataTree, Mapping)
        ) or _is_arviz_loo_source(sample_value):
            return {str(key): value for key, value in vi_or_factors.items()}

    losses_per_block = vi_or_factors.get("losses_per_block")
    pareto_k_per_block = vi_or_factors.get("pareto_k_per_block")
    if losses_per_block is None and pareto_k_per_block is None:
        return {"0": vi_or_factors}

    block_count = 0
    if losses_per_block is not None:
        block_count = max(
            block_count, int(np.asarray(losses_per_block).shape[0])
        )
    if pareto_k_per_block is not None:
        block_count = max(
            block_count,
            int(np.asarray(pareto_k_per_block).reshape(-1).shape[0]),
        )

    split: dict[str, Mapping[str, Any]] = {}
    shared_items = {
        str(key): value
        for key, value in vi_or_factors.items()
        if key not in {"losses_per_block", "pareto_k_per_block"}
    }
    for idx in range(block_count):
        item: dict[str, Any] = dict(shared_items)
        if losses_per_block is not None:
            item["losses"] = np.asarray(losses_per_block)[idx]
        if pareto_k_per_block is not None:
            item["pareto_k"] = np.asarray(pareto_k_per_block).reshape(-1)[
                idx : idx + 1
            ]
        split[str(idx)] = item
    return split


def build_vi_summary_table(
    vi_or_factors: (
        xr.DataTree | Mapping[str, Any] | Sequence[Any] | Mapping[str, Any]
    ),
    *,
    elbo_window: int = 50,
    true_psd: Any = None,
) -> pd.DataFrame:
    """Return one VI diagnostics row per factor."""

    rows: list[dict[str, Any]] = []
    split_inputs = _split_vi_inputs(vi_or_factors)
    shared_truth_metrics = _shared_truth_metrics(
        vi_or_factors, true_psd=true_psd
    )

    for factor, source in split_inputs.items():
        losses = _extract_losses(source)
        final_elbo = float(losses[-1]) if losses.size else np.nan
        window = min(max(int(elbo_window), 1), int(losses.size))
        if window > 1:
            elbo_improvement = float(losses[-window] - losses[-1])
        else:
            elbo_improvement = np.nan

        pareto_k, loo_warning = _extract_pareto_k_from_data(source)
        pareto_k = pareto_k[np.isfinite(pareto_k)]

        row = {
            "factor": factor,
            "final_elbo": final_elbo,
            "elbo_improvement_last_window": elbo_improvement,
            "pareto_k_max": (
                float(np.max(pareto_k)) if pareto_k.size else np.nan
            ),
            "pareto_k_median": (
                float(np.median(pareto_k)) if pareto_k.size else np.nan
            ),
            "loo_warning": bool(loo_warning),
            "n_draws": (
                _n_draws(source, group="vi_posterior")
                if isinstance(source, xr.DataTree)
                else (
                    int(source.posterior.sizes.get("chain", 0))
                    * int(source.posterior.sizes.get("draw", 0))
                    if _is_arviz_loo_source(source)
                    else 0
                )
            ),
        }
        row.update(shared_truth_metrics)
        row.update(
            _truth_metrics_from_mapping(source)
            if isinstance(source, Mapping)
            else {}
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)
