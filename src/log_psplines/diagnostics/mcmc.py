"""MCMC diagnostics: ESS, Rhat, divergences, PSIS."""

from __future__ import annotations

import re
from typing import Callable, Dict

import numpy as np
import xarray as xr
from arviz_stats.base import array_stats

from ..logger import logger
from ._utils import khat_status

DEFAULT_MCMC_DIAG_MAX_ELEMENTS = 250_000
DEFAULT_MCMC_DIAG_WEIGHT_POINTS = 6
DEFAULT_MCMC_DIAG_MAX_DRAWS = 500


def _require_dataset(idata: xr.DataTree, group: str) -> xr.Dataset:
    dataset = idata[group].dataset
    if dataset is None:
        raise TypeError(f"DataTree group '{group}' must contain a dataset.")
    return dataset


def _has_group(idata: xr.DataTree, group: str) -> bool:
    return group in idata.children


def _select_draw_slice(idata: xr.DataTree, draw_slice: slice) -> xr.DataTree:
    out = xr.DataTree()
    out.attrs.update(dict(idata.attrs))
    for name, group in idata.children.items():
        dataset = group.dataset
        if dataset is None:
            out[name] = group
            continue
        if "draw" in dataset.dims:
            dataset = dataset.sel(draw=draw_slice)
        out[name] = xr.DataTree(dataset=dataset)
    return out


def _metric_from_attr_or_compute(
    idata,
    *,
    attr_key: str,
    attr_metrics: Callable[[np.ndarray], Dict[str, float]],
    compute_fn: Callable[[], np.ndarray],
    compute_metrics: Callable[[np.ndarray], Dict[str, float]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    attrs = idata.attrs or {}

    attr_values = attrs.get(attr_key)
    if attr_values is not None:
        vals = np.asarray(attr_values).reshape(-1)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            return attr_metrics(finite)

    try:
        computed = np.asarray(compute_fn()).reshape(-1)
    except Exception:
        return metrics

    finite = computed[np.isfinite(computed)]
    if finite.size:
        metrics.update(compute_metrics(finite))
    return metrics


def _posterior_from_idata(idata: xr.DataTree) -> xr.Dataset:
    return _require_dataset(idata, "posterior")


def _safe_rhat_values(posterior) -> np.ndarray:
    values = []
    for var in posterior.data_vars.values():
        arr = np.asarray(var)
        if arr.ndim < 2 or arr.shape[0] < 2:
            continue
        try:
            arr = np.asarray(array_stats.rhat(arr, chain_axis=0, draw_axis=1))
        except Exception:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr.ravel())
    if not values:
        return np.array([])
    return np.concatenate(values)


def _safe_ess_values(posterior, *, method: str) -> np.ndarray:
    values = []
    for var in posterior.data_vars.values():
        arr = np.asarray(var)
        if arr.ndim < 2:
            continue
        try:
            arr = np.asarray(
                array_stats.ess(
                    arr,
                    method=method,
                    chain_axis=0,
                    draw_axis=1,
                )
            )
        except Exception:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr.ravel())
    if not values:
        return np.array([])
    return np.concatenate(values)


def _compute_rhat(idata, *, skip_compute: bool = False) -> Dict[str, float]:
    attrs = idata.attrs or {}
    if attrs.get("rhat") is not None:
        return _metric_from_attr_or_compute(
            idata,
            attr_key="rhat",
            attr_metrics=lambda vals: {
                "rhat_max": float(np.max(vals)),
                "rhat_mean": float(np.mean(vals)),
            },
            compute_fn=lambda: _safe_rhat_values(_posterior_from_idata(idata)),
            compute_metrics=lambda vals: {
                "rhat_max": float(np.max(vals)),
                "rhat_mean": float(np.mean(vals)),
            },
        )
    if skip_compute:
        return {}
    posterior = _posterior_from_idata(idata)
    if posterior is None:
        return {}
    vals = _safe_rhat_values(posterior)
    if vals.size == 0:
        return {}
    return {
        "rhat_max": float(np.max(vals)),
        "rhat_mean": float(np.mean(vals)),
    }


def _compute_ess(idata, *, skip_compute: bool = False) -> Dict[str, float]:
    attrs = idata.attrs or {}
    has_ess_attr = (
        attrs.get("ess") is not None or attrs.get("ess_tail") is not None
    )
    if skip_compute and not has_ess_attr:
        return {}

    posterior = _posterior_from_idata(idata)
    if posterior is None:
        return {}

    metrics: Dict[str, float] = {}
    bulk_vals = _safe_ess_values(posterior, method="bulk")
    if bulk_vals.size:
        metrics.update(
            {
                "ess_bulk_min": float(np.min(bulk_vals)),
                "ess_bulk_median": float(np.median(bulk_vals)),
            }
        )

    tail_vals = _safe_ess_values(posterior, method="tail")
    if tail_vals.size:
        metrics.update(
            {
                "ess_tail_min": float(np.min(tail_vals)),
                "ess_tail_median": float(np.median(tail_vals)),
            }
        )

    return metrics


def _estimate_posterior_elements(idata) -> int:
    posterior = _require_dataset(idata, "posterior")
    total = 0
    for var in posterior.data_vars.values():
        total += int(var.size)
    return int(total)


def _thin_idata_draws(idata, *, step: int):
    """Return a thinned ``DataTree`` (every ``step`` draws)."""
    if step <= 1:
        return idata, False
    posterior = _require_dataset(idata, "posterior")
    if "draw" not in posterior.dims:
        return idata, False
    return _select_draw_slice(idata, slice(None, None, int(step))), True


def _cap_idata_draws(idata, *, max_draws: int):
    if max_draws <= 0:
        return idata, False
    posterior = _require_dataset(idata, "posterior")
    if "draw" not in posterior.dims:
        return idata, False
    n_draws = int(posterior.sizes.get("draw", 0))
    if n_draws <= max_draws:
        return idata, False
    start = max(0, n_draws - int(max_draws))
    return _select_draw_slice(idata, slice(start, None)), True


def _select_weight_indices(size: int, max_points: int) -> np.ndarray:
    if size <= 0 or max_points <= 0:
        return np.array([], dtype=int)
    if size <= max_points:
        return np.arange(size, dtype=int)
    idx = np.unique(
        np.linspace(0, size - 1, num=max_points, dtype=int, endpoint=True)
    )
    return idx


def _block_index_from_name(name: str) -> int | None:
    match = re.search(r"\d+", name)
    if not match:
        return None
    return int(match.group(0))


def _group_vars_by_block(posterior) -> dict[int, list[str]]:
    blocks: dict[int, list[str]] = {}
    for name in posterior.data_vars:
        idx = _block_index_from_name(str(name))
        if idx is None:
            continue
        blocks.setdefault(idx, []).append(str(name))
    return blocks


def _posterior_counts(posterior) -> dict:
    counts: dict[str, int | float | None | list] = {
        "chains": 0,
        "draws": 0,
        "params_total": None,
        "elements": 0,
        "vars": 0,
        "top_vars": [],
    }
    if posterior is None:
        return counts

    chains = int(posterior.sizes.get("chain", 0) or 0)
    draws = int(posterior.sizes.get("draw", 0) or 0)
    denom = chains * draws if chains and draws else None

    params_total = 0.0
    var_sizes = []
    elements_count = 0
    for name, var in posterior.data_vars.items():
        size = int(np.prod(var.shape))
        var_sizes.append((str(name), size))
        elements_count += size
        if denom and "chain" in var.dims and "draw" in var.dims:
            params_total += size / denom

    counts["chains"] = chains
    counts["draws"] = draws
    counts["vars"] = len(posterior.data_vars)
    counts["elements"] = elements_count
    counts["params_total"] = params_total if denom else None
    counts["top_vars"] = sorted(var_sizes, key=lambda x: x[1], reverse=True)[
        :3
    ]
    return counts


def _log_posterior_summary(stage: str, posterior, weight_meta=None) -> None:
    posterior_ds = posterior
    counts = _posterior_counts(posterior_ds)
    if not counts["vars"]:
        return
    params = counts["params_total"]
    params_text = f"{params:.1f}" if params is not None else "unknown"
    logger.info(
        "MCMC diagnostics: "
        f"{stage}: chains={counts['chains']}, "
        f"draws={counts['draws']}, "
        f"params≈{params_text}, "
        f"elements={counts['elements']:,}, "
        f"vars={counts['vars']}"
    )
    if weight_meta:
        items = list(weight_meta.items())[:3]
        details = ", ".join(
            f"{name}: {meta['size']}-> {meta['selected']}"
            for name, meta in items
        )
        logger.info(
            "MCMC diagnostics: weight subset "
            f"(showing {len(items)} of {len(weight_meta)}): {details}"
        )
    if counts["top_vars"]:
        top = ", ".join(
            f"{name}={size:,}" for name, size in counts["top_vars"]
        )
        logger.info(f"MCMC diagnostics: top vars by size: {top}")


def _select_posterior_subset(idata):
    posterior = _require_dataset(idata, "posterior")

    subset = {}
    weight_meta: dict[str, dict[str, int]] = {}
    for name, var in posterior.data_vars.items():
        var_name = str(name)
        if var_name.startswith("weights"):
            dims = [d for d in var.dims if d not in ("chain", "draw")]
            if not dims:
                subset[var_name] = var
                continue
            dim = dims[-1]
            size = int(var.sizes.get(dim, 0))
            idx = _select_weight_indices(size, DEFAULT_MCMC_DIAG_WEIGHT_POINTS)
            if idx.size == 0:
                continue
            weight_meta[var_name] = {
                "size": size,
                "selected": int(idx.size),
            }
            subset[var_name] = var.isel({dim: idx})
            continue
        subset[var_name] = var

    if not subset:
        return None, {}
    # Reconstruct a DataTree-compatible idata with only the subset vars
    subset_ds = xr.Dataset(subset, attrs=posterior.attrs)
    new_idata = xr.DataTree()
    new_idata.attrs.update(dict(idata.attrs))
    new_idata["posterior"] = xr.DataTree(dataset=subset_ds)
    return new_idata, weight_meta


def _compute_divergences(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not _has_group(idata, "sample_stats"):
        return metrics
    sample_stats = _require_dataset(idata, "sample_stats")

    div_keys = [key for key in sample_stats if "diverging" in str(key)]
    values = []
    for key in div_keys:
        values.append(np.asarray(sample_stats[key]))

    if not values:
        return metrics

    div = np.concatenate([v.reshape(-1) for v in values])
    metrics["divergence_fraction"] = float(np.mean(div))
    metrics["divergence_total"] = float(np.sum(div))
    return metrics


def _compute_acceptance(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not _has_group(idata, "sample_stats"):
        return metrics
    sample_stats = _require_dataset(idata, "sample_stats")

    for key in ("accept_prob", "acceptance_rate"):
        if key in sample_stats:
            arr = np.asarray(sample_stats[key])
            metrics["acceptance_rate_mean"] = float(np.mean(arr))
            return metrics

    channel_means = []
    for key in sample_stats:
        if not str(key).startswith("accept_prob_channel_"):
            continue
        channel_means.append(float(np.mean(np.asarray(sample_stats[key]))))

    if channel_means:
        metrics["acceptance_rate_mean"] = float(np.mean(channel_means))
    return metrics


def _flatten_sample_stats(idata, key_prefix: str) -> np.ndarray:
    if not _has_group(idata, "sample_stats"):
        return np.array([])
    sample_stats = _require_dataset(idata, "sample_stats")

    if key_prefix in sample_stats:
        return np.asarray(sample_stats[key_prefix]).reshape(-1)

    values = []
    for key in sample_stats:
        if not str(key).startswith(f"{key_prefix}_channel_"):
            continue
        values.append(np.asarray(sample_stats[key]).reshape(-1))
    if not values:
        return np.array([])
    return np.concatenate(values)


def _compute_step_size(idata) -> Dict[str, float]:
    arr = _flatten_sample_stats(idata, "step_size")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "step_size_median": float(np.median(arr)),
        "step_size_min": float(np.min(arr)),
    }


def _resolve_max_tree_depth(idata, config) -> int | None:
    attrs = idata.attrs or {}
    max_td = attrs.get("max_tree_depth")
    if max_td is None and config is not None:
        max_td = getattr(config, "max_tree_depth", None)
    if max_td is None:
        return None
    return int(max_td)


def _compute_tree_depth_saturation(idata, config=None) -> Dict[str, float]:
    max_td = _resolve_max_tree_depth(idata, config)
    if max_td is None:
        return {}

    tree_depth = _flatten_sample_stats(idata, "tree_depth")
    tree_depth = tree_depth[np.isfinite(tree_depth)]
    if tree_depth.size:
        hit = float(np.mean(tree_depth >= float(max_td)))
        return {
            "max_tree_depth_hit_frac": hit,
            "tree_depth_max": float(np.max(tree_depth)),
        }

    num_steps = _flatten_sample_stats(idata, "num_steps")
    if num_steps.size == 0:
        num_steps = _flatten_sample_stats(idata, "n_steps")
    num_steps = num_steps[np.isfinite(num_steps)]
    if num_steps.size == 0:
        return {}

    # In NUTS, hitting max tree depth implies 2**max_tree_depth steps.
    # Use a >= comparison for robustness.
    threshold = float(2 ** int(max_td))
    hit = float(np.mean(num_steps >= threshold))
    return {"max_tree_depth_hit_frac": hit}


def _compute_psis(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not _has_group(idata, "log_likelihood"):
        return metrics
    try:
        log_likelihood = _require_dataset(idata, "log_likelihood")
        values = []
        for var in log_likelihood.data_vars.values():
            arr = np.asarray(var)
            if arr.ndim < 2:
                continue
            values.append(arr.reshape(arr.shape[0], arr.shape[1], -1))
        if not values:
            return metrics
        stacked = np.concatenate(values, axis=-1)
        _, pareto = array_stats.psislw(-stacked, axis=(0, 1))
        if pareto.size:
            khat_max = float(np.nanmax(pareto))
            status_code, _ = khat_status(khat_max)
            metrics["psis_khat_max"] = khat_max
            if status_code is not None:
                metrics["psis_khat_status"] = float(status_code)
    except Exception:
        pass
    return metrics


def _run(
    *,
    idata=None,
    config=None,
    truth=None,
    signals=None,
    psd_ref=None,
    idata_vi=None,
) -> Dict[str, float]:
    """Aggregate core MCMC diagnostic scalars."""
    if idata is None:
        return {}

    attrs = idata.attrs or {}
    compute_psis_flag = bool(attrs.get("compute_psis", True))
    summary_mode = "full"  # Always use full diagnostics mode
    light_mode = False
    posterior_full = _require_dataset(idata, "posterior")
    _log_posterior_summary("base posterior", posterior_full)

    idata_heavy, weight_meta = _select_posterior_subset(idata)
    heavy_available = idata_heavy is not None
    if not heavy_available:
        logger.info(
            "MCMC diagnostics: skipping ESS/Rhat/PSIS (no non-weight variables)."
        )
        idata_heavy = idata
        weight_meta = {}

    idata_heavy, capped = _cap_idata_draws(
        idata_heavy, max_draws=DEFAULT_MCMC_DIAG_MAX_DRAWS
    )
    if capped:
        logger.info(
            f"MCMC diagnostics: capping draws to last {DEFAULT_MCMC_DIAG_MAX_DRAWS}."
        )
    total_elements = _estimate_posterior_elements(idata_heavy)
    if heavy_available:
        _log_posterior_summary(
            "filtered posterior",
            _require_dataset(idata_heavy, "posterior"),
            weight_meta,
        )
    max_elements = int(
        getattr(
            config, "mcmc_diag_max_elements", DEFAULT_MCMC_DIAG_MAX_ELEMENTS
        )
        or 0
    )
    thinned = False
    if max_elements > 0 and total_elements > max_elements:
        step = int(np.ceil(total_elements / max_elements))
        idata_heavy, thinned = _thin_idata_draws(idata_heavy, step=step)
        if thinned:
            total_heavy = _estimate_posterior_elements(idata_heavy)
            logger.info(
                "MCMC diagnostics: thinning draws by "
                f"{step} for ESS/Rhat/PSIS "
                f"({total_elements:,} -> {total_heavy:,} elements)."
            )
            if heavy_available:
                _log_posterior_summary(
                    "thinned posterior",
                    _require_dataset(idata_heavy, "posterior"),
                    weight_meta,
                )
        else:
            logger.warning(
                "MCMC diagnostics: unable to thin draws; "
                "skipping ESS/Rhat/PSIS to avoid OOM."
            )
    if not thinned:
        logger.info(f"MCMC diagnostics: posterior elements={total_elements:,}")

    rhat_drop = 0
    if rhat_drop > 0:
        idata_heavy = _select_draw_slice(idata_heavy, slice(rhat_drop, None))
    # TODO: remove light_mode  option.
    if light_mode:
        logger.info("MCMC diagnostics: light mode, skipping ESS/PSIS compute.")

    skip_heavy = (
        not thinned and max_elements > 0 and total_elements > max_elements
    ) or not heavy_available
    metrics: Dict[str, float] = {}
    metrics.update(
        _compute_rhat(idata_heavy, skip_compute=skip_heavy or light_mode)
    )
    metrics.update(
        _compute_ess(idata_heavy, skip_compute=skip_heavy or light_mode)
    )
    if compute_psis_flag and not (skip_heavy or light_mode):
        metrics.update(_compute_psis(idata_heavy))

    # Per-block ESS/Rhat (block index inferred from parameter names).
    posterior = _require_dataset(idata_heavy, "posterior")
    blocks = _group_vars_by_block(posterior)
    for block_idx, var_names in sorted(blocks.items()):
        block_ds = posterior[var_names]
        block_idata = xr.DataTree()
        block_idata.attrs.update(dict(idata_heavy.attrs))
        block_idata["posterior"] = xr.DataTree(dataset=block_ds)
        block_metrics = {}
        block_metrics.update(
            _compute_rhat(
                block_idata,
                skip_compute=skip_heavy or light_mode,
            )
        )
        block_metrics.update(
            _compute_ess(
                block_idata,
                skip_compute=skip_heavy or light_mode,
            )
        )
        for key, val in block_metrics.items():
            metrics[f"block_{block_idx}_{key}"] = val
    metrics.update(_compute_divergences(idata))
    metrics.update(_compute_acceptance(idata))
    metrics.update(_compute_step_size(idata))
    metrics.update(_compute_tree_depth_saturation(idata, config))

    return {
        key: float(val)
        for key, val in metrics.items()
        if val is not None and np.isfinite(val)
    }
