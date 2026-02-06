"""MCMC diagnostics: ESS, Rhat, divergences, PSIS."""

from __future__ import annotations

from typing import Callable, Dict

import arviz as az
import numpy as np

from ._utils import khat_status


def _metric_from_attr_or_compute(
    idata,
    *,
    attr_key: str,
    attr_metrics: Callable[[np.ndarray], Dict[str, float]],
    compute_fn: Callable[[], np.ndarray],
    compute_metrics: Callable[[np.ndarray], Dict[str, float]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    attrs = getattr(idata, "attrs", {}) or {}

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


def _compute_rhat(idata) -> Dict[str, float]:
    return _metric_from_attr_or_compute(
        idata,
        attr_key="rhat",
        attr_metrics=lambda vals: {
            "rhat_max": float(np.max(vals)),
            "rhat_mean": float(np.mean(vals)),
        },
        compute_fn=lambda: np.asarray(az.rhat(idata).to_array()),
        compute_metrics=lambda vals: {
            "rhat_max": float(np.max(vals)),
            "rhat_mean": float(np.mean(vals)),
        },
    )


def _compute_ess(idata) -> Dict[str, float]:
    return _metric_from_attr_or_compute(
        idata,
        attr_key="ess",
        attr_metrics=lambda vals: {
            "ess_bulk_min": float(np.min(vals)),
            "ess_bulk_median": float(np.median(vals)),
        },
        compute_fn=lambda: np.asarray(az.ess(idata, method="bulk").to_array()),
        compute_metrics=lambda vals: {
            "ess_bulk_min": float(np.min(vals)),
            "ess_bulk_median": float(np.median(vals)),
        },
    )


def _compute_divergences(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not hasattr(idata, "sample_stats"):
        return metrics

    div_keys = [key for key in idata.sample_stats if "diverging" in str(key)]
    values = []
    for key in div_keys:
        try:
            values.append(np.asarray(idata.sample_stats[key]))
        except Exception:
            continue

    if not values:
        return metrics

    div = np.concatenate([v.reshape(-1) for v in values])
    metrics["divergence_fraction"] = float(np.mean(div))
    metrics["divergence_total"] = float(np.sum(div))
    return metrics


def _compute_acceptance(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not hasattr(idata, "sample_stats"):
        return metrics

    for key in ("accept_prob", "acceptance_rate"):
        if key in idata.sample_stats:
            arr = np.asarray(idata.sample_stats[key])
            metrics["acceptance_rate_mean"] = float(np.mean(arr))
            return metrics

    channel_means = []
    for key in idata.sample_stats:
        if not str(key).startswith("accept_prob_channel_"):
            continue
        try:
            channel_means.append(
                float(np.mean(np.asarray(idata.sample_stats[key])))
            )
        except Exception:
            continue

    if channel_means:
        metrics["acceptance_rate_mean"] = float(np.mean(channel_means))
    return metrics


def _flatten_sample_stats(idata, key_prefix: str) -> np.ndarray:
    if not hasattr(idata, "sample_stats"):
        return np.array([])

    if key_prefix in idata.sample_stats:
        try:
            return np.asarray(idata.sample_stats[key_prefix]).reshape(-1)
        except Exception:
            return np.array([])

    values = []
    for key in idata.sample_stats:
        if not str(key).startswith(f"{key_prefix}_channel_"):
            continue
        try:
            values.append(np.asarray(idata.sample_stats[key]).reshape(-1))
        except Exception:
            continue
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
    attrs = getattr(idata, "attrs", {}) or {}
    max_td = None
    try:
        max_td = attrs.get("max_tree_depth")
    except Exception:
        max_td = None
    if max_td is None and config is not None:
        max_td = getattr(config, "max_tree_depth", None)
    if max_td is None:
        return None
    try:
        return int(max_td)
    except Exception:
        return None


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
    try:
        loo_res = az.loo(idata, pointwise=True)
        pareto = np.asarray(getattr(loo_res, "pareto_k", []))
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

    attrs = getattr(idata, "attrs", {}) or {}
    compute_psis_flag = bool(attrs.get("compute_psis", True))

    fns = [
        _compute_rhat,
        _compute_ess,
        _compute_divergences,
        _compute_acceptance,
        _compute_step_size,
        lambda _idata: _compute_tree_depth_saturation(_idata, config),
    ]
    if compute_psis_flag:
        fns.append(_compute_psis)

    metrics: Dict[str, float] = {}
    for fn in fns:
        metrics.update(fn(idata))

    return {
        key: float(val)
        for key, val in metrics.items()
        if val is not None and np.isfinite(val)
    }
