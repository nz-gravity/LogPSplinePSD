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
