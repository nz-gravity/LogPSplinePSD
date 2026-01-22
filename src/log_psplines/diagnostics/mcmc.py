"""MCMC diagnostics: ESS, Rhat, divergences, PSIS."""

from __future__ import annotations

from typing import Dict

import arviz as az
import numpy as np

from ._utils import khat_status


def _compute_rhat(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # Prefer precomputed Rhat when available (avoids expensive ArviZ scans).
    try:
        attrs = getattr(idata, "attrs", {}) or {}
        rhat_attr = attrs.get("rhat")
        if rhat_attr is not None:
            vals = np.asarray(rhat_attr).reshape(-1)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                metrics["rhat_max"] = float(np.max(finite))
                metrics["rhat_mean"] = float(np.mean(finite))
                return metrics
    except Exception:
        pass
    try:
        rhat = az.rhat(idata)
        vals = np.asarray(rhat.to_array()).reshape(-1)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            metrics["rhat_max"] = float(np.max(finite))
            metrics["rhat_mean"] = float(np.mean(finite))
    except Exception:
        pass
    return metrics


def _compute_ess(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # Prefer precomputed ESS when available (avoids expensive ArviZ scans).
    try:
        attrs = getattr(idata, "attrs", {}) or {}
        ess_attr = attrs.get("ess")
        if ess_attr is not None:
            vals = np.asarray(ess_attr).reshape(-1)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                metrics["ess_bulk_min"] = float(np.min(finite))
                metrics["ess_bulk_median"] = float(np.median(finite))
                return metrics
    except Exception:
        pass
    try:
        ess = az.ess(idata, method="bulk")
        vals = np.asarray(ess.to_array())
        metrics["ess_bulk_min"] = float(np.nanmin(vals))
        metrics["ess_bulk_median"] = float(np.nanmedian(vals))
    except Exception:
        pass
    return metrics


def _compute_divergences(idata) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not hasattr(idata, "sample_stats"):
        return metrics

    div_keys = [key for key in idata.sample_stats if "diverging" in str(key)]
    values = []
    for key in div_keys:
        try:
            arr = np.asarray(idata.sample_stats[key])
            values.append(arr)
        except Exception:
            continue

    if values:
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
        if str(key).startswith("accept_prob_channel_"):
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
            status_code, status_label = khat_status(khat_max)
            metrics["psis_khat_max"] = khat_max
            if status_code is not None:
                metrics["psis_khat_status"] = float(status_code)
    except Exception:
        pass
    return metrics


def run(
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

    metrics: Dict[str, float] = {}
    # Read optional toggle from idata attrs to skip PSIS
    attrs = getattr(idata, "attrs", {}) or {}
    compute_psis_flag = True
    try:
        flag = attrs.get("compute_psis")
        if flag is not None:
            compute_psis_flag = bool(flag)
    except Exception:
        pass

    fns = [
        _compute_rhat,
        _compute_ess,
        _compute_divergences,
        _compute_acceptance,
    ]
    if compute_psis_flag:
        fns.append(_compute_psis)
    for fn in fns:
        metrics.update(fn(idata))

    return {
        key: float(val)
        for key, val in metrics.items()
        if val is not None and np.isfinite(val)
    }
