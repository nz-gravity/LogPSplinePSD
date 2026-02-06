"""Variational inference diagnostics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ._utils import as_scalar, khat_status
from .psd_compare import _get_psd_dataset
from .psd_compare import _run as _run_psd_compare


def _extract_field(obj, key: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, "attrs") and key in getattr(obj, "attrs", {}):
        try:
            return obj.attrs.get(key)
        except Exception:
            return None
    return getattr(obj, key, None)


def _extract_losses(idata_vi) -> Optional[np.ndarray]:
    losses = _extract_field(idata_vi, "losses")
    if losses is None:
        return None
    arr = np.asarray(losses, dtype=float).reshape(-1)
    return arr if arr.size else None


def _psd_variance_from_ds(psd_ds) -> Optional[float]:
    if psd_ds is None:
        return None
    if "psd" in psd_ds:
        psd = psd_ds["psd"]
        freqs = np.asarray(psd.coords.get("freq", np.arange(psd.shape[-1])))
        perc = np.asarray(psd.coords.get("percentile", []), dtype=float)
        values = np.asarray(psd.values)
        if perc.size == 0:
            perc = np.arange(values.shape[0], dtype=float)
        q50 = values[int(np.argmin(np.abs(perc - 50.0)))]
        return float(np.trapezoid(q50, freqs))
    if "psd_matrix_real" in psd_ds:
        psd = psd_ds["psd_matrix_real"]
        freqs = np.asarray(psd.coords.get("freq", np.arange(psd.shape[-1])))
        perc = np.asarray(psd.coords.get("percentile", []), dtype=float)
        values = np.asarray(psd.values)
        if perc.size == 0:
            perc = np.arange(values.shape[0], dtype=float)
        q50 = values[int(np.argmin(np.abs(perc - 50.0)))]
        diag = np.real(
            q50[:, np.arange(q50.shape[1]), np.arange(q50.shape[2])]
        )
        var = np.trapezoid(diag, freqs, axis=0)
        return float(np.mean(var))
    return None


def _variance_ratio_vs_mcmc(idata_vi, idata) -> Optional[float]:
    vi_ds = _get_psd_dataset(idata_vi, idata_vi)
    ref_ds = _get_psd_dataset(idata, idata)
    if vi_ds is None or ref_ds is None:
        return None
    vi_var = _psd_variance_from_ds(vi_ds)
    ref_var = _psd_variance_from_ds(ref_ds)
    if vi_var is None or ref_var in (None, 0):
        return None
    return float(vi_var / ref_var)


def _moment_bias(summary: dict) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not summary:
        return metrics
    hyper = summary.get("hyperparameters") or []
    if hyper:
        bias = [abs(entry.get("bias_pct", 0.0)) for entry in hyper]
        var_ratio = [entry.get("var_ratio", np.nan) for entry in hyper]
        metrics["moment_bias_pct"] = float(np.nanmax(bias))
        metrics["moment_var_ratio_median"] = float(np.nanmedian(var_ratio))
    weights_stats = summary.get("weights") or {}
    if weights_stats:
        for key in ("var_ratio_median", "bias_median_abs", "bias_max_abs"):
            val = weights_stats.get(key)
            if val is not None:
                metrics[f"weights_{key}"] = float(val)
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
    """Collect scalar VI diagnostics."""
    if idata_vi is None:
        return {}

    metrics: Dict[str, float] = {}

    losses = _extract_losses(idata_vi)
    if losses is not None:
        metrics["elbo_final"] = float(losses[-1])
        metrics["elbo_delta"] = float(losses[-1] - losses[0])
        window = min(20, losses.size)
        slope = (losses[-1] - losses[-window]) / max(1, window - 1)
        metrics["elbo_converged"] = 1.0 if abs(slope) < 1e-3 else 0.0
        recent_improve = np.abs(np.diff(losses[-window:]))
        stall_iters = int(np.sum(recent_improve < 1e-3))
        metrics["elbo_stall_iters"] = float(stall_iters)

    khat_val = as_scalar(_extract_field(idata_vi, "psis_khat_max"))
    if khat_val is not None:
        metrics["psis_khat_max"] = khat_val
        status_code, _ = khat_status(khat_val)
        if status_code is not None:
            metrics["psis_khat_status"] = float(status_code)

    summary = _extract_field(idata_vi, "psis_moment_summary")
    if summary:
        metrics.update(_moment_bias(summary))

    ratio = _variance_ratio_vs_mcmc(idata_vi, idata)
    if ratio is not None and np.isfinite(ratio):
        metrics["variance_ratio_vs_mcmc"] = ratio

    psd_metrics = _run_psd_compare(
        idata=None,
        idata_vi=idata_vi,
        truth=truth if truth is not None else psd_ref,
    )
    if psd_metrics:
        if "coverage" in psd_metrics:
            metrics["coverage_vs_truth"] = psd_metrics["coverage"]
        if "riae" in psd_metrics:
            metrics["riae_vs_truth"] = psd_metrics["riae"]
        if "riae_matrix" in psd_metrics:
            metrics["riae_vs_truth"] = psd_metrics["riae_matrix"]

    return {
        key: float(val)
        for key, val in metrics.items()
        if val is not None and np.isfinite(val)
    }
