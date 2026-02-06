"""Compare time-domain moments with PSD-implied expectations."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.integrate import simpson

from ._utils import extract_percentile
from .psd_compare import _get_psd_dataset


def _psd_variance(psd_ds) -> float:
    if "psd" in psd_ds:
        psd = psd_ds["psd"]
        freqs = np.asarray(psd.coords.get("freq", np.arange(psd.shape[-1])))
        percentiles = np.asarray(psd.coords.get("percentile", []), dtype=float)
        values = np.asarray(psd.values)
        if percentiles.size == 0:
            percentiles = np.arange(values.shape[0], dtype=float)
        q50 = extract_percentile(values, percentiles, 50.0)
        return float(simpson(q50, x=freqs))

    if "psd_matrix_real" in psd_ds:
        psd = psd_ds["psd_matrix_real"]
        freqs = np.asarray(psd.coords.get("freq", np.arange(psd.shape[-1])))
        percentiles = np.asarray(psd.coords.get("percentile", []), dtype=float)
        values = np.asarray(psd.values)
        if percentiles.size == 0:
            percentiles = np.arange(values.shape[0], dtype=float)
        q50 = extract_percentile(values, percentiles, 50.0)
        diag = np.real(
            q50[:, np.arange(q50.shape[1]), np.arange(q50.shape[2])]
        )
        variances = simpson(diag, x=freqs, axis=0)
        return float(np.mean(variances))
    return float("nan")


def _run(
    *,
    idata=None,
    config=None,
    truth=None,
    signals=None,
    psd_ref=None,
    idata_vi=None,
) -> Dict[str, float]:
    """Return variance comparisons between time and frequency domains."""
    if signals is None:
        return {}

    arr = np.asarray(signals, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]

    empirical_var = np.var(arr, axis=0, ddof=1)
    metrics: Dict[str, float] = {
        "empirical_variance_mean": float(np.mean(empirical_var))
    }

    psd_ds = _get_psd_dataset(idata, idata_vi)
    if psd_ds is not None:
        psd_var = _psd_variance(psd_ds)
        if np.isfinite(psd_var) and psd_var != 0:
            metrics["variance_ratio"] = float(np.mean(empirical_var) / psd_var)
            metrics["variance_bias_abs"] = float(
                abs(np.mean(empirical_var) - psd_var)
            )

    return {
        key: float(val)
        for key, val in metrics.items()
        if val is not None and np.isfinite(val)
    }
