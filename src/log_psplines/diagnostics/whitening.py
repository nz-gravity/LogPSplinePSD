"""Simple whitening diagnostics based on autocorrelation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _lag_autocorr(signal: np.ndarray, lag: int) -> float:
    if signal.size <= lag:
        return np.nan
    mean = np.mean(signal)
    x0 = signal[:-lag] - mean
    x1 = signal[lag:] - mean
    denom = np.std(x0) * np.std(x1)
    if denom == 0:
        return np.nan
    return float(np.dot(x0, x1) / ((x0.size - 1) * denom))


def run(
    *,
    idata=None,
    config=None,
    truth=None,
    signals=None,
    psd_ref=None,
    idata_vi=None,
) -> Dict[str, float]:
    """Autocorrelation-based whiteness checks on provided signals."""
    if signals is None:
        return {}

    arr = np.asarray(signals, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]

    metrics: Dict[str, float] = {}
    for lag in (1, 5, 10):
        vals = []
        for ch in range(arr.shape[1]):
            corr = _lag_autocorr(arr[:, ch], lag)
            if np.isfinite(corr):
                vals.append(abs(corr))
        if vals:
            metrics[f"lag{lag}_autocorr_abs_mean"] = float(np.mean(vals))

    return metrics
