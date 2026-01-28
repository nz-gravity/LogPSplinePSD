"""Shared helpers for diagnostics modules."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import numpy as np
from scipy.integrate import simpson


def as_scalar(value: Any) -> Optional[float]:
    """Best-effort conversion to a Python float."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass

    try:
        arr = np.asarray(value)
        if arr.size:
            return float(arr.reshape(-1)[0])
    except Exception:
        return None
    return None


def khat_status(khat_max: Optional[float]) -> Tuple[Optional[float], str]:
    """Map PSIS k-hat into a numeric status and human-readable label."""
    if khat_max is None or not np.isfinite(khat_max):
        return None, "unknown"
    if khat_max < 0.5:
        return 0.0, "ok"
    if khat_max <= 0.7:
        return 1.0, "warn"
    return 2.0, "fail"


def compute_riae(
    median_psd: np.ndarray, true_psd: np.ndarray, freqs: Iterable[float]
) -> float:
    """Relative integrated absolute error (univariate)."""
    freqs_arr = np.asarray(freqs, dtype=float)
    numerator = float(simpson(np.abs(median_psd - true_psd), x=freqs_arr))
    denominator = float(simpson(true_psd, x=freqs_arr))
    return float(numerator / denominator) if denominator != 0 else float("nan")


def compute_matrix_riae(
    median_psd_matrix: np.ndarray,
    true_psd_matrix: np.ndarray,
    freqs: Iterable[float],
) -> float:
    """RIAE for multivariate PSD matrices using the Frobenius norm."""
    freqs_arr = np.asarray(freqs, dtype=float)
    diff_frobenius = np.array(
        [
            np.linalg.norm(median_psd_matrix[k] - true_psd_matrix[k], "fro")
            for k in range(len(freqs_arr))
        ]
    )
    true_frobenius = np.array(
        [
            np.linalg.norm(true_psd_matrix[k], "fro")
            for k in range(len(freqs_arr))
        ]
    )
    numerator = float(simpson(diff_frobenius, x=freqs_arr))
    denominator = float(simpson(true_frobenius, x=freqs_arr))
    return float(numerator / denominator) if denominator != 0 else float("nan")


def compute_riae_errorbars(
    psd_samples: np.ndarray, true_psd: np.ndarray, freqs: Iterable[float]
) -> dict:
    """Quantiles of RIAE across a collection of PSD samples."""
    riae_samples = [
        compute_riae(psd, true_psd, freqs) for psd in np.asarray(psd_samples)
    ]
    arr = np.asarray(riae_samples, dtype=float)
    return {
        "q05": float(np.percentile(arr, 5)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "q95": float(np.percentile(arr, 95)),
    }


def compute_ci_coverage_univar(
    psd_samples: np.ndarray, true_psd: np.ndarray
) -> float:
    """Compute 90% credible interval coverage for univariate PSD."""
    arr = np.asarray(psd_samples)
    if arr.ndim == 2 and arr.shape[0] == 3:
        posterior_lower = arr[0]
        posterior_upper = arr[-1]
    else:
        posterior_lower = np.percentile(arr, 5.0, axis=0)
        posterior_upper = np.percentile(arr, 95.0, axis=0)
    coverage = np.mean(
        (true_psd >= posterior_lower) & (true_psd <= posterior_upper)
    )
    return float(coverage)


def compute_ci_coverage_multivar(
    psd_matrix_samples: np.ndarray, true_psd_real: np.ndarray
) -> float:
    """Compute 90% credible interval coverage for multivariate PSD matrices."""
    true_psd_arr = np.asarray(true_psd_real)
    true_psd = np.zeros(true_psd_arr.shape, dtype=np.float64)
    for i in range(true_psd_arr.shape[0]):
        true_psd[i] = _complex_to_real(true_psd_arr[i])

    arr = np.asarray(psd_matrix_samples)
    if arr.ndim == 4 and arr.shape[0] == 3:
        posterior_lower_raw = arr[0]
        posterior_upper_raw = arr[-1]
        posterior_lower = np.zeros(posterior_lower_raw.shape, dtype=np.float64)
        posterior_upper = np.zeros(posterior_upper_raw.shape, dtype=np.float64)
        for i in range(posterior_lower_raw.shape[0]):
            posterior_lower[i] = _complex_to_real(posterior_lower_raw[i])
            posterior_upper[i] = _complex_to_real(posterior_upper_raw[i])
    else:
        if np.iscomplexobj(arr):
            psd_matrix_real = np.zeros_like(arr, dtype=np.float64)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    psd_matrix_real[i, j] = _complex_to_real(arr[i, j])
        else:
            psd_matrix_real = np.asarray(arr, dtype=np.float64)
        posterior_lower = np.percentile(psd_matrix_real, 5.0, axis=0)
        posterior_upper = np.percentile(psd_matrix_real, 95.0, axis=0)

    coverage = np.mean(
        (true_psd >= posterior_lower) & (true_psd <= posterior_upper)
    )
    return float(coverage)


def extract_percentile(
    values: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    """Return the slice of ``values`` closest to the requested percentile."""
    idx = int(np.argmin(np.abs(np.asarray(percentiles, dtype=float) - target)))
    return values[idx]


def _complex_to_real(mat: np.ndarray) -> np.ndarray:
    """Convert complex matrices to a real-valued representation for CI checks."""
    arr = np.asarray(mat)
    if not np.iscomplexobj(arr):
        return arr

    n = arr.shape[-1]
    upper = np.triu(np.ones((n, n), dtype=bool))
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)

    out = np.where(upper, arr.real, 0.0)
    out = np.where(lower, arr.imag, out)
    return out
