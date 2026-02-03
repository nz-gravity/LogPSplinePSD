"""Reusable PSD accuracy metrics for diagnostics and studies."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import simpson

from ._utils import (
    compute_ci_coverage_multivar,
    compute_matrix_riae,
    compute_riae,
    extract_percentile,
)


def relative_l2_matrix(
    est_psd: np.ndarray, true_psd: np.ndarray, freqs: np.ndarray
) -> float:
    """Relative L2 error for matrix-valued PSDs (Frobenius, integrated)."""

    diff_norm2 = np.array(
        [
            np.linalg.norm(est_psd[k] - true_psd[k], "fro") ** 2
            for k in range(len(freqs))
        ]
    )
    true_norm2 = np.array(
        [np.linalg.norm(true_psd[k], "fro") ** 2 for k in range(len(freqs))]
    )
    numerator = float(simpson(diff_norm2, x=freqs))
    denominator = float(simpson(true_norm2, x=freqs))
    return (
        float(np.sqrt(numerator / denominator))
        if denominator != 0
        else float("nan")
    )


def relative_l2_vector(
    est: np.ndarray, true: np.ndarray, freqs: np.ndarray
) -> float:
    """Relative L2 error for vector PSDs (integrated)."""

    diff_sq = (est - true) ** 2
    numerator = float(simpson(diff_sq, x=freqs))
    denominator = float(simpson(true**2, x=freqs))
    return (
        float(np.sqrt(numerator / denominator))
        if denominator != 0
        else float("nan")
    )


def summarize_multivar_psd_metrics(
    psd_ds,
    *,
    label: str,
    true_psd: np.ndarray,
    freqs: np.ndarray,
    freq_mask: Optional[np.ndarray] = None,
    log_eps: float = 1e-60,
) -> Optional[dict]:
    """Summarize multivariate PSD accuracy metrics from posterior percentiles."""

    if psd_ds is None:
        return None

    psd_real = np.asarray(psd_ds["psd_matrix_real"].values)
    percentiles = np.asarray(
        psd_ds["psd_matrix_real"].coords.get("percentile", []), dtype=float
    )
    if percentiles.size == 0:
        return None

    psd_imag = (
        np.asarray(psd_ds["psd_matrix_imag"].values)
        if "psd_matrix_imag" in psd_ds
        else np.zeros_like(psd_real)
    )

    q50_real = extract_percentile(psd_real, percentiles, 50.0)
    q05_real = extract_percentile(psd_real, percentiles, 5.0)
    q95_real = extract_percentile(psd_real, percentiles, 95.0)
    q50_im = extract_percentile(psd_imag, percentiles, 50.0)
    q05_im = extract_percentile(psd_imag, percentiles, 5.0)
    q95_im = extract_percentile(psd_imag, percentiles, 95.0)

    if freq_mask is not None:
        freqs = freqs[freq_mask]
        q50_real = q50_real[freq_mask]
        q05_real = q05_real[freq_mask]
        q95_real = q95_real[freq_mask]
        q50_im = q50_im[freq_mask]
        q05_im = q05_im[freq_mask]
        q95_im = q95_im[freq_mask]
        true_psd = true_psd[freq_mask]

    riae = compute_matrix_riae(q50_real, true_psd.real, freqs)
    l2_rel = relative_l2_matrix(q50_real, true_psd.real, freqs)

    true_diag = np.diagonal(true_psd.real, axis1=1, axis2=2)
    est_diag = np.diagonal(q50_real, axis1=1, axis2=2)
    log_true = np.log10(np.maximum(true_diag, log_eps))
    log_est = np.log10(np.maximum(est_diag, log_eps))
    log_riae = float(
        np.mean(
            [
                compute_riae(log_est[:, i], log_true[:, i], freqs)
                for i in range(log_true.shape[1])
            ]
        )
    )
    log_l2 = float(
        np.mean(
            [
                relative_l2_vector(log_est[:, i], log_true[:, i], freqs)
                for i in range(log_true.shape[1])
            ]
        )
    )

    percentiles_stack = np.stack(
        [
            q05_real + 1j * q05_im,
            q50_real + 1j * q50_im,
            q95_real + 1j * q95_im,
        ],
        axis=0,
    )
    coverage = compute_ci_coverage_multivar(percentiles_stack, true_psd.real)

    diag_widths = np.diagonal(q95_real - q05_real, axis1=1, axis2=2)
    width_median = float(np.median(diag_widths))
    width_mean = float(np.mean(diag_widths))

    return {
        "label": label,
        "riae_matrix": float(riae),
        "relative_l2_matrix": float(l2_rel),
        "log_riae_diag": log_riae,
        "log_l2_diag": log_l2,
        "coverage_90": float(coverage),
        "ci_width_median": width_median,
        "ci_width_mean": width_mean,
    }


__all__ = [
    "relative_l2_matrix",
    "relative_l2_vector",
    "summarize_multivar_psd_metrics",
]
