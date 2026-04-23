"""Small PSD comparison helpers still used by VI diagnostics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import simpson

from ._utils import (
    compute_ci_coverage_multivar,
    compute_matrix_l2,
    compute_matrix_riae,
    compute_riae,
    interior_frequency_slice,
)


def _coherence(psd_matrix: np.ndarray) -> np.ndarray:
    diag_entries = np.real(np.diagonal(psd_matrix, axis1=1, axis2=2))
    denom = np.sqrt(
        np.maximum(diag_entries[..., None] * diag_entries[:, None, :], 0.0)
    )
    coherence = np.zeros_like(psd_matrix, dtype=np.float64)
    valid = denom > 0
    coherence[valid] = np.abs(psd_matrix[valid]) ** 2 / denom[valid] ** 2
    return coherence


def _extract_percentile_slice(
    values: np.ndarray,
    percentiles: np.ndarray,
    target: float,
) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - target)))
    return np.asarray(values[idx])


def _compute_multivar_diagnostics_from_arrays(
    estimate_psd: np.ndarray,
    true_psd_real: np.ndarray,
    freqs: np.ndarray,
    posterior_psd_quantiles: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    estimate_psd = np.asarray(estimate_psd)
    true_psd_real = np.asarray(true_psd_real)
    freqs = np.asarray(freqs, dtype=np.float64)

    diagnostics: Dict[str, object] = {}
    coverage_interval = [5.0, 95.0]
    coverage_level = 0.90

    diagnostics["riae_matrix"] = float(
        compute_matrix_riae(estimate_psd, true_psd_real, freqs)
    )
    diagnostics["l2_matrix"] = float(
        compute_matrix_l2(estimate_psd, true_psd_real, freqs)
    )

    per_channel_riae = []
    for channel_idx in range(true_psd_real.shape[1]):
        estimate_diag = np.real(estimate_psd[:, channel_idx, channel_idx])
        true_diag = np.real(true_psd_real[:, channel_idx, channel_idx])
        per_channel_riae.append(compute_riae(estimate_diag, true_diag, freqs))
    diagnostics["riae_per_channel"] = [float(v) for v in per_channel_riae]

    offdiag_mask = ~np.eye(true_psd_real.shape[1], dtype=bool)
    diff_offdiag = np.linalg.norm(
        (estimate_psd - true_psd_real)[:, offdiag_mask], axis=1
    )
    true_offdiag = np.linalg.norm(true_psd_real[:, offdiag_mask], axis=1)
    numerator_offdiag = float(simpson(diff_offdiag, x=freqs))
    denominator_offdiag = float(simpson(true_offdiag, x=freqs))
    diagnostics["riae_offdiag"] = (
        float(numerator_offdiag / denominator_offdiag)
        if denominator_offdiag != 0
        else float("nan")
    )

    estimate_coh = _coherence(estimate_psd)
    true_coh = _coherence(true_psd_real)
    coh_diff = np.linalg.norm(
        (estimate_coh - true_coh)[:, offdiag_mask], axis=1
    )
    coh_true = np.linalg.norm(true_coh[:, offdiag_mask], axis=1)
    coh_num = float(simpson(coh_diff, x=freqs))
    coh_den = float(simpson(coh_true, x=freqs))
    diagnostics["coherence_riae"] = (
        float(coh_num / coh_den) if coh_den != 0 else float("nan")
    )

    freq_quantiles = np.quantile(freqs, [0.0, 0.25, 0.5, 0.75, 1.0])
    freq_edges = np.unique(freq_quantiles)
    riae_bands = []
    for start, end in zip(freq_edges[:-1], freq_edges[1:]):
        mask = (freqs >= start) & (freqs <= end)
        if np.count_nonzero(mask) < 2 or end <= start:
            continue
        riae_band = compute_matrix_riae(
            estimate_psd[mask], true_psd_real[mask], freqs[mask]
        )
        riae_bands.append(
            {
                "start": float(start),
                "end": float(end),
                "value": float(riae_band),
            }
        )
    if riae_bands:
        diagnostics["riae_bands"] = riae_bands

    if (
        posterior_psd_quantiles is not None
        and posterior_psd_quantiles.shape[0] >= 3
    ):
        q05 = np.asarray(posterior_psd_quantiles[0])
        q50 = np.asarray(posterior_psd_quantiles[1])
        q95 = np.asarray(posterior_psd_quantiles[2])

        riae_low = compute_matrix_riae(np.real(q05), true_psd_real, freqs)
        riae_med = compute_matrix_riae(np.real(q50), true_psd_real, freqs)
        riae_high = compute_matrix_riae(np.real(q95), true_psd_real, freqs)
        diagnostics["riae_matrix_errorbars"] = [
            float(riae_low),
            float(riae_low),
            float(riae_med),
            float(riae_high),
            float(riae_high),
        ]

        q50_real_array = np.real(q50)
        diag_mask_2d = np.eye(q50_real_array.shape[1], dtype=bool)
        diag_width = (np.real(q95) - np.real(q05))[:, diag_mask_2d]
        diagnostics["ci_width_diag_mean"] = float(np.mean(diag_width))
        diagnostics["ci_width"] = diagnostics["ci_width_diag_mean"]

        coverage_value = compute_ci_coverage_multivar(
            np.stack([q05, q50, q95], axis=0),
            true_psd_real,
        )
        if np.isfinite(coverage_value):
            diagnostics["coverage"] = float(coverage_value)
            diagnostics["ci_coverage"] = float(coverage_value)
            diagnostics["coverage_interval"] = coverage_interval
            diagnostics["coverage_level"] = coverage_level

    return diagnostics


def _handle_multivariate(
    psd_group, true_psd_real: np.ndarray
) -> Dict[str, object]:
    """Compute multivariate diagnostics from percentile-indexed PSD datasets."""
    freq = np.asarray(psd_group["freq"].values, dtype=np.float64)
    psd_real = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    psd_imag = np.asarray(
        psd_group["psd_matrix_imag"].values, dtype=np.float64
    )
    percentiles = np.asarray(
        psd_group["psd_matrix_real"].coords.get(
            "percentile", np.arange(psd_real.shape[0], dtype=float)
        ),
        dtype=np.float64,
    )

    q05 = _extract_percentile_slice(
        psd_real, percentiles, 5.0
    ) + 1j * np.asarray(_extract_percentile_slice(psd_imag, percentiles, 5.0))
    q50 = _extract_percentile_slice(
        psd_real, percentiles, 50.0
    ) + 1j * np.asarray(_extract_percentile_slice(psd_imag, percentiles, 50.0))
    q95 = _extract_percentile_slice(
        psd_real, percentiles, 95.0
    ) + 1j * np.asarray(_extract_percentile_slice(psd_imag, percentiles, 95.0))
    posterior_psd_quantiles = np.stack([q05, q50, q95], axis=0)

    diagnostics = compute_multivar_riae_diagnostics(
        q50,
        true_psd_real,
        freq,
        psd_quantiles={"posterior_psd": posterior_psd_quantiles},
    )

    per_channel = np.asarray(
        diagnostics.get("riae_per_channel", []), dtype=float
    )
    diagnostics["riae_diag_mean"] = (
        float(np.mean(per_channel)) if per_channel.size else float("nan")
    )

    return diagnostics


def compute_multivar_riae_diagnostics(
    vi_psd: np.ndarray,
    true_psd_real: np.ndarray,
    freqs: np.ndarray,
    psd_quantiles: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """Legacy multivariate diagnostics used by VI adapters."""
    freq_idx = interior_frequency_slice(np.asarray(freqs).size)
    freqs = np.asarray(freqs)[freq_idx]
    vi_psd = np.asarray(vi_psd)[freq_idx, ...]
    true_psd_real = np.asarray(true_psd_real)[freq_idx, ...]
    posterior_psd_quantiles = (
        np.asarray(psd_quantiles.get("posterior_psd"), dtype=np.complex128)
        if psd_quantiles and psd_quantiles.get("posterior_psd") is not None
        else None
    )
    if (
        posterior_psd_quantiles is not None
        and posterior_psd_quantiles.shape[0] >= 3
    ):
        posterior_psd_quantiles = np.asarray(posterior_psd_quantiles)[
            :, freq_idx, ...
        ]

    return _compute_multivar_diagnostics_from_arrays(
        vi_psd,
        true_psd_real,
        freqs,
        posterior_psd_quantiles=posterior_psd_quantiles,
    )
