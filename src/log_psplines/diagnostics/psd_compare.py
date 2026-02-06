"""PSD accuracy diagnostics (RIAE, coverage, coherence)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import simpson

from ._utils import (
    compute_ci_coverage_multivar,
    compute_ci_coverage_univar,
    compute_matrix_riae,
    compute_riae,
    extract_percentile,
)


def _get_psd_dataset(idata, idata_vi):
    """Return the first available PSD dataset from MCMC or VI idata."""
    for source in (idata, idata_vi):
        if source is None:
            continue
        for name in ("posterior_psd", "vi_posterior_psd"):
            dataset = getattr(source, name, None)
            if dataset is not None:
                return dataset
    return None


def _coherence(psd_matrix: np.ndarray) -> np.ndarray:
    diag_entries = np.real(np.diagonal(psd_matrix, axis1=1, axis2=2))
    denom = np.sqrt(
        np.maximum(diag_entries[..., None] * diag_entries[:, None, :], 0.0)
    )
    coherence = np.zeros_like(psd_matrix, dtype=np.float64)
    valid = denom > 0
    coherence[valid] = np.abs(psd_matrix[valid]) ** 2 / denom[valid] ** 2
    return coherence


def _parse_reference(reference: Optional[object]) -> Optional[np.ndarray]:
    if reference is None:
        return None
    try:
        arr = np.asarray(reference)
    except Exception:
        return None
    return arr


def _handle_univariate(psd_ds, reference: np.ndarray) -> Dict[str, float]:
    psd = psd_ds["psd"]
    freqs = np.asarray(psd.coords.get("freq", np.arange(psd.shape[-1])))
    values = np.asarray(psd.values)
    percentiles = np.asarray(psd.coords.get("percentile", []), dtype=float)
    if percentiles.size == 0:
        percentiles = np.arange(values.shape[0], dtype=float)

    metrics: Dict[str, float] = {}

    q50 = extract_percentile(values, percentiles, 50.0)
    metrics["riae"] = compute_riae(q50, reference, freqs)

    if percentiles.size >= 3:
        q05 = extract_percentile(values, percentiles, 5.0)
        q95 = extract_percentile(values, percentiles, 95.0)
        metrics["riae_p05"] = compute_riae(q05, reference, freqs)
        metrics["riae_p95"] = compute_riae(q95, reference, freqs)
        metrics["coverage"] = compute_ci_coverage_univar(values, reference)

    return metrics


def _handle_multivariate(psd_ds, reference: np.ndarray) -> Dict[str, float]:
    psd_real = np.asarray(psd_ds["psd_matrix_real"].values)
    percentiles = np.asarray(
        psd_ds["psd_matrix_real"].coords.get("percentile", []), dtype=float
    )
    freqs = np.asarray(psd_ds["psd_matrix_real"].coords.get("freq"))
    psd_imag = None
    if "psd_matrix_imag" in psd_ds:
        psd_imag = np.asarray(psd_ds["psd_matrix_imag"].values)

    true_psd_real = np.real(reference)

    metrics: Dict[str, float] = {}

    q50_real = extract_percentile(psd_real, percentiles, 50.0)
    metrics["riae_matrix"] = compute_matrix_riae(
        q50_real, true_psd_real, freqs
    )

    diag_riae = []
    for channel in range(true_psd_real.shape[1]):
        diag_riae.append(
            compute_riae(
                q50_real[:, channel, channel],
                true_psd_real[:, channel, channel],
                freqs,
            )
        )
    if diag_riae:
        metrics["riae_diag_mean"] = float(np.mean(diag_riae))
        metrics["riae_diag_max"] = float(np.max(diag_riae))

    offdiag_mask = ~np.eye(true_psd_real.shape[1], dtype=bool)
    diff_offdiag = np.linalg.norm(
        (q50_real - true_psd_real)[:, offdiag_mask], axis=1
    )
    true_offdiag = np.linalg.norm(true_psd_real[:, offdiag_mask], axis=1)
    offdiag_num = float(simpson(diff_offdiag, x=freqs))
    offdiag_den = float(simpson(true_offdiag, x=freqs))
    if offdiag_den != 0:
        metrics["riae_offdiag"] = offdiag_num / offdiag_den

    if percentiles.size >= 3:
        q05_real = extract_percentile(psd_real, percentiles, 5.0)
        q95_real = extract_percentile(psd_real, percentiles, 95.0)
        metrics["riae_matrix_p05"] = compute_matrix_riae(
            q05_real, true_psd_real, freqs
        )
        metrics["riae_matrix_p95"] = compute_matrix_riae(
            q95_real, true_psd_real, freqs
        )

    # Coherence diagnostics on off-diagonal structure
    coherence_est = _coherence(q50_real)
    coherence_true = _coherence(true_psd_real)
    coh_diff = np.linalg.norm(
        (coherence_est - coherence_true)[:, offdiag_mask], axis=1
    )
    coh_true = np.linalg.norm(coherence_true[:, offdiag_mask], axis=1)
    coh_num = float(simpson(coh_diff, x=freqs))
    coh_den = float(simpson(coh_true, x=freqs))
    if coh_den != 0:
        metrics["coherence_riae"] = coh_num / coh_den

    freq_quantiles = np.quantile(freqs, [0.0, 0.25, 0.5, 0.75, 1.0])
    freq_edges = np.unique(freq_quantiles)
    if freq_edges.size >= 2:
        band_values = []
        for start, end in zip(freq_edges[:-1], freq_edges[1:]):
            mask = (freqs >= start) & (freqs <= end)
            if np.count_nonzero(mask) < 2 or end <= start:
                continue
            riae_band = compute_matrix_riae(
                q50_real[mask], true_psd_real[mask], freqs[mask]
            )
            band_values.append(riae_band)
        if band_values:
            metrics["riae_band_max"] = float(np.max(band_values))
            metrics["riae_band_mean"] = float(np.mean(band_values))

    if percentiles.size >= 3:
        q05_im = (
            extract_percentile(psd_imag, percentiles, 5.0)
            if psd_imag is not None
            else np.zeros_like(q50_real)
        )
        q95_im = (
            extract_percentile(psd_imag, percentiles, 95.0)
            if psd_imag is not None
            else np.zeros_like(q50_real)
        )
        q50_im = (
            extract_percentile(psd_imag, percentiles, 50.0)
            if psd_imag is not None
            else np.zeros_like(q50_real)
        )
        percentiles_stack = np.stack(
            [
                q05_real + 1j * q05_im,
                q50_real + 1j * q50_im,
                q95_real + 1j * q95_im,
            ],
            axis=0,
        )
        metrics["coverage"] = compute_ci_coverage_multivar(
            percentiles_stack, true_psd_real
        )

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
    """Compute PSD comparison metrics against a reference spectrum."""

    reference = _parse_reference(truth if truth is not None else psd_ref)
    if reference is None:
        return {}

    psd_ds = _get_psd_dataset(idata, idata_vi)
    if psd_ds is None:
        return {}

    metrics: Dict[str, float] = {}

    if "psd" in psd_ds:
        metrics.update(_handle_univariate(psd_ds, reference))
    elif "psd_matrix_real" in psd_ds:
        metrics.update(_handle_multivariate(psd_ds, reference))

    return {
        key: float(val)
        for key, val in metrics.items()
        if val is not None and np.isfinite(val)
    }


def compute_multivar_riae_diagnostics(
    vi_psd: np.ndarray,
    true_psd_real: np.ndarray,
    freqs: np.ndarray,
    psd_quantiles: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, object]:
    """Legacy multivariate diagnostics used by VI adapters."""
    diagnostics: Dict[str, object] = {}
    coverage_interval = [5.0, 95.0]
    coverage_level = 0.90
    coverage_value: Optional[float] = None

    diagnostics["riae_matrix"] = float(
        compute_matrix_riae(vi_psd, true_psd_real, freqs)
    )

    per_channel_riae = []
    for channel_idx in range(true_psd_real.shape[1]):
        vi_diag = np.real(vi_psd[:, channel_idx, channel_idx])
        true_diag = np.real(true_psd_real[:, channel_idx, channel_idx])
        per_channel_riae.append(compute_riae(vi_diag, true_diag, freqs))
    diagnostics["riae_per_channel"] = [float(v) for v in per_channel_riae]

    offdiag_mask = ~np.eye(true_psd_real.shape[1], dtype=bool)
    diff_offdiag = np.linalg.norm(
        (vi_psd - true_psd_real)[:, offdiag_mask], axis=1
    )
    true_offdiag = np.linalg.norm(true_psd_real[:, offdiag_mask], axis=1)
    numerator_offdiag = float(simpson(diff_offdiag, x=freqs))
    denominator_offdiag = float(simpson(true_offdiag, x=freqs))
    diagnostics["riae_offdiag"] = (
        float(numerator_offdiag / denominator_offdiag)
        if denominator_offdiag != 0
        else float("nan")
    )

    vi_coh = _coherence(vi_psd)
    true_coh = _coherence(true_psd_real)
    coh_diff = np.linalg.norm((vi_coh - true_coh)[:, offdiag_mask], axis=1)
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
            vi_psd[mask], true_psd_real[mask], freqs[mask]
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

    real_quantiles = psd_quantiles.get("real") if psd_quantiles else None
    imag_quantiles = psd_quantiles.get("imag") if psd_quantiles else None
    if real_quantiles:
        q05_real = real_quantiles.get("q05")
        q50_real = real_quantiles.get("q50")
        q95_real = real_quantiles.get("q95")
        q05_imag = imag_quantiles.get("q05") if imag_quantiles else None
        q50_imag = imag_quantiles.get("q50") if imag_quantiles else None
        q95_imag = imag_quantiles.get("q95") if imag_quantiles else None

        if (
            q05_real is not None
            and q95_real is not None
            and q50_real is not None
        ):
            riae_low = compute_matrix_riae(q05_real, true_psd_real, freqs)
            riae_med = compute_matrix_riae(q50_real, true_psd_real, freqs)
            riae_high = compute_matrix_riae(q95_real, true_psd_real, freqs)
            diagnostics["riae_matrix_errorbars"] = [
                float(riae_low),
                float(riae_low),
                float(riae_med),
                float(riae_high),
                float(riae_high),
            ]

        if q05_real is not None and q95_real is not None:
            q05_im = (
                np.asarray(q05_imag)
                if q05_imag is not None
                else np.zeros_like(q05_real)
            )
            q50_im = (
                np.asarray(q50_imag)
                if q50_imag is not None
                else np.zeros_like(
                    q50_real if q50_real is not None else q05_real
                )
            )
            q95_im = (
                np.asarray(q95_imag)
                if q95_imag is not None
                else np.zeros_like(q95_real)
            )

            q50_real_array = (
                np.asarray(q50_real) if q50_real is not None else vi_psd
            )
            percentiles_stack = np.stack(
                [
                    np.asarray(q05_real) + 1j * q05_im,
                    q50_real_array + 1j * q50_im,
                    np.asarray(q95_real) + 1j * q95_im,
                ],
                axis=0,
            )
            coverage_value = compute_ci_coverage_multivar(
                percentiles_stack, true_psd_real
            )
            if np.isfinite(coverage_value):
                diagnostics["coverage"] = float(coverage_value)
                diagnostics["ci_coverage"] = float(coverage_value)
                diagnostics["coverage_interval"] = coverage_interval
                diagnostics["coverage_level"] = coverage_level

    return diagnostics
