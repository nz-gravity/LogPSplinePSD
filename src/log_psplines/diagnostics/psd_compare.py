"""PSD accuracy diagnostics (RIAE, coverage, coherence)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import simpson

from ..arviz_utils.from_arviz import get_psd_dataset
from ._utils import (
    compute_ci_coverage_multivar,
    compute_ci_coverage_multivar_detailed,
    compute_ci_coverage_univar,
    compute_coherence_coverage,
    compute_matrix_l2,
    compute_matrix_riae,
    compute_riae,
    extract_percentile,
    interior_frequency_slice,
)

PSD_PERCENTILES = np.asarray([5.0, 50.0, 95.0], dtype=float)


def _spectral_density_samples(
    psd_ds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(freqs, spectral_density_samples, coherence_samples)``."""
    freqs = np.asarray(psd_ds.coords["frequency"].values, dtype=float)
    spectral_density = np.asarray(psd_ds["spectral_density"].values).reshape(
        -1,
        psd_ds["spectral_density"].shape[2],
        psd_ds["spectral_density"].shape[3],
        psd_ds["spectral_density"].shape[4],
    )
    coherence_samples = None
    if "coherence" in psd_ds:
        coherence_samples = np.asarray(psd_ds["coherence"].values).reshape(
            -1,
            psd_ds["coherence"].shape[2],
            psd_ds["coherence"].shape[3],
            psd_ds["coherence"].shape[4],
        )
    return freqs, spectral_density, coherence_samples


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
    freqs, spectral_density, _ = _spectral_density_samples(psd_ds)
    values = np.percentile(
        np.real(spectral_density[:, 0, 0, :]),
        PSD_PERCENTILES,
        axis=0,
    )

    freq_idx = interior_frequency_slice(freqs.size)
    freqs = freqs[freq_idx]
    values = values[..., freq_idx]
    reference = np.asarray(reference)[freq_idx]

    metrics: Dict[str, float] = {}

    q50 = extract_percentile(values, PSD_PERCENTILES, 50.0)
    metrics["riae"] = compute_riae(q50, reference, freqs)

    q05 = extract_percentile(values, PSD_PERCENTILES, 5.0)
    q95 = extract_percentile(values, PSD_PERCENTILES, 95.0)
    metrics["riae_p05"] = compute_riae(q05, reference, freqs)
    metrics["riae_p95"] = compute_riae(q95, reference, freqs)
    metrics["coverage"] = compute_ci_coverage_univar(values, reference)
    metrics["ci_width"] = float(np.mean(q95 - q05))

    return metrics


def _handle_univariate_no_truth(psd_ds) -> Dict[str, float]:
    freqs, spectral_density, _ = _spectral_density_samples(psd_ds)
    values = np.percentile(
        np.real(spectral_density[:, 0, 0, :]),
        PSD_PERCENTILES,
        axis=0,
    )

    freq_idx = interior_frequency_slice(freqs.size)
    values = values[..., freq_idx]

    metrics: Dict[str, float] = {}
    q05 = extract_percentile(values, PSD_PERCENTILES, 5.0)
    q95 = extract_percentile(values, PSD_PERCENTILES, 95.0)
    metrics["ci_width"] = float(np.mean(q95 - q05))
    return metrics


def _handle_multivariate(psd_ds, reference: np.ndarray) -> Dict[str, float]:
    freqs, spectral_density, coherence_samples = _spectral_density_samples(
        psd_ds
    )
    spectral_density = np.moveaxis(spectral_density, -1, 1)
    psd_real = np.percentile(spectral_density.real, PSD_PERCENTILES, axis=0)
    psd_imag = np.percentile(spectral_density.imag, PSD_PERCENTILES, axis=0)
    coherence_quantiles = None
    if coherence_samples is not None:
        coherence_samples = np.moveaxis(coherence_samples, -1, 1)
        coherence_quantiles = np.percentile(
            coherence_samples, PSD_PERCENTILES, axis=0
        )

    freq_idx = interior_frequency_slice(freqs.size)
    freqs = freqs[freq_idx]
    psd_real = psd_real[:, freq_idx, ...]
    if psd_imag is not None:
        psd_imag = psd_imag[:, freq_idx, ...]
    if coherence_quantiles is not None:
        coherence_quantiles = coherence_quantiles[:, freq_idx, ...]

    true_psd_real = np.asarray(reference)[freq_idx, ...]

    metrics: Dict[str, float] = {}

    q50_real = extract_percentile(psd_real, PSD_PERCENTILES, 50.0)
    metrics["riae_matrix"] = compute_matrix_riae(
        q50_real, true_psd_real, freqs
    )
    metrics["l2_matrix"] = compute_matrix_l2(q50_real, true_psd_real, freqs)

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

    p = true_psd_real.shape[1]
    offdiag_mask = ~np.eye(p, dtype=bool)
    upper_tri_mask = np.triu(np.ones((p, p), dtype=bool), k=1)
    lower_tri_mask = np.tril(np.ones((p, p), dtype=bool), k=-1)

    # Cast truth to complex so Re/Im decomposition is always well-defined.
    # For real-valued true PSDs (e.g. LISA), Im parts are zero — that is correct.
    true_psd_complex = np.asarray(true_psd_real).astype(complex)

    # --- off-diagonal RIAE (norm over all off-diagonal, existing metric) ---
    diff_offdiag = np.linalg.norm(
        (q50_real - np.real(true_psd_complex))[:, offdiag_mask], axis=1
    )
    true_offdiag = np.linalg.norm(
        np.real(true_psd_complex)[:, offdiag_mask], axis=1
    )
    offdiag_num = float(simpson(diff_offdiag, x=freqs))
    offdiag_den = float(simpson(true_offdiag, x=freqs))
    if offdiag_den != 0:
        metrics["riae_offdiag"] = offdiag_num / offdiag_den

    q05_real = extract_percentile(psd_real, PSD_PERCENTILES, 5.0)
    q95_real = extract_percentile(psd_real, PSD_PERCENTILES, 95.0)
    metrics["riae_matrix_p05"] = compute_matrix_riae(
        q05_real, np.real(true_psd_complex), freqs
    )
    metrics["riae_matrix_p95"] = compute_matrix_riae(
        q95_real, np.real(true_psd_complex), freqs
    )
    diag_mask_2d = np.eye(p, dtype=bool)
    diag_width = (q95_real - q05_real)[:, diag_mask_2d]
    metrics["ci_width_diag_mean"] = float(np.mean(diag_width))
    metrics["ci_width"] = metrics["ci_width_diag_mean"]

    # --- RIAE: Re and Im off-diagonal separately ---
    q50_im = (
        extract_percentile(psd_imag, PSD_PERCENTILES, 50.0)
        if psd_imag is not None
        else np.zeros_like(q50_real)
    )
    true_re_upper = np.real(true_psd_complex)[:, upper_tri_mask]
    true_im_upper = np.imag(true_psd_complex)[:, upper_tri_mask]
    est_re_upper = q50_real[:, upper_tri_mask]
    est_im_upper = q50_im[:, upper_tri_mask]

    re_num = float(
        simpson(np.linalg.norm(est_re_upper - true_re_upper, axis=1), x=freqs)
    )
    re_den = float(simpson(np.linalg.norm(true_re_upper, axis=1), x=freqs))
    metrics["riae_offdiag_re"] = (
        re_num / re_den if re_den != 0 else float("nan")
    )

    im_den = float(simpson(np.linalg.norm(true_im_upper, axis=1), x=freqs))
    if im_den != 0:
        im_num = float(
            simpson(
                np.linalg.norm(est_im_upper - true_im_upper, axis=1), x=freqs
            )
        )
        metrics["riae_offdiag_im"] = im_num / im_den
    # If Im true is ~0 (e.g. LISA), riae_offdiag_im is undefined — leave absent.

    # --- Coherence RIAE ---
    coherence_est = (
        extract_percentile(coherence_quantiles, PSD_PERCENTILES, 50.0)
        if coherence_quantiles is not None
        else _coherence(q50_real)
    )
    coherence_true = _coherence(true_psd_complex)
    coh_diff = np.linalg.norm(
        (coherence_est - coherence_true)[:, offdiag_mask], axis=1
    )
    coh_true_norm = np.linalg.norm(coherence_true[:, offdiag_mask], axis=1)
    coh_num = float(simpson(coh_diff, x=freqs))
    coh_den = float(simpson(coh_true_norm, x=freqs))
    if coh_den != 0:
        metrics["coherence_riae"] = coh_num / coh_den

    # --- Frequency-band RIAE ---
    freq_quantiles = np.quantile(freqs, [0.0, 0.25, 0.5, 0.75, 1.0])
    freq_edges = np.unique(freq_quantiles)
    if freq_edges.size >= 2:
        band_values = []
        for start, end in zip(freq_edges[:-1], freq_edges[1:]):
            mask = (freqs >= start) & (freqs <= end)
            if np.count_nonzero(mask) < 2 or end <= start:
                continue
            riae_band = compute_matrix_riae(
                q50_real[mask], np.real(true_psd_complex)[mask], freqs[mask]
            )
            band_values.append(riae_band)
        if band_values:
            metrics["riae_band_max"] = float(np.max(band_values))
            metrics["riae_band_mean"] = float(np.mean(band_values))

    # --- Coverage breakdown ---
    q05_im = (
        extract_percentile(psd_imag, PSD_PERCENTILES, 5.0)
        if psd_imag is not None
        else np.zeros_like(q05_real)
    )
    q95_im = (
        extract_percentile(psd_imag, PSD_PERCENTILES, 95.0)
        if psd_imag is not None
        else np.zeros_like(q95_real)
    )
    percentiles_stack = np.stack(
        [
            q05_real + 1j * q05_im,
            q50_real + 1j * q50_im,
            q95_real + 1j * q95_im,
        ],
        axis=0,
    )
    # Use complex truth so _complex_to_real encodes Im parts correctly.
    detail = compute_ci_coverage_multivar_detailed(
        percentiles_stack, true_psd_complex
    )
    metrics["coverage"] = detail["overall"]
    metrics["coverage_diag"] = detail["diag"]
    metrics["coverage_offdiag_re"] = detail["offdiag_re"]
    metrics["coverage_offdiag_im"] = detail["offdiag_im"]

    # Coherence coverage from sample-derived coherence quantiles.
    if coherence_quantiles is not None:
        metrics["coverage_coherence"] = compute_coherence_coverage(
            coherence_quantiles,
            true_psd_complex,
            PSD_PERCENTILES,
        )

    return metrics


def _handle_multivariate_no_truth(psd_ds) -> Dict[str, float]:
    freqs, spectral_density, _ = _spectral_density_samples(psd_ds)
    spectral_density = np.moveaxis(spectral_density, -1, 1)
    psd_real = np.percentile(spectral_density.real, PSD_PERCENTILES, axis=0)
    freq_idx = interior_frequency_slice(freqs.size)
    psd_real = psd_real[:, freq_idx, ...]

    metrics: Dict[str, float] = {}
    q05_real = extract_percentile(psd_real, PSD_PERCENTILES, 5.0)
    q95_real = extract_percentile(psd_real, PSD_PERCENTILES, 95.0)
    diag_mask = np.eye(q05_real.shape[1], dtype=bool)
    diag_width = (q95_real - q05_real)[:, diag_mask]
    metrics["ci_width_diag_mean"] = float(np.mean(diag_width))
    metrics["ci_width"] = metrics["ci_width_diag_mean"]
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
    psd_ds = None
    for source in (idata, idata_vi):
        if source is None:
            continue
        try:
            psd_ds = get_psd_dataset(source, source="best")
            break
        except (KeyError, TypeError, ValueError, StopIteration):
            continue
    if psd_ds is None:
        return {}

    metrics: Dict[str, float] = {}
    n_channels = int(psd_ds["spectral_density"].shape[2])

    if reference is None:
        if n_channels == 1:
            metrics.update(_handle_univariate_no_truth(psd_ds))
        else:
            metrics.update(_handle_multivariate_no_truth(psd_ds))
    else:
        if n_channels == 1:
            metrics.update(_handle_univariate(psd_ds, reference))
        else:
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
    freq_idx = interior_frequency_slice(np.asarray(freqs).size)
    freqs = np.asarray(freqs)[freq_idx]
    vi_psd = np.asarray(vi_psd)[freq_idx, ...]
    true_psd_real = np.asarray(true_psd_real)[freq_idx, ...]

    diagnostics: Dict[str, object] = {}
    coverage_interval = [5.0, 95.0]
    coverage_level = 0.90
    coverage_value: Optional[float] = None

    diagnostics["riae_matrix"] = float(
        compute_matrix_riae(vi_psd, true_psd_real, freqs)
    )
    diagnostics["l2_matrix"] = float(
        compute_matrix_l2(vi_psd, true_psd_real, freqs)
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
            q05_real_arr = np.asarray(q05_real)[freq_idx, ...]
            q50_real_arr = np.asarray(q50_real)[freq_idx, ...]
            q95_real_arr = np.asarray(q95_real)[freq_idx, ...]
            riae_low = compute_matrix_riae(q05_real_arr, true_psd_real, freqs)
            riae_med = compute_matrix_riae(q50_real_arr, true_psd_real, freqs)
            riae_high = compute_matrix_riae(q95_real_arr, true_psd_real, freqs)
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
                np.asarray(q50_real)[freq_idx, ...]
                if q50_real is not None
                else vi_psd
            )
            diag_mask_2d = np.eye(q50_real_array.shape[1], dtype=bool)
            diag_width = (
                np.asarray(q95_real)[freq_idx, ...]
                - np.asarray(q05_real)[freq_idx, ...]
            )[:, diag_mask_2d]
            diagnostics["ci_width_diag_mean"] = float(np.mean(diag_width))
            diagnostics["ci_width"] = diagnostics["ci_width_diag_mean"]
            q05_im = q05_im[freq_idx, ...]
            q50_im = q50_im[freq_idx, ...]
            q95_im = q95_im[freq_idx, ...]
            percentiles_stack = np.stack(
                [
                    np.asarray(q05_real)[freq_idx, ...] + 1j * q05_im,
                    q50_real_array + 1j * q50_im,
                    np.asarray(q95_real)[freq_idx, ...] + 1j * q95_im,
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
