"""Metric extraction and compact output for LISA simulation study.

Adapted from docs/studies/multivar_psd/3d_study.py patterns.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import arviz as az
import numpy as np

from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.diagnostics import psd_compare
from log_psplines.logger import logger


def _extract_percentile_slice(
    values: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    """Return percentile slice nearest to target."""
    idx = int(np.argmin(np.abs(percentiles - target)))
    return np.asarray(values[idx], dtype=np.float64)


def _compute_ci_width_metrics(idata) -> dict[str, float]:
    """Compute CI-width summaries from posterior PSD quantiles."""
    metrics: dict[str, float] = {}
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is None or "psd_matrix_real" not in psd_group:
        return metrics

    psd_real = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    percentiles = np.asarray(
        psd_group["psd_matrix_real"].coords.get(
            "percentile", np.arange(psd_real.shape[0], dtype=float)
        ),
        dtype=np.float64,
    )
    if psd_real.shape[0] < 2:
        return metrics

    q05 = _extract_percentile_slice(psd_real, percentiles, 5.0)
    q95 = _extract_percentile_slice(psd_real, percentiles, 95.0)
    width_psd = np.maximum(q95 - q05, 0.0)

    p = width_psd.shape[1]
    diag_idx = np.arange(p)
    offdiag_mask = ~np.eye(p, dtype=bool)
    diag_width = width_psd[:, diag_idx, diag_idx]
    offdiag_width = width_psd[:, offdiag_mask]

    metrics["ciw_psd_diag_mean"] = float(np.mean(diag_width))
    metrics["ciw_psd_diag_median"] = float(np.median(diag_width))
    metrics["ciw_psd_diag_max"] = float(np.max(diag_width))
    metrics["ciw_psd_offdiag_mean"] = float(np.mean(offdiag_width))
    metrics["ciw_psd_offdiag_median"] = float(np.median(offdiag_width))
    metrics["ciw_psd_offdiag_max"] = float(np.max(offdiag_width))

    if "coherence" in psd_group:
        coherence = np.asarray(psd_group["coherence"].values, dtype=np.float64)
        coh_percentiles = np.asarray(
            psd_group["coherence"].coords.get(
                "percentile", np.arange(coherence.shape[0], dtype=float)
            ),
            dtype=np.float64,
        )
        if coherence.shape[0] >= 2:
            coh_q05 = _extract_percentile_slice(
                coherence, coh_percentiles, 5.0
            )
            coh_q95 = _extract_percentile_slice(
                coherence, coh_percentiles, 95.0
            )
            coh_width = np.maximum(coh_q95 - coh_q05, 0.0)
            coh_offdiag = coh_width[:, offdiag_mask]
            metrics["ciw_coh_offdiag_mean"] = float(np.mean(coh_offdiag))
            metrics["ciw_coh_offdiag_median"] = float(np.median(coh_offdiag))
            metrics["ciw_coh_offdiag_max"] = float(np.max(coh_offdiag))

    return metrics


def _extract_run_metrics(
    idata,
    *,
    seed: int,
    freq_true: np.ndarray,
    S_true: np.ndarray,
) -> dict[str, float | int | str]:
    """Extract compact scalar metrics for downstream aggregation."""
    attrs = idata.attrs

    # ESS
    ess_raw = attrs.get("ess", np.nan)
    ess_arr = np.asarray(ess_raw, dtype=float)
    ess_median = float(np.nanmedian(ess_arr)) if ess_arr.size else float("nan")

    # Rhat
    rhat_raw = attrs.get("rhat", np.nan)
    rhat_arr = np.asarray(rhat_raw, dtype=float)
    rhat_max = float(np.nanmax(rhat_arr)) if rhat_arr.size else float("nan")

    # Divergences
    n_div = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_div = int(np.sum(idata.sample_stats["diverging"].values))

    metrics: dict[str, float | int | str] = {
        "seed": int(seed),
        "ess_median": ess_median,
        "rhat_max": rhat_max,
        "n_divergences": n_div,
        "runtime": float(attrs.get("runtime", np.nan)),
    }

    # PSD accuracy metrics
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is not None:
        freq_plot = np.asarray(psd_group["freq"].values)
        true_psd_phys = interp_matrix(
            np.asarray(freq_true), np.asarray(S_true), freq_plot
        )
        try:
            acc = psd_compare._handle_multivariate(psd_group, true_psd_phys)
            metrics["riae_matrix"] = float(acc.get("riae_matrix", np.nan))
            metrics["riae_diag_mean"] = float(
                acc.get("riae_diag_mean", np.nan)
            )
            metrics["riae_offdiag"] = float(acc.get("riae_offdiag", np.nan))
            metrics["coherence_riae"] = float(
                acc.get("coherence_riae", np.nan)
            )
            metrics["coverage"] = float(acc.get("coverage", np.nan))
        except Exception as exc:
            logger.warning(f"Could not compute PSD accuracy: {exc}")
            metrics["riae_matrix"] = float("nan")
            metrics["coverage"] = float("nan")

    # CI width metrics
    metrics.update(_compute_ci_width_metrics(idata))

    return metrics


def _save_compact_ci_curves(outdir: str, idata, freq_true, S_true) -> None:
    """Save compact CI-vs-frequency arrays for later visualization."""
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is None:
        logger.warning("No posterior_psd group; skipping CI curves.")
        return
    if (
        "psd_matrix_real" not in psd_group
        or "psd_matrix_imag" not in psd_group
    ):
        logger.warning("Missing psd_matrix_real/imag; skipping CI curves.")
        return

    freq = np.asarray(psd_group.coords["freq"].values, dtype=np.float64)
    percentiles = np.asarray(
        psd_group.coords["percentile"].values, dtype=np.float64
    )

    psd_real = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    psd_imag = np.asarray(
        psd_group["psd_matrix_imag"].values, dtype=np.float64
    )

    q05_real = _extract_percentile_slice(psd_real, percentiles, 5.0)
    q50_real = _extract_percentile_slice(psd_real, percentiles, 50.0)
    q95_real = _extract_percentile_slice(psd_real, percentiles, 95.0)

    q05_imag = _extract_percentile_slice(psd_imag, percentiles, 5.0)
    q50_imag = _extract_percentile_slice(psd_imag, percentiles, 50.0)
    q95_imag = _extract_percentile_slice(psd_imag, percentiles, 95.0)

    _, p, _ = q50_real.shape
    offdiag_pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]

    # Interpolate true PSD to model frequencies.
    true_psd = interp_matrix(np.asarray(freq_true), np.asarray(S_true), freq)
    true_psd_real = np.real(true_psd).astype(np.float64, copy=False)
    true_psd_imag = np.imag(true_psd).astype(np.float64, copy=False)

    save_payload = dict(
        freq=freq,
        psd_real_q05=q05_real,
        psd_real_q50=q50_real,
        psd_real_q95=q95_real,
        psd_imag_q05=q05_imag,
        psd_imag_q50=q50_imag,
        psd_imag_q95=q95_imag,
        offdiag_pairs=np.asarray(offdiag_pairs, dtype=int),
        true_psd_real=true_psd_real,
        true_psd_imag=true_psd_imag,
    )

    outpath = os.path.join(outdir, "compact_ci_curves.npz")
    np.savez_compressed(outpath, **save_payload)
    logger.info(f"Saved compact CI curves to {outpath}")


def extract_and_save_metrics(
    idata,
    *,
    seed: int,
    freq_true: np.ndarray,
    S_true: np.ndarray,
    outdir: str,
) -> dict[str, float | int | str]:
    """Extract all metrics, save JSON/CSV/NPZ, return metrics dict."""
    os.makedirs(outdir, exist_ok=True)

    metrics = _extract_run_metrics(
        idata, seed=seed, freq_true=freq_true, S_true=S_true
    )

    # Save JSON
    out_json = os.path.join(outdir, "compact_run_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # Save CSV
    out_csv = os.path.join(outdir, "compact_run_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"Saved compact run summary to {out_json}")

    # Save CI curves
    _save_compact_ci_curves(outdir, idata, freq_true, S_true)

    return metrics
