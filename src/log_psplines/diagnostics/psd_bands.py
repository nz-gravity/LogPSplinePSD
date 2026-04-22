"""Credible-band summaries for PSDs."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.integrate import simpson

from ._utils import extract_percentile
from .psd_compare import _get_psd_dataset


def _variance_from_psd(psd_values: np.ndarray, freqs: np.ndarray) -> float:
    return float(simpson(psd_values, x=freqs))


def _run(
    *,
    idata=None,
    config=None,
    truth=None,
    signals=None,
    psd_ref=None,
    idata_vi=None,
) -> Dict[str, float]:
    """Summaries of PSD credible bands and coherence."""
    psd_ds = _get_psd_dataset(idata, idata_vi)
    if psd_ds is None:
        return {}

    metrics: Dict[str, float] = {}

    if "spectral_density" in psd_ds:
        freqs = np.asarray(psd_ds.coords["frequency"].values, dtype=float)
        spectral_density = np.asarray(psd_ds["spectral_density"].values)
        spectral_density = spectral_density.reshape(
            -1,
            spectral_density.shape[2],
            spectral_density.shape[3],
            spectral_density.shape[4],
        )
        diag = np.real(
            spectral_density[
                :,
                np.arange(spectral_density.shape[1]),
                np.arange(spectral_density.shape[1]),
                :,
            ]
        )

        if diag.shape[1] == 1:
            q50 = np.percentile(diag[:, 0, :], 50.0, axis=0)
            metrics["variance_median"] = _variance_from_psd(q50, freqs)
            q05 = np.percentile(diag[:, 0, :], 5.0, axis=0)
            q95 = np.percentile(diag[:, 0, :], 95.0, axis=0)
            metrics["variance_ci_width"] = float(
                _variance_from_psd(q95, freqs) - _variance_from_psd(q05, freqs)
            )
        else:
            q50 = np.percentile(diag, 50.0, axis=0)
            variances = simpson(q50, x=freqs, axis=-1)
            metrics["variance_median_mean"] = float(np.mean(variances))
            q05 = np.percentile(diag, 5.0, axis=0)
            q95 = np.percentile(diag, 95.0, axis=0)
            var_low = simpson(q05, x=freqs, axis=-1)
            var_high = simpson(q95, x=freqs, axis=-1)
            metrics["variance_ci_width_mean"] = float(
                np.mean(var_high - var_low)
            )
            coherence = np.asarray(psd_ds["coherence"].values).reshape(
                -1,
                spectral_density.shape[1],
                spectral_density.shape[2],
                spectral_density.shape[3],
            )
            coh_q50 = np.percentile(coherence, 50.0, axis=0)
            metrics["coherence_median_max"] = float(np.nanmax(np.abs(coh_q50)))

    elif "psd" in psd_ds:
        psd = psd_ds["psd"]
        freqs = np.asarray(psd.coords.get("freq", np.arange(psd.shape[-1])))
        percentiles = np.asarray(psd.coords.get("percentile", []), dtype=float)
        values = np.asarray(psd.values)
        if percentiles.size == 0:
            percentiles = np.arange(values.shape[0], dtype=float)

        q50 = extract_percentile(values, percentiles, 50.0)
        metrics["variance_median"] = _variance_from_psd(q50, freqs)

        if percentiles.size >= 3:
            q05 = extract_percentile(values, percentiles, 5.0)
            q95 = extract_percentile(values, percentiles, 95.0)
            low = _variance_from_psd(q05, freqs)
            high = _variance_from_psd(q95, freqs)
            metrics["variance_ci_width"] = float(high - low)

    elif "psd_matrix_real" in psd_ds:
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
        metrics["variance_median_mean"] = float(np.mean(variances))

        if percentiles.size >= 3:
            q05 = extract_percentile(values, percentiles, 5.0)
            q95 = extract_percentile(values, percentiles, 95.0)
            var_low = simpson(
                np.real(
                    q05[:, np.arange(q05.shape[1]), np.arange(q05.shape[2])]
                ),
                x=freqs,
                axis=0,
            )
            var_high = simpson(
                np.real(
                    q95[:, np.arange(q95.shape[1]), np.arange(q95.shape[2])]
                ),
                x=freqs,
                axis=0,
            )
            metrics["variance_ci_width_mean"] = float(
                np.mean(var_high - var_low)
            )

        if "coherence" in psd_ds:
            coh = psd_ds["coherence"]
            coh_percentiles = np.asarray(coh.coords.get("percentile", []))
            if coh_percentiles.size:
                coh_q50 = extract_percentile(
                    np.asarray(coh.values), coh_percentiles, 50.0
                )
                metrics["coherence_median_max"] = float(
                    np.nanmax(np.abs(coh_q50))
                )

    return {
        key: float(val)
        for key, val in metrics.items()
        if val is not None and np.isfinite(val)
    }
