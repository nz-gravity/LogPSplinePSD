"""Unified entry point for diagnostic computations."""

from __future__ import annotations

from typing import Dict

from . import energy, mcmc, psd_bands, psd_compare, time_domain, vi, whitening


def run_all_diagnostics(
    *,
    idata=None,
    config=None,
    truth=None,
    signals=None,
    psd_ref=None,
    idata_vi=None,
) -> Dict[str, Dict[str, float]]:
    """Execute available diagnostics and group results by module name."""
    results: Dict[str, Dict[str, float]] = {}

    if idata is not None:
        metrics = mcmc.run(
            idata=idata,
            config=config,
            truth=truth,
            signals=signals,
            psd_ref=psd_ref,
            idata_vi=idata_vi,
        )
        if metrics:
            results["mcmc"] = metrics

        energy_metrics = energy.run(
            idata=idata,
            config=config,
            truth=truth,
            signals=signals,
            psd_ref=psd_ref,
            idata_vi=idata_vi,
        )
        if energy_metrics:
            results["energy"] = energy_metrics

    if truth is not None or psd_ref is not None:
        compare_metrics = psd_compare.run(
            idata=idata,
            idata_vi=idata_vi,
            config=config,
            truth=truth,
            signals=signals,
            psd_ref=psd_ref,
        )
        if compare_metrics:
            results["psd_compare"] = compare_metrics

        band_metrics = psd_bands.run(
            idata=idata,
            idata_vi=idata_vi,
            config=config,
            truth=truth,
            signals=signals,
            psd_ref=psd_ref,
        )
        if band_metrics:
            results["psd_bands"] = band_metrics

    if signals is not None:
        td_metrics = time_domain.run(
            idata=idata,
            idata_vi=idata_vi,
            config=config,
            truth=truth,
            signals=signals,
            psd_ref=psd_ref,
        )
        if td_metrics:
            results["time_domain"] = td_metrics

        whitening_metrics = whitening.run(
            idata=idata,
            idata_vi=idata_vi,
            config=config,
            truth=truth,
            signals=signals,
            psd_ref=psd_ref,
        )
        if whitening_metrics:
            results["whitening"] = whitening_metrics

    if idata_vi is not None:
        vi_metrics = vi.run(
            idata=idata,
            idata_vi=idata_vi,
            config=config,
            truth=truth,
            signals=signals,
            psd_ref=psd_ref,
        )
        if vi_metrics:
            results["vi"] = vi_metrics

    return results
