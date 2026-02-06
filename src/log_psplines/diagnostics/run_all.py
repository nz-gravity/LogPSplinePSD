"""Unified entry point for diagnostic computations."""

from __future__ import annotations

from typing import Callable, Dict

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
    context = {
        "idata": idata,
        "config": config,
        "truth": truth,
        "signals": signals,
        "psd_ref": psd_ref,
        "idata_vi": idata_vi,
    }

    def _has_psd() -> bool:
        return psd_compare._get_psd_dataset(idata, idata_vi) is not None

    rules: list[
        tuple[str, Callable[..., Dict[str, float]], Callable[[], bool]]
    ] = [
        ("mcmc", mcmc._run, lambda: idata is not None),
        ("energy", energy._run, lambda: idata is not None),
        (
            "psd_compare",
            psd_compare._run,
            lambda: truth is not None or psd_ref is not None,
        ),
        (
            "psd_bands",
            psd_bands._run,
            _has_psd,
        ),
        ("time_domain", time_domain._run, lambda: signals is not None),
        ("whitening", whitening._run, lambda: signals is not None),
        ("vi", vi._run, lambda: idata_vi is not None),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for name, fn, predicate in rules:
        if not predicate():
            continue
        metrics = fn(**context)
        if metrics:
            results[name] = metrics

    return results
