"""ArviZ interoperability for sampling diagnostics only."""

from __future__ import annotations

import arviz as az
import xarray as xr


def to_arviz(result) -> az.InferenceData:
    """Build a minimal ArviZ view from a PSDResult.

    Only posterior samples and sampler statistics are exposed. Spectral
    reconstruction, model storage and persistence remain LogPSplinePSD
    responsibilities.
    """
    groups: dict[str, xr.Dataset] = {"posterior": result.posterior}
    if result.sample_stats is not None:
        groups["sample_stats"] = result.sample_stats
    return az.InferenceData(**groups)


__all__ = ["to_arviz"]
