"""Utilities for computing Rhat values safely."""

from __future__ import annotations

import arviz as az
import numpy as np

DEFAULT_RHAT_WEIGHT_POINTS = 6


def _select_weight_indices(size: int, max_points: int) -> np.ndarray:
    if size <= 0 or max_points <= 0:
        return np.array([], dtype=int)
    if size <= max_points:
        return np.arange(size, dtype=int)
    idx = np.unique(
        np.linspace(0, size - 1, num=max_points, dtype=int, endpoint=True)
    )
    return idx


def _posterior_subset_for_rhat(idata, *, drop_draws: int = 0):
    posterior = getattr(idata, "posterior", None)
    if posterior is None and hasattr(idata, "data_vars"):
        posterior = idata
    if posterior is None:
        return idata

    subset = {}
    for name, var in posterior.data_vars.items():
        var_name = str(name)
        if var_name.startswith("weights"):
            dims = [d for d in var.dims if d not in ("chain", "draw")]
            if not dims:
                subset[var_name] = var
                continue
            dim = dims[-1]
            size = int(var.sizes.get(dim, 0))
            idx = _select_weight_indices(size, DEFAULT_RHAT_WEIGHT_POINTS)
            if idx.size == 0:
                continue
            subset[var_name] = var.isel({dim: idx})
            continue
        subset[var_name] = var

    if not subset:
        return None
    ds = posterior.__class__(
        subset, coords=posterior.coords, attrs=posterior.attrs
    )
    if drop_draws > 0 and "draw" in ds.dims:
        ds = ds.isel(draw=slice(int(drop_draws), None))
    return ds


def extract_rhat_values(
    idata: az.InferenceData,
    *,
    drop_draws: int = 0,
) -> np.ndarray:
    """
    Compute Rhat values and return them as a flat array.

    This avoids the ``xarray.Dataset.to_array`` broadcasting blow-up that
    occurs when variables have disjoint dimension names (common for P-spline
    weights), by iterating over each data variable individually.
    """
    subset = _posterior_subset_for_rhat(idata, drop_draws=drop_draws)
    if subset is None:
        return np.array([])
    rhat_ds = az.rhat(subset)
    values = []
    for var in rhat_ds.data_vars.values():
        arr = np.asarray(var)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr.ravel())

    if not values:
        return np.array([])

    return np.concatenate(values)
