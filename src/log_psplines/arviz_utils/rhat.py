"""Utilities for computing Rhat values safely."""

from __future__ import annotations

import arviz as az
import numpy as np


def extract_rhat_values(idata: az.InferenceData) -> np.ndarray:
    """
    Compute Rhat values and return them as a flat array.

    This avoids the ``xarray.Dataset.to_array`` broadcasting blow-up that
    occurs when variables have disjoint dimension names (common for P-spline
    weights), by iterating over each data variable individually.
    """
    rhat_ds = az.rhat(idata)
    values = []
    for var in rhat_ds.data_vars.values():
        arr = np.asarray(var)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr.ravel())

    if not values:
        return np.array([])

    return np.concatenate(values)
