"""Small helpers for working with canonical ``xarray.DataTree`` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def require_dataset(idata: xr.DataTree, group: str) -> xr.Dataset:
    """Return the dataset stored in ``group`` or raise a helpful error."""
    try:
        candidate = idata[group]
    except Exception:
        candidate = getattr(idata, group, None)
    if candidate is None:
        raise KeyError(f"DataTree missing required group '{group}'.")
    dataset = getattr(candidate, "dataset", None)
    if dataset is None and isinstance(candidate, xr.Dataset):
        dataset = candidate
    if dataset is None:
        raise TypeError(f"DataTree group '{group}' must contain a dataset.")
    return dataset


def select_draw_slice(idata: xr.DataTree, draw_slice: slice) -> xr.DataTree:
    """Return a shallow copy of the tree with the given draw slice applied."""
    out = xr.DataTree()
    out.attrs.update(dict(idata.attrs))
    for name, group in idata.children.items():
        dataset = group.dataset
        if dataset is None:
            out[name] = group
            continue
        if "draw" in dataset.dims:
            dataset = dataset.sel(draw=draw_slice)
        out[name] = xr.DataTree(dataset=dataset)
    return out


def save_inference_data(
    idata: xr.DataTree, path: str | Path, *, engine: str = "h5netcdf"
) -> None:
    """Save inference data using the canonical grouped NetCDF layout."""
    _sanitize_attrs_for_netcdf(idata).to_netcdf(Path(path), engine=engine)


def _netcdf_safe_attr(value: Any) -> Any:
    """Return a NetCDF-compatible attribute value, or ``None`` to omit it."""
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, str | int | float | np.integer | np.floating):
        return value.item() if hasattr(value, "item") else value

    arr = np.asarray(value)
    if arr.dtype == object or np.iscomplexobj(arr):
        return str(value)
    if arr.dtype == bool:
        return arr.astype(np.int8)
    return value


def _safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in attrs.items():
        safe_value = _netcdf_safe_attr(value)
        if safe_value is not None:
            safe[key] = safe_value
    return safe


def _sanitize_attrs_for_netcdf(idata: xr.DataTree) -> xr.DataTree:
    """Return a shallow DataTree copy with NetCDF-compatible attrs."""
    out = xr.DataTree()
    out.attrs.update(_safe_attrs(dict(idata.attrs)))

    if idata.dataset is not None:
        dataset = idata.dataset.copy(deep=False)
        dataset.attrs = _safe_attrs(dict(dataset.attrs))
        out.dataset = dataset

    for name, child in idata.children.items():
        out[name] = _sanitize_attrs_for_netcdf(child)
    return out


def open_inference_data(
    path: str | Path, *, engine: str = "h5netcdf"
) -> xr.DataTree:
    """Load inference data saved by ``save_inference_data``."""
    return xr.open_datatree(Path(path), engine=engine)
