from __future__ import annotations

"""Small helpers for working with canonical ``xarray.DataTree`` objects."""

from pathlib import Path

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
    """Return a shallow copy with the given draw slice applied to each group."""
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
    idata.to_netcdf(Path(path), engine=engine)


def open_inference_data(
    path: str | Path, *, engine: str = "h5netcdf"
) -> xr.DataTree:
    """Load inference data saved by ``save_inference_data``."""
    return xr.open_datatree(Path(path), engine=engine)
