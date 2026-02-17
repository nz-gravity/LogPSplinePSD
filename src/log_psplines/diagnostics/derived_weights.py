"""Derived weight diagnostics computed on demand for plotting."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import arviz as az
import numpy as np
import xarray as xr

REP_WEIGHT_MAX = 10
REP_WEIGHT_ESS_K = 3
REP_WEIGHT_VAR_K = 3
REP_WEIGHT_EVEN_K = 3
HDI_PROB = 0.94


def find_weight_vars(posterior: xr.Dataset) -> List[str]:
    if posterior is None:
        return []
    return [
        str(name)
        for name in posterior.data_vars
        if str(name).startswith("weights")
    ]


def _basis_dim(var: xr.DataArray) -> str | None:
    if var is None or not hasattr(var, "dims"):
        return None
    dims = [d for d in var.dims if d not in ("chain", "draw")]
    return str(dims[-1]) if dims else None


def _get_penalty_matrix(idata, weight_name: str) -> np.ndarray | None:
    spline_model = getattr(idata, "spline_model", None)
    if spline_model is None:
        return None

    key = None
    if weight_name == "weights":
        key = "penalty_matrix"
    elif weight_name.startswith("weights_delta_"):
        suffix = weight_name.split("weights_delta_", 1)[-1]
        key = f"diag_{suffix}_penalty_matrix"
    elif weight_name.startswith("weights_theta_re"):
        key = "offdiag_re_penalty_matrix"
    elif weight_name.startswith("weights_theta_im"):
        key = "offdiag_im_penalty_matrix"

    if key is None or key not in spline_model:
        return None

    try:
        return np.asarray(spline_model[key].values)
    except Exception:
        return None


def compute_weight_summaries(
    idata, *, hdi_prob: float = HDI_PROB
) -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
    """Compute derived weight summaries for plotting.

    Returns
    -------
    tuple of dicts
        (scalar_summaries, vector_summaries)
    """
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return {}, {}

    scalar: Dict[str, xr.DataArray] = {}
    vector: Dict[str, xr.DataArray] = {}

    for name in find_weight_vars(posterior):
        var = posterior[name]
        if "chain" not in var.dims or "draw" not in var.dims:
            continue
        basis_dim = _basis_dim(var)
        if basis_dim is None:
            continue

        axis = var.get_axis_num(basis_dim)
        values = np.asarray(var.values)
        if values.size == 0:
            continue

        w_rms = np.sqrt(np.mean(values * values, axis=axis))
        w_maxabs = np.max(np.abs(values), axis=axis)
        scalar[f"w_rms__{name}"] = xr.DataArray(w_rms, dims=["chain", "draw"])
        scalar[f"w_maxabs__{name}"] = xr.DataArray(
            w_maxabs, dims=["chain", "draw"]
        )

        penalty_matrix = _get_penalty_matrix(idata, name)
        if (
            penalty_matrix is not None
            and penalty_matrix.ndim == 2
            and penalty_matrix.shape[0] == values.shape[axis]
        ):
            penalty = np.einsum(
                "...i,ij,...j->...",
                values,
                penalty_matrix,
                values,
                optimize=True,
            )
            scalar[f"penalty__{name}"] = xr.DataArray(
                penalty, dims=["chain", "draw"]
            )

        try:
            hdi = az.hdi(var, hdi_prob=hdi_prob)
            width = hdi.sel(hdi="higher") - hdi.sel(hdi="lower")
            vector[f"w_hdi_width__{name}"] = width
        except Exception:
            continue

    return scalar, vector


def select_rep_indices(
    var: xr.DataArray,
    ess_k: int = REP_WEIGHT_ESS_K,
    var_k: int = REP_WEIGHT_VAR_K,
    even_k: int = REP_WEIGHT_EVEN_K,
) -> np.ndarray:
    """Pick representative basis indices for a weight vector."""
    basis_dim = _basis_dim(var)
    if basis_dim is None:
        return np.array([], dtype=int)

    size = int(var.sizes.get(basis_dim, 0))
    if size <= 0:
        return np.array([], dtype=int)

    idxs: List[int] = []

    if ess_k > 0:
        try:
            ess = az.ess(var, method="bulk")
            ess_vals = np.asarray(ess)
            ess_vals = np.where(np.isfinite(ess_vals), ess_vals, np.inf)
            worst = np.argsort(ess_vals)[: min(ess_k, size)]
            idxs.extend([int(i) for i in worst])
        except Exception:
            pass

    if var_k > 0:
        try:
            variance = var.var(dim=("chain", "draw"))
            var_vals = np.asarray(variance)
            best = np.argsort(var_vals)[-min(var_k, size) :]
            idxs.extend([int(i) for i in best])
        except Exception:
            pass

    if even_k > 0:
        even = np.unique(
            np.linspace(0, size - 1, num=min(even_k, size), dtype=int)
        )
        idxs.extend([int(i) for i in even])

    if not idxs:
        return np.array([], dtype=int)

    idxs = sorted(set(idxs))
    if len(idxs) > REP_WEIGHT_MAX:
        idxs = idxs[:REP_WEIGHT_MAX]
    return np.asarray(idxs, dtype=int)


def build_plot_dataset(
    idata,
    derived_scalar: Dict[str, xr.DataArray],
    rep_indices: Dict[str, Iterable[int]] | Dict[str, np.ndarray],
) -> xr.Dataset:
    """Build a lightweight posterior dataset for ArviZ diagnostics."""
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return xr.Dataset()

    ds = xr.Dataset()

    for name, arr in derived_scalar.items():
        ds[name] = arr

    weight_vars = set(find_weight_vars(posterior))
    for name, var in posterior.data_vars.items():
        if str(name) in weight_vars:
            continue
        if "chain" not in var.dims or "draw" not in var.dims:
            continue
        extra_dims = [d for d in var.dims if d not in ("chain", "draw")]
        if extra_dims:
            continue
        ds[str(name)] = var

    for wname, idxs in rep_indices.items():
        if wname not in posterior.data_vars:
            continue
        var = posterior[wname]
        basis_dim = _basis_dim(var)
        if basis_dim is None:
            continue
        for idx in idxs:
            try:
                sliced = var.isel({basis_dim: int(idx)})
            except Exception:
                continue
            ds[f"{wname}__idx_{int(idx)}"] = sliced

    return ds
