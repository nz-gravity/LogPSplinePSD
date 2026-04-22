"""Small NUTS plotting helpers built around ArviZ plots."""

from __future__ import annotations

from typing import Sequence

import arviz_plots as azp
import matplotlib.pyplot as plt
import xarray as xr

from ..arviz_utils._datatree import require_dataset as _require_dataset
from ._factors import factor_idatas


def _default_trace_vars(posterior: xr.Dataset, *, max_vars: int) -> list[str]:
    scalar_vars = []
    weight_vars = []
    other_vars = []

    for name, var in posterior.data_vars.items():
        extra_dims = [dim for dim in var.dims if dim not in ("chain", "draw")]
        if not extra_dims:
            scalar_vars.append(str(name))
        elif str(name).startswith("weights"):
            weight_vars.append(str(name))
        else:
            other_vars.append(str(name))

    ordered = scalar_vars + weight_vars + other_vars
    return ordered[: max(1, int(max_vars))]


def plot_factor_traces(
    idata_or_factors: xr.DataTree | dict[str, xr.DataTree],
    *,
    factor: str | None = None,
    var_names: Sequence[str] | None = None,
    max_vars: int = 4,
) -> plt.Figure:
    """Plot traces for a small subset of parameters from one factor."""

    factors = factor_idatas(idata_or_factors)
    if factor is None:
        factor = next(iter(factors))
    idata = factors[str(factor)]
    posterior = _require_dataset(idata, "posterior")
    selected = (
        list(var_names)
        if var_names is not None
        else _default_trace_vars(posterior, max_vars=max_vars)
    )
    if not selected:
        raise ValueError(
            "No posterior variables available for trace plotting."
        )

    plot = azp.plot_trace(
        idata,
        var_names=selected,
        backend="matplotlib",
        figure_kwargs={"figsize": (12, 2.5 * len(selected))},
    )
    fig = plot.viz["figure"].item()
    fig.suptitle(f"Factor {factor} traces", y=1.02)
    return fig
