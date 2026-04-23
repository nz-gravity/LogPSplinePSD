"""Small NUTS plotting helpers built around ArviZ plots."""

from __future__ import annotations

import arviz_plots as azp
import xarray as xr


def plot_traces(
    posteriors: xr.DataTree | dict[str, xr.DataTree],
):
    return azp.plot_trace_dist(posteriors, compact=True)


def plot_energy(posteriors: xr.DataTree | dict[str, xr.DataTree]):
    return azp.plot_energy(posteriors)
