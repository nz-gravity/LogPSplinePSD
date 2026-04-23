"""Small NUTS plotting helpers built around ArviZ plots."""

from __future__ import annotations

import arviz_plots as azp
import xarray as xr


def plot_traces(
    posteriors: xr.DataTree | dict[str, xr.DataTree],
):
    """
    TODO

    pc = azp.combine_plots(
        posteriors_dict,
        plots=(azp.plot_trace_dist, dict(compact=True)),
    )
    return pc.viz["figure"].item()
    """

    return azp.plot_trace_dist(posteriors, compact=True)


def plot_energy(posteriors: xr.DataTree | dict[str, xr.DataTree]):
    """
    TODO

    pc = azp.combine_plots(
        posteriors_dict,
        plots=(azp.plot_energy, {}),
    )
    return pc.viz["figure"].item()
    """

    return azp.plot_energy(posteriors)
