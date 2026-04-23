"""Small NUTS plotting helpers built around ArviZ plots."""

from __future__ import annotations

from typing import Sequence

import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ._factors import factor_idatas


def plot_traces(
    posteriors: xr.DataTree | dict[str, xr.DataTree],
) -> plt.Figure:
    posteriors_dict = factor_idatas(posteriors)
    pc = azp.combine_plots(
        posteriors_dict,
        plots=(azp.plot_trace_dist, dict(compact=True)),
    )
    return pc.viz["figure"].item()


def plot_energy(posteriors: xr.DataTree | dict[str, xr.DataTree]):
    posteriors_dict = factor_idatas(posteriors)
    pc = azp.combine_plots(
        posteriors_dict,
        plots=(azp.plot_energy, {}),
    )
    return pc.viz["figure"].item()
