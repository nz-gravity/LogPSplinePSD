from __future__ import annotations

import xarray as xr

from log_psplines.config import PipelineConfig
from log_psplines.pipeline import fit

__all__ = ["run_mcmc", "PipelineConfig"]


def run_mcmc(data, config=None, **kwargs) -> xr.DataTree:
    if kwargs:
        if config is not None:
            raise ValueError("Cannot use both config and kwargs")
        config = PipelineConfig(**kwargs)
    return fit(data, config).idata
