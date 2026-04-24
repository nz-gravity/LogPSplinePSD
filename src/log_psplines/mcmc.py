from __future__ import annotations

import xarray as xr

from .pipeline.config import PipelineConfig
from .pipeline.make_pipeline import make_pipeline

__all__ = ["run_mcmc", "PipelineConfig"]


def run_mcmc(data, config=None, **kwargs) -> xr.DataTree:
    if kwargs:
        if config is not None:
            raise ValueError("Cannot use both config and kwargs")
        config = PipelineConfig(**kwargs)
    return make_pipeline(data, config).run().idata
