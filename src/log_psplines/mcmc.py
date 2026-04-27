from __future__ import annotations

import xarray as xr

from .pipeline.config import PipelineConfig
from .pipeline.make_pipeline import make_pipeline
from .pipeline.preprocessing import align_true_psd_to_freq

__all__ = ["run_mcmc", "PipelineConfig"]


def run_mcmc(data, config=None, **kwargs) -> xr.DataTree:
    if kwargs:
        if config is not None:
            raise ValueError("Cannot use both config and kwargs")
        config = PipelineConfig(**kwargs)
    pipeline = make_pipeline(data, config)
    result = pipeline.run()
    if config is not None and config.outdir is not None:
        true_psd = align_true_psd_to_freq(config.true_psd, pipeline.data)
        result.save(config.outdir, true_psd=true_psd)
    return result.idata
