"""Integration tests for run_mcmc (pipeline path)."""

from typing import cast

import numpy as np
import pytest
import xarray as xr

from log_psplines.mcmc import run_mcmc
from log_psplines.pipeline.config import PipelineConfig


def test_mcmc_univar(outdir: str):
    from log_psplines.example_datasets.ar_data import ARData

    ar_data = ARData(order=4, duration=1.0, fs=2048, seed=42)
    config = PipelineConfig(
        n_knots=20,
        n_samples=500,
        n_warmup=500,
        num_chains=2,
        verbose=True,
        rng_key=42,
    )
    idata = run_mcmc(ar_data.ts, config=config)

    assert isinstance(idata, xr.DataTree)
    assert "posterior" in idata.children
    assert "sample_stats" in idata.children
    ds = idata["posterior"].dataset
    assert ds is not None
    assert "weights" in ds
    assert ds["weights"].sizes["draw"] == config.n_samples


def test_mcmc_multivar(outdir: str):
    from log_psplines.datatypes.multivar import MultivariateTimeseries
    from log_psplines.example_datasets.varma_data import VARMAData
    from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

    varma_data = VARMAData(n_samples=2**12, fs=64.0, seed=0)
    ts_run = MultivariateTimeseries(
        y=cast(np.ndarray, varma_data.data), t=varma_data.time
    )

    config = PipelineConfig(
        n_knots=10,
        n_samples=200,
        n_warmup=200,
        vi_steps=5000,
        vi_lr=5e-3,
        vi_posterior_draws=100,
        Nb=4,
        fmin=0,
        fmax=32,
        coarse_grain_config=CoarseGrainConfig(enabled=True, Nc=None, Nh=2),
        verbose=True,
        rng_key=42,
    )
    idata = run_mcmc(ts_run, config=config)

    assert isinstance(idata, xr.DataTree)
    assert "posterior" in idata.children
    ds = idata["posterior"].dataset
    assert ds is not None
    assert "weights_delta_0" in ds
    assert ds["weights_delta_0"].sizes["draw"] == config.n_samples
