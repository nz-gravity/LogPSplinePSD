"""Tests for the new InferencePipeline / make_pipeline interface."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from log_psplines import make_pipeline
from log_psplines.datatypes import Periodogram
from log_psplines.datatypes.multivar import MultivarFFT, MultivariateTimeseries
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.pipeline.pipeline import InferencePipeline, PipelineResult
from log_psplines.pipeline.stages import NUTSStage, VIStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def univar_data() -> Periodogram:
    """Small periodogram (N≈32) for fast tests."""
    from log_psplines.example_datasets.ar_data import ARData

    ar = ARData(order=2, duration=1.0, fs=64, seed=7)
    return ar.ts.standardise_for_psd().to_periodogram()


@pytest.fixture(scope="module")
def multivar_data() -> MultivarFFT:
    """Small 2-channel MultivarFFT (N=32, p=2) for fast tests."""
    from log_psplines.example_datasets.varma_data import VARMAData

    varma = VARMAData(n_samples=64, fs=16.0, seed=7)
    mv_ts = MultivariateTimeseries(y=varma.data, t=varma.time)
    return mv_ts.standardise_for_psd().to_wishart_stats(Nb=1)


def _fast_config(**extra) -> PipelineConfig:
    """Return a PipelineConfig tuned for speed in CI."""
    return PipelineConfig(
        n_knots=4,
        n_samples=5,
        n_warmup=5,
        num_chains=1,
        vi_steps=20,
        vi_posterior_draws=5,
        verbose=False,
        **extra,
    )


# ---------------------------------------------------------------------------
# make_pipeline construction
# ---------------------------------------------------------------------------


def test_make_pipeline_univar_returns_inference_pipeline(univar_data):
    pipeline = make_pipeline(univar_data, _fast_config())
    assert isinstance(pipeline, InferencePipeline)
    assert (
        pipeline.coarse_model_kwargs is None
    )  # auto_coarse_vi=False by default


def test_make_pipeline_multivar_returns_inference_pipeline(multivar_data):
    pipeline = make_pipeline(multivar_data, _fast_config())
    assert isinstance(pipeline, InferencePipeline)
    assert pipeline.coarse_model_kwargs is None


def test_make_pipeline_vi_stage_uses_config(univar_data):
    config = PipelineConfig(
        n_knots=4,
        n_samples=5,
        n_warmup=5,
        num_chains=1,
        vi_steps=77,
        vi_lr=3e-3,
        vi_posterior_draws=5,
        verbose=False,
        eta=0.5,
    )
    pipeline = make_pipeline(univar_data, config)
    assert pipeline.vi_stage.steps == 77
    assert pipeline.vi_stage.lr == pytest.approx(3e-3)
    assert pipeline.vi_stage.eta == pytest.approx(0.5)


def test_make_pipeline_nuts_stage_uses_config(univar_data):
    config = PipelineConfig(
        n_knots=4,
        vi_steps=20,
        vi_posterior_draws=5,
        n_samples=13,
        n_warmup=7,
        num_chains=1,
        target_accept_prob=0.9,
        verbose=False,
        eta=0.25,
    )
    pipeline = make_pipeline(univar_data, config)
    assert pipeline.nuts_stage.n_samples == 13
    assert pipeline.nuts_stage.n_warmup == 7
    assert pipeline.nuts_stage.target_accept_prob == pytest.approx(0.9)
    assert pipeline.nuts_stage.eta == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# only_vi mode (univar)
# ---------------------------------------------------------------------------


def test_pipeline_univar_only_vi(univar_data):
    config = _fast_config(only_vi=True)
    result = make_pipeline(univar_data, config).run()

    assert isinstance(result, PipelineResult)
    assert result.vi_coarse is None
    assert result.vi is not None
    assert result.vi.losses is not None
    assert result.vi.losses.shape[0] > 0
    assert result.vi.guide_name is not None
    assert isinstance(result.idata, xr.DataTree)
    assert "posterior" in result.idata.children


# ---------------------------------------------------------------------------
# only_vi mode (multivar)
# ---------------------------------------------------------------------------


def test_pipeline_multivar_only_vi(multivar_data):
    config = _fast_config(only_vi=True)
    result = make_pipeline(multivar_data, config).run()

    assert isinstance(result, PipelineResult)
    assert result.vi is not None
    assert result.vi.losses.shape[0] > 0
    posterior = result.idata.children.get("posterior")
    assert posterior is not None
    # All per-channel weight sites should be present in VI means
    assert "weights_delta_0" in result.vi.init_values
    assert "weights_delta_1" in result.vi.init_values


# ---------------------------------------------------------------------------
# vi_coarse=False (i.e., no coarse stage, auto_coarse_vi=False)
# ---------------------------------------------------------------------------


def test_pipeline_no_coarse_vi(univar_data):
    """With auto_coarse_vi=False (default), vi_coarse should be None."""
    config = _fast_config(auto_coarse_vi=False, only_vi=True)
    result = make_pipeline(univar_data, config).run()

    assert result.vi_coarse is None
    assert result.vi is not None


# ---------------------------------------------------------------------------
# Full univar NUTS run
# ---------------------------------------------------------------------------


def test_pipeline_univar_nuts(univar_data):
    config = _fast_config()
    result = make_pipeline(univar_data, config).run()

    assert isinstance(result, PipelineResult)
    assert result.vi is not None
    assert isinstance(result.idata, xr.DataTree)
    posterior = result.idata.children.get("posterior")
    assert posterior is not None
    ds = posterior.dataset
    assert "weights" in ds
    # Correct number of NUTS draws
    assert ds["weights"].sizes["draw"] == config.n_samples


# ---------------------------------------------------------------------------
# Full multivar NUTS run
# ---------------------------------------------------------------------------


def test_pipeline_multivar_nuts(multivar_data):
    config = _fast_config()
    result = make_pipeline(multivar_data, config).run()

    assert isinstance(result, PipelineResult)
    assert result.vi is not None
    assert isinstance(result.idata, xr.DataTree)
    posterior = result.idata.children.get("posterior")
    assert posterior is not None
    ds = posterior.dataset
    assert "weights_delta_0" in ds
    assert ds["weights_delta_0"].sizes["draw"] == config.n_samples


# ---------------------------------------------------------------------------
# PipelineResult.save()
# ---------------------------------------------------------------------------


def test_pipeline_result_save(tmp_path, univar_data):
    config = _fast_config(only_vi=True)
    result = make_pipeline(univar_data, config).run()
    result.save(str(tmp_path))

    assert (tmp_path / "inference_data.nc").exists()
    assert (tmp_path / "vi_losses.npy").exists()
