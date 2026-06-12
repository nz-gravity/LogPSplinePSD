"""Tests for the new InferencePipeline / make_pipeline interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from log_psplines import make_pipeline
from log_psplines.arviz_utils import (
    get_multivar_vi_psd_quantiles,
    get_psd_dataset,
)
from log_psplines.datatypes import MultivarFFT, MultivariateTimeseries
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.pipeline.pipeline import (
    InferencePipeline,
    PipelineResult,
    _init_values_to_dataset,
)
from log_psplines.pipeline.stages import (
    FactorizedMultivarNUTSStage,
    FactorizedMultivarVIStage,
    StageResult,
)
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def p1_data() -> MultivariateTimeseries:
    """Small one-channel AR series for fast p=1 tests."""
    from log_psplines.example_datasets.varma_data import VARMAData

    return VARMAData.ar(order=2, n_samples=64, fs=64.0, seed=7).ts


@pytest.fixture(scope="module")
def multivar_data() -> MultivarFFT:
    """Small 2-channel MultivarFFT (N=32, p=2) for fast tests."""
    from log_psplines.example_datasets.varma_data import VARMAData

    varma = VARMAData(n_samples=64, fs=16.0, seed=7)
    return varma.ts.standardise_for_psd().to_wishart_stats(Nb=1)


def _fast_config(**extra) -> PipelineConfig:
    """Return a PipelineConfig tuned for speed in CI."""
    defaults = dict(
        n_knots=4,
        n_samples=5,
        n_warmup=5,
        num_chains=1,
        vi_steps=20,
        vi_posterior_draws=5,
        verbose=False,
    )
    defaults.update(extra)
    return PipelineConfig(**defaults)


def test_vi_init_values_dataset_uses_variable_specific_dims():
    ds = _init_values_to_dataset(
        {
            "delta_0": np.zeros(3),
            "weights_delta_0": np.zeros(51),
            "weights_theta_re_1_0": np.zeros((3, 51)),
        }
    )

    assert ds["delta_0"].dims == ("chain", "draw", "delta_0_dim_0")
    assert ds["weights_delta_0"].dims == (
        "chain",
        "draw",
        "weights_delta_0_dim_0",
    )
    assert ds["weights_theta_re_1_0"].dims == (
        "chain",
        "draw",
        "weights_theta_re_1_0_dim_0",
        "weights_theta_re_1_0_dim_1",
    )


# ---------------------------------------------------------------------------
# make_pipeline construction
# ---------------------------------------------------------------------------


def test_make_pipeline_p1_returns_inference_pipeline(p1_data):
    pipeline = make_pipeline(p1_data, _fast_config())
    assert isinstance(pipeline, InferencePipeline)
    assert (
        pipeline.coarse_model_kwargs is None
    )  # auto_coarse_vi=False by default
    assert isinstance(pipeline.data, MultivarFFT)
    assert pipeline.data.p == 1
    assert isinstance(pipeline.vi_stage, FactorizedMultivarVIStage)
    assert isinstance(pipeline.nuts_stage, FactorizedMultivarNUTSStage)


def test_make_pipeline_multivar_returns_inference_pipeline(multivar_data):
    pipeline = make_pipeline(multivar_data, _fast_config())
    assert isinstance(pipeline, InferencePipeline)
    assert pipeline.coarse_model_kwargs is None
    assert isinstance(pipeline.vi_stage, FactorizedMultivarVIStage)
    assert isinstance(pipeline.nuts_stage, FactorizedMultivarNUTSStage)


def test_make_pipeline_vi_stage_uses_config(p1_data):
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
    pipeline = make_pipeline(p1_data, config)
    assert pipeline.vi_stage.steps == 77
    assert pipeline.vi_stage.lr == pytest.approx(3e-3)
    assert pipeline.vi_stage.eta == pytest.approx(0.5)


def test_make_pipeline_nuts_stage_uses_config(p1_data):
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
    pipeline = make_pipeline(p1_data, config)
    assert pipeline.nuts_stage.n_samples == 13
    assert pipeline.nuts_stage.n_warmup == 7
    assert pipeline.nuts_stage.target_accept_prob == pytest.approx(0.9)
    assert pipeline.nuts_stage.eta == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# only_vi mode (p=1)
# ---------------------------------------------------------------------------


def test_pipeline_p1_only_vi(p1_data):
    config = _fast_config(only_vi=True)
    result = make_pipeline(p1_data, config).run()

    assert isinstance(result, PipelineResult)
    assert result.vi_coarse is None
    assert result.vi is not None
    assert result.vi.losses is not None
    assert result.vi.losses.shape[0] > 0
    assert result.vi.guide_name is not None
    assert isinstance(result.idata, xr.DataTree)
    assert "posterior" in result.idata.children
    assert "weights_delta_0" in result.vi.init_values
    assert result.vi.samples is not None
    assert (
        result.idata["posterior"].dataset["weights_delta_0"].sizes["draw"]
        == config.vi_posterior_draws
    )


# ---------------------------------------------------------------------------
# only_vi mode (multivar)
# ---------------------------------------------------------------------------


def test_pipeline_multivar_only_vi(multivar_data):
    config = _fast_config(only_vi=True)
    result = make_pipeline(multivar_data, config).run()

    assert isinstance(result, PipelineResult)
    assert result.vi is not None
    assert result.vi.losses.shape[0] > 0
    assert result.vi.losses_per_block is not None
    assert len(result.vi.losses_per_block) == multivar_data.p
    posterior = result.idata.children.get("posterior")
    assert posterior is not None
    # All per-channel weight sites should be present in VI means
    assert "weights_delta_0" in result.vi.init_values
    assert "weights_delta_1" in result.vi.init_values
    vi_stats = result.idata["vi_sample_stats"].dataset
    assert "losses_per_block" in vi_stats
    vi_posterior = result.idata["vi_posterior"].dataset
    assert (
        vi_posterior["weights_delta_0"].sizes["draw"]
        == config.vi_posterior_draws
    )


def test_pipeline_multivar_vi_reconstructs_and_plots_coherence(multivar_data):
    """Small E2E VI path through ArviZ PSD quantiles and coherence plotting."""
    import matplotlib.pyplot as plt

    config = _fast_config(only_vi=True, vi_posterior_draws=8)
    result = make_pipeline(multivar_data, config).run()

    quantiles = get_multivar_vi_psd_quantiles(result.idata, n_keep=4)
    freq = np.asarray(quantiles["freq"], dtype=float)
    psd = np.asarray(quantiles["spectral_density"], dtype=np.complex128)
    coherence = np.asarray(quantiles["coherence"], dtype=float)

    assert psd.shape[:2] == (3, freq.size)
    assert psd.shape[2:] == (multivar_data.p, multivar_data.p)
    assert coherence.shape == psd.shape
    assert np.all(np.isfinite(psd.real))
    assert np.all(np.isfinite(psd.imag))
    assert np.all(np.isfinite(coherence))
    assert np.all((coherence >= 0.0) & (coherence <= 1.0))

    median_idx = int(
        np.argmin(np.abs(np.asarray(quantiles["percentile"]) - 50.0))
    )
    median_psd = psd[median_idx]
    assert np.allclose(
        median_psd,
        np.swapaxes(median_psd.conj(), 1, 2),
        rtol=1e-6,
        atol=1e-8,
    )

    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(
            idata=result.idata,
            save=False,
            close=False,
            show_coherence=True,
            show_knots=True,
            channel_labels=["x", "y"],
        )
    )
    assert axes.shape == (multivar_data.p, multivar_data.p)
    fig.canvas.draw()
    plt.close(fig)


# ---------------------------------------------------------------------------
# vi_coarse=False (i.e., no coarse stage, auto_coarse_vi=False)
# ---------------------------------------------------------------------------


def test_pipeline_no_coarse_vi(p1_data):
    """With auto_coarse_vi=False (default), vi_coarse should be None."""
    config = _fast_config(auto_coarse_vi=False, only_vi=True)
    result = make_pipeline(p1_data, config).run()

    assert result.vi_coarse is None
    assert result.vi is not None


# ---------------------------------------------------------------------------
# Full p=1 NUTS run
# ---------------------------------------------------------------------------


def test_pipeline_p1_nuts(p1_data):
    config = _fast_config()
    result = make_pipeline(p1_data, config).run()

    assert isinstance(result, PipelineResult)
    assert result.vi is not None
    assert isinstance(result.idata, xr.DataTree)
    posterior = result.idata.children.get("posterior")
    assert posterior is not None
    ds = posterior.dataset
    assert "weights_delta_0" in ds
    # Correct number of NUTS draws
    assert ds["weights_delta_0"].sizes["draw"] == config.n_samples
    stats = result.idata["sample_stats"].dataset
    assert "acceptance_rate_channel_0" in stats
    psd_ds = get_psd_dataset(result.idata, source="posterior")
    spectral_density = psd_ds["spectral_density"].values
    assert spectral_density.shape[:4] == (1, config.n_samples, 1, 1)
    median = np.median(np.real(spectral_density[:, :, 0, 0, :]), axis=(0, 1))
    assert np.all(np.isfinite(median))
    assert np.all(median > 0.0)


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
    assert "weights_delta_1" in ds
    assert ds["weights_delta_0"].sizes["draw"] == config.n_samples
    stats = result.idata["sample_stats"].dataset
    assert "acceptance_rate_channel_0" in stats
    assert "acceptance_rate_channel_1" in stats
    assert result.idata.attrs["factorized"] is True


# ---------------------------------------------------------------------------
# PipelineResult.save()
# ---------------------------------------------------------------------------


def test_pipeline_result_save(tmp_path, p1_data):
    config = _fast_config(only_vi=True)
    result = make_pipeline(p1_data, config).run()
    result.save(str(tmp_path))

    assert (tmp_path / "inference_data.nc").exists()
    assert (tmp_path / "vi_losses.npy").exists()


def test_pipeline_multivar_vi_save_records_truth_metrics(
    tmp_path,
    multivar_data,
):
    config = _fast_config(only_vi=True)
    result = make_pipeline(multivar_data, config).run()
    freq = np.asarray(
        result.idata["observed_data"].dataset["periodogram"].coords["freq"],
        dtype=float,
    )
    p = int(multivar_data.p)
    true_psd = np.tile(np.eye(p, dtype=np.complex128), (freq.size, 1, 1))

    result.save(str(tmp_path), true_psd=true_psd)

    vi_summary = pd.read_csv(tmp_path / "diagnostics" / "vi_summary.csv")
    for col in ("riae", "l2", "coverage"):
        values = pd.to_numeric(vi_summary[col], errors="coerce").to_numpy()
        assert np.all(np.isfinite(values))
    vi_stats = result.idata["vi_sample_stats"].attrs
    assert np.isfinite(float(vi_stats["riae"]))
    assert np.isfinite(float(vi_stats["l2"]))
    assert np.isfinite(float(vi_stats["coverage"]))


def test_posterior_predictive_save_overlays_vi_when_available(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def _fake_plot_psd_matrix(spec):
        captured["spec"] = spec

    monkeypatch.setattr(
        "log_psplines.pipeline.pipeline.plot_psd_matrix",
        _fake_plot_psd_matrix,
    )
    vi = StageResult(
        init_values={"weights_delta_0": np.zeros(2)},
        losses=np.asarray([1.0]),
        khat=None,
        guide_name="diag",
        runtime=0.0,
        samples={"weights_delta_0": np.zeros((3, 2))},
    )
    result = PipelineResult(
        vi_coarse=None,
        vi=vi,
        idata=xr.DataTree(children={"sample_stats": xr.DataTree()}),
    )

    result._save_posterior_predictive(str(tmp_path))

    spec = captured["spec"]
    assert spec.overlay_vi is True
    assert spec.label == "NUTS 90% CI"
    assert spec.vi_label == "VI 90% CI"


def test_posterior_predictive_save_does_not_label_only_vi_as_nuts(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def _fake_plot_psd_matrix(spec):
        captured["spec"] = spec

    monkeypatch.setattr(
        "log_psplines.pipeline.pipeline.plot_psd_matrix",
        _fake_plot_psd_matrix,
    )
    vi = StageResult(
        init_values={"weights_delta_0": np.zeros(2)},
        losses=np.asarray([1.0]),
        khat=None,
        guide_name="diag",
        runtime=0.0,
        samples={"weights_delta_0": np.zeros((3, 2))},
    )
    result = PipelineResult(
        vi_coarse=None,
        vi=vi,
        idata=xr.DataTree(),
    )

    result._save_posterior_predictive(str(tmp_path))

    spec = captured["spec"]
    assert spec.overlay_vi is False
    assert spec.label is None
