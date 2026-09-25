"""Tests for the new InferencePipeline / make_pipeline interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from log_psplines import make_pipeline
from log_psplines.results import _values_to_dataset
from log_psplines.data import WishartData, TimeSeries
from log_psplines.config import PipelineConfig
from log_psplines.pipeline import (
    InferencePipeline,
    PSDResult,
)
from log_psplines.inference.nuts import FactorizedMultivarNUTSStage
from log_psplines.inference.vi import FactorizedMultivarVIStage, StageResult
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def p1_data() -> TimeSeries:
    """Small one-channel AR series for fast p=1 tests."""
    from log_psplines.example_datasets.varma_data import VARMAData

    return VARMAData.ar(order=2, n_samples=64, fs=64.0, seed=7).ts


@pytest.fixture(scope="module")
def multivar_data() -> WishartData:
    """Small 2-channel WishartData (N=32, p=2) for fast tests."""
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
    ds = _values_to_dataset(
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
    assert isinstance(pipeline.data, WishartData)
    assert pipeline.data.p == 1
    # Default method="nuts" never constructs a VI stage.
    assert pipeline.vi_stage is None
    assert isinstance(pipeline.nuts_stage, FactorizedMultivarNUTSStage)


def test_make_pipeline_multivar_returns_inference_pipeline(multivar_data):
    pipeline = make_pipeline(multivar_data, _fast_config())
    assert isinstance(pipeline, InferencePipeline)
    assert pipeline.vi_stage is None
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
        method="vi",
    )
    pipeline = make_pipeline(p1_data, config)
    # method="vi" never constructs a NUTS stage.
    assert pipeline.nuts_stage is None
    assert pipeline.vi_stage is not None
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
    assert pipeline.nuts_stage is not None
    assert pipeline.nuts_stage.n_samples == 13
    assert pipeline.nuts_stage.n_warmup == 7
    assert pipeline.nuts_stage.target_accept_prob == pytest.approx(0.9)
    assert pipeline.nuts_stage.eta == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# only_vi mode (p=1)
# ---------------------------------------------------------------------------


def test_pipeline_p1_only_vi(p1_data):
    config = _fast_config(method="vi")
    result = make_pipeline(p1_data, config).run()

    assert isinstance(result, PSDResult)
    assert result.vi is not None
    assert result.vi.losses is not None
    assert result.vi.losses.shape[0] > 0
    assert result.vi.guide_name is not None
    assert isinstance(result.posterior, xr.Dataset)
    assert "weights_delta_0" in result.vi.init_values
    assert result.vi.samples is not None
    assert (
        result.posterior["weights_delta_0"].sizes["draw"]
        == config.vi_posterior_draws
    )


# ---------------------------------------------------------------------------
# only_vi mode (multivar)
# ---------------------------------------------------------------------------


def test_pipeline_multivar_only_vi(multivar_data):
    config = _fast_config(method="vi")
    result = make_pipeline(multivar_data, config).run()

    assert isinstance(result, PSDResult)
    assert result.vi is not None
    assert result.vi.losses.shape[0] > 0
    assert result.vi.losses_per_block is not None
    assert len(result.vi.losses_per_block) == multivar_data.p
    posterior = result.posterior
    # All per-channel weight sites should be present in VI means
    assert "weights_delta_0" in result.vi.init_values
    assert "weights_delta_1" in result.vi.init_values
    assert result.vi.losses_per_block is not None
    vi_posterior = result.vi_posterior
    assert vi_posterior is not None
    assert (
        vi_posterior["weights_delta_0"].sizes["draw"]
        == config.vi_posterior_draws
    )


def test_pipeline_multivar_vi_reconstructs_and_plots_coherence(multivar_data):
    """Small E2E VI path through ArviZ PSD quantiles and coherence plotting."""
    import matplotlib.pyplot as plt

    config = _fast_config(method="vi", vi_posterior_draws=8)
    result = make_pipeline(multivar_data, config).run()

    quantiles = result.quantiles((5.0, 50.0, 95.0))
    freq = result.frequency
    psd = np.asarray(
        quantiles.transpose(
            "percentile", "frequency", "channel", "channel_aux"
        )
    )
    coherence = np.percentile(
        result.coherence.reshape(-1, *result.coherence.shape[2:]),
        [5.0, 50.0, 95.0],
        axis=0,
    )

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
            idata=result,
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
# Full p=1 NUTS run
# ---------------------------------------------------------------------------


def test_pipeline_p1_nuts(p1_data):
    config = _fast_config()
    result = make_pipeline(p1_data, config).run()

    assert isinstance(result, PSDResult)
    # Default method="nuts" never runs VI.
    assert result.vi is None
    ds = result.posterior
    assert "weights_delta_0" in ds
    # Correct number of NUTS draws
    assert ds["weights_delta_0"].sizes["draw"] == config.n_samples
    stats = result.sample_stats
    assert stats is not None
    assert "acceptance_rate_channel_0" in stats
    spectral_density = result.spectral_density
    assert spectral_density.shape == (
        1, config.n_samples, len(result.frequency), 1, 1
    )
    median = np.median(np.real(spectral_density[..., 0, 0]), axis=(0, 1))
    assert np.all(np.isfinite(median))
    assert np.all(median > 0.0)


# ---------------------------------------------------------------------------
# Full multivar NUTS run
# ---------------------------------------------------------------------------


def test_pipeline_multivar_nuts(multivar_data):
    config = _fast_config()
    result = make_pipeline(multivar_data, config).run()

    assert isinstance(result, PSDResult)
    # Default method="nuts" never runs VI.
    assert result.vi is None
    ds = result.posterior
    assert "weights_delta_0" in ds
    assert "weights_delta_1" in ds
    assert ds["weights_delta_0"].sizes["draw"] == config.n_samples
    stats = result.sample_stats
    assert stats is not None
    assert "acceptance_rate_channel_0" in stats
    assert "acceptance_rate_channel_1" in stats
    assert result.metadata["data_type"] == "multivariate"


# ---------------------------------------------------------------------------
# PSDResult.save()
# ---------------------------------------------------------------------------


def test_pipeline_result_save(tmp_path, p1_data):
    config = _fast_config(method="vi")
    result = make_pipeline(p1_data, config).run()
    result.save(str(tmp_path))

    assert (tmp_path / "inference_data.nc").exists()
    assert (tmp_path / "vi_losses.npy").exists()


def test_pipeline_multivar_vi_save_records_truth_metrics(
    tmp_path,
    multivar_data,
):
    config = _fast_config(method="vi")
    result = make_pipeline(multivar_data, config).run()
    freq = result.frequency
    p = int(multivar_data.p)
    true_psd = np.tile(np.eye(p, dtype=np.complex128), (freq.size, 1, 1))

    result.save(str(tmp_path), true_psd=true_psd)

    vi_summary = pd.read_csv(tmp_path / "diagnostics" / "vi_summary.csv")
    for col in ("riae", "l2", "coverage"):
        values = pd.to_numeric(vi_summary[col], errors="coerce").to_numpy()
        assert np.all(np.isfinite(values))


def test_posterior_predictive_save_overlays_vi_when_available(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def _fake_plot_psd_matrix(spec):
        captured["spec"] = spec

    monkeypatch.setattr(
        "log_psplines.plotting.results.plot_psd_matrix",
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
    posterior = xr.Dataset(
        {"weights_delta_0": (("chain", "draw", "k"), np.zeros((1, 3, 2)))}
    )
    spectrum = xr.DataArray(
        np.ones((1, 3, 4, 1, 1), dtype=complex),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={"frequency": np.arange(4), "channel": [0], "channel_aux": [0]},
    )
    result = PSDResult(
        posterior=posterior,
        spectrum=spectrum,
        sample_stats=xr.Dataset({"diverging": (("chain", "draw"), np.zeros((1, 3)))}),
        vi=vi,
        vi_spectrum=spectrum,
    )

    from log_psplines.plotting.results import plot_posterior_spectrum

    plot_posterior_spectrum(result, tmp_path)

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
        "log_psplines.plotting.results.plot_psd_matrix",
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
    posterior = xr.Dataset(
        {"weights_delta_0": (("chain", "draw", "k"), np.zeros((1, 3, 2)))}
    )
    spectrum = xr.DataArray(
        np.ones((1, 3, 4, 1, 1), dtype=complex),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={"frequency": np.arange(4), "channel": [0], "channel_aux": [0]},
    )
    result = PSDResult(
        posterior=posterior,
        spectrum=spectrum,
        vi=vi,
    )

    from log_psplines.plotting.results import plot_posterior_spectrum

    plot_posterior_spectrum(result, tmp_path)

    spec = captured["spec"]
    assert spec.overlay_vi is False
    assert spec.label is None
