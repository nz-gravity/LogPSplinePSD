"""Scalar VI uses the NUTS model and saved reconstruction, on both geometries."""

import numpy as np
import numpyro
import pytest
import xarray as xr
from jaxtyping import TypeCheckError

from log_psplines import PowerConfig, fit
from log_psplines.basis import SplineBasis
from log_psplines.data.spectral import PowerData
from log_psplines.inference.power import prepare_power_model
from log_psplines.models.anova import ANOVALogPSpline
from log_psplines.models.reconstruction import power_draws_from_basis
from log_psplines.models.spectrum import LogPSpline
from log_psplines.preprocessing.moving_periodogram import (
    scattered_moving_periodogram,
)
from log_psplines.results import PSDResult


@pytest.mark.parametrize(
    "centered,grid,guide",
    [
        (False, False, "diag"),
        (True, False, "lowrank:10"),
        (False, True, "diag"),
    ],
)
def test_public_power_vi_shared_model_and_round_trip(
    tmp_path, centered, grid, guide
):
    data = scattered_moving_periodogram(
        np.random.default_rng(4).normal(size=128), dt=1.0, m=4
    )
    time, freq = np.linspace(0.0, 1.0, 7), np.linspace(0.0, 0.5, 8)
    if grid:
        data = PowerData(
            np.full((7, 8), 1 / np.pi), np.full((7, 8), 2.0), freq, time
        )
    model = LogPSpline(
        SplineBasis.from_grid(freq, 0, degree=1, penalty_order=1),
        time=SplineBasis.from_grid(time, 0, degree=1, penalty_order=1),
    )
    config = PowerConfig(
        method="vi",
        vi_steps=30,
        vi_posterior_draws=8,
        centered=centered,
        vi_guide=guide,
        progress_bar=False,
    )
    result = fit(data, config, model=model)
    assert result.sample_stats is None and result.log_likelihood is None
    assert result.vi is not None and np.isfinite(result.vi.losses).all()
    assert result.posterior.sizes["chain"] == 1
    assert result.posterior.sizes["draw"] == 8
    if guide.startswith("lowrank"):
        assert result.vi.guide_name == "lowrank:6"
    assert np.all(result.posterior.sigma_time > 0)
    assert np.all(result.posterior.sigma_freq > 0)
    assert np.all(result.psd > 0) and np.isfinite(result.psd).all()
    weights = result.posterior.weights.values[0, 0]
    np.testing.assert_allclose(
        result.psd[0, 0],
        np.exp(model.time.basis @ weights @ model.frequency.basis.T),
        rtol=1e-12,
    )
    prepared, _, _ = prepare_power_model(data, model, config)
    params = {
        name: value.values[0, 0]
        for name, value in result.posterior.items()
        if name in ("s", "sigma_time", "sigma_freq")
    }
    trace = numpyro.handlers.trace(
        numpyro.handlers.substitute(prepared, data=params)
    ).get_trace()
    logs = (
        model.time.basis @ weights @ model.frequency.basis.T
        if grid
        else np.einsum(
            "pi,ij,pj->p",
            model.time.design_at(data.time),
            weights,
            model.frequency.design_at(data.frequency),
        )
    )
    np.testing.assert_allclose(
        trace["log_likelihood"]["value"],
        -0.5 * np.sum(data.counts * logs + data.power * np.exp(-logs)),
        rtol=1e-12,
    )
    path = tmp_path / "result.nc"
    result.to_netcdf(path)
    loaded = PSDResult.from_netcdf(path)
    np.testing.assert_allclose(
        power_draws_from_basis(loaded.posterior, loaded.model_data)[..., 0],
        result.psd,
    )
    assert loaded.metadata["units"] == data.units
    assert loaded.observed_data.attrs["units"] == data.units
    np.testing.assert_array_equal(
        loaded.observed_data.time, result.observed_data.time
    )
    np.testing.assert_array_equal(
        loaded.observed_data.frequency, result.observed_data.frequency
    )
    assert loaded.vi.guide_name == result.vi.guide_name
    np.testing.assert_array_equal(loaded.vi.losses, result.vi.losses)
    assert loaded.vi.timings == result.vi.timings
    assert loaded.sample_stats is None
    assert loaded.to_arviz() is not None


def test_power_config_rejects_invalid_dispatch():
    for kwargs in (
        {"method": "bad"},
        {"vi_steps": 0},
        {"vi_lr": np.nan},
        {"vi_posterior_draws": False},
    ):
        with pytest.raises((ValueError, TypeCheckError)):
            PowerConfig(**kwargs)


def test_all_posterior_draws_survive_spectrum_preview_roundtrip(tmp_path):
    time, freq = np.linspace(0, 1, 7), np.linspace(0, 0.5, 8)
    model = LogPSpline(
        SplineBasis.from_grid(freq, 0, degree=1, penalty_order=1),
        time=SplineBasis.from_grid(time, 0, degree=1, penalty_order=1),
    )
    data = PowerData(np.ones((7, 8)), np.full((7, 8), 2.0), freq, time)
    config = PowerConfig(
        method="vi",
        vi_steps=20,
        vi_posterior_draws=16,
        spectrum_draws=2,
        progress_bar=False,
        vi_early_stopping=False,
    )
    result = fit(data, config, model=model)
    result.posterior = result.posterior.assign_coords(
        chain=[3], draw=np.arange(100, 116)
    )
    path = tmp_path / "preview.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    assert restored.spectrum.sizes["draw"] == 2
    assert restored.posterior.sizes["draw"] == 16
    np.testing.assert_array_equal(restored.posterior.draw, np.arange(100, 116))
    np.testing.assert_array_equal(restored.posterior.chain, [3])
    complete = power_draws_from_basis(restored.posterior, restored.model_data)
    assert complete.shape[:2] == (1, 16)
    np.testing.assert_array_equal(restored.vi.losses, result.vi.losses)
    assert restored.vi.timings == result.vi.timings


def test_public_anova_vi_shared_model_and_roundtrip(tmp_path):
    from log_psplines.inference.anova_power import prepare_anova_power_model

    time, freq = np.linspace(0, 1, 7), np.linspace(0, 0.5, 8)
    bt = SplineBasis.from_grid(time, 0, degree=1, penalty_order=1)
    bf = SplineBasis.from_grid(freq, 0, degree=1, penalty_order=1)
    model = ANOVALogPSpline(bf, bt, sigma_eta_prior=0.2)
    data = PowerData(np.ones((7, 8)), np.full((7, 8), 2.0), freq, time)
    config = PowerConfig(
        method="vi", vi_steps=20, vi_posterior_draws=8, progress_bar=False
    )
    result = fit(data, config, model=model)
    assert result.posterior.sizes["draw"] == 8
    assert np.isfinite(result.vi.losses).all()
    assert np.all(result.psd > 0) and np.isfinite(result.psd).all()
    assert np.all(result.posterior.sigma_g > 0)
    assert np.all(result.posterior.sigma_eta > 0)
    logs = model(
        result.posterior.weights_g.values[0, 0],
        result.posterior.weights_eta.values[0, 0],
    )
    np.testing.assert_allclose(result.psd[0, 0], np.exp(logs), rtol=1e-12)
    prepared, _, _ = prepare_anova_power_model(data, model, config)
    params = {
        name: result.posterior[name].values[0, 0]
        for name in ("g", "eta", "sigma_g", "sigma_eta")
    }
    trace = numpyro.handlers.trace(
        numpyro.handlers.substitute(prepared, data=params)
    ).get_trace()
    np.testing.assert_allclose(
        trace["log_likelihood"]["value"],
        -0.5 * np.sum(data.counts * logs + data.power * np.exp(-logs)),
        rtol=1e-12,
    )
    result.to_netcdf(tmp_path / "anova.nc")
    loaded = PSDResult.from_netcdf(tmp_path / "anova.nc")
    np.testing.assert_allclose(
        power_draws_from_basis(loaded.posterior, loaded.model_data)[..., 0],
        result.psd,
    )
    assert loaded.metadata["model"] == "anova"
    np.testing.assert_array_equal(loaded.vi.losses, result.vi.losses)


def test_vi_block_losses_roundtrip(tmp_path):
    from log_psplines.inference.vi import VIResult

    posterior = xr.Dataset({"s": (("chain", "draw"), [[0.0, 1.0]])})
    losses = [np.array([4.0, 3.0, 2.0]), np.array([2.0, 1.0])]
    vi = VIResult(
        posterior=posterior,
        losses=np.array([6.0, 4.0, 3.0]),
        guide_name="mvn",
        losses_per_block=losses,
        timings={"steps_run": 5, "num_blocks": 2},
    )
    spectrum = xr.DataArray(
        np.ones((1, 2, 3, 1, 1), dtype=np.complex128),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={"frequency": [0.1, 0.2, 0.3]},
    )
    result = PSDResult(posterior=posterior, spectrum=spectrum, vi=vi)
    result.to_netcdf(tmp_path / "blocked.nc")
    loaded = PSDResult.from_netcdf(tmp_path / "blocked.nc")
    assert loaded.vi.guide_name == vi.guide_name
    assert loaded.vi.timings == vi.timings
    np.testing.assert_array_equal(loaded.vi.losses, vi.losses)
    for actual, expected in zip(
        loaded.vi.losses_per_block, losses, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
