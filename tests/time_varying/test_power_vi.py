"""Scalar VI uses the NUTS model and saved reconstruction, on both geometries."""

import numpy as np
import numpyro
import pytest
from jaxtyping import TypeCheckError

from log_psplines import PowerConfig, fit
from log_psplines.basis import SplineBasis
from log_psplines.data.spectral import PowerData
from log_psplines.inference.power import prepare_power_model
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


def test_diagnostic_state_and_all_joint_draws_roundtrip(tmp_path):
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
        vi_diagnostics={
            "num_particles": 64,
            "chunk_size": 16,
            "seeds": (81, 82),
            "evaluation_seeds": (91, 92),
            "evaluation_particles": 4,
        },
    )
    result = fit(data, config, model=model)
    result.posterior = result.posterior.assign_coords(
        chain=[3], draw=np.arange(100, 116)
    )
    path = tmp_path / "diagnosed.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    assert restored.spectrum.sizes["draw"] == 2
    assert restored.posterior.sizes["draw"] == 16
    np.testing.assert_array_equal(restored.posterior.draw, np.arange(100, 116))
    np.testing.assert_array_equal(restored.posterior.chain, [3])
    complete = power_draws_from_basis(restored.posterior, restored.model_data)
    assert complete.shape[:2] == (1, 16)
    assert restored.vi.diagnostics.metadata == result.vi.diagnostics.metadata
    for name, values in result.vi.diagnostics.arrays.items():
        np.testing.assert_array_equal(
            restored.vi.diagnostics.arrays[name], values
        )
    from log_psplines.diagnostics.variational import rebuild_guide

    prepared, _, _ = prepare_power_model(data, model, config)
    rebuilt = rebuild_guide(
        restored.vi.diagnostics,
        prepared,
        target_fingerprint=restored.metadata["target_fingerprint"],
    )
    assert rebuilt.latent_dim == 6


def test_custom_anova_prior_changes_target_fingerprint():
    from log_psplines.inference.anova_power import prepare_anova_power_model
    from log_psplines.inference.power import power_target_fingerprint
    from log_psplines.models.anova import ANOVALogPSpline

    time, freq = np.linspace(0, 1, 7), np.linspace(0, 0.5, 8)
    bt = SplineBasis.from_grid(time, 0, degree=1, penalty_order=1)
    bf = SplineBasis.from_grid(freq, 0, degree=1, penalty_order=1)
    first = ANOVALogPSpline(bf, bt, sigma_eta_prior=0.2)
    second = ANOVALogPSpline(bf, bt, sigma_eta_prior=0.8)
    data = PowerData(np.ones((7, 8)), np.full((7, 8), 2.0), freq, time)
    config = PowerConfig(progress_bar=False)
    _, pair, _ = prepare_anova_power_model(data, first, config)
    assert power_target_fingerprint(
        data, first, config, pair
    ) != power_target_fingerprint(data, second, config, pair)
