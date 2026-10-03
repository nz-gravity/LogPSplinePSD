"""Deterministic contracts for the bounded scientific experiment, not recovery CI."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model, log_density

from log_psplines.inference.model import (
    _blocked_channel_model,
    channel_model_kwargs,
    prepare_model,
)


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2] / "examples/vi_nuts_validation"
        )
    )
    import stationary

    return stationary


def test_ar4_stationary_covariance_and_sign(experiment):
    phi, matrix, covariance, innovation, radius = experiment.ar4_definition()
    assert radius < 1
    np.testing.assert_allclose(covariance[0, 0], 1, atol=1e-12)
    np.testing.assert_allclose(
        covariance,
        matrix @ covariance @ matrix.T + np.diag([innovation, 0, 0, 0]),
        atol=1e-12,
    )
    raw, initial, _, _ = experiment.simulate_ar4(32, 7)
    rng = np.random.default_rng(7)
    np.testing.assert_array_equal(
        initial, rng.multivariate_normal(np.zeros(4), covariance)
    )
    noise = rng.normal(scale=np.sqrt(innovation), size=32)
    state = initial.copy()
    for i in range(32):
        np.testing.assert_allclose(raw[i], phi @ state + noise[i], atol=1e-14)
        state = np.r_[raw[i], state[:-1]]
    f = np.linspace(0, 0.5, 10001)
    np.testing.assert_allclose(
        np.trapezoid(experiment.ar4_psd(f), f), 1, atol=1e-8
    )


def test_full_record_fourier_normalization_and_endpoints(experiment):
    raw = np.arange(64, dtype=float) / 64 + np.sin(np.arange(64))
    data = experiment.frequency_observations(raw)
    np.testing.assert_array_equal(data.freq, np.arange(1, 32) / 64)
    power = data.u_re[:, 0, 0] ** 2 + data.u_im[:, 0, 0] ** 2
    np.testing.assert_allclose(
        power, 2 * abs(np.fft.rfft(raw)[1:-1]) ** 2, rtol=2e-12
    )
    assert data.Nb == data.Nh == 1
    assert data.enbw == 1
    assert data.duration == 64


def test_conditioned_density_gradients_and_packing(experiment):
    raw, _, _, _ = experiment.simulate_ar4(128, 8)
    data = experiment.frequency_observations(raw)
    kwargs, components = prepare_model(data, experiment.native_config(20))
    base = partial(_blocked_channel_model, **channel_model_kwargs(kwargs, 0))
    sigma = jnp.asarray(0.6744897501960817 * 1.28)
    fixed = experiment.conditional_model(base, sigma)
    init = {
        "sigma_delta_0": sigma,
        "weights_delta_0": jnp.linspace(
            -0.1, 0.1, components.diagonal_models[0].n_basis
        ),
    }

    def hierarchical(c):
        return log_density(base, (), {}, {**init, "weights_delta_0": c})[0]

    def conditional(c):
        return log_density(fixed, (), {}, {"weights_delta_0": c})[0]

    for offset in (0, 0.2):
        c = init["weights_delta_0"] + offset
        np.testing.assert_allclose(conditional(c), hierarchical(c), rtol=1e-12)
        np.testing.assert_allclose(
            jax.grad(conditional)(c),
            jax.grad(hierarchical)(c),
            rtol=1e-12,
            atol=1e-10,
        )
    for cls in (AutoDiagonalNormal, AutoMultivariateNormal):
        guide = cls(fixed, init_loc_fn=init_to_value(values=init))
        numpyro.handlers.seed(guide, 0)()
        assert list(guide._init_locs) == ["weights_delta_0"]
        assert guide.latent_dim == 20
    starts = []
    settings = {
        "nuts_init_coefficient_jitter_scale": 0.2,
        "nuts_init_log_sigma_jitter_sd": 0.1,
    }
    arrays = {"basis": components.diagonal_models[0].basis}
    for seed in range(4):
        initialized = initialize_model(
            jax.random.PRNGKey(seed),
            fixed,
            init_strategy=experiment.init_strategy(init, arrays, settings),
        )
        starts.append(np.asarray(initialized.param_info.z["weights_delta_0"]))
    assert all(
        np.any(starts[i] != starts[j]) for i in range(4) for j in range(i)
    )


def test_precision_status_is_separate_from_point_screen(experiment):
    assert (
        experiment.screen_interval(0.1005, 0.01, -0.1, 0.1)
        == "mc_precision_limited"
    )
    assert experiment.screen_interval(0.2, 0.01, -0.1, 0.1) == "outside_screen"
    assert experiment.screen_interval(0.01, 0.01, -0.1, 0.1) == "within_screen"
    assert experiment.screen_interval(np.nan, 0.01, -0.1, 0.1) == "unavailable"
