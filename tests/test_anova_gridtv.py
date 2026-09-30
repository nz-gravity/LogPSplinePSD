"""GridTV ANOVA identification, reference and old WDM mathematics."""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpyro import handlers

from log_psplines import (
    ANOVALogPSpline,
    PowerConfig,
    PowerData,
    PowerPartition,
    SplineBasis,
    coarse_grain_power,
    fit,
)
from log_psplines.basis.penalty import whiten_penalty_pair
from log_psplines.inference.anova_power import (
    initialize_anova,
    prepare_anova_power_model,
)
from log_psplines.inference.power import fit_power
from log_psplines.models.reconstruction import reconstruct_power_spectrum


def _model():
    time = np.linspace(0.0, 1.0, 8)
    frequency = np.linspace(0.1, 1.0, 7)
    return ANOVALogPSpline(
        SplineBasis.from_grid(frequency, 1), SplineBasis.from_grid(time, 1)
    )


def test_centered_basis_and_old_wdm_penalty_parity():
    model = _model()
    bt = np.asarray(model.time.basis)
    means = bt.mean(axis=0)
    expected = (bt - means[None, :])[:, :-1]
    np.testing.assert_allclose(model.time_basis, expected, atol=1e-12)
    np.testing.assert_allclose(model.time_basis.mean(axis=0), 0, atol=1e-12)
    assert model.time_basis.shape[1] == bt.shape[1] - 1
    assert np.linalg.matrix_rank(model.time_basis) == model.time_basis.shape[1]
    # wdm_psd used the leading penalty block. This equals C.T P C when
    # the integrated-derivative penalty annihilates the constant vector.
    np.testing.assert_allclose(
        model.time_penalty, np.asarray(model.time.penalty)[:-1, :-1], atol=1e-8
    )


def test_anova_components_and_reference_reconstruction():
    model = _model()
    kt, kf = model.time_basis.shape[1], model.frequency.basis.shape[1]
    rng = np.random.default_rng(8)
    weights_g = rng.normal(size=kf) / 4
    weights_eta = rng.normal(size=(kt, kf)) / 4
    g, eta = model.components(jnp.asarray(weights_g), jnp.asarray(weights_eta))
    np.testing.assert_allclose(np.asarray(eta).mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(model(weights_g, weights_eta)).mean(axis=0), g, atol=1e-6
    )
    zeros = np.zeros_like(weights_eta)
    np.testing.assert_allclose(
        model(weights_g, zeros), np.broadcast_to(g, eta.shape)
    )

    original_g = weights_g.copy()
    posterior = xr.Dataset(
        {
            "weights_g": (
                ("chain", "draw", "frequency_coefficient"),
                weights_g[None, None],
            ),
            "weights_eta": (
                ("chain", "draw", "time_coefficient", "frequency_coefficient"),
                weights_eta[None, None],
            ),
        }
    )
    reference = np.exp(np.linspace(-2, 2, eta.size).reshape(eta.shape))
    no_reference = (
        reconstruct_power_spectrum(posterior, model)
        .values[0, 0, :, :, 0, 0]
        .real
    )
    with_reference = (
        reconstruct_power_spectrum(posterior, model, reference=reference)
        .values[0, 0, :, :, 0, 0]
        .real
    )
    np.testing.assert_allclose(
        no_reference, np.exp(model(weights_g, weights_eta)), rtol=1e-6
    )
    np.testing.assert_allclose(
        with_reference, reference * no_reference, rtol=1e-6
    )
    posterior["weights_g"].values[:] = 0
    posterior["weights_eta"].values[:] = 0
    np.testing.assert_allclose(
        reconstruct_power_spectrum(posterior, model, reference=reference)
        .values[0, 0, :, :, 0, 0]
        .real,
        reference,
    )
    posterior["weights_g"].values[:] = original_g
    stationary = (
        reconstruct_power_spectrum(posterior, model, reference=reference)
        .values[0, 0, :, :, 0, 0]
        .real
    )
    np.testing.assert_allclose(
        stationary, reference * np.exp(np.asarray(g)[None, :])
    )


def test_reference_is_normalized_before_pooling_and_anova_fit():
    model = _model()
    shape = (len(model.time.grid), len(model.frequency.grid))
    reference = np.exp(np.linspace(-3, 3, np.prod(shape)).reshape(shape))
    correction = 0.15
    power = reference * np.exp(correction)
    data = PowerData(power, 1, model.frequency.grid, model.time.grid)
    partition = PowerPartition(np.array([0, 3, 6]), np.array([0, 3, 5]))
    normalized = coarse_grain_power(
        PowerData(power / reference, 1, data.frequency, data.time), partition
    )
    # Use a varying correction to distinguish the two orderings.
    varied_power = power * (1 + np.arange(shape[1])[None, :] / 3)
    varied_data = PowerData(varied_power, 1, data.frequency, data.time)
    correct = coarse_grain_power(
        PowerData(varied_power / reference, 1, data.frequency, data.time),
        partition,
    ).power
    wrong = (
        coarse_grain_power(varied_data, partition).power
        / coarse_grain_power(
            PowerData(reference, 1, data.frequency, data.time), partition
        ).power
        * normalized.counts
    )
    assert np.max(np.abs(correct - wrong)) > 0.1
    np.testing.assert_allclose(
        normalized.power, normalized.counts * np.exp(correction)
    )
    result = fit(
        varied_data,
        PowerConfig(n_warmup=2, n_samples=2, progress_bar=False),
        model=model,
        reference=reference,
        partition=partition,
    )
    assert result.spectrum.dims == (
        "chain",
        "draw",
        "time",
        "frequency",
        "channel",
        "channel_aux",
    )
    assert (
        result.metadata["reference_normalization"]
        == "native_power_divided_before_pooling"
    )
    np.testing.assert_allclose(result.observed_data["power"].values, correct)
    assert {
        "g",
        "eta",
        "sigma_g",
        "sigma_eta",
        "weights_g",
        "weights_eta",
    } <= set(result.posterior)
    assert "phi_g" not in result.posterior
    assert np.isfinite(result.psd).all() and np.all(result.psd > 0)


def test_anova_initialization_has_separate_sites():
    model = _model()
    shape = (len(model.time.grid), len(model.frequency.grid))
    data = PowerData(np.ones(shape), 1, model.frequency.grid, model.time.grid)
    pair = whiten_penalty_pair(model.time_penalty, model.frequency.penalty)
    init = initialize_anova(
        data,
        model.time_basis,
        np.asarray(model.frequency.basis),
        model.time_penalty,
        np.asarray(model.frequency.penalty),
        pair,
        PowerConfig(),
    )
    assert np.isfinite(init["g"]).all()
    assert np.isfinite(init["eta"]).all()
    assert init["sigma_eta"] > 0
    energy = float(np.sum(pair["lam_freq"] * np.asarray(init["g"]) ** 2))
    old_precision = max(1e-2, np.asarray(init["g"]).size / (energy + 1e-6))
    np.testing.assert_allclose(init["sigma_g"], old_precision**-0.5)


def test_anova_prior_scales_match_old_nested_hierarchy():
    model = _model()
    pair = whiten_penalty_pair(model.time_penalty, model.frequency.penalty)
    lam_t, lam_f = pair["lam_time"], pair["lam_freq"]
    null_f = lam_f <= 1e-10 * max(lam_f.max(), 1.0)
    sigma_g, null_precision, ridge = 2.0**-0.5, 1e-4, 1e-6
    g_scale = np.where(
        null_f,
        null_precision**-0.5,
        sigma_g / np.sqrt(lam_f + ridge * sigma_g**2),
    )
    eta_shape = np.where(
        pair["joint_null"],
        1.0,
        (lam_t[:, None] + lam_f[None, :] + ridge) ** -0.5,
    )
    # These are the old wdm_psd stationary_scale and interaction_scale.
    np.testing.assert_allclose(
        g_scale[~null_f], (2.0 * lam_f[~null_f] + ridge) ** -0.5
    )
    np.testing.assert_allclose(g_scale[null_f], null_precision**-0.5)
    np.testing.assert_allclose(eta_shape[pair["joint_null"]], 1.0)
    np.testing.assert_allclose(0.0 * eta_shape, 0.0)
    assert np.all(g_scale > 0) and np.all(eta_shape > 0)
    data = PowerData(
        np.ones((len(model.time.grid), len(model.frequency.grid))),
        1,
        model.frequency.grid,
        model.time.grid,
    )
    sample_model, _, init = prepare_anova_power_model(
        data, model, PowerConfig(centered=True)
    )
    for sigma_eta in (0.5, 1e-5):
        sites = {**init, "sigma_g": sigma_g, "sigma_eta": sigma_eta}
        trace = handlers.trace(
            handlers.substitute(handlers.seed(sample_model, 8), data=sites)
        ).get_trace()
        np.testing.assert_allclose(trace["g"]["fn"].scale, g_scale, rtol=1e-6)
        np.testing.assert_allclose(
            trace["eta"]["fn"].scale,
            (sigma_eta * eta_shape).reshape(-1),
            rtol=1e-6,
        )


def test_native_likelihood_uses_supplied_basis_matrices():
    model = _model()
    # Historical breakpoints bases must not be reevaluated as clamped knots.
    model = ANOVALogPSpline(
        SplineBasis.from_knots(model.frequency.grid, np.linspace(0, 1, 5)),
        model.time,
    )
    power = np.ones((len(model.time.grid), len(model.frequency.grid)))
    data = PowerData(power, 1, model.frequency.grid, model.time.grid)
    sample_model, pair, init = prepare_anova_power_model(
        data, model, PowerConfig(centered=True)
    )
    rng = np.random.default_rng(9)
    g = rng.normal(size=init["g"].shape) * 0.1
    eta = rng.normal(size=init["eta"].shape) * 0.1
    trace = handlers.trace(
        handlers.substitute(
            handlers.seed(sample_model, 9), data={**init, "g": g, "eta": eta}
        )
    ).get_trace()
    correction = np.asarray(
        model(
            pair["U_freq"] @ g,
            pair["U_time"]
            @ eta.reshape(len(pair["lam_time"]), -1)
            @ pair["U_freq"].T,
        )
    )
    expected = -0.5 * np.sum(correction + power * np.exp(-correction))
    np.testing.assert_allclose(
        trace["log_likelihood"]["value"], expected, rtol=1e-6
    )


def test_anova_rejects_scattered_and_invalid_reference():
    model = _model()
    config = PowerConfig(n_warmup=1, n_samples=1, progress_bar=False)
    scattered = PowerData(
        np.ones(3), 1, np.array([0.1, 0.4, 0.7]), np.array([0.0, 0.4, 0.7])
    )
    with pytest.raises(ValueError, match="rectangular"):
        fit_power(scattered, model, config)
    shape = (len(model.time.grid), len(model.frequency.grid))
    grid = PowerData(np.ones(shape), 1, model.frequency.grid, model.time.grid)
    with pytest.raises(ValueError, match="finite, positive"):
        fit_power(grid, model, config, reference=np.zeros(shape))
