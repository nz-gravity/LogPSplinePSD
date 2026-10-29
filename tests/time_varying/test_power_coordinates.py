"""Exact scalar priors and likelihoods under automatic coordinate changes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from log_psplines import (
    ANOVALogPSpline,
    LogPSpline,
    PowerConfig,
    PowerData,
    SplineBasis,
)
from log_psplines.inference.anova_power import prepare_anova_power_model
from log_psplines.inference.power import prepare_power_model


def _fixture(kind):
    time, frequency = np.linspace(0, 1, 5), np.linspace(0.1, 1, 6)
    bt = SplineBasis.from_grid(time, 1, degree=1, penalty_order=1)
    bf = SplineBasis.from_grid(frequency, 1, degree=1, penalty_order=1)
    counts = (np.arange(30).reshape(5, 6) % 3 + 1).astype(float)
    counts[0] = 0
    counts[3, 2] = 0
    power = counts * np.exp(0.2 + 0.1 * time[:, None] * frequency[None])
    if kind == "paired":
        t, f = np.meshgrid(time, frequency, indexing="ij")
        active = counts > 0
        data = PowerData(power[active], counts[active], f[active], t[active])
    else:
        data = PowerData(power, counts, frequency, time)
    spline = (
        ANOVALogPSpline(bf, bt, sigma_eta_prior=0.3)
        if kind == "anova"
        else LogPSpline(bf, bt)
    )
    config = PowerConfig(
        roughness_scale=1.4, null_precision=0.02, ridge_eps=1e-5
    )
    if kind == "anova":
        model, pair, init = prepare_anova_power_model(data, spline, config)
        time_design, frequency_design = spline.design(
            data.time, data.frequency
        )
    else:
        model, init, pair = prepare_power_model(data, spline, config)
        time_design = bt.design_at(data.time)
        frequency_design = bf.design_at(data.frequency)
    te = np.asarray(time_design) @ pair["U_time"]
    fe = np.asarray(frequency_design) @ pair["U_freq"]
    field = (
        np.array([np.kron(t, f) for t, f in zip(te, fe, strict=True)])
        if kind == "paired"
        else np.kron(te, fe)
    )
    return data, spline, config, model, pair, init, field, fe


@pytest.mark.parametrize("kind", ["grid", "paired", "anova"])
def test_scalar_information_matches_expected_likelihood_hessian(kind):
    data, _, _, _, pair, _, field, fe = _fixture(kind)
    expected = 0.5 * np.sum(data.counts.reshape(-1, 1) * field**2, axis=0)
    actual = pair["information_eta" if kind == "anova" else "information"]
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1), expected, rtol=2e-13
    )
    if kind == "anova":
        mean_design = np.tile(fe, (len(data.time), 1))
        expected_mean = 0.5 * np.sum(
            data.counts.reshape(-1, 1) * mean_design**2, axis=0
        )
        np.testing.assert_allclose(
            pair["information_g"], expected_mean, rtol=2e-13
        )


def _normal_log_density(value, scale):
    return jnp.sum(
        -0.5 * (value / scale) ** 2
        - jnp.log(scale)
        - 0.5 * jnp.log(2 * jnp.pi)
    )


@pytest.mark.parametrize("kind", ["grid", "paired", "anova"])
@pytest.mark.parametrize("sigma", [1e-7, 0.6, 8.0])
def test_scalar_full_target_and_scale_gradients_include_exact_jacobian(
    kind, sigma
):
    data, spline, config, model, pair, init, field, fe = _fixture(kind)
    lambda_t = jnp.asarray(pair["lam_time"])
    lambda_f = jnp.asarray(pair["lam_freq"])
    null = jnp.asarray(pair["joint_null"])
    # Fix raw coordinates; compare both full densities and derivatives in the
    # two smoothing scales, including the transformation's changing Jacobian.
    raw = {
        name: jnp.linspace(-0.15, 0.2, np.size(value))
        for name, value in init.items()
        if name.startswith("z_")
    }

    def difference(log_sigma):
        sf, st = jnp.exp(log_sigma)
        if kind == "anova":
            params = {**raw, "sigma_g": sf, "sigma_eta": st}
            scale_g = jnp.where(
                pair["null_freq"],
                config.null_precision**-0.5,
                (lambda_f / sf**2 + config.ridge_eps) ** -0.5,
            )
            scale_eta = st * jnp.where(
                null,
                1.0,
                (lambda_t[:, None] + lambda_f[None] + config.ridge_eps)
                ** -0.5,
            ).reshape(-1)
            cg = (scale_g**-2 + pair["information_g"]) ** -0.5
            ce = (scale_eta**-2 + pair["information_eta"].reshape(-1)) ** -0.5
            g, eta = raw["z_g"] * cg, raw["z_eta"] * ce
            logs = jnp.tile(fe @ g, len(data.time)) + field @ eta
            prior = _normal_log_density(g, scale_g) + _normal_log_density(
                eta, scale_eta
            )
            jacobian = jnp.log(cg).sum() + jnp.log(ce).sum()
            scales, hyper_widths = (
                (sf, st),
                (config.roughness_scale, spline.sigma_eta_prior),
            )
        else:
            params = {**raw, "sigma_freq": sf, "sigma_time": st}
            precision = lambda_t[:, None] / st**2 + lambda_f[None] / sf**2
            scale = jnp.where(
                null,
                config.null_precision**-0.5,
                (precision + config.ridge_eps) ** -0.5,
            ).reshape(-1)
            c = (scale**-2 + pair["information"].reshape(-1)) ** -0.5
            physical = raw["z_s"] * c
            logs = field @ physical
            prior = _normal_log_density(physical, scale)
            jacobian = jnp.log(c).sum()
            scales, hyper_widths = (
                (sf, st),
                (config.roughness_scale, config.roughness_scale),
            )
        for scale, width in zip(scales, hyper_widths, strict=True):
            prior += jnp.log(2) + _normal_log_density(scale, width)
        likelihood = -0.5 * jnp.sum(
            data.counts.reshape(-1) * logs
            + data.power.reshape(-1) * jnp.exp(-logs)
        )
        actual, _ = log_density(model, (), {}, params)
        return actual - (prior + likelihood + jacobian)

    point = jnp.log(jnp.array([sigma, 0.4 * sigma]))
    np.testing.assert_allclose(difference(point), 0, atol=1e-9)
    np.testing.assert_allclose(jax.grad(difference)(point), 0, atol=1e-9)
