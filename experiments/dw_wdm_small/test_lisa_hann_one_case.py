"""Direct transforms and independent paired-predictor/gradient checks."""

import jax
import jax.numpy as jnp
import numpy as np
from experiments.dw_wdm_small.lisa_hann_model import prepare_paired_model
from experiments.dw_wdm_small.lisa_hann_one_case import SelectedHann

from log_psplines import ANOVALogPSpline, PowerConfig, PowerData, SplineBasis
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood


def test_fast_hann_selected_transform_agrees_with_every_direct_window():
    m = 64
    rng = np.random.default_rng(20261015)
    x = rng.normal(size=3 * m)
    engine = SelectedHann(m)
    direct = np.array(
        [
            x[j : j + 2 * m + 1]
            @ (
                engine.h
                * np.exp(
                    -2j * np.pi * (j + 1) * np.arange(2 * m + 1) / (2 * m + 1)
                )
            )
            for j in range(m)
        ]
    )
    np.testing.assert_allclose(engine.block(x), direct, atol=2e-11)


def test_paired_reference_predictor_and_gradient_preserve_exact_coordinates():
    jax.config.update("jax_enable_x64", True)
    frequency = np.linspace(0.01, 0.1, 8)
    cycle = np.array([0.2, 0.5, 0.8])
    time = (cycle[:, None] + np.linspace(-0.01, 0.01, len(frequency))).ravel()
    f = np.tile(frequency, len(cycle))
    data = PowerData(np.linspace(0.1, 4, len(time)), 2, f, time)
    spline = ANOVALogPSpline(
        SplineBasis.from_grid(frequency, 2),
        SplineBasis.from_grid(np.linspace(0, 1, 30), 2),
    )
    _, pair, _, evaluate = prepare_paired_model(
        data, spline, PowerConfig(centered=True), cycle
    )
    bt, bf = spline.design(time, f)
    bt, bf = jnp.asarray(bt @ pair["U_time"]), jnp.asarray(bf @ pair["U_freq"])
    rng = np.random.default_rng(4)
    g = jnp.asarray(rng.normal(size=bf.shape[1]))
    eta = jnp.asarray(rng.normal(size=(bt.shape[1], bf.shape[1])))
    expected = bf @ g + jnp.einsum("qi,ij,qj->q", bt, eta, bf)
    np.testing.assert_allclose(evaluate(g, eta), expected, atol=1e-13)

    def objective(value):
        return power_whittle_log_likelihood(
            data.power, data.counts, evaluate(g, value)
        )

    def reference(value):
        return power_whittle_log_likelihood(
            data.power,
            data.counts,
            bf @ g + jnp.einsum("qi,ij,qj->q", bt, value, bf),
        )

    np.testing.assert_allclose(
        jax.grad(objective)(eta), jax.grad(reference)(eta), atol=1e-12
    )


def test_noncentered_adapter_has_identical_physical_prior_and_likelihood():
    from numpyro.infer.util import log_density

    jax.config.update("jax_enable_x64", True)
    f = np.linspace(0.01, 0.1, 8)
    t = np.array([0.2, 0.5, 0.8])
    data = PowerData(
        np.linspace(0.1, 4, 24), 2, np.tile(f, 3), np.repeat(t, 8)
    )
    spline = ANOVALogPSpline(
        SplineBasis.from_grid(f, 2),
        SplineBasis.from_grid(np.linspace(0, 1, 30), 2),
    )
    config = PowerConfig(centered=True)
    centered, pair, init, _ = prepare_paired_model(data, spline, config, t)
    transformed, _, init_z, _ = prepare_paired_model(
        data, spline, config, t, noncentered_eta=True
    )
    log_centered = log_density(centered, (), {}, init)[0]
    log_z = log_density(transformed, (), {}, init_z)[0]
    jacobian = np.sum(np.log(init["sigma_eta"] * pair["eta_scale"]))
    np.testing.assert_allclose(log_z, log_centered + jacobian, atol=2e-10)
