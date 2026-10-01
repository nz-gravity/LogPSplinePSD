"""Regression checks for stationary HalfNormal spline smoothing."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr
from jax.scipy.linalg import solve_triangular
from numpyro.infer.util import log_density
from scipy.stats import halfnorm

from log_psplines import StationaryConfig, fit
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.inference.log_likelihood import compute_pointwise_lnl
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.inference.model import (
    _blocked_channel_model,
    _sample_pspline_block,
    channel_model_kwargs,
    prepare_model,
)
from log_psplines.preprocessing.spectral import preprocess_to_freq_domain


def test_stationary_model_samples_sigma():
    assert StationaryConfig().roughness_scale == 1.28

    def model():
        _sample_pspline_block("sigma_delta_0", "weights_delta_0", jnp.eye(3), 1.28)

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(2))
    ).get_trace()
    assert {"sigma_delta_0", "weights_delta_0"} <= trace.keys()
    assert np.isfinite(log_density(model, (), {}, {
        "sigma_delta_0": jnp.array(1.0),
        "weights_delta_0": jnp.zeros(3),
    })[0])


def test_sigma_controls_the_existing_stationary_penalty():
    penalty = jnp.array([[2.0, 0.2], [0.2, 1.0]])
    weights = jnp.array([0.5, -0.3])

    def model():
        _sample_pspline_block("sigma_delta_0", "weights_delta_0", penalty, 1.28)

    def density(sigma: float) -> float:
        return float(log_density(model, (), {}, {
            "sigma_delta_0": jnp.array(sigma), "weights_delta_0": weights,
        })[0])

    low, high = 0.5, 2.0
    roughness = float(weights @ penalty @ weights)
    expected = (
        halfnorm.logpdf(low, scale=1.28)
        - halfnorm.logpdf(high, scale=1.28)
        - len(weights) * np.log(low / high)
        - 0.5 * roughness * (low**-2 - high**-2)
    )
    np.testing.assert_allclose(density(low) - density(high), expected, atol=1e-6)


def test_stationary_pipeline_uses_sigma():
    example = VARMAData.ar(order=1, n_samples=512, fs=64.0, seed=8)
    config = StationaryConfig(
        n_knots=5, Nb=4, n_warmup=20, n_samples=20,
        rng_key=11, verbose=False,
    )
    data = preprocess_to_freq_domain(example.ts, config)
    result = fit(data, config)
    assert np.isfinite(result.psd).all() and (result.psd > 0).all()
    assert "sigma_delta_0" in result.posterior
    assert "delta_0" not in result.posterior
    assert not any(name.startswith("phi_") for name in result.posterior)
    assert "sigma_delta_0" in result.to_arviz()["posterior"].dataset



def test_multichannel_sigma_keeps_spectral_matrix_invariants():
    example = VARMAData(n_samples=512, fs=64.0, seed=9)
    config = StationaryConfig(
        n_knots=5, Nb=4, n_warmup=20, n_samples=20,
        rng_key=12, verbose=False,
    )
    data = preprocess_to_freq_domain(example.ts, config)
    result = fit(data, config)
    for name in ("sigma_delta_1", "sigma_theta_re_1_0", "sigma_theta_im_1_0"):
        assert name in result.posterior
    assert "delta_1" not in result.posterior
    spectrum = np.asarray(result.spectral_density)
    np.testing.assert_allclose(spectrum, spectrum.conj().swapaxes(-1, -2))
    assert np.linalg.eigvalsh(spectrum).min() > 0
    assert np.all((result.coherence >= 0) & (result.coherence <= 1 + 1e-10))


def test_noncentered_prior_has_same_density_after_jacobian():
    penalty = jnp.array([[2.0, 0.2], [0.2, 1.0]])

    def model(parameterization: str):
        _sample_pspline_block(
            "sigma_delta_0", "weights_delta_0", penalty, 1.28,
            smoothing_parameterization=parameterization,
        )

    cholesky = np.linalg.cholesky(np.asarray(penalty))
    differences = []
    for sigma, z in ((0.3, np.array([0.5, -1.0])), (2.0, np.array([-0.2, 0.7]))):
        weights = sigma * np.linalg.solve(cholesky.T, z)
        centered = log_density(model, ("centered",), {}, {
            "sigma_delta_0": jnp.array(sigma),
            "weights_delta_0": jnp.asarray(weights),
        })[0]
        noncentered = log_density(model, ("noncentered",), {}, {
            "sigma_delta_0": jnp.array(sigma),
            "weights_delta_0_raw": jnp.asarray(z),
        })[0]
        log_jacobian = 2 * np.log(sigma) - np.log(np.linalg.det(cholesky))
        differences.append(float(centered - noncentered + log_jacobian))
    np.testing.assert_allclose(differences[0], differences[1], atol=1e-5)


def test_noncentered_pipeline_exposes_sampled_coordinates():
    example = VARMAData.ar(order=1, n_samples=512, fs=64.0, seed=8)
    result = fit(example.ts, StationaryConfig(
        smoothing_parameterization="noncentered", n_knots=5, Nb=4,
        n_warmup=20, n_samples=20, rng_key=11, verbose=False,
    ))
    assert "weights_delta_0_raw" in result.posterior
    assert "weights_delta_0" in result.posterior
    assert np.isfinite(result.psd).all()


def test_stationary_ridged_penalty_whitening_preserves_quadratic_form():
    example = VARMAData.ar(order=1, n_samples=512, fs=64.0, seed=8)
    config = StationaryConfig(n_knots=5, Nb=4, verbose=False)
    data = preprocess_to_freq_domain(example.ts, config)
    _, spline = prepare_model(data, config)
    penalty = jnp.asarray(spline.diagonal_models[0].penalty_matrix)
    z = jnp.arange(1, penalty.shape[0] + 1, dtype=penalty.dtype) / 5
    weights = solve_triangular(jnp.linalg.cholesky(penalty).T, z, lower=False)
    np.testing.assert_allclose(
        float(weights @ penalty @ weights), float(z @ z), rtol=5e-3
    )


def test_centered_and_noncentered_full_log_posterior_match():
    example = VARMAData.ar(order=1, n_samples=512, fs=64.0, seed=8)
    centered = StationaryConfig(n_knots=5, Nb=4, verbose=False)
    data = preprocess_to_freq_domain(example.ts, centered)
    kwargs_c, _ = prepare_model(data, centered)
    kwargs_n, _ = prepare_model(
        data, replace(centered, smoothing_parameterization="noncentered")
    )
    penalty = np.asarray(kwargs_c["penalties_delta"][0])
    cholesky = np.linalg.cholesky(penalty)
    differences = []
    for sigma, weights in ((0.4, np.zeros(penalty.shape[0])),
                           (0.8, np.linspace(-0.1, 0.1, penalty.shape[0]))):
        raw = cholesky.T @ weights / sigma
        params_c = {"sigma_delta_0": jnp.array(sigma),
                    "weights_delta_0": jnp.asarray(weights)}
        params_n = {"sigma_delta_0": jnp.array(sigma),
                    "weights_delta_0_raw": jnp.asarray(raw)}
        density_c = log_density(
            _blocked_channel_model, (), channel_model_kwargs(kwargs_c, 0), params_c
        )[0]
        density_n = log_density(
            _blocked_channel_model, (), channel_model_kwargs(kwargs_n, 0), params_n
        )[0]
        log_jacobian = len(weights) * np.log(sigma) - np.log(np.linalg.det(cholesky))
        differences.append(float(density_c - density_n + log_jacobian))
    np.testing.assert_allclose(differences[0], differences[1], atol=0.05)


def test_pointwise_likelihood_matches_sampled_channel_likelihoods():
    config = StationaryConfig(
        n_knots=5, Nb=4, verbose=False, wishart_window="hann"
    )
    data = preprocess_to_freq_domain(
        VARMAData(n_samples=128, fs=32.0, seed=14).ts, config
    )
    kwargs, _ = prepare_model(data, config)
    rng = np.random.default_rng(15)
    variables = {}
    for channel in range(2):
        name = f"weights_delta_{channel}"
        basis = kwargs["bases_delta"][channel]
        variables[name] = (
            ("chain", "draw", f"k_{name}"),
            rng.normal(scale=0.1, size=(1, 2, basis.shape[1])),
        )
        for kind in ("re", "im"):
            for index, basis in enumerate(
                kwargs[f"bases_theta_{kind}"][channel]
            ):
                name = f"weights_theta_{kind}_{channel}_{index}"
                variables[name] = (
                    ("chain", "draw", f"k_{name}"),
                    rng.normal(scale=0.1, size=(1, 2, basis.shape[1])),
                )
    posterior = xr.Dataset(variables, coords={"chain": [0], "draw": [0, 1]})
    likelihood = compute_pointwise_lnl(
        posterior=posterior, data=data, model_kwargs=kwargs
    )
    for draw in range(2):
        total = 0.0
        for channel in range(2):
            variance = (
                kwargs["bases_delta"][channel]
                @ posterior[f"weights_delta_{channel}"].values[0, draw]
            )
            theta = {}
            for kind in ("re", "im"):
                parts = [
                    basis
                    @ posterior[
                        f"weights_theta_{kind}_{channel}_{index}"
                    ].values[0, draw]
                    for index, basis in enumerate(
                        kwargs[f"bases_theta_{kind}"][channel]
                    )
                ]
                theta[kind] = (
                    np.stack(parts, axis=-1)
                    if parts
                    else np.zeros((data.N, 0))
                )
            expected = float(
                wishart_log_likelihood(
                    jnp.asarray(variance),
                    jnp.asarray(theta["re"]),
                    jnp.asarray(theta["im"]),
                    kwargs["u_re"][:, channel, :],
                    kwargs["u_im"][:, channel, :],
                    kwargs["u_re"][:, :channel, :],
                    kwargs["u_im"][:, :channel, :],
                    Nb=kwargs["Nb"],
                    Nh=kwargs["Nh"],
                    duration=kwargs["duration"],
                    enbw=kwargs["enbw"],
                )
            )
            actual = (
                likelihood[f"log_likelihood_channel_{channel}"]
                .values[0, draw]
                .sum()
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-6)
            total += expected
        np.testing.assert_allclose(
            likelihood.log_likelihood.values[0, draw].sum(), total, rtol=2e-6
        )
