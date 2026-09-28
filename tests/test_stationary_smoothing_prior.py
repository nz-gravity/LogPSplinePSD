"""Regression checks for stationary HalfNormal spline smoothing."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.linalg import solve_triangular
from numpyro.infer.util import log_density
from scipy.stats import halfnorm

from log_psplines import StationaryConfig, fit
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.inference.evidence import (
    _build_log_posterior,
    _posterior_param_names,
)
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


def test_stationary_pipeline_uses_sigma_and_evidence_density():
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

    kwargs, _ = prepare_model(data, config)
    names = _posterior_param_names(result.posterior, channel_index=0)
    assert names == ["sigma_delta_0", "weights_delta_0"]
    samples, log_prob, evaluate = _build_log_posterior(
        result.posterior,
        model_fn=_blocked_channel_model,
        model_kwargs=channel_model_kwargs(kwargs, 0),
        param_names=names,
    )
    assert samples.shape[0] == 20
    assert np.isfinite(log_prob).all()
    assert np.isfinite(evaluate(samples[0]))


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
    kwargs, _ = prepare_model(data, config)
    names = _posterior_param_names(result.posterior, channel_index=1)
    assert len(names) == 6
    samples, log_prob, _ = _build_log_posterior(
        result.posterior,
        model_fn=_blocked_channel_model,
        model_kwargs=channel_model_kwargs(kwargs, 1),
        param_names=names,
    )
    assert samples.shape[0] == 20
    assert np.isfinite(log_prob).all()


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


def test_noncentered_pipeline_exposes_sampled_coordinates_for_evidence():
    example = VARMAData.ar(order=1, n_samples=512, fs=64.0, seed=8)
    result = fit(example.ts, StationaryConfig(
        smoothing_parameterization="noncentered", n_knots=5, Nb=4,
        n_warmup=20, n_samples=20, rng_key=11, verbose=False,
    ))
    assert "weights_delta_0_raw" in result.posterior
    assert "weights_delta_0" in result.posterior
    assert _posterior_param_names(result.posterior, channel_index=0) == [
        "sigma_delta_0", "weights_delta_0_raw",
    ]
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
