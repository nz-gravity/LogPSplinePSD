"""Independent Gaussian identities for collapsed Cholesky regression."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from log_psplines.inference.collapsed import (
    _conditional_kwargs,
    collapsed_channel_model,
    gaussian_theta_conditional,
    prepare_collapsed_channel,
    sample_conditional_theta,
)
from log_psplines.likelihoods.wishart import wishart_log_likelihood


def _example(log_scale=0.0):
    rng = np.random.default_rng(83)
    n, replicates, j = 5, 3, 2
    scale = np.exp(log_scale / 2)
    bases, penalties = {}, {}
    for part, counts in (("re", (2, 1)), ("im", (3, 2))):
        bases[part] = tuple(rng.normal(size=(n, k)) for k in counts)
        penalties[part] = []
        for k in counts:
            m = rng.normal(size=(k, k))
            penalties[part].append(m @ m.T + np.eye(k))
    return {
        "channel_index": j,
        "u_re_channel": jnp.asarray(scale * rng.normal(size=(n, replicates))),
        "u_im_channel": jnp.asarray(scale * rng.normal(size=(n, replicates))),
        "u_re_prev": jnp.asarray(scale * rng.normal(size=(n, j, replicates))),
        "u_im_prev": jnp.asarray(scale * rng.normal(size=(n, j, replicates))),
        "basis_delta": jnp.asarray(rng.normal(size=(n, 3))),
        "penalty_delta": jnp.eye(3),
        "basis_theta_re_by_component": bases["re"],
        "penalty_theta_re_by_component": tuple(penalties["re"]),
        "basis_theta_im_by_component": bases["im"],
        "penalty_theta_im_by_component": tuple(penalties["im"]),
        "roughness_scale": 1.28,
        "Nb": 4,
        "Nh": 3,
        "duration": 2.3,
        "eta": 0.37,
        "enbw": 1.5,
    }


def _independent_gaussian(kw, logd, sigmas):
    # Explicit complex regressors and covariance, assembled independently of
    # the implementation's real design and prior-precision representation.
    previous = np.asarray(kw["u_re_prev"]) + 1j * np.asarray(kw["u_im_prev"])
    y = np.asarray(kw["u_re_channel"]) + 1j * np.asarray(kw["u_im_channel"])
    columns, covariances = [], []
    index = 0
    for j in range(kw["channel_index"]):
        for part in ("re", "im"):
            b = np.asarray(kw[f"basis_theta_{part}_by_component"][j])
            p = np.asarray(kw[f"penalty_theta_{part}_by_component"][j])
            columns.append(
                previous[:, j, :, None]
                * b[:, None, :]
                * (1 if part == "re" else 1j)
            )
            covariances.append(sigmas[index] ** 2 * np.linalg.inv(p))
            index += 1
    complex_design = np.concatenate(columns, axis=-1)
    x = np.concatenate(
        [
            complex_design.real.reshape(-1, complex_design.shape[-1]),
            complex_design.imag.reshape(-1, complex_design.shape[-1]),
        ]
    )
    response = np.concatenate([y.real.ravel(), y.imag.ravel()])
    precision = np.tile(
        np.repeat(
            kw["eta"] / (kw["enbw"] * kw["duration"] * np.exp(logd)),
            y.shape[-1],
        ),
        2,
    )
    prior_covariance = block_diag(*covariances)
    conditional_covariance = np.linalg.inv(
        np.linalg.inv(prior_covariance) + 2 * x.T @ (precision[:, None] * x)
    )
    mean = conditional_covariance @ (2 * x.T @ (precision * response))
    return (
        mean,
        conditional_covariance,
        prior_covariance,
        x,
        response,
        precision,
    )


@pytest.mark.parametrize("log_scale", [-100.0, 0.0, 100.0])
def test_collapsed_integral_matches_full_joint_and_conditional_density(
    log_scale,
):
    kw = _example(log_scale)
    prepared = prepare_collapsed_channel(kw)
    for amplitude in (0.03, 0.7, 4.0):
        logd = np.linspace(-0.7, 0.8, 5) + log_scale
        sigmas = amplitude * np.array([1.0, 0.4, 1.3, 0.8])
        expected_mean, covariance, prior, x, y, precision = (
            _independent_gaussian(kw, logd, sigmas)
        )
        marginal, mean, cholesky, scale = gaussian_theta_conditional(
            jnp.asarray(logd),
            jnp.asarray(sigmas),
            **_conditional_kwargs(prepared),
        )
        beta = expected_mean + np.linspace(-0.1, 0.1, len(expected_mean))
        full = -kw["eta"] / kw["enbw"] * kw["Nb"] * kw["Nh"] * np.sum(
            logd
        ) - np.sum(precision * (y - x @ beta) ** 2)
        expected = (
            full
            + multivariate_normal.logpdf(beta, cov=prior)
            - multivariate_normal.logpdf(
                beta, mean=expected_mean, cov=covariance
            )
        )
        np.testing.assert_allclose(marginal, expected, atol=2e-10)
        np.testing.assert_allclose(mean, expected_mean, atol=2e-11)
        conditional_precision = (
            np.asarray(scale)[:, None]
            * (np.asarray(cholesky) @ np.asarray(cholesky).T)
            * np.asarray(scale)[None, :]
        )
        np.testing.assert_allclose(
            np.linalg.inv(conditional_precision), covariance, atol=2e-12
        )


@pytest.mark.parametrize("log_scale", [-100.0, 0.0, 100.0])
def test_marginal_gradients_match_independent_gaussian_identity(log_scale):
    kw = _example(log_scale)
    prepared = prepare_collapsed_channel(kw)
    point = np.r_[
        np.linspace(-0.7, 0.8, 5) + log_scale, np.log([0.3, 0.4, 0.8, 0.6])
    ]

    def target(v):
        return gaussian_theta_conditional(
            v[:5], jnp.exp(v[5:]), **_conditional_kwargs(prepared)
        )[0]

    def reference(v):
        logd, sigmas = v[:5], np.exp(v[5:])
        mean, covariance, prior, x, y, precision = _independent_gaussian(
            kw, logd, sigmas
        )
        # Evaluate the identity at beta=0, avoiding shared solve expressions.
        zero = np.zeros(len(mean))
        ll = -kw["eta"] / kw["enbw"] * kw["Nb"] * kw["Nh"] * np.sum(
            logd
        ) - np.sum(precision * y**2)
        return (
            ll
            + multivariate_normal.logpdf(zero, cov=prior)
            - multivariate_normal.logpdf(zero, mean=mean, cov=covariance)
        )

    step = 1e-5
    expected = np.array(
        [
            (
                reference(point + step * direction)
                - reference(point - step * direction)
            )
            / (2 * step)
            for direction in np.eye(len(point))
        ]
    )
    np.testing.assert_allclose(
        jax.grad(target)(jnp.asarray(point)), expected, atol=2e-8, rtol=2e-7
    )


@pytest.mark.parametrize("log_scale", [-100.0, 0.0, 100.0])
def test_conditional_draws_recover_moments_and_original_likelihood(log_scale):
    kw = _example(log_scale)
    kw["basis_delta"] = kw["basis_delta"].at[:, 0].set(1.0)
    prepared = prepare_collapsed_channel(kw)
    weights = np.array([0.1 + log_scale, -0.3, 0.15])
    sigmas = np.array([0.3, 0.4, 0.8, 0.6])
    shape = (2, 3000)
    posterior = {
        "weights_delta_2": np.broadcast_to(weights, (*shape, len(weights)))
    }
    posterior.update(
        {
            f"sigma_{name}": np.full(shape, sigma)
            for name, sigma in zip(
                prepared["theta_names"], sigmas, strict=True
            )
        }
    )
    output = sample_conditional_theta(
        posterior, prepared, jax.random.PRNGKey(44)
    )
    draws = np.concatenate(
        [
            np.asarray(output[f"weights_{name}"])
            for name in prepared["theta_names"]
        ],
        axis=-1,
    ).reshape(-1, sum(prepared["theta_sizes"]))
    logd = np.asarray(kw["basis_delta"]) @ weights
    mean, covariance, *_ = _independent_gaussian(kw, logd, sigmas)
    standard_error = np.sqrt(np.diag(covariance) / len(draws))
    assert np.max(np.abs(draws.mean(axis=0) - mean) / standard_error) < 4
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), covariance, atol=0.0025, rtol=0.08
    )
    theta = {}
    for part in ("re", "im"):
        theta[part] = jnp.stack(
            [
                jnp.asarray(kw[f"basis_theta_{part}_by_component"][j])
                @ output[f"weights_theta_{part}_2_{j}"][0, 0]
                for j in range(2)
            ],
            axis=-1,
        )
    expected = wishart_log_likelihood(
        jnp.asarray(logd),
        theta["re"],
        theta["im"],
        kw["u_re_channel"],
        kw["u_im_channel"],
        kw["u_re_prev"],
        kw["u_im_prev"],
        **{key: kw[key] for key in ("Nb", "Nh", "duration", "eta", "enbw")},
    )
    assert output["log_likelihood_block_2"].shape == shape
    previous = np.asarray(kw["u_re_prev"]) + 1j * np.asarray(kw["u_im_prev"])
    observed = np.asarray(kw["u_re_channel"]) + 1j * np.asarray(
        kw["u_im_channel"]
    )
    residual = observed - np.einsum(
        "fj,fjr->fr", theta["re"] + 1j * theta["im"], previous
    )
    direct = (
        -kw["eta"]
        / kw["enbw"]
        * (
            kw["Nb"] * kw["Nh"] * np.sum(logd)
            + np.sum(np.exp(-logd) * np.sum(abs(residual) ** 2, axis=-1))
            / kw["duration"]
        )
    )
    np.testing.assert_allclose(expected, direct, atol=1e-10)
    np.testing.assert_allclose(
        output["log_likelihood_block_2"][0, 0], expected, atol=1e-10
    )


def test_model_samples_same_scales_without_explicit_theta_weights():
    prepared = prepare_collapsed_channel(_example())
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(collapsed_channel_model, jax.random.PRNGKey(8))
    ).get_trace(**prepared)
    sample_names = {
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site.get("is_observed")
    }
    assert sample_names == {
        "weights_delta_2_raw",
        "sigma_delta_2",
        *(f"sigma_{name}" for name in prepared["theta_names"]),
    }


def test_ridged_spline_logdet_uses_its_positive_definite_factor():
    from log_psplines.basis.splines import SplineBasis

    kw = _example()
    spline = SplineBasis.from_knots(
        grid=np.linspace(0, 1, 5), knots=np.linspace(0, 1, 10)
    )
    for part in ("re", "im"):
        kw[f"basis_theta_{part}_by_component"] = (spline.basis,) * 2
        kw[f"penalty_theta_{part}_by_component"] = (spline.penalty,) * 2
    # Some BLAS/LAPACK configurations emit LU slogdet arithmetic warnings on
    # this nonsingular block diagonal matrix. The SPD factor is sufficient.
    with np.errstate(all="raise"):
        prepared = prepare_collapsed_channel(kw)
    expected = 4 * np.log(np.linalg.eigvalsh(spline.penalty)).sum()
    np.testing.assert_allclose(
        prepared["theta_logdet_penalty"], expected, atol=2e-8
    )
