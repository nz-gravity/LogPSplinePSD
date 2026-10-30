"""Independent Gaussian identities for collapsed Cholesky regression."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.infer.util import log_density
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from log_psplines.inference.collapsed import (
    _conditional_kwargs,
    _prepare_gram_cache,
    _prepare_linear_cache,
    collapsed_channel_model,
    gaussian_theta_conditional,
    prepare_collapsed_channel,
    sample_conditional_theta,
)
from log_psplines.inference.explicit import explicit_channel_model
from log_psplines.likelihoods.wishart import wishart_log_likelihood


def _example(n=5, replicates=3):
    rng = np.random.default_rng(83)
    j = 2
    bases, penalties = {}, {}
    for part, counts in (("re", (2, 1)), ("im", (3, 2))):
        bases[part] = tuple(rng.normal(size=(n, k)) for k in counts)
        penalties[part] = []
        for k in counts:
            m = rng.normal(size=(k, k))
            penalties[part].append(m @ m.T + np.eye(k))
    return {
        "channel_index": j,
        "u_re_channel": jnp.asarray(rng.normal(size=(n, replicates))),
        "u_im_channel": jnp.asarray(rng.normal(size=(n, replicates))),
        "u_re_prev": jnp.asarray(rng.normal(size=(n, j, replicates))),
        "u_im_prev": jnp.asarray(rng.normal(size=(n, j, replicates))),
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
            kw["eta"]
            / (kw["enbw"] * kw["duration"] * np.exp(np.clip(logd, -80, 80))),
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


def test_collapsed_integral_matches_full_joint_and_conditional_density():
    kw = _example()
    prepared = prepare_collapsed_channel(kw)
    for amplitude in (0.03, 0.7, 4.0):
        logd = np.linspace(-0.7, 0.8, 5)
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


def test_marginal_gradients_match_independent_gaussian_identity():
    kw = _example()
    prepared = prepare_collapsed_channel(kw)
    point = np.r_[np.linspace(-0.7, 0.8, 5), np.log([0.3, 0.4, 0.8, 0.6])]

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


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize("backend", ["dense", "banded"])
def test_explicit_and_marginalized_model_posteriors_match(
    parameterization, backend
):
    """The real models obey p(z, theta) / p(theta | z, y) = p(z)."""
    kw = jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        dict(
            _example(),
            smoothing_parameterization=parameterization,
            theta_backend=backend,
        ),
    )
    prepared = prepare_collapsed_channel(kw)
    sizes = prepared["theta_sizes"]
    names = prepared["theta_names"]
    factors = [
        np.linalg.cholesky(
            np.asarray(kw[f"penalty_theta_{part}_by_component"][i])
        )
        for i in range(2)
        for part in ("re", "im")
    ]
    prior_logdet = sum(2 * np.log(np.diag(factor)).sum() for factor in factors)
    constant = 0.5 * (prior_logdet - sum(sizes) * np.log(2 * np.pi))
    noncentered = parameterization == "noncentered"
    suffix = "_raw" if noncentered else ""

    def hyperparameters(v):
        return {
            f"weights_delta_2{suffix}": v[:3],
            "sigma_delta_2": jnp.exp(v[-1]),
            **{
                f"sigma_{name}": jnp.exp(v[3 + i])
                for i, name in enumerate(names)
            },
        }

    def target(v):
        return log_density(
            collapsed_channel_model, (), prepared, hyperparameters(v)
        )[0]

    def reference(v, beta):
        values = hyperparameters(jnp.asarray(v))
        sigmas = np.exp(v[3:7])
        offset = 0
        for name, size, factor, sigma in zip(
            names, sizes, factors, sigmas, strict=True
        ):
            block = beta[offset : offset + size]
            values[f"weights_{name}{suffix}"] = jnp.asarray(
                factor.T @ block / sigma if noncentered else block
            )
            offset += size
        joint = log_density(explicit_channel_model, (), kw, values)[0]
        delta_weights = np.exp(v[-1]) * v[:3] if noncentered else v[:3]
        mean, covariance, *_ = _independent_gaussian(
            kw, np.asarray(kw["basis_delta"]) @ delta_weights, sigmas
        )
        conditional = multivariate_normal.logpdf(
            beta, mean=mean, cov=covariance
        )
        if noncentered:
            # Transform the independently computed conditional density to
            # the raw coordinates actually sampled by the explicit model.
            conditional += (
                np.asarray(sizes) @ np.log(sigmas) - 0.5 * prior_logdet
            )
        # The centered spline helper omits a fixed Gaussian normalization;
        # it is independent of all sampled parameters and the posterior.
        return float(joint) - conditional + (0 if noncentered else constant)

    point = np.r_[[0.1, -0.3, 0.15], np.log([0.3, 0.4, 0.8, 0.6, 0.5])]
    beta = np.linspace(-0.2, 0.4, sum(sizes))
    for shift in (0, 0.2):
        v = point + shift
        for weights in (beta, -beta):
            np.testing.assert_allclose(
                target(jnp.asarray(v)),
                reference(v, weights),
                atol=2e-9,
                rtol=2e-11,
            )
    step = 1e-5
    expected = np.array(
        [
            (
                reference(point + step * direction, beta)
                - reference(point - step * direction, beta)
            )
            / (2 * step)
            for direction in np.eye(len(point))
        ]
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(target))(jnp.asarray(point)),
        expected,
        atol=2e-7,
        rtol=3e-7,
    )


def test_conditional_draws_recover_moments_and_original_likelihood():
    kw = _example()
    prepared = prepare_collapsed_channel(kw)
    weights = np.array([0.1, -0.3, 0.15])
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
        "weights_delta_2",
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


@pytest.mark.parametrize(
    "cache_kind", ["compact", "sparse", "none", "blocked"]
)
def test_cached_marginal_density_gradient_and_curvature(
    cache_kind, monkeypatch
):
    """Compare optimized derivatives with ordinary AD through the full solve."""
    import log_psplines.inference.collapsed as collapsed

    kw = _example(n=129 if cache_kind == "blocked" else 5)
    if cache_kind == "blocked":
        from log_psplines.basis.splines import SplineBasis

        for part, counts in (("re", (7, 9)), ("im", (8, 10))):
            splines = [
                SplineBasis.from_knots(
                    grid=np.linspace(0, 1, 129),
                    knots=np.linspace(0, 1, count),
                )
                for count in counts
            ]
            kw[f"basis_theta_{part}_by_component"] = tuple(
                spline.basis for spline in splines
            )
            kw[f"penalty_theta_{part}_by_component"] = tuple(
                spline.penalty for spline in splines
            )
    prepared = prepare_collapsed_channel(kw)
    kwargs = _conditional_kwargs(prepared)
    if cache_kind == "blocked":
        assert kwargs["gram_cache"][2].ndim == 3
        # This grid needs padding in the last frequency block.
        assert 129 % kwargs["gram_cache"][2].shape[-1] != 0
    if cache_kind == "sparse":
        pairs, _, by_pair = kwargs["gram_cache"]
        row, freq = np.nonzero(by_pair)
        kwargs["gram_cache"] = (
            pairs[row],
            jnp.asarray(freq),
            by_pair[row, freq],
        )
    elif cache_kind == "none":
        kwargs["gram_cache"] = None
        kwargs["linear_cache"] = None
    point = jnp.asarray(np.r_[[0.1, -0.2, 0.3], np.log([0.3, 0.4, 0.8, 0.6])])

    def conditional(v):
        return gaussian_theta_conditional(
            kw["basis_delta"] @ v[:3], jnp.exp(v[3:]), **kwargs
        )

    def target(v):
        return conditional(v)[0]

    actual_value, actual_gradient = jax.value_and_grad(target)(point)
    actual_hessian = jax.jacfwd(jax.grad(target))(point)
    actual_reverse_hessian = jax.jacrev(jax.grad(target))(point)
    actual_conditional = conditional(point)[1:]
    kwargs["gram_cache"] = None
    kwargs["linear_cache"] = None
    with monkeypatch.context() as context:
        context.setattr(
            collapsed,
            "_quadratic_at_conditional_mean",
            collapsed._conditional_quadratic,
        )
        context.setattr(
            collapsed,
            "_logdet_at_precision",
            lambda precision, cholesky, scale: (
                2
                * (
                    jnp.sum(jnp.log(scale))
                    + jnp.sum(jnp.log(jnp.diag(cholesky)))
                )
            ),
        )
        expected_value, expected_gradient = jax.value_and_grad(target)(point)
        expected_hessian = jax.jacfwd(jax.grad(target))(point)
        expected_conditional = conditional(point)[1:]
    np.testing.assert_allclose(actual_value, expected_value, atol=2e-11)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=2e-10)
    np.testing.assert_allclose(actual_hessian, expected_hessian, atol=2e-9)
    np.testing.assert_allclose(
        actual_reverse_hessian, expected_hessian, atol=2e-9
    )
    for actual, expected in zip(
        actual_conditional, expected_conditional, strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-11)


def test_gram_cache_is_exact_sparse_and_bounded():
    rng = np.random.default_rng(51)
    n, r, k = 64, 3, 24
    re, im = np.zeros((n, r, k)), np.zeros((n, r, k))
    for f in range(n):
        re[f, :, f % k] = rng.normal(size=r)
        im[f, :, (f + 7) % k] = rng.normal(size=r)
    budget = n * 3 * 16  # Only two active coefficients per frequency.
    cache = _prepare_gram_cache(re, im, max_bytes=budget)
    assert cache is not None
    indices, frequencies, values = cache
    assert values.ndim == 1
    assert sum(v.nbytes for v in cache) <= budget
    precision = jnp.exp(jnp.linspace(-2, 2, n))
    upper = jax.ops.segment_sum(
        values * precision[frequencies],
        indices,
        num_segments=k**2,
        indices_are_sorted=True,
    ).reshape(k, k)
    actual = upper + upper.T - jnp.diag(jnp.diag(upper))
    expected = np.einsum("frk,frl,f->kl", re, re, precision)
    expected += np.einsum("frk,frl,f->kl", im, im, precision)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert _prepare_gram_cache(re, im, max_bytes=0) is None


def test_zero_design_gram_cache():
    re = np.zeros((5, 2, 3))
    cache = _prepare_gram_cache(re, re)
    assert cache is not None
    pairs, _, values = cache
    assert pairs.size == values.size == 0


def test_fixed_design_caches_share_a_bounded_budget(monkeypatch):
    import log_psplines.inference.collapsed as collapsed

    budget = 4096
    monkeypatch.setattr(collapsed, "_DESIGN_CACHE_BYTES", budget)
    prepared = prepare_collapsed_channel(_example(n=128))
    caches = [prepared["linear_cache"], *(prepared["gram_cache"] or ())]
    assert sum(cache.nbytes for cache in caches if cache is not None) <= budget
    assert prepared["linear_cache"] is None
    assert prepared["gram_cache"] is None
    prepared = prepare_collapsed_channel(_example())
    assert prepared["linear_cache"] is not None
    caches = [prepared["linear_cache"], *(prepared["gram_cache"] or ())]
    assert sum(cache.nbytes for cache in caches if cache is not None) <= budget
    assert (
        _prepare_linear_cache(
            prepared["design_re"],
            prepared["design_im"],
            prepared["observed_re"],
            prepared["observed_im"],
            max_bytes=0,
        )
        is None
    )


def test_banded_marginal_matches_independent_gaussian_and_dense_derivatives():
    kw = _example()
    prepared = prepare_collapsed_channel(dict(kw, theta_backend="banded"))
    dense = prepare_collapsed_channel(dict(kw, theta_backend="dense"))
    plan = prepared["banded_plan"]
    assert plan is not None
    point = jnp.asarray(
        np.r_[np.linspace(-0.7, 0.8, 5), np.log([0.3, 0.4, 0.8, 0.6])]
    )

    def target(v, prepared):
        return gaussian_theta_conditional(
            v[:5], jnp.exp(v[5:]), **_conditional_kwargs(prepared)
        )[0]

    for operator in (
        jax.value_and_grad,
        jax.hessian,
        lambda f: jax.jacrev(jax.grad(f)),
    ):
        actual = jax.jit(operator(lambda v: target(v, prepared)))(point)
        expected = jax.jit(operator(lambda v: target(v, dense)))(point)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10)
    _, mean, factor, scale = gaussian_theta_conditional(
        point[:5], jnp.exp(point[5:]), **_conditional_kwargs(prepared)
    )
    n, width = factor.shape[0], factor.shape[1] - 1
    lower = np.zeros((n, n))
    for d in range(width + 1):
        lower[np.arange(d, n), np.arange(n - d)] = np.asarray(factor)[
            : n - d, d
        ]
    precision = (
        np.asarray(scale)[:, None]
        * (lower @ lower.T)
        * np.asarray(scale)[None, :]
    )
    covariance = np.linalg.inv(precision)[
        np.ix_(plan.inverse_permutation, plan.inverse_permutation)
    ]
    expected_mean, expected_covariance, *_ = _independent_gaussian(
        kw, np.asarray(point[:5]), np.exp(np.asarray(point[5:]))
    )
    np.testing.assert_allclose(mean, expected_mean, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        covariance, expected_covariance, atol=2e-12, rtol=2e-12
    )


def test_banded_conditional_draws_keep_joint_moments_and_likelihood():
    kw = dict(_example(), theta_backend="banded")
    prepared = prepare_collapsed_channel(kw)
    weights = np.array([0.1, -0.3, 0.15])
    sigmas = np.array([0.3, 0.4, 0.8, 0.6])
    posterior = {"weights_delta_2": np.broadcast_to(weights, (2, 3000, 3))}
    posterior.update(
        {
            f"sigma_{name}": np.full((2, 3000), sigma)
            for name, sigma in zip(
                prepared["theta_names"], sigmas, strict=True
            )
        }
    )
    output = sample_conditional_theta(
        posterior, prepared, jax.random.PRNGKey(44)
    )
    draws = np.concatenate(
        [output[f"weights_{name}"] for name in prepared["theta_names"]],
        axis=-1,
    ).reshape(6000, -1)
    logd = np.asarray(kw["basis_delta"]) @ weights
    mean, covariance, _, x, y, precision = _independent_gaussian(
        kw, logd, sigmas
    )
    # Whiten with an independently assembled dense covariance. This checks
    # joint dependence, including directions with very small uncertainty.
    white = np.linalg.solve(np.linalg.cholesky(covariance), (draws - mean).T).T
    assert np.max(np.abs(white.mean(axis=0)) * np.sqrt(len(draws))) < 4
    identity = np.eye(white.shape[1])
    covariance_error = np.abs(np.cov(white, rowvar=False) - identity)
    covariance_mcse = np.sqrt((1 + identity) / (len(draws) - 1))
    assert np.max(covariance_error / covariance_mcse) < 5
    full = -kw["eta"] / kw["enbw"] * kw["Nb"] * kw["Nh"] * np.sum(
        logd
    ) - np.sum(precision * (y - x @ draws[0]) ** 2)
    np.testing.assert_allclose(
        output["log_likelihood_block_2"][0, 0], full, atol=2e-11, rtol=2e-12
    )


def test_banded_selection_and_cache_fallback(monkeypatch):
    import log_psplines.inference.collapsed as collapsed

    assert (
        prepare_collapsed_channel(dict(_example(), theta_backend="auto"))[
            "banded_plan"
        ]
        is None
    )
    monkeypatch.setattr(collapsed, "_DESIGN_CACHE_BYTES", 0)
    with pytest.raises(ValueError, match="memory budget"):
        prepare_collapsed_channel(dict(_example(), theta_backend="banded"))
    fallback = prepare_collapsed_channel(
        dict(_example(), theta_backend="auto")
    )
    assert fallback["banded_plan"] is fallback["gram_cache"] is None


def test_auto_banded_plan_falls_back_when_cache_exceeds_budget(monkeypatch):
    import log_psplines.inference.collapsed as collapsed

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    kw = dict(_example(n=128), theta_backend="auto")
    basis = np.tile(np.eye(64), (2, 1))
    penalty = 2 * np.eye(64)
    penalty += np.diag(np.full(63, -0.2), 1)
    penalty += np.diag(np.full(63, -0.2), -1)
    for part in ("re", "im"):
        kw[f"basis_theta_{part}_by_component"] = (basis, basis)
        kw[f"penalty_theta_{part}_by_component"] = (penalty, penalty)
    assert prepare_collapsed_channel(kw)["banded_plan"] is not None
    monkeypatch.setattr(collapsed, "_DESIGN_CACHE_BYTES", 0)
    prepared = prepare_collapsed_channel(kw)
    assert prepared["banded_plan"] is prepared["gram_cache"] is None


def test_banded_preserves_small_residual_when_channel_is_almost_predictable():
    from log_psplines.basis.splines import SplineBasis

    kw = _example(n=65)
    for part, counts in (("re", (9, 13)), ("im", (11, 15))):
        splines = [
            SplineBasis.from_knots(
                grid=np.linspace(0, 1, 65), knots=np.linspace(0, 1, count)
            )
            for count in counts
        ]
        kw[f"basis_theta_{part}_by_component"] = tuple(
            s.basis for s in splines
        )
        kw[f"penalty_theta_{part}_by_component"] = tuple(
            s.penalty for s in splines
        )
    # A large, nearly explained channel stresses the residual formulation;
    # subtracting two large marginal quadratic forms loses this information.
    coefficient = 300.0 + 100.0j
    previous = np.asarray(kw["u_re_prev"]) + 1j * np.asarray(kw["u_im_prev"])
    observed = coefficient * previous[:, 0, :]
    observed += 0.01 * (
        np.asarray(kw["u_re_channel"]) + 1j * np.asarray(kw["u_im_channel"])
    )
    kw["u_re_channel"] = jnp.asarray(observed.real)
    kw["u_im_channel"] = jnp.asarray(observed.imag)
    point = jnp.asarray(np.r_[np.full(65, -8.0), np.log([10.0] * 4)])

    def evaluate(backend):
        prepared = prepare_collapsed_channel(dict(kw, theta_backend=backend))

        def target(v):
            return gaussian_theta_conditional(
                v[:65], jnp.exp(v[65:]), **_conditional_kwargs(prepared)
            )[0]

        return jax.jit(jax.value_and_grad(target))(point)

    dense, packed = evaluate("dense"), evaluate("banded")
    for actual, expected in zip(packed, dense, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-7)
