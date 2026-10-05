"""Analytic density identities and reproducible VI diagnostics contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density
from scipy.stats import multivariate_normal, norm

from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    VIDiagnosticState,
    combine_factor_ratios,
    compare_features,
    diagnose_guide,
    fingerprint,
    mmd_squared,
    nuts_reference_health,
    packed_log_densities,
    rebuild_guide,
    reference_transform,
    require_same_target,
    weight_diagnostics,
    whiten_coefficients,
)
from log_psplines.inference.model import _sample_pspline_block
from log_psplines.inference.vi import (
    VIResult,
    fit_vi,
    resolve_guide,
    run_multivariate_vi,
)


def gaussian_model():
    numpyro.sample(
        "x",
        dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=jnp.eye(2)),
    )


def initialized(model, kind="mvn", args=(), kwargs=None):
    guide, name = resolve_guide(kind, model)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(guide, 0)).get_trace(
        *args, **(kwargs or {})
    )
    params = {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "param"
    }
    return guide, name, params


def config(**kwargs):
    return VIDiagnosticConfig(
        seeds=(81, 82),
        num_particles=128,
        chunk_size=32,
        evaluation_seeds=(91, 92),
        evaluation_particles=4,
        target_fingerprint="test-target",
        **kwargs,
    )


def test_exact_gaussian_density_weights_checkpoint_roundtrip(tmp_path):
    guide, name, params = initialized(gaussian_model)
    params.update(auto_loc=jnp.zeros(2), auto_scale_tril=jnp.eye(2))
    state = diagnose_guide(gaussian_model, guide, params, name, config())
    for result in state.metadata["weights"]:
        assert result["status"] == "constant_weights"
        assert result["k"] is None
        assert result["raw_weight_ess"] == pytest.approx(128)
        np.testing.assert_allclose(
            state.arrays[f"seed_{result['seed']}_log_ratios"], 0, atol=1e-12
        )
    state.save(tmp_path)
    loaded = VIDiagnosticState.load(tmp_path)
    rebuilt = rebuild_guide(
        loaded, gaussian_model, target_fingerprint="test-target"
    )
    original_dist, loaded_dist = (
        guide.get_posterior(params),
        rebuilt.get_posterior(loaded.params),
    )
    samples = original_dist.sample(jax.random.PRNGKey(25), (2000,))
    np.testing.assert_allclose(
        loaded_dist.sample(jax.random.PRNGKey(25), (2000,)),
        samples,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        loaded_dist.log_prob(samples),
        original_dist.log_prob(samples),
        atol=1e-12,
    )
    # The moment tolerances are > 5 standard errors for N=2000.
    np.testing.assert_allclose(np.mean(samples, axis=0), 0, atol=0.12)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False), np.eye(2), atol=0.18
    )
    roundtrip = VIDiagnosticState.from_dataset(state.to_dataset())
    np.testing.assert_array_equal(
        roundtrip.params["auto_loc"], params["auto_loc"]
    )
    with pytest.raises(ValueError, match="mismatch"):
        rebuild_guide(loaded, gaussian_model, target_fingerprint="different")


@pytest.mark.parametrize(
    "scale,location", [(0.45, 0.0), (2.0, 0.0), (1.0, 1.0)]
)
def test_gaussian_failure_modes_have_different_weights_and_moments(
    scale, location
):
    # Common deterministic Normal quantiles reduce MC noise in comparisons.
    x = location + scale * norm.ppf((np.arange(4096) + 0.5) / 4096)
    ratios = norm.logpdf(x) - norm.logpdf(x, loc=location, scale=scale)
    weights = weight_diagnostics(ratios)
    assert weights["status"] == "ok"
    assert weights["raw_ess_fraction"] < 0.8
    if scale < 1:
        assert weights["k"] > 0.5
    if scale > 1:
        assert weights["k"] < 0.5  # bounded weights do not establish q=p
        assert np.std(x) > 1.9
    if location:
        assert abs(np.mean(x)) > 0.9


def test_joint_correlation_and_anticorrelation_functional():
    rng = np.random.default_rng(23)
    p = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=1600)
    q = rng.multivariate_normal([0, 0], [[1, -0.9], [-0.9, 1]], size=1600)
    # Analytically both marginals are N(0,1), but the joint differs.
    statistic = mmd_squared(p, q, chunk_size=80)
    baseline = mmd_squared(p[:800], p[800:], chunk_size=80)
    assert statistic > baseline + 0.05
    sigma_p = np.array([[1, -0.9], [-0.9, 1]])
    sigma_q = np.diag([0.19, 0.19])
    assert np.diag(sigma_q).max() < np.diag(sigma_p).min()
    assert np.ones(2) @ sigma_q @ np.ones(2) == pytest.approx(0.38)
    assert np.ones(2) @ sigma_p @ np.ones(2) == pytest.approx(0.2)


def test_positive_noncentered_jacobian_and_factor_identity():
    def model(y):
        scale = numpyro.sample("scale", dist.HalfNormal(2))
        raw = numpyro.sample("raw", dist.Normal(0, 1))
        numpyro.deterministic("coefficient", scale * raw)
        numpyro.factor("data", dist.Normal(scale * raw, 1).log_prob(y))

    guide, _, params = initialized(model, args=(0.7,))
    packed = jnp.array([np.log(1.7), 0.4])
    unpacked = guide._unpack_latent(packed)
    constrained = {"scale": jnp.exp(unpacked["scale"]), "raw": unpacked["raw"]}
    expected, _ = log_density(model, (0.7,), {}, constrained)
    p, q = packed_log_densities(
        model, guide, params, packed, model_args=(0.7,)
    )
    np.testing.assert_allclose(p, expected + unpacked["scale"], atol=1e-12)
    assert (
        abs(float(p - expected)) > 0.1
    )  # omitting the positive-site Jacobian fails
    assert np.isfinite(q)
    assert (
        len(guide._init_locs) == 2
    )  # deterministic coefficient is not a dimension


def test_centered_spline_correction_is_in_joint():
    penalty = jnp.array([[2.0, 0.2], [0.2, 1.0]])

    def model():
        _sample_pspline_block("sigma", "weights", penalty, 1.28)

    guide, _, params = initialized(model)
    packed = jnp.array([np.log(1.5), 0.4, -0.8])
    sites = guide._unpack_latent(packed)
    sigma = jnp.exp(sites["sigma"])
    weights = sites["weights"]
    expected = (
        dist.HalfNormal(1.28).log_prob(sigma)
        + sites["sigma"]
        - 2 * jnp.log(sigma)
        - 0.5 * (weights @ penalty @ weights) / sigma**2
    )
    p, _ = packed_log_densities(model, guide, params, packed)
    np.testing.assert_allclose(p, expected, atol=1e-12)
    wrong = (
        dist.HalfNormal(1.28).log_prob(sigma)
        + sites["sigma"]
        + dist.Normal(0, 1).log_prob(weights).sum()
    )
    assert abs(float(p - wrong)) > 0.1


def test_independent_factor_joint_identity():
    x = norm.ppf((np.arange(1024) + 0.5) / 1024)
    y = x[np.random.default_rng(2).permutation(len(x))]
    a = norm.logpdf(x, scale=1.5) - norm.logpdf(x)
    b = norm.logpdf(y, loc=0.7) - norm.logpdf(y)
    joint, summary = combine_factor_ratios([a, b], factorization_verified=True)
    expected = multivariate_normal.logpdf(
        np.column_stack([x, y]), mean=[0, 0.7], cov=np.diag([1.5**2, 1])
    ) - multivariate_normal.logpdf(
        np.column_stack([x, y]), mean=np.zeros(2), cov=np.eye(2)
    )
    np.testing.assert_allclose(joint, expected, atol=2e-14)
    assert summary["k"] == pytest.approx(
        weight_diagnostics(expected)["k"], abs=1e-12
    )
    with pytest.raises(ValueError, match="verified"):
        combine_factor_ratios([a, b], factorization_verified=False)


def test_chunked_mmd_diagonals_and_negative_estimates():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(51, 3)), rng.normal(size=(37, 3))
    for unbiased in (True, False):
        dense = mmd_squared(x, y, chunk_size=100, unbiased=unbiased)
        chunked = mmd_squared(x, y, chunk_size=7, unbiased=unbiased)
        assert dense == pytest.approx(chunked, abs=1e-14)
    assert mmd_squared(x, x, unbiased=True) < 0
    assert abs(mmd_squared(x, x, unbiased=False)) < 1e-14
    trained = reference_transform(x[:20])
    # Both methods use exactly the same reference-derived transformation.
    transformed = (x - trained["mean"]) @ trained["matrix"]
    assert np.isfinite(transformed).all()
    with pytest.raises(ValueError, match="zero reference"):
        reference_transform(np.zeros((20, 3)))


def test_whitening_complex_conjugation_mask_and_convention():
    rng = np.random.default_rng(13)
    z = (
        rng.normal(size=(10000, 2)) + 1j * rng.normal(size=(10000, 2))
    ) / np.sqrt(2)
    covariance = np.array([[2, 0.3 + 0.5j], [0.3 - 0.5j, 1]])
    coefficients = z @ np.linalg.cholesky(covariance).T
    coefficients[0] = np.nan
    result = whiten_coefficients(
        coefficients,
        covariance,
        complex_coefficients=True,
        mask=np.arange(len(z)) > 0,
    )
    np.testing.assert_allclose(result["residuals"], z[1:], atol=2e-15)
    np.testing.assert_allclose(result["covariance"], np.eye(2), atol=0.04)
    np.testing.assert_allclose(result["pseudo_covariance"], 0, atol=0.05)
    assert result["component_scale"] == np.sqrt(2)
    real = whiten_coefficients(z.real * np.sqrt(2), np.eye(2))
    assert real["component_scale"] == 1
    assert (
        whiten_coefficients(None, covariance)["status"]
        == "unavailable_coefficient_phases"
    )


def test_invalid_density_statuses_and_unavailable_native(monkeypatch):
    assert weight_diagnostics([0, np.inf])["status"] == "invalid_density"
    assert weight_diagnostics([np.nan, np.nan])["status"] == "invalid_density"
    assert weight_diagnostics([0])["status"] == "insufficient_samples"
    assert weight_diagnostics(np.ones(32))["status"] == "constant_weights"
    assert weight_diagnostics([0, 0, 0, 1, 2])["status"] == "insufficient_tail"
    guide, name, params = initialized(gaussian_model)
    monkeypatch.delattr(numpyro.infer, "psis_diagnostic", raising=False)
    state = diagnose_guide(gaussian_model, guide, params, name, config())
    assert state.metadata["native_psis"]["status"] == "unsupported_api"
    assert (
        nuts_reference_health(np.zeros((1, 20, 2)), None)["status"]
        == "failed_reference"
    )


def test_same_target_zero_variance_and_failed_reference():
    with pytest.raises(ValueError, match="unavailable"):
        require_same_target(None, None)
    with pytest.raises(ValueError, match="mismatched"):
        require_same_target("a", "b")
    assert fingerprint({"b": 2, "a": np.ones(4)}) == fingerprint(
        {"a": np.ones(4), "b": 2}
    )
    values = np.ones((4, 10, 1))
    comparison = compare_features(
        values,
        values,
        names=["constant"],
        vi_fingerprint="a",
        reference_fingerprint="a",
        reference_status="failed_reference",
    )
    assert comparison["status"] == "unresolved_reference"
    assert comparison["features"][0]["status"] == "zero_reference_sd"


def test_opt_in_iterator_materialization_and_posterior_rng():
    def model(location, *, value):
        x = numpyro.sample("x", dist.Normal(location, 1))
        numpyro.factor("observed", dist.Normal(x, 1).log_prob(value))

    options = dict(
        rng_key=jax.random.PRNGKey(3),
        vi_steps=30,
        optimizer_lr=0.01,
        model_kwargs={"value": 0.5},
        posterior_draws=10,
    )
    plain = fit_vi(model, model_args=iter([0.2]), **options)
    audited = fit_vi(
        model, model_args=iter([0.2]), diagnostics=config(), **options
    )
    assert plain.diagnostics is None
    np.testing.assert_array_equal(plain.posterior.x, audited.posterior.x)
    np.testing.assert_array_equal(plain.losses, audited.losses)
    state = audited.diagnostics
    assert state.metadata["optimization"]["steps_run"] == 30
    assert state.metadata["optimization"]["checkpoint_steps"] == [30]
    assert state.metadata["density_status"] == "ok"


def test_actual_checkpoint_steps_and_disabled_stopping():
    result = fit_vi(
        gaussian_model,
        rng_key=jax.random.PRNGKey(4),
        vi_steps=205,
        optimizer_lr=0.01,
        early_stopping=False,
        posterior_draws=5,
        diagnostics=config(checkpoint_steps=(55, 155)),
    )
    audit = result.diagnostics.metadata["optimization"]
    assert audit["steps_run"] == 205
    assert audit["checkpoint_steps"] == [55, 155, 205]
    assert audit["stopping_reason"] == "step_limit"
    assert set(result.diagnostics.checkpoints) == {"55", "155", "205"}


def test_checkpoint_callback_requires_diagnostics_before_model_initialization():
    def uncalled_model():
        raise AssertionError(
            "configuration must be validated before initialization"
        )

    with pytest.raises(
        ValueError, match="checkpoint_callback requires.*diagnostics"
    ):
        fit_vi(
            uncalled_model,
            rng_key=jax.random.PRNGKey(4),
            vi_steps=5,
            optimizer_lr=0.01,
            checkpoint_callback=lambda step, params: None,
        )


def test_noise_aware_short_run_preserves_rng_and_final_checkpoint():
    options = dict(
        rng_key=jax.random.PRNGKey(4),
        vi_steps=5,
        optimizer_lr=0.01,
        posterior_draws=8,
    )
    checkpoints = []
    plain = fit_vi(gaussian_model, **options)
    audited = fit_vi(
        gaussian_model,
        diagnostics=config(stopping_rule="noise_aware"),
        checkpoint_callback=lambda step, params: checkpoints.append(
            (step, params)
        ),
        **options,
    )
    np.testing.assert_array_equal(plain.losses, audited.losses)
    np.testing.assert_array_equal(plain.posterior.x, audited.posterior.x)
    assert [step for step, _ in checkpoints] == [5]
    assert all(
        np.isfinite(value).all() for value in checkpoints[0][1].values()
    )


@pytest.mark.parametrize("guide", ["flow:1", "flowbnaf:1"])
@pytest.mark.parametrize("steps", [5, 205])
def test_noise_aware_stopping_rejects_unsupported_flow_moments_before_updates(
    guide, steps, monkeypatch
):
    from numpyro.infer import SVI

    def uncalled_update(*args, **kwargs):
        raise AssertionError(
            "unsupported stopping must be rejected before updates"
        )

    monkeypatch.setattr(SVI, "update", uncalled_update)
    with pytest.raises(
        ValueError,
        match="noise-aware stopping requires.*mean and variance",
    ):
        fit_vi(
            gaussian_model,
            rng_key=jax.random.PRNGKey(4),
            vi_steps=steps,
            optimizer_lr=0.01,
            guide=guide,
            diagnostics=config(stopping_rule="noise_aware"),
        )


@pytest.mark.parametrize("guide", ["flow:1", "flowbnaf:1"])
def test_ordinary_flow_fits_remain_supported(guide):
    fitted = fit_vi(
        gaussian_model,
        rng_key=jax.random.PRNGKey(4),
        vi_steps=5,
        optimizer_lr=0.01,
        guide=guide,
        posterior_draws=8,
    )
    assert fitted.timings["steps_run"] == 5
    assert np.isfinite(fitted.losses).all()
    assert fitted.posterior.x.shape == (1, 8, 2)
    assert np.isfinite(fitted.posterior.x).all()


def test_multivariate_timings_sum_updates_and_keep_each_block(monkeypatch):
    import importlib

    import xarray as xr

    module = importlib.import_module("log_psplines.inference.vi")
    calls = []

    def fit_block(model, **kwargs):
        calls.append(kwargs)
        index = kwargs["model_kwargs"]["channel"]
        multiplier = index + 1
        timings = {
            "steps_run": 10 * multiplier,
            "first_chunk_including_compile_seconds": 0.1 * multiplier,
            "posterior_draw_seconds": 0.01 * multiplier,
        }
        if index == 1:
            timings["remaining_optimization_seconds"] = 0.4
        return VIResult(
            posterior=xr.Dataset({f"x_{index}": ("draw", np.arange(3))}),
            losses=jnp.full(5 + index, multiplier),
            guide_name="diag",
            timings=timings,
        )

    monkeypatch.setattr(module, "fit_vi", fit_block)
    monkeypatch.setattr(
        module,
        "channel_model_kwargs",
        lambda kwargs, index: {"channel": index},
    )
    fitted = run_multivariate_vi(
        {"n_channels": 2},
        rng_key=jax.random.PRNGKey(4),
        steps=50,
        early_stopping=False,
    )
    assert fitted.timings["num_blocks"] == 2
    assert fitted.timings["steps_run"] == 30
    assert fitted.timings["block_0_steps_run"] == 10
    assert fitted.timings["block_1_steps_run"] == 20
    assert fitted.timings[
        "first_chunk_including_compile_seconds"
    ] == pytest.approx(0.3)
    assert fitted.timings["posterior_draw_seconds"] == pytest.approx(0.03)
    assert fitted.timings["remaining_optimization_seconds"] == 0.4
    assert fitted.timings["block_1_remaining_optimization_seconds"] == 0.4
    assert all(
        call["vi_steps"] == 50 and not call["early_stopping"] for call in calls
    )
    np.testing.assert_array_equal(fitted.losses, np.full(5, 3))


def test_legacy_stopping_constant_sensitivity_and_audited_objective():
    def base(constant):
        def model():
            x = numpyro.sample("x", dist.Normal(0, 1))
            numpyro.factor("likelihood", -0.5 * (x - 0.5) ** 2 + constant)

        return model

    # The legacy decision demonstrably depends on an additive data constant.
    options = dict(
        rng_key=jax.random.PRNGKey(18),
        vi_steps=1500,
        optimizer_lr=0.01,
        posterior_draws=5,
    )
    a, b = fit_vi(base(0), **options), fit_vi(base(1e9), **options)
    assert b.timings["steps_run"] < a.timings["steps_run"]
    # The audited rule compares paired changes and location stability, not the
    # absolute objective level. Use a moderate constant to avoid FP cancellation.
    audited_a = fit_vi(
        base(0), diagnostics=config(stopping_rule="noise_aware"), **options
    )
    audited_b = fit_vi(
        base(100), diagnostics=config(stopping_rule="noise_aware"), **options
    )
    assert audited_a.timings["steps_run"] == audited_b.timings["steps_run"]
    np.testing.assert_allclose(
        audited_a.posterior.x, audited_b.posterior.x, atol=1e-12
    )


def test_native_replay_and_packed_density_agree_with_scaled_factors():
    from numpyro.infer.elbo import get_importance_trace

    def model(y, *, strength):
        sigma = numpyro.sample("sigma", dist.HalfNormal(2))
        x = numpyro.sample("x", dist.Normal(0, 1))
        with numpyro.handlers.scale(scale=strength):
            numpyro.factor(
                "full_data", dist.Normal(sigma * x, 1).log_prob(y).sum()
            )

    args = (jnp.array([0.4, 0.8, -0.2]),)
    kwargs = {"strength": 0.7}
    guide, _, params = initialized(model, args=args, kwargs=kwargs)
    model_key, guide_key = jax.random.split(jax.random.PRNGKey(21))
    model_trace, guide_trace = get_importance_trace(
        numpyro.handlers.seed(model, model_key),
        numpyro.handlers.seed(guide, guide_key),
        args,
        kwargs,
        params,
    )
    log_p = sum(
        site["log_prob"].sum()
        for site in model_trace.values()
        if site["type"] == "sample"
    )
    log_q = sum(
        site["log_prob"].sum()
        for site in guide_trace.values()
        if site["type"] == "sample"
    )
    u = guide_trace["_auto_latent"]["value"]
    packed_p, packed_q = packed_log_densities(
        model, guide, params, u, model_args=args, model_kwargs=kwargs
    )
    np.testing.assert_allclose(packed_p - packed_q, log_p - log_q, atol=2e-12)


def test_reference_screen_and_depth_failure():
    rng = np.random.default_rng(31)
    values = rng.normal(size=(4, 1000, 3))
    stats = {
        "diverging": np.zeros((4, 1000)),
        "n_steps": np.full((4, 1000), 7),
        "energy": rng.normal(size=(4, 1000)),
    }
    assert nuts_reference_health(values, stats)["status"] == "accepted_screen"
    stats["n_steps"][0, 0] = 2**10 - 1
    failed = nuts_reference_health(values, stats)
    assert failed["status"] == "failed_reference"
    assert failed["depth_saturation"] == 1


def test_blocked_cholesky_diagnostics_and_persistence(tmp_path):
    from dataclasses import replace

    from log_psplines import StationaryConfig, fit
    from log_psplines.data.spectral import WishartData
    from log_psplines.results import PSDResult

    u = np.broadcast_to(np.eye(2), (12, 2, 2)).copy()
    data = WishartData(
        u, np.zeros_like(u), np.linspace(0.03, 0.48, 12), 12, 2, Nb=2
    )
    fitted = fit(
        data,
        StationaryConfig(
            method="vi",
            n_knots=4,
            degree=1,
            diffMatrixOrder=1,
            vi_steps=20,
            vi_posterior_draws=10,
            verbose=False,
            vi_early_stopping=False,
            vi_diagnostics={
                "num_particles": 64,
                "chunk_size": 16,
                "seeds": (81, 82),
                "evaluation_seeds": (91, 92),
                "evaluation_particles": 4,
            },
        ),
    )
    states = fitted.vi.diagnostics_per_block
    assert len(states) == 2
    assert fitted.vi.timings["num_blocks"] == 2
    assert fitted.vi.timings["steps_run"] == 40
    for index, state in enumerate(states):
        assert fitted.vi.timings[f"block_{index}_steps_run"] == 20
        for name, value in state.metadata["timings"].items():
            assert fitted.vi.timings[f"block_{index}_{name}"] == value
    combined = fitted.vi.diagnostics.arrays["repeat_0_log_ratios"]
    expected = (
        states[0].arrays["seed_81_log_ratios"]
        + states[1].arrays["seed_10081_log_ratios"]
    )
    np.testing.assert_allclose(combined, expected, atol=1e-12)
    assert (
        states[0].metadata["diagnostic_seeds"]
        != states[1].metadata["diagnostic_seeds"]
    )
    path = tmp_path / "blocked.nc"
    fitted.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    assert restored.vi.timings == fitted.vi.timings
    assert len(restored.vi.diagnostics_per_block) == 2
    np.testing.assert_array_equal(
        restored.vi.diagnostics.arrays["repeat_0_log_ratios"], combined
    )
    # Identical normalized sufficient statistics can describe different physical
    # spectra. Reject a saved target identity before fitting those data.
    for changed_data in (
        replace(data, channel_stds=np.array([2.0, 3.0])),
        replace(data, scaling_factor=2.0),
    ):
        with pytest.raises(ValueError, match="fingerprint"):
            fit(
                changed_data,
                StationaryConfig(
                    method="vi",
                    n_knots=4,
                    degree=1,
                    diffMatrixOrder=1,
                    verbose=False,
                    vi_diagnostics={
                        "target_fingerprint": fitted.metadata[
                            "target_fingerprint"
                        ]
                    },
                ),
            )


def test_diagnostic_collection_preserves_default_early_stopping():
    options = dict(
        rng_key=jax.random.PRNGKey(12),
        vi_steps=600,
        optimizer_lr=0.01,
        posterior_draws=8,
    )
    plain = fit_vi(gaussian_model, **options)
    audited = fit_vi(gaussian_model, diagnostics=config(), **options)
    assert plain.timings["steps_run"] == audited.timings["steps_run"]
    np.testing.assert_array_equal(plain.losses, audited.losses)
    np.testing.assert_array_equal(plain.posterior.x, audited.posterior.x)


def test_scheduled_multi_particle_fit_and_recorded_training_contract(tmp_path):
    import optax

    schedule = optax.cosine_decay_schedule(0.02, 1000, alpha=0.1)
    fitted = fit_vi(
        gaussian_model,
        rng_key=jax.random.PRNGKey(12),
        vi_steps=1000,
        optimizer_lr=schedule,
        optimizer_lr_metadata={"type": "cosine", "steps": 1000},
        optimization_particles=8,
        guide="diag",
        posterior_draws=4000,
        early_stopping=False,
        diagnostics=config(checkpoint_steps=(500, 1000)),
    )
    # This analytic target has known moments; training particles change the
    # stochastic objective, while the posterior and checkpoint contracts persist.
    samples = fitted.posterior.x.values.reshape(-1, 2)
    np.testing.assert_allclose(samples.mean(0), 0, atol=0.15)
    np.testing.assert_allclose(samples.std(0), 1, atol=0.15)
    fitted.diagnostics.save(tmp_path)
    state = VIDiagnosticState.load(tmp_path)
    optimization = state.metadata["optimization"]
    assert optimization["optimization_particles"] == 8
    assert optimization["optimizer"]["learning_rate"] == "callable_schedule"
    assert optimization["optimizer"]["schedule_metadata"]["steps"] == 1000
    assert optimization["checkpoint_steps"] == [500, 1000]


def test_checkpoint_persisted_before_expensive_analysis_failure(
    tmp_path, monkeypatch
):
    import importlib

    module = importlib.import_module("log_psplines.inference.vi")

    def preserve(step, params):
        np.savez(tmp_path / f"parameters_{step}.npz", **params)

    def fail_after_persistence(*args, **kwargs):
        assert (tmp_path / "parameters_5.npz").exists()
        raise RuntimeError("deliberate downstream analysis failure")

    monkeypatch.setattr(module, "evaluate_objective", fail_after_persistence)
    with pytest.raises(RuntimeError, match="deliberate downstream"):
        fit_vi(
            gaussian_model,
            rng_key=jax.random.PRNGKey(12),
            vi_steps=10,
            optimizer_lr=0.01,
            guide="diag",
            posterior_draws=8,
            early_stopping=False,
            diagnostics=config(checkpoint_steps=(5,)),
            checkpoint_callback=preserve,
        )
    saved = np.load(tmp_path / "parameters_5.npz")
    assert saved.files
    assert all(np.isfinite(saved[key]).all() for key in saved.files)


@pytest.mark.parametrize("particles", [0, -1, True, 1.5])
def test_invalid_optimization_particle_count_is_rejected(particles):
    with pytest.raises(
        (ValueError, TypeError), match="optimization_particles"
    ):
        fit_vi(
            gaussian_model,
            rng_key=jax.random.PRNGKey(12),
            vi_steps=10,
            optimizer_lr=0.01,
            optimization_particles=particles,
        )
