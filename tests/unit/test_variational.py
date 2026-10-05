"""NumPyro guide, optimization settings and public VI result contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pytest
import xarray as xr
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal

from log_psplines.inference.vi import (
    VIResult,
    fit_vi,
    resolve_guide,
    run_multivariate_vi,
)


def gaussian_model():
    numpyro.sample("x", dist.Normal(jnp.zeros(2), 1).to_event(1))


@pytest.mark.parametrize(
    "guide", ["diag", "mvn", "lowrank:1", "flow:1", "flowbnaf:1"]
)
def test_builtin_guides_return_finite_constrained_draws(guide):
    result = fit_vi(
        gaussian_model,
        rng_key=jax.random.PRNGKey(4),
        vi_steps=5,
        optimizer_lr=0.01,
        guide=guide,
        posterior_draws=8,
    )
    assert result.guide_name == guide
    assert result.timings["steps_run"] == 5
    assert result.posterior.x.shape == (1, 8, 2)
    assert np.isfinite(result.losses).all()
    assert np.isfinite(result.posterior.x).all()


@pytest.mark.parametrize("kind", ["diag", "mvn", "lowrank:1"])
def test_builtin_guides_use_explicit_model_initialization(kind):
    guide, _ = resolve_guide(
        kind, gaussian_model, init_values={"x": jnp.array([1.5, -0.2])}
    )
    svi = SVI(gaussian_model, guide, optax.adam(0.01), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(3))
    np.testing.assert_allclose(
        guide.median(svi.get_params(state))["x"], [1.5, -0.2]
    )


def test_callable_guide_and_argument_iterator_preserve_reproducibility():
    def model(location, *, observed):
        x = numpyro.sample("x", dist.Normal(location, 1))
        numpyro.sample("observed", dist.Normal(x, 1), obs=observed)

    options = dict(
        rng_key=jax.random.PRNGKey(3),
        vi_steps=20,
        optimizer_lr=0.01,
        model_kwargs={"observed": 0.5},
        posterior_draws=10,
    )
    first = fit_vi(model, model_args=iter([0.2]), **options)
    second = fit_vi(
        model, model_args=(0.2,), guide=AutoDiagonalNormal, **options
    )
    np.testing.assert_array_equal(first.losses, second.losses)
    np.testing.assert_array_equal(first.posterior.x, second.posterior.x)


def test_disabled_early_stopping_runs_the_complete_budget():
    def nearly_flat_model():
        numpyro.sample("x", dist.Normal(jnp.zeros(2), 1).to_event(1))
        numpyro.factor("constant", jnp.array(1e9))

    options = dict(
        rng_key=jax.random.PRNGKey(4),
        vi_steps=505,
        optimizer_lr=0.01,
        posterior_draws=5,
    )
    stopped = fit_vi(nearly_flat_model, **options)
    complete = fit_vi(nearly_flat_model, early_stopping=False, **options)
    assert stopped.timings["steps_run"] == 400
    assert complete.timings["steps_run"] == 505
    assert len(complete.losses) == 6


def test_scheduled_multi_particle_fit_recovers_gaussian_moments():
    result = fit_vi(
        gaussian_model,
        rng_key=jax.random.PRNGKey(12),
        vi_steps=1000,
        optimizer_lr=optax.cosine_decay_schedule(0.02, 1000, alpha=0.1),
        optimization_particles=8,
        posterior_draws=4000,
        early_stopping=False,
    )
    samples = result.posterior.x.values.reshape(-1, 2)
    np.testing.assert_allclose(samples.mean(0), 0, atol=0.15)
    np.testing.assert_allclose(samples.std(0), 1, atol=0.15)
    assert result.timings["steps_run"] == 1000


def test_zero_posterior_draws_returns_one_median_draw():
    result = fit_vi(
        gaussian_model,
        rng_key=jax.random.PRNGKey(3),
        vi_steps=5,
        optimizer_lr=0.01,
        posterior_draws=0,
    )
    assert result.posterior.x.shape == (1, 1, 2)


@pytest.mark.parametrize("particles", [0, -1, True, 1.5])
def test_invalid_training_particle_count_is_rejected(particles):
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


@pytest.mark.parametrize("steps, rate", [(0, 0.01), (5, 0.0), (5, np.nan)])
def test_invalid_training_settings_are_rejected(steps, rate):
    with pytest.raises(ValueError, match="vi_steps|optimizer_lr"):
        fit_vi(
            gaussian_model,
            rng_key=jax.random.PRNGKey(12),
            vi_steps=steps,
            optimizer_lr=rate,
        )


def test_multivariate_timings_and_loss_aggregation(monkeypatch):
    import importlib

    module = importlib.import_module("log_psplines.inference.vi")

    def fit_block(model, **kwargs):
        index = kwargs["model_kwargs"]["channel"]
        assert not kwargs["early_stopping"]
        return VIResult(
            posterior=xr.Dataset({f"x_{index}": ("draw", np.arange(3))}),
            losses=jnp.full(5 + index, index + 1),
            guide_name="diag",
            timings={
                "steps_run": 10 * (index + 1),
                "posterior_draw_seconds": 0.1 * (index + 1),
            },
        )

    monkeypatch.setattr(module, "fit_vi", fit_block)
    monkeypatch.setattr(
        module,
        "channel_model_kwargs",
        lambda kwargs, index: {"channel": index},
    )
    result = run_multivariate_vi(
        {"n_channels": 2}, rng_key=jax.random.PRNGKey(4), early_stopping=False
    )
    assert result.timings["num_blocks"] == 2
    assert result.timings["steps_run"] == 30
    assert result.timings["block_0_steps_run"] == 10
    assert result.timings["block_1_steps_run"] == 20
    assert result.timings["posterior_draw_seconds"] == pytest.approx(0.3)
    np.testing.assert_array_equal(result.losses, np.full(5, 3))


def test_public_multivariate_vi_preserves_spectral_invariants():
    from log_psplines import StationaryConfig, fit
    from log_psplines.data.spectral import WishartData

    matrices = np.broadcast_to(np.eye(2), (12, 2, 2)).copy()
    data = WishartData(
        matrices,
        np.zeros_like(matrices),
        np.linspace(0.03, 0.48, 12),
        12,
        2,
        Nb=2,
    )
    result = fit(
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
        ),
    )
    assert result.vi.timings["steps_run"] == 40
    spectrum = result.spectrum.values
    assert np.isfinite(spectrum).all()
    np.testing.assert_allclose(spectrum, np.swapaxes(spectrum.conj(), -1, -2))
    assert (np.linalg.eigvalsh(spectrum) > 0).all()
    coherence = result.coherence
    assert (coherence >= 0).all() and (coherence <= 1).all()
