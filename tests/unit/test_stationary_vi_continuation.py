"""Continuation contracts; posterior recovery remains a separate scientific run."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    packed_log_densities,
)
from log_psplines.inference.vi import fit_vi


@pytest.fixture
def continuation(monkeypatch):
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2] / "examples/vi_nuts_validation"
        )
    )
    import stationary_next

    return stationary_next


def schedule():
    return {
        "type": "warmup_cosine",
        "initial_lr": 0.001,
        "peak_lr": 0.01,
        "warmup_steps": 1000,
        "end_lr": 0.0001,
    }


def test_schedule_prefix_and_constant_tail(continuation):
    old = continuation.learning_rate(schedule(), 40000)
    new = continuation.tail_schedule(schedule())
    steps = jnp.arange(60000)
    values = np.asarray(jax.jit(jax.vmap(new))(steps))
    np.testing.assert_array_equal(
        values[:40000], np.asarray(jax.jit(jax.vmap(old))(steps[:40000]))
    )
    np.testing.assert_array_equal(values[40000:], np.full(20000, 1e-4))
    assert float(new(40000)) == 1e-4


def test_exact_svi_replay_and_observational_checkpoint(continuation):
    def model(observed=0.4):
        c = numpyro.sample("c", dist.Normal(jnp.zeros(2), 1).to_event(1))
        numpyro.factor("data", -0.5 * jnp.sum((c - observed) ** 2))

    tiny = {**schedule(), "warmup_steps": 100}
    cfg = VIDiagnosticConfig(
        seeds=(91, 94),
        num_particles=32,
        chunk_size=16,
        evaluation_seeds=(92, 93),
        evaluation_particles=4,
        checkpoint_steps=(500, 1000),
        target_fingerprint="toy",
    )
    old = fit_vi(
        model,
        rng_key=jax.random.PRNGKey(7),
        vi_steps=1000,
        optimizer_lr=continuation.learning_rate(tiny, 1000),
        optimization_particles=8,
        guide="mvn",
        early_stopping=False,
        posterior_draws=32,
        diagnostics=cfg,
    )
    saved = {}

    def observe(step, params):
        saved[step] = jax.tree.map(np.asarray, params)

    new = fit_vi(
        model,
        rng_key=jax.random.PRNGKey(7),
        vi_steps=1500,
        optimizer_lr=continuation.tail_schedule(tiny, horizon=1000),
        optimization_particles=8,
        guide="mvn",
        early_stopping=False,
        posterior_draws=32,
        diagnostics=cfg,
        checkpoint_callback=observe,
    )
    for key in old.diagnostics.params:
        np.testing.assert_array_equal(
            old.diagnostics.params[key], saved[1000][key]
        )
    np.testing.assert_array_equal(old.losses, new.losses[: len(old.losses)])
    assert set(saved) == {500, 1000, 1500}
    assert new.timings["steps_run"] == 1500
    assert all(
        np.isfinite(value) and value >= 0 for value in new.timings.values()
    )
    # Changed data reaches the full density; cached shape does not retain old observations.
    guide = continuation.rebuild_guide(
        new.diagnostics, model, target_fingerprint="toy"
    )
    point = guide.get_posterior(new.diagnostics.params).mean
    p1, _ = packed_log_densities(
        model,
        guide,
        new.diagnostics.params,
        point,
        model_kwargs={"observed": 0.4},
    )
    p2, _ = packed_log_densities(
        model,
        guide,
        new.diagnostics.params,
        point,
        model_kwargs={"observed": 2.0},
    )
    assert float(p1) != float(p2)


def seed(seed, final="within_screen", late="within_screen"):
    return {
        "seed": seed,
        "final_primary_agreement": final,
        "late_optimization_stability": late,
        "replay_status": "within_screen",
    }


def test_exploratory_unlock_does_not_require_all_seeds(continuation):
    result = continuation.gate_decision(
        [
            seed(7101),
            seed(7102, late="outside_screen"),
            seed(7103, final="mc_precision_limited"),
        ],
        [{"status": "within_screen"}] * 3,
    )
    assert result["exploratory_hierarchy_unlock"] is True
    assert result["repeatable_recipe"] is False
    assert result["passing_seeds"] == [7101]


def test_missing_precision_and_replay_do_not_pass(continuation):
    fits = [
        seed(7101, late="mc_precision_limited"),
        seed(7102, final="unavailable"),
        {**seed(7103), "replay_status": "outside_screen"},
    ]
    result = continuation.gate_decision(fits, [])
    assert not result["exploratory_hierarchy_unlock"]
    assert not result["repeatable_recipe"]
    assert continuation.agreement_status({}) == "unavailable"


def test_controller_stops_at_fixed_gate_and_retains_history(
    continuation, tmp_path, monkeypatch
):
    resolved = {"seeds": [7101, 7102, 7103]}
    for number in resolved["seeds"]:
        continuation.fit_directory(tmp_path, "fixed", number).mkdir(
            parents=True
        )
    monkeypatch.setattr(continuation, "prepare", lambda *args: resolved)
    result = {
        "continuation_gate": {"exploratory_hierarchy_unlock": False},
        "historical_flag": "outside_screen",
    }
    monkeypatch.setattr(continuation, "analyze_target", lambda *args: result)
    monkeypatch.setattr(
        continuation,
        "dispatch",
        lambda *args: pytest.fail("unauthorized later inference"),
    )
    recorded = []
    monkeypatch.setattr(
        continuation,
        "summarize",
        lambda *args, **kwargs: recorded.append(kwargs["reason"]),
    )
    continuation.execute(tmp_path, tmp_path)
    assert recorded == ["stopped_fixed_continuation_gate"]
    assert result["historical_flag"] == "outside_screen"
