import jax.numpy as jnp
import numpy as np

import log_psplines.diagnostics.vi_psis as vi_psis_mod
from log_psplines.diagnostics.vi_psis import _compute_psis_khat, _psislw
from log_psplines.samplers.vi_init.mixin import (
    _compute_psis_moment_checks,
)


def test_psis_khat_bounds_regression():
    """Ensure PSIS k-hat flags both well-behaved and heavy-tailed cases."""
    rng = np.random.default_rng(0)
    log_r_good = rng.normal(0.0, 0.1, size=500)
    _, khat_good = _psislw(log_r_good)
    assert float(np.max(khat_good)) < 0.5

    log_r_bad = np.concatenate(
        [rng.normal(0.0, 0.1, size=450), np.full(50, 6.0)]
    )
    _, khat_bad = _psislw(log_r_bad)
    assert float(np.max(khat_bad)) > 0.9


def test_psis_moment_correction_detects_bias():
    """Moment correction should expose sizable bias/variance shifts."""
    samples = {
        "phi": np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float),
    }
    weights = np.array([0.05, 0.1, 0.15, 0.7], dtype=float)
    summary = _compute_psis_moment_checks(samples, weights)
    hyper = summary.get("hyperparameters") or []
    assert hyper, "Expected hyperparameter summary."
    entry = hyper[0]
    assert entry["bias_pct"] > 10.0  # fail if bias shrinks below threshold
    assert entry["var_ratio"] < 0.7  # under-dispersion should be detected


def test_psis_khat_runs_below_recommended_draw_count(monkeypatch):
    class _DummyGuide:
        prefix = "auto"

    def _fake_log_density(model, model_args, model_kwargs, sample):
        return jnp.array(0.0), None

    def _fake_psislw(log_weights):
        lw = np.asarray(log_weights, dtype=np.float64)
        return lw, np.array([0.25], dtype=np.float64)

    monkeypatch.setattr(vi_psis_mod, "log_density", _fake_log_density)
    monkeypatch.setattr(vi_psis_mod, "_psislw", _fake_psislw)

    n_draws = 50
    result = _compute_psis_khat(
        model=lambda *args, **kwargs: None,
        model_args=(),
        model_kwargs={},
        guide=_DummyGuide(),
        guide_params={},
        vi_samples={"phi": jnp.zeros((n_draws, 1), dtype=jnp.float32)},
        latent_samples=jnp.zeros((n_draws, 1), dtype=jnp.float32),
    )

    assert result is not None
    assert result["psis_khat_max"] == 0.25
