import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density

from log_psplines.samplers.pspline_block import (
    build_log_density_fn,
    sample_pspline_block,
)


def _branching_model(flag: int, scale: jnp.ndarray) -> None:
    if flag > 0:
        loc = 0.0
    else:
        loc = 1.0
    numpyro.sample("x", dist.Normal(loc=loc, scale=scale))


def test_build_log_density_fn_preserves_scalar_kwargs_for_control_flow():
    logpost_fn = build_log_density_fn(
        _branching_model,
        {"flag": 1, "scale": np.asarray(1.0)},
    )
    log_prob = float(logpost_fn({"x": jnp.asarray(0.25)}))

    expected = float(dist.Normal(loc=0.0, scale=1.0).log_prob(0.25))
    assert np.isfinite(log_prob)
    assert log_prob == pytest.approx(expected)


def test_sample_pspline_block_exact_log_phi_prior_is_finite():
    penalty = jnp.eye(3)

    def model():
        sample_pspline_block(
            delta_name="delta",
            phi_name="log_phi",
            weights_name="weights",
            penalty_matrix=penalty,
            alpha_phi=2.0,
            beta_phi=1.5,
            alpha_delta=1.0,
            beta_delta=1.0,
        )

    params = {
        "delta": jnp.asarray(1.25),
        "log_phi": jnp.asarray(-0.2),
        "weights": jnp.asarray([0.1, -0.3, 0.2]),
    }
    log_prob, _ = log_density(model, (), {}, params)

    assert np.isfinite(float(log_prob))


def test_sample_pspline_block_returns_positive_phi_and_expected_shapes():
    penalty = jnp.eye(4)
    captured = {}

    def model():
        out = sample_pspline_block(
            delta_name="delta",
            phi_name="log_phi",
            weights_name="weights",
            penalty_matrix=penalty,
            alpha_phi=2.0,
            beta_phi=1.0,
            alpha_delta=1.0,
            beta_delta=1.0,
        )
        captured.update(out)

    params = {
        "delta": jnp.asarray(0.75),
        "log_phi": jnp.asarray(0.4),
        "weights": jnp.asarray([0.0, 0.1, -0.2, 0.3]),
    }
    log_prob, _ = log_density(model, (), {}, params)

    assert np.isfinite(float(log_prob))
    assert set(captured) == {"weights", "delta", "phi"}
    assert captured["weights"].shape == (4,)
    assert captured["delta"].shape == ()
    assert captured["phi"].shape == ()
    assert float(captured["phi"]) > 0.0
