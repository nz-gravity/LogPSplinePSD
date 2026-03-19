import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from log_psplines.samplers.pspline_block import build_log_density_fn


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
