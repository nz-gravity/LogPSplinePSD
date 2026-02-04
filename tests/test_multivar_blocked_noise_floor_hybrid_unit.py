import jax.numpy as jnp
import numpy as np

from log_psplines.samplers.multivar.multivar_blocked_nuts import (
    compute_noise_floor_eps2,
    smooth_max,
)


def test_smooth_max_matches_max_in_limit():
    a = jnp.asarray(1e-4, dtype=jnp.float32)
    b = jnp.asarray(1e-6, dtype=jnp.float32)
    out = smooth_max(a, b, tau=1e-12)
    assert float(out) > 0.0
    assert np.isclose(float(out), float(a), rtol=1e-6, atol=0.0)


def test_hybrid_eps2_and_delta_eff_invariant():
    theory = jnp.asarray([1e-3, 1e1], dtype=jnp.float32)
    eps2 = compute_noise_floor_eps2(
        mode="hybrid",
        theory_psd=theory,
        scale=1e-2,
        constant=1e-4,
        tau=1e-6,
        dtype=jnp.float32,
    )
    eps2_np = np.asarray(eps2)

    # First entry is dominated by constant (scaled=1e-5 < 1e-4)
    assert eps2_np[0] >= 1e-4
    # Second entry is dominated by scaled theory (scaled=1e-1)
    assert eps2_np[1] > 1e-2

    log_delta_sq = jnp.asarray([-50.0, -20.0], dtype=jnp.float32)
    delta_sq = jnp.exp(log_delta_sq)
    delta_eff_sq = delta_sq + eps2

    # Done-condition invariant: delta_eff_sq >= eps2 elementwise.
    assert float(jnp.min(delta_eff_sq - eps2)) >= 0.0
    assert float(jnp.min(delta_eff_sq)) + 1e-30 >= float(jnp.min(eps2))
