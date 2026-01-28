import jax
import jax.numpy as jnp
import numpy as np

from log_psplines.samplers.multivar.cholesky_whittle import (
    unit_lower_from_theta,
    whittle_loglik_cholesky_blocks,
)


def _make_hermitian_pd(key, p: int) -> jnp.ndarray:
    a_re = jax.random.normal(key, (p, p))
    a_im = jax.random.normal(jax.random.fold_in(key, 1), (p, p))
    A = a_re + 1j * a_im
    return A @ jnp.conj(A).T


def test_whittle_loglik_matches_trace_formula_single_matrix():
    key = jax.random.PRNGKey(0)
    p = 3

    Y = _make_hermitian_pd(key, p)
    log_delta_sq = 0.3 * jax.random.normal(jax.random.fold_in(key, 2), (p,))

    theta = jnp.zeros((p, p), dtype=jnp.complex64)
    theta = theta.at[1, 0].set(0.2 + 0.1j)
    theta = theta.at[2, 0].set(-0.3 + 0.05j)
    theta = theta.at[2, 1].set(0.15 - 0.2j)

    nu = 2.0
    got_total, got_blocks = whittle_loglik_cholesky_blocks(
        Y, log_delta_sq, theta, nu=nu
    )

    T = unit_lower_from_theta(theta)
    D_inv = jnp.diag(jnp.exp(-log_delta_sq))
    Sinv = jnp.swapaxes(jnp.conj(T), -1, -2) @ D_inv @ T

    expected_total = -nu * jnp.sum(log_delta_sq) - jnp.real(
        jnp.trace(Sinv @ Y)
    )

    assert np.allclose(
        np.asarray(got_total), np.asarray(expected_total), rtol=1e-6, atol=1e-6
    )
    assert np.allclose(
        np.asarray(jnp.sum(got_blocks)),
        np.asarray(got_total),
        rtol=1e-6,
        atol=1e-6,
    )


def test_blocks_depend_only_on_row_parameters_p3():
    key = jax.random.PRNGKey(1)
    p = 3

    Y = _make_hermitian_pd(key, p)
    log_delta_sq = 0.2 * jax.random.normal(jax.random.fold_in(key, 2), (p,))

    theta = jnp.zeros((p, p), dtype=jnp.complex64)
    theta = theta.at[1, 0].set(0.2 + 0.1j)  # theta_21
    theta = theta.at[2, 0].set(-0.3 + 0.05j)  # theta_31
    theta = theta.at[2, 1].set(0.15 - 0.2j)  # theta_32

    _, blocks_base = whittle_loglik_cholesky_blocks(Y, log_delta_sq, theta)

    theta_perturbed = theta.at[1, 0].add(0.7 - 0.4j)
    _, blocks_perturbed = whittle_loglik_cholesky_blocks(
        Y, log_delta_sq, theta_perturbed
    )

    # Only block 2 (j=1, 0-based) should change when perturbing theta_21.
    assert np.allclose(
        np.asarray(blocks_base[0]),
        np.asarray(blocks_perturbed[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(
        np.asarray(blocks_base[1]),
        np.asarray(blocks_perturbed[1]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(blocks_base[2]),
        np.asarray(blocks_perturbed[2]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_weights_and_batch_dimensions_match_manual_sum():
    key = jax.random.PRNGKey(2)
    p = 3
    n_freq = 5

    keys = jax.random.split(key, n_freq)
    Y = jax.vmap(_make_hermitian_pd, in_axes=(0, None))(keys, p)
    log_delta_sq = 0.1 * jax.random.normal(
        jax.random.fold_in(key, 3), (n_freq, p)
    )

    theta = jnp.zeros((n_freq, p, p), dtype=jnp.complex64)
    theta = theta.at[:, 1, 0].set(0.2 + 0.1j)
    theta = theta.at[:, 2, 0].set(-0.3 + 0.05j)
    theta = theta.at[:, 2, 1].set(0.15 - 0.2j)

    weights = jnp.linspace(0.5, 1.5, n_freq)
    nu = 3.0

    got_total, got_blocks = whittle_loglik_cholesky_blocks(
        Y, log_delta_sq, theta, nu=nu, weights=weights
    )

    # Manual: sum weights * per-frequency per-row terms.
    T = unit_lower_from_theta(theta)
    TYT_H = jnp.matmul(jnp.matmul(T, Y), jnp.swapaxes(jnp.conj(T), -1, -2))
    diag_power = jnp.real(jnp.diagonal(TYT_H, axis1=-2, axis2=-1))
    inv_delta_sq = jnp.exp(-log_delta_sq)
    manual_blocks = (-nu * log_delta_sq - diag_power * inv_delta_sq) * weights[
        :, None
    ]
    manual_total = jnp.sum(manual_blocks)

    assert np.allclose(
        np.asarray(got_total), np.asarray(manual_total), rtol=1e-6, atol=1e-6
    )
    assert np.allclose(
        np.asarray(got_blocks), np.asarray(manual_blocks), rtol=1e-6, atol=1e-6
    )
