"""Banded operators agree with independent dense SPD calculations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import linalg

from log_psplines.inference import banded


def _pack(matrix, width):
    out = np.zeros((len(matrix), width + 1))
    for d in range(width + 1):
        out[: len(matrix) - d, d] = np.diag(matrix, -d)
    return out


def _spd(n, width):
    rng = np.random.default_rng(94)
    factor = np.tril(rng.normal(scale=0.15, size=(n, n)))
    factor[np.subtract.outer(np.arange(n), np.arange(n)) > width] = 0
    factor[np.diag_indices(n)] = np.exp(np.linspace(-1, 1, n))
    return factor @ factor.T


@pytest.mark.parametrize("n,width", [(1, 0), (7, 0), (7, 1), (13, 4), (6, 5)])
def test_factor_solve_and_selected_inverse(n, width):
    matrix = _spd(n, width)
    factor = jax.jit(banded.cholesky)(jnp.asarray(_pack(matrix, width)))
    expected = linalg.cholesky(matrix, lower=True)
    np.testing.assert_allclose(
        factor, _pack(expected, width), atol=2e-13, rtol=2e-13
    )
    rhs = jnp.linspace(-0.4, 0.9, n)
    for transpose in (False, True):
        actual = jax.jit(
            lambda b, transpose=transpose: banded.solve_triangular(
                factor, b, transpose=transpose
            )
        )(rhs)
        reference = linalg.solve_triangular(
            expected, rhs, lower=True, trans="T" if transpose else "N"
        )
        np.testing.assert_allclose(actual, reference, atol=2e-13, rtol=2e-13)
    selected = jax.jit(banded.inverse_subset)(factor)
    np.testing.assert_allclose(
        selected, _pack(np.linalg.inv(matrix), width), atol=2e-12, rtol=2e-12
    )


def test_banded_logdet_derivatives_and_solve_derivatives():
    n, width = 9, 3
    matrix = jnp.asarray(_spd(n, width))
    packed = jnp.asarray(_pack(matrix, width))
    rhs = jnp.linspace(-0.3, 0.8, n)

    def target(v, use_band):
        scale = jnp.exp(v)
        if use_band:
            h = packed.at[:, 0].add(scale)
            diagonal = jnp.sqrt(h[:, 0])
            rows = jnp.arange(n)[:, None] + jnp.arange(width + 1)
            balanced = h / (
                diagonal[:, None]
                * jnp.pad(diagonal, (0, width), constant_values=1)[rows]
            )
            factor = banded.cholesky(balanced)
            logdet = banded.logdet_from_factor(h, factor, diagonal)
            x = banded.solve_triangular(factor, rhs / diagonal)
            x = banded.solve_triangular(factor, x, transpose=True) / diagonal
            return logdet + rhs @ x
        h = matrix + jnp.diag(scale)
        return jnp.linalg.slogdet(h)[1] + rhs @ jnp.linalg.solve(h, rhs)

    point = jnp.linspace(-2, 1, n)

    def reference(v):
        return target(v, False)

    def actual(v):
        return target(v, True)

    for operator in (
        jax.value_and_grad,
        jax.hessian,
        lambda f: jax.jacrev(jax.grad(f)),
    ):
        a, b = (
            jax.jit(operator(actual))(point),
            jax.jit(operator(reference))(point),
        )
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, rtol=2e-11, atol=2e-11)


def test_auto_plan_respects_structural_zeros_and_tiny_nonzeros(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    n = 256
    design = np.eye(n)[:, None, :]
    imaginary = np.zeros_like(design)
    penalty = np.diag(np.full(n, 2.0))
    penalty += np.diag(np.full(n - 1, -0.2), 1)
    penalty += np.diag(np.full(n - 1, -0.2), -1)
    plan = banded.prepare_plan(design, imaginary, penalty)
    assert plan is not None and plan.penalty.shape == (n, 2)
    np.testing.assert_array_equal(
        np.asarray(plan.permutation)[plan.inverse_permutation], np.arange(n)
    )
    dense_penalty = np.where(penalty == 0, 1e-30, penalty)
    assert banded.prepare_plan(design, imaginary, dense_penalty) is None
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert banded.prepare_plan(design, imaginary, penalty) is None
    assert (
        banded.prepare_plan(design, imaginary, penalty, force=True) is not None
    )
