"""Parity with frozen WDM mathematics, without a sibling-repo dependency."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from log_psplines import LogPSpline, SpectralMatrix, SplineBasis
from log_psplines.arviz_utils.spline_storage import to_storage_payload
from log_psplines.basis.penalty import eigen_prior_scale, whiten_penalty_pair
from log_psplines.likelihoods.whittle import (
    power_whittle_log_likelihood,
    whittle_log_likelihood,
)

REFERENCE = Path(__file__).parent / "reference" / "wdm_core.npz"


@pytest.fixture
def reference():
    # Local precision context: do not change the stationary regression setup.
    with jax.enable_x64(True), np.load(REFERENCE) as values:
        yield values


def _basis(matrix, penalty):
    return SplineBasis(
        grid=np.linspace(0, 1, matrix.shape[0]),
        knots=np.array([0.0, 1.0]),
        basis=jnp.asarray(matrix),
        penalty=jnp.asarray(penalty),
    )


def test_surface_likelihood_and_gradient_match_wdm(reference):
    r = reference
    model = LogPSpline(
        frequency=_basis(r["basis_freq"], r["penalty_freq"]),
        time=_basis(r["basis_time"], r["penalty_time"]),
    )
    weights = jnp.asarray(r["coefficients"])
    surface = jax.jit(model.__call__)(weights)
    np.testing.assert_allclose(surface, r["surface"], rtol=1e-12, atol=1e-12)
    assert model.weights.shape == weights.shape
    np.testing.assert_array_equal(model(), np.zeros_like(surface))

    def likelihood(w):
        return power_whittle_log_likelihood(r["power"], r["counts"], model(w))

    np.testing.assert_allclose(
        jax.jit(likelihood)(weights), r["likelihood"], rtol=1e-12
    )
    np.testing.assert_allclose(
        jax.grad(likelihood)(weights), r["gradient"], rtol=1e-12, atol=1e-12
    )
    with pytest.raises(ValueError, match="shape"):
        model(jnp.zeros(weights.shape[::-1]))
    with pytest.raises(ValueError, match="shape"):
        LogPSpline(model.frequency, time=model.time, weights=jnp.zeros(5))
    with pytest.raises(NotImplementedError, match="storage"):
        to_storage_payload(model)


def test_tensor_prior_matches_wdm_and_kronecker_precision(reference):
    r = reference
    pair = whiten_penalty_pair(r["penalty_time"], r["penalty_freq"])
    for name in ("lam_time", "lam_freq", "joint_null"):
        np.testing.assert_allclose(pair[name], r[name], atol=1e-14)
    scale = eigen_prior_scale(
        2.0,
        3.0,
        jnp.asarray(pair["lam_time"]),
        jnp.asarray(pair["lam_freq"]),
        jnp.asarray(pair["joint_null"]),
    )
    np.testing.assert_allclose(scale, r["scale"], rtol=1e-12)
    assert pair["joint_null"].sum() == 4  # two marginal linear null spaces
    np.testing.assert_array_equal(np.asarray(scale)[pair["joint_null"]], 100.0)

    rotation = np.kron(pair["U_time"], pair["U_freq"])
    null_vectors = rotation[:, pair["joint_null"].ravel()]
    precision = 2 * np.kron(r["penalty_time"], np.eye(5)) + 3 * np.kron(
        np.eye(4), r["penalty_freq"]
    )
    precision += 1e-6 * np.eye(20) + (1e-4 - 1e-6) * (
        null_vectors @ null_vectors.T
    )
    np.testing.assert_allclose(
        rotation.T @ precision @ rotation,
        np.diag(1 / np.asarray(scale).ravel() ** 2),
        atol=2e-14,
    )


def test_stationary_surface_limit_and_matrix_composition(reference):
    r = reference
    frequency = _basis(r["basis_freq"], r["penalty_freq"])
    constant_time = SplineBasis(
        np.linspace(0, 1, 7),
        np.array([0.0, 1.0]),
        jnp.ones((7, 1)),
        jnp.zeros((1, 1)),
        degree=0,
        penalty_order=0,
    )
    stationary = LogPSpline(frequency)
    varying = LogPSpline(frequency, time=constant_time)
    weights = jnp.asarray(r["coefficients"][0])
    np.testing.assert_allclose(
        varying(weights[None, :]),
        np.broadcast_to(stationary(weights), (7, 11)),
        atol=1e-14,
    )
    logs = np.stack(
        [varying(weights[None, :]), varying((weights + 0.2)[None, :])], axis=-1
    )
    theta = np.full((7, 11, 1), 0.1)
    matrices = SpectralMatrix(2)(logs, theta, theta / 2)
    assert matrices.shape == (7, 11, 2, 2)
    assert np.linalg.eigvalsh(matrices).min() > 0
    assert SpectralMatrix.coherence(matrices).max() <= 1 + 1e-14


def test_power_counts_fourier_conversion_masking_and_tiny_units(reference):
    r = reference
    logs = jnp.asarray(r["surface"])
    counts = jnp.asarray(r["counts"])
    power = jnp.asarray(r["power"])
    gradient = jax.grad(
        lambda s: power_whittle_log_likelihood(power, counts, s)
    )(logs)
    np.testing.assert_array_equal(np.asarray(gradient)[counts == 0], 0.0)
    shift = np.log(1e-40)
    shifted = power_whittle_log_likelihood(power * 1e-40, counts, logs + shift)
    np.testing.assert_allclose(
        shifted, r["likelihood"] - 0.5 * counts.sum() * shift, rtol=1e-12
    )

    fourier = whittle_log_likelihood(logs, power, count=3, duration=4.0)
    real_components = power_whittle_log_likelihood(
        2 * power / 4.0, jnp.full(logs.shape, 6), logs
    )
    np.testing.assert_allclose(fourier, real_components, rtol=1e-12)
