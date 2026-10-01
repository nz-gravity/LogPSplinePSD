"""Small mathematical contracts for spline PSD estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from scipy.integrate import quad
from scipy.interpolate import BSpline

from log_psplines import SpectralMatrix, SplineBasis
from log_psplines.likelihoods.whittle import whittle_log_likelihood
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.models.reconstruction import (
    compute_psd_quantiles,
    reconstruct_psd_matrix,
)


def test_bspline_basis_and_roughness_penalty_match_definition():
    knots = np.array([0.0, 0.2, 0.65, 1.0])
    grid = np.linspace(0, 1, 17) ** 1.3
    basis = SplineBasis.from_knots(grid, knots)
    padded = np.r_[np.repeat(0.0, 3), knots, np.repeat(1.0, 3)]
    reference = BSpline(padded, np.eye(6), 3)
    np.testing.assert_allclose(basis.basis, reference(grid), atol=5e-8)

    second = reference.derivative(2)
    penalty = np.array(
        [
            [
                quad(
                    lambda x, i=i, j=j: second(x)[i] * second(x)[j],
                    0,
                    1,
                    points=knots[1:-1],
                )[0]
                for j in range(6)
            ]
            for i in range(6)
        ]
    )
    penalty /= penalty.max()
    penalty += 1e-6 * np.eye(6)
    np.testing.assert_allclose(basis.penalty, penalty, atol=5e-8)


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_spectral_matrix_is_hermitian_positive_and_coherence_is_defined(
    channels,
):
    rng = np.random.default_rng(23)
    logs = rng.normal(scale=0.4, size=(2, channels))
    row, col = np.tril_indices(channels, k=-1)
    theta_re = rng.normal(scale=0.2, size=(2, len(row)))
    theta_im = rng.normal(scale=0.2, size=(2, len(row)))
    matrix = SpectralMatrix(channels)
    spectrum = matrix(logs, theta_re, theta_im)
    triangular = np.broadcast_to(
        np.eye(channels, dtype=complex), spectrum.shape
    ).copy()
    triangular[:, row, col] = -theta_re - 1j * theta_im
    precision = triangular.conj().swapaxes(-1, -2) @ (
        np.exp(-logs)[..., :, None] * triangular
    )
    np.testing.assert_allclose(spectrum, np.linalg.inv(precision), atol=1e-14)
    np.testing.assert_allclose(
        spectrum, spectrum.conj().swapaxes(-1, -2), atol=1e-14
    )
    assert np.linalg.eigvalsh(spectrum).min() > 0
    coherence = matrix.coherence(spectrum)
    diagonal = np.diagonal(spectrum, axis1=-2, axis2=-1).real
    expected = np.abs(spectrum) ** 2 / (
        diagonal[..., :, None] * diagonal[..., None, :]
    )
    np.testing.assert_allclose(coherence, expected)
    assert np.all((coherence >= 0) & (coherence <= 1 + 1e-14))
    zero = matrix(logs, np.zeros_like(theta_re), np.zeros_like(theta_im))
    np.testing.assert_allclose(
        zero, np.exp(logs)[..., :, None] * np.eye(channels)
    )
    np.testing.assert_allclose(
        matrix(logs + np.log(3), theta_re, theta_im), 3 * spectrum, atol=1e-14
    )


def test_whittle_likelihood_matches_direct_sum_and_has_finite_gradient():
    log_psd = jnp.log(jnp.array([1.2, 0.7, 2.0]))
    power = np.array([0.3, 1.4, 0.8])
    duration = 4.0
    actual = whittle_log_likelihood(log_psd, power, duration=duration)
    expected = -np.sum(log_psd) - np.sum(power / (duration * np.exp(log_psd)))
    np.testing.assert_allclose(actual, expected)
    gradient = jax.grad(
        lambda value: whittle_log_likelihood(value, power, duration=duration)
    )(log_psd)
    assert np.isfinite(gradient).all()


def test_wishart_likelihood_matches_independent_factor_calculation():
    logs = jnp.array([0.1, -0.2])
    theta_re = jnp.array([[0.2], [-0.1]])
    theta_im = jnp.array([[0.1], [0.3]])
    u_re = jnp.array([[0.4, -0.2], [0.1, 0.7]])
    u_im = jnp.array([[0.3, 0.1], [-0.2, 0.4]])
    previous_re = jnp.array([[[0.1, 0.0]], [[-0.2, 0.0]]])
    previous_im = jnp.array([[[0.0, 0.0]], [[0.1, 0.0]]])
    kwargs = dict(Nb=2, Nh=3, duration=4.0, enbw=1.5, eta=0.7)
    actual = wishart_log_likelihood(
        logs,
        theta_re,
        theta_im,
        u_re,
        u_im,
        previous_re,
        previous_im,
        **kwargs,
    )
    u = np.asarray(u_re) + 1j * np.asarray(u_im)
    previous = np.asarray(previous_re) + 1j * np.asarray(previous_im)
    theta = np.asarray(theta_re) + 1j * np.asarray(theta_im)
    residual = u - np.einsum("fl,flr->fr", theta, previous)
    log_variance = np.asarray(logs)
    expected = (
        (
            -6 * np.sum(log_variance)
            - np.sum(np.abs(residual) ** 2 / np.exp(log_variance)[:, None] / 4)
        )
        * 0.7
        / 1.5
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    gradient = jax.grad(
        lambda value: wishart_log_likelihood(
            value,
            theta_re,
            theta_im,
            u_re,
            u_im,
            previous_re,
            previous_im,
            **kwargs,
        )
    )(logs)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize(
    "power",
    [np.ones(4), np.ones((3, 1)), np.ones(3, dtype=np.int64)],
    ids=["different-length", "different-rank", "integer-dtype"],
)
def test_pytest_typechecking_rejects_invalid_likelihood_arrays(power):
    with pytest.raises(TypeCheckError):
        whittle_log_likelihood(jnp.zeros(3), power)


def test_pytest_typechecking_rejects_invalid_scalar_type():
    with pytest.raises(TypeCheckError):
        whittle_log_likelihood(jnp.zeros(3), np.ones(3), duration="4")


@pytest.fixture
def cholesky_draws():
    rng = np.random.default_rng(24)
    logs = rng.normal(scale=0.4, size=(2, 31, 17, 3))
    logs[1] += 1.0
    return (
        logs,
        rng.normal(scale=0.2, size=logs.shape),
        rng.normal(scale=0.2, size=logs.shape),
    )


@pytest.mark.parametrize("chunk_size", [1, 4, 7, 32, None, 0, -1])
def test_three_channel_chunked_reconstruction(cholesky_draws, chunk_size):
    logs, real, imag = cholesky_draws
    expected = SpectralMatrix(3)(logs, real, imag).reshape(62, 17, 3, 3)
    actual = reconstruct_psd_matrix(
        logs, real, imag, chunk_size=chunk_size, n_samples_max=None
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert actual.shape == (62, 17, 3, 3) and actual.dtype == np.complex128
    np.testing.assert_allclose(
        actual, actual.conj().swapaxes(-1, -2), atol=1e-14
    )
    assert np.diagonal(actual, axis1=-2, axis2=-1).real.min() > 0
    assert np.linalg.eigvalsh(actual).min() > 0
    limited = reconstruct_psd_matrix(
        logs, real, imag, chunk_size=chunk_size, n_samples_max=5
    )
    np.testing.assert_allclose(limited, expected[:5], atol=1e-14)


def test_chunked_matrix_quantiles_use_all_chains_and_draws(cholesky_draws):
    logs, real, imag = cholesky_draws
    full = SpectralMatrix(3)(logs, real, imag).reshape(62, 17, 3, 3)
    expected = np.percentile(
        full.real, [5, 50, 95], axis=0
    ) + 1j * np.percentile(full.imag, [5, 50, 95], axis=0)
    diagonal = np.diagonal(full, axis1=-2, axis2=-1).real
    coherence = np.abs(full) ** 2 / (
        diagonal[..., :, None] * diagonal[..., None, :]
    )
    actual_real, actual_imag, actual_coherence = compute_psd_quantiles(
        logs, real, imag, chunk_size=4, compute_coherence=True
    )
    np.testing.assert_allclose(
        actual_real + 1j * actual_imag, expected, atol=1e-14
    )
    np.testing.assert_allclose(
        actual_coherence,
        np.percentile(coherence, [5, 50, 95], axis=0),
        atol=1e-14,
    )
    assert not np.allclose(
        actual_real, np.percentile(full[:31].real, [5, 50, 95], axis=0)
    )


def test_scientific_environment_uses_float64():
    assert jax.config.jax_enable_x64, (
        "Run scientific tests with JAX_ENABLE_X64=true"
    )
    assert jnp.asarray([1.0], dtype=jnp.float64).dtype == jnp.float64
