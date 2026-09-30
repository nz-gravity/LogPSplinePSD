"""Small mathematical contracts for spline PSD estimation."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline

from log_psplines import SpectralMatrix, SplineBasis
from log_psplines.likelihoods.whittle import (
    power_whittle_log_likelihood,
    whittle_log_likelihood,
)
from log_psplines.likelihoods.wishart import wishart_log_likelihood


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


def test_spectral_matrix_is_hermitian_positive_and_coherence_is_defined():
    logs = np.array([[0.2, -0.3], [0.4, 0.1]])
    theta_re = np.array([[0.25], [-0.1]])
    theta_im = np.array([[0.15], [0.2]])
    matrix = SpectralMatrix(2)
    spectrum = np.asarray(matrix(logs, theta_re, theta_im))

    for f in range(logs.shape[0]):
        triangular = np.array(
            [[1, 0], [-theta_re[f, 0] - 1j * theta_im[f, 0], 1]],
            dtype=complex,
        )
        expected = np.linalg.inv(
            triangular.conj().T @ np.diag(np.exp(-logs[f])) @ triangular
        )
        np.testing.assert_allclose(spectrum[f], expected)
    np.testing.assert_allclose(spectrum, spectrum.conj().swapaxes(-1, -2))
    assert np.linalg.eigvalsh(spectrum).min() > 0
    coherence = np.asarray(matrix.coherence(spectrum))
    expected_coherence = np.abs(spectrum[..., 0, 1]) ** 2 / (
        spectrum[..., 0, 0].real * spectrum[..., 1, 1].real
    )
    np.testing.assert_allclose(coherence[..., 0, 1], expected_coherence)
    assert np.all((coherence >= 0) & (coherence <= 1))


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


def test_masked_wdm_powers_have_zero_likelihood_and_gradient():
    logs = jnp.array([np.nan, -1000.0, np.inf, np.log(2.0)])
    powers = jnp.array([np.nan, 0.0, np.inf, 3.0])
    counts = jnp.array([0.0, 0.0, 0.0, 1.0])

    def likelihood(logs):
        return power_whittle_log_likelihood(powers, counts, logs)

    value, gradient = jax.jit(jax.value_and_grad(likelihood))(logs)
    np.testing.assert_allclose(value, -0.5 * (np.log(2.0) + 1.5))
    np.testing.assert_allclose(gradient, [0, 0, 0, 0.25])


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
