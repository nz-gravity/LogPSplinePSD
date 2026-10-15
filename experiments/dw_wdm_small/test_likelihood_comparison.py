"""Independent controls for the restricted Gaussian likelihood reference."""

import numpy as np
from experiments.dw_wdm_small import likelihood_comparison as comparison
from scipy import sparse, stats


def test_observed_reference_is_marginal_not_conditional_or_periodic():
    diagonal = np.array([1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    lower = np.array([0.4, 0.1, -0.2, 0.3, -0.1, 0.2])
    observed = np.array([True, True, False, False, True, True])
    indices, cholesky = comparison.observed_cholesky(diagonal, lower, observed)
    covariance = (
        np.diag(diagonal) + np.diag(lower[1:], 1) + np.diag(lower[1:], -1)
    )
    marginal = covariance[np.ix_(indices, indices)]
    records = np.array(
        [[0.2, -1, 100, -400, 0.9, 0.3], [-0.3, 0.5, 1e20, -1e20, 0.1, 1]]
    )
    expected = np.einsum(
        "bi,ij,bj->b",
        records[:, indices],
        np.linalg.inv(marginal),
        records[:, indices],
    )
    np.testing.assert_allclose(
        comparison.exact_quadratic(records, indices, cholesky),
        expected,
        rtol=1e-14,
    )
    # Omitting a correlated neighbour removes its conditioning information.
    conditional = np.linalg.inv(
        np.linalg.inv(covariance)[np.ix_(indices, indices)]
    )
    assert np.max(abs(conditional - marginal)) > 0.01


def test_complex_moments_match_real_quadratic_form_including_pseudo_covariance():
    rng = np.random.default_rng(391)
    rows = rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))
    rows /= np.linalg.norm(rows, axis=1)[:, None]
    diagonal = np.arange(6) / 10 + 1
    lower = np.full(6, 0.1)
    covariance = (
        np.diag(diagonal) + np.diag(lower[1:], 1) + np.diag(lower[1:], -1)
    )
    reference = np.array([1.1, 1.2, 1.4, 1.3])
    quadratic = 2 * np.real(rows.conjugate().T @ np.diag(1 / reference) @ rows)
    expected_mean = np.trace(quadratic @ covariance)
    expected_variance = 2 * np.trace(
        quadratic @ covariance @ quadratic @ covariance
    )
    mean, variance = comparison.mp_moments(
        sparse.csr_matrix(rows), diagonal, lower, reference
    )
    np.testing.assert_allclose(
        [mean, variance], [expected_mean, expected_variance], rtol=2e-14
    )


def test_proper_fourier_and_improper_real_limits_have_different_information():
    j = np.arange(7)
    proper = np.exp(-2j * np.pi * np.array([1, 2])[:, None] * j / 7) / np.sqrt(
        7
    )
    real = np.eye(7)[:2].astype(complex)
    arguments = (np.ones(7), np.zeros(7), np.ones(2))
    np.testing.assert_allclose(
        comparison.mp_moments(sparse.csr_matrix(proper), *arguments),
        [4, 8],
        atol=2e-14,
    )
    np.testing.assert_allclose(
        comparison.mp_moments(sparse.csr_matrix(real), *arguments),
        [4, 16],
        atol=2e-14,
    )


def test_inverse_gamma_posterior_matches_reciprocal_gamma_quantiles():
    q = np.array([200.0, 199.1, 204.9])
    nu = 200
    expected = (1 + q[None, :] / 2) / stats.gamma.ppf(
        np.array([0.95, 0.5, 0.05])[:, None], a=2 + nu / 2
    )
    np.testing.assert_allclose(
        comparison.posterior_quantiles(q, nu), expected, rtol=2e-14
    )
