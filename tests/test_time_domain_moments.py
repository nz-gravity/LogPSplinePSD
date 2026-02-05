import numpy as np

from log_psplines.diagnostics.time_domain_moments import (
    compute_empirical_covariances,
    compute_empirical_variances,
    compute_psd_covariances,
    compute_psd_variances,
)


def test_compute_psd_variances_constant_psd():
    freqs = np.array([0.0, 0.5, 1.0])
    psd = np.ones((2, 2, 3))  # (n_samples, p, N)
    # Integral of 1 over [0, 1] using trapezoid rule with points [0, 0.5, 1.0] is 1.0
    expected = np.ones((2, 2))
    result = compute_psd_variances(psd, freqs)
    np.testing.assert_allclose(result, expected)


def test_compute_psd_covariances_real_cross_psd():
    freqs = np.array([0.0, 1.0, 2.0])
    # cross-PSD is linear in frequency for pair (0,1)
    base = np.stack([freqs, freqs], axis=0)  # (n_samples=2, N)
    psd = np.zeros((2, 2, 2, 3), dtype=float)
    psd[:, 0, 1, :] = base
    psd[:, 1, 0, :] = base
    covariances = compute_psd_covariances(psd, freqs, [(0, 1)])
    expected = np.trapezoid(base, freqs, axis=-1)[:, None]
    np.testing.assert_allclose(covariances, expected)


def test_empirical_moments():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 3))
    variances = compute_empirical_variances(data)
    covariances = compute_empirical_covariances(data, [(0, 1), (1, 2)])

    np.testing.assert_allclose(variances, np.var(data, axis=0, ddof=1))
    cov_matrix = np.cov(data, rowvar=False, ddof=1)
    np.testing.assert_allclose(
        covariances, [cov_matrix[0, 1], cov_matrix[1, 2]]
    )
