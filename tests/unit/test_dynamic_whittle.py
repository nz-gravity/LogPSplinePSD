"""Deterministic moving-periodogram and stationarity contracts."""

import jax
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from scipy.special import logsumexp

from log_psplines.diagnostics.stationarity import (
    horizon_summary,
    stationarity_loss,
    whittle_loss,
)
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.preprocessing.moving_periodogram import (
    scattered_moving_periodogram,
    tang_moving_periodogram,
)


@pytest.mark.parametrize(
    "m,thin,n", [(1, 1, 17), (3, 2, 83), (8, 3, 139), (16, 2, 197)]
)
@pytest.mark.parametrize("dt", [0.25, 2.5])
def test_direct_dft_coordinates_and_truncation(m, thin, n, dt):
    x = np.random.default_rng(1).normal(size=n)
    x -= x.mean()
    raw = tang_moving_periodogram(x, m=m, thin=thin)
    coefficients, centers, frequency = [], [], []
    for block in range((n - 2 * m) // (thin * m)):
        for rung in range(1, m + 1):
            start = thin * m * block + rung - 1
            omega = 2 * np.pi * rung / (2 * m + 1)
            z = sum(
                x[start + k] * np.exp(-1j * omega * k)
                for k in range(2 * m + 1)
            )
            coefficients.append(z / np.sqrt(2 * np.pi * (2 * m + 1)))
            centers.append((start + m + 1) / n)
            frequency.append(omega / (2 * np.pi * dt))
            assert start >= 0 and start + 2 * m < n
    np.testing.assert_allclose(raw["coeff"], coefficients, atol=2e-14)
    np.testing.assert_array_equal(raw["u"], centers)
    data = scattered_moving_periodogram(x, dt=dt, m=m, thin=thin)
    np.testing.assert_allclose(data.frequency, frequency)
    np.testing.assert_array_equal(data.time, centers)
    np.testing.assert_allclose(data.power, 2 * np.abs(coefficients) ** 2)
    assert np.all(data.counts == 2)


def test_fourier_probe_and_exponential_likelihood_gradient():
    m, n = 8, 100
    x = np.cos(2 * np.pi * 3 * np.arange(n) / (2 * m + 1))
    raw = tang_moving_periodogram(x, m=m)
    # First complete window at rung 3 has the exact Fourier-probe modulus.
    np.testing.assert_allclose(
        abs(raw["coeff"][2]), np.sqrt((2 * m + 1) / (8 * np.pi))
    )
    data = scattered_moving_periodogram(x, dt=0.2, m=m)
    eta = np.linspace(-1.0, 1.0, data.power.size)
    def likelihood(e):
        return power_whittle_log_likelihood(
            data.power, data.counts, e
        )
    np.testing.assert_allclose(
        likelihood(eta), -np.sum(eta + raw["mi"] * np.exp(-eta)), rtol=1e-14
    )
    step = 1e-5
    finite = np.array(
        [
            (likelihood(eta + step * v) - likelihood(eta - step * v))
            / (2 * step)
            for v in np.eye(eta.size)
        ]
    )
    np.testing.assert_allclose(jax.grad(likelihood)(eta), finite, atol=2e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": 0},
        {"m": True},
        {"m": 1.5},
        {"m": 2, "thin": 0},
        {"m": 2, "thin": True},
        {"m": 2, "dt": 0.0},
        {"m": 2, "dt": np.inf},
        {"m": 2, "dt": np.nan},
    ],
)
def test_invalid_transform_arguments(kwargs):
    with pytest.raises((ValueError, TypeCheckError)):
        scattered_moving_periodogram(np.ones(64), **{"dt": 1.0, **kwargs})


@pytest.mark.parametrize(
    "x",
    [
        np.ones((8, 2)),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.ones(2),
    ],
)
def test_invalid_raw_data(x):
    with pytest.raises((ValueError, TypeCheckError)):
        tang_moving_periodogram(x, m=2)


def test_jensen_gap_scale_invariance_and_direct_identity():
    logs = np.random.default_rng(3).normal(size=(4, 7, 5))
    a, w = np.arange(1, 8, dtype=float), np.arange(1, 6, dtype=float)
    a /= a.sum()
    w /= w.sum()
    mean_log = logsumexp(logs, b=a[:, None], axis=-2)
    direct = (whittle_loss(logs, mean_log[:, None, :]) * a[:, None] * w).sum(
        axis=(-2, -1)
    )
    gap = stationarity_loss(logs, time_weights=a, frequency_weights=w)
    np.testing.assert_allclose(gap, direct, atol=1e-14)
    np.testing.assert_allclose(
        stationarity_loss(
            logs + np.linspace(-100, 100, 5),
            time_weights=a,
            frequency_weights=w,
        ),
        gap,
        atol=1e-14,
    )
    constant = np.broadcast_to(np.array([-20.0, 4.0, 0.0, -4.0, 10.0]), (7, 5))
    np.testing.assert_allclose(stationarity_loss(constant), 0, atol=1e-14)


@pytest.mark.parametrize("slope", [0.0, 1e-8, 0.01, 1.0, 100.0])
def test_discrete_ramp(slope):
    m, dt = 12, 0.25
    x = slope * dt
    logs = (x * np.arange(-m, m + 1))[:, None]
    if abs(x) < 1e-6:
        expected = x * x * m * (m + 1) / 6
    else:

        def logsinh(z):
            return z + np.log(-np.expm1(-2 * z)) - np.log(2)

        expected = (
            logsinh((2 * m + 1) * abs(x) / 2)
            - logsinh(abs(x) / 2)
            - np.log(2 * m + 1)
        )
    np.testing.assert_allclose(stationarity_loss(logs), expected, atol=2e-13)


def test_nonmonotonic_excursion_and_joint_prefix_event():
    gap = []
    for m in (1, 4, 16, 64):
        logs = np.zeros((2 * m + 1, 1))
        logs[m] = 2
        gap.append(stationarity_loss(logs))
    assert gap[-1] < gap[0]
    raw = horizon_summary(
        np.array([gap]), np.array([1, 4, 16, 64]), epsilon=0.1
    )
    prefix = horizon_summary(
        np.array([gap]), np.array([1, 4, 16, 64]), epsilon=0.1, prefix=True
    )
    assert raw["selected"] == 64 and raw["right_censored"]
    assert prefix["selected"] == -1 and prefix["none_acceptable"]
    losses = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = horizon_summary(
        losses, np.array([1, 2]), epsilon=0.5, q=0.5, prefix=True
    )
    np.testing.assert_array_equal(result["probability"], [0.5, 0.0])


def test_quadrature_refinement_and_invalid_support():
    values = [
        stationarity_loss(
            np.linspace(-0.4, 0.4, n)[:, None],
            time_weights=np.r_[0.5, np.ones(n - 2), 0.5],
        )
        for n in (17, 65, 257)
    ]
    assert abs(values[2] - values[1]) < abs(values[1] - values[0])
    for logs, weights in [
        (np.empty((0, 1)), None),
        (np.ones((2, 0)), None),
        (np.full((2, 1), np.nan), None),
        (np.ones((2, 1)), np.array([-1.0, 2.0])),
        (np.ones((2, 1)), np.zeros(2)),
    ]:
        with pytest.raises(ValueError):
            stationarity_loss(logs, time_weights=weights)
