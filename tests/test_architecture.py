"""Core contracts independent of orchestration and frozen sampling traces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import BSpline

from log_psplines import (
    LogPSpline,
    PipelineConfig,
    PSDResult,
    SpectralMatrix,
    SplineBasis,
    TimeSeries,
    fit,
)
from log_psplines.likelihoods.whittle import whittle_log_likelihood
from log_psplines.likelihoods.wishart import wishart_log_likelihood


def test_basis_and_integrated_penalty_independent_reference():
    knots = np.array([0.0, 0.2, 0.65, 1.0])
    grid = np.linspace(0, 1, 17) ** 1.3
    frequency = SplineBasis.from_knots(grid, knots)
    padded = np.r_[np.repeat(0.0, 3), knots, np.repeat(1.0, 3)]
    splines = BSpline(padded, np.eye(len(padded) - 4), 3)
    np.testing.assert_allclose(frequency.basis, splines(grid), atol=5e-8)
    second = splines.derivative(2)
    penalty = np.array(
        [
            [
                quad(
                    lambda x: second(x)[i] * second(x)[j],
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
    np.testing.assert_allclose(frequency.penalty, penalty, atol=5e-8)
    model = LogPSpline(frequency)
    weights = jnp.arange(6.0) / 10
    np.testing.assert_allclose(
        model(weights), splines(grid) @ weights, atol=5e-8
    )
    surface = LogPSpline(frequency, time=frequency)(jnp.zeros((6, 6)))
    np.testing.assert_array_equal(surface, np.zeros((17, 17)))


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_matrix_trailing_dimensions(channels):
    rng = np.random.default_rng(12)
    logs = rng.normal(size=(2, 7, channels)) / 3
    theta = rng.normal(size=(2, 7, channels * (channels - 1) // 2)) / 5
    matrix = SpectralMatrix(channels)
    spectrum = matrix(logs, theta, theta / 2)
    assert spectrum.shape == (2, 7, channels, channels)
    for t in range(2):
        for f in range(7):
            triangular = np.eye(channels, dtype=complex)
            triangular[np.tril_indices(channels, -1)] = -theta[t, f] * (
                1 + 0.5j
            )
            expected = np.linalg.inv(
                triangular.conj().T @ np.diag(np.exp(-logs[t, f])) @ triangular
            )
            np.testing.assert_allclose(spectrum[t, f], expected, atol=1e-14)
    np.testing.assert_allclose(
        spectrum, spectrum.conj().swapaxes(-1, -2), atol=1e-14
    )
    assert np.linalg.eigvalsh(spectrum).min() > 0
    coherence = matrix.coherence(spectrum)
    assert coherence.min() >= 0 and coherence.max() <= 1 + 1e-14


def test_likelihoods_normalization_and_gradients():
    rng = np.random.default_rng(10)
    logs = jnp.asarray(rng.normal(size=7))
    u = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    previous = rng.normal(size=(7, 2, 3)) + 1j * rng.normal(size=(7, 2, 3))
    theta = rng.normal(size=(7, 2)) + 1j * rng.normal(size=(7, 2))
    residual = u - np.einsum("fl,flr->fr", theta, previous)
    expected = (
        (
            -6 * np.sum(logs)
            - np.sum(np.abs(residual) ** 2 / np.exp(logs)[:, None] / 4)
        )
        * 0.7
        / 1.5
    )

    def likelihood(logs):
        return wishart_log_likelihood(
            logs,
            theta.real,
            theta.imag,
            u.real,
            u.imag,
            previous.real,
            previous.imag,
            Nb=2,
            Nh=3,
            duration=4,
            enbw=1.5,
            eta=0.7,
        )

    np.testing.assert_allclose(jax.jit(likelihood)(logs), expected, rtol=2e-6)
    assert np.isfinite(jax.grad(likelihood)(logs)).all()
    scalar = whittle_log_likelihood(logs, np.abs(u[:, 0]) ** 2, duration=4)
    one_channel = wishart_log_likelihood(
        logs,
        jnp.zeros((7, 0)),
        jnp.zeros((7, 0)),
        u[:, :1].real,
        u[:, :1].imag,
        jnp.zeros((7, 0, 1)),
        jnp.zeros((7, 0, 1)),
        duration=4,
    )
    np.testing.assert_allclose(scalar, one_channel, rtol=1e-6)


@pytest.mark.parametrize("channels", [1, 2])
def test_fit_result_roundtrip(channels, tmp_path):
    data = TimeSeries(
        data=np.random.default_rng(15).normal(size=(32, channels))
    )
    assert data.data.shape == (32, channels)
    result = fit(
        data,
        PipelineConfig(
            n_knots=4,
            vi_steps=3,
            vi_posterior_draws=3,
            n_samples=3,
            n_warmup=3,
            rng_key=5,
            verbose=False,
            vi_progress_bar=False,
        ),
    )
    spectra = result.spectral_density
    assert spectra.shape == (1, 3, len(result.frequency), channels, channels)
    assert np.isfinite(spectra).all()
    assert np.linalg.eigvalsh(spectra).min() > 0
    assert result.coherence.min() >= 0 and result.coherence.max() <= 1 + 1e-12
    assert result.time is None
    path = tmp_path / "result.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    np.testing.assert_allclose(restored.spectral_density, spectra)
    np.testing.assert_array_equal(restored.frequency, result.frequency)
    for name in result.posterior:
        np.testing.assert_array_equal(
            result.posterior[name], restored.posterior[name]
        )
