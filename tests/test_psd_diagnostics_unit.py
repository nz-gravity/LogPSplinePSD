import numpy as np
import pytest

from log_psplines.diagnostics.psd_diagnostics import (
    PSDDiagnostics,
    anderson_darling_statistic,
    anderson_p_value,
)


def test_psd_diagnostics_shape_mismatch_raises():
    ts_data = np.zeros(8)
    freqs = np.array([0.5, 1.0, 1.5])
    psd = np.ones(2)
    with pytest.raises(ValueError):
        PSDDiagnostics(ts_data, 4.0, freqs, psd)


def test_psd_diagnostics_short_series_rayleigh_defaults():
    ts_data = np.linspace(0.0, 1.0, 4)
    freqs = np.array([0.5, 1.0, 1.5])
    psd = np.ones_like(freqs)
    diag = PSDDiagnostics(
        ts_data,
        4.0,
        freqs,
        psd,
        reference_psd=psd * 2.0,
        fftlength=2.0,
    )
    np.testing.assert_allclose(diag.rayleigh_spectrum, np.ones_like(freqs))
    np.testing.assert_allclose(diag.rayleigh_freqs, freqs)
    np.testing.assert_allclose(diag.rayleigh_spectrum_ref, np.ones_like(freqs))


def test_psd_diagnostics_summary_stats_with_reference():
    ts_data = np.zeros(8)
    freqs = np.array([0.5, 1.0, 1.5])
    psd = np.ones_like(freqs)
    reference = psd * 2.0
    diag = PSDDiagnostics(
        ts_data,
        4.0,
        freqs,
        psd,
        reference_psd=reference,
        fftlength=2.0,
    )
    stats = diag.summary_stats()
    expected_mse = float(np.mean((psd - reference) ** 2))
    expected_mae = float(np.mean(np.abs(psd - reference)))
    assert stats["mse"] == pytest.approx(expected_mse)
    assert stats["mae"] == pytest.approx(expected_mae)
    assert stats["rayleigh_mean"] == pytest.approx(1.0)
    assert stats["rayleigh_median"] == pytest.approx(1.0)
    assert stats["fraction_coherent"] == pytest.approx(0.0)
    assert stats["fraction_gaussian"] == pytest.approx(1.0)


def test_fbins_anderson_p_value_returns_bins():
    rng = np.random.default_rng(42)
    ts_data = rng.normal(size=256)
    freqs = np.linspace(0.5, 10.0, 64)
    psd = np.ones_like(freqs)
    diag = PSDDiagnostics(ts_data, 64.0, freqs, psd, fftlength=10.0)
    data = rng.normal(size=64) + 1j * rng.normal(size=64)
    fbins, pvals = diag._fbins_anderson_p_value(
        freqs, data, bin_width_Hz=2.0, fmin=0.5, fmax=10.0
    )
    assert fbins.shape == pvals.shape
    assert fbins.size > 0


def test_fbins_anderson_p_value_empty_range():
    rng = np.random.default_rng(0)
    ts_data = rng.normal(size=128)
    freqs = np.linspace(0.5, 5.0, 32)
    psd = np.ones_like(freqs)
    diag = PSDDiagnostics(ts_data, 32.0, freqs, psd, fftlength=10.0)
    data = rng.normal(size=32) + 1j * rng.normal(size=32)
    fbins, pvals = diag._fbins_anderson_p_value(
        freqs, data, bin_width_Hz=1.0, fmin=100.0, fmax=200.0
    )
    assert fbins.size == 0
    assert pvals.size == 0


def test_anderson_stats_edge_cases():
    assert np.isnan(anderson_darling_statistic(np.array([1.0])))
    assert np.isnan(anderson_p_value(np.array([])))
    assert np.isnan(anderson_p_value(np.array([1.0 + 0.0j, 2.0 + 0.0j])))
