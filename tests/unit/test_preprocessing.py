"""Spectral normalization and frequency selection independent of inference."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import csd, welch

from log_psplines import StationaryConfig
from log_psplines.data.timeseries import TimeSeries
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.preprocessing.periodogram import (
    _regularize_wishart,
    compute_wishart,
    empirical_spectrum,
)
from log_psplines.preprocessing.spectral import (
    align_true_psd_to_freq,
    preprocess_to_freq_domain,
)


@pytest.mark.parametrize(
    "window,detrend", [(None, False), ("hann", "constant"), ("hann", "linear")]
)
def test_wishart_auto_spectra_match_welch(window, detrend):
    x = np.random.default_rng(8).normal(size=(256, 2))
    data = compute_wishart(x, fs=32.0, Nb=4, window=window, detrend=detrend)
    for channel in range(2):
        freq, expected = welch(
            x[:, channel],
            fs=32.0,
            nperseg=64,
            noverlap=0,
            window="boxcar" if window is None else window,
            detrend=detrend,
        )
        np.testing.assert_allclose(data.freq, freq[1:])
        np.testing.assert_allclose(
            data.raw_psd[:, channel, channel].real, expected[1:], rtol=1e-12
        )
    np.testing.assert_allclose(data.enbw, 1.0 if window is None else 1.5)


def test_empirical_cross_spectrum_matches_scipy():
    x = np.random.default_rng(9).normal(size=(128, 2))
    x[:, 1] += 0.5 * x[:, 0]
    empirical = empirical_spectrum(x, fs=32.0, nperseg=32, noverlap=16)
    freq, expected = csd(x[:, 0], x[:, 1], fs=32.0, nperseg=32, noverlap=16)
    np.testing.assert_allclose(empirical.freq, freq)
    np.testing.assert_allclose(empirical.psd[:, 0, 1], expected)
    np.testing.assert_allclose(
        empirical.psd, empirical.psd.conj().swapaxes(-1, -2)
    )
    assert np.all((empirical.coherence >= 0) & (empirical.coherence <= 1))


@pytest.mark.parametrize("bins", [{"Nh": 4}, {"Nc": 8}, {"Nc": 7}])
def test_coarse_graining_preserves_wishart_sums_and_psd_means(bins):
    data = compute_wishart(
        np.random.default_rng(10).normal(size=(256, 2)), fs=32.0, Nb=4
    )
    spec = compute_binning_structure(data.freq, **bins)
    coarse = apply_coarse_grain_multivar_fft(data, spec)
    n = spec.Nc * spec.Nh
    expected_y = data.Y[:n].reshape(spec.Nc, spec.Nh, 2, 2).sum(axis=1)
    expected_psd = (
        data.raw_psd[:n].reshape(spec.Nc, spec.Nh, 2, 2).mean(axis=1)
    )
    np.testing.assert_allclose(coarse.Y, expected_y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        coarse.raw_psd, expected_psd, rtol=1e-12, atol=1e-12
    )
    assert np.linalg.eigvalsh(coarse.raw_psd).min() > 0
    assert coarse.Nh == spec.Nh and coarse.Nb == data.Nb


def test_coarse_graining_rejects_nondivisible_explicit_bin_width():
    with pytest.raises(ValueError, match="must divide"):
        compute_binning_structure(np.arange(10.0), Nh=3)
    with pytest.raises(ValueError, match="Exactly one"):
        CoarseGrainConfig(Nc=2, Nh=3)


def test_pipeline_coarse_graining_and_frequency_mask():
    ts = TimeSeries(
        np.random.default_rng(11).normal(size=(256, 2)), np.arange(256) / 32.0
    )
    config = StationaryConfig(
        Nb=4, verbose=False, coarse_grain_config={"enabled": True, "Nc": 8}
    )
    baseline = preprocess_to_freq_domain(ts, config)
    masked_config = replace(
        config,
        exclude_freq_bands=[
            (baseline.freq[2], baseline.freq[1]),
            (baseline.freq[2], baseline.freq[3]),
        ],
    )
    masked = preprocess_to_freq_domain(ts, masked_config)
    keep = np.ones(8, dtype=bool)
    keep[1:4] = False
    np.testing.assert_allclose(masked.freq, baseline.freq[keep])
    np.testing.assert_allclose(masked.Y, baseline.Y[keep])
    masked_config = replace(config, exclude_freq_bands=[(0.0, 32.0)])
    with pytest.raises(ValueError, match="removed all"):
        preprocess_to_freq_domain(ts, masked_config)


def test_truth_alignment_interpolates_complex_spectra_without_changing_data():
    data = compute_wishart(
        np.random.default_rng(12).normal(size=(64, 2)), fs=16.0, Nb=2
    )
    original = data.Y.copy()
    freq = np.array([0.0, 8.0])
    truth = np.array([np.eye(2), np.eye(2) * 3], dtype=complex)
    truth[:, 0, 1] = [0.1j, 0.3j]
    truth[:, 1, 0] = truth[:, 0, 1].conj()
    aligned = align_true_psd_to_freq({"freq": freq, "psd": truth}, data)
    expected_scale = 1 + data.freq / 4
    np.testing.assert_allclose(aligned[:, 0, 0], expected_scale)
    np.testing.assert_allclose(aligned[:, 0, 1], 0.1j * expected_scale)
    np.testing.assert_array_equal(data.Y, original)
    with pytest.raises(ValueError, match="matching lengths"):
        align_true_psd_to_freq((freq, truth[:1]), data)


def test_nearly_singular_wishart_regularization_respects_local_scale():
    scales = np.array([1e-30, 1e-15, 1.0, 1e5])
    vector = np.array([1, 1 + 1e-10j, 0.7 - 0.2j])
    original = scales[:, None, None] * np.outer(vector, vector.conj())
    matrices = _regularize_wishart(original, 1e-6)
    assert np.isfinite(matrices).all()
    normalized = (
        matrices / np.trace(matrices, axis1=-2, axis2=-1).real[:, None, None]
    )
    np.testing.assert_allclose(
        normalized, normalized.conj().swapaxes(-1, -2), atol=1e-14
    )
    assert np.linalg.eigvalsh(normalized).min() > 0
    np.testing.assert_allclose(
        normalized,
        np.broadcast_to(normalized[-1], normalized.shape),
        atol=1e-14,
    )
    recovered_scale = np.trace(matrices, axis1=-2, axis2=-1).real
    np.testing.assert_allclose(
        recovered_scale / recovered_scale[-1], scales / scales[-1], rtol=1e-10
    )


@pytest.mark.parametrize("channels", [1, 3])
def test_known_signal_periodogram_normalization_dc_and_nyquist(channels):
    n, fs, peak = 128, 128.0, 11.0
    time = np.arange(n) / fs
    signal = (
        2.0 * np.cos(2 * np.pi * peak * time)
        + 0.5 * (-1.0) ** np.arange(n)
        + 7.0
    )
    x = np.repeat(signal[:, None], channels, axis=1)
    data = compute_wishart(x, fs=fs, Nb=1, detrend="constant")
    assert data.raw_psd.shape == (n // 2, channels, channels)
    assert data.freq[0] == fs / n and data.freq[-1] == fs / 2
    power = data.raw_psd[:, 0, 0].real
    assert data.freq[np.argmax(power)] == peak
    np.testing.assert_allclose(
        power.sum() * fs / n, 2.0**2 / 2 + 0.5**2, atol=1e-12
    )
    np.testing.assert_allclose(power[-1] * fs / n, 0.5**2, atol=1e-12)
