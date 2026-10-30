"""Adaptive frequency binning preserves multivariate sufficient statistics."""

import jax.numpy as jnp
import numpy as np
import pytest

from log_psplines.data.spectral import WishartData
from log_psplines.data.spectral_utils import Y_to_U
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    CoarseGrainSpec,
    apply_coarse_grain_multivar_fft,
    compute_adaptive_binning_structure,
)
from log_psplines.preprocessing.periodogram import compute_wishart


def test_irregular_partition_preserves_complex_wishart_sums_and_counts():
    rng = np.random.default_rng(45)
    fine = compute_wishart(rng.normal(size=(192, 2)), fs=32.0, Nb=4)
    stops = np.array([1, 4, 6, 10, fine.N])
    starts = np.r_[0, stops[:-1]]
    mids = starts + (stops - starts - 1) // 2
    spec = CoarseGrainSpec(
        f_coarse=fine.freq[mids],
        Nh=1,
        J_start=starts,
        J_stop=stops,
        J_mid=mids,
        Nc=len(stops),
        freq_bin_counts=stops - starts,
    )
    pooled = apply_coarse_grain_multivar_fft(fine, spec)
    expected = np.add.reduceat(fine.Y, starts, axis=0)
    np.testing.assert_allclose(pooled.Y, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(pooled.freq_bin_counts, stops - starts)
    np.testing.assert_array_equal(pooled.counts, fine.Nb * (stops - starts))
    expected_psd = expected / (
        fine.Nb * (stops - starts)[:, None, None] * fine.duration
    ) * fine.scaling_factor
    np.testing.assert_allclose(pooled.raw_psd, expected_psd, rtol=1e-12)
    np.testing.assert_allclose(
        pooled.apply_mask(np.array([True, False, True, True, False])).counts,
        fine.Nb * (stops - starts)[[0, 2, 3]],
    )
    # The imaginary cross-spectrum must survive, not just the diagonal powers.
    assert np.max(np.abs(pooled.Y[:, 0, 1].imag)) > 0


def test_binned_likelihood_equals_fine_likelihood_for_piecewise_constant_model():
    rng = np.random.default_rng(16)
    fine = compute_wishart(rng.normal(size=(160, 2)), fs=40.0, Nb=4)
    stops = np.array([2, 5, fine.N])
    starts = np.r_[0, stops[:-1]]
    mids = starts + (stops - starts - 1) // 2
    spec = CoarseGrainSpec(
        f_coarse=fine.freq[mids], Nh=1, J_start=starts,
        J_stop=stops, J_mid=mids, Nc=len(stops),
    )
    pooled = apply_coarse_grain_multivar_fft(fine, spec)
    logd_bins = np.array([0.3, -0.2, 0.1])
    theta_bins = np.array([0.2 + 0.1j, 0.25 - 0.1j, -0.3 + 0.2j])
    logd_fine = np.repeat(logd_bins, spec.freq_bin_counts)
    theta_fine = np.repeat(theta_bins, spec.freq_bin_counts)

    def evaluate(data, logd, theta):
        u = data.U
        return wishart_log_likelihood(
            jnp.asarray(logd),
            jnp.asarray(theta.real[:, None]),
            jnp.asarray(theta.imag[:, None]),
            jnp.asarray(u[:, 1, :].real),
            jnp.asarray(u[:, 1, :].imag),
            jnp.asarray(u[:, :1, :].real),
            jnp.asarray(u[:, :1, :].imag),
            Nb=data.Nb,
            Nh=data.Nh,
            counts=jnp.asarray(data.counts),
            duration=data.duration,
            enbw=data.enbw,
            eta=0.73,
        )

    np.testing.assert_allclose(
        evaluate(fine, logd_fine, theta_fine),
        evaluate(pooled, logd_bins, theta_bins),
        rtol=1e-6, atol=1e-6,
    )


def test_adaptive_bins_shrink_at_cross_spectral_phase_change():
    n = 48
    frequency = np.arange(1, n + 1, dtype=float) * 0.1
    phase = np.where(np.arange(n) < n // 2, 0.0, np.pi)
    y = np.broadcast_to(np.diag([1.0, 3.0]).astype(complex), (n, 2, 2)).copy()
    y[:, 0, 1] = 0.7 * np.exp(1j * phase)
    y[:, 1, 0] = y[:, 0, 1].conj()
    u = Y_to_U(y)
    fine = WishartData(
        u_re=u.real, u_im=u.imag, freq=frequency, N=n, p=2,
        Nb=1, raw_psd=y,
    )
    spec = compute_adaptive_binning_structure(
        fine, tolerance=0.15, max_bin_width=12, pilot_window=5
    )
    assert spec.freq_bin_counts.sum() == n
    assert spec.freq_bin_counts.max() == 12
    assert spec.freq_bin_counts.min() < 12
    assert np.any(spec.freq_bin_counts == 12)
    # No aggregate may mix two substantially different cross-spectral phases.
    for start, stop in zip(spec.J_start, spec.J_stop):
        assert not (start < n // 2 < stop)


def test_adaptive_respects_gaps_and_excludes_masked_frequencies():
    n = 40
    frequency = np.arange(1, n + 1, dtype=float)
    y = np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy()
    u = Y_to_U(y)
    fine = WishartData(
        u_re=u.real, u_im=u.imag, freq=frequency, N=n, p=2, raw_psd=y,
    )
    mask = ~((frequency >= 12) & (frequency <= 18))
    filtered = fine.apply_mask(mask)
    spec = compute_adaptive_binning_structure(
        filtered, max_bin_width=10, pilot_window=3
    )
    assert spec.freq_bin_counts.sum() == mask.sum()
    for start, stop in zip(spec.J_start, spec.J_stop):
        assert not (filtered.freq[start] < 12 and filtered.freq[stop - 1] > 18)


def test_adaptive_config_validates_inputs():
    config = CoarseGrainConfig(method="adaptive", Nc=None)
    assert config.method == "adaptive"
    with pytest.raises(ValueError, match="positive"):
        CoarseGrainConfig(method="adaptive", tolerance=0.0)
    with pytest.raises(ValueError, match="odd"):
        CoarseGrainConfig(method="adaptive", pilot_window=8)
