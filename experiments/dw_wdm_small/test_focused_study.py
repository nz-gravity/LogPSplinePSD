"""Small regression fixtures for the focused study; no inference."""

import numpy as np
import pytest
from experiments.dw_wdm_small import benchmark as b
from experiments.dw_wdm_small import focused_study as study
from experiments.dw_wdm_small import gap_bias_audit as audit
from experiments.dw_wdm_small import gap_smoke as gap
from experiments.dw_wdm_small.lisa_preflight import SelectedFourier, kernel

from log_psplines import SplineBasis
from log_psplines.models.anova import ANOVALogPSpline
from log_psplines.preprocessing.moving_periodogram import (
    tang_moving_periodogram,
)


@pytest.mark.parametrize("m", [1, 4, 32, 97])
def test_selected_fft_matches_original_coefficients(m):
    x = np.random.default_rng(m).normal(size=8 * m + 1)
    native = tang_moving_periodogram(x, m=m, thin=2)
    engine = SelectedFourier(m)
    for block in range(len(native["coeff"]) // m):
        start = 2 * m * block
        actual = engine.block(x[start : start + 3 * m]) / np.sqrt(
            2 * np.pi * (2 * m + 1)
        )
        np.testing.assert_allclose(
            actual,
            native["coeff"][block * m : (block + 1) * m],
            rtol=1e-10,
            atol=2e-12,
        )


def test_large_window_selected_fft_direct_sum():
    m = 7680
    length = 2 * m + 1
    x = np.random.default_rng(11).normal(size=3 * m)
    actual = SelectedFourier(m).block(x)
    for j in (0, 1, 33, 3840, 7679):
        expected = x[j : j + length] @ np.exp(
            -2j * np.pi * (j + 1) * np.arange(length) / length
        )
        np.testing.assert_allclose(actual[j], expected, rtol=2e-8, atol=1e-8)


def test_real_signal_kernel_white_units_and_nonpointwise_response():
    length = 65
    dt = 1.0
    f = (np.arange(8 * length) + 0.5) / (16 * length * dt)
    df = 1 / (16 * length * dt)
    weights = kernel(f, 1 / (length * dt), length, dt) * df
    np.testing.assert_allclose(weights.sum(), 1, atol=2e-13)
    # A smooth coloured spectrum demonstrates that its projected response
    # differs from its value at the selected frequency.
    spectrum = 1 + 30 * f**2
    projected = weights @ spectrum
    assert abs(projected - (1 + 30 / (length * dt) ** 2)) > 0.01


def test_mask_preserves_schedule_and_power_count_factors():
    x = b.simulate(6101)
    observed = study.masks()["single"]
    full = b.prepare(x, "dw")
    masked = gap.dw_data(x, observed)
    keep = gap.dw_keep(observed)
    np.testing.assert_array_equal(masked.time, full.time[keep])
    np.testing.assert_array_equal(masked.frequency, full.frequency[keep])
    np.testing.assert_array_equal(masked.power, full.power[keep])
    np.testing.assert_array_equal(
        gap.dw_data(np.where(observed, x, np.nan), observed).power,
        masked.power,
    )
    assert (masked.counts == 2).all()
    assert (b.prepare(x, "wdm").counts == 1).all()


def test_masked_wdm_rows_have_correct_mean_and_units():
    observed = study.masks()["single"]
    rho, _, masked = gap.kernel_expectations(observed)
    retained = np.argwhere(gap.wdm_keep(rho))
    for ti, fi in retained[[0, len(retained) // 2, -1]]:
        row = (
            b.wdm_row(int(ti + 1), int(fi + 1))
            * observed
            / np.sqrt(rho[ti, fi])
        )
        np.testing.assert_allclose(row @ row, 1, atol=3e-12)
        np.testing.assert_allclose(
            2 * b.DT * ((row**2) @ b.variance(np.arange(b.N) / b.N)),
            2 * b.DT * masked[ti, fi] / rho[ti, fi],
            rtol=3e-12,
        )


def test_region_domain_includes_wdm_excluded_observed_neighbors():
    u, f, _ = study.e.grid()
    reg = audit.regions(u, study.masks()["single"], study.RADIUS)
    np.testing.assert_array_equal(sum(reg.values()), np.ones(len(u), int))
    # MP can retain centres in these scored observed strips even though the
    # WDM whole-column gate excludes them.
    lo, hi = audit.endpoints(study.masks()["single"])[0] / b.N
    assert reg["left_observed"][np.argmin(abs(u - (lo - 0.015)))]
    assert reg["right_observed"][np.argmin(abs(u - (hi + 0.015)))]


def test_band_average_precedes_posterior_quantiles():
    f = np.array([0.05, 0.15, 0.3, 0.45])
    weights = audit.band_weights(f, (0.05, 0.45))
    # Opposite-frequency draws: averaging quantile endpoints overstates width.
    draws = np.array([[1.0, 100.0, 1.0, 100.0], [100.0, 1.0, 100.0, 1.0]])
    actual = np.quantile(draws @ weights, [0.05, 0.5, 0.95])
    endpoints = np.quantile(draws, [0.05, 0.5, 0.95], axis=0) @ weights
    assert actual[2] - actual[0] < endpoints[2] - endpoints[0]


def test_annual_time_centering_is_fixed_when_observations_are_removed():
    annual = np.arange(1, 2047) / 2048
    model = ANOVALogPSpline(
        SplineBasis.from_grid(np.linspace(0.0001, 0.1, 65), 12),
        SplineBasis.from_grid(annual, 3),
        sigma_eta_prior=0.5,
    )
    assert model.time_basis.shape == (2046, 6)
    assert model.frequency.basis.shape[1] == 16
    np.testing.assert_allclose(model.time_basis.mean(axis=0), 0, atol=3e-13)
    keep = (annual < 0.4) | (annual > 0.7)
    evaluated, _ = model.design(annual[keep], np.array([0.001, 0.01]))
    np.testing.assert_allclose(evaluated, model.time_basis[keep], atol=3e-13)
    # Removing observations must not redefine the scientific time average.
    assert np.max(abs(evaluated.mean(axis=0))) > 0.01
