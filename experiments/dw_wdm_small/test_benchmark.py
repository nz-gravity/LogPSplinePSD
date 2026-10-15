"""Contracts that matter for a common-unit, common-prior comparison."""

import numpy as np
from experiments.dw_wdm_small.benchmark import (
    DT,
    NT,
    THIN,
    M,
    N,
    area_mean,
    master_model,
    on_grid,
    prepare,
    wdm_row,
)
from wdm_transform import TimeSeries as WDMTimeSeries

from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.preprocessing.moving_periodogram import (
    tang_moving_periodogram,
)


def test_area_weighted_coverage_casts_boolean_indicators_before_quadrature():
    u, f = np.linspace(0.05, 0.95, 5), np.linspace(0.025, 0.475, 7)
    np.testing.assert_allclose(
        area_mean(np.ones((5, 7), dtype=bool), u, f), 1.0
    )
    assert area_mean(np.zeros((5, 7), dtype=bool), u, f) == 0.0
    # An exact linear ramp has normalized integral one half.
    ramp = np.broadcast_to(np.linspace(0, 1, 5)[:, None], (5, 7))
    np.testing.assert_allclose(area_mean(ramp, u, f), 0.5)


def test_physical_units_equal_direct_normalized_transform_likelihoods():
    x = np.random.default_rng(34).normal(size=N)
    dw = prepare(x, "dw")
    raw = tang_moving_periodogram(x, m=M, thin=THIN)
    np.testing.assert_allclose(dw.time * N, np.rint(raw["u"] * N) - 1)
    eta = np.log(2 * DT)
    expected_dw = -np.sum(eta + 2 * np.pi * raw["mi"])
    np.testing.assert_allclose(
        power_whittle_log_likelihood(
            dw.power, dw.counts, np.full_like(dw.power, eta)
        ),
        expected_dw,
    )
    wdm = prepare(x, "wdm")
    w = np.asarray(WDMTimeSeries(x, dt=DT).to_wdm(nt=NT).coeffs)[0, 1:-1, 1:-1]
    expected_wdm = -0.5 * np.sum(eta + w**2 / N)
    np.testing.assert_allclose(
        power_whittle_log_likelihood(
            wdm.power, wdm.counts, np.full_like(wdm.power, eta)
        ),
        expected_wdm,
    )


def test_inverse_basis_is_the_forward_analysis_row_with_unit_white_variance():
    x = np.random.default_rng(35).normal(size=N)
    w = np.asarray(WDMTimeSeries(x, dt=DT).to_wdm(nt=NT).coeffs)[0]
    for t, f in [(1, 1), (127, 16), (254, 31)]:
        row = wdm_row(t, f)
        np.testing.assert_allclose(row @ row, 1, atol=1e-12)
        np.testing.assert_allclose(row @ x, w[t, f] / np.sqrt(N), atol=1e-12)


def test_native_wdm_grid_retains_identical_spline_and_penalty():
    master = master_model()
    data = prepare(np.random.default_rng(36).normal(size=N), "wdm")
    native = on_grid(master, data.time, data.frequency)
    weights = np.random.default_rng(37).normal(size=master.weights.shape)
    np.testing.assert_allclose(
        native(weights),
        master.time.design_at(data.time)
        @ weights
        @ master.frequency.design_at(data.frequency).T,
        atol=1e-12,
    )
    for axis in ("time", "frequency"):
        for attribute in ("knots", "penalty"):
            np.testing.assert_array_equal(
                getattr(getattr(master, axis), attribute),
                getattr(getattr(native, axis), attribute),
            )
