import numpy as np
from numpyro.infer.util import log_density

from log_psplines import (
    LogPSpline,
    PowerSplineConfig,
    SplineBasis,
    moving_periodogram,
    scattered_moving_periodogram,
)
from log_psplines.inference.power import prepare_scattered_power_model
from log_psplines.preprocessing.moving_periodogram import (
    bin_tang_ordinates,
    tang_moving_periodogram,
)


def _definition_one(x: np.ndarray, t: int, m: int) -> tuple[complex, float]:
    j = 1 + ((t - 1) % m)
    lam = 2.0 * j / (2 * m + 1)
    nu = np.arange(2 * m + 1)
    window = x[t - m - 1 : t + m]
    coeff = np.sum(window * np.exp(-1j * np.pi * nu * lam))
    return coeff / np.sqrt(2.0 * np.pi * (2 * m + 1)), np.pi * lam


def test_tang_transform_matches_definition_one() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal(127)
    out = tang_moving_periodogram(x, m=4, thin=2)
    centres = np.rint(out["u"] * len(x)).astype(int)
    expected = np.asarray([_definition_one(x, int(t), 4)[0] for t in centres])
    expected_omega = np.asarray(
        [_definition_one(x, int(t), 4)[1] for t in centres]
    )
    np.testing.assert_allclose(out["coeff"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out["mi"], np.abs(expected) ** 2)
    np.testing.assert_allclose(out["omega"], expected_omega)
    assert centres.min() == 5


def test_binning_sums_power_and_counts() -> None:
    raw = tang_moving_periodogram(
        np.random.default_rng(5).standard_normal(191), m=6, thin=2
    )
    pooled = bin_tang_ordinates(raw, time_bin=2, freq_bin=2)
    assert pooled["summed_power"].size < raw["mi"].size
    assert np.all(pooled["counts"] > 0)
    assert np.isclose(pooled["summed_power"].sum(), 2.0 * raw["mi"].sum())
    assert np.isclose(pooled["counts"].sum(), 2.0 * raw["mi"].size)


def test_moving_periodogram_adapts_to_power_spectrum() -> None:
    data = moving_periodogram(
        np.random.default_rng(6).standard_normal(256),
        dt=0.1,
        m=8,
        thin=2,
    )
    assert data.power.ndim == 2
    assert data.power.shape == data.counts.shape
    assert data.power.shape == (data.time.size, data.frequency.size)
    assert np.all(data.power >= 0)
    assert np.all(data.counts == 2)
    assert np.all(np.diff(data.time) > 0)
    assert np.all(np.diff(data.frequency) > 0)


def test_scattered_adapter_preserves_raw_ordinates() -> None:
    x = np.random.default_rng(7).standard_normal(127)
    raw = tang_moving_periodogram(x, m=4, thin=2)
    data = scattered_moving_periodogram(x, dt=0.25, m=4, thin=2)
    np.testing.assert_array_equal(data.time, raw["u"])
    np.testing.assert_allclose(data.frequency, raw["omega"] / (0.5 * np.pi))
    np.testing.assert_allclose(data.power, 2.0 * raw["mi"])
    assert np.all(data.counts == 2.0)


def test_scattered_ordinates_enter_the_power_likelihood() -> None:
    x = np.random.default_rng(8).standard_normal(127)
    data = scattered_moving_periodogram(x, dt=0.25, m=4, thin=2)
    spline = LogPSpline(
        SplineBasis.from_grid(
            np.linspace(data.frequency.min(), data.frequency.max(), 7), 4
        ),
        time=SplineBasis.from_grid(
            np.linspace(data.time.min(), data.time.max(), 7), 4
        ),
    )
    model, initial_sites, _ = prepare_scattered_power_model(
        data, spline, PowerSplineConfig()
    )
    density, trace = log_density(model, (), {}, initial_sites)
    assert np.isfinite(density)
    assert np.isfinite(trace["log_likelihood"]["value"])
