import numpy as np

from log_psplines import moving_periodogram
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
