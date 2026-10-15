"""Independent normalization and missing-sample contracts for tapered windows."""

import numpy as np
import pytest
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import gap_smoke as gap
from experiments.dw_wdm_small import taper_smoke as taper
from scipy.signal import periodogram


@pytest.mark.parametrize("name", ["boxcar", "hann"])
def test_normalization_matches_scipy_without_detrending(name):
    x = np.random.default_rng(4001).normal(size=bench.N)
    observed = gap.masks()["contiguous"]
    data = taper.prepare(x, name, observed)
    h = taper.weights(name)
    centres = np.rint(data.time * bench.N).astype(int)
    windows = np.lib.stride_tricks.sliding_window_view(x, len(h))[
        centres - bench.M
    ]
    _, psd = periodogram(
        windows, fs=1 / bench.DT, window=h, detrend=False, scaling="density"
    )
    rungs = np.rint(data.frequency * len(h) * bench.DT).astype(int)
    np.testing.assert_allclose(
        data.power / data.counts,
        psd[np.arange(len(data.time)), rungs],
        atol=2e-13,
    )


@pytest.mark.parametrize("name", ["boxcar", "hann"])
def test_missing_samples_never_enter_retained_tapered_windows(name):
    x = bench.simulate(6101)
    for observed in gap.masks().values():
        original = taper.prepare(x, name, observed)
        x_nan = np.where(observed, x, np.nan)
        np.testing.assert_array_equal(
            original.power, taper.prepare(x_nan, name, observed).power
        )
        centres = np.rint(original.time * bench.N).astype(int)
        assert all(
            observed[c - bench.M : c + bench.M + 1].all() for c in centres
        )


def test_unknown_taper_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown taper"):
        taper.weights("unsupported")
