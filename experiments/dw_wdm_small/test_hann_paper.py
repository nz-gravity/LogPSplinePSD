"""Focused regressions for frozen headline observation/reconstruction contracts."""

import numpy as np
import pytest
import xarray as xr
from experiments.dw_wdm_small import hann_paper as study
from experiments.dw_wdm_small import taper_smoke as reference
from scipy.signal import periodogram


@pytest.mark.parametrize(
    "n,dt", [(1024, 0.1), (8192, 1.0), (32768, 0.1), (1001, 0.1)]
)
@pytest.mark.parametrize("rungs", [None, tuple(range(1, 33))])
def test_hann_matches_direct_power_and_original_phase(n, dt, rungs):
    x = np.random.default_rng(14).normal(size=n)
    data = study.hann_observations(x, np.ones(n, bool), dt, rungs=rungs)
    centers = np.rint(data.time * n).astype(int)
    expected_rungs = set(range(3, 31, 3)) if rungs is None else set(rungs)
    rungs = np.rint(data.frequency * 65 * dt).astype(int)
    assert set(rungs) == expected_rungs
    assert np.all((centers - 32) % 64 == rungs - 1)
    assert centers.min() >= 32 and centers.max() + 32 < n
    h = np.hanning(65)
    assert h @ h == 24 and h[0] == h[-1] == 0
    windows = np.lib.stride_tricks.sliding_window_view(x, 65)[centers - 32]
    _, powers = periodogram(
        windows, fs=1 / dt, window=h, detrend=False, scaling="density", axis=-1
    )
    np.testing.assert_allclose(
        data.power / 2,
        powers[np.arange(len(centers)), rungs],
        atol=2e-13,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(data.counts, 2)


def test_sparse_gap_rule_matches_saved_reference_before_bin_selection():
    n = 8192
    x = np.random.default_rng(23).normal(size=n)
    observed = np.ones(n, bool)
    observed[3686:4505] = False
    data = study.hann_observations(x, observed, 1.0)
    expected = reference.prepare(x, "hann", observed)
    for field in ("power", "counts", "time", "frequency"):
        np.testing.assert_allclose(
            getattr(data, field),
            getattr(expected, field),
            atol=2e-13,
            rtol=1e-12,
        )
    for hidden in (np.nan, 0.0, 1e50):
        changed = np.where(observed, x, hidden)
        np.testing.assert_array_equal(
            data.power, study.hann_observations(changed, observed, 1.0).power
        )


def test_linear_median_is_computed_after_exponentiation():
    model = study.endpoint_model(1024)
    w = np.zeros((1, 2, *model.weights.shape))
    w[:, 1] = 2
    posterior = xr.Dataset({"weights": (("chain", "draw", "t", "f"), w)})
    q = study.quantiles(posterior, model, np.array([0.3]), np.array([0.4]))
    np.testing.assert_allclose(q[1], (1 + np.exp(2)) / 2)
    assert not np.isclose(q[1], np.e)


def test_frozen_budget_and_extension_settings():
    cells = study.targets()
    assert len(cells) == 104
    assert (
        len(
            [
                c
                for c in cells
                if c["method"] == "MP" and c["dataset"] != "LISA"
            ]
        )
        == 50
    )
    for n in (1024, 32768):
        c = next(c for c in cells if c["dataset"] == "LS2" and c["n"] == n)
        assert study.config(c, 0).n_samples == (1000 if n == 1024 else 2000)
        assert study.config(c, 1).n_samples == 4000
