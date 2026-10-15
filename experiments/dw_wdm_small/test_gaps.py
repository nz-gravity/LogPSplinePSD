"""Missing samples cannot enter retained Tang ordinates."""

import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small.gap_smoke import (
    dw_data,
    dw_keep,
    masks,
    wdm_data,
    wdm_keep,
)


def test_no_gap_preserves_every_original_selected_site_and_power():
    x = bench.simulate(6101)
    data = dw_data(x, np.ones(bench.N, dtype=bool))
    baseline = bench.prepare(x, "dw")
    for field in ("power", "counts", "frequency", "time"):
        np.testing.assert_array_equal(
            getattr(data, field), getattr(baseline, field)
        )


def test_gaps_preserve_thinning_and_make_missing_values_irrelevant():
    x = bench.simulate(6101)
    full = bench.prepare(x, "dw")
    for observed in masks().values():
        keep = dw_keep(observed)
        data = dw_data(x, observed)
        np.testing.assert_array_equal(data.power, full.power[keep])
        np.testing.assert_array_equal(data.time, full.time[keep])
        np.testing.assert_array_equal(data.frequency, full.frequency[keep])
        for replacement in (np.nan, 1e50, -1e50):
            modified = x.copy()
            modified[~observed] = replacement
            np.testing.assert_array_equal(
                dw_data(modified, observed).power, data.power
            )


def test_every_retained_window_has_only_observed_samples():
    x = bench.simulate(6101)
    for observed in masks().values():
        data = dw_data(x, observed)
        centres = np.rint(data.time * bench.N).astype(int)
        assert all(
            observed[c - bench.M : c + bench.M + 1].all() for c in centres
        )
        assert np.all(data.counts == 2)


def test_whole_cycle_control_is_a_conservative_subset():
    for observed in masks().values():
        individual, whole = (
            dw_keep(observed),
            dw_keep(observed, whole_cycles=True),
        )
        assert np.all(individual[whole])
        assert not np.any(
            whole.reshape(-1, bench.M).any(axis=1)
            != whole.reshape(-1, bench.M).all(axis=1)
        )


def test_wdm_column_gate_removes_partial_columns_and_preserves_normalization():
    x = bench.simulate(6101)
    rho = np.load(bench.ROOT / "gaps/contiguous_mask.npz")["rho"]
    observed = masks()["contiguous"]
    keep = wdm_keep(rho)
    assert np.count_nonzero(keep) == 6479
    assert np.count_nonzero(keep.all(axis=1)) == 209
    assert np.array_equal(keep.any(axis=1), keep.all(axis=1))
    assert np.all(rho[keep] >= 0.9999)
    data = wdm_data(x, observed, rho)
    native = bench.prepare(np.where(observed, x, 0.0), "wdm")
    np.testing.assert_array_equal(data.counts > 0, keep)
    np.testing.assert_allclose(
        data.power[keep], native.power[keep] / rho[keep]
    )
    assert np.all(data.power[~keep] == 0)
    assert not wdm_keep(np.full_like(rho, 0.5)).any()
    for replacement in (np.nan, 1e50, -1e50):
        hidden = x.copy()
        hidden[~observed] = replacement
        np.testing.assert_array_equal(
            wdm_data(hidden, observed, rho).power, data.power
        )
