"""Scope, normalization and matched-site contracts of the extension."""

import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import taper_bias_extension as study


def test_hann_and_control_use_exactly_the_same_paired_sites():
    x = study.simulate("LS2", 8201)
    for mask in study.MASKS:
        a = study.observations(x, "boxcar_sparse", mask, "LS2")
        b = study.observations(x, "hann_sparse", mask, "LS2")
        for field in ("time", "frequency", "counts"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        assert np.all(a.counts == 2)
        assert np.unique(a.frequency).size == 10


def test_coloured_band_keeps_every_sparse_rung_without_resetting_time():
    x = study.simulate("LS2", 8201)
    for mask in study.MASKS:
        old = study.observations(x, "hann_sparse", mask, "broadband")
        new = study.observations(x, "hann_sparse", mask, "LS2")
        for field in ("time", "frequency", "counts", "power"):
            np.testing.assert_array_equal(
                getattr(old, field), getattr(new, field)
            )


def test_hidden_coloured_samples_cannot_enter_any_variant():
    x = study.simulate("LS2", 8201)
    observed = study.saved_mask("single")["observed"]
    for variant in study.VARIANTS[1:]:
        a = study.observations(x, variant, "single", "LS2")
        for value in (np.nan, 1e50, -1e50):
            b = study.observations(
                np.where(observed, x, value), variant, "single", "LS2"
            )
            np.testing.assert_array_equal(a.power, b.power)


def test_coloured_ensemble_never_reuses_broadband_posteriors():
    for variant in study.VARIANTS[1:]:
        for mask in study.MASKS:
            assert study.existing("LS2", mask, variant, 8201) is None


def test_coloured_basis_can_represent_the_primary_target():
    u, f = np.linspace(0.05, 0.95, 241), np.linspace(0.05, 0.45, 161)
    m = study.model("LS2")
    bt, bf = m.time.design_at(u), m.frequency.design_at(f)
    truth = np.log(study.truth("LS2", u, f))
    weights = np.linalg.lstsq(bt, truth, rcond=None)[0] @ np.linalg.pinv(bf.T)
    error = bt @ weights @ bf.T - truth
    assert m.weights.shape == (36, 10)
    assert np.sqrt(np.mean(error**2)) < 0.002
    assert abs(error).max() < 0.012


def test_LS2_truth_agrees_with_its_zero_evolution_white_limit_units():
    from log_psplines.example_datasets.ls2_data import LS2Data

    simulator = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=8200)
    u, f = np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.25, 0.4])
    np.testing.assert_allclose(
        study.truth("LS2", u, f),
        2 * bench.DT * simulator.get_true_psd(time_grid=u, freq_grid=f),
    )


def test_selected_broadband_repair_preserves_target_and_passing_screen():
    from experiments.dw_wdm_small import gap_bias_audit as audit

    for variant in ("boxcar_sparse", "hann_sparse"):
        path = study.existing("broadband", "single", variant, 6101)
        row = audit.read(path / "metrics.json")
        assert audit.screen(row["diagnostics"])
        assert row["raw_sha256"] == bench.digest(
            study.simulate("broadband", 6101)
        )
