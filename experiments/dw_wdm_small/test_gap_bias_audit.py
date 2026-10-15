"""Scientific identities and pairing contracts for the first-stage bias audit."""

import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import gap_bias_audit as audit
from experiments.dw_wdm_small import taper_coloured_audit as coloured
from scipy import sparse


def test_fragmented_mask_has_equal_missing_duration_and_frozen_endpoints():
    masks = audit.masks()
    assert (~masks["single"]).sum() == (~masks["fragmented"]).sum() == 819
    intervals = audit.endpoints(masks["fragmented"])
    np.testing.assert_array_equal(
        np.diff(intervals, axis=1).ravel(), [205, 205, 205, 204]
    )
    np.testing.assert_array_equal(
        intervals, [[1536, 1741], [3175, 3380], [4813, 5018], [6452, 6656]]
    )
    assert np.all(intervals[:-1, 1] < intervals[1:, 0])


def test_boundary_gap_endpoints_are_half_open():
    np.testing.assert_array_equal(
        audit.endpoints(np.array([False, False, True, False])),
        [[0, 2], [3, 4]],
    )


def test_nearest_edge_regions_never_duplicate_observed_sites():
    observed = audit.masks()["fragmented"]
    u = np.arange(bench.N) / bench.N
    regions = audit.regions(u, observed, 0.3)
    assigned = sum(values.astype(int) for values in regions.values())
    np.testing.assert_array_equal(assigned, np.ones(bench.N))
    np.testing.assert_array_equal(regions["gap_interior"], ~observed)
    assert not np.any(regions["left_observed"] & regions["right_observed"])


def test_expected_MA_power_matches_explicit_covariance_projection():
    rng = np.random.default_rng(2026)
    b = rng.uniform(-0.8, 0.8, size=bench.N)
    times = np.array([32, 4000, 8159]) / bench.N
    frequencies = np.array([3, 16, 30]) / 65
    h = np.ones(65)
    actual = coloured.expected_dw(h, times, frequencies, b)
    for i, (u, f) in enumerate(zip(times, frequencies, strict=True)):
        centre = round(u * bench.N)
        bb = b[centre - 32 : centre + 33]
        covariance = (
            np.diag(1 + bb**2) + np.diag(bb[1:], 1) + np.diag(bb[1:], -1)
        )
        row = np.exp(-2j * np.pi * f * np.arange(65)) / np.sqrt(65)
        expected = 2 * bench.DT * np.real(row @ covariance @ row.conj())
        np.testing.assert_allclose(actual[i], expected, atol=1e-12)


def test_dependence_includes_nonoverlapping_covariance_connected_rows():
    rows = sparse.csr_matrix(np.eye(3))
    result = audit.gaussian_dependence(
        rows, np.ones(3), np.array([0.0, 0.3, 0.4])
    )
    np.testing.assert_allclose(result["max_power_correlation"], 0.4**2)
    assert result["unordered_potential_pairs"] == 2


def test_record_resampling_preserves_method_and_mask_pairing():
    x = np.arange(10.0)
    records = np.column_stack((x, x + 3, 2 * x))
    indices = np.random.default_rng(1).integers(0, 10, (50, 10))
    samples = audit.resample_means(records, indices)
    np.testing.assert_allclose(samples[:, 1] - samples[:, 0], 3)
    np.testing.assert_allclose(samples[:, 2], 2 * samples[:, 0])


def test_oracle_reproduces_truth_in_the_frozen_basis():
    master = bench.master_model()
    u, f = np.linspace(0, 1, 45), np.linspace(0.025, 0.475, 41)
    weights = np.random.default_rng(3).normal(size=(12, 6))
    truth = (
        master.time.design_at(u) @ weights @ master.frequency.design_at(f).T
    )
    result, _ = audit.oracle_basis(u, f, truth)
    assert result["max_absolute_log_error"] < 1e-12


def test_small_ensemble_coverage_cannot_establish_calibration():
    lo, hi = audit.wilson(9, 10)
    assert lo < 0.85 and hi > 0.95


def test_band_quadrature_integrates_affine_spectra_with_shared_endpoints():
    f = np.array([0.05, 0.09, 0.15, 0.22, 0.30, 0.36, 0.45])
    for lo, hi in audit.BANDS.values():
        weights = audit.band_weights(f, (lo, hi))
        np.testing.assert_allclose(weights.sum(), hi - lo)
        np.testing.assert_allclose(
            weights @ (2 + 3 * f), 2 * (hi - lo) + 1.5 * (hi**2 - lo**2)
        )


def test_sampler_effort_can_change_without_changing_the_target():
    a = {
        "n_samples": 1000,
        "target_accept_prob": 0.95,
        "roughness_scale": 10.0,
    }
    b = {
        "n_samples": 1500,
        "target_accept_prob": 0.99,
        "roughness_scale": 10.0,
    }
    assert audit.config_target(a) == audit.config_target(b)
    b["roughness_scale"] = 5.0
    assert audit.config_target(a) != audit.config_target(b)
