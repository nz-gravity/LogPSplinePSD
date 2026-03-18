"""
Tests for src/log_psplines/psplines/penalty.py

Covers:
  - _build_knot_vector (phantom)
  - _build_full_knot_vector (clamped, private utility)
  - build_general_difference_matrix  (including Li & Cao 2022 worked example)
  - build_gps_penalty
  - build_bspline_basis_scipy
  - Integration via init_basis_and_penalty and LogPSplines
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline

from log_psplines.psplines.penalty import (
    _build_full_knot_vector,
    _build_knot_vector,
    build_bspline_basis_scipy,
    build_general_difference_matrix,
    build_gps_penalty,
)

# ---------------------------------------------------------------------------
# Fixtures / shared constants
# ---------------------------------------------------------------------------

# Li & Cao (2022) Section 2.2 worked example
# cubic (degree=3, order d=4), domain [0,4], interior knots s1=1, s2=3
# Stored knots (including boundaries): [0, 1, 3, 4]
# Full clamped vector: [0,0,0,0, 1, 3, 4,4,4,4]  (p=6 basis functions)
_PAPER_DEGREE = 3
_PAPER_KNOTS = np.array([0.0, 1.0, 3.0, 4.0])  # stored (includes boundaries)
_PAPER_T_FULL = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0])


# ---------------------------------------------------------------------------
# _build_knot_vector (phantom)
# ---------------------------------------------------------------------------


class TestBuildKnotVector:
    def test_same_length_as_clamped(self):
        for degree in [1, 2, 3]:
            knots = np.linspace(0, 1, 6)
            t_clamped = _build_full_knot_vector(knots, degree)
            t_phantom = _build_knot_vector(knots, degree)
            assert len(t_phantom) == len(t_clamped), (
                f"degree={degree}: phantom len={len(t_phantom)}, "
                f"clamped len={len(t_clamped)}"
            )

    def test_phantoms_strictly_outside_domain(self):
        knots = np.linspace(0, 1, 5)
        degree = 3
        t = _build_knot_vector(knots, degree)
        # First `degree` entries must be < 0
        assert np.all(t[:degree] < 0.0)
        # Last `degree` entries must be > 1
        assert np.all(t[-degree:] > 1.0)

    def test_strictly_increasing(self):
        for n in [4, 6, 10]:
            knots = np.linspace(0, 1, n)
            t = _build_knot_vector(knots, degree=3)
            assert np.all(
                np.diff(t) > 0.0
            ), "Knot vector not strictly increasing"

    def test_boundary_values_appear_once(self):
        """0.0 and 1.0 should appear exactly once (no clamping)."""
        knots = np.linspace(0, 1, 8)
        t = _build_knot_vector(knots, degree=3)
        assert np.sum(t == 0.0) == 1
        assert np.sum(t == 1.0) == 1

    def test_spacing_at_left_boundary(self):
        """Left phantoms should be spaced by h_left = knots[1] - knots[0]."""
        knots = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        degree = 2
        t = _build_knot_vector(knots, degree)
        # t[0] = 0.0 - 2*0.2 = -0.4,  t[1] = 0.0 - 1*0.2 = -0.2
        np.testing.assert_allclose(t[0], -0.4, atol=1e-12)
        np.testing.assert_allclose(t[1], -0.2, atol=1e-12)

    def test_simple_example(self):
        """Three-knot example: phantom vector matches expected."""
        t = _build_knot_vector(np.array([0.0, 0.5, 1.0]), degree=2)
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(t, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# _build_full_knot_vector (private clamped utility)
# ---------------------------------------------------------------------------


class TestBuildFullKnotVector:
    def test_paper_example(self):
        t = _build_full_knot_vector(_PAPER_KNOTS, _PAPER_DEGREE)
        np.testing.assert_array_equal(t, _PAPER_T_FULL)

    def test_length(self):
        for degree in [1, 2, 3]:
            knots = np.linspace(0, 1, 7)  # 5 interior knots
            t = _build_full_knot_vector(knots, degree)
            n_basis = len(knots) + degree - 1
            assert len(t) == n_basis + degree + 1

    def test_boundary_multiplicity(self):
        knots = np.array([0.0, 0.3, 0.6, 1.0])
        degree = 3
        t = _build_full_knot_vector(knots, degree)
        assert np.all(t[: degree + 1] == 0.0)
        assert np.all(t[-(degree + 1) :] == 1.0)

    def test_interior_knots_preserved(self):
        interior = np.array([0.25, 0.5, 0.75])
        knots = np.concatenate([[0.0], interior, [1.0]])
        t = _build_full_knot_vector(knots, degree=2)
        for si in interior:
            assert np.sum(t == si) == 1


# ---------------------------------------------------------------------------
# build_general_difference_matrix
# ---------------------------------------------------------------------------


class TestBuildGeneralDifferenceMatrix:
    def test_paper_W1_weights(self):
        """
        Verify W1 from Li & Cao Section 2.2, page 9.
        t = [0,0,0,0,1,3,4,4,4,4], d=4, m=1, lag=3, n_rows=5
        Expected diagonal: [1/3, 1, 4/3, 1, 1/3]
        D1 = W1^{-1} @ Delta: the j-th row has -1/w[j] at column j and
        +1/w[j] at column j+1.  Check the +1/w[j] entries (column j+1).
        """
        D1 = build_general_difference_matrix(_PAPER_T_FULL, _PAPER_DEGREE, 1)
        expected_w1 = np.array([1 / 3, 1.0, 4 / 3, 1.0, 1 / 3])
        for j in range(5):
            np.testing.assert_allclose(
                D1[j, j + 1],
                1.0 / expected_w1[j],
                rtol=1e-10,
                err_msg=f"W1 mismatch at j={j}",
            )

    def test_paper_W2_weights(self):
        """
        Verify W2 from the paper, page 10.
        lag = 2, n_rows = 4
        j=0: (t[4]-t[2])/2 = (1-0)/2 = 1/2
        j=1: (t[5]-t[3])/2 = (3-0)/2 = 3/2
        j=2: (t[6]-t[4])/2 = (4-1)/2 = 3/2
        j=3: (t[7]-t[5])/2 = (4-3)/2 = 1/2
        """
        t = _PAPER_T_FULL
        d = 4
        paper_w2 = np.array([0.5, 1.5, 1.5, 0.5])
        computed_w2 = np.array([(t[j + d] - t[j + 2]) / 2 for j in range(4)])
        np.testing.assert_allclose(computed_w2, paper_w2, rtol=1e-10)

    def test_shape(self):
        """D_m has shape (n_basis - m, n_basis) for m = 1..degree."""
        degree = 3
        knots = np.linspace(0, 1, 9)  # 7 interior knots
        t = _build_full_knot_vector(knots, degree)
        n_basis = len(t) - degree - 1
        for m in range(1, degree + 1):
            D = build_general_difference_matrix(t, degree, diff_order=m)
            assert D.shape == (
                n_basis - m,
                n_basis,
            ), f"m={m}: expected ({n_basis - m}, {n_basis}), got {D.shape}"

    def test_null_space_contains_constants(self):
        """
        A constant function lies in the null space of D_m for any m >= 1.
        Uses phantom knot vector (same as build_bspline_basis_scipy) for consistency.
        """
        degree = 3
        knots = np.linspace(0, 1, 9)
        t = _build_knot_vector(knots, degree)
        grid = np.linspace(0.0, 1.0, 500)
        B = build_bspline_basis_scipy(knots, degree, grid)
        w, *_ = np.linalg.lstsq(B, np.ones(500), rcond=None)
        for m in [1, 2]:
            D = build_general_difference_matrix(t, degree, diff_order=m)
            np.testing.assert_allclose(
                D @ w,
                0.0,
                atol=1e-7,
                err_msg=f"Constant not in null space of D_{m}",
            )

    def test_null_space_contains_linear(self):
        """Linear function lies in null space of D_m for m >= 2.
        Uses clamped knot vector (same as build_bspline_basis_scipy) for consistency.
        """
        degree = 3
        knots = np.linspace(0, 1, 9)
        t = _build_full_knot_vector(knots, degree)
        grid = np.linspace(0.0, 1.0, 500)
        B = build_bspline_basis_scipy(knots, degree, grid)
        w, *_ = np.linalg.lstsq(B, grid, rcond=None)  # fit y = x
        D2 = build_general_difference_matrix(t, degree, diff_order=2)
        np.testing.assert_allclose(
            D2 @ w,
            0.0,
            atol=1e-7,
            err_msg="Linear function not in null space of D_2",
        )

    def test_equidistant_knots_proportional_to_standard_diff(self):
        """
        For equidistant knots, D_m must be proportional to Delta_m.
        Uses phantom knot vector where all knot spacings are truly uniform.
        """
        degree = 3
        n_interior = 7
        knots = np.linspace(0, 1, n_interior + 2)
        # Phantom knot vector: all spacings are equal (truly equidistant)
        t = _build_knot_vector(knots, degree)
        n_basis = len(t) - degree - 1
        D2 = build_general_difference_matrix(t, degree, diff_order=2)
        # Standard second-difference matrix
        Delta2 = np.diff(np.eye(n_basis), n=2, axis=0)
        mask = np.abs(Delta2) > 1e-12
        ratios = D2[mask] / Delta2[mask]
        np.testing.assert_allclose(
            ratios,
            ratios[0],
            rtol=1e-8,
            err_msg="GPS D_2 not proportional to standard Delta_2 for equidistant knots",
        )

    def test_raises_diff_order_zero(self):
        with pytest.raises(ValueError, match="diff_order must be >= 1"):
            build_general_difference_matrix(_PAPER_T_FULL, _PAPER_DEGREE, 0)

    def test_raises_diff_order_too_large(self):
        # diff_order must be < degree + 1 = 4
        with pytest.raises(ValueError, match="diff_order"):
            build_general_difference_matrix(_PAPER_T_FULL, _PAPER_DEGREE, 4)


# ---------------------------------------------------------------------------
# build_bspline_basis_scipy
# ---------------------------------------------------------------------------


class TestBuildBsplineBasisScipy:
    def test_shape(self):
        knots = np.linspace(0, 1, 8)
        degree = 3
        n_basis = len(knots) + degree - 1
        grid = np.linspace(0, 1, 100)
        B = build_bspline_basis_scipy(knots, degree, grid)
        assert B.shape == (100, n_basis)

    def test_partition_of_unity(self):
        """Each row sums to 1 (standard B-spline property)."""
        knots = np.linspace(0, 1, 10)
        grid = np.linspace(0.01, 0.99, 200)
        B = build_bspline_basis_scipy(knots, 3, grid)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegative(self):
        knots = np.linspace(0, 1, 8)
        grid = np.linspace(0, 1, 100)
        B = build_bspline_basis_scipy(knots, 3, grid)
        assert np.all(B >= -1e-14), f"Min value: {B.min()}"

    def test_no_nan_in_output(self):
        knots = np.linspace(0, 1, 8)
        grid = np.linspace(0, 1, 100)
        B = build_bspline_basis_scipy(knots, 3, grid)
        assert not np.any(np.isnan(B))

    def test_endpoint_basis_pinned_to_one(self):
        """
        With clamped knots (degree+1 boundary multiplicity), B_0(0) = 1 and
        B_{n-1}(1) = 1 — the standard B-spline interpolation property at
        endpoints.
        """
        knots = np.linspace(0, 1, 10)
        degree = 3
        B = build_bspline_basis_scipy(knots, degree, np.array([0.0, 1.0]))

        assert B[0, 0] == pytest.approx(
            1.0, abs=1e-10
        ), f"Clamped: B_0(0)={B[0, 0]:.6f} should be 1"
        assert B[1, -1] == pytest.approx(
            1.0, abs=1e-10
        ), f"Clamped: B_{{n-1}}(1)={B[1, -1]:.6f} should be 1"


# ---------------------------------------------------------------------------
# build_gps_penalty
# ---------------------------------------------------------------------------


class TestBuildGpsPenalty:
    def test_shape(self):
        for degree in [1, 2, 3]:
            knots = np.linspace(0, 1, 7)
            n_basis = len(knots) + degree - 1
            P = build_gps_penalty(knots, degree, diff_order=min(2, degree))
            assert P.shape == (n_basis, n_basis)

    def test_symmetric(self):
        knots = np.linspace(0, 1, 8)
        P = build_gps_penalty(knots, 3, 2)
        np.testing.assert_allclose(P, P.T, atol=1e-12)

    def test_positive_definite(self):
        """After ridge, all eigenvalues > 0."""
        knots = np.linspace(0, 1, 8)
        P = build_gps_penalty(knots, 3, 2, epsilon=1e-6)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0), f"Min eigenvalue: {eigvals.min():.3e}"

    def test_null_space_size_without_ridge(self):
        """
        D_m^T D_m has exactly `diff_order` near-zero eigenvalues before ridge.
        """
        diff_order = 2
        knots = np.linspace(0, 1, 10)
        P_no_ridge = build_gps_penalty(knots, 3, diff_order, epsilon=0.0)
        eigvals = np.sort(np.linalg.eigvalsh(P_no_ridge))
        assert np.all(
            eigvals[:diff_order] < 1e-6
        ), f"Expected {diff_order} near-zero eigenvalues, got {eigvals[:diff_order]}"

    def test_diff_order_zero_returns_ridge(self):
        knots = np.linspace(0, 1, 6)
        degree = 2
        n_basis = len(knots) + degree - 1
        epsilon = 1e-4
        P = build_gps_penalty(knots, degree, 0, epsilon=epsilon)
        np.testing.assert_allclose(P, epsilon * np.eye(n_basis), atol=1e-14)

    def test_non_uniform_log_knots(self):
        """Log-spaced knots (as used by LVK knot allocator) must work."""
        raw = np.logspace(-3, 0, 10)  # 0.001 .. 1
        knots = np.unique(np.concatenate([[0.0], raw / raw.max(), [1.0]]))
        knots[0] = 0.0
        knots[-1] = 1.0
        P = build_gps_penalty(knots, 3, 2)
        assert P.shape[0] == P.shape[1]
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0)


# ---------------------------------------------------------------------------
# Integration tests: init_basis_and_penalty + LogPSplines
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_init_basis_and_penalty_shapes(self):
        from log_psplines.psplines.initialisation import init_basis_and_penalty

        knots = np.linspace(0, 1, 8)  # 6 interior knots
        degree = 3
        diff_order = 2
        n_grid = 64
        n_basis = len(knots) + degree - 1  # should be 10
        basis, penalty = init_basis_and_penalty(
            knots, degree, n_grid, diff_order
        )
        assert basis.shape == (
            n_grid,
            n_basis,
        ), f"basis: expected ({n_grid}, {n_basis}), got {basis.shape}"
        assert penalty.shape == (
            n_basis,
            n_basis,
        ), f"penalty: expected ({n_basis}, {n_basis}), got {penalty.shape}"

    def test_init_basis_and_penalty_pd(self):
        from log_psplines.psplines.initialisation import init_basis_and_penalty

        knots = np.linspace(0, 1, 8)
        _, penalty = init_basis_and_penalty(knots, 3, 64, 2)
        eigvals = np.linalg.eigvalsh(np.asarray(penalty))
        assert np.all(eigvals > 0)

    def test_init_basis_partition_of_unity(self):
        from log_psplines.psplines.initialisation import init_basis_and_penalty

        knots = np.linspace(0, 1, 8)
        grid = np.linspace(0.01, 0.99, 100)
        basis, _ = init_basis_and_penalty(
            knots, 3, len(grid), 2, grid_points=grid
        )
        row_sums = np.asarray(basis).sum(axis=1)
        # atol=1e-5 accommodates float32 precision from JAX conversion
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_log_psplines_from_periodogram_smoke(self):
        """End-to-end: LogPSplines.from_periodogram must not raise."""
        from log_psplines.datatypes import Periodogram
        from log_psplines.psplines.psplines import LogPSplines

        freqs = np.linspace(1e-3, 0.5, 128)
        power = 1.0 / (freqs**2 + 0.01)
        pdgrm = Periodogram(freqs=freqs, power=power)
        model = LogPSplines.from_periodogram(
            pdgrm, n_knots=10, degree=3, diffMatrixOrder=2
        )
        assert model.basis.shape == (128, model.n_basis)
        assert model.penalty_matrix.shape == (model.n_basis, model.n_basis)
        assert np.all(np.isfinite(np.asarray(model.basis)))
        assert np.all(np.isfinite(np.asarray(model.penalty_matrix)))

    def test_log_psplines_log_knots(self):
        """Log-spaced knots (LVK-style) must work end-to-end."""
        from log_psplines.datatypes import Periodogram
        from log_psplines.psplines.psplines import LogPSplines

        freqs = np.logspace(-4, -1, 256)
        power = 1e-10 / freqs**2
        pdgrm = Periodogram(freqs=freqs, power=power)
        model = LogPSplines.from_periodogram(
            pdgrm,
            n_knots=12,
            degree=3,
            diffMatrixOrder=2,
            knot_kwargs=dict(method="log"),
        )
        assert np.all(np.isfinite(np.asarray(model.basis)))
        assert np.all(np.isfinite(np.asarray(model.penalty_matrix)))

    def test_n_basis_property_consistent(self):
        """LogPSplines.n_basis must equal basis.shape[1] and penalty.shape[0]."""
        from log_psplines.datatypes import Periodogram
        from log_psplines.psplines.psplines import LogPSplines

        freqs = np.linspace(0.001, 0.5, 200)
        power = np.ones_like(freqs)
        pdgrm = Periodogram(freqs=freqs, power=power)
        model = LogPSplines.from_periodogram(
            pdgrm, n_knots=8, degree=3, diffMatrixOrder=2
        )
        assert model.n_basis == model.basis.shape[1]
        assert model.n_basis == model.penalty_matrix.shape[0]
        assert model.n_basis == model.penalty_matrix.shape[1]
