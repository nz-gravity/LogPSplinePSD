"""
General P-Spline (GPS) penalty and B-spline basis via scipy.

Implements the general difference penalty from:
    Li, Z. & Cao, J. (2022). General P-Splines for Non-Uniform B-Splines.
    arXiv:2201.06808v2, Section 2.2–2.3, equations (4a) and (5).

The key insight is that the standard difference matrix Delta_m is only correct
for equidistant knots. For non-uniform knots the general difference matrix D_m
accounts for the actual knot spacing via diagonal weight matrices W_m.

Glossary (paper notation vs code):
    d   = degree + 1  (paper calls this the "order")
    p   = n_basis     (number of B-spline basis functions)
    k   = number of interior knots
    K   = total knots in the full knot vector = k + 2*d
    m   = diff_order  (penalty order)

All basis functions use a phantom knot vector so that endpoint functions
have the same bell shape as interior ones, removing the clamped-boundary
bias that causes undercoverage in posterior credible intervals.

Public API
----------
build_gps_penalty(knots, degree, diff_order, epsilon)
build_bspline_basis_scipy(knots, degree, grid_points)
build_general_difference_matrix(t, degree, diff_order)
_build_knot_vector(knots, degree)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_gps_penalty",
    "build_bspline_basis_scipy",
    "build_general_difference_matrix",
    "_build_knot_vector",
]


# ---------------------------------------------------------------------------
# Knot vector constructors
# ---------------------------------------------------------------------------


def _build_knot_vector(knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Build a knot vector with phantom knots outside [0, 1].

    Instead of repeating the boundary values ``degree`` times (clamped), we
    place ``degree`` equally-spaced phantom knots below 0 and above 1 using
    the local knot spacing.  Each boundary point appears exactly once.

    This gives every basis function — including the endpoint ones — the same
    bell shape, removing the structural asymmetry that causes boundary bias
    in posterior coverage.

    The returned vector has the **same length** as a standard clamped vector
    would have, so ``n_basis`` is identical.

    Parameters
    ----------
    knots : np.ndarray
        Stored knot array **including** the boundary values 0.0 and 1.0,
        shape (n_knots,).
    degree : int
        B-spline polynomial degree.

    Returns
    -------
    np.ndarray
        Strictly increasing knot vector of length n_basis + degree + 1
        where n_basis = len(knots) + degree - 1.

    Examples
    --------
    >>> _build_knot_vector(np.array([0., 0.5, 1.]), degree=2)
    array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """
    h_left = float(knots[1] - knots[0])  # spacing at left boundary
    h_right = float(knots[-1] - knots[-2])  # spacing at right boundary
    # `degree` phantom knots below 0, ascending order
    left_phantoms = knots[0] - h_left * np.arange(degree, 0, -1)
    # `degree` phantom knots above 1, ascending order
    right_phantoms = knots[-1] + h_right * np.arange(1, degree + 1)
    # knots already contains exactly one 0.0 and one 1.0
    return np.concatenate([left_phantoms, knots, right_phantoms])


def _build_full_knot_vector(knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Build a clamped full knot vector with (degree+1)-fold boundary multiplicity.

    This is a **private utility used only in tests** to verify that both knot
    vector conventions produce the same ``n_basis``.  Production code always
    uses ``_build_knot_vector`` (phantom knots).

    Parameters
    ----------
    knots : np.ndarray
        Stored knot array including 0.0 and 1.0, shape (n_knots,).
    degree : int
        B-spline polynomial degree.

    Returns
    -------
    np.ndarray
        Full clamped knot vector of length n_basis + degree + 1.
    """
    interior = knots[1:-1]
    return np.concatenate(
        [
            np.repeat(knots[0], degree + 1),
            interior,
            np.repeat(knots[-1], degree + 1),
        ]
    )


# ---------------------------------------------------------------------------
# General difference matrix  (Li & Cao 2022, eq. 4a + 5)
# ---------------------------------------------------------------------------


def build_general_difference_matrix(
    t: np.ndarray,
    degree: int,
    diff_order: int,
) -> np.ndarray:
    """
    Compute the general difference matrix D_m for non-uniform B-splines.

    Implements the iterative construction from Li & Cao (2022), equations
    (4a) and (5)::

        beta_m = W_m^{-1} @ Delta @ beta_{m-1}
        D_m    = W_m^{-1} @ Delta @ D_{m-1},   D_0 = I_p

    where Delta is the (p-m) x (p-m+1) first-difference matrix and W_m is
    the diagonal weight matrix with::

        W_m[j, j] = (t[j + d] - t[j + m]) / (d - m)     (0-indexed j)

    Here ``d = degree + 1`` (the paper's "order") and ``p = len(t) - d``.

    For equidistant knots all W_m are proportional to identity, so D_m is
    proportional to the standard difference matrix Delta_m, recovering
    standard P-splines as a special case.

    Parameters
    ----------
    t : np.ndarray
        Full knot vector (phantom or clamped), shape (n_basis + degree + 1,).
    degree : int
        B-spline polynomial degree.
    diff_order : int
        Penalty order m.  Must satisfy 1 <= m <= degree.

    Returns
    -------
    np.ndarray
        Matrix D_m of shape (n_basis - diff_order, n_basis), dtype float64.

    Raises
    ------
    ValueError
        If diff_order < 1 or diff_order >= degree + 1.
        If any diagonal weight w[j] <= 0 (degenerate knot sequence).

    Notes
    -----
    Verified against the worked example in Section 2.2 of the paper:
    cubic splines (degree=3), domain [0,4], interior knots at 1 and 3,
    clamped boundaries.  t = [0,0,0,0, 1, 3, 4,4,4,4].  The computed W1
    diagonal is [1/3, 1, 4/3, 1, 1/3] which matches the paper exactly.
    """
    t = np.asarray(t, dtype=np.float64)
    d = degree + 1  # paper's order
    p = len(t) - d  # number of basis functions

    if diff_order < 1:
        raise ValueError(
            f"diff_order must be >= 1, got {diff_order}. "
            "For a ridge-only penalty use diff_order=0 separately."
        )
    if diff_order >= d:
        raise ValueError(
            f"diff_order={diff_order} must be < d=degree+1={d}. "
            f"Ensure degree ({degree}) >= diffMatrixOrder ({diff_order})."
        )

    D = np.eye(p, dtype=np.float64)  # D_0 = I_p

    for m in range(1, diff_order + 1):
        n_rows = p - m  # rows of D_m
        lag = d - m  # = order - step number

        # Diagonal weights: w[j] = (t[j+d] - t[j+m]) / lag,  j = 0,...,n_rows-1
        w = np.array(
            [(t[j + d] - t[j + m]) / lag for j in range(n_rows)],
            dtype=np.float64,
        )

        if np.any(w <= 0.0):
            bad_indices = np.where(w <= 0.0)[0].tolist()
            raise ValueError(
                f"Non-positive weight w[j] at step m={m} for indices "
                f"{bad_indices} (min={w.min():.3e}). "
                "This means two adjacent active knots are identical. "
                "Check for duplicate interior knots in the knot sequence."
            )

        W_inv = np.diag(1.0 / w)

        # First-difference matrix: shape (n_rows, n_rows + 1)
        Delta = np.zeros((n_rows, n_rows + 1), dtype=np.float64)
        for i in range(n_rows):
            Delta[i, i] = -1.0
            Delta[i, i + 1] = 1.0

        # D_m = W_m^{-1} @ Delta @ D_{m-1},  shape: (n_rows, p)
        D = W_inv @ Delta @ D

    return D  # shape (p - diff_order, p)


# ---------------------------------------------------------------------------
# Basis matrix (scipy)
# ---------------------------------------------------------------------------


def build_bspline_basis_scipy(
    knots: np.ndarray,
    degree: int,
    grid_points: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the B-spline basis matrix using scipy.interpolate.BSpline.

    Uses phantom knots so all basis functions — including endpoint ones —
    have the same bell shape.

    Parameters
    ----------
    knots : np.ndarray
        Stored knot array including 0.0 and 1.0.
    degree : int
        B-spline polynomial degree.
    grid_points : np.ndarray
        1-D array of evaluation points, nominally in [0, 1].
        Values outside [0, 1] are clipped to the boundary.

    Returns
    -------
    np.ndarray
        Basis matrix, shape (len(grid_points), n_basis), dtype float64.
        Entry [i, j] is B_j(grid_points[i]).  NaN from scipy extrapolation
        (produced outside the B-spline's support when extrapolate=False) is
        replaced with 0.0.

    Notes
    -----
    n_basis = len(knots) + degree - 1.
    """
    from scipy.interpolate import BSpline

    t = _build_knot_vector(np.asarray(knots, dtype=np.float64), degree)
    n_basis = len(t) - degree - 1

    x = np.clip(np.asarray(grid_points, dtype=np.float64), 0.0, 1.0)
    B = np.zeros((len(x), n_basis), dtype=np.float64)

    for i in range(n_basis):
        c = np.zeros(n_basis, dtype=np.float64)
        c[i] = 1.0
        spl = BSpline(t, c, degree, extrapolate=False)
        vals = spl(x)
        # extrapolate=False → NaN outside support; treat as zero
        B[:, i] = np.where(np.isnan(vals), 0.0, vals)

    return B


# ---------------------------------------------------------------------------
# GPS penalty matrix
# ---------------------------------------------------------------------------


def build_gps_penalty(
    knots: np.ndarray,
    degree: int,
    diff_order: int,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Build the General P-Spline penalty matrix P = D_m^T D_m.

    Uses phantom knots (same as ``build_bspline_basis_scipy``) so that
    basis and penalty are always mathematically consistent.

    The penalty is normalised so that max(|P|) = 1, matching the convention
    used by scikit-fda's L2Regularization.  A small ridge term (epsilon × I)
    is added for strict positive definiteness.

    Parameters
    ----------
    knots : np.ndarray
        Stored knot array including 0.0 and 1.0.
    degree : int
        B-spline polynomial degree.
    diff_order : int
        Penalty order m.  If 0, returns epsilon * I (pure ridge).
    epsilon : float, default 1e-6
        Ridge regularisation added to diagonal for numerical stability.

    Returns
    -------
    np.ndarray
        Symmetric positive (semi-)definite matrix of shape
        (n_basis, n_basis), dtype float64.
        After ridge regularisation it is strictly positive definite.
    """
    knots = np.asarray(knots, dtype=np.float64)
    n_basis = len(knots) + degree - 1

    if diff_order == 0:
        return epsilon * np.eye(n_basis, dtype=np.float64)

    t = _build_knot_vector(knots, degree)
    D = build_general_difference_matrix(t, degree, diff_order)
    P = D.T @ D  # shape (n_basis, n_basis), symmetric PSD

    # Normalise so max(|P|) = 1, matching the convention used by scikit-fda's
    # L2Regularization.penalty_matrix().  This keeps the penalty on a unit
    # scale so the φ (precision) prior remains well-calibrated regardless of
    # the number of knots or the knot spacing.
    max_val = float(np.max(np.abs(P)))
    if max_val > 0.0:
        P = P / max_val
    return P + epsilon * np.eye(n_basis, dtype=np.float64)
