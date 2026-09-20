"""Tensor-penalty algebra reused from wdm_psd.tv_pspline_psd.model.

Operate on supplied marginal penalties, without constructing a dense
Kronecker precision or choosing a smoothing hyperprior. WDM supplies
trace-normalized penalties before adding a ridge. The existing stationary
SplineBasis default is max-normalized and already ridged: it must not be
silently substituted when reproducing a WDM prior.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import interpolate


def whiten_penalty_pair(
    penalty_time: np.ndarray,
    penalty_freq: np.ndarray,
    *,
    null_tol: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Diagonalize marginal penalties and identify their joint null space.

    Inputs are symmetric positive semidefinite marginal penalties. As in the
    source, clip numerical negative eigenvalues and use a relative null cutoff.
    Keep the null space separate so it receives its own proper weak prior.
    """
    lam_t, U_t = np.linalg.eigh(penalty_time)
    lam_f, U_f = np.linalg.eigh(penalty_freq)
    lam_t = np.clip(lam_t, 0.0, None)
    lam_f = np.clip(lam_f, 0.0, None)
    null_t = lam_t <= null_tol * max(lam_t.max(), 1.0)
    null_f = lam_f <= null_tol * max(lam_f.max(), 1.0)
    return {
        "lam_time": lam_t,
        "lam_freq": lam_f,
        "U_time": U_t,
        "U_freq": U_f,
        "joint_null": np.outer(null_t, null_f),
    }


def eigen_prior_scale(
    phi_time: Array,
    phi_freq: Array,
    lam_time: Array,
    lam_freq: Array,
    joint_null: Array,
    *,
    null_precision: float = 1e-4,
    ridge_eps: float = 1e-6,
) -> Array:
    """Normal scales (Kt,Kf) for the anisotropic tensor eigen-coefficients.

    Precision is phi_time*lambda_time + phi_freq*lambda_freq outside the
    joint null space. Constants match the WDM defaults, without depending
    on its configuration class. A singleton zero time penalty gives a 1D
    prior with the WDM convention, not this package's stationary hierarchy.
    """
    precision = phi_time * lam_time[:, None] + phi_freq * lam_freq[None, :]
    return jnp.where(
        joint_null,
        1.0 / jnp.sqrt(null_precision),
        1.0 / jnp.sqrt(precision + ridge_eps),
    )


def create_bspline_roughness_penalty(
    knots: np.ndarray,
    *,
    degree: int,
    derivative_order: int = 2,
    quad_order: int = 8,
) -> np.ndarray:
    r"""Derivative-based B-spline roughness matrix.

    Entries are ``R_{ij} = \int B_i^{(q)}(x) B_j^{(q)}(x) dx`` with
    ``q = derivative_order``, evaluated by Gauss-Legendre quadrature on each
    non-degenerate knot span and normalized by its trace.
    """
    if derivative_order > degree:
        raise ValueError("derivative_order must be <= degree.")
    n_basis = len(knots) - degree - 1
    coeffs = np.eye(n_basis)
    deriv_splines = [
        interpolate.BSpline(
            knots, coeffs[i], degree, extrapolate=False
        ).derivative(derivative_order)
        for i in range(n_basis)
    ]
    penalty = np.zeros((n_basis, n_basis))
    abscissa, weights = np.polynomial.legendre.leggauss(quad_order)
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        x_eval = midpoint + half_width * abscissa
        values = np.stack([spline(x_eval) for spline in deriv_splines], axis=0)
        values = np.nan_to_num(values)
        penalty += (values * weights[None, :]) @ values.T * half_width
    penalty = 0.5 * (penalty + penalty.T)
    return penalty / np.maximum(np.trace(penalty), 1e-12)
