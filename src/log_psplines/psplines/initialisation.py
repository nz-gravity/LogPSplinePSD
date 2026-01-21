import warnings
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis

from ..datatypes import Periodogram
from .knots_locator import init_knots

__all__ = [
    "init_weights",
    "init_basis_and_penalty",
    "init_knots",
    "make_penalty_spd",
]


def make_penalty_spd(
    P_np: np.ndarray,
    *,
    eps_rel: float = 1e-6,
    eps_abs: float = 0.0,
    do_eig_floor: bool = True,
):
    """Return a symmetric positive-definite penalty matrix and its Cholesky factor.

    Note on numerical stability:
    The P-spline penalty matrix constructed from finite differences is theoretically positive semidefinite and has a known nullspace (e.g. constant/linear components for a second-difference penalty). In finite precision arithmetic, this can lead to tiny negative eigenvalues or extreme ill-conditioning, which in turn causes Cholesky factorisation to fail (e.g. NaNs during whitening).

    We enforce positive definiteness by symmetrising and adding diagonal jitter
    that is *relative to the scale of the matrix*:

        tau = eps_rel * lam_scale + eps_abs,

    where ``lam_scale`` is a representative magnitude (typically the max
    diagonal, falling back to an eigenvalue-based estimate).

    Important:
    Do not normalise the penalty matrix here unless the model resales the
    smoothness parameter accordingly.

    Parameters
    ----------
    P_np : np.ndarray
        Input penalty matrix that may be slightly non-symmetric or indefinite.
    eps_rel : float, optional
        Relative jitter size scaled to a typical diagonal magnitude.
    eps_abs : float, optional
        Absolute jitter floor, helpful if the diagonal scale is tiny.
    do_eig_floor : bool, optional
        If True, clamp negative eigenvalues to zero before adding jitter.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        The SPD matrix, its lower-triangular Cholesky factor, and info about
        jitter used.
    """

    P = np.asarray(P_np, dtype=np.float64)

    # Symmetrise to drop antisymmetric numerical noise.
    P = 0.5 * (P + P.T)

    diag = np.diag(P)
    diag_max = float(np.max(np.abs(diag))) if diag.size else 0.0
    if np.isfinite(diag_max) and diag_max > 0.0:
        lam_scale = diag_max
    else:
        try:
            eig_max = float(np.max(np.linalg.eigvalsh(P)))
        except np.linalg.LinAlgError:
            eig_max = 0.0
        lam_scale = eig_max if np.isfinite(eig_max) and eig_max > 0.0 else 1.0

    if do_eig_floor:
        w, V = np.linalg.eigh(P)
        w = np.maximum(w, 0.0)
        P = (V * w) @ V.T
        P = 0.5 * (P + P.T)

    tau = float(eps_rel) * float(lam_scale) + float(eps_abs)
    if not np.isfinite(tau) or tau <= 0.0:
        tau = max(float(eps_abs), 1e-12)

    # Ensure PD by increasing tau if needed.
    chol = None
    tau_used = tau
    for _ in range(8):
        P_pd = P + tau_used * np.eye(P.shape[0], dtype=np.float64)
        try:
            chol = np.linalg.cholesky(P_pd)
            break
        except np.linalg.LinAlgError:
            tau_used *= 10.0

    if chol is None:
        raise np.linalg.LinAlgError(
            "Failed to compute Cholesky factor for penalty matrix after jitter."
        )

    if np.any(np.isnan(chol)):
        raise RuntimeError(
            "Cholesky factor contains NaNs despite jitter addition."
        )

    return P_pd, chol, {"lam_scale": lam_scale, "jitter_used": tau_used}


def init_weights(
    log_pdgrm: jnp.ndarray,
    log_psplines: "LogPSplines",
    init_weights: jnp.ndarray = None,
    num_steps: int = 5000,
) -> jnp.ndarray:
    """
    Optimize spline weights by directly minimizing the MSE between
    log periodogram and log model.

    Parameters
    ----------
    log_pdgrm : jnp.ndarray
        Log of the periodogram values
    log_psplines : LogPSplines
        The log P-splines model object
    init_weights : jnp.ndarray, optional
        Initial weights. If None, uses zeros.
    num_steps : int, default=5000
        Number of optimization steps


    Returns
    -------
    jnp.ndarray
        Optimized weights
    """
    if init_weights is None:
        init_weights = jnp.zeros(log_psplines.n_basis)

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(init_weights)

    @jax.jit
    def compute_loss(weights: jnp.ndarray) -> float:
        """Compute MSE loss between log periodogram and log model"""
        log_model = log_psplines(weights) + log_psplines.log_parametric_model
        return jnp.mean((log_pdgrm - log_model) ** 2)

    def step(i, state):
        """Single optimization step"""
        weights, opt_state = state
        loss, grads = jax.value_and_grad(compute_loss)(weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (weights, opt_state)

    # Run optimization loop
    init_state = (init_weights, opt_state)
    final_state = jax.lax.fori_loop(0, num_steps, step, init_state)
    final_weights, _ = final_state

    return final_weights


def init_basis_and_penalty(
    knots: np.ndarray,
    degree: int,
    n_grid_points: int,
    diff_matrix_order: int,
    epsilon: float = 1e-6,
    grid_points: np.ndarray | None = None,
    spd_penalty: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate B-spline basis matrix and penalty matrix.

    Parameters
    ----------
    knots : np.ndarray
        Array of knots (values between 0 and 1)
    degree : int
        Degree of the B-spline
    n_grid_points : int
        Number of grid points. Ignored if `grid_points` is provided.
    diff_matrix_order : int
        Order of the differential operator for regularization
    epsilon : float, default=1e-6
        Small constant for numerical stability
    grid_points : np.ndarray, optional
        Locations in [0, 1] at which to evaluate the basis. If None,
        uses a uniform grid of length `n_grid_points`.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        (basis_matrix, penalty_matrix_pd, penalty_chol) as JAX arrays
    """
    order = degree + 1
    basis = BSplineBasis(domain_range=[0, 1], order=order, knots=knots)
    if grid_points is None:
        grid_points = np.linspace(0, 1, n_grid_points)
    else:
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.ndim != 1:
            raise ValueError("grid_points must be 1-D if provided")
        if grid_points.size != n_grid_points:
            raise ValueError("grid_points length must match n_grid_points")
        # Clip to [0,1] for numerical safety
        grid_points = np.clip(grid_points, 0.0, 1.0)

    # Compute basis matrix and keep it explicitly 2-D (n_grid, n_basis)
    basis_eval = basis.to_basis().to_grid(grid_points).data_matrix
    basis_eval = np.asarray(basis_eval, dtype=np.float64)
    basis_matrix = np.squeeze(basis_eval, axis=-1).T
    basis_matrix = jnp.array(basis_matrix)

    # Compute penalty matrix using L2 regularization
    regularization = L2Regularization(
        LinearDifferentialOperator(diff_matrix_order)
    )
    penalty_matrix = regularization.penalty_matrix(basis)

    if spd_penalty:
        penalty_matrix_pd, penalty_chol, _ = make_penalty_spd(
            penalty_matrix,
            eps_rel=epsilon,
            eps_abs=0.0,
            do_eig_floor=True,
        )
    else:
        penalty_matrix_pd = 0.5 * (
            np.asarray(penalty_matrix) + np.asarray(penalty_matrix).T
        )
        penalty_chol = np.linalg.cholesky(
            penalty_matrix_pd
            + float(epsilon) * np.eye(penalty_matrix_pd.shape[0])
        )

    return (
        basis_matrix,
        jnp.asarray(penalty_matrix_pd),
        jnp.asarray(penalty_chol),
    )
