import warnings
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..datatypes import Periodogram
from .knots_locator import init_knots
from .penalty import build_bspline_basis_scipy, build_gps_penalty

__all__ = ["init_weights", "init_basis_and_penalty", "init_knots"]

if TYPE_CHECKING:
    from .psplines import LogPSplines


def init_weights(
    log_pdgrm: jnp.ndarray,
    log_psplines: "LogPSplines",
    init_weights: jnp.ndarray | None = None,
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
    def compute_loss(weights: jnp.ndarray) -> jnp.ndarray:
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate B-spline basis matrix and GPS penalty matrix.

    Uses scipy.interpolate.BSpline with phantom knots for the basis and the
    General P-Spline difference penalty (Li & Cao 2022) for the penalty.
    Both use the same phantom knot vector for mathematical consistency.

    Parameters
    ----------
    knots : np.ndarray
        Array of knots in [0, 1], including the boundary values 0.0 and 1.0.
    degree : int
        Polynomial degree of the B-spline.
    n_grid_points : int
        Number of evaluation points.  Ignored when ``grid_points`` is given.
    diff_matrix_order : int
        Penalty order m (0 = pure ridge, 1 = first differences, 2 = second
        differences, ...).  Must satisfy m < degree + 1.
    epsilon : float, default 1e-6
        Ridge regularisation added to the penalty diagonal.
    grid_points : np.ndarray, optional
        Explicit 1-D evaluation grid in [0, 1].  When provided,
        ``n_grid_points`` is still checked for length consistency.

    Returns
    -------
    basis_matrix : jnp.ndarray, shape (n_grid_points, n_basis)
        B-spline design matrix.  Entry [i, j] = B_j(grid_points[i]).
    penalty_matrix : jnp.ndarray, shape (n_basis, n_basis)
        Symmetric positive definite penalty P = D_m^T D_m + epsilon * I,
        normalised so max(|P|) = 1 before adding the ridge term.
    """
    if grid_points is None:
        grid_points = np.linspace(0.0, 1.0, n_grid_points)
    else:
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.ndim != 1:
            raise ValueError("grid_points must be 1-D if provided")
        if grid_points.size != n_grid_points:
            raise ValueError("grid_points length must match n_grid_points")
        grid_points = np.clip(grid_points, 0.0, 1.0)

    # --- Basis matrix, shape (n_grid_points, n_basis) ---
    basis_matrix_np = build_bspline_basis_scipy(knots, degree, grid_points)
    basis_matrix = jnp.asarray(basis_matrix_np)

    # --- GPS penalty matrix, shape (n_basis, n_basis) ---
    penalty_matrix_np = build_gps_penalty(
        knots, degree, diff_matrix_order, epsilon=epsilon
    )

    return basis_matrix, jnp.asarray(penalty_matrix_np)
