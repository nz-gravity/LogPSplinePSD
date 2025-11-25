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

__all__ = ["init_weights", "init_basis_and_penalty", "init_knots"]


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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    Tuple[jnp.ndarray, jnp.ndarray]
        (basis_matrix, penalty_matrix) as JAX arrays
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

    # Normalize basis matrix elements for numerical stability
    # knots_with_boundary = np.concatenate(
    #     [np.repeat(0, degree), knots, np.repeat(1, degree)]
    # )
    # n_knots_total = len(knots_with_boundary)
    # mid_to_end = knots_with_boundary[degree + 1 :]
    # start_to_mid = knots_with_boundary[: (n_knots_total - degree - 1)]
    # norm_factor = (mid_to_end - start_to_mid) / (degree + 1)
    # norm_factor[norm_factor == 0] = np.inf  # Prevent division by zero
    # basis_matrix = basis_matrix / norm_factor

    basis_matrix = jnp.array(basis_matrix)

    # Compute penalty matrix using L2 regularization
    regularization = L2Regularization(
        LinearDifferentialOperator(diff_matrix_order)
    )
    penalty_matrix = regularization.penalty_matrix(basis)
    penalty_matrix = penalty_matrix / np.max(penalty_matrix)
    penalty_matrix = penalty_matrix + epsilon * np.eye(penalty_matrix.shape[1])

    return basis_matrix, jnp.array(penalty_matrix)
