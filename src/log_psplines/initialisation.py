import warnings
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.sparse import BCOO
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis

from .datatypes import Periodogram

__all__ = ["init_weights", "init_basis_and_penalty", "init_knots"]


def init_weights(
    log_pdgrm: jnp.ndarray,
    log_psplines: "LogPSplines",
    init_weights: jnp.ndarray = None,
    num_steps: int = 5000,
) -> jnp.ndarray:
    """
    Optimize spline weights by directly minimizing the negative Whittle log likelihood.

    This function wraps the optimization loop in a JAX-compiled loop using jax.lax.fori_loop.
    """
    log_param = log_psplines.log_parametric_model

    if init_weights is None:
        init_weights = jnp.zeros(log_psplines.n_basis)
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(init_weights)

    @jax.jit
    def compute_loss(weights: jnp.ndarray) -> float:
        lnmodel = log_psplines(weights) + log_param
        mse = jnp.mean((log_pdgrm - lnmodel) ** 2)
        return mse

    def step(i, state):
        weights, opt_state = state
        loss, grads = jax.value_and_grad(compute_loss)(weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (weights, opt_state)

    init_state = (init_weights, opt_state)
    final_state = jax.lax.fori_loop(0, num_steps, step, init_state)
    final_weights, _ = final_state
    return final_weights


def init_basis_and_penalty(
    knots: np.ndarray,
    degree: int,
    n_grid_points: int,
    diffMatrixOrder: int,
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a B-spline basis matrix and penalty matrix.

    Args:
        knots: Array of knots (values between 0 and 1).
        degree: Degree of the B-spline.
        n_grid_points: Number of grid points.
        diffMatrixOrder: Order of the differential operator for regularization.
        epsilon: Small constant for numerical stability.

    Returns:
        A tuple (basis_matrix, penalty_matrix) as JAX arrays.
    """
    order = degree + 1
    basis = BSplineBasis(domain_range=[0, 1], order=order, knots=knots)
    grid_points = np.linspace(0, 1, n_grid_points)
    basis_matrix = (
        basis.to_basis().to_grid(grid_points).data_matrix.squeeze().T
    )
    # normalise basis matrix elements (for numerical stability)
    knots_with_boundary = np.concatenate(
        [np.repeat(0, degree), knots, np.repeat(1, degree)]
    )
    n_knots_total = len(knots_with_boundary)
    mid_to_end = knots_with_boundary[degree + 1 :]
    start_to_mid = knots_with_boundary[: (n_knots_total - degree - 1)]
    norm_factor = (mid_to_end - start_to_mid) / (degree + 1)
    norm_factor[norm_factor == 0] = np.inf  # Prevent division by zero.
    basis_matrix = basis_matrix / norm_factor

    # swap all values below epsilon to 0
    # TODO: this is a hack... might not work
    # basis_matrix[basis_matrix < epsilon] = 0.0
    # basis_matrix = dense_to_sparse_jax(basis_matrix, threshold=epsilon)

    basis_matrix = jnp.array(basis_matrix)

    # Compute the penalty matrix using L2 regularization.
    regularization = L2Regularization(
        LinearDifferentialOperator(diffMatrixOrder)
    )
    p = regularization.penalty_matrix(basis)
    p = p / np.max(p)
    p = p + epsilon * np.eye(p.shape[1])
    return basis_matrix, jnp.array(p)


def dense_to_sparse_jax(matrix: jnp.ndarray, threshold=1e-10) -> BCOO:
    mask = jnp.abs(matrix) >= threshold
    values = matrix[mask]
    indices = jnp.argwhere(mask)
    return BCOO((values, indices), shape=matrix.shape)


def init_knots(
    n_knots: int,
    periodogram: Periodogram,
    parametric_model: jnp.ndarray = None,
    frac_uniform: float = 0.0,
    frac_log: float = 0.5,
) -> np.ndarray:
    """Select knots with a mix of uniform, log-spaced, and density-based placement.

             Instead of using a fixed grid (via log‐ or geomspace) to “force” a knot allocation,
         you can let the periodogram’s power distribution guide you. For example,
         you can interpret the (normalized) power as a probability density over frequency,
         compute its cumulative distribution function (CDF), and then choose knots at equally
         spaced quantiles of that CDF. In regions where the power (and hence “spikiness”) is higher,
          the CDF rises faster, so more knots will be allocated there.

        Ensures the first and last knots are at the min and max frequency.
        The remaining knots are allocated:
        - `frac_uniform`: Using uniform spacing.
        - `frac_log`: Using logarithmic spacing.
        - The rest: Using power-based density sampling.

    Args:
            periodogram: Periodogram object with freqs and power.
            n_knots: Total number of knots to select.
            frac_uniform: Fraction of knots to place uniformly (can be 0).
            frac_log: Fraction of knots to place logarithmically (can be 0).

        Returns:
            An array of knot locations (frequencies).
    """
    if n_knots < 2:
        raise ValueError(
            "At least two knots are required (min and max frequencies)."
        )

    min_freq, max_freq = periodogram.freqs[0], periodogram.freqs[-1]

    if n_knots == 2:
        return np.array([min_freq, max_freq])

    # Ensure fractions sum to at most 1
    frac_uniform = max(0.0, min(frac_uniform, 1.0))
    frac_log = max(0.0, min(frac_log, 1.0))
    frac_density = 1.0 - (frac_uniform + frac_log)

    # Compute number of knots in each category
    n_uniform = int(frac_uniform * (n_knots - 2)) if frac_uniform > 0 else 0
    n_log = int(frac_log * (n_knots - 2)) if frac_log > 0 else 0
    n_density = max(0, (n_knots - 2) - (n_uniform + n_log))  # Remaining knots

    # Uniformly spaced knots (excluding min/max)
    uniform_knots = (
        np.linspace(min_freq, max_freq, n_uniform + 2)[1:-1]
        if n_uniform > 0
        else np.array([])
    )

    # Log-spaced knots (excluding min/max)
    log_knots = (
        np.logspace(np.log10(min_freq), np.log10(max_freq), n_log + 2)[1:-1]
        if n_log > 0
        else np.array([])
    )

    # Power-based density sampling
    density_knots = np.array([])
    if n_density > 0:
        power = np.array(periodogram.power.copy(), dtype=np.float64)
        if parametric_model is not None:
            power -= parametric_model
            # ensure power is positive
            power = power + np.abs(np.min(power))

        density = power / np.sum(power)
        cdf = np.cumsum(density)

        # Compute quantiles for density-based knots
        quantiles = np.linspace(0, 1, n_density + 2)[1:-1]
        density_knots = np.interp(quantiles, cdf, periodogram.freqs)

    # Combine and sort
    knots = np.concatenate(
        ([min_freq], uniform_knots, log_knots, density_knots, [max_freq])
    )

    knots = np.sort(knots)  # Ensure order
    knots = (knots - min_freq) / (max_freq - min_freq)  # Normalize to [0, 1]
    # drop any nans that got through
    knots = knots[~np.isnan(knots)]

    unique_knots = np.unique(knots)
    if len(unique_knots) < len(knots):
        warnings.warn(
            f"Some knots were dropped due to duplication. [{n_knots}->{len(unique_knots)}]"
        )

    return unique_knots
