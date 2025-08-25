import warnings
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numba.core.types import double
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis
from sympy.codegen.ast import float128

from ..datatypes import Periodogram

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
        Number of grid points
    diff_matrix_order : int
        Order of the differential operator for regularization
    epsilon : float, default=1e-6
        Small constant for numerical stability

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        (basis_matrix, penalty_matrix) as JAX arrays
    """
    order = degree + 1
    basis = BSplineBasis(domain_range=[0, 1], order=order, knots=knots)
    grid_points = np.linspace(0, 1, n_grid_points)

    # Compute basis matrix
    basis_matrix = (
        basis.to_basis().to_grid(grid_points).data_matrix.squeeze().T
    )

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


def init_knots(
    n_knots: int,
    periodogram: Periodogram,
    parametric_model: jnp.ndarray = None,
    method: str = "density",
    frac_uniform: float = 0.0,
    frac_log: float = 0.5,
    knots: np.ndarray = None,
) -> np.ndarray:
    """
    Select knots using various placement strategies.

    Parameters
    ----------
    n_knots : int
        Total number of knots to select
    periodogram : Periodogram
        Periodogram object with freqs and power
    parametric_model : jnp.ndarray, optional
        Parametric model to subtract from power before knot placement
    method : str, default="density"
        Knot placement method:
        - "uniform": Uniformly spaced knots
        - "log": Logarithmically spaced knots
        - "density": Quantile-based placement using periodogram (Patricio's method)
        - "mixed": Mix of uniform, log, and density-based placement
    frac_uniform : float, default=0.0
        For "mixed" method: fraction of knots placed uniformly
    frac_log : float, default=0.5
        For "mixed" method: fraction of knots placed logarithmically

    Returns
    -------
    np.ndarray
        Array of knot locations normalized to [0, 1]
    """


    if n_knots < 2:
        raise ValueError(
            "At least two knots are required (min and max frequencies)."
        )

    min_freq, max_freq = float(periodogram.freqs[0]), float(periodogram.freqs[-1])

    if n_knots == 2:
        return np.array([0.0, 1.0])

    if knots is not None:
        knots = np.array(knots)
    else:

        if method == "uniform":
            knots = np.linspace(min_freq, max_freq, n_knots)

        elif method == "log":
            # Handle zero frequency by using a small positive value
            min_freq_log = max(min_freq, 1e-10)
            knots = np.logspace(
                np.log10(min_freq_log), np.log10(max_freq), n_knots
            )

        elif method == "density":
            # Implement Patricio's quantile-based knot placement method
            knots = _quantile_based_knots(n_knots, periodogram, parametric_model)

        elif method == "mixed":
            knots = _mixed_knot_placement(
                n_knots, periodogram, parametric_model, frac_uniform, frac_log
            )

        else:
            raise ValueError(f"Unknown knot placement method: {method}")

    # Normalize to [0, 1] and ensure proper ordering
    original_knots = knots.copy()
    knots = np.array(knots, dtype=np.float128)
    knots = np.sort(knots)
    knots = (knots - min_freq) / (max_freq - min_freq)
    knots = np.clip(knots, 0.0, 1.0)
    # print if we have some nanas
    if np.isnan(knots).any():
        missing_knots = original_knots[np.isnan(knots)]
        warnings.warn(
            f"Some knots are NaN after normalization. "
            f"Missing knots: {missing_knots}"
        )
        knots = knots[~np.isnan(knots)]

    # ensure we have knots at ends 0 and 1
    knots = np.concatenate([[0.0], knots, [1.0]])
    unique_knots, counts = np.unique(knots, return_counts=True)
    # non_unique_knots = knots[np.isin(knots, knots[counts > 1])]

    if len(unique_knots) < len(knots):
        # get idx of non-unique knots
        warnings.warn(
            f"Some knots were dropped due to duplication. [{len(knots)}->{len(unique_knots)}]. "
            # f"Non-unique knots: {non_unique_knots}"
        )

    return unique_knots


def _quantile_based_knots(
    n_knots: int,
    periodogram: Periodogram,
    parametric_model: jnp.ndarray = None,
) -> np.ndarray:
    """
    Implement Patricio's quantile-based knot placement method.

    The procedure follows these steps:
    1. Take square root of periodogram values
    2. Standardize the values
    3. Take absolute values and normalize to create a PMF
    4. Interpolate to get a continuous CDF
    5. Place knots at equally spaced quantiles of this CDF
    """
    # Step 1: Square root transformation
    x = np.sqrt(periodogram.power)

    # Optionally subtract parametric model
    if parametric_model is not None:
        # Subtract from power, then take square root
        power_adjusted = periodogram.power - parametric_model
        # Ensure positivity
        power_adjusted = power_adjusted + np.abs(np.min(power_adjusted))
        x = np.sqrt(power_adjusted)

    # Step 2: Standardize
    x_mean = np.mean(x)
    x_std = np.std(x)
    y = (x - x_mean) / x_std

    # Step 3: Absolute values and normalize to create PMF
    z = np.abs(y)
    z = z / np.sum(z)  # Normalize to sum to 1

    # Step 4: Create cumulative distribution function
    cdf_values = np.cumsum(z)

    # Step 5: Place knots at equally spaced quantiles
    # We want n_knots total, including endpoints
    quantiles = np.linspace(0, 1, n_knots)

    # Interpolate to find frequencies corresponding to these quantiles
    knots = np.interp(quantiles, cdf_values, periodogram.freqs)

    return knots


def _mixed_knot_placement(
    n_knots: int,
    periodogram: Periodogram,
    parametric_model: jnp.ndarray = None,
    frac_uniform: float = 0.0,
    frac_log: float = 0.5,
) -> np.ndarray:
    """
    Mixed knot placement strategy combining uniform, log, and density-based placement.
    """
    min_freq, max_freq = periodogram.freqs[0], periodogram.freqs[-1]

    # Ensure fractions sum to at most 1
    frac_uniform = max(0.0, min(frac_uniform, 1.0))
    frac_log = max(0.0, min(frac_log, 1.0))
    frac_density = 1.0 - (frac_uniform + frac_log)

    # Compute number of knots in each category (excluding endpoints)
    n_interior = n_knots - 2
    n_uniform = int(frac_uniform * n_interior) if frac_uniform > 0 else 0
    n_log = int(frac_log * n_interior) if frac_log > 0 else 0
    n_density = max(0, n_interior - (n_uniform + n_log))

    knots_list = [min_freq, max_freq]  # Always include endpoints

    # Uniform knots
    if n_uniform > 0:
        uniform_knots = np.linspace(min_freq, max_freq, n_uniform + 2)[1:-1]
        knots_list.extend(uniform_knots)

    # Log-spaced knots
    if n_log > 0:
        min_freq_log = max(min_freq, 1e-10)
        log_knots = np.logspace(
            np.log10(min_freq_log), np.log10(max_freq), n_log + 2
        )[1:-1]
        knots_list.extend(log_knots)

    # Density-based knots
    if n_density > 0:
        # Create a temporary periodogram for density-based placement
        density_knots = _quantile_based_knots(
            n_density + 2, periodogram, parametric_model
        )[1:-1]
        knots_list.extend(density_knots)

    return np.array(knots_list)
