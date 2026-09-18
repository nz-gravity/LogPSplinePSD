import jax
import jax.numpy as jnp
import optax

from log_psplines.basis.splines import SplineBasis
from log_psplines.models.spectrum import LogPSpline


def _least_squares_weight_initialiser(
    log_pdgrm: jnp.ndarray,
    log_psplines: "LogPSpline",
) -> jnp.ndarray:
    """Return a stabilized least-squares fit for the spline weights."""
    basis = jnp.asarray(log_psplines.basis)
    target = jnp.asarray(log_pdgrm)
    gram = basis.T @ basis
    rhs = basis.T @ target

    n_basis = int(gram.shape[0])
    trace_scale = jnp.trace(gram) / max(n_basis, 1)
    ridge = jnp.asarray(1e-6, dtype=gram.dtype) * jnp.maximum(
        trace_scale, jnp.asarray(1.0, dtype=gram.dtype)
    )
    system = gram + ridge * jnp.eye(n_basis, dtype=gram.dtype)
    return jnp.linalg.solve(system, rhs)


def init_weights(
    log_pdgrm: jnp.ndarray,
    log_psplines: "LogPSpline",
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
    log_psplines : LogPSpline
        The log P-splines model object
    init_weights : jnp.ndarray, optional
        Initial weights to refine. If None, starts from a stabilized least-
        squares spline fit to the observed log spectrum before refinement.
    num_steps : int, default=5000
        Number of optimization steps used to refine the initial weights.

    Returns
    -------
    jnp.ndarray
        Optimized weights
    """
    if init_weights is None:
        init_weights = _least_squares_weight_initialiser(
            log_pdgrm, log_psplines
        )

    if num_steps <= 0:
        return jnp.asarray(init_weights)

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(init_weights)

    @jax.jit
    def compute_loss(weights: jnp.ndarray) -> jnp.ndarray:
        """Compute MSE loss between log periodogram and log model"""
        return jnp.mean((log_pdgrm - log_psplines(weights)) ** 2)

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


def build_component(
    *,
    degree: int,
    diffMatrixOrder: int,
    n: int,
    knots: jnp.ndarray,
    basis: jnp.ndarray | None = None,
    penalty_matrix: jnp.ndarray | None = None,
    weights: jnp.ndarray | None = None,
    grid_points: jnp.ndarray | None = None,
    log_target: jnp.ndarray | None = None,
    init_num_steps: int = 5000,
) -> LogPSpline:
    """Prepare a scalar component, optionally fitting its initial weights."""
    frequency = SplineBasis.create(
        degree=degree,
        penalty_order=diffMatrixOrder,
        n=n,
        knots=knots,
        basis=basis,
        penalty=penalty_matrix,
        grid=grid_points,
    )
    model = LogPSpline(frequency, weights=weights)
    if log_target is not None:
        target = jnp.asarray(log_target)
        if target.ndim != 1 or target.shape[0] != n:
            raise ValueError("log_target must be 1-D with length n")
        model.weights = init_weights(
            target, model, init_weights=model.weights, num_steps=init_num_steps
        )
    return model
