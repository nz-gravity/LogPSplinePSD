import jax
import jax.numpy as jnp
import optax

from log_psplines.datasets import Periodogram

from .psplines import LogPSplines
from .sampling import lnlikelihood


def optimize_logpsplines_weights(
    noise_f: Periodogram,
    log_psplines: LogPSplines,
    init_weights: jnp.ndarray,
    num_steps: int = 1000,
) -> jnp.ndarray:
    """
    Optimize spline weights by directly minimizing the negative Whittle log likelihood.

    This function wraps the optimization loop in a JAX-compiled loop using jax.lax.fori_loop.
    """

    # Now we assume that the likelihood function expects log power,
    # so compute the log of the power spectrum.
    noise_f_log = jnp.log(noise_f.power)

    # Define the loss as the negative log likelihood.
    def compute_loss(weights: jnp.ndarray) -> float:
        return -lnlikelihood(noise_f_log, log_psplines, weights)

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(init_weights)

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
