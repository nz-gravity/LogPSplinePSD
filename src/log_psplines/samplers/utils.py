"""Shared utilities for sampler implementations."""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density


def build_log_density_fn(model, model_kwargs: Dict[str, Any]):
    """Return a JIT-compiled callable that evaluates the NumPyro log posterior.

    Parameters
    ----------
    model:
        NumPyro model callable.
    model_kwargs:
        Keyword arguments passed to ``model``.

    Returns
    -------
    Callable[[Dict[str, jnp.ndarray]], jnp.ndarray]
        Function that maps a pytree of parameter arrays to the log posterior.
    """

    model_kwargs = jax.tree_util.tree_map(jnp.asarray, model_kwargs)

    def _logpost(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        log_prob, _ = log_density(model, (), model_kwargs, params)
        return log_prob

    return jax.jit(_logpost)


def evaluate_log_density_batch(
    logpost_fn, params_batch: Dict[str, jnp.ndarray]
):
    """Evaluate a batched set of parameters with a compiled log posterior."""
    vmapped = jax.vmap(logpost_fn)
    return np.asarray(jax.device_get(vmapped(params_batch)), dtype=np.float64)
