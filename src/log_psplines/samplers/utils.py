"""Shared utilities for sampler implementations."""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
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


def sample_pspline_block(
    delta_name: str,
    phi_name: str,
    weights_name: str,
    penalty_matrix: jnp.ndarray,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
    factor_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Draw hierarchical Gamma-Normal P-spline weights and record log priors."""

    delta_dist = dist.Gamma(concentration=alpha_delta, rate=beta_delta)
    delta = numpyro.sample(delta_name, delta_dist)
    log_prior_delta = delta_dist.log_prob(delta)

    phi_dist = dist.Gamma(concentration=alpha_phi, rate=delta * beta_phi)
    phi = numpyro.sample(phi_name, phi_dist)
    log_prior_phi = phi_dist.log_prob(phi)

    k = penalty_matrix.shape[0]
    base_normal = dist.Normal(0.0, 1.0).expand((k,)).to_event(1)
    weights = numpyro.sample(weights_name, base_normal)

    wPw = jnp.dot(weights, jnp.dot(penalty_matrix, weights))
    log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
    base_log_prob = base_normal.log_prob(weights)
    log_prior_adjustment = log_prior_w - base_log_prob

    if factor_name is None:
        factor_name = f"weights_prior_{weights_name}"
    numpyro.factor(factor_name, log_prior_adjustment)

    log_prior_total = log_prior_delta + log_prior_phi + log_prior_adjustment

    return {
        "weights": weights,
        "delta": delta,
        "phi": phi,
        "log_prior_adjustment": log_prior_adjustment,
        "log_prior_delta": log_prior_delta,
        "log_prior_phi": log_prior_phi,
        "log_prior_total": log_prior_total,
    }


def pspline_hyperparameter_initials(
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
    *,
    divide_phi_by_delta: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return default initial values for delta and phi hyperparameters."""

    delta_init = jnp.asarray(alpha_delta / beta_delta)
    if divide_phi_by_delta:
        phi_init = jnp.asarray(alpha_phi / (beta_phi * delta_init))
    else:
        phi_init = jnp.asarray(alpha_phi / beta_phi)
    return delta_init, phi_init
