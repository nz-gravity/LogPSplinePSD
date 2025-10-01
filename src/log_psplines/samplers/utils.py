"""Shared utilities for sampler implementations."""

from typing import Any, Dict, Tuple

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
    penalty_whiten: jnp.ndarray,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
) -> Dict[str, Any]:
    """Draw hierarchical Gamma-Normal P-spline weights via whitening."""

    delta_dist = dist.Gamma(concentration=alpha_delta, rate=beta_delta)
    delta = numpyro.sample(delta_name, delta_dist)

    # Moment-match the original Gamma prior with a log-normal and sample log(phi)
    # to reduce the funnel geometry seen by NUTS.
    sigma_sq = jnp.log1p(1.0 / alpha_phi)
    sigma = jnp.sqrt(sigma_sq)
    mu = (
        jnp.log(alpha_phi)
        - jnp.log(beta_phi)
        - jnp.log(delta)
        - 0.5 * sigma_sq
    )
    phi_normal = dist.Normal(loc=mu, scale=sigma)
    log_phi = numpyro.sample(
        phi_name,
        phi_normal,
    )
    phi = jnp.exp(log_phi)
    k = penalty_whiten.shape[0]
    base_normal = dist.Normal(0.0, 1.0).expand((k,)).to_event(1)
    latent_name = f"{weights_name}_latent"
    z = numpyro.sample(latent_name, base_normal)
    weights = (penalty_whiten @ z) / jnp.sqrt(phi)
    numpyro.deterministic(weights_name, weights)

    return {
        "weights": weights,
        "weights_latent": z,
        "delta": delta,
        "phi": phi,
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
