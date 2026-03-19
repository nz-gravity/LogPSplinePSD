"""Shared utilities for sampler implementations."""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import log_density

from .._jaxtypes import Float
from .._typecheck import runtime_typecheck


def _prepare_model_kwarg(value: Any) -> Any:
    """Convert array-like model kwargs to JAX arrays, preserving scalars."""
    if isinstance(value, (jax.Array, np.ndarray)):
        return jnp.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_log_density_fn(model, model_kwargs: dict[str, Any]):
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

    model_kwargs = jax.tree_util.tree_map(_prepare_model_kwarg, model_kwargs)

    def _logpost(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        log_prob, _ = log_density(model, (), model_kwargs, params)
        return log_prob

    return jax.jit(_logpost)


@runtime_typecheck
def evaluate_log_density_batch(
    logpost_fn, params_batch: dict[str, jax.Array]
) -> np.ndarray:
    """Evaluate a batched set of parameters with a compiled log posterior."""
    vmapped = jax.vmap(logpost_fn)
    return np.asarray(jax.device_get(vmapped(params_batch)), dtype=np.float64)


def sample_pspline_block(
    delta_name: str,
    phi_name: str,
    weights_name: str,
    penalty_matrix: Float[jax.Array, "k k"],
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
    factor_name: Optional[str] = None,
    w_design: Optional[jax.Array] = None,
    tau: Optional[float] = None,
) -> dict[str, Any]:
    """Draw hierarchical Gamma-Normal P-spline weights and record log priors.

    Optional Parameters
    ----------
    w_design:
        Reference weights to shrink toward instead of zero. When provided,
        the smoothness penalty acts on ``weights - w_design``. If ``None``
        (default), behavior is identical to the original implementation.
    tau:
        Standard deviation for an additional isotropic Gaussian prior on
        ``weights - w_design``. Only applied when ``w_design`` is also
        provided. Smaller values pull the posterior more strongly toward the
        design.
    """
    delta_dist = dist.Gamma(concentration=alpha_delta, rate=beta_delta)
    delta = numpyro.sample(delta_name, delta_dist)
    log_prior_delta = delta_dist.log_prob(delta)

    # Moment-match the original Gamma prior with a log-normal and sample log(phi)
    # to reduce the funnel geometry seen by NUTS.
    # NOTE: spoke to Renate about this in ~Sept 2025, she thinks its fine
    sigma_sq = jnp.log1p(1.0 / alpha_phi)
    sigma = jnp.sqrt(sigma_sq)
    # Guard against delta underflow to 0.0 in float32 which makes log(delta)
    # diverge and can produce inf/NaN loc parameters during VI/NUTS init.
    delta_safe = jnp.maximum(delta, jnp.asarray(1e-12, dtype=delta.dtype))
    mu = (
        jnp.log(alpha_phi)
        - jnp.log(beta_phi)
        - jnp.log(delta_safe)
        - 0.5 * sigma_sq
    )
    phi_normal = dist.Normal(loc=mu, scale=sigma)
    log_phi = numpyro.sample(
        phi_name,
        phi_normal,
    )
    phi = jnp.exp(log_phi)
    log_prior_phi = phi_normal.log_prob(log_phi) - log_phi

    k = penalty_matrix.shape[0]
    base_normal = dist.Normal(0.0, 1.0).expand((k,)).to_event(1)
    weights = numpyro.sample(weights_name, base_normal)

    residual = weights if w_design is None else weights - w_design
    wPw = jnp.dot(residual, jnp.dot(penalty_matrix, residual))
    log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
    if tau is not None and w_design is not None:
        # DEFAULT IS OFF
        # this additional prior is not part of the original model but can help regularise
        # the posterior around the design and improve NUTS exploration when the design is informative
        log_prior_w += -0.5 * jnp.sum(residual**2) / tau**2
    base_log_prob = base_normal.log_prob(weights)
    log_prior_adjustment = log_prior_w - base_log_prob

    if factor_name is None:
        factor_name = f"weights_prior_{weights_name}"
    numpyro.factor(factor_name, log_prior_adjustment)

    # NOTE: no need to 'factor' in prior from delta and phi since they
    # are explicitly sampled as part of the model and will be included
    # in the log posterior automatically via Numpyro.

    return {
        "weights": weights,
        "delta": delta,
        "phi": phi,
    }


@runtime_typecheck
def pspline_hyperparameter_initials(
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
    *,
    divide_phi_by_delta: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return default initial values for delta and phi hyperparameters."""

    delta_init = jnp.asarray(alpha_delta / beta_delta)
    if divide_phi_by_delta:
        phi_init = jnp.asarray(alpha_phi / (beta_phi * delta_init))
    else:
        phi_init = jnp.asarray(alpha_phi / beta_phi)
    return delta_init, phi_init
