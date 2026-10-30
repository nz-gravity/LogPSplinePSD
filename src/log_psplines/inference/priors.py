"""Shared stationary spline priors for explicit and marginalised models."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular


def sample_pspline_block(
    sigma_name: str,
    weights_name: str,
    penalty_matrix: jnp.ndarray,
    roughness_scale: float,
    smoothing_parameterization: str = "centered",
) -> jnp.ndarray:
    """Draw spline weights; return shape ``(K,)`` under a HalfNormal scale."""
    sigma = numpyro.sample(sigma_name, dist.HalfNormal(roughness_scale))

    k = penalty_matrix.shape[0]
    base_normal = dist.Normal(0.0, 1.0).expand((k,)).to_event(1)
    if smoothing_parameterization == "noncentered":
        raw_weights = numpyro.sample(f"{weights_name}_raw", base_normal)
        cholesky = jnp.linalg.cholesky(penalty_matrix)
        weights = sigma * solve_triangular(
            cholesky.T, raw_weights, lower=False
        )
        numpyro.deterministic(weights_name, weights)
    else:
        weights = numpyro.sample(weights_name, base_normal)

    if smoothing_parameterization == "centered":
        wPw = jnp.dot(weights, jnp.dot(penalty_matrix, weights))
        precision = sigma**-2
        log_prior_w = 0.5 * k * jnp.log(precision) - 0.5 * precision * wPw
        base_log_prob = base_normal.log_prob(weights)
        numpyro.factor(
            f"weights_prior_{weights_name}", log_prior_w - base_log_prob
        )

    return weights
