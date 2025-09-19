import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import List
from ...datatypes import MultivarFFT



def whittle_likelihood(
        data: MultivarFFT,
        log_delta_sq: jnp.ndarray,  # (n_freq, n_dim)
        theta_re: jnp.ndarray,      # (n_freq, n_theta)
        theta_im: jnp.ndarray       # (n_freq, n_theta)
) -> jnp.ndarray:
    """
    Multivariate Whittle likelihood for Cholesky PSD parameterization.

    .. math::
        \log L = -\sum_{j,k} \log \delta_{jk}^2 - \sum_{j,k} \frac{|y_{jk} - \sum_l \theta_{jl} y_{lk}|^2}{\delta_{jk}^2}
    """
    sum_log_det = -jnp.sum(log_delta_sq)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)
    if data.Z_re.shape[2] > 0:
        Z_theta_re = jnp.einsum('fij,fj->fi', data.Z_re, theta_re) - jnp.einsum('fij,fj->fi', data.Z_im, theta_im)
        Z_theta_im = jnp.einsum('fij,fj->fi', data.Z_re, theta_im) + jnp.einsum('fij,fj->fi', data.Z_im, theta_re)
        u_re = data.y_re - Z_theta_re
        u_im = data.y_im - Z_theta_im
    else:
        u_re = data.y_re
        u_im = data.y_im
    numerator = u_re ** 2 + u_im ** 2
    internal = numerator * exp_neg_log_delta
    tmp2 = -jnp.sum(internal)
    return sum_log_det + tmp2

def multivariate_psplines_model(
        data: MultivarFFT,
        all_bases: List[jnp.ndarray],
        all_penalties: List[jnp.ndarray],
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
):
    """
    NumPyro model for multivariate PSD estimation using P-splines and Cholesky parameterization.
    """
    n_dim = data.n_dim
    n_freq = data.n_freq
    n_theta = data.Z_re.shape[2]
    component_idx = 0
    log_delta_components = []
    for j in range(n_dim):
        delta = numpyro.sample(f"delta_{j}", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample(f"phi_delta_{j}", dist.Gamma(alpha_phi, delta*beta_phi))
        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample(f"weights_delta_{j}", dist.Normal(0, 1).expand((k,)).to_event(1))
        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor(f"weights_prior_delta_{j}", log_prior_w)
        log_delta_sq_j = all_bases[component_idx] @ weights
        log_delta_components.append(log_delta_sq_j)
        component_idx += 1
    log_delta_sq = jnp.stack(log_delta_components, axis=1)
    if n_theta > 0:
        delta = numpyro.sample("delta_theta_re", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_re", dist.Gamma(alpha_phi, delta*beta_phi))
        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_re", dist.Normal(0, 1).expand((k,)).to_event(1))
        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_re", log_prior_w)
        theta_re_base = all_bases[component_idx] @ weights
        theta_re = jnp.tile(theta_re_base[:, None], (1, max(1, n_theta)))
        component_idx += 1
        delta = numpyro.sample("delta_theta_im", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_im", dist.Gamma(alpha_phi, delta*beta_phi))
        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_im", dist.Normal(0, 1).expand((k,)).to_event(1))
        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_im", log_prior_w)
        theta_im_base = all_bases[component_idx] @ weights
        theta_im = jnp.tile(theta_im_base[:, None], (1, max(1, n_theta)))
    else:
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))
    log_likelihood = whittle_likelihood(data, log_delta_sq, theta_re, theta_im)
    numpyro.factor("likelihood", log_likelihood)
    numpyro.deterministic("log_delta_sq", log_delta_sq)
    numpyro.deterministic("theta_re", theta_re)
    numpyro.deterministic("theta_im", theta_im)
    numpyro.deterministic("log_likelihood", log_likelihood)