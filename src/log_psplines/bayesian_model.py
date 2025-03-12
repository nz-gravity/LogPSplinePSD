import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax
from .psplines import build_spline

LOG2PI = jnp.log(2.0 * jnp.pi)


@jax.jit
def whittle_lnlike(ln_pdgrm: jnp.ndarray, ln_spline: jnp.ndarray):
    integrand = ln_spline + jnp.exp(ln_pdgrm - ln_spline - LOG2PI)
    return -0.5 * jnp.sum(integrand)


def bayesian_model(
        log_pdgrm: jnp.ndarray,  # shape (Nfreq,)  - log of observed periodogram
        lnspline_basis: jnp.ndarray,  # shape (kbasis, Nfreq,)  - matrix of spline basis functions
        penalty_matrix: jnp.ndarray,  # shape (kbasis, kbasis) - penalty matrix P
        alpha_phi, beta_phi,  # for phi | delta  => Gamma(alpha_phi, delta * beta_phi)
        alpha_delta, beta_delta,  # for delta => Gamma(alpha_delta, beta_delta)
):
    """
    A NumPyro model that samples:
      - delta ~ Gamma(alpha_delta, beta_delta)
      - phi   ~ Gamma(alpha_phi, delta * beta_phi)
      - w     ~ (penalized) MVN(0, (phi * P)^-1)
      - Whittle log-likelihood for data
    """

    # 1) Sample delta
    delta_dist = dist.Gamma(concentration=alpha_delta, rate=beta_delta)
    delta = numpyro.sample("delta", delta_dist)

    # 2) Sample phi | delta
    phi_dist = dist.Gamma(concentration=alpha_phi, rate=delta * beta_phi)
    phi = numpyro.sample("phi", phi_dist)

    # 3) Sample v from an unregularized Normal(0,1). We do dimension k from penalty_matrix
    k = penalty_matrix.shape[0]
    w = numpyro.sample(
        "weights",
        dist.Normal(0, 1).expand([k]).to_event(1)
    )

    # 4) Add a custom factor for the prior p(v | phi, delta) ~ MVN(0, (phi*P)^-1)
    #    log p(v) = 0.5*k*log(phi) - 0.5*phi * v^T P v   + (stuff that doesn't depend on v)
    wPw = jnp.dot(w, jnp.dot(penalty_matrix, w))
    log_prior_v = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
    # log_prior_phi = phi_dist.log_prob(phi)
    # log_prior_delta = delta_dist.log_prob(delta)
    # ln_prior = log_prior_v + log_prior_phi + log_prior_delta
    numpyro.factor("ln_prior", log_prior_v)  # really only need the log prior for v

    # 5) Build the log-splinens
    ln_spline = build_spline(lnspline_basis, w)

    # 6) Add the Whittle likelihood
    numpyro.factor("ln_likelihood", whittle_lnlike(log_pdgrm, ln_spline))
