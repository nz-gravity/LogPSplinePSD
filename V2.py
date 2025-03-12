import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

# --- User-supplied functions ---


def whittle_log_likelihood(Y, f):
    """
    Compute the log Whittle likelihood.

    For example, if Y is the periodogram at frequencies λₗ and f(λₗ) is
    the spectral density estimate, one common formulation is
        log L = -∑ [I_n(λₗ)/f(λₗ) + log f(λₗ)]
    Adjust this function to your needs.
    """
    return -jnp.sum(Y / f + jnp.log(f))


def spectral_density(v, phi, tau):
    """
    Compute the spectral density as a function of the parameters.

    This is a placeholder. In your application, f might be defined via the
    pspline representation or another function of v, phi (the penalty parameter),
    and tau. Replace this with your actual function.
    """
    # For example, one might use an exponential form or another model:
    # f = tau * jnp.exp(jnp.dot(v, v)) / phi  <-- (this is just a dummy example)
    f = tau * jnp.exp(jnp.sum(v**2)) / phi
    return f


# --- NumPyro model definition ---


def model(
    Y,
    periodogram,
    P,
    vbar,
    S_sqrt,
    alpha_phi,
    beta_phi,
    alpha_delta,
    beta_delta,
    alpha_tau,
    beta_tau,
    nu,
    K,
):
    """
    Y           : observed data (or a transformation thereof)
    periodogram : periodogram values (used in the likelihood for τ)
    P           : penalty matrix (used in the quadratic form vᵀ P v)
    vbar        : pilot mean vector for v (from a preliminary sample)
    S_sqrt      : square-root matrix of the covariance (from the pilot sample)
    alpha_phi, beta_phi, alpha_delta, beta_delta, alpha_tau, beta_tau, nu : hyperparameters
    K           : number of basis functions + 1 (so that v is (K-1)-dimensional)
    """
    # --- Reparameterization for v ---
    # Instead of sampling v directly, we sample beta ~ N(0,I) and set
    # v = S_sqrt @ beta + vbar.
    beta = numpyro.sample(
        "beta", dist.Normal(jnp.zeros(K - 1), jnp.ones(K - 1))
    )
    v = jnp.matmul(S_sqrt, beta) + vbar
    # Compute the quadratic form: vᵀ P v
    vTPv = jnp.dot(v, jnp.matmul(P, v))

    # --- Sampling φ and δ ---
    # The full conditionals given in your paper are:
    #   φ | … ∼ Gamma((K–1)/2 + α_φ, 0.5*vᵀ P v + δ * β_φ)
    #   δ | … ∼ Gamma(α_φ + α_δ, β_φ * φ + β_δ)
    #
    # In a joint model the ordering is important. One strategy is to sample one of these
    # (say, δ) from a “marginal” prior and then sample φ conditionally, and add an extra
    # factor (via numpyro.factor) to “correct” the joint density.
    #
    # Here we first sample δ from a Gamma distribution that uses only part of the rate;
    # then we sample φ given δ.
    #
    # (This is one of several possible strategies; you may choose to reparameterize the
    # joint density differently if you wish.)

    # Sample δ from a preliminary Gamma (using β_δ alone)
    delta = numpyro.sample(
        "delta", dist.Gamma(alpha_phi + alpha_delta, beta_delta)
    )

    # Now sample φ using δ in the rate.
    phi = numpyro.sample(
        "phi",
        dist.Gamma((K - 1) / 2 + alpha_phi, 0.5 * vTPv + delta * beta_phi),
    )

    # Because the “true” full conditional for δ has rate β_φ * φ + β_δ,
    # add a correction factor so that the joint density is as desired.
    # (That is, we add the log-probability difference between the target and the
    # preliminary δ density.)
    log_correction = dist.Gamma(
        alpha_phi + alpha_delta, beta_phi * phi + beta_delta
    ).log_prob(delta) - dist.Gamma(
        alpha_phi + alpha_delta, beta_delta
    ).log_prob(
        delta
    )
    numpyro.factor("delta_correction", log_correction)

    # --- Sampling τ ---
    # The full conditional is:
    #   τ | … ∼ InverseGamma(α_τ + ν,  Σₗ I_n(λₗ)/[ ... ] + β_τ)
    #
    # For demonstration we use a placeholder for the sum in the rate.
    tau_rate = jnp.sum(periodogram)  # replace with your actual computation
    tau = numpyro.sample(
        "tau", dist.InverseGamma(alpha_tau + nu, tau_rate + beta_tau)
    )

    # --- Likelihood ---
    # Compute the spectral density (or f, as in your notation) from the parameters.
    f = spectral_density(v, phi, tau)

    # Incorporate the Whittle likelihood into the joint log-density.
    numpyro.factor("likelihood", whittle_log_likelihood(Y, f))
