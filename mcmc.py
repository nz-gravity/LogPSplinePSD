import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from log_psplines.psplines import LogPSplines


def whittle_log_likelihood(log_pdgrm, ln_spline):
    """
    Whittle log-likelihood for log power spectral density data:
        ln L = -0.5 * sum( ln_spline + exp( log_pdgrm - ln_spline - ln(2*pi) ) )
    """
    integrand = ln_spline + jnp.exp(
        log_pdgrm - ln_spline - jnp.log(2.0 * jnp.pi)
    )
    return -0.5 * jnp.sum(integrand)


def bayesian_model(
    log_pdgrm: jnp.ndarray,
    log_spline: LogPSplines,
    alpha_phi=1.0,
    beta_phi=1.0,
):
    # 1) Sample phi (precision scalar).
    phi = numpyro.sample(
        "phi", dist.Gamma(concentration=alpha_phi, rate=beta_phi)
    )

    # 2) Sample raw weights w from an unregularized Normal(0,1).
    k = spline_model.n_basis
    w = numpyro.sample("w", dist.Normal(0, 1).expand([k]).to_event(1))

    # 3) Add a custom prior factor for w with penalty matrix:
    #       p(w) ~ MVN(0, (phi * P)^-1)
    #    => log p(w) = 0.5*k*log(phi) - 0.5*phi * w^T P w + const
    w_prior = 0.5 * k * jnp.log(phi) - 0.5 * phi * jnp.dot(
        w, jnp.dot(spline_model.penalty_matrix, w)
    )
    numpyro.factor("w_prior", w_prior)

    # 4) Evaluate the log-spectrum with the current weights
    ln_spline = spline_model(w)

    # 5) Add Whittle log-likelihood
    lnlike = whittle_log_likelihood(log_pdgrm, ln_spline)
    numpyro.factor("whittle_likelihood", lnlike)


def generate_synthetic_data(num_points=256, seed=0):
    """
    Generate a synthetic periodogram and return the log of the power.
    For illustration, we just create a smooth curve plus noise.
    """
    rng = np.random.default_rng(seed)
    freqs = np.linspace(1.0, 50.0, num_points)  # e.g., from 1 to 50 Hz
    # A mock "true" PSD that has a peak near 10 Hz
    true_psd = 1e2 * np.exp(-0.5 * ((freqs - 10) / 3) ** 2) + 1.0
    # Add multiplicative noise
    noisy_psd = true_psd * rng.lognormal(mean=0.0, sigma=0.2, size=num_points)
    log_pdgrm = np.log(noisy_psd)
    return freqs, log_pdgrm


# ---------------------------------------------------------------------------
# 5. MCMC wrapper using NUTS
# ---------------------------------------------------------------------------
def run_mcmc(
    log_pdgrm,
    spline_model,
    alpha_phi=1.0,
    beta_phi=1.0,
    num_warmup=500,
    num_samples=1000,
    rng_key=0,
):
    """
    Run NUTS MCMC on the Whittle model.

    log_pdgrm: jnp.ndarray
        The log of the observed periodogram.
    spline_model: LogPSplines
        Instance of the LogPSplines class.
    alpha_phi, beta_phi: float
        Hyperparameters for phi's Gamma prior.
    num_warmup: int
        Number of warmup (burn-in) steps.
    num_samples: int
        Number of samples to draw after warmup.
    rng_key: int
        Random seed for reproducibility.
    """
    rng_key = jax.random.PRNGKey(rng_key)

    # Set up NUTS
    kernel = NUTS(bayesian_model())
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

    # Run the sampler
    mcmc.run(rng_key, log_pdgrm, spline_model, alpha_phi, beta_phi)

    # Return the samples (phi, w, etc.)
    samples = mcmc.get_samples()
    return samples
