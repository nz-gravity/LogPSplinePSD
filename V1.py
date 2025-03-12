import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def psd_from_v(v, tau, B):
    """
    Given spline coefficients v, return the spectral density f at the frequencies.
    Here we assume a log-link: log f = B @ v, so that
        f = tau * exp(B @ v).
    The factor tau (which may control the scale) is multiplied in.
    Adjust this function as needed.
    """
    log_f = jnp.dot(B, v)
    f = tau * jnp.exp(log_f)
    return f


def whittle_loglike(f, periodogram):
    """
    Computes the Whittle log-likelihood.
    One common version is
       - sum( log f + I_n / f )
    where f is the spectral density and I_n is the periodogram.
    """
    return -jnp.sum(jnp.log(f) + periodogram / f)


def model(
    periodogram,
    B,
    P,
    S,
    mean_v,
    I_n,
    nu,
    alpha_phi,
    beta_phi,
    alpha_delta,
    beta_delta,
    alpha_tau,
    beta_tau,
):
    """
    NumPyro model for the spline-based Whittle likelihood.

    Parameters:
      periodogram : jnp.ndarray
          The observed periodogram values, I_n, at the frequency grid.
      B : jnp.ndarray
          B-spline basis matrix (n_frequencies x (K-1)).
      P : jnp.ndarray
          Penalty matrix for the spline coefficients v.
      S : jnp.ndarray
          Covariance matrix from the pilot sample (for reparameterization of v).
      mean_v : jnp.ndarray
          Mean vector for v from the pilot sample.
      I_n : jnp.ndarray
          (Possibly the same as periodogram; here used in the τ full conditional).
      nu : int
          Degrees of freedom or number of frequency bins (used in τ).
      alpha_phi, beta_phi : float
          Hyperparameters in the φ full conditional.
      alpha_delta, beta_delta : float
          Hyperparameters in the δ full conditional.
      alpha_tau, beta_tau : float
          Hyperparameters in the τ full conditional.
    """
    # Number of spline coefficients: note that v is (K-1)-dimensional.
    K_minus_1 = mean_v.shape[0]

    # === Reparameterization for v ===
    # We sample β ~ N(0, I) and then transform v = S^(1/2) β + mean_v.
    beta = numpyro.sample(
        "beta", dist.Normal(jnp.zeros(K_minus_1), jnp.ones(K_minus_1))
    )
    chol_S = jnp.linalg.cholesky(S)
    v = jnp.dot(chol_S, beta) + mean_v  # spline coefficients

    # === Full conditional for δ ===
    # δ | ... ~ Gamma(α_φ + α_δ, β_φ φ + β_δ)
    # Note that φ has not been sampled yet, but we can sample δ first if its prior is independent.
    # In our model we assume a prior for δ; here we use a Gamma with shape (α_φ + α_δ)
    # and a rate parameter (we use β_delta as provided). You might modify this if δ appears in φ.
    delta = numpyro.sample(
        "delta", dist.Gamma(alpha_phi + alpha_delta, beta_delta)
    )

    # === Full conditional for φ ===
    # φ | ... ~ Gamma((K-1)/2 + α_φ, 0.5 * (v^T P v) + δ β_φ)
    vTPv = jnp.dot(v, jnp.dot(P, v))
    phi = numpyro.sample(
        "phi",
        dist.Gamma((K_minus_1) / 2 + alpha_phi, 0.5 * vTPv + delta * beta_phi),
    )

    # === Full conditional for τ ===
    # τ | ... ~ Inverse Gamma(α_τ + ν, sum_{l=1}^ν (I_n(λ_l)/s_r(λ_l/π)) + β_τ)
    # In this example, we approximate the sum by jnp.sum(I_n).
    tau = numpyro.sample(
        "tau", dist.InverseGamma(alpha_tau + nu, jnp.sum(I_n) + beta_tau)
    )

    # === Whittle Likelihood ===
    # Compute the spectral density f from v and τ.
    f_est = psd_from_v(v, tau, B)

    # Add the likelihood as a log-factor.
    # (Note: In NumPyro you can incorporate a custom likelihood via numpyro.factor.)
    ll = whittle_loglike(f_est, periodogram)
    numpyro.factor("whittle", ll)
