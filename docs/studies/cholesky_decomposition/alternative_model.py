import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax
import jax.scipy.linalg as linalg
from log_psplines.bayesian_model import whittle_lnlike, build_spline


def bayesian_model(
        log_pdgrm: jnp.ndarray,  # shape (Nfreq,) - log of the observed periodogram
        lnspline_basis: jnp.ndarray,  # shape (kbasis, Nfreq,) - matrix of spline basis functions
        penalty_matrix: jnp.ndarray,  # shape (kbasis, kbasis) - penalty matrix P
        L: jnp.ndarray,  # Cholesky factor of the penalty matrix (L=cholesky(penalty_matrix, lower=True))
        alpha_phi, beta_phi,  # for phi | delta: Gamma(alpha_phi, delta * beta_phi)
        alpha_delta, beta_delta,  # for delta: Gamma(alpha_delta, beta_delta)
):
    """
    Bayesian model using a reparameterized spline prior with Cholesky decomposition.

    The model specifies:
      - delta ~ Gamma(alpha_delta, beta_delta)
      - phi   ~ Gamma(alpha_phi, delta * beta_phi)
      - w is defined implicitly via tilde_w ~ Normal(0, I) and the transformation
           w = L^{-1} tilde_w / sqrt(phi)
        so that w ~ N(0, (phi P)^{-1})
      - The likelihood is given by the Whittle likelihood.
    """
    # 1) Sample delta and phi.
    delta = numpyro.sample("delta", dist.Gamma(concentration=alpha_delta, rate=beta_delta))
    phi = numpyro.sample("phi", dist.Gamma(concentration=alpha_phi, rate=delta * beta_phi))

    # 2) Reparameterize the spline weights.
    k = penalty_matrix.shape[0]

    # Sample tilde_w ~ Normal(0, I) in k dimensions.
    tilde_w = numpyro.sample("tilde_w", dist.Normal(jnp.zeros(k), jnp.ones(k)).to_event(1))

    # Transform tilde_w to obtain w:
    #    w = L^{-1} tilde_w / sqrt(phi)
    # We use solve_triangular to avoid explicit matrix inversion.
    w = linalg.solve_triangular(L, tilde_w, lower=True) * jnp.sqrt(phi)
    # save the weights for later inspection
    numpyro.deterministic("weights", w)

    # 3) Build the log-spline as a linear combination of the basis functions.
    #    lnspline_basis has shape (kbasis, Nfreq), so we take the weighted sum over the kbasis dimension.
    ln_spline = build_spline(lnspline_basis, w)

    # 4) Add the Whittle likelihood.
    numpyro.factor("ln_likelihood", whittle_lnlike(log_pdgrm, ln_spline))
