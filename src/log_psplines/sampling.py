import jax.numpy as jnp

from .psplines import LogPSplines


def lnlikelihood(
    lndata_log: jnp.ndarray, log_psplines: LogPSplines, weights: jnp.ndarray
) -> float:
    """Compute the Whittle log likelihood.

    Args:
        lndata_log: Log power spectral density data.
        log_psplines: Instance of LogPSplines.
        weights: Spline weights.

    Returns:
        The computed log likelihood.
    """
    lnmodel = log_psplines(weights)
    # Compute the likelihood term on the log-scale.
    integrand = lnmodel + jnp.exp(lndata_log - lnmodel - jnp.log(2 * jnp.pi))
    lnlike = -jnp.sum(integrand) / 2

    # If lnlike is not finite, return a very large negative value.
    lnlike = jnp.where(jnp.isfinite(lnlike), lnlike, -1e300)
    return lnlike
