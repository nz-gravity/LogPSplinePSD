"""Modified-Cholesky Wishart factors, omitting data-only constants."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from log_psplines.likelihoods.whittle import whittle_log_likelihood


def wishart_log_likelihood(
    log_variance: Float[Array | np.ndarray, "*batch N"],
    theta_re: Float[Array | np.ndarray, "*batch N _"],
    theta_im: Float[Array | np.ndarray, "*batch N _"],
    u_re: Float[Array | np.ndarray, "*batch N p"],
    u_im: Float[Array | np.ndarray, "*batch N p"],
    prev_re: Float[Array | np.ndarray, "*batch N _ p"],
    prev_im: Float[Array | np.ndarray, "*batch N _ p"],
    *,
    Nb: int = 1,
    Nh: int = 1,
    duration: float | int = 1.0,
    enbw: float | int = 1.0,
    eta: float | int = 1.0,
) -> Float[Array, ""]:
    """One channel factor; sum factors to obtain the matrix likelihood.

    log_variance: (..., F), theta: (..., F, preceding_channels),
    u: (..., F, replicates), prev: (..., F, preceding_channels, replicates).
    Zero preceding channels gives exactly the univariate Whittle likelihood.
    Inputs are sufficient statistics, independent of package data containers.
    """
    contrib_re = jnp.einsum(
        "...fl,...flr->...fr", theta_re, prev_re
    ) - jnp.einsum("...fl,...flr->...fr", theta_im, prev_im)
    contrib_im = jnp.einsum(
        "...fl,...flr->...fr", theta_re, prev_im
    ) + jnp.einsum("...fl,...flr->...fr", theta_im, prev_re)
    power = jnp.sum(
        (u_re - contrib_re) ** 2 + (u_im - contrib_im) ** 2, axis=-1
    )
    return whittle_log_likelihood(
        log_variance,
        power,
        count=jnp.asarray(Nb, dtype=log_variance.dtype)
        * jnp.asarray(Nh, dtype=log_variance.dtype),
        duration=duration,
        enbw=enbw,
        eta=eta,
    )
