"""Modified-Cholesky Wishart factors, omitting data-only constants."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Real

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
    counts: int | float | Real[Array | np.ndarray, "..."] | None = None,
    clip_log_psd: bool = True,
) -> Float[Array, ""]:
    """One channel factor; sum factors to obtain the matrix likelihood.

    log_variance: (..., F), theta: (..., F, preceding_channels),
    u: (..., F, replicates), prev: (..., F, preceding_channels, replicates).
    counts overrides Nb*Nh; factors contain sums, so it weights only logdet.
    The last factor axis is not the observation count. Zero-count cells are
    neutralized before forming residuals. Zero preceding channels gives the
    univariate complex Whittle likelihood under matched scaling.
    Inputs are sufficient statistics, independent of package data containers.
    """
    count = jnp.asarray(Nb * Nh if counts is None else counts)
    active = count > 0
    theta_re = jnp.where(active[..., None], theta_re, 0.0)
    theta_im = jnp.where(active[..., None], theta_im, 0.0)
    u_re = jnp.where(active[..., None], u_re, 0.0)
    u_im = jnp.where(active[..., None], u_im, 0.0)
    prev_re = jnp.where(active[..., None, None], prev_re, 0.0)
    prev_im = jnp.where(active[..., None, None], prev_im, 0.0)
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
        count=count,
        duration=duration,
        enbw=enbw,
        eta=eta,
        clip_log_psd=clip_log_psd,
    )
