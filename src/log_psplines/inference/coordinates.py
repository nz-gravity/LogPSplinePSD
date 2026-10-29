"""Exact Gaussian spline priors in automatically scaled coordinates."""

import jax
import jax.numpy as jnp
import numpy as np


def coordinate_scales(prior_scales, information):
    """Interpolate between prior and fixed expected-likelihood scales.

    If s=c*z and z~Normal(0,a/c), then s~Normal(0,a) exactly, including
    when a depends on sampled smoothing parameters. Information is a fixed
    design/count approximation; it is never a fitted truth spectrum.
    """
    return jax.lax.rsqrt(prior_scales**-2 + information)


def spline_coordinates(basis, penalty, information_weight):
    """Prepare a proper stationary penalty's eigenbasis and Fisher diagonal."""
    penalty = np.asarray(penalty)
    np.linalg.cholesky(penalty)
    eigenvalues, rotation = np.linalg.eigh(penalty)
    design = np.asarray(basis) @ rotation
    information = np.sum(
        np.asarray(information_weight)[..., None] * design**2, axis=0
    )
    return {
        "rotation": jnp.asarray(rotation),
        "eigenvalues": jnp.asarray(eigenvalues),
        "information": jnp.asarray(information),
    }
