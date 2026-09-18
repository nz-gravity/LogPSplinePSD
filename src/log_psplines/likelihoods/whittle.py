"""Whittle likelihood without data-only constants."""

import jax.numpy as jnp
from jax import Array

# Preserve the historical inference overflow guard.
LOG_SPECTRUM_LIMIT = 80.0


def whittle_log_likelihood(
    log_psd: Array,
    power: Array,
    *,
    count: float = 1.0,
    duration: float = 1.0,
    enbw: float = 1.0,
    eta: float = 1.0,
) -> Array:
    """Scalar log spectrum and summed power (..., F); return total log L.

    power is summed, not averaged, over `count` independent replicates.
    duration converts FFT power to spectral density. Window bandwidth and
    likelihood tempering follow the existing Wishart convention.
    """
    variance = jnp.exp(
        jnp.clip(log_psd, -LOG_SPECTRUM_LIMIT, LOG_SPECTRUM_LIMIT)
    )
    determinant = -jnp.asarray(count, dtype=variance.dtype) * jnp.sum(
        jnp.log(variance)
    )
    quadratic = jnp.sum(
        power / (jnp.asarray(duration, dtype=variance.dtype) * variance)
    )
    return (
        (determinant - quadratic)
        / jnp.asarray(enbw, dtype=variance.dtype)
        * jnp.asarray(eta, dtype=variance.dtype)
    )
