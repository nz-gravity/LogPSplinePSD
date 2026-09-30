"""Whittle likelihood without data-only constants."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Real

# Preserve the historical inference overflow guard.
LOG_SPECTRUM_LIMIT = 80.0


def power_whittle_log_likelihood(
    summed_power: Float[Array | np.ndarray, "*grid"],
    counts: Real[Array | np.ndarray, "*grid"],
    log_psd: Float[Array | np.ndarray, "*grid"],
) -> Float[Array, ""]:
    """Power/count likelihood for scalar time-frequency surfaces.

    Inputs broadcast to (F,) or (T, F). Counts are numbers of independent
    real Gaussian components, not complex Fourier ordinates. A WDM cell
    has P=w**2, count=1. A complex Fourier ordinate has P=2*power/duration,
    count=2. Pool raw powers and counts, never a smoothed pilot spectrum.
    Omitted cells have both power and count zero, with finite log_psd.

    This likelihood is unclipped and omits data-only constants. The stationary
    Fourier entry point below keeps its existing clipping, ENBW and tempering
    conventions.
    """
    return -0.5 * jnp.sum(counts * log_psd + summed_power * jnp.exp(-log_psd))


def whittle_log_likelihood(
    log_psd: Float[Array | np.ndarray, "*grid"],
    power: Float[Array | np.ndarray, "*grid"],
    *,
    count: float | int | Real[Array | np.ndarray, "..."] = 1.0,
    duration: float | int = 1.0,
    enbw: float | int = 1.0,
    eta: float | int = 1.0,
    clip_log_psd: bool = True,
) -> Float[Array, ""]:
    """Scalar log spectrum and summed power (..., F); return total log L.

    power is summed, not averaged, over `count` independent replicates.
    duration converts FFT power to spectral density. Window bandwidth and
    likelihood tempering follow the existing Wishart convention. Array counts
    weight each cell. Zero-count cells have zero value and gradient, even for
    nonfinite omitted inputs. Normalized GridTV coefficients use duration=1,
    enbw=1, eta=1 and disable the historical stationary clipping guard.
    """
    count = jnp.asarray(count)
    active = count > 0
    logs = jnp.where(active, log_psd, 0.0)
    if clip_log_psd:
        logs = jnp.clip(logs, -LOG_SPECTRUM_LIMIT, LOG_SPECTRUM_LIMIT)
    variance = jnp.exp(logs)
    determinant = -jnp.sum(count * logs)
    quadratic = jnp.sum(
        jnp.where(active, power, 0.0)
        / (jnp.asarray(duration, dtype=variance.dtype) * variance)
    )
    return (
        (determinant - quadratic)
        / jnp.asarray(enbw, dtype=variance.dtype)
        * jnp.asarray(eta, dtype=variance.dtype)
    )
