from typing import Tuple, Union

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import gamma


def sample_gamma(
    alpha: float, rate: float, key: random.PRNGKey, shape: Union[Tuple, int]
) -> jnp.ndarray:
    return random.gamma(key, alpha, shape) * 1 / rate


def gamma_logpdf(
    x: jnp.ndarray,
    alpha: Union[float, jnp.array],
    beta: Union[float, jnp.array],
) -> jnp.ndarray:
    return gamma.logpdf(x, a=alpha, scale=1 / beta)
