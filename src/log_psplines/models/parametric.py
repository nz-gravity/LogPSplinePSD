"""A deterministic positive spectrum and its independent parameter priors."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpyro.distributions import Distribution


@dataclass(frozen=True)
class ParametricSpectrum:
    """Supply physics without writing a Bayesian model or sampler.

    ``spectrum(parameters, frequency_slice)`` must be JAX differentiable and
    return variances matching PowerData: (T, F) or (T, F, channel). Slicing
    bounds posterior reconstruction memory. Parameters use scalar independent
    NumPyro priors; transforms such as exponentiation belong in ``spectrum``.
    The supplied spectrum must already include any response projection and
    likelihood pooling. No automatic reference or partition is applied.
    """

    spectrum: Callable
    priors: Mapping[str, Distribution]
    initial_values: Mapping[str, float]

    def __post_init__(self) -> None:
        if not callable(self.spectrum) or not self.priors:
            raise ValueError(
                "a spectrum callable and parameter priors are required"
            )
        if set(self.priors) != set(self.initial_values):
            raise ValueError(
                "initial_values must name every prior exactly once"
            )
        for name, prior in self.priors.items():
            if (
                not isinstance(prior, Distribution)
                or prior.batch_shape
                or prior.event_shape
            ):
                raise ValueError(f"{name} must have a scalar prior")
            value = self.initial_values[name]
            if not np.isfinite(value) or not np.isfinite(
                prior.log_prob(value)
            ):
                raise ValueError(f"invalid initial value for {name}")
