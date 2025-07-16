"""
Sampler class hierarchy for log P-splines spectral density estimation.

Provides a clean object-oriented interface with a base Sampler class and
specialized subclasses for different sampling algorithms.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp

from ..plotting.diagnostics import plot_diagnostics
from ..psplines import LogPSplines, build_spline


@jax.jit
def log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
) -> jnp.ndarray:
    ln_spline = build_spline(basis_matrix, weights)
    ln_model = ln_spline + log_parametric
    integrand = ln_model + jnp.exp(log_pdgrm - ln_model)
    return -0.5 * jnp.sum(integrand)


@dataclass
class SamplerConfig:
    """Base configuration for all samplers."""

    alpha_phi: float = 1.0
    beta_phi: float = 1.0
    alpha_delta: float = 1e-4
    beta_delta: float = 1e-4
    rng_key: int = 42
    verbose: bool = True
    outdir: str = None

    def __post_init__(self):
        os.makedirs(self.outdir, exist_ok=True)


class BaseSampler(ABC):

    def __init__(
        self,
        log_pdgrm: jnp.ndarray,
        spline_model: LogPSplines,
        config: SamplerConfig = None,
    ):
        """
        Initialize base sampler.

        Parameters
        ----------
        log_pdgrm : jnp.ndarray
            Log periodogram data
        spline_model : LogPSplines
            Spline model object
        config : SamplerConfig
            Sampler configuration
        """
        self.log_pdgrm = log_pdgrm
        self.spline_model = spline_model
        self.config = config

        # Common attributes
        self.n_weights = len(spline_model.weights)

        # JAX arrays for mathematical operations
        self.penalty_matrix = jnp.array(spline_model.penalty_matrix)
        self.basis_matrix = jnp.array(spline_model.basis)
        self.log_parametric = jnp.array(spline_model.log_parametric_model)

        # Random state
        self.rng_key = jax.random.PRNGKey(config.rng_key)

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        n_warmup: int = 1000,
        thin: int = 1,
        chains: int = 1,
        **kwargs,
    ) -> az.InferenceData:
        """
        Run MCMC sampling. Must be implemented by subclasses.

        Parameters
        ----------
        n_samples : int
            Number of samples to collect
        n_warmup : int
            Number of warmup iterations
        thin : int
            Thinning interval
        chains : int
            Number of chains
        **kwargs
            Additional sampler-specific arguments

        Returns
        -------
        dict
            Dictionary containing samples and diagnostics
        """
        pass

    @abstractmethod
    def to_arviz(self, results: Any) -> az.InferenceData:
        pass

    def plot_diagnostics(
        self,
        idata: az.InferenceData,
        outdir: str = None,
        variables: list = ["phi", "delta", "weights"],
        figsize: tuple = (12, 8),
    ) -> None:
        if outdir is None:
            outdir = self.config.outdir

        plot_diagnostics(idata, outdir, variables=variables, figsize=figsize)
