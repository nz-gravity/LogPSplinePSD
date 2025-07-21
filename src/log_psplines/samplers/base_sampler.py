import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np


from ..plotting import plot_diagnostics, plot_pdgrm
from ..psplines import LogPSplines, Periodogram, build_spline
from ..arviz_utils.to_arviz import results_to_arviz


@jax.jit
def log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
) -> jnp.ndarray:
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    # Use log-sum-exp trick for numerical stability
    # Original: ln_model + exp(log_pdgrm - ln_model)
    # Rewrite as: ln_model + exp(log_pdgrm - ln_model) = exp(ln_model) + exp(log_pdgrm)
    # Then use logsumexp

    # Stack the two terms for logsumexp
    terms = jnp.stack([ln_model, log_pdgrm], axis=-1)
    integrand = jax.scipy.special.logsumexp(terms, axis=-1)

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
        if self.outdir is not None:
            os.makedirs(self.outdir, exist_ok=True)


class BaseSampler(ABC):

    def __init__(
        self,
        periodogram: Periodogram,
        spline_model: LogPSplines,
        config: SamplerConfig = None,
    ):
        self.periodogram = periodogram
        self.spline_model = spline_model
        self.config = config

        # Common attributes
        self.n_weights = len(spline_model.weights)

        # JAX arrays for mathematical operations
        self.log_pdgrm = jnp.log(periodogram.power)
        self.penalty_matrix = jnp.array(spline_model.penalty_matrix)
        self.basis_matrix = jnp.array(spline_model.basis)
        self.log_parametric = jnp.array(spline_model.log_parametric_model)

        # Random state
        self.rng_key = jax.random.PRNGKey(config.rng_key)

        # Runtime tracking
        self.runtime = np.nan

        # GPU/CPU device
        self.device = jax.devices()[0].platform

    @abstractmethod
    def sample(self, n_samples: int, n_warmup: int = 1000, **kwargs) -> az.InferenceData:
        pass


    def to_arviz(self, samples: Dict[str, np.ndarray], sample_stats:Dict[str, Any] ) -> az.InferenceData:
        idata = results_to_arviz(
            samples=samples,
            sample_stats=sample_stats,
            periodogram=self.periodogram,
            spline_model=self.spline_model,
            config=self.config,
            attributes=dict(
                device= str(self.device),
                runtime=self.runtime,
            )
        )

        # Summary statistics
        if self.config.verbose:
            print("Summary Statistics:")
            print(az.summary(idata))

        if self.config.outdir is not None:
            az.to_netcdf(idata, f"{self.config.outdir}/inference_data.nc")
            plot_diagnostics(idata, self.config.outdir)
            fig, _ = plot_pdgrm(idata=idata)
            fig.savefig(f"{self.config.outdir}/posterior_predictive.png")
            az.summary(idata).to_csv(f"{self.config.outdir}/summary_statistics.csv")

        return idata
