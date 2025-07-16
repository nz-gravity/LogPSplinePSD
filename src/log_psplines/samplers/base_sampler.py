import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

from ..arviz_utils import _make_dataset_from_dict, get_weights
from ..plotting import plot_diagnostics, plot_pdgrm
from ..psplines import LogPSplines, Periodogram, build_spline


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
        az.InferenceData
            Object containing the sampling results and diagnostics
        """
        pass

    @abstractmethod
    def to_arviz(self, results: Any) -> az.InferenceData:
        pass

    def _add_common_attrs_and_save(
        self, idata: az.InferenceData
    ) -> az.InferenceData:
        idata.attrs["runtime"] = self.runtime
        idata.attrs["sampler"] = self.__class__.__name__
        for key, value in asdict(self.config).items():
            if isinstance(value, bool):
                value = int(value)
            idata.attrs[key] = value

        # Add spline model and configuration to InferenceData
        spline = self.spline_model
        spline_data = _make_dataset_from_dict(
            {
                "knots": ("knot", np.array(spline.knots)),
                "degree": int(spline.degree),
                "diffMatrixOrder": int(spline.diffMatrixOrder),
                "n": int(spline.n),
                "basis": (["obs", "basis_dim"], np.array(spline.basis)),
                "penalty_matrix": (
                    ["basis_dim", "basis_dim"],
                    np.array(spline.penalty_matrix),
                ),
                "parametric_model": ("obs", np.array(spline.parametric_model)),
            },
            coords={
                "knot": np.arange(len(spline.knots)),
                "obs": np.arange(spline.basis.shape[0]),
                "basis_dim": np.arange(spline.basis.shape[1]),
            },
        )
        idata.add_groups(spline_model=spline_data)
        idata.add_groups(
            periodogram=_make_dataset_from_dict(
                {"power": ("freq", self.periodogram.power)},
                {"freq": self.periodogram.freqs},
            )
        )

        # Summary statistics
        if self.config.verbose:
            print("Summary Statistics:")
            print(az.summary(idata))

        if self.config.outdir is not None:
            az.to_netcdf(idata, f"{self.config.outdir}/inference_data.nc")
            plot_diagnostics(idata, self.config.outdir)
            fig, _ = plot_pdgrm(
                self.periodogram,
                self.spline_model,
                get_weights(idata),
            )
            fig.savefig(f"{self.config.outdir}/posterior_predictive.png")

        return idata
