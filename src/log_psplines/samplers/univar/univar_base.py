"""
Base class for univariate PSD samplers.
"""

import tempfile
from typing import Any, Dict, Tuple

import arviz as az
import jax.numpy as jnp
import morphZ
import numpy as np
import jax

from ...arviz_utils.to_arviz import results_to_arviz
from ...datatypes import Periodogram
from ...plotting import plot_pdgrm
from ...psplines import LogPSplines, build_spline
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
) -> jnp.ndarray:
    """Univariate log-likelihood function."""
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    integrand = ln_model + jnp.exp(log_pdgrm - ln_model)
    return -0.5 * jnp.sum(integrand)


class UnivarBaseSampler(BaseSampler):
    """
    Base class for univariate PSD samplers.

    Handles single-channel periodogram data with LogPSplines models.
    """

    def __init__(
        self,
        periodogram: Periodogram,
        spline_model: LogPSplines,
        config: SamplerConfig
    ):
        # Type hints for clarity
        self.periodogram: Periodogram = periodogram
        self.spline_model: LogPSplines = spline_model

        super().__init__(periodogram, spline_model, config)

    def _setup_data(self) -> None:
        """Setup univariate-specific data attributes."""
        self.n_weights = len(self.spline_model.weights)
        self.log_pdgrm = jnp.log(self.periodogram.power)
        self.penalty_matrix = jnp.array(self.spline_model.penalty_matrix)
        self.basis_matrix = jnp.array(self.spline_model.basis)
        self.log_parametric = jnp.array(self.spline_model.log_parametric_model)

    @property
    def data_type(self) -> str:
        return "univariate"



    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save univariate-specific plots."""
        fig, _ = plot_pdgrm(idata=idata)
        fig.savefig(f"{self.config.outdir}/posterior_predictive.png")

    def _get_lnz(
        self,
        samples: Dict[str, np.ndarray],
        sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Default implementation for univariate LnZ computation."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        # Combine all parameters into single posterior sample array
        post_smp = np.concatenate([
            samples['weights'],
            samples['phi'][:, None],
            samples['delta'][:, None]
        ], axis=1)
        lp = sample_stats['lp']

        def lp_fn(sample):
            weights = sample[:self.n_weights]
            phi = sample[self.n_weights]
            delta = sample[self.n_weights + 1]
            return self._compute_log_posterior(weights, phi, delta)

        lnz_res = morphZ.evidence(
            post_smp, lp, lp_fn, output_path=tempfile.gettempdir()
        )[0]
        return float(lnz_res[0]), float(lnz_res[1])

    @property
    def _logp_kwargs(self) -> Dict[str, Any]:
        """Common log posterior kwargs for univariate case."""
        return dict(
            log_pdgrm=self.log_pdgrm,
            basis_matrix=self.basis_matrix,
            log_parametric=self.log_parametric,
            penalty_matrix=self.penalty_matrix,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )

    def _compute_log_posterior(self, weights: jnp.ndarray, phi: float, delta: float) -> float:
        """Compute log posterior for LnZ calculation. To be implemented by concrete samplers."""
        raise NotImplementedError("Concrete sampler must implement _compute_log_posterior")
