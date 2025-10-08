"""
Base class for univariate PSD samplers.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import morphZ
import numpy as np

from ...arviz_utils.to_arviz import results_to_arviz
from ...datatypes import Periodogram
from ...plotting import (
    plot_pdgrm,
    save_vi_diagnostics_univariate,
)
from ...psplines import LogPSplines, build_spline
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def whittle_log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
    freq_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Whittle likelihood on the (possibly coarse) frequency grid."""
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    integrand = ln_model + jnp.exp(log_pdgrm - ln_model)
    return -0.5 * jnp.sum(freq_weights * integrand)


@jax.jit
def gaussian_log_likelihood(
    weights: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
    observed_power: jnp.ndarray,
    sigma_obs: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian likelihood using coarse-grained power and per-bin uncertainties."""
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    model_power = jnp.exp(ln_model)
    sigma = jnp.maximum(sigma_obs, 1e-30)
    resid = (observed_power - model_power) / sigma
    log_det = 2.0 * jnp.log(sigma)
    return -0.5 * jnp.sum(resid**2 + log_det + jnp.log(2.0 * jnp.pi))


class UnivarBaseSampler(BaseSampler):
    """
    Base class for univariate PSD samplers.

    Handles single-channel periodogram data with LogPSplines models.
    """

    def __init__(
        self,
        periodogram: Periodogram,
        spline_model: LogPSplines,
        config: SamplerConfig,
    ):
        # Always ensure periodogram is the correct (standardized) one with scaling_factor
        self.periodogram: Periodogram = periodogram
        self.spline_model: LogPSplines = spline_model
        super().__init__(periodogram, spline_model, config)

    def _setup_data(self) -> None:
        """Setup univariate-specific data attributes."""
        self.n_weights = len(self.spline_model.weights)
        self.log_pdgrm = jnp.log(self.periodogram.power)
        self.penalty_matrix = jnp.array(self.spline_model.penalty_matrix)
        self.basis_matrix = jnp.asarray(
            self.spline_model.basis, dtype=jnp.float32
        )
        self.log_parametric = jnp.array(self.spline_model.log_parametric_model)
        if self.config.freq_weights is not None:
            freq_weights = jnp.asarray(
                self.config.freq_weights,
                dtype=self.log_pdgrm.dtype,
            )
            if freq_weights.shape[0] != self.log_pdgrm.shape[0]:
                raise ValueError(
                    "Frequency weights must match periodogram length"
                )
            self.freq_weights = freq_weights
        else:
            self.freq_weights = jnp.ones_like(self.log_pdgrm)
        self.observed_power = jnp.asarray(
            self.periodogram.power,
            dtype=self.log_pdgrm.dtype,
        )
        if self.config.coarse_sigma is not None:
            sigma = jnp.asarray(
                self.config.coarse_sigma,
                dtype=self.log_pdgrm.dtype,
            )
            if sigma.shape[0] != self.observed_power.shape[0]:
                raise ValueError("Coarse sigma must match periodogram length")
            self.sigma_obs = jnp.maximum(sigma, 1e-30)
            self._use_coarse_gaussian = True
        else:
            self.sigma_obs = jnp.maximum(self.observed_power, 1e-30)
            self._use_coarse_gaussian = False

    @property
    def data_type(self) -> str:
        return "univariate"

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save univariate-specific plots."""
        fig, _ = plot_pdgrm(idata=idata)
        fig.savefig(f"{self.config.outdir}/posterior_predictive.png")

        self._save_vi_diagnostics()

    def _save_vi_diagnostics(self) -> None:
        """Persist VI diagnostics if available."""
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if vi_diag:
            save_vi_diagnostics_univariate(
                outdir=self.config.outdir,
                periodogram=self.periodogram,
                spline_model=self.spline_model,
                diagnostics=vi_diag,
            )

    def _get_lnz(
        self, samples: Dict[str, np.ndarray], sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Default implementation for univariate LnZ computation."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        # Combine all parameters into single posterior sample array
        weights = np.asarray(samples["weights"])
        phi = np.asarray(samples["phi"])
        delta = np.asarray(samples["delta"])
        lp = np.asarray(sample_stats["lp"])

        post_smp = np.concatenate(
            [weights, phi[:, None], delta[:, None]],
            axis=1,
        )

        def lp_fn(sample):
            weights = sample[: self.n_weights]
            phi = sample[self.n_weights]
            delta = sample[self.n_weights + 1]
            return self._compute_log_posterior(weights, phi, delta)

        lnz_res = morphZ.evidence(
            post_smp,
            lp,
            lp_fn,
            output_path=tempfile.gettempdir(),
            kde_bw="scott",
        )[0]
        return float(lnz_res[0]), float(lnz_res[1])

    @property
    def _logp_kwargs(self) -> Dict[str, Any]:
        """Common log posterior kwargs for univariate case."""
        return dict(
            log_pdgrm=self.log_pdgrm,
            basis_matrix=self.basis_matrix,
            log_parametric=self.log_parametric,
            observed_power=self.observed_power,
            sigma_obs=self.sigma_obs,
            penalty_matrix=self.penalty_matrix,
            freq_weights=self.freq_weights,
            alpha_phi=self.config.alpha_phi,
            beta_phi=self.config.beta_phi,
            alpha_delta=self.config.alpha_delta,
            beta_delta=self.config.beta_delta,
        )

    def _compute_log_posterior(
        self, weights: jnp.ndarray, phi: float, delta: float
    ) -> float:
        """Compute log posterior for LnZ calculation. To be implemented by concrete samplers."""
        raise NotImplementedError(
            "Concrete sampler must implement _compute_log_posterior"
        )
