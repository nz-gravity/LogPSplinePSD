"""
Base class for univariate PSD samplers.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import morphZ
import numpy as np
from xarray import DataArray, Dataset

from ...arviz_utils.to_arviz import results_to_arviz
from ...datatypes import Periodogram
from ...logger import logger
from ...plotting import (
    plot_pdgrm,
    save_vi_diagnostics_univariate,
)
from ...psplines import LogPSplines, build_spline
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
    freq_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Univariate log-likelihood function."""
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    sum_log_det = jnp.sum(freq_weights * ln_model)
    quad = jnp.sum(jnp.exp(log_pdgrm - ln_model))
    return -0.5 * (sum_log_det + quad)


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

        if self.config.verbose:
            basis_shape = tuple(self.basis_matrix.shape)
            logger.info(
                f"Frequency bins used for inference (N): {self.periodogram.n}"
            )
            logger.info(f"B-spline basis shape: {basis_shape}")

    @property
    def data_type(self) -> str:
        return "univariate"

    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save univariate-specific plots."""
        fig, _ = plot_pdgrm(idata=idata)
        fig.savefig(f"{self.config.outdir}/posterior_predictive.png")

        self._save_vi_diagnostics(log_summary=False)

    def _save_vi_diagnostics(self, *, log_summary: bool = True) -> None:
        """Persist VI diagnostics if available."""
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if vi_diag:
            save_vi_diagnostics_univariate(
                outdir=self.config.outdir,
                periodogram=self.periodogram,
                spline_model=self.spline_model,
                diagnostics=vi_diag,
            )
            try:
                from ...plotting import generate_vi_diagnostics_summary

                generate_vi_diagnostics_summary(
                    vi_diag,
                    outdir=self.config.outdir,
                    log=log_summary,
                )
            except Exception:
                logger.debug(
                    "Could not log VI diagnostics summary.", exc_info=True
                )

    def _get_lnz(
        self, samples: Dict[str, np.ndarray], sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Default implementation for univariate LnZ computation."""
        if not self.config.compute_lnz:
            return np.nan, np.nan

        # Combine all parameters into single posterior sample array
        weights = np.asarray(samples["weights"])
        if weights.ndim >= 3:
            weights = weights.reshape((-1, weights.shape[-1]))
        elif weights.ndim == 2 and weights.shape[0] == self.config.num_chains:
            weights = weights.reshape((-1, weights.shape[-1]))

        phi = np.asarray(samples["phi"])
        if phi.ndim >= 2:
            phi = phi.reshape(-1)

        delta = np.asarray(samples["delta"])
        if delta.ndim >= 2:
            delta = delta.reshape(-1)

        lp = np.asarray(sample_stats["lp"])
        if lp.ndim >= 2:
            lp = lp.reshape(-1)

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
            penalty_matrix=self.penalty_matrix,
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

    def _create_vi_inference_data(
        self,
        samples: Dict[str, jnp.ndarray],
        sample_stats: Dict[str, jnp.ndarray],
        diagnostics: Optional[Dict[str, Any]],
    ) -> az.InferenceData:
        """Convert VI samples to ArviZ and attach VI-specific diagnostics."""

        idata = self._create_inference_data(
            samples,
            sample_stats,
            lnz=np.nan,
            lnz_err=np.nan,
        )

        self._attach_vi_psd_group(idata, diagnostics)
        return idata

    def _attach_vi_psd_group(
        self, idata: az.InferenceData, diagnostics: Optional[Dict[str, Any]]
    ) -> None:
        """Store VI PSD quantiles in a dedicated ArviZ group."""

        if not diagnostics:
            return

        psd_quantiles = diagnostics.get("psd_quantiles")
        freq = np.asarray(self.periodogram.freqs, dtype=np.float32)
        if psd_quantiles:
            entries = []
            perc = []
            for label, percentile in [
                ("q05", 5.0),
                ("q50", 50.0),
                ("q95", 95.0),
            ]:
                value = psd_quantiles.get(label)
                if value is None:
                    continue
                entries.append(np.asarray(value))
                perc.append(percentile)
            if not entries:
                return
            psd_array = np.stack(entries, axis=0)
            percentiles = np.asarray(perc, dtype=np.float32)
        else:
            psd = diagnostics.get("psd")
            if psd is None:
                return
            psd_array = np.asarray(psd)[None, :]
            percentiles = np.asarray([50.0], dtype=np.float32)

        dataset = Dataset(
            {
                "psd": DataArray(
                    psd_array,
                    dims=["percentile", "freq"],
                    coords={
                        "percentile": percentiles,
                        "freq": freq,
                    },
                )
            }
        )

        idata.add_groups(vi_posterior_psd=dataset)
