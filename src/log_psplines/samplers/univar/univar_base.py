from __future__ import annotations

"""
Base class for univariate PSD samplers.
"""

import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import morphZ
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from ...arviz_utils.to_arviz import _pack_spline_model
from ...datatypes import Periodogram
from ...logger import logger
from ...plotting import plot_pdgrm
from ...psplines import LogPSplines, build_spline
from ..base_sampler import BaseSampler, SamplerConfig


@jax.jit
def log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
    Nh: int,
) -> jnp.ndarray:
    """Univariate log-likelihood function."""
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    nh = jnp.asarray(Nh, dtype=ln_model.dtype)
    sum_log_det = nh * jnp.sum(ln_model)
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
        if self.spline_model.weights is None:
            raise ValueError("spline_model.weights must be initialized.")
        self.n_weights = len(self.spline_model.weights)
        self.log_pdgrm = jnp.log(self.periodogram.power)
        self.log_pdgrm_np = np.asarray(self.log_pdgrm, dtype=np.float64)
        self.penalty_matrix = jnp.array(self.spline_model.penalty_matrix)
        self.basis_matrix = jnp.asarray(
            self.spline_model.basis, dtype=jnp.float32
        )
        self.basis_matrix_np = np.asarray(self.basis_matrix, dtype=np.float64)
        self.log_parametric = jnp.array(self.spline_model.log_parametric_model)
        self.log_parametric_np = np.asarray(
            self.log_parametric, dtype=np.float64
        )
        self.freq_coords = np.asarray(self.periodogram.freqs, dtype=np.float32)
        Nh = getattr(self.periodogram, "Nh", 1)
        if isinstance(Nh, bool) or not isinstance(Nh, (int, np.integer)):
            raise TypeError("periodogram.Nh must be a positive integer")
        self.Nh = int(Nh)
        if self.Nh <= 0:
            raise ValueError("periodogram.Nh must be positive")

        if self.config.verbose:
            basis_shape = tuple(self.basis_matrix.shape)
            logger.info(
                f"Frequency bins used for inference (N): {self.periodogram.n}"
            )
            logger.info(f"B-spline basis shape: {basis_shape}")

    @property
    def data_type(self) -> str:
        return "univariate"

    def _arviz_coords(self) -> Dict[str, Any]:
        return {
            "freq": self.freq_coords,
            "weight_dim": np.arange(self.n_weights),
        }

    def _arviz_dims(
        self, samples: Optional[Dict[str, Any]] = None
    ) -> Dict[str, list[str]]:
        return {"weights": ["weight_dim"]}

    def _attach_custom_groups(
        self,
        idata: xr.DataTree,
        *,
        samples: Dict[str, Any],
        sample_stats: Dict[str, Any],
    ) -> None:
        weights = np.asarray(samples["weights"], dtype=np.float64)
        observed_power = np.asarray(
            self.periodogram.power, dtype=np.float64
        ) * float(self.config.scaling_factor)
        log_psd = (
            np.einsum("fk,cdk->cdf", self.basis_matrix_np, weights)
            + self.log_parametric_np[None, None, :]
        )
        pointwise_log_likelihood = -0.5 * (
            float(self.Nh) * log_psd
            + np.exp(self.log_pdgrm_np[None, None, :] - log_psd)
        )
        freq_coords = {"freq": self.freq_coords}

        idata["observed_data"] = xr.DataTree(
            dataset=Dataset(
                {
                    "periodogram": DataArray(
                        observed_power,
                        dims=["freq"],
                        coords=freq_coords,
                    )
                }
            )
        )
        idata["log_likelihood"] = xr.DataTree(
            dataset=Dataset(
                {
                    "periodogram": DataArray(
                        pointwise_log_likelihood.astype(np.float32),
                        dims=["chain", "draw", "freq"],
                        coords={
                            "chain": np.arange(weights.shape[0]),
                            "draw": np.arange(weights.shape[1]),
                            "freq": self.freq_coords,
                        },
                    )
                }
            )
        )
        idata["spline_model"] = xr.DataTree(
            dataset=_pack_spline_model(self.spline_model)
        )

    def _save_plots(self, idata: xr.DataTree) -> None:
        """Save univariate-specific plots."""
        t0 = time.perf_counter()
        logger.info("save_plots(univar): plotting posterior predictive")
        fig, _ = plot_pdgrm(idata=idata)
        fig.savefig(f"{self.config.outdir}/posterior_predictive.png")
        logger.info(
            f"save_plots(univar): posterior predictive done in {time.perf_counter() - t0:.2f}s"
        )

        t0 = time.perf_counter()
        logger.info("save_plots(univar): saving VI diagnostics")
        self._save_vi_diagnostics(log_summary=False)
        logger.info(
            f"save_plots(univar): VI diagnostics done in {time.perf_counter() - t0:.2f}s"
        )

    def _get_lnz(
        self, samples: Dict[str, Any], sample_stats: Dict[str, Any]
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

    def _attach_vi_group(
        self, idata: xr.DataTree, diagnostics: Optional[Dict[str, Any]]
    ) -> None:
        """Store optional VI posterior draws and diagnostics."""

        if not diagnostics:
            return

        if bool(idata.attrs.get("only_vi")):
            return

        vi_samples = diagnostics.get("vi_samples")
        if vi_samples:
            sample_vars = {}
            coords: Dict[str, Any] = {"chain": np.arange(1)}
            for key, value in vi_samples.items():
                if key not in {"weights", "phi", "delta"}:
                    continue
                array = np.asarray(value)
                if array.ndim == 1:
                    array = (
                        array[None, :, None]
                        if key != "weights"
                        else array[None, ...]
                    )
                if key == "weights":
                    if array.ndim != 2:
                        continue
                    array = array[None, ...]
                    coords["weight_dim"] = np.arange(array.shape[-1])
                    sample_vars[key] = (
                        ["chain", "draw", "weight_dim"],
                        array,
                    )
                else:
                    sample_vars[key] = (
                        ["chain", "draw"],
                        array.reshape(1, -1),
                    )
            self._attach_vi_posterior_dataset(idata, sample_vars, coords)
            vi_log_likelihood = self._build_vi_log_likelihood_dataset(
                vi_samples
            )
            if vi_log_likelihood is not None:
                idata["vi_log_likelihood"] = xr.DataTree(
                    dataset=vi_log_likelihood
                )

        self._attach_vi_sample_stats(
            idata,
            diagnostics,
            attr_keys=(
                "guide",
                "riae",
                "coverage",
                "ci_width",
                "psis_khat_max",
                "psis_khat_status",
                "psis_khat_threshold",
            ),
        )

    def _build_vi_log_likelihood_dataset(
        self, vi_samples: Dict[str, Any]
    ) -> Optional[Dataset]:
        """Build pointwise VI log-likelihood arrays for univariate VI."""

        weights = vi_samples.get("weights")
        if weights is None:
            return None

        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] == 0:
            return None

        log_psd = (
            weights @ self.basis_matrix_np.T + self.log_parametric_np[None, :]
        )
        pointwise = -0.5 * (
            float(self.Nh) * log_psd
            + np.exp(self.log_pdgrm_np[None, :] - log_psd)
        )
        return Dataset(
            {
                "log_likelihood": DataArray(
                    pointwise[None, ...].astype(np.float32),
                    dims=["chain", "draw", "freq"],
                    coords={
                        "chain": np.arange(1),
                        "draw": np.arange(weights.shape[0]),
                        "freq": self.freq_coords,
                    },
                )
            }
        )
