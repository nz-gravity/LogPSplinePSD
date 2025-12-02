"""
Abstract base classes for MCMC samplers.

Provides foundation for both univariate and multivariate PSD estimation samplers.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

from ..arviz_utils.rhat import extract_rhat_values
from ..arviz_utils.to_arviz import results_to_arviz
from ..logger import logger
from ..plotting import plot_diagnostics, plot_pdgrm


@dataclass
class SamplerConfig:
    """Base configuration for all MCMC samplers."""

    alpha_phi: float = 1.0
    beta_phi: float = 1.0
    alpha_delta: float = 1e-4
    beta_delta: float = 1e-4
    num_chains: int = 1
    rng_key: int = 42
    verbose: bool = True
    outdir: Optional[str] = None
    compute_lnz: bool = False
    scaling_factor: float = 1.0  # To track any data scaling
    channel_stds: Optional[np.ndarray] = (
        None  # Per-channel stds for multivariate scaling
    )
    true_psd: Optional[jnp.ndarray] = None  # True PSD for diagnostics
    freq_weights: Optional[np.ndarray] = None  # Optional frequency weights
    vi_psd_max_draws: int = (
        64  # Cap PSD reconstructions from VI/posterior draws
    )
    only_vi: bool = False  # Skip MCMC and rely on VI draws only

    def __post_init__(self):
        if self.outdir is not None:
            os.makedirs(self.outdir, exist_ok=True)


class BaseSampler(ABC):
    """
    Abstract base class for all MCMC samplers in LogPSplinePSD.

    Provides common interface and structure for both univariate and multivariate
    spectral estimation samplers.
    """

    def __init__(self, data, model, config: SamplerConfig):
        self.data = data
        self.model = model
        self.config = config

        # Common attributes for all samplers
        self.rng_key = jax.random.PRNGKey(config.rng_key)
        self.chain_method = self._select_chain_method()
        self.runtime = np.nan
        self.device = jax.devices()[0].platform

        # Setup data-specific attributes
        self._setup_data()

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        n_warmup: int = 1000,
        *,
        only_vi: bool = False,
        **kwargs,
    ) -> az.InferenceData:
        """Run MCMC sampling and return inference data."""
        pass

    @abstractmethod
    def _setup_data(self) -> None:
        """Setup data attributes for sampling (univar vs multivar specific)."""
        pass

    @abstractmethod
    def _get_lnz(
        self, samples: Dict[str, jnp.ndarray], sample_stats: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Extract log normalizing constant from samples."""
        pass

    @property
    @abstractmethod
    def sampler_type(self) -> str:
        """Return string identifier for the sampler type."""
        pass

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return string identifier for the data type."""
        pass

    def to_arviz(
        self, samples: Dict[str, jnp.ndarray], sample_stats: Dict[str, Any]
    ) -> az.InferenceData:
        """Convert samples to ArviZ InferenceData with diagnostics and plotting."""
        lnz, lnz_err = self._get_lnz(samples, sample_stats)

        # Call the appropriate results_to_arviz based on data type
        idata = self._create_inference_data(
            samples, sample_stats, lnz, lnz_err
        )
        logger.debug(" InferenceData created.")

        # Attach VI PSD summaries when available (from VI initialisation)
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if vi_diag and hasattr(self, "_attach_vi_psd_group"):
            try:
                self._attach_vi_psd_group(idata, vi_diag)
            except Exception as exc:  # pragma: no cover - best-effort hook
                logger.warning(f"Could not attach VI diagnostics: {exc}")

        # Summary statistics
        rhat_vals = None

        # Compute and store Rhat when multiple chains are available
        if (
            self.config.num_chains > 1
            and idata.posterior.sizes.get("chain", 1) > 1
        ):
            try:
                rhat_vals = extract_rhat_values(idata)
                if rhat_vals.size:
                    idata.attrs["rhat"] = rhat_vals
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(f"  Could not compute Rhat: {exc}")

        if self.config.verbose:
            if not (np.isnan(lnz) or np.isnan(lnz_err)):
                logger.info(f"  lnz: {lnz:.2f} ± {lnz_err:.2f}")

            if hasattr(idata.attrs, "get"):

                if "ess" in idata.attrs:
                    ess = idata.attrs["ess"]
                    logger.info(
                        f"  ESS min: {np.min(ess):.1f}, max: {np.max(ess):.1f}"
                    )

                if rhat_vals is not None and rhat_vals.size:
                    logger.info(
                        "  Rhat: min=%.3f, mean=%.3f, max=%.3f",
                        np.min(rhat_vals),
                        np.mean(rhat_vals),
                        np.max(rhat_vals),
                    )
                    logger.info(
                        "  Rhat ≤ 1.01: %.1f%%",
                        (rhat_vals <= 1.01).mean() * 100,
                    )

                if "riae" in idata.attrs:
                    # Univariate case
                    riae_median = idata.attrs["riae"]
                    errorbars = idata.attrs.get("riae_errorbars", [])
                    if len(errorbars) >= 5:
                        iqr_half = (errorbars[3] - errorbars[1]) / 2.0
                        logger.info(
                            f"  RIAE: {riae_median:.3f} ± {iqr_half:.3f}"
                        )
                    else:
                        logger.info(f"  RIAE: {riae_median:.3f}")

                # Check for multivariate RIAE
                if "riae_matrix" in idata.attrs:
                    riae_matrix = idata.attrs["riae_matrix"]
                    logger.info(f"  RIAE (matrix): {riae_matrix:.3f}")

        # Save outputs if requested
        if self.config.outdir is not None:
            logger.debug(" Saving results to disk...")
            self._save_results(idata)

        return idata

    def _create_inference_data(
        self,
        samples: Dict[str, jnp.ndarray],
        sample_stats: Dict[str, Any],
        lnz: float,
        lnz_err: float,
    ) -> az.InferenceData:
        """Create InferenceData object for both univar and multivar cases."""
        return results_to_arviz(
            samples=samples,
            sample_stats=sample_stats,
            data=self.data,
            model=self.model,
            config=self.config,
            attributes=dict(
                device=str(self.device),
                runtime=self.runtime,
                lnz=lnz,
                lnz_err=lnz_err,
                sampler_type=self.sampler_type,
                data_type=self.data_type,
            ),
        )

    def _save_results(self, idata: az.InferenceData) -> None:
        """Save inference results to disk."""
        az.to_netcdf(idata, f"{self.config.outdir}/inference_data.nc")
        plot_diagnostics(idata, self.config.outdir)
        az.summary(idata).to_csv(
            f"{self.config.outdir}/summary_statistics.csv"
        )

        # Data-type specific plotting
        self._save_plots(idata)

    @abstractmethod
    def _save_plots(self, idata: az.InferenceData) -> None:
        """Save data-type specific plots."""
        pass

    def _select_chain_method(self) -> str:
        """Choose an appropriate NumPyro chain_method based on hardware.

        NumPyro defaults to ``parallel`` for ``num_chains > 1`` which fails when
        only a single device is available. Fall back to ``vectorized`` in that
        case so multi-chain sampling works out of the box on CPUs/GPUs with a
        single device.
        """

        if self.config.num_chains <= 1:
            return "sequential"

        n_devs = len(jax.devices())
        if n_devs >= self.config.num_chains:
            logger.info(
                f"Running {self.config.num_chains} chains in parallel across {n_devs} device(s)."
            )
            return "parallel"

        logger.info(
            f"Running {self.config.num_chains} chains on {n_devs} device(s) with NumPyro vectorized chaining (no multiprocessing)."
        )
        if n_devs > 1 and self.config.num_chains > n_devs:
            logger.warning(
                "num_chains (%d) exceeds available JAX devices (%d); "
                "for true multiprocessing with chain_method='parallel', "
                "set num_chains <= number of devices.",
                self.config.num_chains,
                n_devs,
            )

        return "vectorized"
