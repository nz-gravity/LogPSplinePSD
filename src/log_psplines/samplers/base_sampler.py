"""
Abstract base classes for MCMC samplers.

Provides foundation for both univariate and multivariate PSD estimation samplers.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, Literal, Optional, Tuple, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

from ..arviz_utils.rhat import extract_rhat_values
from ..arviz_utils.to_arviz import results_to_arviz
from ..diagnostics import run_all_diagnostics
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
    chain_method: Optional[Literal["parallel", "vectorized", "sequential"]] = (
        None
    )
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
    # Fast-diagnostics controls
    posterior_psd_max_draws: int = 50  # Cap posterior PSD reconstructions
    compute_coherence_quantiles: bool = True  # Compute coherence bands
    compute_psis: bool = True  # Enable PSIS-LOO diagnostics
    skip_plot_diagnostics: bool = False  # Skip plotting heavy diagnostics
    diagnostics_summary_mode: Literal["off", "light", "full"] = "light"
    diagnostics_summary_position: Literal["start", "end"] = "end"
    # Optional extra empirical overlays for multivariate PSD matrix plots.
    # Intended for things like full-resolution Welch estimates alongside
    # coarse-grained empirical periodograms.
    extra_empirical_psd: Optional[list[Any]] = None
    extra_empirical_labels: Optional[list[str]] = None
    extra_empirical_styles: Optional[list[Dict[str, Any]]] = None

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
        if config.chain_method is not None:
            allowed = {"parallel", "vectorized", "sequential"}
            if config.chain_method not in allowed:
                raise ValueError(
                    f"Unknown chain_method='{config.chain_method}'. "
                    f"Expected one of {sorted(allowed)}."
                )
            self.chain_method = config.chain_method
            logger.info(
                f"Using requested NumPyro chain_method='{self.chain_method}'."
            )
        else:
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
        idata = self._create_inference_data(
            samples, sample_stats, lnz, lnz_err
        )
        logger.debug(" InferenceData created.")
        self._attach_vi_group_safe(idata)
        rhat_vals = self._compute_chain_summaries(idata)
        self._cache_full_diagnostics(idata)
        self._log_summary_metrics(idata, lnz, lnz_err, rhat_vals)
        self._maybe_save_outputs(idata)

        return idata

    def _cache_full_diagnostics(self, idata: az.InferenceData) -> None:
        """Compute scalar diagnostics and store them in ``idata.attrs``.

        This runs regardless of whether ``outdir`` is set. Any diagnostic plots
        remain gated by the diagnostic modules via ``config.outdir``.
        """

        attrs = getattr(idata, "attrs", None)
        if attrs is None or not hasattr(attrs, "__setitem__"):
            return

        true_psd = getattr(self.config, "true_psd", None)
        idata_vi = getattr(self, "_vi_diagnostics", None)

        try:
            results = run_all_diagnostics(
                idata=idata,
                config=self.config,
                psd_ref=true_psd,
                idata_vi=idata_vi,
            )
        except Exception as exc:  # pragma: no cover - best-effort hook
            if getattr(self.config, "verbose", False):
                logger.warning(f"Full diagnostics computation failed: {exc}")
            return

        canonical_keys = {
            "riae",
            "riae_matrix",
            "coverage",
            "coherence_riae",
            "riae_diag_mean",
            "riae_diag_max",
            "riae_offdiag",
        }

        for module, metrics in (results or {}).items():
            if not metrics:
                continue
            for key, val in metrics.items():
                try:
                    fval = float(val)
                except Exception:
                    continue
                if not np.isfinite(fval):
                    continue

                out_key = key if key in canonical_keys else f"{module}_{key}"

                if out_key in canonical_keys and out_key in attrs:
                    try:
                        existing = float(attrs.get(out_key))
                        if np.isfinite(existing):
                            continue
                    except Exception:
                        pass

                attrs[out_key] = fval

        attrs["full_diagnostics_computed"] = 1
        attrs["full_diagnostics_timestamp"] = datetime.now(UTC).isoformat()

    def _attach_vi_group_safe(self, idata: az.InferenceData) -> None:
        """Attach optional VI diagnostics to the InferenceData object."""
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if not vi_diag or not hasattr(self, "_attach_vi_psd_group"):
            return
        try:
            self._attach_vi_psd_group(idata, vi_diag)
        except Exception as exc:  # pragma: no cover - best-effort hook
            logger.warning(f"Could not attach VI diagnostics: {exc}")

    def _compute_chain_summaries(
        self, idata: az.InferenceData
    ) -> Optional[np.ndarray]:
        """Compute optional per-parameter chain summaries such as R-hat."""
        multi_chain = (
            self.config.num_chains > 1
            and idata.posterior.sizes.get("chain", 1) > 1
        )
        if not multi_chain:
            return None
        try:
            rhat_vals = extract_rhat_values(idata)
            if rhat_vals.size:
                idata.attrs["rhat"] = rhat_vals
            return rhat_vals
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(f"  Could not compute Rhat: {exc}")
            return None

    def _log_summary_metrics(
        self,
        idata: az.InferenceData,
        lnz: float,
        lnz_err: float,
        rhat_vals: Optional[np.ndarray],
    ) -> None:
        """Log human-readable summary metrics for quick terminal inspection."""
        if not self.config.verbose:
            return
        if not hasattr(idata.attrs, "get"):
            return

        attrs = idata.attrs
        if not (np.isnan(lnz) or np.isnan(lnz_err)):
            logger.info(f"  lnz: {lnz:.2f} ± {lnz_err:.2f}")

        ess = attrs.get("ess")
        if ess is not None:
            logger.info(
                f"  ESS min: {np.min(ess):.1f}, max: {np.max(ess):.1f}"
            )
            names = attrs.get("ess_lowest_names")
            vals = attrs.get("ess_lowest_values")
            if names is not None and vals is not None:
                if isinstance(names, np.ndarray):
                    names_list = names.tolist()
                elif isinstance(names, (list, tuple)):
                    names_list = list(names)
                else:
                    names_list = [str(names)]

                if isinstance(vals, np.ndarray):
                    vals_list = vals.tolist()
                elif isinstance(vals, (list, tuple)):
                    vals_list = list(vals)
                else:
                    vals_list = [vals]

                if len(names_list) == len(vals_list) and len(names_list) > 0:
                    try:
                        pairs = ", ".join(
                            f"{n}: {float(v):.1f}"
                            for n, v in zip(names_list, vals_list)
                        )
                        logger.info(f"  ESS lowest: {pairs}")
                    except Exception as exc:  # pragma: no cover
                        logger.debug(
                            f"  Could not format ESS lowest values: {exc}"
                        )

        if rhat_vals is not None and rhat_vals.size:
            logger.info(
                f"  Rhat: min={np.min(rhat_vals):.3f}, mean={np.mean(rhat_vals):.3f}, max={np.max(rhat_vals):.3f}"
            )
            logger.info(
                f"  Rhat ≤ 1.01: {(rhat_vals <= 1.01).mean() * 100:.1f}%"
            )

        riae = attrs.get("riae")
        if riae is not None:
            errorbars = attrs.get("riae_errorbars", [])
            if len(errorbars) >= 5:
                iqr_half = (errorbars[3] - errorbars[1]) / 2.0
                logger.info(f"  RIAE: {riae:.3f} ± {iqr_half:.3f}")
            else:
                logger.info(f"  RIAE: {riae:.3f}")

        riae_matrix = attrs.get("riae_matrix")
        if riae_matrix is not None:
            logger.info(f"  RIAE (matrix): {riae_matrix:.3f}")

    def _maybe_save_outputs(self, idata: az.InferenceData) -> None:
        """Persist sampler outputs when an output directory is configured."""
        if self.config.outdir is None:
            return
        logger.debug(" Saving results to disk...")
        self._save_results(idata)

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
        # Optionally skip heavy MCMC diagnostics plots/summaries.
        if not getattr(self.config, "skip_plot_diagnostics", False):
            plot_diagnostics(
                idata,
                self.config.outdir,
                summary_mode=getattr(
                    self.config, "diagnostics_summary_mode", "light"
                ),
                summary_position=getattr(
                    self.config, "diagnostics_summary_position", "end"
                ),
            )
            t0 = time.perf_counter()
            logger.info("Writing summary_statistics.csv...")
            az.summary(idata).to_csv(
                f"{self.config.outdir}/summary_statistics.csv"
            )
            logger.info(
                f"Wrote summary_statistics.csv in {time.perf_counter() - t0:.2f}s"
            )

        # Always save the main PSD plots for the given data type.
        try:
            self._save_plots(idata)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not save plots: {exc}")

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
