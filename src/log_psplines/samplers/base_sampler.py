from __future__ import annotations

"""
Abstract base classes for MCMC samplers.

Provides foundation for both univariate and multivariate PSD estimation samplers.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import arviz_plots as azp
import arviz_stats as azs
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arviz_base import from_dict, from_numpyro
from arviz_stats.base import array_stats

from ..arviz_utils._datatree import require_dataset as _require_dataset
from ..arviz_utils._datatree import save_inference_data as _save_inference_data
from ..arviz_utils._datatree import select_draw_slice as _select_draw_slice
from ..logger import logger

ChainMethod = Literal["parallel", "vectorized", "sequential"]
SampleArray = Union[jnp.ndarray, np.ndarray]


def _sanitize_attr_value(value: Any) -> Any | None:
    """Return a NetCDF/HDF5-friendly attribute value or ``None``."""
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float, str, np.number)):
        return value

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return None
        if value.dtype.kind == "U":
            return value.astype("S")
        return value

    if isinstance(value, (list, tuple)):
        if not value:
            return np.asarray(value, dtype=float)
        if all(isinstance(v, (bool, int, float, np.number)) for v in value):
            return np.asarray(
                [int(v) if isinstance(v, bool) else v for v in value],
                dtype=float,
            )
        if all(isinstance(v, str) for v in value):
            if len(value) == 1:
                return value[0]
            return list(value)
        return None

    return None


def _normalize_chain_draw_array(
    value: Any,
    *,
    num_chains: int,
    is_sample_stats: bool,
) -> np.ndarray:
    """Normalize arrays to ``(chain, draw, ...)`` or ``(chain, draw)`` layout."""
    arr = np.asarray(value)

    if num_chains <= 1:
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr[np.newaxis, :]
        return arr if arr.shape[0] == 1 else arr[np.newaxis, ...]

    if not is_sample_stats:
        return arr

    if arr.ndim == 0:
        return np.broadcast_to(arr.reshape(1, 1), (num_chains, 1))
    if arr.ndim == 1:
        if arr.shape[0] == num_chains:
            return arr[:, np.newaxis]
        return np.broadcast_to(arr[np.newaxis, :], (num_chains, arr.shape[0]))
    if arr.shape[0] != num_chains and arr.shape[1] == num_chains:
        axes = list(range(arr.ndim))
        axes[0], axes[1] = 1, 0
        return np.transpose(arr, axes)
    return arr


def _normalize_samples_and_stats(
    samples: Mapping[str, SampleArray],
    sample_stats: Dict[str, Any],
    *,
    num_chains: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return normalized explicit sample/stat dictionaries for ``from_dict``."""
    normalized_samples = {
        str(key): _normalize_chain_draw_array(
            value, num_chains=num_chains, is_sample_stats=False
        )
        for key, value in samples.items()
    }
    normalized_stats = {
        str(key): _normalize_chain_draw_array(
            value, num_chains=num_chains, is_sample_stats=True
        )
        for key, value in sample_stats.items()
    }
    return normalized_samples, normalized_stats


@dataclass
class SamplerConfig:
    """Base configuration for all MCMC samplers."""

    alpha_phi: float = 1.0
    beta_phi: float = 1.0
    alpha_delta: float = 1e-4
    beta_delta: float = 1e-4
    num_chains: int = 1
    chain_method: Optional[ChainMethod] = None
    rng_key: int = 42
    verbose: bool = True
    outdir: Optional[str] = None
    compute_lnz: bool = True
    scaling_factor: float = 1.0  # To track any data scaling
    channel_stds: Optional[np.ndarray] = (
        None  # Per-channel stds for multivariate scaling
    )
    true_psd: Optional[jnp.ndarray] = None  # True PSD for diagnostics
    vi_psd_max_draws: int = (
        50  # Cap PSD reconstructions from VI/posterior draws
    )
    only_vi: bool = False  # Skip MCMC and rely on VI draws only
    # Fast-diagnostics controls
    posterior_psd_max_draws: int = 50  # Cap posterior PSD reconstructions
    compute_coherence_quantiles: bool = True  # Compute coherence bands
    compute_psis: bool = True  # Enable PSIS-LOO diagnostics
    run_full_diagnostics: bool = True  # Run full post-sampling diagnostics
    # Cap ESS/Rhat/PSIS computations (posterior elements) to avoid OOM.
    mcmc_diag_max_elements: int = 250_000
    max_saved_draws: int = 1000  # Cap saved/diagnostic draws per chain
    max_save_bytes: int = 750_000_000  # Skip heavy outputs above this size
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
        self.rng_key: jax.Array = jax.random.PRNGKey(config.rng_key)
        if config.chain_method is not None:
            allowed = {"parallel", "vectorized", "sequential"}
            if config.chain_method not in allowed:
                raise ValueError(
                    f"Unknown chain_method='{config.chain_method}'. "
                    f"Expected one of {sorted(allowed)}."
                )
            self.chain_method: ChainMethod = config.chain_method
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
        vi_warm_start_plan: Any | None = None,
        **kwargs,
    ) -> xr.DataTree:
        """Run MCMC sampling and return inference results."""
        pass

    @abstractmethod
    def _setup_data(self) -> None:
        """Setup data attributes for sampling (univar vs multivar specific)."""
        pass

    @abstractmethod
    def _get_lnz(
        self, samples: Dict[str, Any], sample_stats: Dict[str, Any]
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

    def _arviz_coords(self) -> Dict[str, Any]:
        """Return ArviZ coordinates for posterior variables."""
        return {}

    def _arviz_dims(
        self, samples: Optional[Mapping[str, SampleArray]] = None
    ) -> Dict[str, list[str]]:
        """Return ArviZ dimension names for posterior variables."""
        return {}

    def _attach_custom_groups(
        self,
        idata: xr.DataTree,
        *,
        samples: Mapping[str, SampleArray],
        sample_stats: Dict[str, Any],
    ) -> None:
        """Attach repo-specific groups after core ArviZ conversion."""
        return

    def _build_tree_attrs(self, lnz: float, lnz_err: float) -> Dict[str, Any]:
        """Assemble root ``DataTree`` attributes."""
        attrs = dict(
            device=str(self.device),
            runtime=self.runtime,
            lnz=lnz,
            lnz_err=lnz_err,
            sampler_type=self.sampler_type,
            data_type=self.data_type,
        )
        extra_attrs = getattr(self, "_extra_idata_attrs", None)
        if extra_attrs:
            attrs.update(dict(extra_attrs))

        config_attrs: Dict[str, Any] = {}
        for key, value in asdict(self.config).items():
            if value is None:
                continue
            sanitized = _sanitize_attr_value(value)
            if sanitized is None:
                continue
            config_attrs[key] = sanitized
        if config_attrs.pop("true_psd", None) is not None:
            config_attrs["true_psd_provided"] = 1

        attrs.update(config_attrs)
        return attrs

    def _create_vi_inference_data(
        self,
        samples: Mapping[str, SampleArray],
        sample_stats: Dict[str, Any],
        diagnostics: Optional[Dict[str, Any]],
    ) -> xr.DataTree:
        """Convert VI samples to the canonical ``DataTree`` container."""
        idata = self._create_inference_data(
            samples,
            sample_stats,
            lnz=np.nan,
            lnz_err=np.nan,
        )
        self._attach_vi_group(idata, diagnostics)
        return idata

    def _assemble_vi_only_inference_data(
        self,
        *,
        diagnostics: Optional[Dict[str, Any]],
        posterior_draws: Optional[Mapping[str, Any]],
        means: Optional[Mapping[str, Any]],
        site_filter: Callable[[str], bool],
        normalize_samples: Callable[
            [Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]
        ],
        sample_stats_builder: Optional[
            Callable[[Dict[str, jnp.ndarray]], Dict[str, Any]]
        ] = None,
        required_sites: Optional[tuple[str, ...]] = None,
        missing_sites_error: Optional[str] = None,
        no_samples_error: str = "No variational posterior samples available.",
        no_diagnostics_error: Optional[str] = None,
    ) -> xr.DataTree:
        """Build VI-only ``DataTree`` using sampler-specific callbacks."""
        if diagnostics is None and no_diagnostics_error is not None:
            raise ValueError(no_diagnostics_error)

        if posterior_draws:
            sample_dict = {
                name: jnp.asarray(array)
                for name, array in posterior_draws.items()
                if site_filter(name)
            }
        elif means:
            sample_dict = {
                name: jnp.asarray(value)[None, ...]
                for name, value in means.items()
                if site_filter(name)
            }
        else:
            raise ValueError(no_samples_error)

        if not sample_dict:
            raise ValueError(no_samples_error)

        if required_sites is not None:
            missing = set(required_sites) - set(sample_dict)
            if missing:
                raise ValueError(
                    missing_sites_error
                    or "Variational-only mode is missing required parameter sites."
                )

        sample_stats: Dict[str, Any] = {}
        if sample_stats_builder is not None:
            try:
                sample_stats = sample_stats_builder(sample_dict) or {}
            except Exception:
                sample_stats = {}

        samples = normalize_samples(dict(sample_dict))

        self.runtime = 0.0
        idata = self._create_vi_inference_data(
            samples, sample_stats, diagnostics
        )
        self._cache_full_diagnostics(idata)
        return idata

    def _attach_vi_group(
        self, idata: xr.DataTree, diagnostics: Optional[Dict[str, Any]]
    ) -> None:
        """Attach optional VI outputs to the result tree."""
        return

    def _attach_vi_posterior_dataset(
        self,
        idata: xr.DataTree,
        sample_vars: Dict[str, tuple[list[str], np.ndarray]],
        coords: Dict[str, Any],
    ) -> None:
        """Attach a VI posterior dataset when sample variables are present."""
        if sample_vars:
            idata["vi_posterior"] = xr.DataTree(
                dataset=xr.Dataset(sample_vars, coords=coords)
            )

    def _attach_vi_sample_stats(
        self,
        idata: xr.DataTree,
        diagnostics: Dict[str, Any],
        *,
        attr_keys: tuple[str, ...] | list[str],
    ) -> None:
        """Attach VI diagnostics and losses inside ``vi_sample_stats``."""
        stat_vars = {}
        coords: Dict[str, Any] = {}
        losses = diagnostics.get("losses")
        if losses is not None:
            losses_arr = np.asarray(losses, dtype=np.float64).reshape(-1)
            coords["vi_step"] = np.arange(losses_arr.size)
            stat_vars["losses"] = (["vi_step"], losses_arr)
        dataset = xr.Dataset(stat_vars, coords=coords)
        for key in attr_keys:
            value = diagnostics.get(key)
            if value is not None:
                dataset.attrs[key] = value
        for src_key, out_key in (
            ("riae", "riae"),
            ("riae_matrix", "riae"),
            ("l2", "l2"),
            ("l2_matrix", "l2"),
            ("coverage", "coverage"),
            ("ci_coverage", "coverage"),
            ("pareto_k_max", "pareto_k_max"),
            ("psis_khat_max", "pareto_k_max"),
        ):
            if out_key in dataset.attrs:
                continue
            value = diagnostics.get(src_key)
            if value is not None:
                dataset.attrs[out_key] = value
        if dataset.data_vars or dataset.attrs:
            idata["vi_sample_stats"] = xr.DataTree(dataset=dataset)

    def _save_vi_diagnostics(self, *args, **kwargs) -> None:
        # TODO, standardise between both univar and multivar
        # 0. compute PSIS loo
        # 1. save vi summary stats csv
        # 2. save loss plots
        # 3. Save VI posterior prediction
        pass

    def to_arviz(
        self,
        samples: Mapping[str, SampleArray],
        sample_stats: Dict[str, Any],
        *,
        mcmc: Any = None,
    ) -> xr.DataTree:
        """Convert an MCMC run into the canonical ``DataTree`` container."""
        lnz, lnz_err = self._get_lnz(dict(samples), sample_stats)
        if mcmc is None:
            # Some tests and helper paths construct explicit sample dictionaries
            # without a live NumPyro MCMC object.
            idata = self._create_inference_data(
                samples=samples,
                sample_stats=sample_stats,
                lnz=lnz,
                lnz_err=lnz_err,
            )
        else:
            idata = self._create_mcmc_inference_data(
                mcmc=mcmc,
                samples=samples,
                sample_stats=sample_stats,
                lnz=lnz,
                lnz_err=lnz_err,
            )
        logger.debug(" DataTree created.")
        vi_diag = getattr(self, "_vi_diagnostics", None)
        if vi_diag:
            try:
                self._attach_vi_group(idata, vi_diag)
            except Exception as exc:  # pragma: no cover - best-effort hook
                logger.warning(f"Could not attach VI diagnostics: {exc}")
        rhat_vals = self._compute_chain_summaries(idata)
        self._cache_full_diagnostics(idata)
        self._log_summary_metrics(idata, lnz, lnz_err, rhat_vals)
        self._maybe_save_outputs(idata)
        return idata

    def _create_mcmc_inference_data(
        self,
        *,
        mcmc: Any,
        samples: Mapping[str, SampleArray],
        sample_stats: Dict[str, Any],
        lnz: float,
        lnz_err: float,
    ) -> xr.DataTree:
        attrs = self._build_tree_attrs(lnz, lnz_err)
        coords = self._arviz_coords()
        dims = self._arviz_dims(samples)
        idata = from_numpyro(
            mcmc,
            coords=coords,
            dims=dims,
            log_likelihood=True,
        )
        idata.attrs.update(attrs)
        self._attach_custom_groups(
            idata,
            samples=samples,
            sample_stats=sample_stats,
        )
        return idata

    def _cache_full_diagnostics(self, idata: xr.DataTree) -> None:
        """Legacy hook retained for API stability; no extra diagnostics."""
        del idata
        return

    def _compute_chain_summaries(
        self, idata: xr.DataTree
    ) -> Optional[np.ndarray]:
        """Compute optional per-parameter chain summaries."""
        posterior = _require_dataset(idata, "posterior")
        chain_count = int(posterior.sizes.get("chain", 1))
        multi_chain = self.config.num_chains > 1 and chain_count > 1
        rhat_values: list[np.ndarray] = []
        ess_values: list[np.ndarray] = []

        for name, var in posterior.data_vars.items():
            arr = np.asarray(var)
            if arr.ndim < 2:
                continue
            try:
                ess = np.asarray(
                    array_stats.ess(
                        arr,
                        method="bulk",
                        chain_axis=0,
                        draw_axis=1,
                    )
                )
                ess = ess[np.isfinite(ess)]
                if ess.size:
                    ess_values.append(ess.reshape(-1))
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(f"Could not compute ESS for {name}: {exc}")

            if not multi_chain:
                continue

            try:
                rhat = np.asarray(
                    array_stats.rhat(arr, chain_axis=0, draw_axis=1)
                )
                rhat = rhat[np.isfinite(rhat)]
                if rhat.size:
                    rhat_values.append(rhat.reshape(-1))
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(f"Could not compute R-hat for {name}: {exc}")

        ess_flat = np.concatenate(ess_values) if ess_values else np.array([])
        if ess_flat.size:
            idata.attrs["ess"] = ess_flat

        rhat_flat = (
            np.concatenate(rhat_values) if rhat_values else np.array([])
        )
        if rhat_flat.size:
            idata.attrs["rhat"] = rhat_flat
            return rhat_flat
        return None

    def _log_summary_metrics(
        self,
        idata: xr.DataTree,
        lnz: float,
        lnz_err: float,
        rhat_vals: Optional[np.ndarray],
    ) -> None:
        """Log human-readable summary metrics for quick terminal inspection."""
        if not self.config.verbose:
            return

        attrs = idata.attrs
        if not (np.isnan(lnz) or np.isnan(lnz_err)):
            logger.info(f"  lnz: {lnz:.2f} ± {lnz_err:.2f}")

        ess = attrs.get("ess")
        if ess is not None:
            logger.info(
                f"  ESS min: {np.min(ess):.1f}, max: {np.max(ess):.1f}"
            )

        if rhat_vals is not None and rhat_vals.size:
            logger.info(
                f"  Rhat: min={np.min(rhat_vals):.3f}, mean={np.mean(rhat_vals):.3f}, max={np.max(rhat_vals):.3f}"
            )
            logger.info(
                f"  Rhat ≤ 1.01: {(rhat_vals <= 1.01).mean() * 100:.1f}%"
            )

    def _maybe_save_outputs(self, idata: xr.DataTree) -> None:
        """Persist sampler outputs when an output directory is configured."""
        if self.config.outdir is None:
            return
        logger.debug(" Saving results to disk...")
        self._save_results(idata)

    def _create_inference_data(
        self,
        samples: Mapping[str, SampleArray],
        sample_stats: Dict[str, Any],
        lnz: float,
        lnz_err: float,
        *,
        mcmc: Any | None = None,
    ) -> xr.DataTree:
        """Create a ``DataTree`` from explicit sample dictionaries."""
        del mcmc
        attrs = self._build_tree_attrs(lnz, lnz_err)
        coords = self._arviz_coords()
        dims = self._arviz_dims(samples)
        samples, sample_stats = _normalize_samples_and_stats(
            samples, sample_stats, num_chains=self.config.num_chains
        )
        idata = from_dict(
            {
                "posterior": samples,
                "sample_stats": sample_stats,
            },
            coords=coords,
            dims=dims,
        )
        idata.attrs.update(attrs)

        self._attach_custom_groups(
            idata,
            samples=samples,
            sample_stats=sample_stats,
        )
        return idata

    def _save_results(self, idata: xr.DataTree) -> None:
        """Save inference results to disk."""
        assert self.config.outdir is not None
        idata_out = idata
        logger.info("save_results: start")
        estimated_bytes = None
        total = 0
        for group in ("posterior", "sample_stats", "log_likelihood"):
            if group not in idata.children:
                continue
            dataset = _require_dataset(idata, group)
            for var in dataset.data_vars.values():
                total += int(var.size) * int(var.dtype.itemsize)
        estimated_bytes = total

        if estimated_bytes is not None:
            logger.info(
                f"Estimated idata size: {estimated_bytes / 1e6:.1f} MB"
            )
        posterior = _require_dataset(idata, "posterior")
        n_draws = int(posterior.sizes.get("draw", 0))
        max_draws = int(getattr(self.config, "max_saved_draws", 0) or 0)
        if max_draws > 0 and n_draws > max_draws:
            step = int(np.ceil(n_draws / max_draws))
            idata_out = _select_draw_slice(idata, slice(None, None, step))
            posterior_out = _require_dataset(idata_out, "posterior")
            logger.info(
                f"Thinning outputs: draw step {step} "
                f"({n_draws} -> {posterior_out.sizes.get('draw', 0)})"
            )

        skip_heavy = False
        max_save_bytes = int(getattr(self.config, "max_save_bytes", 0) or 0)
        if (
            max_save_bytes > 0
            and estimated_bytes is not None
            and estimated_bytes > max_save_bytes
        ):
            skip_heavy = True
            logger.warning(
                "Skipping netcdf/summary due to size "
                f"({estimated_bytes / 1e6:.1f} MB > "
                f"{max_save_bytes / 1e6:.1f} MB)."
            )

        diagnostics_dir = Path(self.config.outdir) / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        logger.info("save_results: writing minimal diagnostics")
        try:
            from ..diagnostics import (
                build_nuts_summary_table,
                plot_energy,
            )

            nuts_summary = build_nuts_summary_table(
                idata_out,
                true_psd=self.config.true_psd,
            )
            nuts_summary.to_csv(
                diagnostics_dir / "nuts_summary.csv", index=False
            )

            nuts_metric_attrs: Dict[str, float] = {}
            for col in ("riae", "l2", "coverage", "rhat_max"):
                if col not in nuts_summary.columns:
                    continue
                vals = np.asarray(nuts_summary[col], dtype=float)
                vals = vals[np.isfinite(vals)]
                nuts_metric_attrs[col] = (
                    float(np.median(vals)) if vals.size else float("nan")
                )

            if nuts_metric_attrs:
                for target in (idata, idata_out):
                    if "sample_stats" not in target.children:
                        continue
                    sample_stats_ds = _require_dataset(target, "sample_stats")
                    sample_stats_ds.attrs.update(nuts_metric_attrs)

            trace_plot = azp.plot_trace_dist(idata_out, compact=True)
            trace_plot.savefig(
                diagnostics_dir / "traces.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")

        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"Could not save minimal NUTS diagnostics: {exc}",
                exc_info=True,
            )
        try:
            energy_plot = plot_energy(idata_out)
            energy_plot.savefig(
                diagnostics_dir / "energy.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"Could not save energy diagnostics: {exc}",
                exc_info=True,
            )
        logger.debug("save_results: minimal diagnostics done")

        if not skip_heavy:
            logger.info("save_results: writing inference_data.nc")
            _save_inference_data(
                idata_out,
                f"{self.config.outdir}/inference_data.nc",
                engine="h5netcdf",
            )
            logger.info("save_results: wrote inference_data.nc")

        t0 = time.perf_counter()
        logger.debug("Writing summary_statistics.csv...")
        if not skip_heavy:
            azs.summary(idata_out["posterior"]).to_csv(
                Path(self.config.outdir) / "summary_statistics.csv"
            )
            logger.debug(
                f"Wrote summary_statistics.csv in {time.perf_counter() - t0:.2f}s"
            )
        else:
            logger.info("summary_statistics.csv skipped (size guard).")

        # Always save the main PSD plots for the given data type.
        try:
            logger.debug("save_results: saving PSD plots")
            self._save_plots(idata_out)
            logger.debug("save_results: saving PSD plots done")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not save plots: {exc}")

    @abstractmethod
    def _save_plots(self, idata: xr.DataTree) -> None:
        """Save data-type specific plots."""
        pass

    def _select_chain_method(self) -> ChainMethod:
        """Choose an appropriate NumPyro chain_method based on hardware.

        NumPyro defaults to ``parallel`` for ``num_chains > 1`` which fails when
        only a single device is available. Fall back to ``vectorized`` in that
        case so multi-chain sampling works out of the box on CPUs/GPUs with a
        single device.
        """

        if self.config.num_chains <= 1:
            return "sequential"

        n_devs = len(jax.devices())
        # Only use parallel when we have *at least* as many devices as chains AND
        # the device count is a small multiple of num_chains.  On some HPC nodes
        # JAX may see many virtual CPU "devices" (e.g. via
        # XLA_FLAGS=--xla_force_host_platform_device_count=N); blindly using
        # chain_method='parallel' in that case can cause NumPyro to launch more
        # chains than requested, leading to shape mismatches in ArviZ.
        if (
            n_devs >= self.config.num_chains
            and n_devs <= 2 * self.config.num_chains
        ):
            logger.info(
                f"Running {self.config.num_chains} chains in parallel across {n_devs} device(s)."
            )
            return "parallel"

        logger.info(
            f"Running {self.config.num_chains} chains on {n_devs} device(s) with NumPyro vectorized chaining (no multiprocessing)."
        )
        if n_devs > 1 and self.config.num_chains > n_devs:
            logger.warning(
                f"num_chains ({self.config.num_chains}) exceeds available JAX devices ({n_devs}); "
                "for true multiprocessing with chain_method='parallel', "
                "set num_chains <= number of devices."
            )

        return "vectorized"
