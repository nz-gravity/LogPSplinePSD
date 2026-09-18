"""InferencePipeline and PSDResult."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from log_psplines.arviz_utils.to_arviz import (
    _pack_spline_model_multivar,
)
from log_psplines.config import PipelineConfig
from log_psplines.data.spectral import WishartData
from log_psplines.inference.evidence import (
    compute_pointwise_lnl,
    estimate_pipeline_lnz,
)
from log_psplines.inference.model import _joint_multivar_model, prepare_model
from log_psplines.inference.nuts import FactorizedMultivarNUTSStage, NUTSStage
from log_psplines.inference.vi import (
    FactorizedMultivarVIStage,
    StageResult,
    VIStage,
)
from log_psplines.preprocessing.checks import _save_preprocessing_plot
from log_psplines.preprocessing.spectral import (
    coarse_vi_freq_domain,
    preprocess_to_freq_domain,
)
from log_psplines.results import PSDResult, _losses_per_block_array

from .logger import logger


def _vi_result_to_idata(result: StageResult) -> xr.DataTree:
    """Wrap VI posterior draws into a minimal xr.DataTree."""
    has_samples = result.samples is not None
    values = result.samples if has_samples else result.init_values
    if not values:
        return xr.DataTree()
    ds = _posterior_values_to_dataset(values, values_are_draws=has_samples)
    return xr.DataTree(children={"posterior": xr.DataTree(dataset=ds)})


def _posterior_values_to_dataset(
    values: dict[str, jnp.ndarray],
    *,
    values_are_draws: bool,
) -> xr.Dataset:
    """Pack posterior-like values using ``chain``/``draw`` leading dims."""
    data_vars = {}
    draw_count: int | None = None
    for name, value in values.items():
        array = np.asarray(value)
        if values_are_draws:
            if array.ndim == 0:
                raise ValueError(
                    f"Posterior samples for '{name}' must include a draw axis."
                )
            array = array[None, ...]
        else:
            array = array[None, None, ...]
        if draw_count is None:
            draw_count = int(array.shape[1])
        elif int(array.shape[1]) != draw_count:
            raise ValueError(
                f"Posterior value '{name}' has {array.shape[1]} draws; "
                f"expected {draw_count}."
            )

        tail_dims = tuple(
            f"{name}_dim_{axis}" for axis in range(array.ndim - 2)
        )
        data_vars[name] = xr.DataArray(
            array,
            dims=("chain", "draw", *tail_dims),
        )

    n_draws = int(draw_count or 0)
    return xr.Dataset(
        data_vars,
        coords={"chain": [0], "draw": np.arange(n_draws)},
    )


def _init_values_to_dataset(values: dict[str, jnp.ndarray]) -> xr.Dataset:
    """Pack VI point estimates using variable-specific trailing dimensions."""
    data_vars = {}
    for name, value in values.items():
        array = np.asarray(value)[None, None, ...]
        tail_dims = tuple(
            f"{name}_dim_{axis}" for axis in range(array.ndim - 2)
        )
        data_vars[name] = xr.DataArray(
            array,
            dims=("chain", "draw", *tail_dims),
        )
    return xr.Dataset(
        data_vars,
        coords={"chain": [0], "draw": [0]},
    )


class InferencePipeline:
    """Sequential vi_coarse → vi → nuts inference pipeline.

    Each stage receives ``init_values`` from the previous stage so that
    downstream optimisation / sampling starts near the posterior mode.
    """

    def __init__(
        self,
        model_fn: Callable,
        full_model_kwargs: dict,
        coarse_model_kwargs: dict | None,
        data: WishartData,
        spline_model,
        config,
        vi_stage: VIStage,
        nuts_stage: NUTSStage,
        *,
        rng_key: int | jax.Array = 42,
        verbose: bool = False,
        vi_progress_bar: bool | None = None,
        only_vi: bool = False,
        init_from_vi: bool = True,
        vi_coarse_only: bool = False,
    ) -> None:
        self.model_fn = model_fn
        self.full_model_kwargs = full_model_kwargs
        self.coarse_model_kwargs = coarse_model_kwargs
        self.data = data
        self.spline_model = spline_model
        self.config = config
        self.vi_stage = vi_stage
        self.nuts_stage = nuts_stage
        self.rng_key = rng_key
        self.verbose = verbose
        self.vi_progress_bar = (
            verbose if vi_progress_bar is None else vi_progress_bar
        )
        self.only_vi = only_vi
        self.init_from_vi = init_from_vi
        self.vi_coarse_only = vi_coarse_only

    def _observed_data_dataset(self) -> xr.Dataset:
        freq = np.asarray(self.data.freq, dtype=float)
        channel_coords = np.arange(int(self.data.p))
        coords = {
            "freq": freq,
            "channels": channel_coords,
            "channels_aux": channel_coords,
        }
        dims = ("freq", "channels", "channels_aux")
        if self.data.raw_psd is not None:
            values = np.asarray(self.data.raw_psd, dtype=np.complex128)
        else:
            values = np.zeros(
                (freq.size, int(self.data.p), int(self.data.p)),
                dtype=np.complex128,
            )

        return xr.Dataset(
            {
                "periodogram": xr.DataArray(
                    values,
                    dims=dims,
                    coords=coords,
                )
            }
        )

    def _vi_posterior_dataset(self, vi: StageResult) -> xr.Dataset:
        has_samples = vi.samples is not None
        values = vi.samples if has_samples else vi.init_values
        if not values:
            return xr.Dataset()
        return _posterior_values_to_dataset(
            values,
            values_are_draws=has_samples,
        )

    def _attach_pipeline_metadata(
        self,
        idata: xr.DataTree,
        vi: StageResult | None,
    ) -> xr.DataTree:
        """Attach model/data groups needed by diagnostics and plotting."""
        spline_ds = _pack_spline_model_multivar(self.spline_model)
        attrs = {
            "data_type": "multivariate",
            "scaling_factor": float(self.data.scaling_factor or 1.0),
            "channel_stds": (
                None
                if self.data.channel_stds is None
                else np.asarray(self.data.channel_stds)
            ),
            "sampler": "factorized_multivar_nuts",
        }

        attrs.update(
            {
                "max_tree_depth": int(self.config.max_tree_depth),
                "posterior_psd_max_draws": int(self.config.vi_psd_max_draws),
                "vi_psd_max_draws": int(self.config.vi_psd_max_draws),
                "alpha_phi": float(self.config.alpha_phi),
                "beta_phi": float(self.config.beta_phi),
                "alpha_delta": float(self.config.alpha_delta),
                "beta_delta": float(self.config.beta_delta),
                "eta": float(self.config.eta),
                "sampling_eta": float(self.nuts_stage.eta),
            }
        )
        attrs["compute_lnz"] = bool(self.config.compute_lnz)
        if self.config.target_accept_prob_by_channel is not None:
            attrs["target_accept_prob_by_channel"] = list(
                self.config.target_accept_prob_by_channel
            )
        if self.config.max_tree_depth_by_channel is not None:
            attrs["max_tree_depth_by_channel"] = list(
                self.config.max_tree_depth_by_channel
            )
        for channel_index in range(int(self.data.p)):
            attrs[f"sampling_eta_channel_{channel_index}"] = float(
                self.nuts_stage.eta
            )
        idata.attrs.update(attrs)
        idata["observed_data"] = xr.DataTree(
            dataset=self._observed_data_dataset()
        )
        idata["spline_model"] = xr.DataTree(dataset=spline_ds)

        if vi is not None:
            idata["vi_posterior"] = xr.DataTree(
                dataset=self._vi_posterior_dataset(vi)
            )
            losses = (
                np.asarray(vi.losses, dtype=float)
                if vi.losses is not None
                else np.asarray([], dtype=float)
            )
            vi_stats = xr.Dataset(
                {
                    "losses": xr.DataArray(
                        losses,
                        dims=("draw",),
                        coords={"draw": np.arange(losses.size)},
                    )
                }
            )
            losses_per_block = _losses_per_block_array(vi.losses_per_block)
            if losses_per_block.size:
                vi_stats["losses_per_block"] = xr.DataArray(
                    losses_per_block,
                    dims=("factor", "draw_per_factor"),
                    coords={
                        "factor": np.arange(losses_per_block.shape[0]),
                        "draw_per_factor": np.arange(
                            losses_per_block.shape[1]
                        ),
                    },
                )
            idata["vi_sample_stats"] = xr.DataTree(dataset=vi_stats)
            idata["vi_log_likelihood"] = xr.DataTree(
                dataset=xr.Dataset(
                    {
                        f"log_likelihood_block_{j}": xr.DataArray(
                            np.zeros((1, 1, int(self.data.N))),
                            dims=("chain", "draw", "freq"),
                            coords={
                                "chain": [0],
                                "draw": [0],
                                "freq": np.asarray(
                                    self.data.freq, dtype=float
                                ),
                            },
                        )
                        for j in range(int(self.data.p))
                    }
                )
            )
        return idata

    def _attach_lnz_metadata(self, idata: xr.DataTree) -> xr.DataTree:
        """Compute optional lnZ and store summary attrs on ``idata``."""
        if not bool(self.config.compute_lnz):
            return idata

        try:
            lnz_model_kwargs = dict(self.full_model_kwargs)
            lnz_model_kwargs["eta"] = float(self.nuts_stage.eta)
            result = estimate_pipeline_lnz(
                idata=idata,
                data=self.data,
                model_kwargs=lnz_model_kwargs,
                outdir=self.config.outdir,
                extra_kwargs=self.config.extra_kwargs,
                verbose=self.verbose,
            )
        except Exception as exc:
            logger.warning(f"Could not compute lnZ: {exc}", exc_info=True)
            idata.attrs.update(
                {
                    "lnz": float("nan"),
                    "lnz_err": float("nan"),
                    "lnz_valid": False,
                    "lnz_n_estimations": 0,
                    "lnz_nonconverged_count": 0,
                    "lnz_method": "morphZ",
                }
            )
            return idata

        idata.attrs.update(
            {
                "lnz": float(result.lnz),
                "lnz_err": float(result.lnz_err),
                "lnz_valid": bool(result.is_valid),
                "lnz_n_estimations": int(result.n_estimations),
                "lnz_nonconverged_count": int(result.nonconverged_count),
                "lnz_method": "morphZ",
            }
        )
        for factor_index, factor_result in enumerate(result.factor_results):
            idata.attrs[f"lnz_factor_{factor_index}"] = float(
                factor_result.lnz
            )
            idata.attrs[f"lnz_err_factor_{factor_index}"] = float(
                factor_result.lnz_err
            )
            idata.attrs[f"lnz_valid_factor_{factor_index}"] = bool(
                factor_result.is_valid
            )
        return idata

    def _attach_pointwise_log_likelihood(
        self, idata: xr.DataTree
    ) -> xr.DataTree:
        """Attach per-frequency pointwise log-likelihood draws for PSIS-LOO."""
        try:
            log_likelihood = compute_pointwise_lnl(
                idata=idata,
                data=self.data,
                model_kwargs=self.full_model_kwargs,
            )
        except Exception as exc:
            logger.warning(
                f"Could not compute pointwise log-likelihood: {exc}",
                exc_info=True,
            )
            return idata

        idata["log_likelihood"] = xr.DataTree(dataset=log_likelihood)
        return idata

    def run(self) -> PSDResult:
        """Execute the pipeline and return a PSDResult."""
        rng = (
            jax.random.PRNGKey(self.rng_key)
            if isinstance(self.rng_key, int)
            else self.rng_key
        )

        vi_coarse: StageResult | None = None
        init_values: dict[str, jnp.ndarray] | None = None

        if self.coarse_model_kwargs is not None:
            rng, key = jax.random.split(rng)
            vi_coarse = self.vi_stage.run(
                self.model_fn,
                self.coarse_model_kwargs,
                init_values=None,
                rng_key=key,
                verbose=self.vi_progress_bar,
            )
            init_values = vi_coarse.init_values

        if self.vi_coarse_only:
            if vi_coarse is None:
                raise ValueError(
                    "vi_coarse_only=True requires coarse_model_kwargs. "
                    "Set coarse_grain_config_vi or auto_coarse_vi."
                )
            idata = _vi_result_to_idata(vi_coarse)
            idata = self._attach_pipeline_metadata(idata, vi_coarse)
            return PSDResult(vi_coarse=vi_coarse, vi=None, idata=idata)

        rng, key = jax.random.split(rng)
        vi = self.vi_stage.run(
            self.model_fn,
            self.full_model_kwargs,
            init_values=init_values,
            rng_key=key,
            verbose=self.vi_progress_bar,
        )
        init_values = vi.init_values if self.init_from_vi else None

        if self.only_vi:
            idata = _vi_result_to_idata(vi)
            idata = self._attach_pipeline_metadata(idata, vi)
            return PSDResult(vi_coarse=vi_coarse, vi=vi, idata=idata)

        logger.info(f"Spline model: {self.spline_model}")

        rng, key = jax.random.split(rng)
        idata = self.nuts_stage.run(
            self.model_fn,
            self.full_model_kwargs,
            init_values=init_values,
            rng_key=key,
            verbose=self.verbose,
        )
        idata = self._attach_pipeline_metadata(idata, vi)
        idata = self._attach_pointwise_log_likelihood(idata)
        idata = self._attach_lnz_metadata(idata)
        return PSDResult(vi_coarse=vi_coarse, vi=vi, idata=idata)


def make_pipeline(
    data,
    config: PipelineConfig | None = None,
) -> InferencePipeline:
    """Build an InferencePipeline from data and config.

    Parameters
    ----------
    data:
        Time-domain ``TimeSeries`` (including ``y.shape == (n,)``)
        or pre-processed ``WishartData``.
    config:
        Pipeline configuration.  Defaults to :class:`PipelineConfig` with all
        default values.

    Returns
    -------
    InferencePipeline
        Ready-to-run pipeline.  Call ``.run()`` to execute it.
    """
    if config is None:
        config = PipelineConfig()

    if not isinstance(data, WishartData):
        data = preprocess_to_freq_domain(data, config)

    model_fn = _joint_multivar_model

    full_kwargs, spline_model = prepare_model(
        data,
        config,
    )
    if isinstance(data, WishartData) and config.outdir is not None:
        _save_preprocessing_plot(data, config, spline_model=spline_model)

    coarse_data = (
        coarse_vi_freq_domain(data, config)
        if config.init_from_vi and config.use_coarse_vi_for_init
        else None
    )
    coarse_kwargs = (
        prepare_model(coarse_data, config)[0]
        if coarse_data is not None
        else None
    )

    eta = float(config.eta)
    vi_stage = FactorizedMultivarVIStage(
        steps=config.vi_steps,
        lr=config.vi_lr,
        guide=config.vi_guide or "diag",
        posterior_draws=config.vi_posterior_draws,
        eta=eta,
    )
    nuts_stage = FactorizedMultivarNUTSStage(
        n_samples=config.n_samples,
        n_warmup=config.n_warmup,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
        dense_mass=config.dense_mass,
        num_chains=config.num_chains,
        eta=eta,
        target_accept_prob_by_channel=config.target_accept_prob_by_channel,
        max_tree_depth_by_channel=config.max_tree_depth_by_channel,
    )

    return InferencePipeline(
        model_fn=model_fn,
        full_model_kwargs=full_kwargs,
        coarse_model_kwargs=coarse_kwargs,
        data=data,
        spline_model=spline_model,
        config=config,
        vi_stage=vi_stage,
        nuts_stage=nuts_stage,
        rng_key=config.rng_key,
        verbose=config.verbose,
        vi_progress_bar=config.vi_progress_bar,
        only_vi=config.only_vi,
        init_from_vi=config.init_from_vi,
        vi_coarse_only=config.vi_coarse_only,
    )


def fit(data, config=None) -> PSDResult:
    """Fit stationary one- or multi-channel data with VI and blocked NUTS."""
    pipeline = make_pipeline(data, config)
    result = pipeline.run()
    if pipeline.config.outdir is not None:
        from log_psplines.preprocessing.spectral import align_true_psd_to_freq

        result.save(
            pipeline.config.outdir,
            true_psd=align_true_psd_to_freq(
                pipeline.config.true_psd, pipeline.data
            ),
        )
    return result
