"""InferencePipeline and PSDResult."""

from __future__ import annotations

import jax
import xarray as xr

from log_psplines.arviz_utils.to_arviz import (
    _vi_result_to_idata,
    pack_stationary_result,
)
from log_psplines.config import PipelineConfig
from log_psplines.data.spectral import WishartData
from log_psplines.inference.components import SpectralComponents
from log_psplines.inference.evidence import (
    compute_pointwise_lnl,
    estimate_pipeline_lnz,
)
from log_psplines.inference.model import prepare_model
from log_psplines.inference.nuts import FactorizedMultivarNUTSStage
from log_psplines.inference.vi import FactorizedMultivarVIStage
from log_psplines.preprocessing.checks import _save_preprocessing_plot
from log_psplines.preprocessing.spectral import preprocess_to_freq_domain
from log_psplines.results import PSDResult

from .logger import logger


class InferencePipeline:
    """Runs either a standalone VI fit or a standalone NUTS fit.

    ``config.method`` selects the stage; VI never seeds NUTS initial values.
    """

    def __init__(
        self,
        full_model_kwargs: dict,
        data: WishartData,
        spline_model: SpectralComponents,
        config: PipelineConfig,
        vi_stage: FactorizedMultivarVIStage | None,
        nuts_stage: FactorizedMultivarNUTSStage | None,
    ) -> None:
        self.full_model_kwargs = full_model_kwargs
        self.data = data
        self.spline_model = spline_model
        self.config = config
        self.vi_stage = vi_stage
        self.nuts_stage = nuts_stage

    def _attach_lnz_metadata(
        self, idata: xr.DataTree, *, eta: float
    ) -> xr.DataTree:
        """Compute optional lnZ and store summary attrs on ``idata``."""
        if not bool(self.config.compute_lnz):
            return idata

        try:
            lnz_model_kwargs = dict(self.full_model_kwargs)
            lnz_model_kwargs["eta"] = float(eta)
            result = estimate_pipeline_lnz(
                idata=idata,
                data=self.data,
                model_kwargs=lnz_model_kwargs,
                outdir=self.config.outdir,
                extra_kwargs=self.config.extra_kwargs,
                verbose=self.config.verbose,
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
        log_likelihood = compute_pointwise_lnl(
            idata=idata,
            data=self.data,
            model_kwargs=self.full_model_kwargs,
        )

        idata["log_likelihood"] = xr.DataTree(dataset=log_likelihood)
        return idata

    def run(self) -> PSDResult:
        """Execute the pipeline and return a PSDResult."""
        rng = (
            jax.random.PRNGKey(self.config.rng_key)
            if isinstance(self.config.rng_key, int)
            else self.config.rng_key
        )

        if self.config.method == "vi":
            assert self.vi_stage is not None
            rng, key = jax.random.split(rng)
            vi = self.vi_stage.run(
                self.full_model_kwargs,
                init_values=None,
                rng_key=key,
                verbose=(
                    self.config.verbose
                    if self.config.vi_progress_bar is None
                    else self.config.vi_progress_bar
                ),
            )
            idata = _vi_result_to_idata(vi)
            idata = pack_stationary_result(
                idata,
                self.data,
                self.spline_model,
                self.config,
                self.config.eta,
                vi,
            )
            return PSDResult(vi=vi, idata=idata)

        assert self.nuts_stage is not None
        logger.info(f"Spline model: {self.spline_model}")

        rng, key = jax.random.split(rng)
        idata = self.nuts_stage.run(
            self.full_model_kwargs,
            init_values=None,
            rng_key=key,
            verbose=self.config.verbose,
        )
        idata = pack_stationary_result(
            idata,
            self.data,
            self.spline_model,
            self.config,
            self.nuts_stage.eta,
            None,
        )
        idata = self._attach_pointwise_log_likelihood(idata)
        idata = self._attach_lnz_metadata(idata, eta=self.nuts_stage.eta)
        return PSDResult(vi=None, idata=idata)


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

    full_kwargs, spline_model = prepare_model(
        data,
        config,
    )
    if isinstance(data, WishartData) and config.outdir is not None:
        _save_preprocessing_plot(data, config, spline_model=spline_model)

    eta = float(config.eta)
    vi_stage = (
        FactorizedMultivarVIStage(
            steps=config.vi_steps,
            lr=config.vi_lr,
            guide=config.vi_guide or "diag",
            posterior_draws=config.vi_posterior_draws,
            eta=eta,
        )
        if config.method == "vi"
        else None
    )
    nuts_stage = (
        FactorizedMultivarNUTSStage(
            n_samples=config.n_samples,
            n_warmup=config.n_warmup,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            num_chains=config.num_chains,
            chain_method=config.chain_method,
            eta=eta,
            target_accept_prob_by_channel=config.target_accept_prob_by_channel,
            max_tree_depth_by_channel=config.max_tree_depth_by_channel,
        )
        if config.method == "nuts"
        else None
    )

    return InferencePipeline(
        full_model_kwargs=full_kwargs,
        data=data,
        spline_model=spline_model,
        config=config,
        vi_stage=vi_stage,
        nuts_stage=nuts_stage,
    )


def fit(data, config=None, *, model=None, partition=None) -> PSDResult:
    """Fit stationary Wishart data or scalar time-frequency powers.

    PowerSpectrum requires an explicit LogPSpline and PowerSplineConfig.
    Its NUTS path uses the WDM prior; the stationary VI/blocked path retains
    its historical prior and configuration. ScatteredPowerSpectrum uses the
    same prior but evaluates log S(u, omega) directly at each ordinate,
    without pooling onto a rectangular time/frequency grid.

    ``partition`` optionally pools rectangular PowerSpectrum statistics while
    reconstructing posterior spectra on the model's native grid.
    """
    from log_psplines.config import PowerSplineConfig
    from log_psplines.data.spectral import (
        PowerSpectrum,
        ScatteredPowerSpectrum,
    )
    from log_psplines.inference.power import (
        fit_power_spline,
        fit_scattered_power_spline,
    )

    if isinstance(data, (PowerSpectrum, ScatteredPowerSpectrum)):
        if model is None:
            raise ValueError(
                f"{type(data).__name__} fitting requires model=LogPSpline(...)"
            )
        config = PowerSplineConfig() if config is None else config
        if not isinstance(config, PowerSplineConfig):
            raise TypeError(
                f"{type(data).__name__} requires PowerSplineConfig"
            )
        if isinstance(data, ScatteredPowerSpectrum):
            if partition is not None:
                raise ValueError(
                    "partition requires rectangular PowerSpectrum"
                )
            return fit_scattered_power_spline(data, model, config)
        return fit_power_spline(data, model, config, partition=partition)
    if partition is not None:
        raise ValueError("partition requires PowerSpectrum")
    if model is not None:
        raise ValueError(
            "explicit model requires PowerSpectrum or ScatteredPowerSpectrum"
        )
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
