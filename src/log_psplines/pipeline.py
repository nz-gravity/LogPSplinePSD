"""Public fitting orchestration for stationary and time-frequency spectra."""

from __future__ import annotations

import jax
import numpy as np

from log_psplines.config import PipelineConfig, PowerSplineConfig
from log_psplines.data.spectral import (
    PowerData,
    WishartData,
)
from log_psplines.inference.evidence import (
    compute_pointwise_lnl,
    estimate_pipeline_lnz,
)
from log_psplines.inference.model import prepare_model
from log_psplines.inference.nuts import run_multivariate_nuts
from log_psplines.inference.power import fit_power
from log_psplines.inference.vi import run_multivariate_vi
from log_psplines.models.reconstruction import reconstruct_stationary_spectrum
from log_psplines.preprocessing.checks import _save_preprocessing_plot
from log_psplines.preprocessing.spectral import (
    align_true_psd_to_freq,
    preprocess_to_freq_domain,
)
from log_psplines.results import PSDResult, observed_wishart_data

from .logger import logger


def _attach_lnz_metadata(
    result: PSDResult,
    *,
    data: WishartData,
    model_kwargs: dict,
    config: PipelineConfig,
) -> None:
    """Add optional evidence diagnostics to a completed stationary fit."""
    if not config.compute_lnz:
        return
    try:
        evidence = estimate_pipeline_lnz(
            posterior=result.posterior,
            data=data,
            model_kwargs=model_kwargs,
            outdir=config.outdir,
            lnz_kwargs=config.lnz_kwargs,
            verbose=config.verbose,
        )
    except Exception as exc:
        logger.warning(f"Could not compute lnZ: {exc}", exc_info=True)
        result.metadata.update(
            {
                "lnz": float("nan"),
                "lnz_err": float("nan"),
                "lnz_valid": False,
                "lnz_n_estimations": 0,
                "lnz_nonconverged_count": 0,
                "lnz_method": "morphZ",
            }
        )
        return

    result.metadata.update(
        {
            "lnz": float(evidence.lnz),
            "lnz_err": float(evidence.lnz_err),
            "lnz_valid": bool(evidence.is_valid),
            "lnz_n_estimations": int(evidence.n_estimations),
            "lnz_nonconverged_count": int(evidence.nonconverged_count),
            "lnz_method": "morphZ",
        }
    )
    for index, factor in enumerate(evidence.factor_results):
        result.metadata[f"lnz_factor_{index}"] = float(factor.lnz)
        result.metadata[f"lnz_err_factor_{index}"] = float(factor.lnz_err)
        result.metadata[f"lnz_valid_factor_{index}"] = bool(factor.is_valid)


def _fit_stationary(data, config: PipelineConfig) -> PSDResult:
    """Preprocess, sample, reconstruct and optionally save a Wishart fit."""
    if not isinstance(data, WishartData):
        data = preprocess_to_freq_domain(data, config)
    model_kwargs, spline_model = prepare_model(data, config)
    if config.outdir is not None:
        _save_preprocessing_plot(data, config, spline_model=spline_model)

    rng = (
        jax.random.PRNGKey(config.rng_key)
        if isinstance(config.rng_key, int)
        else config.rng_key
    )
    _, key = jax.random.split(rng)
    if config.method == "vi":
        vi = run_multivariate_vi(
            model_kwargs,
            rng_key=key,
            steps=config.vi_steps,
            lr=config.vi_lr,
            guide=config.vi_guide or "diag",
            posterior_draws=config.vi_posterior_draws,
            eta=float(config.eta),
            verbose=(
                config.verbose
                if config.vi_progress_bar is None
                else config.vi_progress_bar
            ),
        )
        posterior = vi.posterior
        sample_stats = None
        log_likelihood = None
    else:
        logger.info(f"Spline model: {spline_model}")
        mcmc = run_multivariate_nuts(
            model_kwargs,
            rng_key=key,
            n_samples=config.n_samples,
            n_warmup=config.n_warmup,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            num_chains=config.num_chains,
            chain_method=config.chain_method,
            eta=float(config.eta),
            target_accept_prob_by_channel=config.target_accept_prob_by_channel,
            max_tree_depth_by_channel=config.max_tree_depth_by_channel,
            verbose=config.verbose,
        )
        log_likelihood = compute_pointwise_lnl(
            posterior=mcmc.posterior,
            data=data,
            model_kwargs=model_kwargs,
        )
        posterior = mcmc.posterior
        sample_stats = mcmc.sample_stats
        vi = None
    result = PSDResult(
        posterior=posterior,
        sample_stats=sample_stats,
        spectrum=reconstruct_stationary_spectrum(
            posterior, spline_model, data
        ),
        metadata={
            "data_type": "multivariate",
            "scaling_factor": float(data.scaling_factor or 1.0),
            "channel_stds": (
                None
                if data.channel_stds is None
                else np.asarray(data.channel_stds)
            ),
            "max_tree_depth": int(config.max_tree_depth),
            "max_tree_depth_by_channel": config.max_tree_depth_by_channel,
            "eta": float(config.eta),
            "sampling_eta": float(config.eta),
            "compute_lnz": config.compute_lnz,
        },
        vi=vi,
        log_likelihood=log_likelihood,
        observed_data=observed_wishart_data(data),
    )
    if config.method != "vi":
        _attach_lnz_metadata(
            result,
            data=data,
            model_kwargs={**model_kwargs, "eta": float(config.eta)},
            config=config,
        )

    if config.outdir is not None:
        result.save(
            config.outdir,
            true_psd=align_true_psd_to_freq(config.true_psd, data),
        )
    return result


def fit(data, config=None, *, model=None, partition=None) -> PSDResult:
    """Fit stationary Wishart data or scalar time-frequency powers.

    Power data require an explicit LogPSpline and PowerSplineConfig.
    A partition may pool rectangular powers while retaining native-grid
    reconstruction. Scattered ordinates are evaluated at their exact points.
    """
    if isinstance(data, PowerData):
        if model is None:
            raise ValueError(
                "PowerData fitting requires model=LogPSpline(...)"
            )
        config = PowerSplineConfig() if config is None else config
        if not isinstance(config, PowerSplineConfig):
            raise TypeError("PowerData requires PowerSplineConfig")
        return fit_power(data, model, config, partition=partition)
    if partition is not None:
        raise ValueError("partition requires PowerData")
    if model is not None:
        raise ValueError("explicit model requires PowerData")
    config = PipelineConfig() if config is None else config
    if not isinstance(config, PipelineConfig):
        raise TypeError("stationary fitting requires PipelineConfig")
    return _fit_stationary(data, config)
