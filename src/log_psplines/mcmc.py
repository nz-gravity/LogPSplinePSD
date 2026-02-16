from __future__ import annotations

from typing import Union

import arviz as az

from .datatypes import Periodogram
from .datatypes.multivar import MultivarFFT, MultivariateTimeseries
from .datatypes.univar import Timeseries
from .mcmc_utils import (
    DiagnosticsConfig,
    ModelConfig,
    NUTSConfigOverride,
    RunMCMCConfig,
    SamplerName,
    TruePSDInput,
    VIConfig,
    _align_true_psd_to_freq,
    _build_config_from_kwargs,
    _build_model_from_data,
    _build_sampler_inputs,
    _coarse_grain_processed_data,
    _create_sampler,
    _interp_psd_array,
    _maybe_build_welch_overlay,
    _normalize_coarse_grain_config,
    _normalize_run_config,
    _prepare_processed_data,
    _run_preprocessing_checks,
)

__all__ = [
    "DiagnosticsConfig",
    "ModelConfig",
    "NUTSConfigOverride",
    "RunMCMCConfig",
    "SamplerName",
    "TruePSDInput",
    "VIConfig",
    "_build_model_from_data",
    "_create_sampler",
    "_interp_psd_array",
    "_maybe_build_welch_overlay",
    "run_mcmc",
]


def run_mcmc(
    data: Union[Timeseries, MultivariateTimeseries, Periodogram, MultivarFFT],
    config: RunMCMCConfig | None = None,
    **kwargs,
) -> az.InferenceData:
    """
    Unified MCMC entrypoint for univariate and multivariate PSD estimation.

    Can be called with either a config object or legacy-style kwargs:

    - config-based: run_mcmc(data, config=RunMCMCConfig(...))
    - kwargs-based: run_mcmc(data, n_knots=10, n_samples=1000, ...)
    """
    # If kwargs provided, build config from them
    if kwargs:
        if config is not None:
            raise ValueError(
                "Cannot specify both 'config' and keyword arguments. "
                "Use either config=RunMCMCConfig(...) or pass kwargs directly."
            )
        config = _build_config_from_kwargs(**kwargs)

    # This collects all arguments and config options, applies defaults, and performs validation
    run_config = _normalize_run_config(config)
    processed_data, raw_multivar_ts, sampler_type = _prepare_processed_data(
        data,
        run_config,
    )
    scaled_true_psd = _align_true_psd_to_freq(
        run_config.model.true_psd,
        processed_data,
    )

    processed_after, scaled_true_psd = _coarse_grain_processed_data(
        processed_data,
        _normalize_coarse_grain_config(run_config.coarse_grain_config),
        scaled_true_psd,
    )
    if processed_after is None:
        raise ValueError(
            "Processed data unexpectedly None after coarse graining."
        )
    processed_data = processed_after

    # Plot the periodogram before analsysis
    (
        extra_empirical_psd,
        extra_empirical_labels,
        extra_empirical_styles,
    ) = _maybe_build_welch_overlay(raw_multivar_ts, processed_data, run_config)

    # Align true PSD to frequencies of processed data (after coarse graining if enabled)
    scaled_true_psd = _align_true_psd_to_freq(scaled_true_psd, processed_data)
    _run_preprocessing_checks(processed_data, run_config)

    # Build spline basis and compile model
    model = _build_model_from_data(processed_data, run_config.model)
    sampler_inputs = _build_sampler_inputs(
        processed_data,
        run_config,
        sampler_type,
        scaled_true_psd,
        extra_empirical_psd,
        extra_empirical_labels,
        extra_empirical_styles,
    )

    sampler_obj = _create_sampler(
        data=processed_data,
        model=model,
        config=sampler_inputs,
    )

    # Finally, run MCMC and return results
    return sampler_obj.sample(
        n_samples=run_config.n_samples,
        n_warmup=run_config.n_warmup,
        only_vi=run_config.vi.only_vi,
    )
