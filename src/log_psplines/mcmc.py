from __future__ import annotations

from typing import Union

import arviz as az

from .datatypes.multivar import MultivariateTimeseries
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
    _build_welch_overlay,
    _normalize_coarse_grain_config,
    _normalize_run_config,
    _prepare_processed_data,
    _run_preprocessing_checks,
    _preprocess_data,
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
    "_build_welch_overlay",
    "run_mcmc",
]


def run_mcmc(
    data: Union[Timeseries, MultivariateTimeseries],
    config: RunMCMCConfig | None = None,
    **kwargs,
) -> az.InferenceData:
    """
    Unified MCMC entrypoint for univariate and multivariate PSD estimation.
    Expects time-domain inputs (Timeseries or MultivariateTimeseries).

    Can be called with either a config object or legacy-style kwargs:

    - config-based: run_mcmc(data, config=RunMCMCConfig(...))
    - kwargs-based: run_mcmc(data, n_knots=10, n_samples=1000, ...)
    """
    preproc_input = _preprocess_data(data, config)

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
