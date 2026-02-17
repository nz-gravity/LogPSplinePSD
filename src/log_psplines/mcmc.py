from __future__ import annotations

from typing import Union

import arviz as az

from .datatypes.multivar import MultivariateTimeseries
from .datatypes.univar import Timeseries
from .preprocessing.configs import (
    DiagnosticsConfig,
    ModelConfig,
    NUTSConfigOverride,
    RunMCMCConfig,
    SamplerFactoryConfig,
    SamplerName,
    TruePSDInput,
    VIConfig,
)
from .preprocessing.preprocessing import (
    PreprocessedMCMCInput,
    _build_model_from_data,
    _build_sampler_inputs,
    _build_welch_overlay,
    _create_sampler,
    _interp_psd_array,
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

    Sampler type is automatically inferred from input data dimensionality:
    - 1D timeseries (Timeseries) → NUTS sampler
    - P-D timeseries (MultivariateTimeseries) → Multivariate blocked NUTS sampler

    Can be called with either a config object or legacy-style kwargs:

    - config-based: run_mcmc(data, config=RunMCMCConfig(...))
    - kwargs-based: run_mcmc(data, n_knots=10, n_samples=1000, ...)
    """
    preproc_input = _preprocess_data(data, config, **kwargs)

    # Build spline basis and compile model
    model = _build_model_from_data(
        preproc_input.processed_data,
        preproc_input.run_config.model,
    )
    sampler_inputs = _build_sampler_inputs(
        preproc_input.processed_data,
        preproc_input.run_config,
        preproc_input.sampler_type,
        preproc_input.scaled_true_psd,
        preproc_input.extra_empirical_psd,
        preproc_input.extra_empirical_labels,
        preproc_input.extra_empirical_styles,
    )

    sampler_obj = _create_sampler(
        data=preproc_input.processed_data,
        model=model,
        config=sampler_inputs,
    )

    # Finally, run MCMC and return results
    return sampler_obj.sample(
        n_samples=preproc_input.run_config.n_samples,
        n_warmup=preproc_input.run_config.n_warmup,
        only_vi=preproc_input.run_config.vi.only_vi,
    )
