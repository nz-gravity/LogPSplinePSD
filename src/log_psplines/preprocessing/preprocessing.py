from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import EmpiricalPSD, MultivarFFT
from .checks import _run_preprocessing_checks
from .config_utils import _build_config_from_kwargs, _normalize_run_config
from .configs import RunMCMCConfig, SamplerName
from .data_prep import (
    _build_welch_overlay,
    _coarse_grain_processed_data,
    _normalize_coarse_grain_config,
    _prepare_processed_data,
)
from .sampler_factory import (
    _build_common_sampler_kwargs,
    _build_model_from_data,
    _build_multivar_blocked_sampler,
    _build_sampler_inputs,
    _build_univar_sampler,
    _create_sampler,
    _validate_extra_kwargs,
    _validate_sampler_selection,
)
from .true_psd import (
    _align_true_psd_to_freq,
    _interp_psd_array,
    _prepare_true_psd_for_freq,
    _unpack_true_psd,
)

__all__ = [
    "PreprocessedMCMCInput",
    "_align_true_psd_to_freq",
    "_build_common_sampler_kwargs",
    "_build_config_from_kwargs",
    "_build_model_from_data",
    "_build_sampler_inputs",
    "_build_univar_sampler",
    "_build_multivar_blocked_sampler",
    "_build_welch_overlay",
    "_coarse_grain_processed_data",
    "_create_sampler",
    "_interp_psd_array",
    "_normalize_coarse_grain_config",
    "_normalize_run_config",
    "_prepare_processed_data",
    "_prepare_true_psd_for_freq",
    "_preprocess_data",
    "_run_preprocessing_checks",
    "_unpack_true_psd",
    "_validate_extra_kwargs",
    "_validate_sampler_selection",
]


@dataclass(frozen=True)
class PreprocessedMCMCInput:
    processed_data: Union[Periodogram, MultivarFFT]
    scaled_true_psd: Optional[np.ndarray]
    sampler_type: SamplerName
    extra_empirical_psd: list[EmpiricalPSD] | None
    extra_empirical_labels: list[str] | None
    extra_empirical_styles: list[dict] | None
    run_config: RunMCMCConfig


def _preprocess_data(data, config=None, **kwargs) -> PreprocessedMCMCInput:
    if kwargs:
        if config is not None:
            raise ValueError("Cant use both 'config' and kwargs")
        config = _build_config_from_kwargs(**kwargs)

    # This collects all arguments and config options, applies defaults, and performs validation
    run_config = _normalize_run_config(config)
    fft_data, raw_ts, sampler_type = _prepare_processed_data(
        data,
        run_config,
    )
    scaled_true_psd = _align_true_psd_to_freq(
        run_config.model.true_psd,
        fft_data,
    )

    fft_data_cg, scaled_true_psd = _coarse_grain_processed_data(
        fft_data,
        _normalize_coarse_grain_config(run_config.coarse_grain_config),
        scaled_true_psd,
    )
    # _coarse_grain_processed_data preserves the type of fft_data.
    # Since fft_data is non-None from _prepare_processed_data, result is non-None.
    assert fft_data_cg is not None, "Coarse graining should preserve non-None fft_data"
    fft_data = fft_data_cg
    # Compute + plot welch-estimate before analsysis
    (
        extra_empirical_psd,
        extra_empirical_labels,
        extra_empirical_styles,
    ) = _build_welch_overlay(raw_ts, fft_data, run_config)

    # Align true PSD to frequencies of processed data (after coarse graining if enabled)
    scaled_true_psd = _align_true_psd_to_freq(scaled_true_psd, fft_data)
    _run_preprocessing_checks(fft_data, run_config)
    return PreprocessedMCMCInput(
        processed_data=fft_data,
        scaled_true_psd=scaled_true_psd,
        sampler_type=sampler_type,
        extra_empirical_psd=extra_empirical_psd,
        extra_empirical_labels=extra_empirical_labels,
        extra_empirical_styles=extra_empirical_styles,
        run_config=run_config,
    )
