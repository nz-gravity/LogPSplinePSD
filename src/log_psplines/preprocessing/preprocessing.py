from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Optional, Union

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import EmpiricalPSD, MultivarFFT
from ..logger import logger
from .checks import _run_preprocessing_checks
from .coarse_grain import (
    CoarseGrainConfig,
    _closest_divisor,
    _smallest_divisor_geq,
)
from .config_utils import _build_config_from_kwargs, _normalize_run_config
from .configs import RunMCMCConfig, SamplerName
from .data_prep import (
    _apply_multivar_frequency_exclusion,
    _build_welch_overlay,
    _coarse_grain_processed_data,
    _normalize_coarse_grain_config,
    _normalize_excluded_frequency_bands,
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
    coarse_vi_context: Optional["CoarseVIContext"] = None


@dataclass(frozen=True)
class CoarseVIContext:
    processed_data: Union[Periodogram, MultivarFFT]
    scaled_true_psd: Optional[np.ndarray]
    sampler_type: SamplerName
    metadata: dict[str, object]


def _get_frequency_count(data: Union[Periodogram, MultivarFFT]) -> int:
    if isinstance(data, Periodogram):
        return int(len(data.freqs))
    return int(len(data.freq))


def _derive_vi_coarse_grain_config(
    processed_data: Union[Periodogram, MultivarFFT],
    run_config: RunMCMCConfig,
):
    vi_cfg = run_config.vi
    full_nfreq = _get_frequency_count(processed_data)
    k_basis = int(run_config.model.n_knots + run_config.model.degree - 1)

    explicit_cfg = _normalize_coarse_grain_config(
        vi_cfg.coarse_grain_config_vi
    )
    if explicit_cfg.enabled:
        metadata: dict[str, object] = {
            "coarse_vi_mode": "config",
            "coarse_vi_full_nfreq": full_nfreq,
        }

        if explicit_cfg.Nc is not None:
            requested_nc = int(explicit_cfg.Nc)
            if full_nfreq % requested_nc != 0:
                adjusted_nc = _closest_divisor(full_nfreq, requested_nc)
                logger.info(
                    "Adjusting explicit coarse VI bins: "
                    f"N_full={full_nfreq} not divisible by Nc={requested_nc}; "
                    f"using Nc={adjusted_nc}."
                )
                explicit_cfg = CoarseGrainConfig(
                    enabled=True,
                    Nc=adjusted_nc,
                    Nh=None,
                )
                metadata["coarse_vi_requested_Nc"] = requested_nc
                metadata["coarse_vi_adjusted_Nc"] = adjusted_nc
        elif explicit_cfg.Nh is not None:
            requested_nh = int(explicit_cfg.Nh)
            if full_nfreq % requested_nh != 0:
                adjusted_nh = _closest_divisor(full_nfreq, requested_nh)
                logger.info(
                    "Adjusting explicit coarse VI bin width: "
                    f"N_full={full_nfreq} not divisible by Nh={requested_nh}; "
                    f"using Nh={adjusted_nh}."
                )
                explicit_cfg = CoarseGrainConfig(
                    enabled=True,
                    Nc=None,
                    Nh=adjusted_nh,
                )
                metadata["coarse_vi_requested_Nh"] = requested_nh
                metadata["coarse_vi_adjusted_Nh"] = adjusted_nh

        return explicit_cfg, metadata

    if not vi_cfg.auto_coarse_vi:
        return None, None

    min_full_nfreq = int(vi_cfg.auto_coarse_vi_min_full_nfreq)
    min_required_nfreq = max(1, min_full_nfreq, 20 * k_basis)
    if full_nfreq < min_required_nfreq:
        return None, None

    target_nfreq = max(
        1, min(vi_cfg.auto_coarse_vi_target_nfreq, full_nfreq - 1)
    )
    min_nh = max(2, int(ceil(full_nfreq / float(target_nfreq))))
    nh = _smallest_divisor_geq(full_nfreq, min_nh)
    if nh is None or nh <= 1:
        return None, None
    target_nfreq = full_nfreq // nh

    return _normalize_coarse_grain_config(
        {
            "enabled": True,
            "Nc": None,
            "Nh": nh,
        }
    ), {
        "coarse_vi_mode": "auto",
        "coarse_vi_full_nfreq": full_nfreq,
        "coarse_vi_target_nfreq": target_nfreq,
        "coarse_vi_min_required_nfreq": min_required_nfreq,
        "coarse_vi_basis_target_floor": 10 * k_basis,
    }


def _preprocess_with_run_config(
    data,
    run_config: RunMCMCConfig,
    *,
    include_overlays: bool,
) -> PreprocessedMCMCInput:
    """Preprocess data for one concrete run configuration."""
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
    assert (
        fft_data_cg is not None
    ), "Coarse graining should preserve non-None fft_data"
    fft_data = fft_data_cg
    exclude_bands = _normalize_excluded_frequency_bands(
        run_config.model.exclude_freq_bands
    )
    fft_data = _apply_multivar_frequency_exclusion(fft_data, exclude_bands)
    assert fft_data is not None, "Frequency exclusion should preserve data."

    if include_overlays:
        (
            extra_empirical_psd,
            extra_empirical_labels,
            extra_empirical_styles,
        ) = _build_welch_overlay(raw_ts, fft_data, run_config)
    else:
        extra_empirical_psd = None
        extra_empirical_labels = None
        extra_empirical_styles = None

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


def _preprocess_data(data, config=None, **kwargs) -> PreprocessedMCMCInput:
    if kwargs:
        if config is not None:
            raise ValueError("Cant use both 'config' and kwargs")
        config = _build_config_from_kwargs(**kwargs)

    # This collects all arguments and config options, applies defaults, and performs validation
    run_config = _normalize_run_config(config)
    preproc_input = _preprocess_with_run_config(
        data,
        run_config,
        include_overlays=True,
    )

    coarse_vi_context = None
    vi_cfg = run_config.vi
    if (
        vi_cfg.init_from_vi
        and vi_cfg.use_coarse_vi_for_init
        and not vi_cfg.only_vi
    ):
        vi_cg_config, metadata = _derive_vi_coarse_grain_config(
            preproc_input.processed_data,
            run_config,
        )
        if vi_cg_config is not None and vi_cg_config.enabled:
            vi_run_config = replace(
                run_config,
                coarse_grain_config=vi_cg_config,
            )
            coarse_input = _preprocess_with_run_config(
                data,
                vi_run_config,
                include_overlays=False,
            )
            coarse_vi_context = CoarseVIContext(
                processed_data=coarse_input.processed_data,
                scaled_true_psd=coarse_input.scaled_true_psd,
                sampler_type=coarse_input.sampler_type,
                metadata={
                    **(metadata or {}),
                    "coarse_vi_attempted": 0,
                    "coarse_vi_success": 0,
                    "coarse_vi_nfreq": _get_frequency_count(
                        coarse_input.processed_data
                    ),
                },
            )
            if run_config.diagnostics.verbose:
                coarse_nfreq = coarse_vi_context.metadata["coarse_vi_nfreq"]
                logger.info(
                    "Using a separate coarse VI grid for warm start: "
                    f"mode={coarse_vi_context.metadata['coarse_vi_mode']}, "
                    f"N_full={coarse_vi_context.metadata['coarse_vi_full_nfreq']}, "
                    f"N_vi={coarse_nfreq}"
                )

    return replace(preproc_input, coarse_vi_context=coarse_vi_context)
