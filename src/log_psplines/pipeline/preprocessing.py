"""Pipeline preprocessing helpers.

This module converts input data to the frequency-domain objects consumed by
``build_model_kwargs``. It contains no sampler logic.
"""

from __future__ import annotations

from math import ceil

import numpy as np

from .._jaxtypes import Complex, Float
from .._typecheck import runtime_typecheck
from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..datatypes.multivar_utils import _interp_frequency_indexed_array
from ..logger import logger
from ..preprocessing.coarse_grain import (
    CoarseGrainConfig,
    _closest_divisor,
    _smallest_divisor_geq,
)
from ..preprocessing.data_prep import (
    _apply_frequency_exclusion,
    _coarse_grain_processed_data,
    _normalize_coarse_grain_config,
    _normalize_excluded_frequency_bands,
    _prepare_processed_data,
)
from .config import PipelineConfig

FrequencyData = Periodogram | MultivarFFT


def preprocess_to_freq_domain(data, config: PipelineConfig) -> FrequencyData:
    """Convert time-domain data to frequency-domain inference data."""
    freq_data, _, _ = _prepare_processed_data(data, config)

    cg_config = _normalize_coarse_grain_config(config.coarse_grain_config)
    freq_data, _ = _coarse_grain_processed_data(freq_data, cg_config, None)
    assert freq_data is not None, "Coarse graining removed all input data."

    excl_bands = _normalize_excluded_frequency_bands(config.exclude_freq_bands)
    freq_data = _apply_frequency_exclusion(freq_data, excl_bands)

    return freq_data


def _frequency_count(data: FrequencyData) -> int:
    if isinstance(data, Periodogram):
        return int(len(data.freqs))
    return int(len(data.freq))


def _max_n_knots(n_knots: int | dict[str, int]) -> int:
    if isinstance(n_knots, int):
        return int(n_knots)
    return max(int(value) for value in n_knots.values())


def _unpack_true_psd(
    true_psd,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(freq, psd)`` from accepted ``true_psd`` formats."""
    if true_psd is None:
        return None, None
    if isinstance(true_psd, dict):
        freq = true_psd.get("freq")
        psd = true_psd.get("psd")
        if psd is None:
            raise ValueError(
                "true_psd dict must contain a 'psd' entry (optional 'freq')."
            )
        return None if freq is None else np.asarray(freq), np.asarray(psd)
    if isinstance(true_psd, tuple | list) and len(true_psd) == 2:
        freq = None if true_psd[0] is None else np.asarray(true_psd[0])
        return freq, np.asarray(true_psd[1])
    return None, np.asarray(true_psd)


@runtime_typecheck
def _interp_psd_array(
    psd: Complex[np.ndarray, "f_src ..."] | Float[np.ndarray, "f_src ..."],  # noqa: F722
    freq_src: Float[np.ndarray, f_src],  # noqa: F821
    freq_tgt: Float[np.ndarray, f_tgt],  # noqa: F821
) -> Complex[np.ndarray, "f_tgt ..."] | Float[np.ndarray, "f_tgt ..."]:  # noqa: F722
    """Interpolate PSD arrays onto target frequencies."""
    return _interp_frequency_indexed_array(
        freq_src,
        freq_tgt,
        psd,
        sort_and_dedup=True,
    )


def align_true_psd_to_freq(
    true_psd, data: FrequencyData | None
) -> np.ndarray | None:
    """Align an optional true PSD to the frequency grid of processed data."""
    if true_psd is None:
        return None
    if data is None:
        _, psd = _unpack_true_psd(true_psd)
        return None if psd is None else np.asarray(psd)

    freq_tgt = data.freqs if isinstance(data, Periodogram) else data.freq
    freq_src, psd = _unpack_true_psd(true_psd)
    if psd is None:
        return None
    if freq_src is None:
        if psd.shape[0] == len(freq_tgt):
            return np.asarray(psd)
        logger.warning(
            f"true_psd length {psd.shape[0]} does not match target "
            f"frequencies {len(freq_tgt)}; assuming uniform spacing."
        )
        freq_src = np.linspace(freq_tgt[0], freq_tgt[-1], psd.shape[0])
    elif len(freq_src) != psd.shape[0]:
        raise ValueError(
            "true_psd frequency and value arrays must have matching lengths."
        )
    return _interp_psd_array(
        np.asarray(psd), np.asarray(freq_src), np.asarray(freq_tgt)
    )


def _resolve_explicit_coarse_vi_config(
    data: FrequencyData,
    config: PipelineConfig,
) -> CoarseGrainConfig | None:
    cg_config = _normalize_coarse_grain_config(config.coarse_grain_config_vi)
    if not cg_config.enabled:
        return None

    full_nfreq = _frequency_count(data)
    if cg_config.Nc is not None and full_nfreq % int(cg_config.Nc) != 0:
        requested = int(cg_config.Nc)
        adjusted = _closest_divisor(full_nfreq, requested)
        logger.info(
            f"Adjusting explicit coarse VI bins: N_full={full_nfreq} "
            f"not divisible by Nc={requested}; using Nc={adjusted}."
        )
        return CoarseGrainConfig(enabled=True, Nc=adjusted, Nh=None)

    if cg_config.Nh is not None and full_nfreq % int(cg_config.Nh) != 0:
        requested = int(cg_config.Nh)
        adjusted = _closest_divisor(full_nfreq, requested)
        logger.info(
            f"Adjusting explicit coarse VI bin width: N_full={full_nfreq} "
            f"not divisible by Nh={requested}; using Nh={adjusted}."
        )
        return CoarseGrainConfig(enabled=True, Nc=None, Nh=adjusted)

    return cg_config


def coarse_vi_freq_domain(
    data: FrequencyData,
    config: PipelineConfig,
) -> FrequencyData | None:
    """Return a separate coarse VI grid, or ``None`` when not requested."""
    explicit = _resolve_explicit_coarse_vi_config(data, config)
    if explicit is not None:
        coarse_data, _ = _coarse_grain_processed_data(data, explicit, None)
        assert coarse_data is not None, "Coarse VI gridding removed all data."
        return coarse_data

    if not config.auto_coarse_vi:
        return None

    full_nfreq = _frequency_count(data)
    k_basis = int(_max_n_knots(config.n_knots) + config.degree - 1)
    min_required = max(
        1,
        int(config.auto_coarse_vi_min_full_nfreq),
        20 * k_basis,
    )
    if full_nfreq < min_required:
        return None

    target = max(1, min(config.auto_coarse_vi_target_nfreq, full_nfreq - 1))
    min_nh = max(2, int(ceil(full_nfreq / float(target))))
    nh = _smallest_divisor_geq(full_nfreq, min_nh)
    if nh is None or nh <= 1:
        return None

    cg_config = CoarseGrainConfig(enabled=True, Nc=None, Nh=nh)
    coarse_data, _ = _coarse_grain_processed_data(data, cg_config, None)
    assert coarse_data is not None, "Coarse VI gridding removed all data."
    return coarse_data


__all__ = [
    "FrequencyData",
    "align_true_psd_to_freq",
    "preprocess_to_freq_domain",
    "coarse_vi_freq_domain",
]
