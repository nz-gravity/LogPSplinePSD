"""Pipeline preprocessing helpers.

This module converts input data to the frequency-domain objects consumed by
``inference.model.prepare_model``. It contains no sampler logic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from log_psplines.config import StationaryConfig
from log_psplines.data.spectral import WishartData
from log_psplines.data.spectral_utils import _interp_frequency_indexed_array
from log_psplines.data.timeseries import TimeSeries
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)

from ..logger import logger


def preprocess_to_freq_domain(
    data: TimeSeries, config: StationaryConfig
) -> WishartData:
    """Standardize, form Wishart data, bin, and exclude frequencies."""
    if not isinstance(data, TimeSeries):
        data = TimeSeries(data=np.asarray(data.data), t=np.asarray(data.t))
    processed = data.standardise_for_psd().to_wishart_stats(
        Nb=config.Nb,
        fmin=config.fmin,
        fmax=config.fmax,
        window=config.wishart_window,
        detrend=config.wishart_detrend,
        wishart_floor_fraction=config.wishart_floor_fraction,
    )
    if config.verbose:
        logger.info(
            f"Standardized data: scale ~{processed.scaling_factor:.2e}"
        )

    coarse_config = config.coarse_grain_config
    if coarse_config is None:
        coarse_config = CoarseGrainConfig()
    elif isinstance(coarse_config, dict):
        coarse_config = CoarseGrainConfig(**coarse_config)
    if coarse_config.enabled:
        spec = compute_binning_structure(
            processed.freq, Nc=coarse_config.Nc, Nh=coarse_config.Nh
        )
        processed = apply_coarse_grain_multivar_fft(processed, spec)
        kept_percent = 100.0 / float(spec.Nh)
        logger.info(
            f"Coarse-grained multivariate FFT: {spec} "
            f"(kept {kept_percent:.1f}%, "
            f"decimated {100.0 - kept_percent:.1f}%)."
        )

    bands = _normalize_excluded_frequency_bands(config.exclude_freq_bands)
    if bands:
        mask = np.ones(processed.freq.shape, dtype=bool)
        for low, high in bands:
            mask &= ~((processed.freq >= low) & (processed.freq <= high))
        n_excluded = int((~mask).sum())
        if n_excluded:
            if not np.any(mask):
                raise ValueError(
                    "Frequency masking removed all inference bins."
                )
            logger.info(
                f"Null-band excision: removing {n_excluded} bins across "
                f"{len(bands)} band(s). "
                f"{int(np.count_nonzero(mask))} bins retained."
            )
            processed = processed.apply_mask(mask)
    return processed


def _normalize_excluded_frequency_bands(
    bands: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float], ...]:
    """Return sorted, merged excluded frequency bands."""
    if bands is None:
        return ()

    cleaned: list[tuple[float, float]] = []
    for band in bands:
        if len(band) != 2:
            raise ValueError(
                "Each excluded frequency band must be a length-2 tuple."
            )
        low = float(band[0])
        high = float(band[1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("Excluded frequency bands must be finite.")
        if high < low:
            low, high = high, low
        cleaned.append((low, high))

    if not cleaned:
        return ()

    cleaned.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = [cleaned[0]]
    for low, high in cleaned[1:]:
        prev_low, prev_high = merged[-1]
        if low <= prev_high:
            merged[-1] = (prev_low, max(prev_high, high))
        else:
            merged.append((low, high))
    return tuple(merged)


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


def _interp_psd_array(
    psd: np.ndarray,
    freq_src: np.ndarray,
    freq_tgt: np.ndarray,
) -> np.ndarray:
    """Interpolate PSD arrays onto target frequencies."""
    return _interp_frequency_indexed_array(
        freq_src,
        freq_tgt,
        psd,
        sort_and_dedup=True,
    )


def align_true_psd_to_freq(
    true_psd, data: WishartData | None
) -> np.ndarray | None:
    """Align an optional true PSD to the frequency grid of processed data."""
    if true_psd is None:
        return None
    if data is None:
        _, psd = _unpack_true_psd(true_psd)
        return None if psd is None else np.asarray(psd)

    freq_tgt = data.freq
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


__all__ = [
    "align_true_psd_to_freq",
    "preprocess_to_freq_domain",
]
