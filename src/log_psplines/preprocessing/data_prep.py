from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import (
    EmpiricalPSD,
    MultivarFFT,
    MultivariateTimeseries,
)
from ..datatypes.univar import Timeseries
from ..logger import logger
from ..pipeline.config import PipelineConfig
from .coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    apply_coarse_graining_univar,
    compute_binning_structure,
)
from .configs import SamplerName


def _normalize_coarse_grain_config(
    coarse_grain_config: Optional[CoarseGrainConfig | dict],
) -> CoarseGrainConfig:
    if coarse_grain_config is None:
        return CoarseGrainConfig()
    if isinstance(coarse_grain_config, dict):
        return CoarseGrainConfig(**coarse_grain_config)
    return coarse_grain_config


def _normalize_excluded_frequency_bands(
    bands: Sequence[tuple[float, float]] | None,
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


def _build_frequency_exclusion_mask(
    freq: np.ndarray,
    bands: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Return a boolean mask retaining frequencies outside ``bands``."""
    freq = np.asarray(freq, dtype=np.float64)
    mask = np.ones(freq.shape, dtype=bool)
    for low, high in bands:
        mask &= ~((freq >= float(low)) & (freq <= float(high)))
    return mask


def _filter_empirical_psd(
    empirical: EmpiricalPSD,
    bands: Sequence[tuple[float, float]],
) -> EmpiricalPSD:
    """Return an EmpiricalPSD with excluded frequency bands removed."""
    if len(bands) == 0:
        return empirical
    mask = _build_frequency_exclusion_mask(empirical.freq, bands)
    if not np.any(mask):
        raise ValueError(
            "Frequency masking removed all empirical overlay bins."
        )
    return EmpiricalPSD(
        freq=np.asarray(empirical.freq)[mask],
        psd=np.asarray(empirical.psd)[mask],
        coherence=np.asarray(empirical.coherence)[mask],
        channels=empirical.channels,
    )


def _apply_multivar_frequency_exclusion(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    bands: Sequence[tuple[float, float]],
) -> Optional[Union[Periodogram, MultivarFFT]]:
    """Apply post-coarse excluded bands to multivariate processed data."""
    if processed_data is None or not isinstance(processed_data, MultivarFFT):
        return processed_data
    if len(bands) == 0:
        return processed_data
    return processed_data.exclude_frequency_bands(bands)


def _coarse_grain_processed_data(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    cg_config: CoarseGrainConfig,
    scaled_true_psd: Optional[np.ndarray],
) -> tuple[
    Optional[Union[Periodogram, MultivarFFT]],
    Optional[np.ndarray],
]:
    """Apply coarse graining to the already-processed data if configured."""
    if processed_data is None or not cg_config.enabled:
        return processed_data, scaled_true_psd

    if isinstance(processed_data, Periodogram):
        spec = compute_binning_structure(
            processed_data.freqs,
            Nc=cg_config.Nc,
            Nh=cg_config.Nh,
        )

        freqs = processed_data.freqs
        power_coarse = apply_coarse_graining_univar(
            np.asarray(processed_data.power), spec, freqs
        )

        processed_data = Periodogram(
            spec.f_coarse,
            power_coarse,
            scaling_factor=processed_data.scaling_factor,
            Nh=int(spec.Nh),
        )

        nl = int(spec.Nc * spec.Nh)
        percent_retained = 100.0 * float(spec.Nc) / float(nl)
        percent_decimated = 100.0 - percent_retained
        logger.info(
            f"Coarse-grained periodogram: {spec} "
            f"(kept {percent_retained:.1f}% of points, "
            f"decimated {percent_decimated:.1f}%)."
        )

        if scaled_true_psd is not None:
            try:
                true_coarse = apply_coarse_graining_univar(
                    np.asarray(scaled_true_psd), spec, freqs
                )
                scaled_true_psd = true_coarse
            except Exception:
                logger.warning(
                    "Could not coarse-grain provided true_psd; leaving unchanged."
                )

        return processed_data, scaled_true_psd

    if isinstance(processed_data, MultivarFFT):
        spec = compute_binning_structure(
            processed_data.freq,
            Nc=cg_config.Nc,
            Nh=cg_config.Nh,
        )
        processed_data = apply_coarse_grain_multivar_fft(processed_data, spec)
        nl = int(spec.Nc * spec.Nh)
        percent_retained = 100.0 * float(spec.Nc) / float(nl)
        percent_decimated = 100.0 - percent_retained
        logger.info(
            f"Coarse-grained multivariate FFT: {spec} "
            f"(kept {percent_retained:.1f}% of points, "
            f"decimated {percent_decimated:.1f}%)."
        )
        return processed_data, scaled_true_psd

    return processed_data, scaled_true_psd


def _prepare_processed_data(
    data: Union[Timeseries, MultivariateTimeseries],
    config: PipelineConfig,
) -> tuple[
    Union[Periodogram, MultivarFFT],
    Optional[MultivariateTimeseries],
    SamplerName,
]:
    raw_multivar_ts: Optional[MultivariateTimeseries] = None

    standardized_ts = data.standardise_for_psd()

    # Infer sampler type from input data type
    if isinstance(data, Timeseries):
        # Univariate: use NUTS
        sampler: SamplerName = "nuts"
        processed = standardized_ts.to_periodogram(
            fmin=config.fmin,
            fmax=config.fmax,
        )
    else:
        # Multivariate: prefer multivar_blocked_nuts, fall back to nuts
        sampler = "multivar_blocked_nuts"
        raw_multivar_ts = data
        processed = standardized_ts.to_wishart_stats(
            Nb=config.Nb,
            fmin=config.fmin,
            fmax=config.fmax,
            window=config.wishart_window,
            detrend=config.wishart_detrend,
            wishart_floor_fraction=config.wishart_floor_fraction,
        )

    if config.verbose:
        logger.info(
            f"Standardized data: original scale ~{processed.scaling_factor:.2e}"
        )
        logger.info(f"Inferred sampler type: {sampler}")

    if processed is None:
        raise ValueError("Processed data unexpectedly None.")
    return processed, raw_multivar_ts, sampler


def _build_welch_overlay(
    raw_multivar_ts: Optional[MultivariateTimeseries],
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    config: PipelineConfig,
) -> tuple[
    list[EmpiricalPSD] | None,
    list[str] | None,
    list[dict] | None,
]:
    if raw_multivar_ts is None:
        return None, None, None
    if not isinstance(processed_data, MultivarFFT):
        return None, None, None

    try:
        welch_emp = raw_multivar_ts.get_empirical_psd(
            nperseg=config.welch_nperseg,
            noverlap=config.welch_noverlap,
            window=config.welch_window,
        )
        freq_target = np.asarray(processed_data.freq, dtype=float)
        f_lo = float(freq_target[0])
        f_hi = float(freq_target[-1])
        keep = (
            (welch_emp.freq > 0.0)
            & (welch_emp.freq >= f_lo)
            & (welch_emp.freq <= f_hi)
        )
        if not np.any(keep):
            if config.verbose:
                logger.warning(
                    "Welch overlay requested but produced no in-range positive-frequency bins; skipping."
                )
            return None, None, None

        overlay = EmpiricalPSD(
            freq=np.asarray(welch_emp.freq)[keep],
            psd=np.asarray(welch_emp.psd)[keep],
            coherence=np.asarray(welch_emp.coherence)[keep],
            channels=welch_emp.channels,
        )
        overlay = _filter_empirical_psd(
            overlay,
            _normalize_excluded_frequency_bands(config.exclude_freq_bands),
        )
        return (
            [overlay],
            ["Welch"],
            [
                {
                    "color": "0.25",
                    "lw": 0.9,
                    "alpha": 0.18,
                    "ls": ":",
                    "zorder": -10,
                }
            ],
        )
    except Exception as exc:
        if config.verbose:
            logger.warning(f"Could not compute Welch overlay: {exc}")
        return None, None, None
