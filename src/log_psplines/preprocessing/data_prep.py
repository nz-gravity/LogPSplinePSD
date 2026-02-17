from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import (
    EmpiricalPSD,
    MultivarFFT,
    MultivariateTimeseries,
)
from ..datatypes.univar import Timeseries
from ..logger import logger
from .coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    apply_coarse_graining_univar,
    compute_binning_structure,
)
from .configs import RunMCMCConfig, SamplerName


def _normalize_coarse_grain_config(
    coarse_grain_config: Optional[CoarseGrainConfig | dict],
) -> CoarseGrainConfig:
    if coarse_grain_config is None:
        return CoarseGrainConfig()
    if isinstance(coarse_grain_config, dict):
        return CoarseGrainConfig(**coarse_grain_config)
    return coarse_grain_config


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

        selection_mask = spec.selection_mask
        power_selected = np.asarray(processed_data.power[selection_mask])
        freqs_selected = processed_data.freqs[selection_mask]
        power_coarse = apply_coarse_graining_univar(
            power_selected, spec, freqs_selected
        )

        processed_data = Periodogram(
            spec.f_coarse,
            power_coarse,
            scaling_factor=processed_data.scaling_factor,
            Nh=int(spec.Nh),
        )

        logger.info(f"Coarse-grained periodogram: {spec}")

        if scaled_true_psd is not None:
            try:
                true_selected = np.asarray(scaled_true_psd)[selection_mask]
                true_coarse = apply_coarse_graining_univar(
                    true_selected, spec, freqs_selected
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
        logger.info(f"Coarse-grained multivariate FFT: {spec}")
        return processed_data, scaled_true_psd

    return processed_data, scaled_true_psd


def _truncate_frequency_range(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    fmin: Optional[float],
    fmax: Optional[float],
) -> Optional[Union[Periodogram, MultivarFFT]]:
    if processed_data is None or (fmin is None and fmax is None):
        return processed_data

    freq_attr = "freqs" if isinstance(processed_data, Periodogram) else "freq"
    freqs = np.asarray(getattr(processed_data, freq_attr), dtype=float)
    if freqs.size == 0:
        raise ValueError("Processed data contains no frequencies.")

    freq_min = float(freqs[0])
    freq_max = float(freqs[-1])
    lower = freq_min if fmin is None else float(fmin)
    upper = freq_max if fmax is None else float(fmax)

    lower = min(max(lower, freq_min), freq_max)
    upper = min(max(upper, freq_min), freq_max)
    if upper < lower:
        upper = lower

    truncated = processed_data.cut(lower, upper)
    n_points = (
        truncated.n if isinstance(truncated, Periodogram) else truncated.N
    )
    if n_points == 0:
        raise ValueError(
            "Frequency truncation removed all data points. Check fmin/fmax."
        )
    return truncated


def _prepare_processed_data(
    data: Union[Timeseries, MultivariateTimeseries],
    config: RunMCMCConfig,
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
            fmin=config.model.fmin,
            fmax=config.model.fmax,
        )
    else:
        # Multivariate: prefer multivar_blocked_nuts, fall back to nuts
        sampler = "multivar_blocked_nuts"
        raw_multivar_ts = data
        processed = standardized_ts.to_wishart_stats(
            Nb=config.Nb,
            fmin=config.model.fmin,
            fmax=config.model.fmax,
        )

    if config.diagnostics.verbose:
        logger.info(
            f"Standardized data: original scale ~{processed.scaling_factor:.2e}"
        )
        logger.info(f"Inferred sampler type: {sampler}")

    processed = _truncate_frequency_range(
        processed,
        config.model.fmin,
        config.model.fmax,
    )
    if processed is None:
        raise ValueError("Processed data unexpectedly None.")
    return processed, raw_multivar_ts, sampler


def _build_welch_overlay(
    raw_multivar_ts: Optional[MultivariateTimeseries],
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    config: RunMCMCConfig,
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
            if config.diagnostics.verbose:
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
        if config.diagnostics.verbose:
            logger.warning(f"Could not compute Welch overlay: {exc}")
        return None, None, None
