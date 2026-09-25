"""Rectangular likelihood partitions for time-varying power spectra.

The pilot chooses boundaries only. The likelihood always pools original,
unsmoothed power and the exact counts carried by each native cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from log_psplines.data.spectral import PowerSpectrum


def _starts(value: np.ndarray, size: int, name: str) -> np.ndarray:
    starts = np.asarray(value)
    if (
        starts.ndim != 1
        or starts.size == 0
        or not np.issubdtype(starts.dtype, np.integer)
        or starts[0] != 0
        or starts[-1] >= size
        or np.any(np.diff(starts) <= 0)
    ):
        raise ValueError(f"{name} must be increasing integer starts from 0")
    return starts.astype(int, copy=True)


@dataclass(frozen=True)
class PowerPartition:
    """Zero-based starts of separable time and frequency likelihood blocks."""

    time_starts: np.ndarray
    frequency_starts: np.ndarray


def mask_power(data: PowerSpectrum, mask: np.ndarray) -> PowerSpectrum:
    """Exclude native cells while retaining their grid coordinates."""
    if data.time is None:
        raise ValueError("mask_power requires a time-frequency PowerSpectrum")
    mask = np.asarray(mask)
    if mask.shape != data.power.shape or mask.dtype != np.bool_:
        raise ValueError("mask must be boolean with the native power shape")
    return PowerSpectrum(
        np.where(mask, data.power, 0.0),
        np.where(mask, data.counts, 0.0),
        data.frequency,
        data.time,
        data.units,
    )


def select_power_partition(
    pilot_log_psd: np.ndarray,
    time: np.ndarray,
    *,
    time_bin: int,
    max_frequency_bin: int,
    max_log_range: float,
    counts: np.ndarray | None = None,
    max_time_gap: float | None = None,
    time_breaks: np.ndarray | None = None,
) -> PowerPartition:
    """Choose bins from a finite training-only pilot, splitting at gaps/masks.

    ``pilot_log_psd`` has shape (pilot_times, frequencies). It may be a
    smoothed guide but is never used as likelihood power. ``time_breaks`` is
    a boolean vector of length ``len(time)`` marking starts of new runs.
    Bin size and pilot tolerance must be chosen by the caller.
    """
    pilot = np.asarray(pilot_log_psd, dtype=float)
    time = np.asarray(time, dtype=float)
    if pilot.ndim != 2 or not pilot.size or not np.isfinite(pilot).all():
        raise ValueError("pilot_log_psd must be a finite nonempty 2-D array")
    if (
        time.ndim != 1
        or not time.size
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("time must be finite and strictly increasing")
    if not isinstance(time_bin, (int, np.integer)) or time_bin < 1:
        raise ValueError("time_bin must be a positive integer")
    if (
        not isinstance(max_frequency_bin, (int, np.integer))
        or max_frequency_bin < 1
    ):
        raise ValueError("max_frequency_bin must be a positive integer")
    if not np.isfinite(max_log_range) or max_log_range <= 0:
        raise ValueError("max_log_range must be finite and positive")
    if counts is not None:
        counts = np.asarray(counts)
        if counts.shape != (time.size, pilot.shape[1]):
            raise ValueError(
                "counts must match the native time/frequency grid"
            )
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("counts must be finite and non-negative")

    frequency_starts = [0]
    low = high = pilot[:, 0].copy()
    start = 0
    for frequency in range(1, pilot.shape[1]):
        new_low = np.minimum(low, pilot[:, frequency])
        new_high = np.maximum(high, pilot[:, frequency])
        if (
            frequency - start >= max_frequency_bin
            or np.max(new_high - new_low) > max_log_range
        ):
            frequency_starts.append(frequency)
            start = frequency
            low = high = pilot[:, frequency].copy()
        else:
            low, high = new_low, new_high

    breaks = np.zeros(time.size, dtype=bool)
    breaks[0] = True
    if time.size > 1:
        gaps = np.diff(time)
        threshold = (
            1.5 * np.median(gaps) if max_time_gap is None else max_time_gap
        )
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("max_time_gap must be finite and positive")
        breaks[1:] |= gaps > threshold
    if counts is not None and time.size > 1:
        breaks[1:] |= np.any((counts[1:] > 0) != (counts[:-1] > 0), axis=1)
    if time_breaks is not None:
        supplied = np.asarray(time_breaks)
        if supplied.shape != breaks.shape or supplied.dtype != np.bool_:
            raise ValueError("time_breaks must be boolean with time shape")
        breaks |= supplied
    time_starts = []
    run_starts = np.flatnonzero(breaks)
    for run_start, run_stop in zip(
        run_starts, np.r_[run_starts[1:], time.size], strict=True
    ):
        time_starts.extend(range(int(run_start), int(run_stop), int(time_bin)))
    return PowerPartition(
        np.asarray(time_starts), np.asarray(frequency_starts)
    )


def coarse_grain_power(
    data: PowerSpectrum, partition: PowerPartition
) -> PowerSpectrum:
    """Pool native power and exact counts; use native-center block means.

    A block's single spline evaluation approximates its constituent PSDs.
    Empty blocks stay present with zero power/count so the tensor grid remains
    rectangular, and contribute zero to the Whittle likelihood.
    """
    if data.time is None:
        raise ValueError("coarse_grain_power requires time-frequency power")
    ts = _starts(partition.time_starts, len(data.time), "time_starts")
    fs = _starts(
        partition.frequency_starts, len(data.frequency), "frequency_starts"
    )
    power = np.add.reduceat(
        np.add.reduceat(data.power, ts, axis=0), fs, axis=1
    )
    counts = np.add.reduceat(
        np.add.reduceat(data.counts, ts, axis=0), fs, axis=1
    )
    tsize = np.diff(np.r_[ts, len(data.time)])
    fsize = np.diff(np.r_[fs, len(data.frequency)])
    time = np.add.reduceat(data.time, ts) / tsize
    frequency = np.add.reduceat(data.frequency, fs) / fsize
    return PowerSpectrum(power, counts, frequency, time, data.units)
