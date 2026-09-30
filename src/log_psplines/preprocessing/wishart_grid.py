"""Local rectangular FFT coefficients and fixed rectangular statistic bins."""

from dataclasses import replace

import numpy as np

from log_psplines.data.spectral_utils import Y_to_U, _as_positive_int
from log_psplines.data.timeseries import TimeSeries
from log_psplines.data.wishart_grid import WishartGridData


def local_wishart_grid(
    series: TimeSeries,
    segment_length: int,
    *,
    units: str = "data_unit^2/Hz",
) -> WishartGridData:
    """Nonoverlapping rectangular FFTs, in physical one-sided PSD units.

    x=sqrt(2/(fs*L))*rfft(segment); E[x x^H] approximates S(t,f).
    There is no standardization, taper, overlap, detrending, or extra ENBW
    correction. DC and even-L Nyquist are excluded because they are real.
    Time is the arithmetic mean of sample times in each segment; frequency
    is k*fs/L. Require full segments and uniform sampling. Independent,
    proper complex ordinates and local stationarity are Whittle assumptions,
    exact for interior Fourier bins of white Gaussian data, approximate for
    finite colored/time-varying records. Each native cell has count one.
    """
    length = _as_positive_int("segment_length", segment_length)
    values, time = np.asarray(series.data), np.asarray(series.t)
    if length < 3 or len(time) < length or len(time) % length:
        raise ValueError("require segment_length >= 3 and complete segments")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or not np.allclose(np.diff(time), time[1] - time[0])
    ):
        raise ValueError("local FFTs require finite uniformly sampled data")
    fs = series.fs
    segments = values.reshape(-1, length, series.p)
    stop = -1 if length % 2 == 0 else None
    coefficients = np.fft.rfft(segments, axis=1)[:, 1:stop]
    coefficients *= np.sqrt(2.0 / (fs * length))
    return WishartGridData.from_coefficients(
        coefficients,
        time.reshape(-1, length).mean(axis=1),
        np.fft.rfftfreq(length, d=1.0 / fs)[1:stop],
        units=units,
        normalization="sqrt(2/(fs*segment_length))*rfft; boxcar; interior bins",
    )


def coarse_grain_wishart_grid(
    data: WishartGridData, *, time_bin: int = 1, frequency_bin: int = 1
) -> WishartGridData:
    """Sum Y and counts in disjoint fixed rectangles, including edge bins.

    Bin coordinates are unweighted means of the input cell coordinates.
    Already-pooled inputs retain counts and original reference grids. A
    spectrum evaluated once at each bin centre is exact only when S is
    constant within that bin under the independent-coefficient model.
    Pooling over time loses temporal detail, including cross-spectrum phase.
    """
    time_bin = _as_positive_int("time_bin", time_bin)
    frequency_bin = _as_positive_int("frequency_bin", frequency_bin)
    if time_bin == frequency_bin == 1:
        return data
    ts = np.arange(0, len(data.time), time_bin)
    fs = np.arange(0, len(data.frequency), frequency_bin)
    sums = np.add.reduceat(np.add.reduceat(data.Y, ts, axis=0), fs, axis=1)
    counts = np.add.reduceat(
        np.add.reduceat(data.counts, ts, axis=0), fs, axis=1
    )
    shape = sums.shape
    factors = Y_to_U(sums.reshape(-1, data.p, data.p)).reshape(shape)
    return replace(
        data,
        u_re=factors.real,
        u_im=factors.imag,
        counts=counts,
        time=np.add.reduceat(data.time, ts)
        / np.diff(np.r_[ts, len(data.time)]),
        frequency=np.add.reduceat(data.frequency, fs)
        / np.diff(np.r_[fs, len(data.frequency)]),
    )
