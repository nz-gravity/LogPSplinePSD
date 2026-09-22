"""Optional WDM transform adapter; inference consumes plain powers/counts."""

import numpy as np

from log_psplines.data.spectral import PowerSpectrum
from log_psplines.data.timeseries import TimeSeries


def wdm_periodogram(
    series: TimeSeries,
    *,
    nt: int,
    trim_time: int = 1,
    trim_low: int = 1,
    trim_high: int = 1,
) -> PowerSpectrum:
    """Transform one channel, preserving WDM coefficient-variance units.

    Time is divided by the full duration. No automatic cropping, calibration,
    coarse graining or detrending is performed.
    """
    from wdm_transform import TimeSeries as WDMTimeSeries

    if series.data.shape[1] != 1:
        raise NotImplementedError(
            "WDM inference currently supports one channel"
        )
    n = len(series.data)
    if not isinstance(nt, int) or nt <= 0 or n % nt or nt % 2 or (n // nt) % 2:
        raise ValueError(
            "WDM requires N divisible by nt and both nt and N/nt even"
        )
    for trim in (trim_time, trim_low, trim_high):
        if not isinstance(trim, int) or trim < 0:
            raise ValueError("trimming must use non-negative integers")
    steps = np.diff(series.t)
    if (
        not np.isfinite(series.data).all()
        or steps.size == 0
        or not np.isfinite(steps).all()
        or steps[0] <= 0
        or not np.allclose(steps, steps[0])
    ):
        raise ValueError("WDM requires finite data and uniformly spaced times")
    wdm = WDMTimeSeries(series.data[:, 0], dt=float(steps[0])).to_wdm(nt=nt)
    coeffs = np.asarray(wdm.coeffs)
    if coeffs.ndim == 3 and coeffs.shape[0] == 1:
        coeffs = coeffs[0]
    keep_t = np.arange(trim_time, wdm.nt - trim_time)
    keep_f = np.arange(trim_low, wdm.nf + 1 - trim_high)
    if not keep_t.size or not keep_f.size:
        raise ValueError("WDM trimming leaves an empty grid")
    return PowerSpectrum(
        power=coeffs[np.ix_(keep_t, keep_f)] ** 2,
        counts=1,
        frequency=np.asarray(wdm.freq_grid)[keep_f],
        time=np.asarray(wdm.time_grid)[keep_t] / wdm.duration,
        units="WDM coefficient variance",
    )
