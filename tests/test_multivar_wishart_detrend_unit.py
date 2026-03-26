from __future__ import annotations

import numpy as np

from log_psplines.datatypes.multivar import MultivarFFT


def test_compute_wishart_constant_detrend_reduces_first_bin_dc_leakage():
    fs = 1.0
    n = 2048
    t = np.arange(n, dtype=float) / fs
    data = (3.0 + 0.2 * np.sin(2.0 * np.pi * 0.02 * t))[:, None]

    fft_none = MultivarFFT.compute_wishart(
        data,
        fs=fs,
        Nb=4,
        window="hann",
        detrend=False,
    )
    fft_const = MultivarFFT.compute_wishart(
        data,
        fs=fs,
        Nb=4,
        window="hann",
        detrend="constant",
    )

    psd_none = np.real(fft_none.raw_psd[:, 0, 0])
    psd_const = np.real(fft_const.raw_psd[:, 0, 0])

    assert psd_const[0] < psd_none[0] * 1e-2


def test_compute_wishart_accepts_linear_detrend():
    fs = 1.0
    n = 1024
    t = np.arange(n, dtype=float) / fs
    data = (0.5 * t / n + np.sin(2.0 * np.pi * 0.03 * t))[:, None]

    fft_linear = MultivarFFT.compute_wishart(
        data,
        fs=fs,
        Nb=2,
        window="hann",
        detrend="linear",
    )

    assert fft_linear.N > 0
    assert fft_linear.raw_psd is not None
