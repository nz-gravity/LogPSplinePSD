import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.psd_diagnostics import PSDDiagnostics



def test_plot_whitening_ar2(outdir):
    outdir = os.path.join(outdir, "out_psd_diagnostics")
    os.makedirs(outdir, exist_ok=True)

    ar_data = ARData(order=2, duration=4.0, fs=512.0, sigma=1.0, seed=42)

    # add 0 element to freq, psd, and reference_psd
    freqs = np.concatenate(([0], ar_data.periodogram.freqs))
    psd = np.concatenate(
        (
            [ar_data.periodogram.power[0]],
            np.atleast_1d(ar_data.periodogram.power),
        )
    )
    reference_psd = np.concatenate(
        ([ar_data.psd_theoretical[0]], ar_data.psd_theoretical)
    )

    psd_diag = PSDDiagnostics(
        ts_data=ar_data.ts.y,
        fs=ar_data.fs,
        psd=psd,
        freqs=freqs,
        reference_psd=reference_psd,
        fftlength=1.0,
        overlap=0.5,
    )
    psd_diag.plot_diagnostics(f"{outdir}/ar_psd_diagostics.png")


def test_lvk_psd_diagnostics(outdir):
    from log_psplines.example_datasets.lvk_data import LVKData

    outdir = os.path.join(outdir, "out_psd_diagnostics")
    os.makedirs(outdir, exist_ok=True)

    lvk_data = LVKData.download_data(
        detector="L1", gps_start=1126259462, duration=4, fmin=20, fmax=2048
    )
    lvk_data.plot_psd(fname=os.path.join(outdir, "lvk_psd_analysis.png"))
    ref_psd = scipy.signal.medfilt(lvk_data.psd, kernel_size=65)
    ref_psd = np.where(np.isnan(ref_psd), 0, ref_psd)

    fmask = (lvk_data.freqs >= 20) & (lvk_data.freqs <= 2048)
    freqs = lvk_data.freqs[fmask]
    psd = lvk_data.psd[fmask]
    ref_psd = ref_psd[fmask]

    psd_diag = PSDDiagnostics(
        ts_data=lvk_data.strain,
        fs=4096,
        psd=psd,
        freqs=freqs,
        reference_psd=ref_psd,
        fftlength=0.5,
        overlap=0,
    )
    psd_diag.plot_diagnostics(f"{outdir}/lvk_psd_diagostics.png")
