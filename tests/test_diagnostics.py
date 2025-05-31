import os.path

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.psd_diagnostics import PSDDiagnostics


def test_plot_whitening_ar2(outdir):
    ar_data = ARData(order=2, duration=8.0, fs=1024.0, sigma=1.0, seed=42)
    psd_diag = PSDDiagnostics(
        ts_data=ar_data.ts,
        fs=ar_data.fs,
        psd=ar_data.psd_theoretical,
        freqs=ar_data.freqs,
    )
    psd_diag.plot_whiten_diagnostics(f"{outdir}/psd_diagostics.png")
