from log_psplines.example_datasets.lvk_data import  LVKData
from log_psplines.datatypes import Periodogram, Timeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
import os
import numpy as np

def test_lvk_mcmc(outdir):
    out = os.path.join(outdir, "out_lvk_mcmc")
    os.makedirs(out, exist_ok=True)
    lvk_data = LVKData.download_data(
        detector="L1",
        gps_start=1126259462,
        duration=4,
        fmin=256,
        fmax=512
    )
    lvk_data.plot_psd_analysis(
        fname=os.path.join(out, "lvk_psd_analysis.png")
    )
    # rescale the PSD to a better scale to work with
    power = lvk_data.psd / np.nanmax(lvk_data.psd) * 1e-3
    pdgrm = Periodogram(
        freqs=lvk_data.freqs,
        power=power,
    )
    pdgrm = pdgrm.cut(256, 512)
    run_mcmc(
        pdgrm,
        n_samples=200,
        n_warmup=200,
        outdir=out,
        rng_key=42,
        n_knots=10,
    )