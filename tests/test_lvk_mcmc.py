import os

import numpy as np

from log_psplines.datatypes import Periodogram
from log_psplines.example_datasets.lvk_data import LVKData
from log_psplines.mcmc import run_mcmc
from log_psplines.psplines.knots_locator import init_knots


def test_lvk_mcmc(outdir):
    out = os.path.join(outdir, "out_lvk_mcmc")
    os.makedirs(out, exist_ok=True)
    lvk_data = LVKData.download_data(
        detector="L1", gps_start=1126259462, duration=4, fmin=256, fmax=512
    )
    lvk_data.plot_psd(fname=os.path.join(out, "lvk_psd_analysis.png"))
    # rescale the PSD to a better scale to work with
    power = lvk_data.psd / np.nanmax(lvk_data.psd) * 1e-3
    pdgrm = Periodogram(
        freqs=lvk_data.freqs,
        power=power,
    )
    pdgrm = pdgrm.cut(256, 512)

    lvk_knots = init_knots(
        n_knots=100,
        periodogram=pdgrm,
        method="lvk",
        knots_plotfn=os.path.join(out, "lvk_psd_analysis.png"),
    )

    run_mcmc(
        pdgrm,
        n_samples=200,
        n_warmup=200,
        outdir=out,
        rng_key=42,
        n_knots=5,
        knot_kwargs=dict(
            method="uniform",
        ),
    )
