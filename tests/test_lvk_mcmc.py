from log_psplines.example_datasets.lvk_data import  LVKData
from log_psplines.datatypes import Periodogram, Timeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
import os

def test_lvk_mcmc(outdir):
    out = os.path.join(outdir, "out_lvk_mcmc")
    os.makedirs(out, exist_ok=True)
    lvk_data = LVKData.download_data(
        detector="L1",
        gps_start=1126259462,
        duration=4,
        fmin=20,
        fmax=1024
    )
    lvk_data.plot_psd_analysis(
        include_lines=True,
        fname=os.path.join(out, "lvk_psd_analysis.png")
    )

    ts = Timeseries(
        t=lvk_data.time,
        y=lvk_data.strain
    )
    pdgrm =  ts.standardise().to_periodogram()
    run_mcmc(
        pdgrm,
        n_samples=2000,
        n_warmup=2000,
        outdir=out,
        rng_key=42,
        knot_kwargs=dict(knots=lvk_data.knots_locations)
    )