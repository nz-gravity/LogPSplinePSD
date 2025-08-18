import matplotlib.pyplot as plt

from log_psplines.example_datasets.lvk_data import  LVKData
from log_psplines.datatypes import Periodogram, Timeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
from log_psplines.psplines import LogPSplines
import os
import numpy as np

FMIN, FMAX = 20, 1024

out = os.path.join("out_lvk_mcmc")
os.makedirs(out, exist_ok=True)
lvk_data = LVKData.download_data(
    detector="L1",
    gps_start=1126259462,
    duration=4,
    fmin=FMIN,
    fmax=FMAX
)
lvk_data.plot_psd_analysis(
    include_lines=True,
    fname=os.path.join(out, "lvk_psd_analysis.png")
)
# rescale the PSD to a better scale to work with
power  = lvk_data.psd / np.nanmax(lvk_data.psd) * 1e-3
pdgrm = Periodogram(
    freqs= lvk_data.freqs,
    power=power,
)
pdgrm = pdgrm.cut(FMIN, FMAX)


spline_model = LogPSplines.from_periodogram(
    pdgrm,
    n_knots=len(lvk_data.knots_locations),
    degree=3,
    diffMatrixOrder=2,
    knot_kwargs=dict(knots=lvk_data.knots_locations)
)
# plot initial fit with optimised weights
fig, ax = plot_pdgrm(pdgrm=pdgrm, spline_model=spline_model, figsize=(12, 6))
ax.set_xscale('linear')
fig.savefig(os.path.join(out, f"test_spline_init.png"))



idata  = run_mcmc(
    pdgrm,
    sampler='mh',
    n_samples=2000,
    n_warmup=2000,
    outdir=out,
    rng_key=42,
    knot_kwargs=dict(knots=lvk_data.knots_locations)
)

fig, ax = plot_pdgrm(idata=idata, figsize=(12, 6))
ax.set_xscale('linear')
fig.savefig(os.path.join(out, f"test_mcmc.png"))

fig, ax = plot_pdgrm(idata=idata, figsize=(12, 6))
ax.set_xscale('log')
fig.savefig(os.path.join(out, f"test_mcmc_log.png"))