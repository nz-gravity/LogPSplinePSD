import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from log_psplines.bayesian_model import whittle_lnlike
from log_psplines.datatypes import Periodogram, Timeseries
from log_psplines.example_datasets.ar_data import ARData
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_basis, plot_pdgrm, plot_trace
from log_psplines.psplines import LogPSplines


def test_spline_init(mock_pdgrm: Periodogram, outdir):
    t0 = time.time()
    ln_pdgrm = jnp.log(mock_pdgrm.power)
    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
    )
    zero_weights = jnp.zeros(spline_model.weights.shape)
    # Compute the log likelihood with weights at 0
    lnl_initial = whittle_lnlike(ln_pdgrm, spline_model(zero_weights))
    lnl_final = whittle_lnlike(ln_pdgrm, spline_model())
    runtime = float(time.time()) - t0

    plot_basis(
        spline_model.basis, os.path.join(outdir, "test_spline_init_basis.png")
    )

    assert lnl_final > lnl_initial
    assert runtime < 5

    fig, ax = plot_pdgrm(mock_pdgrm, spline_model)
    fig.savefig(os.path.join(outdir, "test_spline_init.png"))


def test_mcmc(mock_pdgrm: Periodogram, outdir):

    ar_data = ARData(order=2, duration=8.0, fs=1024.0, sigma=1.0, seed=42)
    pdgm = (
        Timeseries(ar_data.ts, ar_data.psd_theoretical)
        .to_periodogram()
        .highpass(5)
    )

    mcmc, spline_model = run_mcmc(
        mock_pdgrm, n_knots=30, num_samples=250, num_warmup=1000
    )
    samples = mcmc.get_samples()

    fig, ax = plot_pdgrm(mock_pdgrm, spline_model, samples["weights"])
    fig.savefig(os.path.join(outdir, f"test_mcmc.png"))
    plt.close(fig)

    assert mcmc.runtime < 30
    plot_trace(mcmc, os.path.join(outdir, "traceplot.png"))
