import os
import time

import jax.numpy as jnp

from log_psplines.datasets import Periodogram
from log_psplines.psplines import LogPSplines
from log_psplines.plotting import plot_pdgrm
from log_psplines.bayesian_model import whittle_lnlike
from log_psplines.mcmc import run_mcmc


def test_spline_init(mock_pdgrm: Periodogram, outdir):
    t0 = time.time()
    ln_pdgrm = jnp.log(mock_pdgrm.power)
    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=20,
        degree=3,
        diffMatrixOrder=2,
    )
    zero_weights = jnp.zeros(spline_model.weights.shape)
    # Compute the log likelihood with weights at 0
    lnl_initial = whittle_lnlike(ln_pdgrm, spline_model(zero_weights))
    lnl_final = whittle_lnlike(ln_pdgrm, spline_model())
    runtime = float(time.time()) - t0

    assert lnl_final > lnl_initial
    assert runtime < 5

    fig, ax = plot_pdgrm(mock_pdgrm, spline_model)
    fig.savefig(os.path.join(outdir, "test_spline_init.png"))


def test_mcmc(mock_pdgrm: Periodogram, outdir):
    t0 = time.time()
    samples, spline_model = run_mcmc(mock_pdgrm)
    runtime = float(time.time()) - t0

    assert runtime < 30

    fig, ax = plot_pdgrm(mock_pdgrm, spline_model, samples['weights'])
    fig.savefig(os.path.join(outdir, "test_mcmc.png"))