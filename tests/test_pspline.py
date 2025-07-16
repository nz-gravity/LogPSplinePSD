import os
import time

import jax.numpy as jnp

from log_psplines.datatypes import Periodogram
from log_psplines.plotting import plot_basis, plot_pdgrm
from log_psplines.psplines import LogPSplines
from log_psplines.samplers.base_sampler import log_likelihood


def test_spline_init(mock_pdgrm: Periodogram, outdir):
    out = os.path.join(outdir, "out_spline_init")
    os.makedirs(out, exist_ok=True)

    # init splines
    t0 = time.time()
    ln_pdgrm = jnp.log(mock_pdgrm.power)
    zero_param = jnp.zeros(ln_pdgrm.shape[0])
    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
    )
    zero_weights = jnp.zeros(spline_model.weights.shape)  # model == zeros
    optim_weights = spline_model.weights

    # compute LnL at init and optimized weights
    lnl_args = (ln_pdgrm, spline_model.basis, zero_param)
    lnl_initial = log_likelihood(zero_weights, *lnl_args)
    lnl_final = log_likelihood(optim_weights, *lnl_args)
    runtime = float(time.time()) - t0

    print(
        f"LnL initial: {lnl_initial}, LnL final: {lnl_final}, runtime: {runtime:.2f} seconds"
    )

    # plotting for verification
    fig, ax = plot_pdgrm(mock_pdgrm, spline_model)
    fig.savefig(f"{out}/test_spline_init.png")
    plot_basis(spline_model.basis, f"{out}/test_spline_init_basis.png")

    assert (
        lnl_final > lnl_initial
    ), "Optimized weights should yield a higher log-likelihood than initial zeros."
    assert (
        runtime < 5
    ), "Initialization should complete in less than 5 seconds."
