import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from log_psplines.adaptive_mcmc import AdaptiveMCMCSampler
from log_psplines.datatypes import Periodogram
from log_psplines.plotting import plot_pdgrm
from log_psplines.plotting.utils import get_weights, plot_diagnostics
from log_psplines.psplines import LogPSplines


def test_adaptive_mcmc(mock_pdgrm: Periodogram, outdir):
    log_pdgrm = jnp.log(mock_pdgrm.power)
    spline_model = LogPSplines.from_periodogram(
        mock_pdgrm,
        n_knots=30,
        degree=3,
        diffMatrixOrder=2,
        # parametric_model=parametric_model,
    )

    # Create and run sampler
    sampler = AdaptiveMCMCSampler(
        log_pdgrm=log_pdgrm,
        spline_model=spline_model,
    )

    inference_data = sampler.sample(
        n_samples=1000, n_warmup=1000, thin=1, verbose=True
    )

    weights = get_weights(inference_data, thin=10)

    out = f"{outdir}/out_adaptive_mcmc"
    os.makedirs(out, exist_ok=True)

    fig, ax = plot_pdgrm(mock_pdgrm, spline_model, weights)
    fig.savefig(os.path.join(out, f"test_adaptive.png"))
    plt.close(fig)

    plot_diagnostics(inference_data, out)
