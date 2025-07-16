import os

import matplotlib.pyplot as plt
import pytest

from log_psplines.mcmc import Periodogram, run_mcmc
from log_psplines.plotting import plot_pdgrm
from log_psplines.plotting.utils import get_weights


@pytest.mark.parametrize("sampler", ["nuts", "mh"])
def test_mcmc(mock_pdgrm: Periodogram, outdir, sampler):
    idata, spline_model = run_mcmc(
        mock_pdgrm,
        sampler=sampler,
        n_knots=30,
        n_samples=1000,
        n_warmup=1000,
        outdir=f"{outdir}/out_{sampler}",
    )
    weights = get_weights(idata)

    fig, ax = plot_pdgrm(mock_pdgrm, spline_model, weights)
    fig.savefig(os.path.join(outdir, f"test_mcmc_{sampler}.png"))
    plt.close(fig)
