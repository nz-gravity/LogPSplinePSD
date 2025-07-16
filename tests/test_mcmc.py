import os

import arviz as az
import matplotlib.pyplot as plt
import pytest

from log_psplines.arviz_utils import compare_runs, get_weights
from log_psplines.mcmc import Periodogram, run_mcmc
from log_psplines.plotting import plot_pdgrm


def test_mcmc(mock_pdgrm: Periodogram, outdir):
    for sampler in ["nuts", "mh"]:
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

        # check inference data saved
        fname = os.path.join(outdir, f"out_{sampler}", "inference_data.nc")
        assert os.path.exists(
            fname
        ), f"Inference data file {fname} does not exist."
        # check we can load the inference data
        idata_loaded = az.from_netcdf(fname)
        # print inference data summary
        assert hasattr(
            idata_loaded, "posterior"
        ), "Loaded inference data does not have posterior samples."
        assert hasattr(
            idata_loaded, "spline_model"
        ), "Loaded inference data does not have posterior predictive samples."
        assert isinstance(
            idata_loaded.attrs["runtime"], float
        ), "Runtime attribute is not a float."

    compare_runs(
        az.from_netcdf(os.path.join(outdir, "out_nuts", "inference_data.nc")),
        az.from_netcdf(os.path.join(outdir, "out_mh", "inference_data.nc")),
        labels=["NUTS", "MH"],
        outdir=f"{outdir}/out_comparison",
    )
