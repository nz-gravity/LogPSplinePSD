import os

import arviz as az
import matplotlib.pyplot as plt

from log_psplines.arviz_utils import compare_results, get_weights
from log_psplines.mcmc import Periodogram, run_mcmc
from log_psplines.plotting import plot_pdgrm


def test_mcmc(mock_pdgrm: Periodogram, outdir: str):
    for sampler in ["mh", "nuts"]:
        idata = run_mcmc(
            mock_pdgrm,
            sampler=sampler,
            n_knots=20,
            n_samples=2000,
            n_warmup=2000,
            outdir=f"{outdir}/out_{sampler}",
            rng_key=42,
        )

        fig, ax = plot_pdgrm(idata=idata)
        fig.savefig(os.path.join(outdir, f"test_mcmc_{sampler}.png"))
        plt.close(fig)

        # check inference data saved
        fname = os.path.join(outdir, f"out_{sampler}", "inference_data.nc")
        assert os.path.exists(
            fname
        ), f"Inference data file {fname} does not exist."
        # check we can load the inference data
        idata_loaded = az.from_netcdf(fname)
        assert idata_loaded is not None, "Inference data could not be loaded."

    compare_results(
        az.from_netcdf(os.path.join(outdir, "out_nuts", "inference_data.nc")),
        az.from_netcdf(os.path.join(outdir, "out_mh", "inference_data.nc")),
        labels=["NUTS", "MH"],
        outdir=f"{outdir}/out_comparison",
    )
