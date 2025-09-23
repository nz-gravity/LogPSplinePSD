import os

import arviz as az
import matplotlib.pyplot as plt

from log_psplines.arviz_utils import compare_results, get_weights
from log_psplines.mcmc import Periodogram, run_mcmc
from log_psplines.plotting import plot_pdgrm


def test_mcmc(mock_pdgrm: Periodogram, outdir: str):
    outdir = os.path.join(outdir, "out_mcmc")
    os.makedirs(outdir, exist_ok=True)

    for sampler in ["nuts", "mh"]:
        compute_lnz = sampler == "mh"  # only compute Lnz for MH sampler (OTHER IS BROKEN)

        idata = run_mcmc(
            mock_pdgrm,
            sampler=sampler,
            n_knots=4,
            n_samples=200,
            n_warmup=200,
            outdir=f"{outdir}/out_{sampler}",
            rng_key=42,
            compute_lnz=compute_lnz,
        )

        fig, ax = plot_pdgrm(idata=idata, show_data=False)
        ax.set_xscale("linear")
        fig.savefig(
            os.path.join(outdir, f"test_mcmc_{sampler}.png"), transparent=False
        )
        plt.close(fig)

        # check inference data saved
        fname = os.path.join(outdir, f"out_{sampler}", "inference_data.nc")
        assert os.path.exists(
            fname
        ), f"Inference data file {fname} does not exist."
        # check we can load the inference data
        idata_loaded = az.from_netcdf(fname)
        assert idata_loaded is not None, "Inference data could not be loaded."

        # assert that lp is present for idata
        assert "lp" in idata_loaded.sample_stats, "Log-posterior 'lp' not found in sample_stats."
        # assert that weights are present and have correct shape
        weights = get_weights(idata_loaded)
        assert weights is not None, "Weights not found in posterior."

    compare_results(
        az.from_netcdf(os.path.join(outdir, "out_nuts", "inference_data.nc")),
        az.from_netcdf(os.path.join(outdir, "out_mh", "inference_data.nc")),
        labels=["NUTS", "MH"],
        outdir=f"{outdir}/out_comparison",
    )

    fig = plot_pdgrm(idata=idata, interactive=True)  # test interactive mode
    fig.write_html(os.path.join(outdir, "test_mcmc_interactive.html"))
