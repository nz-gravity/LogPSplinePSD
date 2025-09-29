import arviz as az
import matplotlib.pyplot as plt

from log_psplines.arviz_utils import compare_results, get_weights
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc
from log_psplines.plotting import plot_pdgrm, plot_psd_matrix
from log_psplines.example_datasets.ar_data import ARData

import numpy as np

from log_psplines.example_datasets.varma_data import VARMAData
import os


def test_multivar_mcmc(outdir, test_mode):
    """Test basic multivariate PSD analysis with VARMA data."""
    outdir = f"{outdir}/out_mcmc/multivar"
    os.makedirs(outdir, exist_ok=True)

    n = 1024
    n_knots = 10
    n_samples = n_warmup = 600
    if test_mode == "fast":
        n_samples = n_warmup = 10
        n = 256
        n_knots = 5

    # Generate test data
    np.random.seed(42)
    varma = VARMAData(n_samples=n)
    n_dim = varma.dim
    varma.plot(fname=os.path.join(outdir, "varma_data.png"))


    print(f"VARMA data shape: {varma.data.shape}, dim={n_dim}")

    timeseries = MultivariateTimeseries(
        t=varma.time,
        y=varma.data,
    )
    print(f"Timeseries: {timeseries}")


    # Run unified MCMC (multivariate NUTS)
    idata = run_mcmc(
        data=timeseries,
        sampler="nuts",
        n_knots=n_knots,  # Small for fast testing
        degree=3,
        diffMatrixOrder=2,
        n_samples=n_samples,
        n_warmup=n_warmup,
        outdir=outdir,
        verbose=True,
        target_accept_prob=0.8,
        true_psd=varma.get_true_psd()
    )

    # Basic checks
    assert idata is not None
    assert "posterior" in idata.groups()
    assert idata.posterior.sizes["draw"] == n_samples
    print(f"Inference data posterior variables: {idata.posterior}", )

    # check sampler type in attributes
    assert hasattr(idata, "attrs") and "sampler_type" in idata.attrs, "Sampler type not found in InferenceData attributes."
    assert idata.attrs["sampler_type"] == 'multivariate_nuts', f"Unexpected sampler type: {idata.attrs['sampler_type']}"

    # Check key parameters exist
    assert "log_likelihood" in idata.sample_stats.data_vars
    assert "lp" in idata.sample_stats.data_vars
    print(f"log_likelihood shape: {idata.sample_stats['log_likelihood'].shape}")
    print(f"lp shape: {idata.sample_stats['lp'].shape}")

    # Check diagonal parameters
    for j in range(n_dim):
        assert f"delta_{j}" in idata.posterior.data_vars
        assert f"phi_delta_{j}" in idata.posterior.data_vars
        assert f"weights_delta_{j}" in idata.posterior.data_vars

    # Print some results
    ll_samples = idata.sample_stats["log_likelihood"].values.flatten()
    print(f"Log likelihood range: {ll_samples.min():.2f} to {ll_samples.max():.2f}")

    # check the posterior psd matrix shape
    psd_matrix = idata.posterior_psd["psd_matrix"].values
    psd_matrix_shape = psd_matrix.shape
    expected_shape = (n_samples, varma.freq.shape[0], n_dim, n_dim)
    assert psd_matrix_shape[1:] == expected_shape[1:], f"Posterior PSD matrix shape mismatch (excluding 0th dim)! Expected {expected_shape[1:]}, got {psd_matrix_shape[1:]}"

    # Check RIAE computation for multivariate
    print(f"InferenceData attributes: {list(idata.attrs.keys())}")
    if 'riae_matrix' in idata.attrs:
        print(f"RIAE Matrix: {idata.attrs['riae_matrix']:.3f}")

    # check that results saved, and plots created
    result_fn = os.path.join(outdir, "inference_data.nc")
    plot_fn = os.path.join(outdir, "psd_matrix_posterior.png")
    assert os.path.exists(result_fn), "InferenceData file not found!"
    assert os.path.exists(plot_fn), "PSD matrix plot file not found!"


    plot_psd_matrix(
        idata=idata,
        n_channels=n_dim,
        freq=varma.freq,
        empirical_psd=varma.get_periodogram(),
        outdir=outdir,
        filename="psd_matrix_posterior_check.png",
        xscale='linear',
        diag_yscale='log',
    )



def test_mcmc(outdir: str, test_mode: str):
    outdir = os.path.join(outdir, "out_mcmc/univar")
    os.makedirs(outdir, exist_ok=True)

    psd_scale = 1 # e-42

    n = 1024
    n_samples = n_warmup = 500
    n_knots = 10
    if test_mode == "fast":
        n_samples = n_warmup = 10
        n = 256
        n_knots = 4
    ar_data = ARData(order=4, duration=1.0, fs=n, seed=42, sigma=np.sqrt(psd_scale))
    print(f"{ar_data.ts}")

    for sampler in ["nuts", "mh"]:
        compute_lnz = sampler == "mh"  # only compute Lnz for MH sampler (OTHER IS BROKEN)
        sampler_out = f"{outdir}/out_{sampler}"
        idata = run_mcmc(
            ar_data.ts,
            sampler=sampler,
            n_knots=n_knots,
            n_samples=n_samples,
            n_warmup=n_warmup,
            outdir=sampler_out,
            rng_key=42,
            compute_lnz=compute_lnz,
            true_psd=ar_data.psd_theoretical,
        )

        print(f"Inference data posterior variables: {idata.posterior}", )

        fig, ax = plot_pdgrm(idata=idata, show_data=False)
        ax.set_xscale("linear")
        fig.savefig(
            os.path.join(sampler_out, f"test_mcmc_{sampler}.png"), transparent=False
        )
        plt.close(fig)

        # check inference data saved
        fname = os.path.join(sampler_out, "inference_data.nc")
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

        post_psd = idata_loaded.posterior_psd.psd.median(dim=['pp_draw'])
        posd_psd_scale = post_psd.median().item()
        print(f"Posterior PSD scale (median): {posd_psd_scale:.2e}, expected ~{psd_scale:.2e}")
        # should be within 1 order of magnitude
        assert np.isclose(posd_psd_scale, psd_scale, rtol=1.0), "Posterior PSD scale is not within expected range."

    compare_results(
        az.from_netcdf(os.path.join(outdir, "out_nuts", "inference_data.nc")),
        az.from_netcdf(os.path.join(outdir, "out_mh", "inference_data.nc")),
        labels=["NUTS", "MH"],
        outdir=f"{outdir}/out_comparison",
    )

    fig = plot_pdgrm(idata=idata, interactive=True)  # test interactive mode
    fig.write_html(os.path.join(outdir, "test_mcmc_interactive.html"))
