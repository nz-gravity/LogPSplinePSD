import os

import arviz as az
import matplotlib.pyplot as plt

from log_psplines.arviz_utils import compare_results, get_weights
from log_psplines.mcmc import Periodogram, run_mcmc
from log_psplines.plotting import plot_pdgrm

import numpy as np

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.datatypes import MultivarFFT
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines
from log_psplines.samplers.multivar.multivar_nuts import MultivarNUTSSampler, MultivarNUTSConfig
import os
from log_psplines.plotting.psd_matrix import plot_psd_matrix, compute_empirical_psd


def test_multivar_mcmc(outdir):
    """Test basic multivariate PSD analysis with VARMA data."""
    outdir = f"{outdir}/out_mcmc/multivar"
    os.makedirs(outdir, exist_ok=True)

    # Generate test data
    np.random.seed(42)
    varma = VARMAData(n_samples=1024)
    x = varma.data
    n_dim = varma.dim

    print(f"VARMA data shape: {x.shape}, dim={n_dim}")

    # Convert to FFT
    fft_data = MultivarFFT.compute_fft(x, fs=1.0)
    print(f"FFT shapes: y_re={fft_data.y_re.shape}, Z_re={fft_data.Z_re.shape}")

    # Run unified MCMC (multivariate NUTS)
    idata = run_mcmc(
        data=fft_data,
        sampler="nuts",
        n_knots=5,  # Small for fast testing
        degree=3,
        diffMatrixOrder=2,
        n_samples=50,
        n_warmup=50,
        outdir=outdir,
        verbose=True,
        target_accept_prob=0.8
    )

    # Basic checks
    assert idata is not None
    assert "posterior" in idata.groups()
    assert idata.posterior.sizes["draw"] == 50

    # Check key parameters exist
    assert "log_likelihood" in idata.sample_stats.data_vars
    print(f"log_likelihood shape: {idata.sample_stats['log_likelihood'].shape}")

    # Check diagonal parameters
    for j in range(n_dim):
        assert f"delta_{j}" in idata.posterior.data_vars
        assert f"phi_delta_{j}" in idata.posterior.data_vars
        assert f"weights_delta_{j}" in idata.posterior.data_vars

    # Print some results
    ll_samples = idata.sample_stats["log_likelihood"].values.flatten()
    print(f"Log likelihood range: {ll_samples.min():.2f} to {ll_samples.max():.2f}")

    # Test PSD reconstruction if we have the required samples
    if all(key in idata.sample_stats.data_vars for key in ["log_delta_sq", "theta_re", "theta_im"]):
        mv_model = MultivariateLogPSplines.from_multivar_fft(
            fft_data,
            n_knots=5,
            degree=3,
            diffMatrixOrder=2
        )
        psd_reconstructed = mv_model.reconstruct_psd_matrix(
            idata.sample_stats["log_delta_sq"].values,
            idata.sample_stats["theta_re"].values,
            idata.sample_stats["theta_im"].values,
            n_samples_max=4
        )
        print(f"Reconstructed PSD shape: {psd_reconstructed.shape}")
        assert psd_reconstructed.shape[1] == fft_data.n_freq
        assert psd_reconstructed.shape[2] == n_dim
        assert psd_reconstructed.shape[3] == n_dim

    # check that results saved, and plots created
    result_fn = os.path.join(outdir, "inference_data.nc")
    assert os.path.exists(result_fn), "InferenceData file not found!"

    plot_fn = os.path.join(outdir, "psd_matrix_posterior.png")
    assert os.path.exists(plot_fn), "PSD matrix plot file not found!"

    plot_psd_matrix(
        idata=idata,
        n_channels=n_dim,
        freq=fft_data.freq,
        empirical_psd=compute_empirical_psd(fft_data.y_re, fft_data.y_im, n_channels=n_dim),
        outdir=outdir,
        filename="psd_matrix_posterior.png",
        dpi=100,
        xscale='linear',
        diag_yscale='log',
    )

    print("✓ Multivariate analysis test passed!")


def test_mcmc(mock_pdgrm: Periodogram, outdir: str):
    outdir = os.path.join(outdir, "out_mcmc/univar")
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
