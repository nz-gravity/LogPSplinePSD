import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from log_psplines.arviz_utils import compare_results, get_weights
from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.example_datasets.ar_data import ARData
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc
from log_psplines.plotting import plot_pdgrm, plot_psd_matrix


def test_multivar_mcmc(outdir, test_mode):
    """Test basic multivariate PSD analysis with VARMA data."""
    outdir = f"{outdir}/out_mcmc/multivar"
    os.makedirs(outdir, exist_ok=True)
    print(f"++++ Running multivariate MCMC test {test_mode} ++++")

    n = 1024
    n_knots = 10
    n_samples = n_warmup = 1200
    verbose = True
    if test_mode == "fast":
        n_samples = n_warmup = 4
        n = 128
        n_knots = 4
        verbose = False

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
    empirical_full = timeseries.get_empirical_psd()
    print(f"Timeseries: {timeseries}")

    true_psd = varma.get_true_psd()
    default_blocks = 2 if test_mode == "fast" else 4
    samplers = [
        ("nuts", "multivariate_blocked_nuts", False, default_blocks),
        ("multivar_nuts", "multivariate_nuts", True, 1),
    ]

    for sampler_name, expected_sampler_attr, expect_lp, n_blocks in samplers:
        save_name = (
            "multivar_blocked_nuts" if sampler_name == "nuts" else sampler_name
        )
        sampler_outdir = os.path.join(outdir, save_name)
        # Run unified MCMC (multivariate sampler)
        idata = run_mcmc(
            data=timeseries,
            sampler=sampler_name,
            n_knots=n_knots,  # Small for fast testing
            degree=3,
            diffMatrixOrder=2,
            n_samples=n_samples,
            n_warmup=n_warmup,
            outdir=sampler_outdir,
            verbose=verbose,
            target_accept_prob=0.8,
            true_psd=true_psd,
            n_time_blocks=n_blocks,
        )

        # Basic checks
        assert idata is not None
        assert "posterior" in idata.groups()
        assert idata.posterior.sizes["draw"] == n_samples
        print(
            f"[{sampler_name}] posterior variables: {idata.posterior}",
        )

        # check sampler type in attributes
        assert hasattr(idata, "attrs") and "sampler_type" in idata.attrs
        # assert (
        #     idata.attrs["sampler_type"] == expected_sampler_attr
        # ), f"Unexpected sampler type for {sampler_name}: {idata.attrs['sampler_type']}"

        # Check key parameters exist
        assert "log_likelihood" in idata.sample_stats.data_vars
        # if expect_lp:
        #     assert "lp" in idata.sample_stats.data_vars
        # else:
        #     assert "lp" not in idata.sample_stats.data_vars
        print(
            f"[{sampler_name}] log_likelihood shape: {idata.sample_stats['log_likelihood'].shape}"
        )
        if "lp" in idata.sample_stats.data_vars:
            print(
                f"[{sampler_name}] lp shape: {idata.sample_stats['lp'].shape}"
            )

        # Check diagonal parameters
        for j in range(n_dim):
            assert f"delta_{j}" in idata.posterior.data_vars
            assert f"phi_delta_{j}" in idata.posterior.data_vars
            assert f"weights_delta_{j}" in idata.posterior.data_vars

        # Print some results
        ll_samples = idata.sample_stats["log_likelihood"].values.flatten()
        print(
            f"[{sampler_name}] Log likelihood range: {ll_samples.min():.2f} to {ll_samples.max():.2f}"
        )

        # check the posterior psd matrix shape
        psd_matrix = idata.posterior_psd["psd_matrix"].values
        psd_matrix_shape = psd_matrix.shape
        freq_dim = idata.posterior_psd["psd_matrix"].sizes["freq"]
        assert (
            psd_matrix_shape[1] == freq_dim
        ), "Posterior PSD frequency dimension mismatch."
        assert psd_matrix_shape[2:] == (
            n_dim,
            n_dim,
        ), f"Posterior PSD matrix channel dims mismatch: expected {(n_dim, n_dim)}, got {psd_matrix_shape[2:]}"

        # Check RIAE and CI coverage computation for multivariate
        print(
            f"[{sampler_name}] InferenceData attributes: {list(idata.attrs.keys())}"
        )
        if "riae_matrix" in idata.attrs:
            print(
                f"[{sampler_name}] RIAE Matrix: {idata.attrs['riae_matrix']:.3f}"
            )
        if "ci_coverage" in idata.attrs:
            print(
                f"[{sampler_name}] CI Coverage: {idata.attrs['ci_coverage']:.3f}"
            )

        # check that results saved, and plots created
        result_fn = os.path.join(sampler_outdir, "inference_data.nc")
        plot_fn = os.path.join(sampler_outdir, "psd_matrix.png")
        assert os.path.exists(result_fn), "InferenceData file not found!"
        assert os.path.exists(plot_fn), "PSD matrix plot file not found!"

        plot_psd_matrix(
            idata=idata,
            outdir=sampler_outdir,
            filename=f"psd_matrix_posterior_check_{sampler_name}.png",
            xscale="linear",
            diag_yscale="log",
        )

    res_multiar_nuts = az.from_netcdf(
        os.path.join(outdir, "multivar_nuts", "inference_data.nc")
    )
    res_multiar_blocked_nuts = az.from_netcdf(
        os.path.join(outdir, "multivar_blocked_nuts", "inference_data.nc")
    )
    fig, ax = plot_psd_matrix(
        idata=res_multiar_nuts,
        true_psd=true_psd,
        xscale="linear",
        diag_yscale="log",
        label="Multivar NUTS (1 block)",
        save=False,
        close=False,
    )
    fig, ax = plot_psd_matrix(
        idata=res_multiar_blocked_nuts,
        true_psd=true_psd,
        xscale="linear",
        diag_yscale="log",
        label=f"Multivar Factorised NUTS (Nb={default_blocks})",
        fig=fig,
        ax=ax,
        save=False,
        close=False,
        empirical_psd=empirical_full,
    )
    fig.savefig(os.path.join(outdir, "psd_matrix_comparison.png"))
    plt.close(fig)

    print(f"++++ multivariate MCMC test {test_mode} COMPLETE ++++")


def test_mcmc(outdir: str, test_mode: str):
    outdir = os.path.join(outdir, "out_mcmc/univar")
    os.makedirs(outdir, exist_ok=True)

    print(f"++++ Running univariate MCMC test {test_mode} ++++")

    psd_scale = 1  # e-42

    n = 1024
    n_samples = n_warmup = 500
    n_knots = 10
    compute_lnz = True
    sampler_names = ["nuts", "mh"]
    if test_mode == "fast":
        n_samples = n_warmup = 4
        n = 128
        n_knots = 3
        compute_lnz = False
        sampler_names = ["nuts", "mh"]
    ar_data = ARData(
        order=4, duration=1.0, fs=n, seed=42, sigma=np.sqrt(psd_scale)
    )
    print(f"{ar_data.ts}")

    # coarse_grain = CoarseGrainConfig(
    #     enabled=True,
    #     f_transition=10**2,
    #     f_max=ar_data.ts.fs / 2,
    #     n_log_bins=100,
    # )

    for sampler in sampler_names:
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
            verbose=(test_mode != "fast"),
            # coarse_grain=coarse_grain,
        )

        print(
            f"Inference data posterior variables: {idata.posterior}",
        )

        fig, ax = plot_pdgrm(idata=idata, show_data=False)
        ax.set_xscale("linear")
        fig.savefig(
            os.path.join(sampler_out, f"test_mcmc_{sampler}.png"),
            transparent=False,
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
        assert (
            "lp" in idata_loaded.sample_stats
        ), "Log-posterior 'lp' not found in sample_stats."
        # assert that weights are present and have correct shape
        weights = get_weights(idata_loaded)
        assert weights is not None, "Weights not found in posterior."

        post_psd = idata_loaded.posterior_psd.psd.median(dim=["pp_draw"])
        posd_psd_scale = post_psd.median().item()
        print(
            f"Posterior PSD scale (median): {posd_psd_scale:.2e}, expected ~{psd_scale:.2e}"
        )
        # should be within 1 order of magnitude
        assert np.isclose(
            posd_psd_scale, psd_scale, rtol=1.0
        ), "Posterior PSD scale is not within expected range."

    compare_results(
        az.from_netcdf(os.path.join(outdir, "out_nuts", "inference_data.nc")),
        az.from_netcdf(os.path.join(outdir, "out_mh", "inference_data.nc")),
        labels=["NUTS", "MH"],
        outdir=f"{outdir}/out_comparison",
    )

    fig = plot_pdgrm(idata=idata, interactive=True)  # test interactive mode
    fig.write_html(os.path.join(outdir, "test_mcmc_interactive.html"))

    print(f"++++ univariate MCMC test {test_mode} COMPLETE ++++")
