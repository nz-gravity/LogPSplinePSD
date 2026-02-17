import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.arviz_utils import get_weights
from log_psplines.arviz_utils.compare_results import compare_results
from log_psplines.arviz_utils.to_arviz import _prepare_samples_and_stats
from log_psplines.datatypes.univar import Timeseries
from log_psplines.example_datasets.ar_data import ARData
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    MultivariateTimeseries,
    NUTSConfigOverride,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)
from log_psplines.plotting import (
    PSDMatrixPlotSpec,
    plot_pdgrm,
    plot_psd_matrix,
)
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)


@pytest.mark.slow
def test_multivar_mcmc(outdir, test_mode):
    """Test basic multivariate PSD analysis with VARMA data."""
    outdir = f"{outdir}/out_mcmc/multivar"
    os.makedirs(outdir, exist_ok=True)
    print(f"++++ Running multivariate MCMC test {test_mode} ++++")

    n = 256
    n_knots = 8
    n_samples = n_warmup = 200
    verbose = True
    if test_mode == "fast":
        n_samples = n_warmup = 3
        n = 64
        n_knots = 3
        verbose = False

    # Generate test data
    np.random.seed(42)
    varma = VARMAData(n_samples=n)
    p = varma.p
    varma.plot(fname=os.path.join(outdir, "varma_data.png"))

    print(f"VARMA data shape: {varma.data.shape}, p={p}")

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
    ]

    for sampler_name, expected_sampler_attr, expect_lp, Nb in samplers:
        save_name = (
            "multivar_blocked_nuts" if sampler_name == "nuts" else sampler_name
        )
        sampler_outdir = os.path.join(outdir, save_name)
        # Run unified MCMC (multivariate sampler)
        model_cfg = ModelConfig(
            n_knots=n_knots,
            degree=3,
            diffMatrixOrder=2,
            true_psd=true_psd,
        )
        diagnostics_cfg = DiagnosticsConfig(
            outdir=sampler_outdir,
            verbose=verbose,
        )
        nuts_cfg = NUTSConfigOverride(target_accept_prob=0.8)
        run_cfg = RunMCMCConfig(
            n_samples=n_samples,
            n_warmup=n_warmup,
            Nb=Nb,
            model=model_cfg,
            diagnostics=diagnostics_cfg,
            nuts=nuts_cfg,
        )
        idata = run_mcmc(
            data=timeseries,
            config=run_cfg,
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
        for j in range(p):
            assert f"delta_{j}" in idata.posterior.data_vars
            assert f"phi_delta_{j}" in idata.posterior.data_vars
            assert f"weights_delta_{j}" in idata.posterior.data_vars

        # Print some results
        ll_samples = idata.sample_stats["log_likelihood"].values.flatten()
        print(
            f"[{sampler_name}] Log likelihood range: {ll_samples.min():.2f} to {ll_samples.max():.2f}"
        )

        # check the posterior psd matrix shape
        psd_matrix_real = idata.posterior_psd["psd_matrix_real"]
        psd_matrix_shape = psd_matrix_real.shape
        freq_dim = psd_matrix_real.sizes["freq"]
        assert (
            psd_matrix_shape[1] == freq_dim
        ), "Posterior PSD frequency dimension mismatch."
        assert psd_matrix_shape[2:] == (
            p,
            p,
        ), f"Posterior PSD matrix channel dims mismatch: expected {(p, p)}, got {psd_matrix_shape[2:]}"

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

        spec = PSDMatrixPlotSpec(
            idata=idata,
            outdir=sampler_outdir,
            filename=f"psd_matrix_posterior_check_{sampler_name}.png",
            xscale="linear",
            diag_yscale="log",
        )
        plot_psd_matrix(spec)

    res_multiar_blocked_nuts = az.from_netcdf(
        os.path.join(outdir, "multivar_blocked_nuts", "inference_data.nc")
    )
    spec = PSDMatrixPlotSpec(
        idata=res_multiar_blocked_nuts,
        true_psd=true_psd,
        xscale="linear",
        diag_yscale="log",
        label=f"Multivar Blocked NUTS (Nb={default_blocks})",
        save=False,
        close=False,
        empirical_psd=empirical_full,
    )
    fig, ax = plot_psd_matrix(spec)
    fig.savefig(os.path.join(outdir, "psd_matrix_posterior.png"))
    plt.close(fig)

    print(f"++++ multivariate MCMC test {test_mode} COMPLETE ++++")


def test_multivar_mcmc_unit(synthetic_multivar_timeseries):
    timeseries = synthetic_multivar_timeseries
    model_cfg = ModelConfig(n_knots=3)
    diagnostics_cfg = DiagnosticsConfig(
        outdir=None,
        verbose=False,
        compute_lnz=False,
    )
    vi_cfg = VIConfig(
        vi_steps=10,
        vi_posterior_draws=6,
        vi_psd_max_draws=2,
        only_vi=True,
    )
    run_cfg = RunMCMCConfig(
        n_samples=1,
        n_warmup=1,
        Nb=1,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
    )
    idata = run_mcmc(
        data=timeseries,
        config=run_cfg,
    )

    assert idata is not None
    assert "posterior" in idata.groups()
    assert idata.attrs.get("full_diagnostics_computed") == 1
    assert "full_diagnostics_timestamp" in idata.attrs
    psd = idata.posterior_psd["psd_matrix_real"].sel(
        percentile=50, method="nearest"
    )
    psd_vals = np.asarray(psd.values)
    assert psd_vals.ndim == 3
    assert np.allclose(
        psd_vals, np.swapaxes(psd_vals, 1, 2), rtol=1e-6, atol=1e-8
    )
    diag = np.diagonal(psd_vals, axis1=1, axis2=2)
    assert np.all(diag > 0.0)


def test_mcmc_unit(synthetic_univar_timeseries):
    model_cfg = ModelConfig(n_knots=3)
    diagnostics_cfg = DiagnosticsConfig(
        outdir=None,
        verbose=False,
        compute_lnz=False,
    )
    vi_cfg = VIConfig(init_from_vi=False)
    run_cfg = RunMCMCConfig(
        n_samples=3,
        n_warmup=3,
        rng_key=0,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
    )
    idata = run_mcmc(
        synthetic_univar_timeseries,
        config=run_cfg,
    )

    assert "posterior" in idata.groups()
    assert "sample_stats" in idata.groups()
    assert idata.attrs.get("full_diagnostics_computed") == 1
    assert "full_diagnostics_timestamp" in idata.attrs
    assert "lp" in idata.sample_stats.data_vars
    weights = get_weights(idata)
    assert weights.size > 0
    psd = idata.posterior_psd["psd"].sel(percentile=50, method="nearest")
    assert np.all(psd.values > 0.0)


@pytest.mark.slow
def test_mcmc(outdir: str, test_mode: str):
    outdir = os.path.join(outdir, "out_mcmc/univar")
    os.makedirs(outdir, exist_ok=True)

    print(f"++++ Running univariate MCMC test {test_mode} ++++")

    psd_scale = 1  # e-42

    n = 256
    n_samples = n_warmup = 120
    n_knots = 8
    compute_lnz = True
    sampler_names = ["nuts"]
    if test_mode == "fast":
        n_samples = n_warmup = 3
        n = 64
        n_knots = 3
        compute_lnz = False
        sampler_names = ["nuts"]
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
        model_cfg = ModelConfig(
            n_knots=n_knots,
            true_psd=ar_data.psd_theoretical,
        )
        diagnostics_cfg = DiagnosticsConfig(
            outdir=sampler_out,
            verbose=(test_mode != "fast"),
            compute_lnz=compute_lnz,
        )
        vi_cfg = VIConfig(init_from_vi=(test_mode != "fast"))
        run_cfg = RunMCMCConfig(
            n_samples=n_samples,
            n_warmup=n_warmup,
            rng_key=42,
            model=model_cfg,
            diagnostics=diagnostics_cfg,
            vi=vi_cfg,
        )
        idata = run_mcmc(
            ar_data.ts,
            config=run_cfg,
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

        post_psd = idata_loaded.posterior_psd.psd.sel(
            percentile=50.0, method="nearest"
        )
        posd_psd_scale = post_psd.median().item()
        print(
            f"Posterior PSD scale (median): {posd_psd_scale:.2e}, expected ~{psd_scale:.2e}"
        )
        # should be within 1 order of magnitude
        assert np.isclose(
            posd_psd_scale, psd_scale, rtol=1.0
        ), "Posterior PSD scale is not within expected range."

    fig = plot_pdgrm(idata=idata, interactive=True)  # test interactive mode
    fig.write_html(os.path.join(outdir, "test_mcmc_interactive.html"))

    print(f"++++ univariate MCMC test {test_mode} COMPLETE ++++")


def _synthetic_univariate_series():
    rng = np.random.default_rng(12345)
    n = 128
    t = np.linspace(0, 4, n, endpoint=False)
    signal = 0.05 * np.sin(2 * np.pi * 3.0 * t)
    signal += 0.03 * np.cos(2 * np.pi * 1.3 * t)
    y = signal + 0.02 * rng.normal(size=n)
    return t, y


def _expected_coarse_freq_univar(
    ts: Timeseries,
    fmin: float,
    fmax: float,
    cfg: CoarseGrainConfig,
) -> np.ndarray:
    standardized = ts.standardise_for_psd()
    pdgrm = standardized.to_periodogram(fmin=fmin, fmax=fmax)
    spec = compute_binning_structure(
        pdgrm.freqs,
        Nc=cfg.Nc,
        Nh=cfg.Nh,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    return np.asarray(spec.f_coarse, dtype=np.float64)


def test_run_mcmc_coarse_grain_univariate_mcmc():
    t, y = _synthetic_univariate_series()
    ts_run = Timeseries(t=t.copy(), y=y.copy())
    ts_spec = Timeseries(t=t.copy(), y=y.copy())
    fmin, fmax = 0.02, 0.8
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=None,
        Nh=1,
        f_min=fmin,
        f_max=fmax,
    )
    expected_freq = _expected_coarse_freq_univar(
        ts_spec,
        fmin=fmin,
        fmax=fmax,
        cfg=coarse_cfg,
    )

    model_cfg = ModelConfig(
        n_knots=6,
        degree=3,
        diffMatrixOrder=2,
        fmin=fmin,
        fmax=fmax,
    )
    diagnostics_cfg = DiagnosticsConfig(verbose=False)
    vi_cfg = VIConfig(
        only_vi=True,
        vi_steps=20,
        vi_posterior_draws=8,
        vi_psd_max_draws=4,
    )
    run_cfg = RunMCMCConfig(
        n_samples=1,
        n_warmup=1,
        coarse_grain_config=coarse_cfg,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
    )
    idata = run_mcmc(
        data=ts_run,
        config=run_cfg,
    )
    freq = np.asarray(idata.posterior_psd["freq"].values)
    assert freq.shape[0] == expected_freq.shape[0]
    assert np.allclose(freq, expected_freq)


def _synthetic_multivar_series():
    rng = np.random.default_rng(67890)
    n = 32
    t = np.linspace(0, 4, n, endpoint=False)
    base = np.stack(
        (
            np.sin(2 * np.pi * 0.25 * t),
            np.cos(2 * np.pi * 0.3 * t),
        ),
        axis=1,
    )
    noise = 0.05 * rng.normal(size=base.shape)
    y = base + noise
    y[:, 1] += 0.2 * y[:, 0]
    return t, y


def _expected_coarse_freq_multivar(
    ts: MultivariateTimeseries,
    Nb: int,
    fmin: float,
    fmax: float,
    cfg: CoarseGrainConfig,
) -> np.ndarray:
    standardized = ts.standardise_for_psd()
    fft = standardized.to_wishart_stats(
        Nb=Nb,
        fmin=fmin,
        fmax=fmax,
    )
    spec = compute_binning_structure(
        fft.freq,
        Nc=cfg.Nc,
        Nh=cfg.Nh,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    return np.asarray(spec.f_coarse, dtype=np.float64)


def test_run_mcmc_coarse_grain_multivar_only_vi():
    t, y = _synthetic_multivar_series()
    ts_run = MultivariateTimeseries(y=y.copy(), t=t.copy())
    ts_spec = MultivariateTimeseries(y=y.copy(), t=t.copy())
    fmin, fmax = 0.01, 0.4
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=None,
        Nh=1,
        f_min=fmin,
        f_max=fmax,
    )
    Nb = 2
    expected_freq = _expected_coarse_freq_multivar(
        ts_spec,
        Nb=Nb,
        fmin=fmin,
        fmax=fmax,
        cfg=coarse_cfg,
    )

    model_cfg = ModelConfig(
        n_knots=5,
        degree=3,
        diffMatrixOrder=2,
        fmin=fmin,
        fmax=fmax,
    )
    diagnostics_cfg = DiagnosticsConfig(verbose=False)
    vi_cfg = VIConfig(
        only_vi=True,
        vi_steps=20,
        vi_lr=5e-3,
        vi_progress_bar=False,
        vi_posterior_draws=6,
        vi_psd_max_draws=2,
    )
    run_cfg = RunMCMCConfig(
        n_samples=1,
        n_warmup=1,
        Nb=Nb,
        coarse_grain_config=coarse_cfg,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
    )
    idata = run_mcmc(
        data=ts_run,
        config=run_cfg,
    )
    freq = np.asarray(idata.posterior_psd["freq"].values)
    assert freq.shape[0] == expected_freq.shape[0]
    assert np.allclose(freq, expected_freq)


def test_multivar_blocked_nuts_records_step_size():
    t, y = _synthetic_multivar_series()
    ts_run = MultivariateTimeseries(y=y.copy(), t=t.copy())

    model_cfg = ModelConfig(
        n_knots=4,
        degree=3,
        diffMatrixOrder=2,
    )
    diagnostics_cfg = DiagnosticsConfig(verbose=False)
    vi_cfg = VIConfig(init_from_vi=False)
    run_cfg = RunMCMCConfig(
        n_samples=1,
        n_warmup=2,
        num_chains=1,
        Nb=1,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
    )
    idata = run_mcmc(
        data=ts_run,
        config=run_cfg,
    )

    step_size_keys = [
        key
        for key in idata.sample_stats.data_vars
        if str(key).startswith("step_size_channel_")
    ]
    assert step_size_keys, "Blocked NUTS should record per-channel step size."


def test_prepare_samples_and_stats_preserves_chain_dim():
    samples = {"weights_delta_0": np.zeros((2, 3, 4), dtype=float)}
    sample_stats = {"log_likelihood": np.zeros((2, 3), dtype=float)}
    out_samples, out_stats = _prepare_samples_and_stats(
        samples, sample_stats, num_chains=2
    )

    assert out_samples["weights_delta_0"].shape[0] == 2
    assert out_stats["log_likelihood"].shape[0] == 2


def test_prepare_samples_and_stats_adds_chain_dim():
    samples = {"weights_delta_0": np.zeros((3, 4), dtype=float)}
    sample_stats = {"log_likelihood": np.zeros((3,), dtype=float)}
    out_samples, out_stats = _prepare_samples_and_stats(
        samples, sample_stats, num_chains=1
    )

    assert out_samples["weights_delta_0"].shape[0] == 1
    assert out_stats["log_likelihood"].shape[0] == 1


def test_compare_results_minimal(tmp_path):
    rng = np.random.default_rng(0)
    draws = 60
    run1 = az.from_dict(posterior={"weights": rng.normal(size=(1, draws))})
    run2 = az.from_dict(
        posterior={"weights": rng.normal(loc=0.1, size=(1, draws))}
    )

    compare_results(
        run1,
        run2,
        labels=["Run1", "Run2"],
        outdir=str(tmp_path),
    )

    assert (tmp_path / "ess_comparison.png").exists()
    assert (tmp_path / "summary_diff.csv").exists()
