import csv
import os
import shutil
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.arviz_utils import (
    get_multivar_posterior_psd_quantiles,
    get_posterior_psd,
    get_weights,
)
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    MultivariateTimeseries,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)
from log_psplines.plotting import plot_pdgrm
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)


def test_mcmc_univar(outdir: str):
    from log_psplines.example_datasets.ar_data import ARData

    outdir = os.path.join(outdir, "out_mcmc/univar")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"++++ Running univariate MCMC test ++++")

    psd_scale = 1  # e-42

    n = 2048
    n_samples = n_warmup = 500
    n_knots = 20
    compute_lnz = True

    ar_data = ARData(
        order=4, duration=1.0, fs=n, seed=42, sigma=np.sqrt(psd_scale)
    )
    print(f"{ar_data.ts}")

    sampler_out = f"{outdir}/out_nuts"
    model_cfg = ModelConfig(
        n_knots=n_knots,
        true_psd=ar_data.psd_theoretical,
    )
    diagnostics_cfg = DiagnosticsConfig(
        outdir=sampler_out,
        verbose=True,
        compute_lnz=compute_lnz,
    )
    vi_cfg = VIConfig(init_from_vi=True)
    run_cfg = RunMCMCConfig(
        n_samples=n_samples,
        n_warmup=n_warmup,
        rng_key=42,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
        num_chains=2,
    )
    idata = run_mcmc(
        ar_data.ts,
        config=run_cfg,
    )

    print(f"Inference data posterior variables: {idata.posterior}")

    fig, ax = plot_pdgrm(
        idata=idata,
        show_data=False,
        true_psd=np.asarray(ar_data.psd_theoretical, dtype=float),
    )
    ax.set_xscale("linear")
    fig.savefig(
        os.path.join(sampler_out, "test_mcmc_nuts.png"),
        transparent=False,
    )
    plt.close(fig)

    # Check inference data saved
    fname = os.path.join(sampler_out, "inference_data.nc")
    assert os.path.exists(
        fname
    ), f"Inference data file {fname} does not exist."

    # Assert that lp is present for idata
    assert (
        "lp" in idata.sample_stats
    ), "Log-posterior 'lp' not found in sample_stats."

    # Assert that weights are present and have correct shape
    weights = get_weights(idata)
    assert weights is not None, "Weights not found in posterior."

    # Verify VI diagnostics are persisted.
    univar_diag_csv = os.path.join(
        sampler_out, "diagnostics", "vi_summary.csv"
    )
    assert os.path.exists(univar_diag_csv), "Missing univariate vi_summary.csv"

    # Verify NUTS truth metrics are saved and attached.
    univar_nuts_csv = os.path.join(
        sampler_out, "diagnostics", "nuts_summary.csv"
    )
    assert os.path.exists(
        univar_nuts_csv
    ), "Missing univariate nuts_summary.csv"
    with open(univar_nuts_csv, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    for key in ("riae", "l2", "coverage"):
        assert (
            key in row
        ), f"Column {key} missing from univariate nuts_summary.csv"
        assert (
            key in idata.sample_stats.attrs
        ), f"sample_stats missing NUTS metric attr: {key}"

    _, median_psd, _, _ = get_posterior_psd(idata)
    post_psd_scale = float(np.median(median_psd))
    print(
        f"Posterior PSD scale (median): {post_psd_scale:.2e}, expected ~{psd_scale:.2e}"
    )
    # Should be within 1 order of magnitude
    assert np.isclose(
        post_psd_scale, psd_scale, rtol=1.0
    ), "Posterior PSD scale is not within expected range."

    print(f"++++ univariate MCMC test COMPLETE ++++")


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
    )
    return np.asarray(spec.f_coarse, dtype=np.float64)


def test_mcmc_multivar(outdir):
    """Test multivariate PSD analysis with coarse grain, VI init, and blocking."""
    from log_psplines.example_datasets.varma_data import VARMAData

    outdir = os.path.join(outdir, "out_mcmc/multivar")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    varma_data = VARMAData(n_samples=2**12, fs=64.0, seed=0)
    assert varma_data.data is not None
    ts_data = cast(np.ndarray, varma_data.data)
    ts_run = MultivariateTimeseries(y=ts_data, t=varma_data.time)

    fmin, fmax = 0, 32
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=None,
        Nh=2,
    )

    n_samples = n_warmup = 1000
    vi_steps = 1000
    Nb = 4  # Number of blocks for Welch periodogram

    expected_freq = _expected_coarse_freq_multivar(
        ts_run,
        Nb=Nb,
        fmin=fmin,
        fmax=fmax,
        cfg=coarse_cfg,
    )

    model_cfg = ModelConfig(
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        fmin=fmin,
        fmax=fmax,
        true_psd=varma_data.get_true_psd(),
    )
    diagnostics_cfg = DiagnosticsConfig(verbose=True, outdir=outdir)
    vi_cfg = VIConfig(
        init_from_vi=True,
        only_vi=False,
        vi_steps=vi_steps,
        vi_lr=5e-3,
        vi_progress_bar=False,
        vi_posterior_draws=100,
        vi_psd_max_draws=100,
    )
    run_cfg = RunMCMCConfig(
        n_samples=n_samples,
        n_warmup=n_warmup,
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

    # Verify coarse-grained frequency structure
    quantiles = get_multivar_posterior_psd_quantiles(idata, n_keep=2)
    freq = np.asarray(quantiles["freq"], dtype=float)
    assert freq.shape[0] == expected_freq.shape[0], (
        f"Frequency dimension mismatch: got {freq.shape[0]}, "
        f"expected {expected_freq.shape[0]}"
    )
    assert np.allclose(
        freq, expected_freq
    ), "Coarse-grained frequencies do not match."

    # Verify PSD matrix structure for multivariate
    spectral_density = np.asarray(
        quantiles["spectral_density"], dtype=np.complex128
    )
    psd_shape = spectral_density.shape
    assert psd_shape[1] == freq.shape[0], "PSD frequency dimension mismatch."
    assert psd_shape[2:] == (2, 2), "PSD matrix should be 2x2 per frequency."

    # Verify Hermitian and positive definite
    idx50 = int(np.argmin(np.abs(np.asarray(quantiles["percentile"]) - 50.0)))
    psd_median = spectral_density[idx50]
    diag = np.real(np.diagonal(psd_median, axis1=1, axis2=2))
    assert np.all(diag > 0.0), "PSD diagonal elements should be positive."
    assert np.allclose(
        psd_median,
        np.swapaxes(psd_median.conj(), 1, 2),
        rtol=1e-6,
        atol=1e-8,
    ), "PSD should be Hermitian."

    vi_log_likelihood = idata["vi_log_likelihood"].dataset
    assert vi_log_likelihood is not None
    assert "log_likelihood_block_0" in vi_log_likelihood
    assert "log_likelihood_block_1" in vi_log_likelihood
    assert vi_log_likelihood["log_likelihood_block_0"].ndim == 3

    diagnostics_dir = os.path.join(outdir, "diagnostics")
    summary_path = os.path.join(diagnostics_dir, "vi_summary.csv")
    assert os.path.exists(summary_path)

    nuts_summary_path = os.path.join(diagnostics_dir, "nuts_summary.csv")
    assert os.path.exists(nuts_summary_path)

    # Plot should include true PSD overlay when true_psd is provided.
    assert os.path.exists(os.path.join(outdir, "psd_matrix.png"))

    vi_stats = idata["vi_sample_stats"].dataset
    for key in ("riae_matrix", "l2_matrix", "coverage"):
        assert key in vi_stats.attrs, f"Missing VI metric attr: {key}"
        assert np.isfinite(
            float(vi_stats.attrs[key])
        ), f"VI metric {key} should be finite."

    with open(summary_path, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    for key in ("riae", "l2", "coverage"):
        assert (
            key in row
        ), f"Column {key} missing from multivariate vi_summary.csv"
        assert np.isfinite(
            float(row[key])
        ), f"Column {key} in multivariate vi_summary.csv should be finite."

    with open(nuts_summary_path, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    for key in ("riae", "l2", "coverage"):
        assert (
            key in row
        ), f"Column {key} missing from multivariate nuts_summary.csv"
        assert np.isfinite(
            float(row[key])
        ), f"Column {key} in multivariate nuts_summary.csv should be finite."
        assert (
            key in idata.sample_stats.attrs
        ), f"sample_stats missing NUTS metric attr: {key}"
        assert np.isfinite(
            float(idata.sample_stats.attrs[key])
        ), f"sample_stats NUTS metric attr {key} should be finite."

    print(f"++++ multivariate MCMC test COMPLETE ++++")
