import csv
import os
import shutil
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from log_psplines.arviz_utils import (
    get_multivar_posterior_psd_quantiles,
    get_posterior_psd,
    get_weights,
    open_inference_data,
)
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    MultivariateTimeseries,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)


def test_mcmc_univar(outdir: str):
    print(f"++++ Running univariate MCMC test ++++")
    idata_orig, ar_data, psd_scale = _run_univar_mcmc(outdir)

    ### NOW WE CHECK THE OUTPUTS ###
    files_to_check = [
        "inference_data.nc",
        "posterior_predictive.png",
        "diagnostics/vi_summary.csv",
        "diagnostics/nuts_summary.csv",
    ]
    _check_for_files(files_to_check, outdir)

    # load idata
    idata_path = os.path.join(outdir, "inference_data.nc")
    idata = open_inference_data(idata_path)
    xr.testing.assert_identical(idata_orig, idata)

    # Check inference data contents
    assert "posterior" in idata.children
    assert "sample_stats" in idata.children
    assert idata["posterior"].dataset is not None
    assert idata["sample_stats"].dataset is not None
    assert "lp" in idata["sample_stats"].dataset
    assert "riae" in idata["sample_stats"].attrs
    assert "l2" in idata["sample_stats"].attrs
    assert "coverage" in idata["sample_stats"].attrs
    assert get_weights(idata) is not None

    nuts_stats = pd.read_csv(f"{outdir}/diagnostics/nuts_summary.csv")
    assert isinstance(nuts_stats["r_hat"].values[0], float)
    assert isinstance(nuts_stats["riae"].values[0], float)

    # numerical checks
    _, median_psd, _, _ = get_posterior_psd(idata)
    post_psd_scale = float(np.median(median_psd))
    assert np.isclose(post_psd_scale, psd_scale, rtol=1.0)
    assert idata.sample_stats.attrs["riae"] < 0.5
    assert idata.sample_stats.attrs["l2"] < 0.5
    assert idata.sample_stats.attrs["coverage"] > 0.8

    # check for diagnostic plots
    _check_for_files(
        [
            "diagnostics/vi_loss.png",
            "diagnostics/traces.png",
            "diagnostics/energy.png",
        ],
        outdir,
    )


def test_mcmc_multivar(outdir):
    expected_freq = _run_multivar_mcmc(outdir)
    ### NOW WE CHECK THE OUTPUTS ###
    # load idata
    idata_path = os.path.join(outdir, "inference_data.nc")
    idata = open_inference_data(idata_path)

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

    ## Check that all expected output files are present
    files_to_check = [
        "inference_data.nc",
        "psd_matrix.png",
        "diagnostics/vi_summary.csv",
        "diagnostics/nuts_summary.csv",
        "diagnostics/preprocessing_eignvalue_ratios.png",
        "diagnostics/traces.png",
        "diagnostics/energy.png",
        "diagnostics/vi_elbo.png",
    ]
    _check_for_files(files_to_check, outdir)

    print(f"++++ multivariate MCMC test COMPLETE ++++")


def _check_for_files(expected_files, outdir):
    missing_files = []
    for fname in expected_files:
        path = os.path.join(outdir, fname)
        if not os.path.exists(path):
            missing_files.append(fname)
    assert not missing_files, f"Missing expected output files: {missing_files}"


#### RUNNERS


def _run_univar_mcmc(outdir):
    from log_psplines.example_datasets.ar_data import ARData

    psd_scale = 1  # e-42

    n = 2048
    n_samples = n_warmup = 500
    n_knots = 20
    compute_lnz = True

    ar_data = ARData(
        order=4, duration=1.0, fs=n, seed=42, sigma=np.sqrt(psd_scale)
    )
    print(f"{ar_data.ts}")

    model_cfg = ModelConfig(
        n_knots=n_knots,
        true_psd=ar_data.psd_theoretical,
    )
    diagnostics_cfg = DiagnosticsConfig(
        outdir=outdir,
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
    return idata, ar_data, psd_scale


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


def _run_multivar_mcmc(outdir):
    from log_psplines.example_datasets.varma_data import VARMAData

    varma_data = VARMAData(n_samples=2**12, fs=64.0, seed=0)
    ts_data = cast(np.ndarray, varma_data.data)
    ts_run = MultivariateTimeseries(y=ts_data, t=varma_data.time)

    fmin, fmax = 0, 32
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=None,
        Nh=2,
    )

    n_samples = n_warmup = 200
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
    run_mcmc(
        data=ts_run,
        config=run_cfg,
    )
    return expected_freq
