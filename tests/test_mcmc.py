"""Integration tests for run_mcmc (pipeline path)."""

import csv
import os
import shutil
from typing import List, cast

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
from log_psplines.mcmc import run_mcmc
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)


def test_mcmc_univar(outdir: str):
    print(f"++++ Running univariate MCMC test ++++")
    outdir_str = str(outdir)
    idata_orig, ar_data, psd_scale = _run_univar_mcmc(outdir_str)

    ### NOW WE CHECK THE OUTPUTS ###
    files_to_check = [
        "inference_data.nc",
        "posterior_predictive.png",
        "diagnostics/vi_summary.csv",
        "diagnostics/nuts_summary.csv",
    ]
    _check_for_files(files_to_check, outdir_str)

    # load idata
    idata_path = os.path.join(outdir_str, "inference_data.nc")
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

    _check_stats_are_finite(idata, outdir_str)

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
        outdir_str,
    )
    print(f"++++ univariate MCMC test COMPLETE ++++")


def test_mcmc_multivar(outdir):
    print(f"++++ Running multivariate MCMC test ++++")
    outdir_str = str(outdir)
    idata_orig, expected_freq = _run_multivar_mcmc(outdir_str)
    ### NOW WE CHECK THE OUTPUTS ###

    _check_for_files(
        [
            "inference_data.nc",
            "posterior_predictive.png",
            "diagnostics/vi_summary.csv",
            "diagnostics/nuts_summary.csv",
            "diagnostics/preprocessing_eigenvalue_ratios.png",
        ],
        outdir_str,
    )

    # load idata
    idata_path = os.path.join(outdir_str, "inference_data.nc")
    idata = open_inference_data(idata_path)
    xr.testing.assert_identical(idata_orig, idata)

    # Verify coarse-grained frequency structure
    qtl = get_multivar_posterior_psd_quantiles(idata, n_keep=2)
    freq = np.asarray(qtl["freq"], dtype=float)
    assert np.allclose(freq, expected_freq)

    # Verify PSD matrix structure for multivariate
    psd = np.asarray(qtl["psd"], dtype=np.complex128)
    psd_shape = psd.shape
    assert psd_shape[1] == freq.shape[0]
    assert psd_shape[2:] == (2, 2)

    # Verify Hermitian and positive definite
    idx50 = int(np.argmin(np.abs(np.asarray(qtl["percentile"]) - 50.0)))
    psd_median = psd[idx50]
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

    _check_stats_are_finite(idata, outdir_str)

    ## Check that all expected output files are present
    files_to_check = [
        "diagnostics/traces.png",
        "diagnostics/energy.png",
        "diagnostics/vi_loss.png",
    ]
    _check_for_files(files_to_check, outdir_str)

    print(f"++++ multivariate MCMC test COMPLETE ++++")


def _check_stats_are_finite(idata, outdir) -> None:
    vi_stats = idata["vi_sample_stats"].dataset
    nuts_stats = idata["sample_stats"].dataset
    vi_stats_pd = pd.read_csv(f"{outdir}/diagnostics/vi_summary.csv")
    nuts_stats_pd = pd.read_csv(f"{outdir}/diagnostics/nuts_summary.csv")

    def check_finite(d: dict, key: List[str]) -> None:
        for k in key:
            assert k in d, f"Key '{k}' not found in {d}."

    nuts_keys = ["riae", "l2", "coverage", "rhat_max"]
    vi_keys = ["riae", "l2", "coverage", "pareto_k_max"]

    check_finite(nuts_stats.attrs, nuts_keys)
    check_finite(nuts_stats_pd.iloc[0], nuts_keys)
    check_finite(vi_stats.attrs, vi_keys)
    check_finite(vi_stats_pd.iloc[0], vi_keys)


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

    psd_scale = 1.0

    n = 2048
    n_samples = n_warmup = 500
    n_knots = 20
    compute_lnz = True

    ar_data = ARData(
        order=4, duration=1.0, fs=n, seed=42, sigma=np.sqrt(psd_scale)
    )
    print(f"{ar_data.ts}")

    config = PipelineConfig(
        n_knots=n_knots,
        n_samples=n_samples,
        n_warmup=n_warmup,
        rng_key=42,
        true_psd=ar_data.psd_theoretical,
        verbose=True,
        outdir=outdir,
        compute_lnz=compute_lnz,
        init_from_vi=True,
        num_chains=2,
    )
    idata = run_mcmc(
        ar_data.ts,
        config=config,
    )
    return idata, ar_data, psd_scale


def _expected_coarse_freq_multivar(
    ts,
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
    from log_psplines.datatypes.multivar import MultivariateTimeseries
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
    vi_steps = 5000
    Nb = 4  # Number of blocks for Welch periodogram

    expected_freq = _expected_coarse_freq_multivar(
        ts_run,
        Nb=Nb,
        fmin=fmin,
        fmax=fmax,
        cfg=coarse_cfg,
    )

    config = PipelineConfig(
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        fmin=fmin,
        fmax=fmax,
        n_samples=n_samples,
        n_warmup=n_warmup,
        Nb=Nb,
        coarse_grain_config=coarse_cfg,
        true_psd=varma_data.get_true_psd(),
        verbose=True,
        outdir=outdir,
        init_from_vi=True,
        only_vi=False,
        vi_steps=vi_steps,
        vi_lr=5e-3,
        vi_progress_bar=False,
        vi_posterior_draws=100,
        vi_psd_max_draws=100,
    )
    idata = run_mcmc(
        data=ts_run,
        config=config,
    )
    return idata, expected_freq
