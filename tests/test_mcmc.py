"""Integration tests for the canonical fit() pipeline path."""

import os
from typing import cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from log_psplines import PSDResult, fit
from log_psplines.config import PipelineConfig
from log_psplines.inference.evidence import (
    MorphZEvidenceResult,
    estimate_pipeline_lnz,
    run_morphz_evidence,
)
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)


def test_mcmc_p1(outdir: str):
    print("_____________p=1 MCMC_____________")
    outdir_str = str(outdir)
    result_orig, _data, psd_scale = _run_p1_mcmc(outdir_str)

    ### NOW WE CHECK THE OUTPUTS ###
    files_to_check = [
        "inference_data.nc",
        "posterior_spectrum.png",
        "diagnostics/nuts_summary.csv",
    ]
    _check_for_files(files_to_check, outdir_str)

    result = PSDResult.from_netcdf(
        os.path.join(outdir_str, "inference_data.nc")
    )
    assert set(result_orig.posterior) == set(result.posterior)
    assert result.sample_stats is not None
    assert "lp_channel_0" in result.sample_stats
    assert "step_size_channel_0" in result.sample_stats
    assert "n_steps_channel_0" in result.sample_stats
    assert bool(result.metadata["compute_lnz"])
    assert bool(result.metadata["lnz_valid"])
    assert np.isfinite(result.metadata["lnz"])
    assert np.isfinite(result.metadata["lnz_err"])

    _check_stats_are_finite(outdir_str)

    post_psd_scale = float(np.median(np.median(result.psd, axis=(0, 1))))
    assert np.isclose(post_psd_scale, psd_scale, rtol=1.0)

    # check for diagnostic plots
    _check_for_files(
        [
            "diagnostics/traces.png",
            "diagnostics/energy.png",
        ],
        outdir_str,
    )


def test_mcmc_multivar(outdir):
    outdir_str = str(outdir)
    result_orig, expected_freq = _run_multivar_mcmc(outdir_str)
    ### NOW WE CHECK THE OUTPUTS ###
    print("_____________multivariate MCMC_____________")
    _check_for_files(
        [
            "inference_data.nc",
            "posterior_spectrum.png",
            "diagnostics/nuts_summary.csv",
            "diagnostics/preprocessing_eigenvalue_ratios.png",
        ],
        outdir_str,
    )

    result = PSDResult.from_netcdf(
        os.path.join(outdir_str, "inference_data.nc")
    )
    np.testing.assert_allclose(result.spectral_density, result_orig.spectral_density)

    freq = result.frequency
    assert np.allclose(freq, expected_freq)
    qtl = result.quantiles()
    psd = np.asarray(qtl)
    assert psd.shape[1] == freq.shape[0]
    assert psd.shape[2:] == (2, 2)
    psd_median = psd[1]
    diag = np.real(np.diagonal(psd_median, axis1=1, axis2=2))
    assert np.all(diag > 0.0), "PSD diagonal elements should be positive."
    assert np.allclose(
        psd_median,
        np.swapaxes(psd_median.conj(), 1, 2),
        rtol=1e-6,
        atol=1e-8,
    ), "PSD should be Hermitian."

    assert result.vi_posterior is None
    _check_stats_are_finite(outdir_str)

    ## Check that all expected output files are present
    files_to_check = [
        "diagnostics/traces.png",
        "diagnostics/energy.png",
    ]
    _check_for_files(files_to_check, outdir_str)


@pytest.mark.skip(reason="LnZ not currently in use")
def test_multivar_morphz_all_nonconverged_is_invalid(outdir) -> None:
    rng = np.random.default_rng(0)
    post_samples = rng.normal(size=(96, 2))

    def log_posterior_fn(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float64)
        return float(
            -0.5 * np.sum(theta**2) - 0.5 * theta.size * np.log(2.0 * np.pi)
        )

    log_posterior_values = np.apply_along_axis(
        log_posterior_fn, 1, post_samples
    )
    result = run_morphz_evidence(
        post_samples=post_samples,
        log_posterior_values=log_posterior_values,
        log_posterior_function=log_posterior_fn,
        n_resamples=48,
        thin=2,
        kde_fraction=0.5,
        bridge_start_fraction=0.5,
        max_iter=1,
        tol=0.0,
        morph_type="pair",
        output_path=str(outdir),
        n_estimations=2,
        kde_bw="silverman",
        verbose=False,
        plot=False,
        show_progress=False,
    )

    assert result.n_estimations == 2
    assert result.nonconverged_count == 2
    assert not result.is_valid
    assert np.isnan(result.lnz)
    assert np.isnan(result.lnz_err)


@pytest.mark.skip(reason="LnZ not currently in use")
def test_multivar_lnz_sums_factor_results(monkeypatch) -> None:
    from log_psplines.data.timeseries import TimeSeries
    from log_psplines.example_datasets.varma_data import VARMAData
    from log_psplines.pipeline import make_pipeline

    factor_calls: list[int] = []

    def _fake_run_morphz_evidence(**kwargs) -> MorphZEvidenceResult:
        factor_index = len(factor_calls)
        factor_calls.append(factor_index)
        if factor_index == 0:
            return MorphZEvidenceResult(
                lnz=10.0,
                lnz_err=0.3,
                is_valid=True,
                n_estimations=1,
                nonconverged_count=0,
                estimates=np.asarray([[10.0, 0.3]], dtype=float),
            )
        return MorphZEvidenceResult(
            lnz=20.0,
            lnz_err=0.4,
            is_valid=True,
            n_estimations=1,
            nonconverged_count=0,
            estimates=np.asarray([[20.0, 0.4]], dtype=float),
        )

    monkeypatch.setattr(
        "log_psplines.inference.evidence.run_morphz_evidence",
        _fake_run_morphz_evidence,
    )

    varma_data = VARMAData(n_samples=2**8, fs=32.0, seed=1)
    ts_run = TimeSeries(
        data=cast(np.ndarray, varma_data.data),
        t=varma_data.time,
    )
    config = PipelineConfig(
        n_knots=6,
        degree=3,
        diffMatrixOrder=2,
        fmin=0,
        fmax=16,
        n_samples=8,
        n_warmup=8,
        Nb=2,
        coarse_grain_config=CoarseGrainConfig(enabled=True, Nc=None, Nh=2),
        verbose=False,
        outdir=None,
        compute_lnz=False,
    )
    pipeline = make_pipeline(ts_run, config)
    result = pipeline.run()

    lnz_result = estimate_pipeline_lnz(
        posterior=result.posterior,
        data=pipeline.data,
        model_kwargs=pipeline.full_model_kwargs,
        outdir=None,
        extra_kwargs={"lnz_kwargs": {"show_progress": False}},
        verbose=False,
    )

    assert factor_calls == [0, 1]
    assert lnz_result.is_valid
    assert len(lnz_result.factor_results) == 2
    assert lnz_result.lnz == pytest.approx(
        sum(factor.lnz for factor in lnz_result.factor_results)
    )
    assert lnz_result.lnz_err == pytest.approx(np.sqrt(0.3**2 + 0.4**2))


def _check_stats_are_finite(outdir) -> None:
    nuts_stats_pd = pd.read_csv(f"{outdir}/diagnostics/nuts_summary.csv")
    for key in ("step_size", "max_treedepth_hits"):
        assert key in nuts_stats_pd.columns
        assert np.isfinite(
            pd.to_numeric(nuts_stats_pd[key], errors="coerce")
        ).any()
    # R-hat is undefined for a single chain and may legitimately be NaN.
    assert "rhat_max" in nuts_stats_pd.columns

def _check_for_files(expected_files, outdir):
    missing_files = []
    for fname in expected_files:
        path = os.path.join(outdir, fname)
        if not os.path.exists(path):
            missing_files.append(fname)
    assert not missing_files, f"Missing expected output files: {missing_files}"


#### RUNNERS


def _run_p1_mcmc(outdir):
    from log_psplines.example_datasets.varma_data import VARMAData

    psd_scale = 1.0

    n = 2048
    n_samples = n_warmup = 500
    n_knots = 20
    compute_lnz = True

    data = VARMAData.ar(
        order=2,
        n_samples=n,
        fs=float(n),
        seed=42,
        sigma=np.sqrt(psd_scale),
    )
    print(f"{data.ts}")

    config = PipelineConfig(
        n_knots=n_knots,
        n_samples=n_samples,
        n_warmup=n_warmup,
        rng_key=42,
        true_psd=data.get_true_psd(),
        verbose=True,
        outdir=outdir,
        compute_lnz=compute_lnz,
        num_chains=2,
    )
    result = fit(data.ts, config=config)
    return result, data, psd_scale


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
    from log_psplines.data.timeseries import TimeSeries
    from log_psplines.example_datasets.varma_data import VARMAData

    varma_data = VARMAData(n_samples=2**12, fs=64.0, seed=0)
    ts_data = cast(np.ndarray, varma_data.data)
    ts_run = TimeSeries(data=ts_data, t=varma_data.time)

    fmin, fmax = 0, 32
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=None,
        Nh=2,
    )

    n_samples = n_warmup = 200
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
        compute_lnz=False,
        extra_kwargs={
            "lnz_kwargs": {
                "morph_type": "pair",
                "n_resamples": 64,
                "n_estimations": 1,
                "kde_bw": "silverman",
                "max_iter": 200,
                "tol": 1e-2,
                "verbose": True,
            }
        },
    )
    result = fit(data=ts_run, config=config)
    return result, expected_freq
