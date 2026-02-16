import os
import re

import numpy as np
import pytest

from log_psplines.arviz_utils.to_arviz import _prepare_samples_and_stats
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


def _mean_divergence(idata):
    if not hasattr(idata, "sample_stats"):
        return 0.0
    div = idata.sample_stats.get("diverging")
    if div is None:
        return 0.0
    values = np.asarray(div.values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(values.mean())


def test_univariate_vi_initialisation_smoke(outdir):
    ar_data = ARData(order=2, duration=0.5, fs=64, sigma=0.5, seed=0)
    outdir = f"{outdir}/univar_vi"

    model_cfg = ModelConfig(n_knots=4)
    diagnostics_cfg = DiagnosticsConfig(
        outdir=str(outdir),
        verbose=False,
        compute_lnz=False,
    )
    vi_cfg = VIConfig(
        vi_steps=20,
        vi_lr=5e-2,
        vi_posterior_draws=12,
        vi_psd_max_draws=4,
        only_vi=True,
    )
    run_cfg = RunMCMCConfig(
        sampler="nuts",
        n_samples=2,
        n_warmup=2,
        num_chains=1,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
    )
    idata = run_mcmc(
        ar_data.ts,
        config=run_cfg,
    )

    assert "posterior" in idata.groups()
    assert "delta" in idata.posterior.data_vars
    assert _mean_divergence(idata) < 0.5
    assert os.path.exists(f"{outdir}/diagnostics/vi_initial_psd.png")
    assert os.path.exists(f"{outdir}/diagnostics/vi_elbo_trace.png")
    summary_file = os.path.join(outdir, "vi_diagnostics_summary.txt")
    assert os.path.exists(summary_file)
    with open(summary_file) as f:
        summary_text = f.read()
    assert "VI Diagnostics Summary" in summary_text
    match = re.search(r"PSIS k-hat \(max\): ([0-9eE+\-.]+)", summary_text)
    if match is not None:
        khat_val = float(match.group(1))
        assert ("PSIS alert" in summary_text) == (khat_val > 0.7)


@pytest.mark.parametrize("num_chains", [1, 2])
def test_univariate_arviz_chain_dims(num_chains):
    samples = {"weights": np.zeros((num_chains, 3, 4), dtype=float)}
    sample_stats = {"accept_prob": np.zeros((num_chains, 3), dtype=float)}
    out_samples, out_stats = _prepare_samples_and_stats(
        samples, sample_stats, num_chains=num_chains
    )

    assert out_samples["weights"].shape[0] == num_chains
    assert out_stats["accept_prob"].shape[0] == num_chains


def test_multivariate_vi_initialisation_smoke(outdir):
    varma = VARMAData(n_samples=64, seed=0)
    timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)
    outdir = f"{outdir}/multivar_vi"

    model_cfg = ModelConfig(n_knots=5)
    diagnostics_cfg = DiagnosticsConfig(
        outdir=str(outdir),
        verbose=False,
        compute_lnz=False,
    )
    vi_cfg = VIConfig(
        vi_steps=20,
        vi_lr=1e-2,
        vi_posterior_draws=12,
        vi_psd_max_draws=4,
        only_vi=True,
    )
    nuts_cfg = NUTSConfigOverride(
        target_accept_prob=0.9,
        max_tree_depth=8,
    )
    run_cfg = RunMCMCConfig(
        sampler="nuts",
        n_samples=2,
        n_warmup=2,
        num_chains=1,
        Nb=2,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
        nuts=nuts_cfg,
    )
    idata = run_mcmc(
        timeseries,
        config=run_cfg,
    )

    assert "posterior" in idata.groups()
    assert any(name.startswith("delta_") for name in idata.posterior.data_vars)
    assert _mean_divergence(idata) < 0.5
    assert os.path.exists(f"{outdir}/diagnostics/vi_initial_psd_matrix.png")
    assert os.path.exists(f"{outdir}/diagnostics/vi_elbo_trace.png")
    summary_file = os.path.join(outdir, "vi_diagnostics_summary.txt")
    assert os.path.exists(summary_file)
    with open(summary_file) as f:
        summary_text = f.read()
    assert "VI Diagnostics Summary" in summary_text
    match = re.search(r"PSIS k-hat \(max\): ([0-9eE+\-.]+)", summary_text)
    if match is not None:
        khat_val = float(match.group(1))
        assert ("PSIS alert" in summary_text) == (khat_val > 0.7)


def test_multivariate_blocked_vi_initialisation_smoke(outdir):
    varma = VARMAData(n_samples=64, seed=1)
    timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)
    outdir = f"{outdir}/multivar_blocked_vi"

    model_cfg = ModelConfig(n_knots=5)
    diagnostics_cfg = DiagnosticsConfig(
        outdir=str(outdir),
        verbose=False,
        compute_lnz=False,
    )
    vi_cfg = VIConfig(
        vi_steps=20,
        vi_lr=1e-2,
        vi_posterior_draws=12,
        vi_psd_max_draws=4,
        only_vi=True,
    )
    nuts_cfg = NUTSConfigOverride(
        target_accept_prob=0.9,
        max_tree_depth=8,
    )
    run_cfg = RunMCMCConfig(
        sampler="multivar_blocked_nuts",
        n_samples=2,
        n_warmup=2,
        num_chains=1,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
        nuts=nuts_cfg,
    )
    idata = run_mcmc(
        timeseries,
        config=run_cfg,
    )

    print(idata)

    assert "posterior" in idata.groups()
    assert "vi_posterior_psd" in idata.groups()
    assert "posterior_psd" in idata.groups()
    assert any(
        name.startswith("weights_delta_") for name in idata.posterior.data_vars
    )
    assert _mean_divergence(idata) < 0.5
    assert os.path.exists(f"{outdir}/diagnostics/vi_initial_psd_matrix.png")
    assert os.path.exists(f"{outdir}/diagnostics/vi_elbo_trace.png")
