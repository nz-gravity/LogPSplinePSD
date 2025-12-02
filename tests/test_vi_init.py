import os
import re

import numpy as np
import pytest

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc


def _mean_divergence(idata):
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

    idata = run_mcmc(
        ar_data.ts,
        sampler="nuts",
        n_knots=4,
        n_samples=40,
        n_warmup=60,
        num_chains=1,
        vi_steps=300,
        vi_lr=5e-2,
        vi_posterior_draws=128,
        verbose=False,
        compute_lnz=False,
        outdir=str(outdir),
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
    assert "PSIS k-hat" in summary_text
    match = re.search(r"PSIS k-hat \(max\): ([0-9eE+\-.]+)", summary_text)
    assert match is not None
    khat_val = float(match.group(1))
    assert ("PSIS alert" in summary_text) == (khat_val > 0.7)


@pytest.mark.parametrize("num_chains", [1, 2])
def test_univariate_arviz_chain_dims(outdir, num_chains):
    ar_data = ARData(order=1, duration=0.25, fs=32, sigma=0.25, seed=1)
    outdir = f"{outdir}/univar_arviz_chains_{num_chains}"

    idata = run_mcmc(
        ar_data.ts,
        sampler="nuts",
        n_knots=3,
        n_samples=8,
        n_warmup=8,
        num_chains=num_chains,
        vi_steps=50,
        vi_posterior_draws=32,
        verbose=False,
        compute_lnz=False,
        outdir=str(outdir),
    )

    assert idata.posterior.sizes.get("chain") == num_chains
    assert idata.sample_stats.sizes.get("chain") == num_chains


def test_multivariate_vi_initialisation_smoke(outdir):
    varma = VARMAData(n_samples=128, seed=0)
    timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)
    outdir = f"{outdir}/multivar_vi"

    idata = run_mcmc(
        timeseries,
        sampler="nuts",
        n_knots=5,
        n_samples=30,
        n_warmup=80,
        num_chains=1,
        vi_steps=500,
        vi_lr=1e-2,
        vi_posterior_draws=128,
        target_accept_prob=0.9,
        max_tree_depth=8,
        verbose=False,
        compute_lnz=False,
        outdir=str(outdir),
        n_time_blocks=2,
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
    assert "PSIS k-hat" in summary_text
    match = re.search(r"PSIS k-hat \(max\): ([0-9eE+\-.]+)", summary_text)
    assert match is not None
    khat_val = float(match.group(1))
    assert ("PSIS alert" in summary_text) == (khat_val > 0.7)


def test_multivariate_blocked_vi_initialisation_smoke(outdir):
    varma = VARMAData(n_samples=128, seed=1)
    timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)
    outdir = f"{outdir}/multivar_blocked_vi"

    idata = run_mcmc(
        timeseries,
        sampler="multivar_blocked_nuts",
        n_knots=5,
        n_samples=30,
        n_warmup=80,
        num_chains=2,
        vi_steps=400,
        vi_lr=1e-2,
        vi_posterior_draws=128,
        target_accept_prob=0.9,
        max_tree_depth=8,
        verbose=False,
        compute_lnz=False,
        outdir=str(outdir),
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
