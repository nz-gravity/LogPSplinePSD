import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.arviz_utils import (
    get_multivar_posterior_psd_quantiles,
    get_posterior_psd,
    get_weights,
)
from log_psplines.example_datasets.ar_data import ARData
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
    if test_mode == "fast":
        n_samples = n_warmup = 3
        n = 64
        n_knots = 3
        compute_lnz = False

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

    print(f"Inference data posterior variables: {idata.posterior}")

    fig, ax = plot_pdgrm(idata=idata, show_data=False)
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

    _, median_psd, _, _ = get_posterior_psd(idata)
    post_psd_scale = float(np.median(median_psd))
    print(
        f"Posterior PSD scale (median): {post_psd_scale:.2e}, expected ~{psd_scale:.2e}"
    )
    # Should be within 1 order of magnitude
    assert np.isclose(
        post_psd_scale, psd_scale, rtol=1.0
    ), "Posterior PSD scale is not within expected range."

    print(f"++++ univariate MCMC test {test_mode} COMPLETE ++++")


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
    )
    return np.asarray(spec.f_coarse, dtype=np.float64)


@pytest.mark.slow
def test_run_mcmc_coarse_grain_multivar(test_mode: str):
    """Test multivariate PSD analysis with coarse grain, VI init, and blocking."""
    t, y = _synthetic_multivar_series()
    ts_run = MultivariateTimeseries(y=y.copy(), t=t.copy())
    ts_spec = MultivariateTimeseries(y=y.copy(), t=t.copy())

    fmin, fmax = 0.01, 0.4
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=None,
        Nh=1,
    )
    Nb = 2 if test_mode != "fast" else 1
    n_samples = n_warmup = 120 if test_mode != "fast" else 3
    vi_steps = 30 if test_mode != "fast" else 5

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
    diagnostics_cfg = DiagnosticsConfig(verbose=(test_mode != "fast"))
    vi_cfg = VIConfig(
        init_from_vi=True,
        only_vi=False,
        vi_steps=vi_steps,
        vi_lr=5e-3,
        vi_progress_bar=False,
        vi_posterior_draws=6,
        vi_psd_max_draws=2,
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
    psd_real = np.asarray(quantiles["real"])
    psd_shape = psd_real.shape
    assert psd_shape[1] == freq.shape[0], "PSD frequency dimension mismatch."
    assert psd_shape[2:] == (2, 2), "PSD matrix should be 2x2 per frequency."

    # Verify Hermitian and positive definite
    idx50 = int(np.argmin(np.abs(np.asarray(quantiles["percentile"]) - 50.0)))
    psd_median = psd_real[idx50]
    diag = np.diagonal(psd_median, axis1=1, axis2=2)
    assert np.all(diag > 0.0), "PSD diagonal elements should be positive."
    assert np.allclose(
        psd_median, np.swapaxes(psd_median, 1, 2), rtol=1e-6, atol=1e-8
    ), "PSD should be Hermitian."

    print(f"++++ multivariate MCMC test {test_mode} COMPLETE ++++")
