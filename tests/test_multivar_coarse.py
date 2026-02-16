import os
import warnings

import numpy as np
import pytest

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    MultivariateTimeseries,
    RunMCMCConfig,
    run_mcmc,
)
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") is not None,
    reason="Skip multivariate coarse-grain test on CI for time",
)
def test_multivar_coarse_vs_full(outdir, test_mode):
    """Compare multivariate results with/without coarse-grained likelihood."""
    outdir = f"{outdir}/out_coarse_grain/multivar"
    os.makedirs(outdir, exist_ok=True)

    # Problem size and sampling budget
    n = 256 if test_mode != "fast" else 64
    n_samples = n_warmup = 120 if test_mode != "fast" else 8
    n_knots = 12 if test_mode != "fast" else 3

    # Simulated data
    np.random.seed(0)
    varma = VARMAData(n_samples=n)
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)
    std_ts = ts.standardise_for_psd()
    channel_stds = std_ts.original_stds
    scaling_factor = float(std_ts.scaling_factor or 1.0)
    if channel_stds is not None:
        scale_matrix = np.outer(channel_stds, channel_stds) / scaling_factor
    else:
        scale_matrix = scaling_factor * np.ones((ts.p, ts.p))

    def to_physical(psd: np.ndarray) -> np.ndarray:
        return psd * scale_matrix

    # Full run (baseline)
    full_dir = os.path.join(outdir, "multivar_full")
    model_cfg = ModelConfig(n_knots=n_knots, true_psd=varma.get_true_psd())
    diagnostics_cfg = DiagnosticsConfig(outdir=full_dir, verbose=False)
    run_cfg_full = RunMCMCConfig(
        sampler="nuts",
        n_samples=n_samples,
        n_warmup=n_warmup,
        Nb=2 if test_mode != "fast" else 1,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
    )
    idata_full = run_mcmc(
        data=ts,
        config=run_cfg_full,
    )

    # Coarse-grained run
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=16 if test_mode != "fast" else 16,
        f_min=None,
        f_max=None,
    )
    coarse_dir = os.path.join(outdir, "multivar_coarse")
    diagnostics_cfg = DiagnosticsConfig(outdir=coarse_dir, verbose=False)
    run_cfg_coarse = RunMCMCConfig(
        sampler="nuts",
        n_samples=n_samples,
        n_warmup=n_warmup,
        Nb=2 if test_mode != "fast" else 1,
        coarse_grain_config=coarse_cfg,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
    )
    idata_coarse = run_mcmc(
        data=ts,
        config=run_cfg_coarse,
    )

    # Sanity checks on shapes
    psd_full = idata_full.posterior_psd["psd_matrix_real"]
    psd_coarse = idata_coarse.posterior_psd["psd_matrix_real"]
    nfreq_full = psd_full.sizes["freq"]
    nfreq_coarse = psd_coarse.sizes["freq"]
    assert nfreq_coarse <= nfreq_full
    # Coarse graining should reduce frequency count unless very small n
    if nfreq_full > 32:
        assert nfreq_coarse < nfreq_full

    # Compare median PSD along diagonal channels on the coarse grid
    q50_full = psd_full.sel(percentile=50).values  # (freq, p, p)
    q50_coarse = psd_coarse.sel(percentile=50).values

    freqs_full = np.asarray(idata_full.posterior_psd["freq"].values)
    freqs_coarse = np.asarray(idata_coarse.posterior_psd["freq"].values)

    from log_psplines.mcmc import _interp_psd_array

    q50_full_interp = to_physical(
        _interp_psd_array(q50_full, freqs_full, freqs_coarse)
    )
    q50_coarse_phys = to_physical(q50_coarse)
    true_psd_full = varma.get_true_psd()
    true_interp = to_physical(
        _interp_psd_array(true_psd_full, varma.freq, freqs_coarse)
    )

    # Require reasonable relative agreement on the diagonal elements.
    # Use a symmetric relative error to avoid blow-ups near zeros.
    diag_full = np.diagonal(q50_full_interp, axis1=1, axis2=2)
    diag_coarse = np.diagonal(q50_coarse_phys, axis1=1, axis2=2)
    denom = np.abs(diag_full) + np.abs(diag_coarse) + 1e-12
    rel_err = 2.0 * np.abs(diag_full - diag_coarse) / denom
    assert np.nanmedian(rel_err) < 0.35  # coarse ≈ full within 35% median
    true_diag = np.diagonal(true_interp, axis1=1, axis2=2)
    denom_true = np.abs(true_diag) + np.abs(diag_coarse) + 1e-12
    rel_err_true = 2.0 * np.abs(diag_coarse - true_diag) / denom_true

    # Overlay posterior matrices together with the true PSD for quick inspection
    true_psd = varma.get_true_psd()
    spec = PSDMatrixPlotSpec(
        idata=idata_full,
        true_psd=true_psd,
        label="Full",
        save=False,
        close=False,
    )
    fig, ax = plot_psd_matrix(spec)
    overlay_spec = PSDMatrixPlotSpec(
        idata=idata_coarse,
        true_psd=None,
        label="Coarse",
        model_color="tab:orange",
        fig=fig,
        ax=ax,
        save=False,
        close=False,
    )
    plot_psd_matrix(overlay_spec)
    fig.savefig(os.path.join(outdir, "psd_matrix_overlay.png"), dpi=150)

    # Diagnostics: empirical PSD stored in the coarse run should match the
    # coarse-grained Wishart statistics computed directly from the data.
    Nb = 2 if test_mode != "fast" else 1
    standardized_ts = ts.standardise_for_psd()
    fft_full = standardized_ts.to_wishart_stats(
        Nb=Nb,
        fmin=None,
        fmax=None,
    )
    spec_manual = compute_binning_structure(
        fft_full.freq,
        Nc=coarse_cfg.Nc,
        Nh=coarse_cfg.Nh,
        f_min=coarse_cfg.f_min,
        f_max=coarse_cfg.f_max,
    )
    fft_manual_coarse = apply_coarse_grain_multivar_fft(fft_full, spec_manual)
    periodogram_obs = idata_coarse.observed_data["periodogram"].values
    manual_psd_physical = to_physical(fft_manual_coarse.raw_psd)

    assert periodogram_obs.shape == manual_psd_physical.shape
    diff = np.abs(periodogram_obs - manual_psd_physical)
    denom = np.abs(manual_psd_physical) + 1e-12
    rel_max = np.max(diff / denom)

    # Allow a slightly looser tolerance to accommodate stochastic variation
    assert (
        np.nanmedian(rel_err_true) < 0.45
    ), "Coarse PSD should match true within 45% median"
    # The coarse-grained observed periodogram should still track the manual
    # coarse computation within a modest tolerance.
    assert rel_max < 0.3, f"Max rel error {rel_max:.2e} too large"
