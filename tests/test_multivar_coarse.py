import os

import numpy as np
import pytest

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc
from log_psplines.plotting import plot_psd_matrix


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") is not None,
    reason="Skip multivariate coarse-grain test on CI for time",
)
def test_multivar_coarse_vs_full(outdir, test_mode):
    """Compare multivariate results with/without coarse-grained likelihood."""
    outdir = f"{outdir}/out_coarse_grain/multivar"
    os.makedirs(outdir, exist_ok=True)

    # Problem size and sampling budget
    n = 512 if test_mode != "fast" else 128
    n_samples = n_warmup = 300 if test_mode != "fast" else 16
    n_knots = 8 if test_mode != "fast" else 4

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
        scale_matrix = scaling_factor * np.ones((ts.n_channels, ts.n_channels))

    def to_physical(psd: np.ndarray) -> np.ndarray:
        return psd * scale_matrix

    # Full run (baseline)
    full_dir = os.path.join(outdir, "multivar_full")
    idata_full = run_mcmc(
        data=ts,
        sampler="nuts",  # maps to multivar_blocked_nuts
        n_knots=n_knots,
        n_samples=n_samples,
        n_warmup=n_warmup,
        outdir=full_dir,
        verbose=False,
        n_time_blocks=2 if test_mode != "fast" else 1,
        true_psd=varma.get_true_psd(),
    )

    # Coarse-grained run
    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        f_transition=varma.freq[len(varma.freq) // 4],  # quarter band
        n_log_bins=10 if test_mode != "fast" else 6,
        f_min=None,
        f_max=None,
    )
    coarse_dir = os.path.join(outdir, "multivar_coarse")
    idata_coarse = run_mcmc(
        data=ts,
        sampler="nuts",
        n_knots=n_knots,
        n_samples=n_samples,
        n_warmup=n_warmup,
        outdir=coarse_dir,
        verbose=False,
        n_time_blocks=2 if test_mode != "fast" else 1,
        coarse_grain_config=coarse_cfg,
        true_psd=varma.get_true_psd(),
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

    # Compare median PSD along diagonal channels in overlapping low-band
    q50_full = psd_full.sel(percentile=50).values  # (freq, dim, dim)
    q50_coarse = psd_coarse.sel(percentile=50).values

    # Restrict to the low (non-aggregated) region of the coarse run
    from log_psplines.coarse_grain import compute_binning_structure as _cbs

    freqs_full = np.asarray(idata_full.posterior_psd["freq"].values)
    freqs_coarse = np.asarray(idata_coarse.posterior_psd["freq"].values)
    spec = _cbs(
        freqs_coarse,
        f_transition=coarse_cfg.f_transition,
        n_log_bins=coarse_cfg.n_log_bins,
        f_min=coarse_cfg.f_min,
        f_max=coarse_cfg.f_max,
    )
    n_low = int(spec.n_low)
    if n_low > 4:  # only compare when we kept a few low bins
        # Interpolate full median PSD onto coarse frequencies in the low region
        from log_psplines.mcmc import _interp_psd_array

        q50_full_low = to_physical(
            _interp_psd_array(q50_full, freqs_full, freqs_coarse[:n_low])
        )
        q50_coarse_low = to_physical(q50_coarse[:n_low])
        true_psd_full = varma.get_true_psd()
        true_low = _interp_psd_array(
            true_psd_full, varma.freq, freqs_coarse[:n_low]
        )

    # Require reasonable relative agreement on the diagonal elements.
    # Use a symmetric relative error to avoid blow-ups near zeros.
    diag_full = np.diagonal(q50_full_low, axis1=1, axis2=2)
    diag_coarse = np.diagonal(q50_coarse_low, axis1=1, axis2=2)
    denom = np.abs(diag_full) + np.abs(diag_coarse) + 1e-12
    rel_err = 2.0 * np.abs(diag_full - diag_coarse) / denom
    assert np.nanmedian(rel_err) < 0.25  # coarse ≈ full within 25% median
    true_diag = np.diagonal(true_low, axis1=1, axis2=2)
    denom_true = np.abs(true_diag) + np.abs(diag_coarse) + 1e-12
    rel_err_true = 2.0 * np.abs(diag_coarse - true_diag) / denom_true

    # Overlay posterior matrices together with the true PSD for quick inspection
    true_psd = varma.get_true_psd()
    fig, ax = plot_psd_matrix(
        idata=idata_full,
        true_psd=true_psd,
        label="Full",
        save=False,
        close=False,
    )
    plot_psd_matrix(
        idata=idata_coarse,
        true_psd=None,  # avoid duplicating the true curve
        label="Coarse",
        model_color="tab:orange",
        fig=fig,
        ax=ax,
        save=False,
        close=False,
    )
    fig.savefig(os.path.join(outdir, "psd_matrix_overlay.png"), dpi=150)

    # Diagnostics: empirical PSD stored in the coarse run should match the
    # coarse-grained Wishart statistics computed directly from the data.
    n_blocks = 2 if test_mode != "fast" else 1
    standardized_ts = ts.standardise_for_psd()
    fft_full = standardized_ts.to_wishart_stats(
        n_blocks=n_blocks,
        fmin=None,
        fmax=None,
    )
    spec_manual = compute_binning_structure(
        fft_full.freq,
        f_transition=coarse_cfg.f_transition,
        n_log_bins=coarse_cfg.n_log_bins,
        f_min=coarse_cfg.f_min,
        f_max=coarse_cfg.f_max,
    )
    fft_manual_coarse, weights_manual = coarse_grain_multivar_fft(
        fft_full, spec_manual
    )
    periodogram_obs = idata_coarse.observed_data["periodogram"].values
    manual_psd_physical = to_physical(fft_manual_coarse.raw_psd)

    assert periodogram_obs.shape == manual_psd_physical.shape
    diff = np.abs(periodogram_obs - manual_psd_physical)
    denom = np.abs(manual_psd_physical) + 1e-12
    rel_max = np.max(diff / denom)

    assert (
        np.nanmedian(rel_err_true) < 0.3
    ), "Coarse PSD should match true within 30% median"
    assert rel_max < 5e-6, f"Max rel error {rel_max:.2e} too large"
