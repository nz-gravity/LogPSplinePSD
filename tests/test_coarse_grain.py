import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_graining_multivar_fft,
    apply_coarse_graining_univar,
    compute_binning_structure,
    plot_coarse_vs_original,
    plot_coarse_vs_original_multivar,
)
from log_psplines.datatypes import MultivarFFT
from log_psplines.plotting import plot_pdgrm


def test_compute_binning_structure_simple():
    freqs = np.linspace(1e-5, 1e-2, 100)
    cfg = CoarseGrainConfig(
        enabled=True,
        f_transition=5e-3,
        n_log_bins=5,
        f_min=2e-5,
        f_max=8e-3,
    )

    spec = compute_binning_structure(
        freqs,
        f_transition=cfg.f_transition,
        n_log_bins=cfg.n_log_bins,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )

    assert spec.f_coarse[0] == np.min(spec.f_coarse)
    assert np.all(spec.f_coarse[:-1] <= spec.f_coarse[1:])
    assert spec.n_low == spec.mask_low.sum()
    assert spec.n_bins_high == spec.bin_counts.size

    indices = np.nonzero(spec.selection_mask)[0]
    power_selected = np.sin(indices * 0.1) ** 2 + 1.0
    freqs_selected = freqs[indices]
    power_coarse, weights = apply_coarse_graining_univar(
        power_selected, spec, freqs_selected
    )

    assert power_coarse.shape[0] == weights.shape[0]
    assert power_coarse.shape[0] == spec.f_coarse.shape[0]
    assert np.all(weights[: spec.n_low] == 1.0)
    # total weight should equal number of selected fine frequencies
    assert np.isclose(
        weights.sum(), spec.mask_low.sum() + spec.bin_counts.sum()
    )


def test_coarse_weights_properties_log_bins():
    freqs = np.linspace(0.5, 128.0, 512)
    power = np.ones_like(freqs)
    cfg = CoarseGrainConfig(
        enabled=True,
        f_transition=16.0,
        n_log_bins=50,
        f_min=1.0,
        f_max=120.0,
    )

    spec = compute_binning_structure(
        freqs,
        f_transition=cfg.f_transition,
        n_log_bins=cfg.n_log_bins,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    selection = spec.selection_mask
    power_sel = power[selection]
    freqs_sel = freqs[selection]

    _, weights = apply_coarse_graining_univar(power_sel, spec, freqs_sel)

    assert np.all(weights[: spec.n_low] == 1.0)
    # Non-negativity and correct total mass
    assert np.all(weights >= 0)
    assert np.isclose(
        weights.sum(), spec.mask_low.sum() + spec.bin_counts.sum()
    )


def test_coarse_grain_multivar_fft():
    rng = np.random.default_rng(0)
    n_freq = 64
    n_dim = 3
    freqs = np.linspace(1.0, 256.0, n_freq)
    n_theta = n_dim * (n_dim - 1) // 2

    y_re = rng.standard_normal((n_freq, n_dim))
    y_im = rng.standard_normal((n_freq, n_dim))
    Z_re = rng.standard_normal((n_freq, n_dim, n_theta))
    Z_im = rng.standard_normal((n_freq, n_dim, n_theta))

    fft = MultivarFFT(
        y_re=y_re,
        y_im=y_im,
        Z_re=Z_re,
        Z_im=Z_im,
        freq=freqs,
        n_freq=n_freq,
        n_dim=n_dim,
        scaling_factor=1.0,
        fs=512.0,
    )

    spec = compute_binning_structure(
        freqs,
        f_transition=64.0,
        n_log_bins=8,
        f_min=2.0,
        f_max=240.0,
    )

    result = apply_coarse_graining_multivar_fft(fft, spec)

    assert result.fft.n_freq == spec.n_low + np.count_nonzero(spec.bin_counts)
    assert result.fft.y_re.shape == (result.fft.n_freq, n_dim)
    assert result.weights.shape[0] == result.fft.n_freq
    assert np.all(result.weights[: spec.n_low] == 1.0)
    total_expected = spec.mask_low.sum() + spec.bin_counts.sum()
    assert np.isclose(result.weights.sum(), total_expected)
    assert result.fft.csd_sums.shape == (result.fft.n_freq, n_dim, n_dim)
    assert result.fft.bin_weights.shape[0] == result.fft.n_freq
    selected = (y_re + 1j * y_im)[spec.selection_mask]
    expected_diag = np.abs(selected[spec.mask_low, 0]) ** 2
    np.testing.assert_allclose(
        result.fft.csd_sums[: spec.n_low, 0, 0].real, expected_diag
    )
    assert np.allclose(result.weights, result.fft.bin_weights)
    plt.close("all")


def test_plot_coarse_vs_original_multivar(outdir):
    outdir = f"{outdir}/out_coarse_grain/plot_multivar"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(10)
    n_freq = 32
    n_dim = 2
    freqs = np.linspace(1.0, 64.0, n_freq)
    n_theta = n_dim * (n_dim - 1) // 2

    fft = MultivarFFT(
        y_re=rng.standard_normal((n_freq, n_dim)),
        y_im=rng.standard_normal((n_freq, n_dim)),
        Z_re=rng.standard_normal((n_freq, n_dim, n_theta)),
        Z_im=rng.standard_normal((n_freq, n_dim, n_theta)),
        freq=freqs,
        n_freq=n_freq,
        n_dim=n_dim,
        scaling_factor=1.0,
        fs=128.0,
    )

    spec = compute_binning_structure(
        freqs,
        f_transition=16.0,
        n_log_bins=6,
        f_min=2.0,
        f_max=60.0,
    )

    fig, axes_pair, weights = plot_coarse_vs_original_multivar(fft, spec)
    fig.savefig(f"{outdir}/coarse_vs_fine_multivar.png")

    assert weights.shape[0] == spec.f_coarse.shape[0]
    assert len(axes_pair) == 2
    plt.close(fig)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") is not None,
    reason="Skip on GitHub Actions for time",
)
def test_coarse_lnl_with_univar_mcmc(outdir):
    from log_psplines.example_datasets import ARData
    from log_psplines.mcmc import run_mcmc

    outdir = f"{outdir}/out_coarse_grain/univar"
    os.makedirs(outdir, exist_ok=True)

    ar_data = ARData(order=4, duration=1, fs=2048, seed=0)

    standardized = ar_data.ts.standardise_for_psd()
    periodogram_full = standardized.to_periodogram()

    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        f_transition=100.0,
        f_max=ar_data.ts.fs / 2,
        n_log_bins=100,
    )

    spec = compute_binning_structure(
        np.asarray(periodogram_full.freqs),
        f_transition=coarse_cfg.f_transition,
        n_log_bins=coarse_cfg.n_log_bins,
        f_min=coarse_cfg.f_min,
        f_max=coarse_cfg.f_max,
    )

    fig, ax, weights = plot_coarse_vs_original(
        periodogram_full.freqs,
        periodogram_full.power,
        spec,
        scaling_factor=standardized.scaling_factor,
        transition_freq=coarse_cfg.f_transition,
    )

    # Plot the weights for diagnostics
    from log_psplines.coarse_grain import plot_coarse_grain_weights

    fig_weights, ax_weights = plot_coarse_grain_weights(
        spec,
        weights,
        transition_freq=coarse_cfg.f_transition,
    )
    fig_weights.savefig(f"{outdir}/coarse_grain_weights.png")
    ax = fig.gca()
    ax.loglog(
        ar_data.freqs, ar_data.psd_theoretical, "k--", label="Theoretical PSD"
    )
    fig.savefig(f"{outdir}/coarse_grain_example.png")

    idata = run_mcmc(
        ar_data.ts,
        sampler="nuts",
        n_knots=15,
        n_samples=500,
        n_warmup=500,
        outdir=f"{outdir}/coarse_grain",
        rng_key=0,
        compute_lnz=False,
        init_from_vi=True,
        verbose=False,
        coarse_grain_config=coarse_cfg,
        knot_kwargs=dict(method="uniform"),
    )

    idata2 = run_mcmc(
        ar_data.ts,
        sampler="nuts",
        n_knots=15,
        n_samples=500,
        n_warmup=500,
        outdir=f"{outdir}/full_freq",
        rng_key=0,
        compute_lnz=False,
        init_from_vi=True,
        verbose=False,
        knot_kwargs=dict(method="uniform"),
    )

    # check that the frequency coordinates match the coarse frequencies
    freq_coords = np.asarray(idata.posterior_psd["psd"].coords["freq"].values)
    obs_freqs = np.asarray(
        idata.observed_data["periodogram"].coords["freq"].values
    )
    assert freq_coords.shape[0] == spec.f_coarse.shape[0]
    assert obs_freqs.shape[0] == spec.f_coarse.shape[0]
    assert np.allclose(freq_coords, spec.f_coarse, rtol=1e-6, atol=1e-9)

    # finally, lets make a comparison PSD plot between coarse and full
    fig, ax, _ = plot_coarse_vs_original(
        periodogram_full.freqs,
        periodogram_full.power,
        spec,
        scaling_factor=standardized.scaling_factor,
        transition_freq=coarse_cfg.f_transition,
    )
    ax = fig.gca()
    ax.loglog(
        ar_data.freqs, ar_data.psd_theoretical, "k--", label="Theoretical PSD"
    )
    plot_pdgrm(
        idata=idata,
        ax=ax,
        show_data=False,
        model_label="Coarse-grained NUTS",
    )
    plot_pdgrm(
        idata=idata2,
        ax=ax,
        show_data=False,
        model_label="Full freq",
        model_color="tab:green",
    )
    ax.legend()
    fig.savefig(f"{outdir}/coarse_vs_full_psd.png")

    fig, ax = plot_pdgrm(
        idata=idata,
        show_data=False,
        model_label="Coarse-grained NUTS",
        knot_color="tab:orange",
    )
    plot_pdgrm(
        idata=idata2,
        ax=ax,
        show_data=False,
        model_label="Full freq NUTS",
        model_color="tab:green",
        knot_color="tab:green",
    )
    ax.loglog(
        ar_data.freqs, ar_data.psd_theoretical, "k--", label="Theoretical PSD"
    )
    ax.legend()
    fig.savefig(f"{outdir}/coarse_vs_full_psd_just_splines.png")


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") is not None,
    reason="Skip on GitHub Actions for time",
)
def test_coarse_lnl_with_multivar_mcmc(outdir):

    from log_psplines.datatypes import MultivariateTimeseries
    from log_psplines.example_datasets.varma_data import VARMAData
    from log_psplines.mcmc import run_mcmc

    outdir = f"{outdir}/out_coarse_grain/multivar"
    os.makedirs(outdir, exist_ok=True)

    varma = VARMAData(
        n_samples=1024,
    )
    n_dim = varma.dim
    varma.plot(fname=os.path.join(outdir, "varma_data.png"))

    print(f"VARMA data shape: {varma.data.shape}, dim={n_dim}")

    timeseries = MultivariateTimeseries(
        t=varma.time,
        y=varma.data,
    )
    coarse_settings = CoarseGrainConfig(
        enabled=True,
        f_transition=varma.fs / 5,
        f_max=varma.fs / 2,
        n_log_bins=50,
    )

    run_mcmc(
        data=timeseries,
        sampler="multivar_blocked_nuts",
        n_knots=10,
        n_samples=500,
        n_warmup=500,
        outdir=f"{outdir}/full_freq",
        rng_key=0,
        compute_lnz=False,
        init_from_vi=True,
        verbose=False,
    )

    run_mcmc(
        data=timeseries,
        sampler="multivar_blocked_nuts",
        n_knots=10,
        n_samples=500,
        n_warmup=500,
        outdir=f"{outdir}/coarse_grain",
        rng_key=0,
        compute_lnz=False,
        init_from_vi=True,
        verbose=False,
        coarse_grain_config=coarse_settings,
    )
