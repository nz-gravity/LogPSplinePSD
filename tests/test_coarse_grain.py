import os

import numpy as np
import pytest

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_graining_univar,
    compute_binning_structure,
    plot_coarse_vs_original,
)
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
