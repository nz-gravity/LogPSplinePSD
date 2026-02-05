import os

import jax.numpy as jnp
import numpy as np
import pytest

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_graining_univar,
    coarse_grain_multivar_fft,
    compute_binning_structure,
    plot_coarse_vs_original,
)
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.diagnostics.coarse_grain_checks import (
    bin_doubling_stiffness_check_from_u_stacks,
    coarse_bin_likelihood_equivalence_check,
)
from log_psplines.plotting import plot_pdgrm
from log_psplines.samplers.univar.univar_base import log_likelihood
from log_psplines.spectrum_utils import (
    sum_wishart_outer_products,
    wishart_matrix_to_psd,
)


def test_compute_binning_structure_simple():
    freqs = np.linspace(1e-5, 1e-2, 100)
    cfg = CoarseGrainConfig(
        enabled=True,
        n_bins=5,
        f_min=2e-5,
        f_max=8e-3,
    )

    spec = compute_binning_structure(
        freqs,
        n_bins=cfg.n_bins,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )

    assert spec.f_coarse[0] == np.min(spec.f_coarse)
    assert np.all(spec.f_coarse[:-1] <= spec.f_coarse[1:])
    assert spec.n_low == 0
    assert spec.n_bins_high == spec.bin_counts.size

    indices = np.nonzero(spec.selection_mask)[0]
    power_selected = np.sin(indices * 0.1) ** 2 + 1.0
    freqs_selected = freqs[indices]
    power_coarse, weights = apply_coarse_graining_univar(
        power_selected, spec, freqs_selected
    )

    assert power_coarse.shape[0] == weights.shape[0]
    assert power_coarse.shape[0] == spec.f_coarse.shape[0]
    # total weight should equal number of selected fine frequencies
    assert np.isclose(weights.sum(), spec.bin_counts.sum())


def test_compute_binning_structure_linear_bins_equal_width():
    freqs = np.linspace(1.0, 101.0, 1001)
    spec = compute_binning_structure(
        freqs,
        n_bins=9,
        f_min=1.0,
        f_max=101.0,
    )

    assert spec.n_bins_high > 0
    assert spec.bin_counts.shape == (spec.n_bins_high,)
    assert np.all(spec.bin_counts % 2 == 1)
    assert np.max(spec.bin_counts) - np.min(spec.bin_counts) <= 2


def test_compute_binning_structure_linear_fixed_size_midpoint_frequency():
    freqs = np.arange(1.0, 21.0, 1.0)  # 20 freqs
    spec = compute_binning_structure(
        freqs,
        n_freqs_per_bin=5,
        f_min=freqs[0],
        f_max=freqs[-1],
    )

    assert spec.n_low == 0
    assert spec.n_bins_high == 4
    assert np.all(spec.bin_counts == 5)
    assert np.allclose(spec.f_coarse, np.array([3.0, 8.0, 13.0, 18.0]))


def test_compute_binning_structure_linear_fixed_size_allows_remainder():
    freqs = np.arange(1.0, 24.0, 1.0)  # 23 freqs -> 4 bins of 5 + remainder 3
    spec = compute_binning_structure(
        freqs,
        n_freqs_per_bin=5,
        f_min=freqs[0],
        f_max=freqs[-1],
    )

    assert spec.n_low == 0
    assert spec.n_bins_high == 5
    assert np.all(spec.bin_counts[:-1] == 5)
    assert spec.bin_counts[-1] == 3
    assert np.all(spec.bin_counts % 2 == 1)
    assert np.allclose(spec.f_coarse, np.array([3.0, 8.0, 13.0, 18.0, 22.0]))


def test_compute_binning_structure_linear_fixed_size_absorbs_even_remainder():
    freqs = np.arange(1.0, 23.0, 1.0)  # 22 freqs -> 3 bins of 5 + last bin 7
    spec = compute_binning_structure(
        freqs,
        n_freqs_per_bin=5,
        f_min=freqs[0],
        f_max=freqs[-1],
    )

    assert spec.n_low == 0
    assert spec.n_bins_high == 4
    assert np.all(spec.bin_counts[:3] == 5)
    assert spec.bin_counts[-1] == 7
    assert np.all(spec.bin_counts % 2 == 1)
    assert np.allclose(spec.f_coarse, np.array([3.0, 8.0, 13.0, 19.0]))


def test_apply_coarse_graining_univar_uses_sum():
    freqs = np.arange(1.0, 10.0, 1.0)  # 9 freqs
    power = np.arange(1.0, 10.0, 1.0)
    spec = compute_binning_structure(
        freqs,
        n_bins=3,
        f_min=freqs[0],
        f_max=freqs[-1],
    )
    power_coarse, weights = apply_coarse_graining_univar(power, spec, freqs)

    expected = np.array(
        [np.sum(power[:3]), np.sum(power[3:6]), np.sum(power[6:9])]
    )
    np.testing.assert_allclose(power_coarse, expected)
    np.testing.assert_allclose(weights, spec.bin_counts.astype(float))


def test_coarse_weights_properties_full_band():
    freqs = np.linspace(0.5, 128.0, 512)
    power = np.ones_like(freqs)
    cfg = CoarseGrainConfig(
        enabled=True,
        n_bins=24,
        f_min=1.0,
        f_max=120.0,
    )

    spec = compute_binning_structure(
        freqs,
        n_bins=cfg.n_bins,
        n_freqs_per_bin=cfg.n_freqs_per_bin,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    selection = spec.selection_mask
    power_sel = power[selection]
    freqs_sel = freqs[selection]

    _, weights = apply_coarse_graining_univar(power_sel, spec, freqs_sel)

    assert np.all(weights > 0)
    assert np.all(weights == spec.bin_counts)
    assert np.isclose(weights.sum(), spec.bin_counts.sum())


def test_univar_loglik_fine_equals_coarse_sum():
    power = np.array([1.5, 2.0, 3.5, 4.0], dtype=np.float64)
    n_h = power.size
    log_s = np.log(2.5)

    basis = jnp.zeros((n_h, 1), dtype=jnp.float32)
    weights = jnp.zeros((1,), dtype=jnp.float32)
    log_parametric = jnp.full((n_h,), log_s, dtype=jnp.float32)
    ll_fine = log_likelihood(
        weights,
        jnp.log(jnp.asarray(power)),
        basis,
        log_parametric,
        jnp.ones((n_h,), dtype=jnp.float32),
    )

    power_sum = np.sum(power)
    basis_c = jnp.zeros((1, 1), dtype=jnp.float32)
    log_parametric_c = jnp.full((1,), log_s, dtype=jnp.float32)
    ll_coarse = log_likelihood(
        weights,
        jnp.log(jnp.asarray([power_sum])),
        basis_c,
        log_parametric_c,
        jnp.asarray([float(n_h)], dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        np.asarray(ll_coarse), np.asarray(ll_fine), rtol=1e-10, atol=1e-10
    )


def test_multivar_coarse_psd_matches_bin_average():
    rng = np.random.default_rng(12)
    n = 512
    t = np.arange(n)
    base = rng.normal(size=n)
    y = np.column_stack(
        [
            base + 0.1 * rng.normal(size=n),
            0.8 * base + 0.2 * rng.normal(size=n),
        ]
    )
    ts = MultivariateTimeseries(y=y, t=t)
    fft_full = ts.to_wishart_stats(n_blocks=4)

    spec = compute_binning_structure(
        fft_full.freq,
        n_bins=8,
        f_min=fft_full.freq[0],
        f_max=fft_full.freq[-1],
    )

    fft_coarse, weights = coarse_grain_multivar_fft(fft_full, spec)
    assert np.isclose(weights.sum(), spec.bin_counts.sum())

    u_complex = (fft_full.u_re + 1j * fft_full.u_im)[spec.selection_mask]
    u_high_sorted = u_complex[spec.sort_indices]
    bin_counts = spec.bin_counts.astype(int)

    manual_psd = []
    pos = 0
    base_nu = int(max(int(fft_full.nu), 1))
    scaling = float(fft_full.scaling_factor or 1.0)
    duration = float(getattr(fft_full, "duration", 1.0) or 1.0)

    for idx in range(spec.f_coarse.shape[0]):
        count = bin_counts[idx]
        if count <= 0:
            continue
        sl = slice(pos, pos + count)
        pos += count
        u_stack = u_high_sorted[sl]
        weight = np.array([float(count)])
        y_sum = sum_wishart_outer_products(u_stack)
        psd = wishart_matrix_to_psd(
            y_sum[None, ...],
            nu=base_nu,
            duration=duration,
            scaling_factor=scaling,
            weights=weight,
        )[0]
        manual_psd.append(psd)

    manual_psd = np.stack(manual_psd, axis=0)
    assert manual_psd.shape == fft_coarse.raw_psd.shape
    max_rel = np.max(
        np.abs(manual_psd - fft_coarse.raw_psd) / (np.abs(manual_psd) + 1e-12)
    )
    assert max_rel < 1e-10


def test_multivar_coarse_likelihood_equivalence_per_bin_theta_off():
    rng = np.random.default_rng(123)
    # Choose n so that with n_blocks=4 the retained positive-frequency count is
    # divisible by an odd n_freqs_per_bin (here 5). With block_len=100, rfft
    # yields 51 bins including 0, and we drop the 0 bin -> 50.
    n = 400
    t = np.arange(n)
    base = rng.normal(size=n)
    y = np.column_stack(
        [
            base + 0.2 * rng.normal(size=n),
            0.7 * base + 0.3 * rng.normal(size=n),
        ]
    )
    ts = MultivariateTimeseries(y=y, t=t)
    fft_full = ts.to_wishart_stats(n_blocks=4)

    # Paper-style full-band linear coarse graining with fixed odd bin size.
    spec = compute_binning_structure(
        fft_full.freq,
        n_freqs_per_bin=5,
        f_min=fft_full.freq[0],
        f_max=fft_full.freq[-1],
    )

    # Choose a non-degenerate constant parameter value to make scaling visible.
    # theta is set to zero to avoid basis-invariance concerns in the equivalence check.
    log_delta_sq_bin = np.log(np.array([2.0, 3.0], dtype=np.float64))

    res = coarse_bin_likelihood_equivalence_check(
        fft_fine=fft_full,
        spec=spec,
        bin_index=0,
        nu=int(fft_full.nu),
        log_delta_sq_bin=log_delta_sq_bin,
        include_theta=False,
    )
    # Expect equivalence up to numerical roundoff.
    assert abs(res.delta) < 1e-8


def test_bin_doubling_stiffness_proxy_scales_linearly():
    rng = np.random.default_rng(0)
    n_dim = 3
    n_small = 5
    n_large = 10

    u_stack_small = (
        rng.normal(size=(n_small, n_dim, n_dim))
        + 1j * rng.normal(size=(n_small, n_dim, n_dim))
    ).astype(np.complex128)
    u_stack_large = (
        rng.normal(size=(n_large, n_dim, n_dim))
        + 1j * rng.normal(size=(n_large, n_dim, n_dim))
    ).astype(np.complex128)

    log_delta_sq_bin = np.log(np.array([1.5, 2.0, 3.0], dtype=np.float64))
    n_theta = n_dim * (n_dim - 1) // 2
    theta_re = np.zeros((n_theta,), dtype=np.float64)
    theta_im = np.zeros((n_theta,), dtype=np.float64)

    # Direction along the last diagonal log-scale.
    direction = np.zeros((n_dim + 2 * n_theta,), dtype=np.float64)
    direction[n_dim - 1] = 1.0

    res = bin_doubling_stiffness_check_from_u_stacks(
        u_stack_small=u_stack_small,
        u_stack_large=u_stack_large,
        nu=4.0,
        log_delta_sq_bin=log_delta_sq_bin,
        theta_re_bin=theta_re,
        theta_im_bin=theta_im,
        direction=direction,
        epsilons=(1e-3, 1e-2),
    )

    assert res.n_members_small == n_small
    assert res.n_members_large == n_large
    for entry in res.eps_results:
        assert np.isfinite(entry.ratio)
        assert abs(entry.ratio - entry.expected_ratio) < 0.05


@pytest.mark.slow
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
        f_max=ar_data.ts.fs / 2,
        n_bins=100,
    )

    spec = compute_binning_structure(
        np.asarray(periodogram_full.freqs),
        n_bins=coarse_cfg.n_bins,
        f_min=coarse_cfg.f_min,
        f_max=coarse_cfg.f_max,
    )

    fig, ax, weights = plot_coarse_vs_original(
        periodogram_full.freqs,
        periodogram_full.power,
        spec,
        scaling_factor=standardized.scaling_factor,
    )

    # Plot the weights for diagnostics
    from log_psplines.coarse_grain import plot_coarse_grain_weights

    fig_weights, ax_weights = plot_coarse_grain_weights(
        spec,
        weights,
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
        n_samples=80,
        n_warmup=80,
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
        n_samples=80,
        n_warmup=80,
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
