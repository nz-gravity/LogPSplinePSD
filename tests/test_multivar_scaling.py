import numpy as np
import pytest

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries, Timeseries
from log_psplines.datatypes.multivar import _interp_complex_matrix
from log_psplines.example_datasets.varma_data import (
    VARMAData,
    _calculate_true_varma_psd,
)
from log_psplines.mcmc import _coarse_grain_processed_data, run_mcmc
from log_psplines.samplers.multivar.multivar_base import MultivarBaseSampler


def _ratio_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    return {
        "min": float(np.nanmin(arr)),
        "median": float(np.nanmedian(arr)),
        "max": float(np.nanmax(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p90": float(np.nanpercentile(arr, 90)),
    }


def test_multivar_rescale_matches_empirical():
    rng = np.random.default_rng(1234)
    var_coeffs = np.array([[[0.45, 0.0], [0.08, 0.28]]], dtype=float)
    vma_coeffs = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=float)
    sigma_base = np.array([[1.0, 0.15], [0.15, 0.7]], dtype=float)
    amplitude = 5.0
    sigma = sigma_base * amplitude**2

    n_time = 120
    n_time_blocks = 2
    varma = VARMAData(
        n_samples=n_time,
        var_coeffs=var_coeffs,
        vma_coeffs=vma_coeffs,
        sigma=sigma,
        seed=rng.integers(0, 1_000_000),
    )
    timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)

    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        f_transition=varma.freq[len(varma.freq) // 5],
        n_log_bins=6,
    )

    standardized = timeseries.standardise_for_psd()
    assert standardized.scaling_factor > 5.0
    fft_data = standardized.to_wishart_stats(n_blocks=n_time_blocks)
    processed_data, _, _ = _coarse_grain_processed_data(
        fft_data, coarse_cfg, scaled_true_psd=None
    )
    assert processed_data.raw_psd is not None

    dummy_sampler = type("DummySampler", (), {})()
    dummy_sampler.fft_data = processed_data
    raw_psd = np.asarray(processed_data.raw_psd)
    scale_matrix = np.outer(
        processed_data.channel_stds, processed_data.channel_stds
    )
    sf = float(processed_data.scaling_factor or 1.0)
    expected_empirical = raw_psd / sf * scale_matrix[None, :, :]
    rescaled_empirical = MultivarBaseSampler._rescale_psd(
        dummy_sampler, raw_psd
    )
    np.testing.assert_allclose(
        rescaled_empirical,
        expected_empirical,
        rtol=1e-6,
        atol=1e-12,
    )


@pytest.mark.filterwarnings("ignore:Matplotlib")
@pytest.mark.slow
def test_multivar_scaling_matches_periodogram_and_truth(outdir):
    """Blocked NUTS posterior should align with the periodogram and analytic PSD."""

    rng = np.random.default_rng(1234)
    var_coeffs = np.array([[[0.45, 0.0], [0.08, 0.28]]], dtype=float)
    vma_coeffs = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=float)
    sigma_base = np.array([[1.0, 0.15], [0.15, 0.7]], dtype=float)
    amplitude = 5.0
    sigma = sigma_base * amplitude**2

    n_time = 240
    n_time_blocks = 3
    varma = VARMAData(
        n_samples=n_time,
        var_coeffs=var_coeffs,
        vma_coeffs=vma_coeffs,
        sigma=sigma,
        seed=rng.integers(0, 1_000_000),
    )
    timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)

    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        f_transition=varma.freq[len(varma.freq) // 5],
        n_log_bins=8,
    )

    idata = run_mcmc(
        data=timeseries,
        sampler="multivar_blocked_nuts",
        n_knots=5,
        n_samples=12,
        n_warmup=12,
        n_time_blocks=n_time_blocks,
        coarse_grain_config=coarse_cfg,
        vi_steps=80,
        vi_posterior_draws=24,
        vi_progress_bar=False,
        rng_key=0,
        verbose=False,
        outdir=f"{outdir}/multivar_scaling",
    )

    freq_grid = np.asarray(idata.posterior_psd["freq"].values)
    psd_med = idata.posterior_psd["psd_matrix_real"].sel(percentile=50).values
    periodogram = idata.observed_data["periodogram"].values

    true_psd_full = _calculate_true_varma_psd(
        freqs_hz=varma.freq,
        dim=varma.dim,
        var_coeffs=var_coeffs,
        vma_coeffs=vma_coeffs,
        sigma=sigma,
        fs=varma.fs,
        channel_stds=None,
        scaling_factor=1.0,
    )
    true_psd_interp = _interp_complex_matrix(
        varma.freq, freq_grid, true_psd_full
    )

    diag_ratios = {}
    for idx in (0, 1):
        post_over_periodogram = np.real(psd_med[:, idx, idx]) / np.real(
            periodogram[:, idx, idx]
        )
        post_over_true = np.real(psd_med[:, idx, idx]) / np.real(
            true_psd_interp[:, idx, idx]
        )
        diag_ratios[idx] = {
            "post_periodogram": _ratio_stats(post_over_periodogram),
            "post_true": _ratio_stats(post_over_true),
        }
        print(
            f"Channel {idx}: post/periodogram {diag_ratios[idx]['post_periodogram']} "
            f"post/true {diag_ratios[idx]['post_true']}"
        )

    for idx in diag_ratios:
        stats_pp = diag_ratios[idx]["post_periodogram"]
        stats_pt = diag_ratios[idx]["post_true"]
        assert 0.6 < stats_pp["median"] < 1.6
        assert 0.4 < stats_pt["median"] < 2.0
        assert stats_pp["p90"] < 4.0
        assert stats_pp["p10"] > 0.2


def _simulate_independent_ar1(
    n: int, phi: float, sigma: float, seed: int
) -> np.ndarray:
    """Simulate two independent AR(1) channels with identical dynamics."""

    rng = np.random.default_rng(seed)
    eps = rng.normal(scale=sigma, size=(n + 1, 2))
    data = np.zeros((n + 1, 2), dtype=float)
    for idx in range(1, n + 1):
        data[idx] = phi * data[idx - 1] + eps[idx]
    return data[1:]


def test_univariate_and_multivar_scaling_consistency():
    """Rescaling should align univariate and multivariate empirical PSDs."""
    n_time = 128
    phi = 0.65
    sigma = 0.9

    data = _simulate_independent_ar1(n_time, phi=phi, sigma=sigma, seed=123)
    t = np.arange(n_time)
    ts_single_0 = Timeseries(t=t, y=data[:, 0])
    ts_single_1 = Timeseries(t=t, y=data[:, 1])
    ts_multi = MultivariateTimeseries(t=t, y=data)

    pdgrm0 = ts_single_0.standardise_for_psd().to_periodogram()
    pdgrm1 = ts_single_1.standardise_for_psd().to_periodogram()
    fft = ts_multi.standardise_for_psd().to_wishart_stats(
        n_blocks=1, window=None
    )
    assert fft.raw_psd is not None

    dummy_sampler = type("DummySampler", (), {})()
    dummy_sampler.fft_data = fft
    rescaled = MultivarBaseSampler._rescale_psd(
        dummy_sampler, np.asarray(fft.raw_psd)
    )

    np.testing.assert_allclose(pdgrm0.freqs, fft.freq)
    np.testing.assert_allclose(pdgrm1.freqs, fft.freq)

    phys0 = pdgrm0.power * pdgrm0.scaling_factor
    phys1 = pdgrm1.power * pdgrm1.scaling_factor
    max_idx = -1

    np.testing.assert_allclose(
        rescaled[:max_idx, 0, 0].real,
        phys0[:max_idx],
        rtol=1e-6,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rescaled[:max_idx, 1, 1].real,
        phys1[:max_idx],
        rtol=1e-6,
        atol=1e-10,
    )
