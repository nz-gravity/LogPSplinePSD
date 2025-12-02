import numpy as np
import pytest

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
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


@pytest.mark.filterwarnings("ignore:Matplotlib")
def test_multivar_scaling_matches_periodogram_and_truth(outdir):
    """Blocked NUTS posterior should align with the periodogram and analytic PSD."""

    rng = np.random.default_rng(1234)
    var_coeffs = np.array([[[0.45, 0.0], [0.08, 0.28]]], dtype=float)
    vma_coeffs = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=float)
    sigma = np.array([[1.0, 0.15], [0.15, 0.7]], dtype=float)

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

    # Mirror the preprocessing in run_mcmc to validate the internal PSD rescaling
    standardized = timeseries.standardise_for_psd()
    fft_data = standardized.to_wishart_stats(n_blocks=n_time_blocks)
    processed_data, _, _ = _coarse_grain_processed_data(
        fft_data, coarse_cfg, scaled_true_psd=None
    )
    assert processed_data.raw_psd is not None

    # Empirical PSD rescaling should remove the global scaling_factor before
    # applying channel standard deviations.
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

    idata = run_mcmc(
        data=timeseries,
        sampler="multivar_blocked_nuts",
        n_knots=5,
        n_samples=24,
        n_warmup=24,
        n_time_blocks=n_time_blocks,
        coarse_grain_config=coarse_cfg,
        vi_steps=200,
        vi_posterior_draws=32,
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
        assert 0.75 < stats_pp["median"] < 1.25
        assert 0.6 < stats_pt["median"] < 1.35
        assert stats_pp["p90"] < 3.0
        assert stats_pp["p10"] > 0.25
