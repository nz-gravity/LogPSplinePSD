import numpy as np

from log_psplines.arviz_utils import (
    get_multivar_posterior_psd_quantiles,
    get_multivar_vi_psd_quantiles,
)
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    MultivariateTimeseries,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)


def _diag_ratio(idata_vi, idata_nuts, channel_index):
    vi_quantiles = get_multivar_vi_psd_quantiles(idata_vi)
    vi_idx50 = int(
        np.argmin(np.abs(np.asarray(vi_quantiles["percentile"]) - 50.0))
    )
    vi_psd = np.asarray(vi_quantiles["real"])[vi_idx50]
    nuts_quantiles = get_multivar_posterior_psd_quantiles(idata_nuts)
    idx50 = int(
        np.argmin(np.abs(np.asarray(nuts_quantiles["percentile"]) - 50.0))
    )
    nuts_psd = np.asarray(nuts_quantiles["real"])[idx50]

    vi_diag = np.asarray(vi_psd[:, channel_index, channel_index])
    nuts_diag = np.asarray(nuts_psd[:, channel_index, channel_index])
    return vi_diag / nuts_diag


def test_vi_matches_mcmc_psd_scaling():
    """VI and MCMC PSD outputs should share the same physical scaling."""

    rng_seed = 1234
    np.random.seed(rng_seed)
    varma = VARMAData(n_samples=16, seed=rng_seed)
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)
    idata_vi = run_mcmc(data=ts, init_from_vi=True, only_vi=True)
    idata_nuts = run_mcmc(data=ts, init_from_vi=False, only_vi=False)

    median_ratios = []

    for ch in range(2):
        ratio = _diag_ratio(idata_vi, idata_nuts, ch)
        finite_ratio = ratio[np.isfinite(ratio)]
        assert finite_ratio.size > 0

        median_ratio = np.nanmedian(finite_ratio)
        median_ratios.append(median_ratio)
        mad = np.nanmedian(np.abs(finite_ratio - 1.0))

        assert 0.3 < median_ratio < 3.5
        assert mad < 2.5, f"Channel {ch} median deviation {mad:.2f} too large"

    # Both channels should show consistent rescaling behaviour.
    np.testing.assert_allclose(
        median_ratios[0], median_ratios[1], rtol=1.0, atol=1.0
    )
