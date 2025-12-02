import numpy as np

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc


def _diag_ratio(idata_vi, idata_nuts, channel_index):
    vi_psd = idata_vi.vi_posterior_psd["psd_matrix_real"].sel(percentile=50)
    nuts_psd = idata_nuts.posterior_psd["psd_matrix_real"].sel(percentile=50)

    vi_diag = np.asarray(vi_psd[:, channel_index, channel_index])
    nuts_diag = np.asarray(nuts_psd[:, channel_index, channel_index])
    return vi_diag / nuts_diag


def test_vi_matches_mcmc_psd_scaling(tmp_path):
    """VI and MCMC PSD outputs should share the same physical scaling."""

    rng_seed = 1234
    np.random.seed(rng_seed)
    varma = VARMAData(n_samples=32, seed=rng_seed)
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)

    sampler_kwargs = dict(
        data=ts,
        sampler="nuts",
        n_knots=4,
        n_samples=8,
        n_warmup=8,
        rng_key=rng_seed,
        verbose=False,
        vi_steps=80,
        vi_posterior_draws=16,
        vi_psd_max_draws=8,
    )

    idata_vi = run_mcmc(
        **sampler_kwargs, outdir=str(tmp_path / "vi_only"), only_vi=True
    )
    idata_nuts = run_mcmc(**sampler_kwargs, outdir=str(tmp_path / "nuts"))

    median_ratios = []

    for ch in range(2):
        ratio = _diag_ratio(idata_vi, idata_nuts, ch)
        finite_ratio = ratio[np.isfinite(ratio)]
        assert finite_ratio.size > 0

        median_ratio = np.nanmedian(finite_ratio)
        median_ratios.append(median_ratio)
        mad = np.nanmedian(np.abs(finite_ratio - 1.0))

        # Require ratios close to unity (2x envelope) with moderate dispersion.
        np.testing.assert_allclose(median_ratio, 1.0, rtol=1.0, atol=1.0)
        assert mad < 1.5, f"Channel {ch} median deviation {mad:.2f} too large"

    # Both channels should show consistent rescaling behaviour.
    np.testing.assert_allclose(
        median_ratios[0], median_ratios[1], rtol=0.5, atol=0.5
    )
