import numpy as np

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
    vi_psd = idata_vi.vi_posterior_psd["psd_matrix_real"].sel(percentile=50)
    nuts_psd = idata_nuts.posterior_psd["psd_matrix_real"].sel(percentile=50)

    vi_diag = np.asarray(vi_psd[:, channel_index, channel_index])
    nuts_diag = np.asarray(nuts_psd[:, channel_index, channel_index])
    return vi_diag / nuts_diag


def test_vi_matches_mcmc_psd_scaling():
    """VI and MCMC PSD outputs should share the same physical scaling."""

    rng_seed = 1234
    np.random.seed(rng_seed)
    varma = VARMAData(n_samples=16, seed=rng_seed)
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)

    model_cfg = ModelConfig(n_knots=3)
    diagnostics_cfg = DiagnosticsConfig(outdir=None, verbose=False)
    vi_cfg = VIConfig(
        vi_steps=10,
        vi_posterior_draws=6,
        vi_psd_max_draws=3,
        vi_progress_bar=False,
    )
    run_cfg = RunMCMCConfig(
        sampler="nuts",
        n_samples=2,
        n_warmup=2,
        rng_key=rng_seed,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
    )
    vi_only_cfg = RunMCMCConfig(
        sampler=run_cfg.sampler,
        n_samples=run_cfg.n_samples,
        n_warmup=run_cfg.n_warmup,
        rng_key=run_cfg.rng_key,
        model=run_cfg.model,
        diagnostics=run_cfg.diagnostics,
        vi=VIConfig(
            only_vi=True,
            init_from_vi=run_cfg.vi.init_from_vi,
            vi_steps=run_cfg.vi.vi_steps,
            vi_lr=run_cfg.vi.vi_lr,
            vi_guide=run_cfg.vi.vi_guide,
            vi_posterior_draws=run_cfg.vi.vi_posterior_draws,
            vi_progress_bar=run_cfg.vi.vi_progress_bar,
            vi_psd_max_draws=run_cfg.vi.vi_psd_max_draws,
        ),
        nuts=run_cfg.nuts,
        extra_kwargs=run_cfg.extra_kwargs,
    )

    idata_vi = run_mcmc(data=ts, config=vi_only_cfg)
    idata_nuts = run_mcmc(data=ts, config=run_cfg)

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
