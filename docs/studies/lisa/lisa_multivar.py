from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD, _get_coherence
from log_psplines.example_datasets.lisa_data import (
    LISAData,
    covariance_matrix,
    lisa_link_noises_ldc,
    tdi2_psd_and_csd,
)
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import create_sampler, run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix
from log_psplines.plotting.vi import plot_vi_initial_psd_matrix
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines
from log_psplines.samplers.multivar.multivar_blocked_nuts import (
    _blocked_channel_model,
)
from log_psplines.samplers.vi_init.adapters import prepare_block_vi

set_level("DEBUG")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FN = RESULTS_DIR / "inference_data.nc"

RUN_VI_ONLY = True

lisa_data = LISAData.load()
lisa_data.plot(f"{RESULTS_DIR}/lisa_raw.png")

t = lisa_data.time
raw_series = MultivariateTimeseries(y=lisa_data.data, t=t)
standardized_ts = raw_series.standardise_for_psd()

# detrmine number of time blocks based on data length
n = raw_series.y.shape[0]

# make n_blocks so each block is ~ 1week long, power of 2
target_blocks = max(1, 2 ** int(np.round(np.log2(n / (24 * 7)))))
while target_blocks > 1 and n % target_blocks != 0:
    target_blocks //= 2

# Require each block to contain at least one quarter of the samples.
while target_blocks > 4:
    target_blocks //= 2

n_blocks = target_blocks
n_inside_block = n // n_blocks
logger.info(
    f"Using n_blocks={n_blocks} x {n_inside_block} (n_time={n})",
)


FMIN, FMAX = 10**-4, 10**-1

fft_data = standardized_ts.to_wishart_stats(
    n_blocks=n_blocks,
    fmin=FMIN,
    fmax=FMAX,
)
logger.info(fft_data)

freq_weights = None
if RUN_VI_ONLY:
    spec = compute_binning_structure(
        fft_data.freq,
        f_transition=5e-3,
        n_log_bins=200,
        f_min=FMIN,
        f_max=FMAX,
    )
    fft_data, freq_weights = coarse_grain_multivar_fft(fft_data, spec)

freqs = np.asarray(fft_data.freq)
coarse_cfg = CoarseGrainConfig(
    enabled=True,
    f_transition=5e-3,
    n_log_bins=200,
    f_min=FMIN,
    f_max=FMAX,
)

dt = t[1] - t[0]
fs = 1.0 / dt
fmin_full = 1.0 / (len(t) * dt)
Spm_data, Sop_data = lisa_link_noises_ldc(freqs, fs=fs, fmin=fmin_full)
diag_data, csd_data = tdi2_psd_and_csd(freqs, Spm_data, Sop_data)
true_psd_physical_data = covariance_matrix(diag_data, csd_data)
true_psd_standardized_data = true_psd_physical_data


if RUN_VI_ONLY:
    spline_model = MultivariateLogPSplines.from_multivar_fft(
        fft_data,
        n_knots=50,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
    )

    sampler = create_sampler(
        data=fft_data,
        model=spline_model,
        sampler_type="multivar_blocked_nuts",
        num_chains=1,
        verbose=True,
        outdir=str(RESULTS_DIR),
        target_accept_prob=0.8,
        max_tree_depth=10,
        init_from_vi=True,
        vi_steps=30_000,
        vi_lr=1e-3,
        vi_guide=None,
        vi_posterior_draws=256,
        vi_progress_bar=True,
        scaling_factor=fft_data.scaling_factor,
        channel_stds=fft_data.channel_stds,
        true_psd=true_psd_standardized_data,
        freq_weights=freq_weights,
    )

    vi_artifacts = prepare_block_vi(
        sampler, rng_key=sampler.rng_key, block_model=_blocked_channel_model
    )
    sampler.rng_key = vi_artifacts.rng_key
    sampler._vi_diagnostics = vi_artifacts.diagnostics
    vi_diag = sampler._vi_diagnostics or {}

    if not vi_diag.get("psd_quantiles"):
        raise RuntimeError("VI diagnostics missing PSD quantiles.")

    empirical_psd = sampler._compute_empirical_psd()
    freq_plot = np.asarray(sampler.freq)

    plot_vi_initial_psd_matrix(
        outfile=str(RESULTS_DIR / "psd_matrix.png"),
        freq=freq_plot,
        empirical_psd=empirical_psd,
        true_psd=true_psd_standardized_data,
        psd_quantiles=vi_diag["psd_quantiles"],
        coherence_quantiles=vi_diag.get("coherence_quantiles"),
        diag_yscale="log",
        offdiag_yscale="log",
        xscale="log",
        show_csd_magnitude=True,
        show_coherence=False,
    )

elif RESULT_FN.exists() and True:
    logger.info(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(str(RESULT_FN))

else:

    idata = run_mcmc(
        data=fft_data,
        sampler="multivar_blocked_nuts",
        n_samples=1000,
        n_warmup=1000,
        n_knots=20,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir=str(RESULTS_DIR),
        verbose=True,
        coarse_grain_config=coarse_cfg,
        fmin=FMIN,
        fmax=FMAX,
        true_psd=true_psd_standardized_data,
    )

if not RUN_VI_ONLY:
    logger.info(idata)

    freq_plot = np.asarray(idata.posterior_psd["freq"].values)

    if fft_data.raw_psd is not None:
        psd_array = np.asarray(fft_data.raw_psd)
        empirical_psd = EmpiricalPSD(
            freq=freqs,
            psd=psd_array,
            coherence=_get_coherence(psd_array),
        )
    else:
        empirical_psd = fft_data.empirical_psd

    Spm_plot, Sop_plot = lisa_link_noises_ldc(freq_plot, fs=fs, fmin=fmin_full)
    diag_true, csd_true = tdi2_psd_and_csd(freq_plot, Spm_plot, Sop_plot)
    true_psd_physical = covariance_matrix(diag_true, csd_true)
    true_psd_standardized = true_psd_physical

    plot_psd_matrix(
        idata=idata,
        freq=freq_plot,
        empirical_psd=empirical_psd,
        true_psd=true_psd_standardized,
        outdir=str(RESULTS_DIR),
        filename=f"psd_matrix.png",
        diag_yscale="log",
        offdiag_yscale="log",
        xscale="log",
        show_csd_magnitude=True,
        show_coherence=False,
    )
