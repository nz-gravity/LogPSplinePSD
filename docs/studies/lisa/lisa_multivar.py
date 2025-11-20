from pathlib import Path

import numpy as np

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.example_datasets.lisa_data import (
    LISAData,
    covariance_matrix,
    lisa_link_noises_ldc,
    tdi2_psd_and_csd,
)
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix

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

idata = None

if RESULT_FN.exists():
    logger.info(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(str(RESULT_FN))

else:
    logger.info(f"No existing {RESULT_FN} found, running inference...")
    n_knots = 50 if RUN_VI_ONLY else 20
    idata = run_mcmc(
        data=fft_data,
        sampler="multivar_blocked_nuts",
        n_samples=1000,
        n_warmup=1000,
        n_knots=n_knots,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir=str(RESULTS_DIR),
        verbose=True,
        coarse_grain_config=coarse_cfg,
        fmin=FMIN,
        fmax=FMAX,
        true_psd=dict(freq=freqs, psd=true_psd_standardized_data),
        only_vi=RUN_VI_ONLY,
        vi_steps=10_000 if RUN_VI_ONLY else 1_500,
        vi_lr=1e-3 if RUN_VI_ONLY else 1e-2,
        vi_progress_bar=True,
    )

if idata is None:
    raise RuntimeError("Inference data was not produced or loaded.")

idata.to_netcdf(str(RESULT_FN))
logger.info(f"Saved results to {RESULT_FN}")

logger.info(idata)

freq_plot = np.asarray(idata["posterior_psd"]["freq"].values)

Spm_plot, Sop_plot = lisa_link_noises_ldc(freq_plot, fs=fs, fmin=fmin_full)
diag_true, csd_true = tdi2_psd_and_csd(freq_plot, Spm_plot, Sop_plot)
true_psd_physical = covariance_matrix(diag_true, csd_true)
true_psd_standardized = true_psd_physical

plot_psd_matrix(
    idata=idata,
    freq=freq_plot,
    true_psd=true_psd_standardized,
    outdir=str(RESULTS_DIR),
    filename="psd_matrix.png",
    diag_yscale="log",
    offdiag_yscale="log",
    xscale="log",
    show_csd_magnitude=True,
    show_coherence=False,
    overlay_vi=True,
)
