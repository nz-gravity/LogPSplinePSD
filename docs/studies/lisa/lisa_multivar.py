from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lisa_data import (
    TEN_DAYS,
    LISAData,
    OMS_model,
    TM_model,
    Total_model,
    download,
)

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix

set_level("DEBUG")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FN = RESULTS_DIR / "inference_data.nc"
DATA_DIR = HERE / "data"
DATA_FPATH = DATA_DIR / "tdi.h5"
if not DATA_FPATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download(TEN_DAYS, dest_folder=str(DATA_DIR))
lisa_data = LISAData.from_hdf5(str(DATA_FPATH))

t = lisa_data.t
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

freqs = fft_data.raw_freq if fft_data.raw_freq is not None else fft_data.freq
diag_labels = ["X2", "Y2", "Z2"]
periodogram_fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
tm_psd = (
    TM_model(
        freqs,
        duration=lisa_data.duration,
        fs=lisa_data.fs,
    )
    ** 2
)
oms_psd = (
    OMS_model(
        freqs,
        duration=lisa_data.duration,
        fs=lisa_data.fs,
    )
    ** 2
)
total_psd = (
    Total_model(
        freqs,
        duration=lisa_data.duration,
        fs=lisa_data.fs,
    )
    ** 2
)
for idx, channel in enumerate(diag_labels):
    psd_diag = np.real(fft_data.raw_psd[:, idx, idx])
    axes[idx].loglog(freqs, psd_diag, color="C0", label="Block PSD")
    axes[idx].loglog(
        freqs, tm_psd, "tab:blue", linestyle="--", label="TM model"
    )
    axes[idx].loglog(
        freqs, oms_psd, "tab:red", linestyle="--", label="OMS model"
    )
    axes[idx].loglog(freqs, total_psd, "k-", label="Total model")
    axes[idx].set_ylabel(f"{channel} PSD [1/Hz]")
    axes[idx].set_xlim(FMIN, FMAX)
    axes[idx].legend(loc="upper right")

axes[-1].set_xlabel("Frequency [Hz]")
for ax in axes:
    ax.set_yscale("log")
    ax.set_xscale("log")
periodogram_path = RESULTS_DIR / "periodograms.png"
periodogram_fig.tight_layout()
periodogram_fig.savefig(periodogram_path, dpi=200, bbox_inches="tight")
plt.close(periodogram_fig)
logger.info(
    "Saved block-averaged periodogram with analytical overlays to %s",
    periodogram_path,
)

coarse_cfg = CoarseGrainConfig(
    enabled=True,
    f_transition=5e-3,
    n_log_bins=200,
    f_min=FMIN,
    f_max=FMAX,
)


if RESULT_FN.exists() and False:
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
    )

logger.info(idata)


freq_plot = np.asarray(idata.posterior_psd["freq"].values)

plot_psd_matrix(
    idata=idata,
    freq=freq_plot,
    empirical_psd=fft_data.empirical_psd,
    outdir=str(RESULTS_DIR),
    filename=f"psd_matrix.png",
    diag_yscale="log",
    xscale="log",
)
