import os
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Print number of JAX devices
import jax
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
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix

logger.info(f"JAX devices: {jax.devices()}")

set_level("DEBUG")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FN = RESULTS_DIR / "inference_data.nc"

RUN_VI_ONLY = False
REUSE_EXISTING = False  # set True to skip sampling when results already exist

# Hyperparameters and spline configuration for this study
ALPHA_DELTA = 1.0
BETA_DELTA = 1.0
N_KNOTS = 50
TARGET_ACCEPT = 0.95
MAX_TREE_DEPTH = 12

lisa_data = LISAData.load(data_path="data/tdi.h5")
lisa_data.plot(f"{RESULTS_DIR}/lisa_raw.png")

# Trim data so that n_time is divisible by the desired number of blocks.
desired_blocks = 4
t_full = lisa_data.time
y_full = lisa_data.data
n_time = y_full.shape[0]
remainder = n_time % desired_blocks
if remainder != 0:
    n_trim = remainder
    logger.info(
        f"Trimming {n_trim} samples from end to make n_time divisible by {desired_blocks} blocks."
    )
    t_full = t_full[:-n_trim]
    y_full = y_full[:-n_trim]

t = t_full
raw_series = MultivariateTimeseries(y=y_full, t=t)
standardized_ts = raw_series.standardise_for_psd()

n = raw_series.y.shape[0]
n_blocks = desired_blocks
n_inside_block = n // n_blocks
logger.info(
    f"Using n_blocks={n_blocks} x {n_inside_block} (n_time={n})",
)


FMIN, FMAX = 10**-4, 3 * 10**-2

fft_data = standardized_ts.to_wishart_stats(
    n_blocks=n_blocks,
    fmin=FMIN,
    fmax=FMAX,
)
logger.info(fft_data)

freqs = np.asarray(fft_data.freq)
coarse_cfg = CoarseGrainConfig(
    enabled=True,
    f_transition=1e-3,
    n_log_bins=200,
    f_min=FMIN,
    f_max=FMAX,
)

# Build an explicit coarse-grained FFT on the same grid that the sampler sees.
spec = compute_binning_structure(
    freqs,
    f_transition=coarse_cfg.f_transition,
    n_log_bins=coarse_cfg.n_log_bins,
    f_min=coarse_cfg.f_min,
    f_max=coarse_cfg.f_max,
)
fft_coarse, freq_weights = coarse_grain_multivar_fft(fft_data, spec)
logger.info(fft_coarse)

dt = t[1] - t[0]
fs = 1.0 / dt
fmin_full = 1.0 / (len(t) * dt)
Spm_data, Sop_data = lisa_link_noises_ldc(freqs, fs=fs, fmin=fmin_full)
diag_data, csd_data = tdi2_psd_and_csd(freqs, Spm_data, Sop_data)
true_psd_physical_data = covariance_matrix(diag_data, csd_data)
true_psd_standardized_data = true_psd_physical_data

idata = None

if RESULT_FN.exists() and REUSE_EXISTING:
    logger.info(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(str(RESULT_FN))

else:
    logger.info(f"No existing {RESULT_FN} found, running inference...")
    idata = run_mcmc(
        data=fft_coarse,
        sampler="multivar_blocked_nuts",
        n_samples=1500,
        n_warmup=1500,
        num_chains=4,
        n_knots=N_KNOTS,
        degree=2,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir=str(RESULTS_DIR),
        verbose=True,
        fmin=FMIN,
        fmax=FMAX,
        alpha_delta=ALPHA_DELTA,
        beta_delta=BETA_DELTA,
        only_vi=False,
        vi_steps=30_000,
        vi_lr=1e-3,
        vi_posterior_draws=256,
        vi_progress_bar=True,
        target_accept_prob=TARGET_ACCEPT,
        max_tree_depth=MAX_TREE_DEPTH,
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

# Empirical PSD on the *coarse Wishart grid* (what the likelihood fits)
if fft_coarse.raw_psd is not None:
    psd_array = np.asarray(fft_coarse.raw_psd)
    empirical_wishart = EmpiricalPSD(
        freq=np.asarray(fft_coarse.freq),
        psd=psd_array,
        coherence=_get_coherence(psd_array),
    )
else:
    empirical_wishart = fft_coarse.empirical_psd

# Traditional Welch-style empirical PSD on the original (unstandardised) data
empirical_welch = EmpiricalPSD.from_timeseries_data(
    data=raw_series.y,
    fs=raw_series.fs,
)

plot_psd_matrix(
    idata=idata,
    freq=freq_plot,
    empirical_psd=empirical_wishart,
    extra_empirical_psd=[empirical_welch],
    extra_empirical_labels=["Welch"],
    outdir=str(RESULTS_DIR),
    filename="psd_matrix.png",
    diag_yscale="log",
    offdiag_yscale="log",
    xscale="log",
    show_csd_magnitude=True,
    show_coherence=False,
    overlay_vi=True,
    freq_range=(FMIN, FMAX),
)
