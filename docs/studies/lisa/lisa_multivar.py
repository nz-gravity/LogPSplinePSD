import os
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Print number of JAX devices
import jax
import matplotlib.pyplot as plt
import numpy as np

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.example_datasets.lisa_data import LISAData
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix

logger.info(f"JAX devices: {jax.devices()}")

set_level("DEBUG")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FN = RESULTS_DIR / "inference_data.nc"

RUN_VI_ONLY = True
REUSE_EXISTING = False  # set True to skip sampling when results already exist

# Hyperparameters and spline configuration for this study
ALPHA_DELTA = 1.0
BETA_DELTA = 1.0
N_KNOTS = 50
TARGET_ACCEPT = 0.95
MAX_TREE_DEPTH = 12

lisa_data = LISAData.load()
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

n = y_full.shape[0]
n_blocks = desired_blocks
n_inside_block = n // n_blocks
logger.info(
    f"Using n_blocks={n_blocks} x {n_inside_block} (n_time={n})",
)


FMIN, FMAX = 10**-4, 10**-1

coarse_cfg = CoarseGrainConfig(
    enabled=True,
    f_transition=5 * 10**-4,
    n_log_bins=200,
    f_min=FMIN,
    f_max=FMAX,
)

raw_series = MultivariateTimeseries(y=y_full, t=t_full)

dt = t_full[1] - t_full[0]
fs = 1.0 / dt
fmin_full = 1.0 / (len(t_full) * dt)

idata = None

if RESULT_FN.exists() and REUSE_EXISTING:
    logger.info(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(str(RESULT_FN))

else:
    logger.info(f"No existing {RESULT_FN} found, running inference...")
    idata = run_mcmc(
        data=raw_series,
        sampler="multivar_blocked_nuts",
        n_samples=300,
        n_warmup=300,
        num_chains=4,
        n_knots=N_KNOTS,
        degree=2,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir=str(RESULTS_DIR),
        verbose=True,
        coarse_grain_config=coarse_cfg,
        n_time_blocks=n_blocks,
        fmin=FMIN,
        fmax=FMAX,
        alpha_delta=ALPHA_DELTA,
        beta_delta=BETA_DELTA,
        only_vi=RUN_VI_ONLY,
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


def _interp_matrix(
    freq_src: np.ndarray, mat: np.ndarray, freq_tgt: np.ndarray
):
    if freq_src.shape == freq_tgt.shape and np.allclose(freq_src, freq_tgt):
        return np.asarray(mat)
    flat = mat.reshape(mat.shape[0], -1)
    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_src, flat[:, i].real)
            for i in range(flat.shape[1])
        ]
    ).T
    imag_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_src, flat[:, i].imag)
            for i in range(flat.shape[1])
        ]
    ).T
    return (real_interp + 1j * imag_interp).reshape(
        (freq_tgt.size,) + mat.shape[1:]
    )


true_psd_physical = _interp_matrix(
    np.asarray(lisa_data.freq), np.asarray(lisa_data.true_matrix), freq_plot
)

# Traditional Welch-style empirical PSD on the original (unstandardised) data
empirical_welch = EmpiricalPSD.from_timeseries_data(
    data=y_full,
    fs=fs,
    nperseg=4096,
    noverlap=0,
    window="hann",
)


def _compute_coherence(psd: np.ndarray) -> np.ndarray:
    n_freq, n_chan, _ = psd.shape
    coh = np.zeros((n_freq, n_chan, n_chan))
    for i in range(n_chan):
        coh[:, i, i] = 1.0
        for j in range(i + 1, n_chan):
            denom = np.abs(psd[:, i, i]) * np.abs(psd[:, j, j])
            coh[:, i, j] = np.abs(psd[:, i, j]) ** 2 / denom
            coh[:, j, i] = coh[:, i, j]
    return coh


def plot_coherence_matrix(
    freq: np.ndarray,
    coh_true: np.ndarray,
    coh_emp: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    labels = ["X", "Y", "Z"]
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                ax.text(
                    0.5,
                    0.5,
                    labels[i],
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            ax.semilogx(freq, coh_true[:, i, j], label="True")
            ax.semilogx(freq, coh_emp[:, i, j], label="Welch", alpha=0.7)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{labels[i]}–{labels[j]}")
            if i == 2 and j == 0:
                ax.set_xlabel("Frequency [Hz]")
            if i == 2 and j == 1:
                ax.set_xlabel("Frequency [Hz]")
            if i == 1 and j == 0:
                ax.set_ylabel("Coherence")
            if i == 1 and j == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


plot_psd_matrix(
    idata=idata,
    freq=freq_plot,
    empirical_psd=None,  # will be extracted from idata.observed_data
    extra_empirical_psd=[empirical_welch],
    extra_empirical_labels=["Welch"],
    outdir=str(RESULTS_DIR),
    filename="psd_matrix.png",
    diag_yscale="log",
    offdiag_yscale="log",
    xscale="log",
    show_csd_magnitude=False,
    show_coherence=True,
    overlay_vi=True,
    freq_range=(FMIN, FMAX),
    true_psd=true_psd_physical,
)
