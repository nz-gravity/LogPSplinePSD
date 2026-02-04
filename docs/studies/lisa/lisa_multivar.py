import os
import shutil
import sys
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Ensure local package imports work even when running outside the repo .venv.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Print number off JAX devices
import jax
import numpy as np

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.diagnostics.psd_metrics import summarize_multivar_psd_metrics
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix
from log_psplines.spectrum_utils import interp_matrix, resolve_psd_plot_units
from log_psplines.utils.blocking import infer_time_blocks

logger.info(f"JAX devices: {jax.devices()}")

set_level("DEBUG")

HERE = Path(__file__).resolve().parent
BASE_RESULTS_DIR = HERE / "results" / "lisa"


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    return float(val)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    return int(val)


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val)


def _env_blocks(name: str, default: tuple[int, ...] | str):
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val == "all":
        return "all"
    if val == "":
        return default
    parts = [p for p in val.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


PRESET = _env_str("LISA_PRESET", "auto").strip().lower()
RUN_TAG = _env_str("LISA_RUN_TAG", "")

# Simple presets to reduce "forgot env vars" failures.
# If preset is set and RUN_TAG is empty, we use the preset name as the run tag.
if PRESET not in {"", "auto"} and RUN_TAG == "":
    RUN_TAG = PRESET

RESULTS_DIR = (
    BASE_RESULTS_DIR if RUN_TAG == "" else HERE / "results" / f"lisa_{RUN_TAG}"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FN = RESULTS_DIR / "inference_data.nc"

RUN_VI_ONLY = _env_flag("LISA_RUN_VI_ONLY", False)
INIT_FROM_VI = _env_flag("LISA_INIT_FROM_VI", True)
REUSE_EXISTING = _env_flag(
    "LISA_REUSE_EXISTING", False
)  # set True to skip sampling when results already exist
USE_LISATOOLS_SYNTH = _env_flag("LISA_USE_LISATOOLS_SYNTH", True)
LISATOOLS_SYNTH_NPZ = RESULTS_DIR / "lisatools_synth_data.npz"

# Hyperparameters and spline configuration for this study
ALPHA_DELTA = _env_float("LISA_ALPHA_DELTA", 3.0)
BETA_DELTA = _env_float("LISA_BETA_DELTA", 3.0)
N_KNOTS = _env_int("LISA_N_KNOTS", 30)
TARGET_ACCEPT = _env_float("LISA_TARGET_ACCEPT", 0.7)
MAX_TREE_DEPTH = _env_int("LISA_MAX_TREE_DEPTH", 10)
TARGET_ACCEPT_BY_CHANNEL: list[float] | None = [0.7, 0.6, 0.75]
# Avoid raising max_tree_depth unless you have to: if a channel already hits the
# max steps, increasing max_tree_depth can dramatically increase walltime.
MAX_TREE_DEPTH_BY_CHANNEL: list[int] | None = [10, 10, 10]
DENSE_MASS = True
VI_GUIDE = _env_str("LISA_VI_GUIDE", "lowrank:16")
VI_STEPS = _env_int("LISA_VI_STEPS", 200_000)
VI_LR = _env_float("LISA_VI_LR", 1e-4)
VI_POSTERIOR_DRAWS = _env_int("LISA_VI_POSTERIOR_DRAWS", 1024)
MAX_TIME_BLOCKS = 12
N_TIME_BLOCKS_OVERRIDE: int | None = None
MAX_DAYS = _env_float("LISA_MAX_DAYS", 0.0)
MAX_MONTHS = _env_float("LISA_MAX_MONTHS", 0.0)
N_FREQS_PER_BIN = _env_int("LISA_N_FREQS_PER_BIN", 0)
CHECK_COARSE_EQUIV = _env_flag("LISA_CHECK_COARSE_EQUIV", False)
CHECK_COARSE_EQUIV_BINS = _env_blocks("LISA_CHECK_COARSE_EQUIV_BINS", (0,))

if ".venv" in str(HERE.parent / ".venv") and ".venv" not in sys.executable:
    # Running with a non-venv python is a common source of "why didn't my changes apply?"
    logger.warning(
        f"Python executable does not look like the repo .venv: {sys.executable}. "
        "Recommended: run with .venv/bin/python (and set PYTHONPATH=./src if needed)."
    )

C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LASER_FREQ = 2.81e14  # Hz
METRICS_MIN_PCT = 5.0
METRICS_LOG_EPS = 1e-60
PLOT_PSD_UNITS = "freq"  # "freq" -> Hz^2/Hz, "strain" -> 1/Hz


if USE_LISATOOLS_SYNTH:
    if not LISATOOLS_SYNTH_NPZ.exists():
        fallback_npz = BASE_RESULTS_DIR / "lisatools_synth_data.npz"
        if fallback_npz.exists():
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                LISATOOLS_SYNTH_NPZ.symlink_to(fallback_npz)
                logger.info(
                    f"Symlinked synth data from {fallback_npz} -> {LISATOOLS_SYNTH_NPZ}."
                )
            except OSError:
                shutil.copy2(fallback_npz, LISATOOLS_SYNTH_NPZ)
                logger.info(
                    f"Copied synth data from {fallback_npz} -> {LISATOOLS_SYNTH_NPZ}."
                )
        else:
            raise FileNotFoundError(
                f"{LISATOOLS_SYNTH_NPZ} not found. Run lisatools_synth_check.py first."
            )
    with np.load(LISATOOLS_SYNTH_NPZ, allow_pickle=False) as synth:
        t_full = synth["time"]
        y_full = synth["data"]
        true_matrix = synth["true_matrix"]
        use_freq_units = bool(
            synth["use_freq_units"]
            if "use_freq_units" in synth.files
            else False
        )
        if use_freq_units and "true_matrix_freq" not in synth.files:
            logger.warning(
                "Synthetic NPZ stores PSD in frequency units only; "
                "regenerate to include strain units for consistent overlays."
            )
        if "block_len_samples" in synth.files:
            block_len_samples = int(synth["block_len_samples"])
        else:
            block_len_samples = None
        base_psd_units = "strain"
        true_psd_source = (synth["freq_true"], true_matrix)
else:
    from log_psplines.example_datasets.lisa_data import LISAData

    lisa_data = LISAData.load()
    lisa_data.plot(f"{RESULTS_DIR}/lisa_raw.png")
    t_full = lisa_data.time
    y_full = lisa_data.data
    base_psd_units = "freq"
    true_psd_source = (lisa_data.freq, lisa_data.true_matrix)

dt = t_full[1] - t_full[0]
fs = 1.0 / dt
if MAX_DAYS > 0.0 or MAX_MONTHS > 0.0:
    month_days = 30.0
    max_days = MAX_DAYS
    if MAX_MONTHS > 0.0:
        max_days = max_days if max_days > 0.0 else MAX_MONTHS * month_days
        if MAX_DAYS > 0.0:
            max_days = min(max_days, MAX_DAYS)
    max_seconds = max_days * 86_400.0
    max_samples = int(max_seconds / dt)
    if max_samples < 2:
        raise ValueError(
            "LISA_MAX_DAYS/MONTHS too small to retain at least 2 samples."
        )
    if max_samples < len(t_full):
        t_full = t_full[:max_samples]
        y_full = y_full[:max_samples]
        logger.info(
            f"Trimmed to {max_days:.2f} days "
            f"({max_samples} samples) for faster runs."
        )

fmin_full = 1.0 / (len(t_full) * dt)

n_time = y_full.shape[0]
n_duration = t_full[-1] - t_full[0]
n_duration_days = n_duration / 86_400.0

# Choose block structure. Prefer NPZ metadata when available to avoid crossing
# chunk boundaries in synthetic generators.
if N_TIME_BLOCKS_OVERRIDE is not None:
    n_blocks = int(N_TIME_BLOCKS_OVERRIDE)
    block_len_samples = n_time // n_blocks
elif block_len_samples is not None:
    # Preserve the generator chunk length exactly and trim to a whole number of blocks.
    # This avoids stitching artifacts when the synthetic series was generated by
    # concatenating independent 7-day chunks.
    n_blocks = max(1, int(n_time // block_len_samples))
else:
    n_blocks = infer_time_blocks(n_time, max_blocks=MAX_TIME_BLOCKS)
    block_len_samples = n_time // n_blocks

block_seconds = block_len_samples * dt
n_used = n_blocks * block_len_samples
if n_used != n_time:
    n_trim = n_time - n_used
    logger.info(
        f"Trimming {n_trim} samples to fit {n_blocks} blocks of {block_len_samples} samples ({block_seconds:.0f} s each).",
    )
    t_full = t_full[:n_used]
    y_full = y_full[:n_used]

# Recompute duration stats after any trimming.
n_time = y_full.shape[0]
n_duration = t_full[-1] - t_full[0]
n_duration_days = n_duration / 86_400.0

n = n_time
logger.info(
    f"Using n_blocks={n_blocks} x {block_len_samples} (n_time={n}, block_seconds={block_seconds:.0f}).",
)
logger.info(
    f"Total duration: {n_duration_days:.2f} days ({n_time} samples).",
)
logger.info(
    f"Per-block duration: {block_seconds / 86_400.0:.2f} days ({block_len_samples} samples).",
)


FMIN, FMAX = 10**-4, 10**-1

coarse_cfg = CoarseGrainConfig(
    enabled=True,
    # Paper-style coarse graining: bin the full retained frequency band into
    # disjoint subsets J_h and use the midpoint Fourier frequency per bin.
    keep_low=False,
    n_log_bins=4096,
    binning="linear",
    representative="middle",
    n_freqs_per_bin=(None if N_FREQS_PER_BIN <= 0 else int(N_FREQS_PER_BIN)),
    f_min=FMIN,
    f_max=FMAX,
)

raw_series = MultivariateTimeseries(y=y_full, t=t_full)

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
        n_samples=4000,
        n_warmup=4000,
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
        init_from_vi=INIT_FROM_VI,
        vi_steps=VI_STEPS,
        vi_lr=VI_LR,
        vi_guide=VI_GUIDE,
        vi_posterior_draws=VI_POSTERIOR_DRAWS,
        vi_progress_bar=True,
        target_accept_prob=TARGET_ACCEPT,
        target_accept_prob_by_channel=TARGET_ACCEPT_BY_CHANNEL,
        max_tree_depth=MAX_TREE_DEPTH,
        max_tree_depth_by_channel=MAX_TREE_DEPTH_BY_CHANNEL,
        dense_mass=DENSE_MASS,
        true_psd=true_psd_source,
    )
    idata.to_netcdf(str(RESULT_FN))

if idata is None:
    raise RuntimeError("Inference data was not produced or loaded.")


logger.info(f"Saved results to {RESULT_FN}")

logger.info(idata)


def _posterior_median(idata, name: str) -> np.ndarray | None:
    if not hasattr(idata, "sample_stats"):
        return None
    if name not in idata.sample_stats:
        return None
    array = np.asarray(idata.sample_stats[name].values)
    if array.ndim < 2:
        return array
    return np.median(array, axis=(0, 1))


def _theta_slice_for_block(block_idx: int) -> slice:
    theta_start = block_idx * (block_idx - 1) // 2
    theta_count = block_idx
    return slice(theta_start, theta_start + theta_count)


def _compute_block_residual_power(
    u_re: np.ndarray,
    u_im: np.ndarray,
    theta_re: np.ndarray,
    theta_im: np.ndarray,
    block_idx: int,
) -> np.ndarray:
    u_re_channel = u_re[:, block_idx, :]
    u_im_channel = u_im[:, block_idx, :]
    if block_idx > 0:
        u_re_prev = u_re[:, :block_idx, :]
        u_im_prev = u_im[:, :block_idx, :]
        contrib_re = np.einsum("fl,flr->fr", theta_re, u_re_prev) - np.einsum(
            "fl,flr->fr", theta_im, u_im_prev
        )
        contrib_im = np.einsum("fl,flr->fr", theta_re, u_im_prev) + np.einsum(
            "fl,flr->fr", theta_im, u_re_prev
        )
        u_re_resid = u_re_channel - contrib_re
        u_im_resid = u_im_channel - contrib_im
    else:
        u_re_resid = u_re_channel
        u_im_resid = u_im_channel

    return np.sum(u_re_resid**2 + u_im_resid**2, axis=1)


freq_plot = np.asarray(idata["posterior_psd"]["freq"].values)

if CHECK_COARSE_EQUIV:
    from log_psplines.diagnostics.coarse_grain_checks import (
        coarse_bin_likelihood_equivalence_check,
    )

    log_delta_sq = _posterior_median(idata, "log_delta_sq")
    theta_re = _posterior_median(idata, "theta_re")
    theta_im = _posterior_median(idata, "theta_im")
    if log_delta_sq is None:
        logger.warning(
            "Skipping coarse equivalence check: log_delta_sq not found."
        )
    else:
        standardized_ts = raw_series.standardise_for_psd()
        fft_fine = standardized_ts.to_wishart_stats(
            n_blocks=n_blocks, fmin=FMIN, fmax=FMAX
        )
        spec = compute_binning_structure(
            fft_fine.freq,
            f_transition=coarse_cfg.f_transition,
            n_log_bins=coarse_cfg.n_log_bins,
            binning=coarse_cfg.binning,
            representative=coarse_cfg.representative,
            keep_low=coarse_cfg.keep_low,
            n_freqs_per_bin=coarse_cfg.n_freqs_per_bin,
            f_min=coarse_cfg.f_min,
            f_max=coarse_cfg.f_max,
        )

        bins = CHECK_COARSE_EQUIV_BINS
        if isinstance(bins, str):
            raise ValueError(
                "LISA_CHECK_COARSE_EQUIV_BINS must be a sequence (e.g. '0,1,2'), not 'all'."
            )

        include_theta = theta_re is not None and theta_im is not None
        for b in bins:
            b = int(b)
            res = coarse_bin_likelihood_equivalence_check(
                fft_fine=fft_fine,
                spec=spec,
                bin_index=b,
                nu=int(fft_fine.nu),
                log_delta_sq_bin=np.asarray(log_delta_sq)[b],
                theta_re_bin=(
                    None if theta_re is None else np.asarray(theta_re)[b]
                ),
                theta_im_bin=(
                    None if theta_im is None else np.asarray(theta_im)[b]
                ),
                include_theta=bool(include_theta),
            )
            logger.info(
                f"Coarse equivalence bin={res.bin_index}, N_h={res.n_members}: "
                f"ll_fine={res.ll_fine:.6e}, ll_coarse={res.ll_coarse:.6e}, "
                f"delta={res.delta:.3e}, rel={res.rel_delta:.3e}"
            )


true_psd_physical = interp_matrix(
    np.asarray(true_psd_source[0]), np.asarray(true_psd_source[1]), freq_plot
)

# Traditional Welch-style empirical PSD on the original (unstandardised) data
empirical_welch = EmpiricalPSD.from_timeseries_data(
    data=y_full,
    fs=fs,
    nperseg=4096,
    noverlap=0,
    window="hann",
)

psd_scale_plot, psd_unit_label_plot = resolve_psd_plot_units(
    base_psd_units,
    PLOT_PSD_UNITS,
    laser_freq=LASER_FREQ,
    arm_length=L_ARM,
    c_light=C_LIGHT,
)


plot_psd_matrix(
    idata=idata,
    freq=freq_plot,
    empirical_psd=None,  # will be extracted from idata.observed_data
    extra_empirical_psd=[empirical_welch],
    extra_empirical_labels=["Welch"],
    outdir=str(RESULTS_DIR),
    filename="psd_matrix.png",
    diag_yscale="log",
    offdiag_yscale="linear",
    xscale="log",
    show_csd_magnitude=False,
    show_coherence=True,
    overlay_vi=True,
    freq_range=(FMIN, FMAX),
    true_psd=true_psd_physical,
    psd_scale=psd_scale_plot,
    psd_unit_label=psd_unit_label_plot,
)

true_diag = np.diagonal(true_psd_physical.real, axis1=1, axis2=2)
true_diag_min = np.min(true_diag, axis=1)
dip_threshold = np.percentile(true_diag_min, METRICS_MIN_PCT)
metrics_mask = (
    (freq_plot >= FMIN) & (freq_plot <= FMAX) & (true_diag_min > dip_threshold)
)
metrics = []
metrics.append(
    summarize_multivar_psd_metrics(
        getattr(idata, "posterior_psd", None),
        label="NUTS",
        true_psd=true_psd_physical,
        freqs=freq_plot,
        freq_mask=metrics_mask,
        log_eps=METRICS_LOG_EPS,
    )
)
metrics.append(
    summarize_multivar_psd_metrics(
        getattr(idata, "vi_posterior_psd", None),
        label="VI",
        true_psd=true_psd_physical,
        freqs=freq_plot,
        freq_mask=metrics_mask,
        log_eps=METRICS_LOG_EPS,
    )
)

metrics = [entry for entry in metrics if entry is not None]
if metrics:
    summary_path = RESULTS_DIR / "psd_accuracy_summary.txt"
    with summary_path.open("w") as handle:
        handle.write(
            f"Dip mask threshold (min diag, p{METRICS_MIN_PCT:.1f}): {dip_threshold:.4g}\n"
        )
        handle.write(
            f"Mask retained {np.count_nonzero(metrics_mask)}/{metrics_mask.size} bins.\n\n"
        )
        for entry in metrics:
            handle.write(f"{entry['label']} metrics\n")
            handle.write(f"  RIAE (matrix): {entry['riae_matrix']:.4g}\n")
            handle.write(
                f"  Relative L2 (matrix): {entry['relative_l2_matrix']:.4g}\n"
            )
            handle.write(
                f"  Log10 RIAE (diag mean): {entry['log_riae_diag']:.4g}\n"
            )
            handle.write(
                f"  Log10 Relative L2 (diag mean): {entry['log_l2_diag']:.4g}\n"
            )
            handle.write(f"  Coverage (90% CI): {entry['coverage_90']:.3f}\n")
            handle.write(
                f"  CI width (median): {entry['ci_width_median']:.4g}\n"
            )
            handle.write(f"  CI width (mean): {entry['ci_width_mean']:.4g}\n")
            handle.write("\n")
    logger.info(f"Saved PSD accuracy summary to {summary_path}")
