import os
import shutil
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Print number of JAX devices
import jax
import numpy as np

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.diagnostics import psd_compare
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_graining_univar,
    compute_binning_structure,
)

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


RUN_TAG = _env_str("LISA_RUN_TAG", "")
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
VI_GUIDE = _env_str("LISA_VI_GUIDE", "diag")
VI_STEPS = _env_int("LISA_VI_STEPS", 200_000)
VI_LR = _env_float("LISA_VI_LR", 1e-4)
VI_POSTERIOR_DRAWS = _env_int("LISA_VI_POSTERIOR_DRAWS", 1024)
MAX_TIME_BLOCKS = 12
N_TIME_BLOCKS_OVERRIDE: int | None = None
MAX_DAYS = _env_float("LISA_MAX_DAYS", 0.0)
MAX_MONTHS = _env_float("LISA_MAX_MONTHS", 0.0)
WELCH_NPERSEG = _env_int("LISA_WELCH_NPERSEG", 0)
WELCH_OVERLAP_FRAC = _env_float("LISA_WELCH_OVERLAP_FRAC", 0.5)
WELCH_WINDOW = _env_str("LISA_WELCH_WINDOW", "hann")
WELCH_BLOCK_AVG = _env_flag("LISA_WELCH_BLOCK_AVG", True)

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
        if "Lb" in synth.files:
            Lb = int(synth["Lb"])
        else:
            Lb = None
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

n = y_full.shape[0]
n_duration = t_full[-1] - t_full[0]
n_duration_days = n_duration / 86_400.0

# Choose block structure. Prefer NPZ metadata when available to avoid crossing
# chunk boundaries in synthetic generators.
if N_TIME_BLOCKS_OVERRIDE is not None:
    Nb = int(N_TIME_BLOCKS_OVERRIDE)
elif Lb is not None:
    Nb = max(1, int(n // Lb))

Lb = n // Nb
block_seconds = Lb * dt
n_used = Nb * Lb
if n_used != n:
    n_trim = n - n_used
    logger.info(
        f"Trimming {n_trim} samples to fit {Nb} blocks of {Lb} samples ({block_seconds:.0f} s each).",
    )
    t_full = t_full[:n_used]
    y_full = y_full[:n_used]

n = y_full.shape[0]
logger.info(
    f"Using Nb={Nb} x {Lb} (n={n}, block_seconds={block_seconds:.0f}).",
)
logger.info(
    f"Total duration: {n_duration_days:.2f} days ({n} samples).",
)
logger.info(
    f"Per-block duration: {block_seconds / 86_400.0:.2f} days ({Lb} samples).",
)


FMIN, FMAX = 10**-4, 10**-1

coarse_cfg = CoarseGrainConfig(
    enabled=True,
    Nc=512,
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
        Nb=Nb,
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

freq_plot = np.asarray(idata["posterior_psd"]["freq"].values)


true_psd_physical = interp_matrix(
    np.asarray(true_psd_source[0]), np.asarray(true_psd_source[1]), freq_plot
)

# Pick Welch settings that can actually resolve the requested low-frequency
# range. With dt=1 s, nperseg=4096 implies df≈2.44e-4 Hz, so anything below that
# is missing/biased. By default we target df≈FMIN via nperseg≈fs/FMIN.
if WELCH_NPERSEG > 0:
    welch_nperseg = int(WELCH_NPERSEG)
else:
    welch_nperseg = int(round(fs / FMIN))
    welch_nperseg = max(256, welch_nperseg)
welch_nperseg = min(welch_nperseg, y_full.shape[0])
if not (0.0 <= WELCH_OVERLAP_FRAC < 1.0):
    raise ValueError("LISA_WELCH_OVERLAP_FRAC must be in [0, 1).")
welch_noverlap = int(round(WELCH_OVERLAP_FRAC * welch_nperseg))
welch_noverlap = min(welch_noverlap, welch_nperseg - 1)
welch_df = fs / welch_nperseg
logger.info(
    "Welch settings: "
    f"nperseg={welch_nperseg} ({welch_nperseg * dt:.0f} s), "
    f"noverlap={welch_noverlap} ({welch_noverlap * dt:.0f} s), "
    f"window={WELCH_WINDOW!r}, df={welch_df:.3g} Hz."
)


def _drop_dc(emp: EmpiricalPSD) -> EmpiricalPSD:
    if emp.freq.size > 0 and np.isclose(emp.freq[0], 0.0):
        return EmpiricalPSD(
            freq=emp.freq[1:],
            psd=emp.psd[1:],
            coherence=emp.coherence[1:],
            channels=emp.channels,
        )
    return emp


def _restrict_freq_range(
    emp: EmpiricalPSD, *, fmin: float, fmax: float
) -> EmpiricalPSD:
    freq = np.asarray(emp.freq, dtype=float)
    mask = (freq >= float(fmin)) & (freq <= float(fmax))
    if not np.any(mask):
        raise ValueError("Welch frequency mask removed all bins.")
    return EmpiricalPSD(
        freq=freq[mask],
        psd=emp.psd[mask],
        coherence=emp.coherence[mask],
        channels=emp.channels,
    )


def _blocked_welch(
    data: np.ndarray,
    *,
    fs: float,
    Lb: int,
    nperseg: int,
    noverlap: int,
    window: str,
    detrend: str | bool,
) -> EmpiricalPSD:
    n, p = data.shape
    if Lb <= 1:
        raise ValueError("Lb must be > 1 for blocked Welch.")
    Nb = n // Lb
    if Nb < 1:
        raise ValueError("Not enough samples for even one Welch block.")
    n_used = Nb * Lb
    if n_used != n:
        data = data[:n_used]

    psd_sum = None
    freq_ref = None
    for idx in range(Nb):
        seg = data[idx * Lb : (idx + 1) * Lb]
        seg_nperseg = min(nperseg, Lb)
        seg_noverlap = min(noverlap, seg_nperseg - 1)
        emp = EmpiricalPSD.from_timeseries_data(
            data=seg,
            fs=fs,
            nperseg=seg_nperseg,
            noverlap=seg_noverlap,
            window=window,
            detrend=detrend,
        )
        if freq_ref is None:
            freq_ref = emp.freq
            psd_sum = np.zeros_like(emp.psd)
        if emp.freq.shape != freq_ref.shape or not np.allclose(
            emp.freq, freq_ref
        ):
            raise ValueError("Blocked Welch produced inconsistent freq grids.")
        psd_sum += emp.psd

    psd_avg = psd_sum / float(Nb)
    coh = np.abs(psd_avg) ** 2
    # coherence_ij = |Sij|^2 / (|Sii| |Sjj|)
    diag = np.abs(np.diagonal(psd_avg, axis1=1, axis2=2))
    denom = diag[:, :, None] * diag[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(denom > 0, coh.real / denom, np.nan)
    for ch in range(p):
        coherence[:, ch, ch] = 1.0

    return EmpiricalPSD(freq=freq_ref, psd=psd_avg, coherence=coherence)


# Traditional Welch-style empirical PSD on the original (unstandardised) data.
# For lisatools synthetic data, the generator can introduce small discontinuities
# at chunk boundaries; computing Welch within each block and averaging avoids
# leakage from crossing those boundaries.
if WELCH_BLOCK_AVG and Lb is not None:
    logger.info(
        f"Welch block-averaging enabled: {Nb} block(s) of {Lb} samples."
    )
    empirical_welch = _blocked_welch(
        y_full,
        fs=fs,
        Lb=Lb,
        nperseg=welch_nperseg,
        noverlap=welch_noverlap,
        window=WELCH_WINDOW,
        detrend=False,
    )
else:
    empirical_welch = EmpiricalPSD.from_timeseries_data(
        data=y_full,
        fs=fs,
        nperseg=welch_nperseg,
        noverlap=welch_noverlap,
        window=WELCH_WINDOW,
        detrend=False,
    )
empirical_welch = _drop_dc(empirical_welch)
empirical_welch = _restrict_freq_range(empirical_welch, fmin=FMIN, fmax=FMAX)


plot_psd_matrix(
    idata=idata,
    freq=freq_plot,
    empirical_psd=None,  # will be extracted from idata.observed_data
    extra_empirical_psd=[empirical_welch],
    extra_empirical_labels=[
        "Welch (block-avg)" if WELCH_BLOCK_AVG else "Welch"
    ],
    extra_empirical_styles=[
        dict(color="0.5", lw=1.3, alpha=0.9, ls="-", zorder=-4),
    ],
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
    psd_scale=None,
    psd_unit_label=None,
)

true_diag = np.diagonal(true_psd_physical.real, axis1=1, axis2=2)
true_diag_min = np.min(true_diag, axis=1)
dip_threshold = np.percentile(true_diag_min, METRICS_MIN_PCT)
metrics_mask = (
    (freq_plot >= FMIN) & (freq_plot <= FMAX) & (true_diag_min > dip_threshold)
)
metrics = []


def _summarize_psd_accuracy(psd_ds, *, label: str):
    if psd_ds is None:
        return None
    try:
        idx = np.where(metrics_mask)[0]
        psd_ds_use = psd_ds.isel(freq=idx) if idx.size else psd_ds
        true_use = true_psd_physical[idx] if idx.size else true_psd_physical
        out = psd_compare._handle_multivariate(psd_ds_use, true_use)
        out["label"] = label
        return out
    except Exception as exc:
        logger.warning(f"Could not summarize PSD accuracy ({label}): {exc}")
        return None


metrics.append(
    _summarize_psd_accuracy(
        getattr(idata, "posterior_psd", None), label="NUTS"
    )
)
metrics.append(
    _summarize_psd_accuracy(
        getattr(idata, "vi_posterior_psd", None), label="VI"
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
            if "riae_matrix" in entry:
                handle.write(f"  RIAE (matrix): {entry['riae_matrix']:.4g}\n")
            if "riae_diag_mean" in entry:
                handle.write(
                    f"  RIAE (diag mean): {entry['riae_diag_mean']:.4g}\n"
                )
            if "riae_diag_max" in entry:
                handle.write(
                    f"  RIAE (diag max): {entry['riae_diag_max']:.4g}\n"
                )
            if "riae_offdiag" in entry:
                handle.write(
                    f"  RIAE (offdiag): {entry['riae_offdiag']:.4g}\n"
                )
            if "coherence_riae" in entry:
                handle.write(
                    f"  Coherence RIAE: {entry['coherence_riae']:.4g}\n"
                )
            if "coverage" in entry:
                handle.write(f"  Coverage (90% CI): {entry['coverage']:.3f}\n")
            handle.write("\n")
    logger.info(f"Saved PSD accuracy summary to {summary_path}")
