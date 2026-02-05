import os
import shutil
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Print number of JAX devices
import jax
import numpy as np

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_graining_univar,
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
USE_NOISE_FLOOR = _env_flag("LISA_USE_NOISE_FLOOR", True)
NOISE_FLOOR_MODE = _env_str(
    "LISA_NOISE_FLOOR_MODE", "theory_scaled"
)  # "constant", "theory_scaled", "array"
NOISE_FLOOR_SCALE = _env_float("LISA_NOISE_FLOOR_SCALE", 1e-4)
NOISE_FLOOR_CONSTANT = _env_float("LISA_NOISE_FLOOR_CONSTANT", 1e-6)
NOISE_FLOOR_BLOCKS: tuple[int, ...] | str = _env_blocks(
    "LISA_NOISE_FLOOR_BLOCKS", (2,)
)  # block 3 (0-based)
NOISE_FLOOR_ARRAY: np.ndarray | None = None

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
elif block_len_samples is not None:
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

n = y_full.shape[0]
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
    Nc=512,
    f_min=FMIN,
    f_max=FMAX,
)

noise_floor_kwargs: dict = {}
if USE_NOISE_FLOOR:
    noise_floor_psd = None
    if NOISE_FLOOR_MODE == "theory_scaled":
        blocks = NOISE_FLOOR_BLOCKS
        if isinstance(blocks, str):
            raise ValueError(
                "NOISE_FLOOR_BLOCKS must be a sequence when using theory_scaled."
            )
        if blocks is None or len(blocks) != 1:
            raise ValueError(
                "theory_scaled noise floor requires exactly one block index."
            )
        block_idx = int(blocks[0])
        if block_idx < 0 or block_idx >= y_full.shape[1]:
            raise ValueError(
                f"Block index {block_idx} is out of range for {y_full.shape[1]} channels."
            )
        freq_true = np.asarray(true_psd_source[0], dtype=float)
        true_matrix = np.asarray(true_psd_source[1])

        freq_fine = np.fft.rfftfreq(block_len_samples, 1 / fs)[1:]
        freq_mask = (freq_fine >= FMIN) & (freq_fine <= FMAX)
        freq_fine = freq_fine[freq_mask]
        if freq_fine.size == 0:
            raise ValueError("No frequencies remain after fmin/fmax.")

        true_matrix_interp = interp_matrix(freq_true, true_matrix, freq_fine)
        true_diag_fine = np.real(
            np.diagonal(true_matrix_interp, axis1=1, axis2=2)
        )

        if coarse_cfg.enabled:
            spec = compute_binning_structure(
                freq_fine,
                Nc=coarse_cfg.Nc,
                f_min=coarse_cfg.f_min,
                f_max=coarse_cfg.f_max,
            )
            diag_bins = []
            for ch in range(true_diag_fine.shape[1]):
                diag_coarse, _ = apply_coarse_graining_univar(
                    true_diag_fine[:, ch], spec, freq_fine
                )
                diag_bins.append(diag_coarse)
            true_diag = np.stack(diag_bins, axis=1)
        else:
            true_diag = true_diag_fine

        stds = np.std(y_full, axis=0)
        scale = 1.0 / (stds**2)
        true_diag_std = true_diag * scale[None, :]
        noise_floor_psd = true_diag_std[:, block_idx]
    elif NOISE_FLOOR_MODE == "array":
        if NOISE_FLOOR_ARRAY is None:
            raise ValueError(
                "NOISE_FLOOR_ARRAY must be set when NOISE_FLOOR_MODE='array'."
            )
        noise_floor_psd = np.asarray(NOISE_FLOOR_ARRAY, dtype=float)

    noise_floor_kwargs = {
        "use_noise_floor": True,
        "noise_floor_mode": NOISE_FLOOR_MODE,
        "noise_floor_constant": NOISE_FLOOR_CONSTANT,
        "noise_floor_scale": NOISE_FLOOR_SCALE,
        "noise_floor_array": NOISE_FLOOR_ARRAY,
        "theory_psd": noise_floor_psd,
        "noise_floor_blocks": NOISE_FLOOR_BLOCKS,
    }
    if noise_floor_psd is not None:
        floor_min = float(np.min(noise_floor_psd))
        floor_med = float(np.median(noise_floor_psd))
        floor_max = float(np.max(noise_floor_psd))
        logger.info(
            f"Noise floor PSD (std units) for block {int(NOISE_FLOOR_BLOCKS[0])}: "
            f"min={floor_min:.3e}, median={floor_med:.3e}, max={floor_max:.3e}."
        )
        effective_floor = noise_floor_psd * float(NOISE_FLOOR_SCALE)
        eff_min = float(np.min(effective_floor))
        eff_med = float(np.median(effective_floor))
        eff_max = float(np.max(effective_floor))
        logger.info(
            f"Effective noise floor (std units): min={eff_min:.3e}, "
            f"median={eff_med:.3e}, max={eff_max:.3e}."
        )
    elif NOISE_FLOOR_MODE == "constant":
        logger.info(
            f"Effective noise floor (std units): constant={NOISE_FLOOR_CONSTANT:.3e}."
        )
    elif NOISE_FLOOR_MODE == "array" and NOISE_FLOOR_ARRAY is not None:
        eff_min = float(np.min(NOISE_FLOOR_ARRAY))
        eff_med = float(np.median(NOISE_FLOOR_ARRAY))
        eff_max = float(np.max(NOISE_FLOOR_ARRAY))
        logger.info(
            f"Effective noise floor (std units): min={eff_min:.3e}, "
            f"median={eff_med:.3e}, max={eff_max:.3e}."
        )
    logger.info(
        f"Noise floor enabled: mode={NOISE_FLOOR_MODE}, blocks={NOISE_FLOOR_BLOCKS}, "
        f"scale={NOISE_FLOOR_SCALE:g}, constant={NOISE_FLOOR_CONSTANT:g}."
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
        **noise_floor_kwargs,
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
    block_len: int,
    nperseg: int,
    noverlap: int,
    window: str,
    detrend: str | bool,
) -> EmpiricalPSD:
    n_time, n_channels = data.shape
    if block_len <= 1:
        raise ValueError("block_len must be > 1 for blocked Welch.")
    n_blocks = n_time // block_len
    if n_blocks < 1:
        raise ValueError("Not enough samples for even one Welch block.")
    n_used = n_blocks * block_len
    if n_used != n_time:
        data = data[:n_used]

    psd_sum = None
    freq_ref = None
    for idx in range(n_blocks):
        seg = data[idx * block_len : (idx + 1) * block_len]
        seg_nperseg = min(nperseg, block_len)
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

    psd_avg = psd_sum / float(n_blocks)
    coh = np.abs(psd_avg) ** 2
    # coherence_ij = |Sij|^2 / (|Sii| |Sjj|)
    diag = np.abs(np.diagonal(psd_avg, axis1=1, axis2=2))
    denom = diag[:, :, None] * diag[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(denom > 0, coh.real / denom, np.nan)
    for ch in range(n_channels):
        coherence[:, ch, ch] = 1.0

    return EmpiricalPSD(freq=freq_ref, psd=psd_avg, coherence=coherence)


# Traditional Welch-style empirical PSD on the original (unstandardised) data.
# For lisatools synthetic data, the generator can introduce small discontinuities
# at chunk boundaries; computing Welch within each block and averaging avoids
# leakage from crossing those boundaries.
if WELCH_BLOCK_AVG and block_len_samples is not None:
    logger.info(
        f"Welch block-averaging enabled: {n_blocks} block(s) of {block_len_samples} samples."
    )
    empirical_welch = _blocked_welch(
        y_full,
        fs=fs,
        block_len=block_len_samples,
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
