import os
import shutil
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Print number of JAX devices
import jax
import numpy as np
from evidence_utils import (
    combine_diag_lnz,
    compare_full_vs_diag,
    extract_lnz,
    parse_channel_names,
    parse_hypothesis_mode,
    write_evidence_summary,
)

from log_psplines.datatypes import MultivariateTimeseries, Timeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.diagnostics import psd_compare
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

logger.info(f"JAX devices: {jax.devices()}")

set_level("DEBUG")

HERE = Path(__file__).resolve().parent
BASE_RESULTS_DIR = HERE / "results" / "lisa"

RUN_TAG = ""
RESULTS_DIR = (
    BASE_RESULTS_DIR if RUN_TAG == "" else HERE / "results" / f"lisa_{RUN_TAG}"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_RESULTS_DIR = RESULTS_DIR / "full"
DIAG_RESULTS_DIR = RESULTS_DIR / "diag"
COMPARE_TXT = RESULTS_DIR / "evidence_comparison.txt"
COMPARE_JSON = RESULTS_DIR / "evidence_comparison.json"
LISA_HYPOTHESIS_MODE = parse_hypothesis_mode("full")
LISA_RUN_COMPARE_SUMMARY = True
_raw_channel_names = "X,Y,Z"
LISA_CHANNEL_NAMES = parse_channel_names(_raw_channel_names)
if ",".join(LISA_CHANNEL_NAMES) != _raw_channel_names.replace(" ", ""):
    logger.warning(
        f"Invalid LISA_CHANNEL_NAMES='{_raw_channel_names}'. Using default channel labels X,Y,Z."
    )

RUN_VI_ONLY = False
INIT_FROM_VI = True
# Main multivariate run can be decoupled from VI init when diagnosing
# pathological adaptation; diagonal runs still use INIT_FROM_VI.
INIT_FROM_VI_FULL = False
REUSE_EXISTING = False  # set True to skip sampling when results already exist
USE_LISATOOLS_SYNTH = True
LISATOOLS_SYNTH_NPZ = RESULTS_DIR / "lisa_data.npz"

# Hyperparameters and spline configuration for this study
ALPHA_DELTA = 3.0
BETA_DELTA = 3.0
N_KNOTS = 10
TARGET_ACCEPT = 0.9
MAX_TREE_DEPTH = 10
TARGET_ACCEPT_BY_CHANNEL: list[float] | None = [0.9, 0.92, 0.95]
# Avoid raising max_tree_depth unless you have to: if a channel already hits the
# max steps, increasing max_tree_depth can dramatically increase walltime.
MAX_TREE_DEPTH_BY_CHANNEL: list[int] | None = [10, 11, 12]
DENSE_MASS = True
VI_GUIDE = "diag"
VI_STEPS = 100_000
VI_LR = 1e-4
VI_POSTERIOR_DRAWS = 1024
MAX_TIME_BLOCKS = 12
N_TIME_BLOCKS_OVERRIDE: int | None = None
BLOCK_DAYS = 1.0
MAX_DAYS = 14.0
MAX_MONTHS = 0.0
WELCH_NPERSEG = 0
WELCH_OVERLAP_FRAC = 0.5
WELCH_WINDOW = "hann"
WELCH_BLOCK_AVG = True
ENABLE_COARSE_GRAIN = False
COARSE_GRAIN_FACTOR = 0
TARGET_COARSE_BINS = 4096

C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LASER_FREQ = 2.81e14  # Hz
METRICS_MIN_PCT = 5.0
METRICS_LOG_EPS = 1e-60
PLOT_PSD_UNITS = "freq"  # "freq" -> Hz^2/Hz, "strain" -> 1/Hz

Nb_hint: int | None = None
Lb_hint: int | None = None

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
                f"{LISATOOLS_SYNTH_NPZ} not found. Run lisa_datagen.py first."
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
        if "Nb" in synth.files:
            Nb_hint = int(synth["Nb"])
        if "Lb" in synth.files:
            Lb_hint = int(synth["Lb"])
        elif "block_len_samples" in synth.files:
            # Backward-compatible read for older synth NPZ files.
            Lb_hint = int(synth["block_len_samples"])
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

# Choose block structure.
# Priority:
# 1) explicit Nb override
# 2) explicit block duration in days
# 3) NPZ metadata hints
# 4) MAX_TIME_BLOCKS fallback
Lb: int | None = None
if N_TIME_BLOCKS_OVERRIDE is not None:
    Nb = int(N_TIME_BLOCKS_OVERRIDE)
    if Nb < 1:
        raise ValueError("N_TIME_BLOCKS_OVERRIDE must be >= 1.")
    Nb = min(Nb, n)
elif BLOCK_DAYS > 0.0:
    Lb_target = int(round(float(BLOCK_DAYS) * 86_400.0 / dt))
    if Lb_target < 2:
        raise ValueError(
            "LISA_BLOCK_DAYS is too small for the current cadence."
        )
    Lb = min(Lb_target, n)
    Nb = max(1, int(n // Lb))
    logger.info(
        f"Using block duration target {BLOCK_DAYS:.3g} days "
        f"(Lb={Lb} samples before trim)."
    )
elif Lb_hint is not None and Lb_hint > 0:
    Lb = int(min(int(Lb_hint), n))
    Nb = max(1, int(n // Lb))
elif Nb_hint is not None and Nb_hint > 0:
    Nb = min(int(Nb_hint), n)
else:
    Nb = min(MAX_TIME_BLOCKS, n)

if Nb < 1:
    raise ValueError("Derived Nb must be >= 1.")

if Lb is None:
    Lb = n // Nb
if Lb < 1:
    raise ValueError(
        f"Derived block length Lb={Lb} is invalid for n={n}, Nb={Nb}."
    )
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


# Pilot stability band: reduce extreme high-frequency geometry while tuning.
FMIN, FMAX = 10**-4, 10**-2

analysis_freq = np.fft.rfftfreq(Lb, d=dt)[1:]
analysis_mask = (analysis_freq >= FMIN) & (analysis_freq <= FMAX)
Nl_analysis = int(np.count_nonzero(analysis_mask))
if Nl_analysis < 1:
    raise ValueError(
        f"No positive frequencies retained in [{FMIN}, {FMAX}] for Lb={Lb}."
    )

if ENABLE_COARSE_GRAIN:
    if COARSE_GRAIN_FACTOR > 0:
        nh_eff = int(COARSE_GRAIN_FACTOR)
        if Nl_analysis % nh_eff != 0:
            Nh_candidates = [
                k for k in range(1, Nl_analysis + 1) if Nl_analysis % k == 0
            ]
            nh_eff = min(Nh_candidates, key=lambda d: (abs(d - nh_eff), d))
            logger.info(
                f"Adjusting coarse factor: Nl={Nl_analysis} not divisible by "
                f"{COARSE_GRAIN_FACTOR}; using Nh={nh_eff}."
            )
        nc_eff = Nl_analysis // nh_eff
        # Guardrail: tiny Nc collapses the spectrum and makes diagnostics misleading.
        if nc_eff < 32:
            logger.warning(
                f"Requested coarse graining gives Nc={nc_eff} (<32) for Nl={Nl_analysis}; "
                "disabling coarse graining."
            )
            coarse_cfg = CoarseGrainConfig(enabled=False)
        else:
            logger.info(
                f"Coarse graining enabled with Nh={nh_eff} "
                f"(~{nh_eff}:1 compression, Nc={nc_eff})."
            )
            coarse_cfg = CoarseGrainConfig(enabled=True, Nh=nh_eff)
    else:
        target_Nc = int(TARGET_COARSE_BINS)
        if target_Nc <= 0:
            raise ValueError("LISA_TARGET_COARSE_BINS must be positive.")
        if Nl_analysis % target_Nc != 0:
            Nc_candidates = [
                k for k in range(1, Nl_analysis + 1) if Nl_analysis % k == 0
            ]
            target_Nc = max(
                c for c in Nc_candidates if c <= min(target_Nc, Nl_analysis)
            )
            logger.info(
                f"Adjusting coarse bins: retained Nl={Nl_analysis} is not divisible by "
                f"{TARGET_COARSE_BINS}; using Nc={target_Nc}."
            )
        if target_Nc < 32:
            logger.warning(
                f"Requested coarse graining would use Nc={target_Nc} (<32) for Nl={Nl_analysis}; "
                "disabling coarse graining."
            )
            coarse_cfg = CoarseGrainConfig(enabled=False)
        else:
            logger.info(f"Coarse graining enabled with Nc={target_Nc}.")
            coarse_cfg = CoarseGrainConfig(enabled=True, Nc=target_Nc)
else:
    logger.info("Coarse graining disabled (full frequency resolution).")
    coarse_cfg = CoarseGrainConfig(enabled=False)

raw_series = MultivariateTimeseries(y=y_full, t=t_full)


def _run_full_hypothesis(
    *,
    full_outdir: Path,
):
    full_outdir.mkdir(parents=True, exist_ok=True)
    result_path = full_outdir / "inference_data.nc"

    if result_path.exists() and REUSE_EXISTING:
        logger.info(
            f"Found existing full-hypothesis results at {result_path}, loading..."
        )
        import arviz as az

        return az.from_netcdf(str(result_path)), result_path

    logger.info(
        f"No existing full-hypothesis results at {result_path}, running inference..."
    )
    idata_full = run_mcmc(
        data=raw_series,
        n_samples=1000,
        n_warmup=1500,
        num_chains=4,
        n_knots=N_KNOTS,
        degree=2,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir=str(full_outdir),
        verbose=True,
        coarse_grain_config=coarse_cfg,
        Nb=Nb,
        fmin=FMIN,
        fmax=FMAX,
        alpha_delta=ALPHA_DELTA,
        beta_delta=BETA_DELTA,
        only_vi=RUN_VI_ONLY,
        init_from_vi=INIT_FROM_VI_FULL,
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
        compute_lnz=True,
    )
    idata_full.to_netcdf(str(result_path))
    return idata_full, result_path


def _run_diag_hypothesis(
    *,
    diag_outdir: Path,
):
    import arviz as az

    diag_outdir.mkdir(parents=True, exist_ok=True)
    true_freq = np.asarray(true_psd_source[0], dtype=np.float64)
    true_psd_matrix = np.asarray(true_psd_source[1])
    channel_results: list[dict[str, object]] = []

    for channel_index, channel_name in enumerate(LISA_CHANNEL_NAMES):
        channel_dir = diag_outdir / channel_name
        channel_dir.mkdir(parents=True, exist_ok=True)
        result_path = channel_dir / "inference_data.nc"

        if result_path.exists() and REUSE_EXISTING:
            logger.info(
                f"Found existing diagonal-channel results [{channel_name}] at {result_path}, loading..."
            )
            idata_channel = az.from_netcdf(str(result_path))
        else:
            logger.info(
                f"No existing diagonal-channel results [{channel_name}] at {result_path}, running inference..."
            )
            channel_series = Timeseries(t=t_full, y=y_full[:, channel_index])
            true_psd_channel = np.real(
                true_psd_matrix[:, channel_index, channel_index]
            )
            idata_channel = run_mcmc(
                data=channel_series,
                n_samples=4000,
                n_warmup=4000,
                num_chains=4,
                n_knots=N_KNOTS,
                degree=2,
                diffMatrixOrder=2,
                knot_kwargs=dict(strategy="log"),
                outdir=str(channel_dir),
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
                max_tree_depth=MAX_TREE_DEPTH,
                dense_mass=DENSE_MASS,
                true_psd=(true_freq, true_psd_channel),
                compute_lnz=True,
            )
            idata_channel.to_netcdf(str(result_path))

        lnz_ch, lnz_err_ch, lnz_valid = extract_lnz(idata_channel)
        channel_results.append(
            {
                "name": channel_name,
                "path": str(result_path),
                "lnz": lnz_ch,
                "lnz_err": lnz_err_ch,
                "valid": lnz_valid,
            }
        )

    return channel_results


run_full = LISA_HYPOTHESIS_MODE in {"full", "both"}
run_diag = LISA_HYPOTHESIS_MODE in {"diag", "both"}
if not (run_full or run_diag):
    raise ValueError(
        f"Invalid LISA_HYPOTHESIS_MODE='{LISA_HYPOTHESIS_MODE}'. Expected full/diag/both."
    )

idata = None
full_result_path = FULL_RESULTS_DIR / "inference_data.nc"
diag_channel_results: list[dict[str, object]] = []
if run_full:
    idata, full_result_path = _run_full_hypothesis(
        full_outdir=FULL_RESULTS_DIR
    )
    logger.info(f"Full-hypothesis results saved at {full_result_path}")
if run_diag:
    diag_channel_results = _run_diag_hypothesis(diag_outdir=DIAG_RESULTS_DIR)
    logger.info(
        f"Diagonal-hypothesis channel outputs saved under {DIAG_RESULTS_DIR}"
    )

full_lnz = np.nan
full_lnz_err = np.nan
full_valid = False
if idata is not None:
    full_lnz, full_lnz_err, full_valid = extract_lnz(idata)

diag_lnz, diag_lnz_err, diag_valid = combine_diag_lnz(diag_channel_results)
log_bf, log_bf_err, log_bf_valid = compare_full_vs_diag(
    full_lnz,
    full_lnz_err,
    diag_lnz,
    diag_lnz_err,
    full_valid=full_valid,
    diag_valid=diag_valid,
)

if LISA_RUN_COMPARE_SUMMARY:
    write_evidence_summary(
        txt_path=COMPARE_TXT,
        json_path=COMPARE_JSON,
        run_mode=LISA_HYPOTHESIS_MODE,
        full=(
            {
                "path": str(full_result_path),
                "lnz": full_lnz,
                "lnz_err": full_lnz_err,
                "valid": full_valid,
            }
            if run_full
            else None
        ),
        diag_channels=diag_channel_results,
        diag_combined=(
            {
                "lnz": diag_lnz,
                "lnz_err": diag_lnz_err,
                "valid": diag_valid,
            }
            if run_diag
            else None
        ),
        comparison={
            "log_bf": log_bf,
            "log_bf_err": log_bf_err,
            "valid": log_bf_valid,
        },
    )
    logger.info(
        f"Saved evidence comparison summaries to {COMPARE_TXT} and {COMPARE_JSON}"
    )

if run_diag:
    for entry in diag_channel_results:
        logger.info(
            f"Diag[{entry['name']}]: lnz={entry['lnz']} lnz_err={entry['lnz_err']} valid={entry['valid']}"
        )
if run_full:
    logger.info(
        f"Full: lnz={full_lnz} lnz_err={full_lnz_err} valid={full_valid}"
    )
if run_full and run_diag:
    logger.info(
        f"logBF(full-diag)={log_bf} ± {log_bf_err} (valid={log_bf_valid})"
    )

if idata is None:
    logger.info(
        "Full-hypothesis run not requested; skipping multivariate PSD matrix plotting/metrics."
    )
    raise SystemExit(0)

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
# Computing Welch within each analysis block and averaging keeps the diagnostic
# estimate aligned with the blocked Wishart preprocessing.
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
    PSDMatrixPlotSpec(
        idata=idata,
        freq=freq_plot,
        empirical_psd=None,  # extracted from idata.observed_data
        extra_empirical_psd=[empirical_welch],
        extra_empirical_labels=[
            "Welch (block-avg)" if WELCH_BLOCK_AVG else "Welch"
        ],
        extra_empirical_styles=[
            dict(color="0.5", lw=1.3, alpha=0.9, ls="-", zorder=-4),
        ],
        outdir=str(FULL_RESULTS_DIR),
        filename="psd_matrix.png",
        diag_yscale="log",
        offdiag_yscale="linear",
        xscale="log",
        show_csd_magnitude=False,
        show_coherence=True,
        overlay_vi=True,
        freq_range=(FMIN, FMAX),
        true_psd=true_psd_physical,
    )
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
    summary_path = FULL_RESULTS_DIR / "psd_accuracy_summary.txt"
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
