import os
import shutil
import argparse
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Print number of JAX devices
import jax
import numpy as np
from evidence_utils import extract_lnz

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.diagnostics import psd_compare
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

logger.info(f"JAX devices: {jax.devices()}")

set_level("INFO")

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
SHARED_CACHE_DIR = RESULTS_ROOT / "lisa_shared"
SHARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_RESULTS_DIR = RESULTS_ROOT / "lisa"
LEGACY_SYNTH_NPZ = LEGACY_RESULTS_DIR / "lisa_data.npz"
LEGACY_FALLBACK_SYNTH_NPZ = LEGACY_RESULTS_DIR / "lisatools_synth_data.npz"
SHARED_SYNTH_NPZ = SHARED_CACHE_DIR / "lisa_data.npz"

FULL_N_SAMPLES = 1000
FULL_N_WARMUP = 1500
FULL_NUM_CHAINS = 4

RUN_VI_ONLY = False
INIT_FROM_VI_FULL = False
REUSE_EXISTING = False  # set True to skip sampling when results already exist
USE_LISATOOLS_SYNTH = True

# Hyperparameters and spline configuration for this study
ALPHA_DELTA = 3.0
BETA_DELTA = 3.0
N_KNOTS = 20
KNOT_METHOD = "density"  # one of: density, log, uniform
DIFF_MATRIX_ORDER = 2
TARGET_ACCEPT = 0.7
MAX_TREE_DEPTH = 10
TARGET_ACCEPT_BY_CHANNEL: list[float] | None = None
DESIGN_PSD_TAU: float | None = None  # None disables soft shrinkage toward true PSD
COMPUTE_LNZ = False
# Avoid raising max_tree_depth unless you have to: if a channel already hits the
# max steps, increasing max_tree_depth can dramatically increase walltime.
MAX_TREE_DEPTH_BY_CHANNEL: list[int] | None = [10, 10, 10]
DENSE_MASS = True
VI_GUIDE = "diag"
VI_STEPS = 1000_000
VI_LR = 1e-4
VI_POSTERIOR_DRAWS = 1024
MAX_TIME_BLOCKS = 12
N_TIME_BLOCKS_OVERRIDE: int | None = None
BLOCK_DAYS = 7.0
MAX_DAYS = 365.0
MAX_MONTHS = 0.0
WELCH_NPERSEG = 0
WELCH_OVERLAP_FRAC = 0.5
WELCH_WINDOW = "hann"
WELCH_TUKEY_ALPHA = 0.1
WELCH_BLOCK_AVG = True
WISHART_WINDOW = "none"  # one of: none, hann, tukey
WISHART_TUKEY_ALPHA = 0.1
ENABLE_COARSE_GRAIN = True
COARSE_GRAIN_FACTOR = 0
TARGET_COARSE_BINS = 8192
FMIN = 10**-4
FMAX = 10**-1

C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LASER_FREQ = 2.81e14  # Hz
METRICS_MIN_PCT = 5.0
METRICS_LOG_EPS = 1e-60
PLOT_PSD_UNITS = "freq"  # "freq" -> Hz^2/Hz, "strain" -> 1/Hz


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LISA multivariate LogPSplinePSD experiment."
    )
    parser.add_argument("--n-knots", type=int, default=None)
    parser.add_argument(
        "--diff-order",
        type=int,
        choices=(1, 2),
        default=None,
        help="P-spline difference penalty order.",
    )
    parser.add_argument(
        "--knot-method",
        type=str,
        choices=("density", "log", "uniform", "linear"),
        default=None,
        help="Knot placement method. 'linear' maps to 'uniform'.",
    )
    parser.add_argument("--fmin", type=float, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--target-coarse-bins", type=int, default=None)
    parser.add_argument("--coarse-grain-factor", type=int, default=None)
    parser.add_argument(
        "--enable-coarse-grain",
        dest="enable_coarse_grain",
        action="store_true",
    )
    parser.add_argument(
        "--disable-coarse-grain",
        dest="enable_coarse_grain",
        action="store_false",
    )
    parser.set_defaults(enable_coarse_grain=None)
    parser.add_argument(
        "--init-from-vi", dest="init_from_vi", action="store_true"
    )
    parser.add_argument(
        "--no-init-from-vi", dest="init_from_vi", action="store_false"
    )
    parser.set_defaults(init_from_vi=None)
    parser.add_argument("--compute-lnz", dest="compute_lnz", action="store_true")
    parser.add_argument(
        "--no-compute-lnz", dest="compute_lnz", action="store_false"
    )
    parser.set_defaults(compute_lnz=None)
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Enable design PSD shrinkage with this tau value.",
    )
    parser.add_argument(
        "--tau-off", action="store_true", help="Disable design PSD shrinkage."
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-warmup", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=None)
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--block-days", type=float, default=None)
    parser.add_argument(
        "--wishart-window",
        type=str,
        choices=("none", "hann", "tukey"),
        default=None,
    )
    parser.add_argument("--wishart-tukey-alpha", type=float, default=None)
    parser.add_argument(
        "--welch-window",
        type=str,
        choices=("none", "hann", "tukey"),
        default=None,
    )
    parser.add_argument("--welch-tukey-alpha", type=float, default=None)
    parser.add_argument(
        "--reuse-existing", dest="reuse_existing", action="store_true"
    )
    parser.add_argument(
        "--no-reuse-existing", dest="reuse_existing", action="store_false"
    )
    parser.set_defaults(reuse_existing=None)
    return parser.parse_args()


args = _parse_cli()
if args.n_knots is not None:
    N_KNOTS = int(args.n_knots)
if args.diff_order is not None:
    DIFF_MATRIX_ORDER = int(args.diff_order)
if args.knot_method is not None:
    KNOT_METHOD = (
        "uniform" if args.knot_method.strip().lower() == "linear" else args.knot_method.strip().lower()
    )
if args.fmin is not None:
    FMIN = float(args.fmin)
if args.fmax is not None:
    FMAX = float(args.fmax)
if FMIN <= 0.0 or FMAX <= 0.0 or FMAX <= FMIN:
    raise ValueError(f"Require 0 < fmin < fmax, got fmin={FMIN}, fmax={FMAX}.")
if args.target_coarse_bins is not None:
    TARGET_COARSE_BINS = int(args.target_coarse_bins)
if args.coarse_grain_factor is not None:
    COARSE_GRAIN_FACTOR = int(args.coarse_grain_factor)
if args.enable_coarse_grain is not None:
    ENABLE_COARSE_GRAIN = bool(args.enable_coarse_grain)
if args.init_from_vi is not None:
    INIT_FROM_VI_FULL = bool(args.init_from_vi)
if args.compute_lnz is not None:
    COMPUTE_LNZ = bool(args.compute_lnz)
if args.tau_off and args.tau is not None:
    raise ValueError("Pass either --tau or --tau-off, not both.")
if args.tau_off:
    DESIGN_PSD_TAU = None
elif args.tau is not None:
    DESIGN_PSD_TAU = float(args.tau)
if args.n_samples is not None:
    FULL_N_SAMPLES = int(args.n_samples)
if args.n_warmup is not None:
    FULL_N_WARMUP = int(args.n_warmup)
if args.num_chains is not None:
    FULL_NUM_CHAINS = int(args.num_chains)
if args.target_accept is not None:
    TARGET_ACCEPT = float(args.target_accept)
if args.max_tree_depth is not None:
    MAX_TREE_DEPTH = int(args.max_tree_depth)
if args.block_days is not None:
    BLOCK_DAYS = float(args.block_days)
if args.wishart_window is not None:
    WISHART_WINDOW = str(args.wishart_window).strip().lower()
if args.wishart_tukey_alpha is not None:
    WISHART_TUKEY_ALPHA = float(args.wishart_tukey_alpha)
if args.welch_window is not None:
    WELCH_WINDOW = str(args.welch_window).strip().lower()
if args.welch_tukey_alpha is not None:
    WELCH_TUKEY_ALPHA = float(args.welch_tukey_alpha)
if args.reuse_existing is not None:
    REUSE_EXISTING = bool(args.reuse_existing)


def _window_spec(
    name: str, *, tukey_alpha: float
) -> str | tuple[str, float] | None:
    key = str(name).strip().lower()
    if key in ("none", "rect", "rectangular"):
        return None
    if key == "hann":
        return "hann"
    if key == "tukey":
        if not (0.0 <= float(tukey_alpha) <= 1.0):
            raise ValueError(
                f"Tukey alpha must be in [0, 1], got {tukey_alpha}."
            )
        return ("tukey", float(tukey_alpha))
    raise ValueError(f"Unsupported window name: {name}")


WISHART_WINDOW_SPEC = _window_spec(
    WISHART_WINDOW, tukey_alpha=WISHART_TUKEY_ALPHA
)
WELCH_WINDOW_SPEC = _window_spec(WELCH_WINDOW, tukey_alpha=WELCH_TUKEY_ALPHA)


def _window_slug(window_spec: str | tuple[str, float] | None) -> str:
    if window_spec is None:
        return "rect"
    if isinstance(window_spec, tuple):
        name, alpha = window_spec
        alpha_slug = _format_decimal_slug(float(alpha))
        return f"{str(name)}{alpha_slug}"
    return str(window_spec)


def _welch_window_arg(
    window_spec: str | tuple[str, float] | None,
) -> str | tuple[str, float]:
    # scipy.signal.welch does not accept None as a valid window argument.
    # "boxcar" is the rectangular/no-taper equivalent.
    if window_spec is None:
        return "boxcar"
    return window_spec

Nb_hint: int | None = None
Lb_hint: int | None = None

if USE_LISATOOLS_SYNTH:
    preferred_npz: Path | None = None
    if LEGACY_FALLBACK_SYNTH_NPZ.exists():
        preferred_npz = LEGACY_FALLBACK_SYNTH_NPZ
    elif LEGACY_SYNTH_NPZ.exists():
        preferred_npz = LEGACY_SYNTH_NPZ

    if preferred_npz is None and not SHARED_SYNTH_NPZ.exists():
        raise FileNotFoundError(
            f"{SHARED_SYNTH_NPZ} not found. Run lisa_datagen.py first."
        )

    if preferred_npz is not None:
        needs_refresh = True
        if SHARED_SYNTH_NPZ.exists():
            try:
                needs_refresh = not SHARED_SYNTH_NPZ.samefile(preferred_npz)
            except OSError:
                needs_refresh = True
        if needs_refresh:
            SHARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if SHARED_SYNTH_NPZ.exists() or SHARED_SYNTH_NPZ.is_symlink():
                SHARED_SYNTH_NPZ.unlink()
            try:
                SHARED_SYNTH_NPZ.symlink_to(preferred_npz)
                logger.info(
                    f"Symlinked synth data from {preferred_npz} -> {SHARED_SYNTH_NPZ}."
                )
            except OSError:
                shutil.copy2(preferred_npz, SHARED_SYNTH_NPZ)
                logger.info(
                    f"Copied synth data from {preferred_npz} -> {SHARED_SYNTH_NPZ}."
                )

    with np.load(SHARED_SYNTH_NPZ, allow_pickle=False) as synth:
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


# Pilot stability band is set via globals FMIN/FMAX (CLI-overridable).

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


def _format_freq_slug(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "")


def _format_decimal_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _coarse_slug(cfg: CoarseGrainConfig) -> str:
    if not cfg.enabled:
        return "cgOff"
    if cfg.Nc is not None:
        return f"cgNc{int(cfg.Nc)}"
    if cfg.Nh is not None:
        return f"cgNh{int(cfg.Nh)}"
    return "cgOn"


def _build_run_slug() -> str:
    vi_enabled = bool(RUN_VI_ONLY or INIT_FROM_VI_FULL)
    total_nuts_steps = FULL_NUM_CHAINS * (FULL_N_WARMUP + FULL_N_SAMPLES)
    parts = [
        "lisa",
        f"nb{Nb}",
        f"lb{Lb}",
        _coarse_slug(coarse_cfg),
        f"k{N_KNOTS}",
        f"d{DIFF_MATRIX_ORDER}",
        f"km{KNOT_METHOD}",
        f"ww{_window_slug(WISHART_WINDOW_SPEC)}",
        f"ew{_window_slug(WELCH_WINDOW_SPEC)}",
        f"f{_format_freq_slug(FMIN)}-{_format_freq_slug(FMAX)}",
        f"ta{_format_decimal_slug(TARGET_ACCEPT)}",
        f"td{MAX_TREE_DEPTH}",
        "dmOn" if DENSE_MASS else "dmOff",
        f"nutsW{FULL_N_WARMUP}S{FULL_N_SAMPLES}",
        f"steps{total_nuts_steps}",
        "viOn" if vi_enabled else "viOff",
        "lnzOn" if COMPUTE_LNZ else "lnzOff",
        (
            f"tau{_format_decimal_slug(DESIGN_PSD_TAU)}"
            if DESIGN_PSD_TAU is not None
            else "tauOff"
        ),
    ]
    return "_".join(parts)


RUN_SLUG = _build_run_slug()
RUN_DIR = RESULTS_ROOT / RUN_SLUG
RUN_DIR.mkdir(parents=True, exist_ok=True)
logger.info(
    "Run config: "
    f"knot_method={KNOT_METHOD}, n_knots={N_KNOTS}, diff_order={DIFF_MATRIX_ORDER}, "
    f"f=[{FMIN:.3g}, {FMAX:.3g}], coarse={_coarse_slug(coarse_cfg)}, "
    f"wishart_window={WISHART_WINDOW_SPEC}, welch_window={WELCH_WINDOW_SPEC}, "
    f"vi_init={INIT_FROM_VI_FULL}, lnz={COMPUTE_LNZ}, tau={DESIGN_PSD_TAU}"
)

if USE_LISATOOLS_SYNTH:
    run_npz = RUN_DIR / "lisa_data.npz"
    if not run_npz.exists():
        try:
            run_npz.symlink_to(SHARED_SYNTH_NPZ)
            logger.info(
                f"Linked run-local synth NPZ: {run_npz} -> {SHARED_SYNTH_NPZ}"
            )
        except OSError:
            shutil.copy2(SHARED_SYNTH_NPZ, run_npz)
            logger.info(f"Copied synth NPZ into run directory: {run_npz}")
else:
    lisa_data.plot(f"{RUN_DIR}/lisa_raw.png")

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
        n_samples=FULL_N_SAMPLES,
        n_warmup=FULL_N_WARMUP,
        num_chains=FULL_NUM_CHAINS,
        n_knots=N_KNOTS,
        degree=2,
        diffMatrixOrder=DIFF_MATRIX_ORDER,
        knot_kwargs=dict(method=KNOT_METHOD),
        outdir=str(full_outdir),
        verbose=True,
        coarse_grain_config=coarse_cfg,
        wishart_window=WISHART_WINDOW_SPEC,
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
        compute_lnz=COMPUTE_LNZ,
        design_psd=true_psd_source if DESIGN_PSD_TAU is not None else None,
        tau=DESIGN_PSD_TAU,
    )
    idata_full.to_netcdf(str(result_path))
    return idata_full, result_path


idata, full_result_path = _run_full_hypothesis(full_outdir=RUN_DIR)
logger.info(f"Full-hypothesis results saved at {full_result_path}")
if COMPUTE_LNZ:
    full_lnz, full_lnz_err, full_valid = extract_lnz(idata)
    logger.info(f"Full: lnz={full_lnz} lnz_err={full_lnz_err} valid={full_valid}")
else:
    logger.info("Full: lnz disabled (COMPUTE_LNZ=False).")

posterior_ds = idata["posterior"]
posterior_var_count = len(posterior_ds.data_vars)
posterior_param_count = 0
for var in posterior_ds.data_vars.values():
    n_elements = int(np.prod(var.shape[2:])) if var.ndim >= 2 else int(np.prod(var.shape))
    posterior_param_count += n_elements
logger.info(
    f"InferenceData summary: posterior vars={posterior_var_count}, "
    f"approx scalar params/draw={posterior_param_count}"
)

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
    f"window={WELCH_WINDOW_SPEC!r}, df={welch_df:.3g} Hz."
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
    window: str | tuple[str, float] | None,
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
        window=_welch_window_arg(WELCH_WINDOW_SPEC),
        detrend=False,
    )
else:
    empirical_welch = EmpiricalPSD.from_timeseries_data(
        data=y_full,
        fs=fs,
        nperseg=welch_nperseg,
        noverlap=welch_noverlap,
        window=_welch_window_arg(WELCH_WINDOW_SPEC),
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
        outdir=str(RUN_DIR),
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
    summary_path = RUN_DIR / "psd_accuracy_summary.txt"
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
