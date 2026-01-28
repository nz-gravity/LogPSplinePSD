import os
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from typing import Callable, Tuple

# Print number of JAX devices
import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.diagnostics._utils import (
    compute_ci_coverage_multivar,
    compute_matrix_riae,
    compute_riae,
    extract_percentile,
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


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


RUN_VI_ONLY = _env_flag("LISA_MULTIVAR_RUN_VI_ONLY", False)
INIT_FROM_VI = _env_flag("LISA_MULTIVAR_INIT_FROM_VI", False)
# Set True to skip sampling when results already exist.
REUSE_EXISTING = _env_flag("LISA_MULTIVAR_REUSE_EXISTING", False)
RECOMPUTE_POSTERIOR_PSD = _env_flag(
    "LISA_MULTIVAR_RECOMPUTE_POSTERIOR_PSD", False
)
USE_LISATOOLS_SYNTH = True
LISATOOLS_SYNTH_NPZ = RESULTS_DIR / "lisatools_synth_data.npz"

# Hyperparameters and spline configuration for this study
ALPHA_DELTA = 3.0
BETA_DELTA = 3.0
N_KNOTS = 30
TARGET_ACCEPT = 0.7
MAX_TREE_DEPTH = 10
TARGET_ACCEPT_BY_CHANNEL: list[float] | None = [0.7, 0.6, 0.75]
# Avoid raising max_tree_depth unless you have to: if a channel already hits the
# max steps, increasing max_tree_depth can dramatically increase walltime.
MAX_TREE_DEPTH_BY_CHANNEL: list[int] | None = [10, 10, 10]
DENSE_MASS = True
VI_GUIDE = "diag"
VI_STEPS = 200_000
VI_LR = 1e-4
VI_POSTERIOR_DRAWS = 1024
BLOCK_DAYS = 14.0  # legacy; superseded by infer_blocks()
MAX_TIME_BLOCKS = 12
N_TIME_BLOCKS_OVERRIDE: int | None = None

C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LASER_FREQ = 2.81e14  # Hz
METRICS_MIN_PCT = 5.0
METRICS_LOG_EPS = 1e-60
PLOT_PSD_UNITS = "freq"  # "freq" -> Hz^2/Hz, "strain" -> 1/Hz


def infer_blocks(n_time: int, *, max_blocks: int = MAX_TIME_BLOCKS) -> int:
    target = max(1, 2 ** int(np.round(np.log2(n_time / (24 * 7)))))
    while target > 1 and n_time % target != 0:
        target //= 2
    while target > max_blocks:
        target //= 2
    return max(1, target)


def _strain_to_freq_scale(freq: np.ndarray) -> np.ndarray:
    return (2.0 * np.pi * freq * LASER_FREQ * L_ARM / C_LIGHT) ** 2


def _resolve_plot_units(
    base_psd_units: str, plot_psd_units: str
) -> Tuple[Callable | None, str]:
    base_psd_units = str(base_psd_units).lower().strip()
    plot_psd_units = str(plot_psd_units).lower().strip()
    if plot_psd_units not in {"freq", "strain"}:
        raise ValueError(
            f"plot_psd_units must be 'freq' or 'strain', got {plot_psd_units!r}."
        )
    if base_psd_units not in {"freq", "strain"}:
        raise ValueError(
            f"base_psd_units must be 'freq' or 'strain', got {base_psd_units!r}."
        )

    unit_label = "Hz^2/Hz" if plot_psd_units == "freq" else "1/Hz"
    if plot_psd_units == base_psd_units:
        return None, unit_label
    if base_psd_units == "strain" and plot_psd_units == "freq":
        return _strain_to_freq_scale, unit_label
    raise NotImplementedError(
        "Only strain→freq conversion is supported for plotting."
    )


def _relative_l2_matrix(
    est_psd: np.ndarray, true_psd: np.ndarray, freqs: np.ndarray
) -> float:
    diff_norm2 = np.array(
        [
            np.linalg.norm(est_psd[k] - true_psd[k], "fro") ** 2
            for k in range(len(freqs))
        ]
    )
    true_norm2 = np.array(
        [np.linalg.norm(true_psd[k], "fro") ** 2 for k in range(len(freqs))]
    )
    numerator = float(simpson(diff_norm2, x=freqs))
    denominator = float(simpson(true_norm2, x=freqs))
    return (
        float(np.sqrt(numerator / denominator))
        if denominator != 0
        else float("nan")
    )


def _relative_l2_vector(
    est: np.ndarray, true: np.ndarray, freqs: np.ndarray
) -> float:
    diff_sq = (est - true) ** 2
    numerator = float(simpson(diff_sq, x=freqs))
    denominator = float(simpson(true**2, x=freqs))
    return (
        float(np.sqrt(numerator / denominator))
        if denominator != 0
        else float("nan")
    )


def _summarize_psd_metrics(
    psd_ds,
    label: str,
    true_psd: np.ndarray,
    freqs: np.ndarray,
    freq_mask: np.ndarray | None = None,
) -> dict | None:
    if psd_ds is None:
        return None

    psd_real = np.asarray(psd_ds["psd_matrix_real"].values)
    percentiles = np.asarray(
        psd_ds["psd_matrix_real"].coords.get("percentile", []), dtype=float
    )
    if percentiles.size == 0:
        logger.warning("%s metrics skipped: missing percentiles.", label)
        return None

    psd_imag = (
        np.asarray(psd_ds["psd_matrix_imag"].values)
        if "psd_matrix_imag" in psd_ds
        else np.zeros_like(psd_real)
    )

    q50_real = extract_percentile(psd_real, percentiles, 50.0)
    q05_real = extract_percentile(psd_real, percentiles, 5.0)
    q95_real = extract_percentile(psd_real, percentiles, 95.0)
    q50_im = extract_percentile(psd_imag, percentiles, 50.0)
    q05_im = extract_percentile(psd_imag, percentiles, 5.0)
    q95_im = extract_percentile(psd_imag, percentiles, 95.0)

    if freq_mask is not None:
        freqs = freqs[freq_mask]
        q50_real = q50_real[freq_mask]
        q05_real = q05_real[freq_mask]
        q95_real = q95_real[freq_mask]
        q50_im = q50_im[freq_mask]
        q05_im = q05_im[freq_mask]
        q95_im = q95_im[freq_mask]
        true_psd = true_psd[freq_mask]

    riae = compute_matrix_riae(q50_real, true_psd.real, freqs)
    l2_rel = _relative_l2_matrix(q50_real, true_psd.real, freqs)

    true_diag = np.diagonal(true_psd.real, axis1=1, axis2=2)
    est_diag = np.diagonal(q50_real, axis1=1, axis2=2)
    log_true = np.log10(np.maximum(true_diag, METRICS_LOG_EPS))
    log_est = np.log10(np.maximum(est_diag, METRICS_LOG_EPS))
    log_riae = float(
        np.mean(
            [
                compute_riae(log_est[:, i], log_true[:, i], freqs)
                for i in range(log_true.shape[1])
            ]
        )
    )
    log_l2 = float(
        np.mean(
            [
                _relative_l2_vector(log_est[:, i], log_true[:, i], freqs)
                for i in range(log_true.shape[1])
            ]
        )
    )

    percentiles_stack = np.stack(
        [
            q05_real + 1j * q05_im,
            q50_real + 1j * q50_im,
            q95_real + 1j * q95_im,
        ],
        axis=0,
    )
    # Coverage should be assessed on the full complex matrix representation:
    # upper triangle -> real part, strict lower triangle -> imaginary part.
    # Passing only ``true_psd.real`` would incorrectly treat all imaginary
    # components as zero and can drive coverage to ~0 even when the posterior
    # is sensible.
    coverage = compute_ci_coverage_multivar(percentiles_stack, true_psd)

    diag_widths = np.diagonal(q95_real - q05_real, axis1=1, axis2=2)
    width_median = float(np.median(diag_widths))
    width_mean = float(np.mean(diag_widths))

    return {
        "label": label,
        "riae_matrix": float(riae),
        "relative_l2_matrix": float(l2_rel),
        "log_riae_diag": log_riae,
        "log_l2_diag": log_l2,
        "coverage_90": float(coverage),
        "ci_width_median": width_median,
        "ci_width_mean": width_mean,
    }


if USE_LISATOOLS_SYNTH:
    if not LISATOOLS_SYNTH_NPZ.exists():
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
    n_blocks = infer_blocks(n_time)

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
    f_transition=FMIN,
    n_log_bins=512,
    f_min=FMIN,
    f_max=FMAX,
)

raw_series = MultivariateTimeseries(y=y_full, t=t_full)

fs = 1.0 / dt
fmin_full = 1.0 / (len(t_full) * dt)

idata = None

if RESULT_FN.exists() and REUSE_EXISTING:
    logger.info(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(str(RESULT_FN))
    if RECOMPUTE_POSTERIOR_PSD:
        logger.info(
            "Recomputing posterior PSD quantiles from stored draws "
            "(LISA_MULTIVAR_RECOMPUTE_POSTERIOR_PSD=1)."
        )
        from xarray import DataArray, Dataset

        from log_psplines.psplines.multivar_psplines import (
            MultivariateLogPSplines,
        )

        sample_stats = idata.sample_stats
        required = {"log_delta_sq", "theta_re", "theta_im"}
        missing = [name for name in required if name not in sample_stats]
        if missing:
            raise KeyError(
                "Cannot recompute posterior PSD group; missing sample_stats: "
                + ", ".join(missing)
            )

        dummy_model = MultivariateLogPSplines.__new__(MultivariateLogPSplines)
        percentiles = [5.0, 50.0, 95.0]
        psd_real_q, psd_imag_q, coh_q = dummy_model.compute_psd_quantiles(
            np.asarray(sample_stats["log_delta_sq"].values),
            np.asarray(sample_stats["theta_re"].values),
            np.asarray(sample_stats["theta_im"].values),
            percentiles=percentiles,
            n_samples_max=200,
            compute_coherence=True,
        )

        channel_stds = idata.attrs.get("channel_stds")
        if channel_stds is not None:
            channel_stds = np.asarray(channel_stds, dtype=float)
            factor_matrix = np.outer(channel_stds, channel_stds).astype(float)
            factor_4d = factor_matrix[None, None, :, :]
            psd_real_q = np.asarray(psd_real_q) * factor_4d
            psd_imag_q = np.asarray(psd_imag_q) * factor_4d

        freq = np.asarray(idata.observed_data["freq"].values, dtype=float)
        channels = np.asarray(idata.observed_data["channels"].values)
        channels2 = np.asarray(idata.observed_data["channels2"].values)

        posterior_psd = Dataset(
            {
                "psd_matrix_real": DataArray(
                    psd_real_q,
                    dims=["percentile", "freq", "channels", "channels2"],
                    coords={
                        "percentile": np.asarray(percentiles, dtype=float),
                        "freq": freq,
                        "channels": channels,
                        "channels2": channels2,
                    },
                ),
                "psd_matrix_imag": DataArray(
                    psd_imag_q,
                    dims=["percentile", "freq", "channels", "channels2"],
                    coords={
                        "percentile": np.asarray(percentiles, dtype=float),
                        "freq": freq,
                        "channels": channels,
                        "channels2": channels2,
                    },
                ),
            }
        )
        if coh_q is not None:
            posterior_psd["coherence"] = DataArray(
                coh_q,
                dims=["percentile", "freq", "channels", "channels2"],
                coords={
                    "percentile": np.asarray(percentiles, dtype=float),
                    "freq": freq,
                    "channels": channels,
                    "channels2": channels2,
                },
            )
        idata.posterior_psd = posterior_psd
        tmp_fn = RESULT_FN.with_name(f"{RESULT_FN.stem}.tmp{RESULT_FN.suffix}")
        idata.to_netcdf(str(tmp_fn))
        try:
            for group in idata.groups():
                getattr(idata, group).close()
        except Exception:
            pass
        os.replace(tmp_fn, RESULT_FN)

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

freq_plot = np.asarray(idata["posterior_psd"]["freq"].values)


def _interp_matrix(
    freq_src: np.ndarray, mat: np.ndarray, freq_tgt: np.ndarray
):
    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    mat = np.asarray(mat)

    # Guard against non-strictly-increasing grids (e.g. some generators replace
    # the zero-frequency bin by copying the first positive frequency).
    sort_idx = np.argsort(freq_src)
    freq_sorted = freq_src[sort_idx]
    mat_sorted = mat[sort_idx]
    freq_unique, uniq_idx = np.unique(freq_sorted, return_index=True)
    mat_unique = mat_sorted[uniq_idx]

    if freq_unique.shape == freq_tgt.shape and np.allclose(
        freq_unique, freq_tgt
    ):
        return np.asarray(mat_unique)

    flat = mat_unique.reshape(mat_unique.shape[0], -1)
    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, i].real)
            for i in range(flat.shape[1])
        ]
    ).T
    imag_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, i].imag)
            for i in range(flat.shape[1])
        ]
    ).T
    return (real_interp + 1j * imag_interp).reshape(
        (freq_tgt.size,) + mat_unique.shape[1:]
    )


true_psd_physical = _interp_matrix(
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

psd_scale_plot, psd_unit_label_plot = _resolve_plot_units(
    base_psd_units, PLOT_PSD_UNITS
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
    _summarize_psd_metrics(
        getattr(idata, "posterior_psd", None),
        "NUTS",
        true_psd_physical,
        freq_plot,
        freq_mask=metrics_mask,
    )
)
metrics.append(
    _summarize_psd_metrics(
        getattr(idata, "vi_posterior_psd", None),
        "VI",
        true_psd_physical,
        freq_plot,
        freq_mask=metrics_mask,
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
