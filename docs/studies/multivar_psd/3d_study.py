"""Multivariate PSD simulation study with in-script VAR data generation.

CLI args:
1) seed (default 0)
2) mode: "large" or "short"
"""

import argparse
import csv
import json
import os
from typing import Literal

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import numpy as np

from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

jax.config.update("jax_enable_x64", True)

set_level("DEBUG")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("out_var3")

DEFAULT_KNOT_METHOD = "density"
DEFAULT_TARGET_ACCEPT_PROB = 0.95
DEFAULT_MAX_TREE_DEPTH = 14
DEFAULT_INIT_FROM_VI = True
DEFAULT_VI_STEPS = 100_000
DEFAULT_VI_GUIDE = "lowrank:16"
VI_LR = 5e-4
DEFAULT_VI_PSD_MAX_DRAWS = 256
DEFAULT_POSTERIOR_PSD_MAX_DRAWS = 256
DEFAULT_ALPHA_DELTA = 1.0
DEFAULT_BETA_DELTA = 1.0
# Total Niter=8000 implemented as 4000 warmup + 4000 posterior samples.
DEFAULT_N_SAMPLES = 4000
DEFAULT_N_WARMUP = 4000
DEFAULT_NUM_CHAINS = 4

DEFAULT_FS = 1.0  # Hz
DEFAULT_BURN_IN = 512
EPS = 1e-12

# VAR(2) setup (3 channels) embedded directly in this script.
A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [
        [-0.2, 0.5, 0.0],  # var2 -> var1 at lag 2
        [0.4, -0.1, 0.0],  # var1 -> var2 at lag 2
        [0.0, 0.0, -0.1],
    ],
    dtype=np.float64,
)
VAR_COEFFS = np.array([A1, A2], dtype=np.float64)

SIGMA_VAL = 0.25
OFF_DIAG = 0.08
SIGMA = np.array(
    [
        [SIGMA_VAL, 0.0, OFF_DIAG],
        [0.0, SIGMA_VAL, OFF_DIAG],
        [OFF_DIAG, OFF_DIAG, SIGMA_VAL],
    ],
    dtype=np.float64,
)

MODE_CONFIG = {
    # short: N=2048, Nb=2 -> block length 1024 -> 512 positive bins
    "short": {"N": 2048, "Nb": 2, "coarse_Nh": None},
    # large: N=16384, Nb=4 with Nh=4 -> effective 512 coarse bins
    "large": {"N": 16 * 1024, "Nb": 4, "coarse_Nh": 4},
}


def _log_var_coefficients() -> None:
    """Log the VAR(p) coefficient matrices used by this study."""
    logger.info("Using VAR coefficients:")
    for lag, coeff in enumerate(VAR_COEFFS, start=1):
        coeff_str = np.array2string(coeff, precision=4, suppress_small=False)
        logger.info(f"A{lag} =\n{coeff_str}")


def _companion_spectral_radius(var_coeffs: np.ndarray) -> float:
    """Return companion-matrix spectral radius for VAR(p) coefficients."""
    ar_order, n_channels, _ = var_coeffs.shape
    companion = np.zeros(
        (n_channels * ar_order, n_channels * ar_order), dtype=np.float64
    )
    companion[:n_channels, : (n_channels * ar_order)] = np.hstack(var_coeffs)
    if ar_order > 1:
        companion[n_channels:, :-n_channels] = np.eye(
            n_channels * (ar_order - 1), dtype=np.float64
        )
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def _simulate_var_process(
    n_samples: int,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    seed: int,
    *,
    fs: float = DEFAULT_FS,
    burn_in: int = DEFAULT_BURN_IN,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate VAR(p): x_t = sum_k A_k x_{t-k} + eps_t.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time grid in seconds.
    x : np.ndarray, shape (N, C)
        Simulated channels.
    """
    ar_order, n_channels, _ = var_coeffs.shape
    n_total = int(n_samples) + int(burn_in)
    rng = np.random.default_rng(int(seed))
    noise = rng.multivariate_normal(np.zeros(n_channels), sigma, size=n_total)
    x = np.zeros((n_total, n_channels), dtype=np.float64)

    for t_idx in range(ar_order, n_total):
        state = noise[t_idx].copy()
        for lag in range(1, ar_order + 1):
            state = state + var_coeffs[lag - 1] @ x[t_idx - lag]
        x[t_idx] = state

    x = x[burn_in:]
    t = np.arange(x.shape[0], dtype=np.float64) / float(fs)
    return t, x


def _calculate_true_var_psd_hz(
    freqs_hz: np.ndarray,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    *,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """Compute one-sided theoretical PSD matrix S(f) on a Hz frequency grid.

    Parameters
    ----------
    freqs_hz : (F,)
        Frequencies in Hz (not angular frequency).
    var_coeffs : (P, C, C)
        VAR coefficient matrices.
    sigma : (C, C)
        Innovation covariance.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    if freqs_hz.ndim != 1:
        raise ValueError("freqs_hz must be one-dimensional.")
    if freqs_hz.size and (
        np.min(freqs_hz) < 0.0 or np.max(freqs_hz) > (fs / 2.0 + 1e-12)
    ):
        raise ValueError("freqs_hz must lie in [0, fs/2].")

    ar_order, n_channels, _ = var_coeffs.shape
    omega = 2.0 * np.pi * freqs_hz / float(fs)  # rad/sample, derived from Hz
    psd = np.empty(
        (freqs_hz.shape[0], n_channels, n_channels), dtype=np.complex128
    )
    ident = np.eye(n_channels, dtype=np.complex128)

    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, ar_order + 1):
            a_f = a_f - var_coeffs[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        s_f = h_f @ sigma @ h_f.conj().T
        psd[idx] = (2.0 / float(fs)) * s_f

    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]

    # Keep matrices numerically Hermitian to protect downstream coherence math.
    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    psd = np.where(np.abs(psd) < EPS, EPS, psd)
    return psd


def _extract_percentile_slice(
    values: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    """Return percentile slice nearest to target."""
    if values.shape[0] == 0:
        raise ValueError("values must contain percentile axis.")
    idx = int(np.argmin(np.abs(percentiles - target)))
    return np.asarray(values[idx], dtype=np.float64)


def _compute_ci_width_metrics(idata) -> dict[str, float]:
    """Compute CI-width summaries from posterior PSD/coherence quantiles."""
    metrics: dict[str, float] = {}
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is None or "psd_matrix_real" not in psd_group:
        return metrics

    psd_real = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    percentiles = np.asarray(
        psd_group["psd_matrix_real"].coords.get(
            "percentile", np.arange(psd_real.shape[0], dtype=float)
        ),
        dtype=np.float64,
    )
    if psd_real.shape[0] < 2:
        return metrics

    q05 = _extract_percentile_slice(psd_real, percentiles, 5.0)
    q95 = _extract_percentile_slice(psd_real, percentiles, 95.0)
    width_psd = np.maximum(q95 - q05, 0.0)

    p = width_psd.shape[1]
    diag_idx = np.arange(p)
    offdiag_mask = ~np.eye(p, dtype=bool)
    diag_width = width_psd[:, diag_idx, diag_idx]
    offdiag_width = width_psd[:, offdiag_mask]

    metrics["ciw_psd_diag_mean"] = float(np.mean(diag_width))
    metrics["ciw_psd_diag_median"] = float(np.median(diag_width))
    metrics["ciw_psd_diag_max"] = float(np.max(diag_width))
    metrics["ciw_psd_offdiag_mean"] = float(np.mean(offdiag_width))
    metrics["ciw_psd_offdiag_median"] = float(np.median(offdiag_width))
    metrics["ciw_psd_offdiag_max"] = float(np.max(offdiag_width))

    if "coherence" in psd_group:
        coherence = np.asarray(psd_group["coherence"].values, dtype=np.float64)
        coh_percentiles = np.asarray(
            psd_group["coherence"].coords.get(
                "percentile", np.arange(coherence.shape[0], dtype=float)
            ),
            dtype=np.float64,
        )
        if coherence.shape[0] >= 2:
            coh_q05 = _extract_percentile_slice(
                coherence, coh_percentiles, 5.0
            )
            coh_q95 = _extract_percentile_slice(
                coherence, coh_percentiles, 95.0
            )
            coh_width = np.maximum(coh_q95 - coh_q05, 0.0)
            coh_offdiag = coh_width[:, offdiag_mask]
            metrics["ciw_coh_offdiag_mean"] = float(np.mean(coh_offdiag))
            metrics["ciw_coh_offdiag_median"] = float(np.median(coh_offdiag))
            metrics["ciw_coh_offdiag_max"] = float(np.max(coh_offdiag))

    return metrics


def _extract_run_metrics(
    idata,
    *,
    seed: int,
    mode: str,
    N: int,
    Nb: int,
    coarse_Nh: int | None,
) -> dict[str, float | int | str]:
    """Extract compact run-level metrics for downstream aggregation."""
    attrs = idata.attrs
    ess_raw = attrs.get("ess", np.nan)
    ess_arr = np.asarray(ess_raw, dtype=float)
    ess_median = float(np.nanmedian(ess_arr)) if ess_arr.size else float("nan")

    metrics: dict[str, float | int | str] = {
        "seed": int(seed),
        "mode": str(mode),
        "N": int(N),
        "Nb": int(Nb),
        "Nh": "OFF" if coarse_Nh is None else int(coarse_Nh),
        "riae_matrix": float(
            attrs.get("riae_matrix", attrs.get("riae", np.nan))
        ),
        "coverage": float(attrs.get("coverage", np.nan)),
        "runtime": float(attrs.get("runtime", np.nan)),
        "ess_median": ess_median,
    }
    metrics.update(_compute_ci_width_metrics(idata))
    return metrics


def _save_metrics_summary(
    outdir: str, metrics: dict[str, float | int | str]
) -> None:
    """Persist compact metrics as JSON and single-row CSV."""
    metrics_json = os.path.join(outdir, "metrics_summary.json")
    metrics_csv = os.path.join(outdir, "metrics_summary.csv")

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"Saved compact metrics to {metrics_json} and {metrics_csv}")


def _extract_lnz_summary(idata) -> tuple[float, float]:
    """Best-effort extraction of total lnZ and its uncertainty."""
    attrs = getattr(idata, "attrs", {})

    lnz = attrs.get("lnz", attrs.get("logz", np.nan))
    lnz_err = attrs.get(
        "lnz_err",
        attrs.get("logz_err", attrs.get("lnz_se", np.nan)),
    )

    return float(lnz), float(lnz_err)


def _save_compact_ci_curves(outdir: str, idata) -> None:
    """Save compact CI-vs-frequency arrays for plotting comparisons."""
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is None:
        raise ValueError("InferenceData has no posterior_psd group.")
    if (
        "psd_matrix_real" not in psd_group
        or "psd_matrix_imag" not in psd_group
    ):
        raise ValueError(
            "posterior_psd must contain psd_matrix_real and psd_matrix_imag."
        )

    freq = np.asarray(psd_group.coords["freq"].values, dtype=np.float64)
    percentiles = np.asarray(
        psd_group.coords["percentile"].values, dtype=np.float64
    )

    psd_real = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    psd_imag = np.asarray(
        psd_group["psd_matrix_imag"].values, dtype=np.float64
    )

    q05_real = _extract_percentile_slice(psd_real, percentiles, 5.0)
    q50_real = _extract_percentile_slice(psd_real, percentiles, 50.0)
    q95_real = _extract_percentile_slice(psd_real, percentiles, 95.0)

    q05_imag = _extract_percentile_slice(psd_imag, percentiles, 5.0)
    q50_imag = _extract_percentile_slice(psd_imag, percentiles, 50.0)
    q95_imag = _extract_percentile_slice(psd_imag, percentiles, 95.0)

    if q50_real.ndim != 3:
        raise ValueError(
            f"Expected q50_real to have shape (F, P, P), got {q50_real.shape}"
        )

    _, p, _ = q50_real.shape
    diag_idx = np.arange(p)
    offdiag_pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]

    psd_diag_q05 = q05_real[:, diag_idx, diag_idx]
    psd_diag_q50 = q50_real[:, diag_idx, diag_idx]
    psd_diag_q95 = q95_real[:, diag_idx, diag_idx]

    if offdiag_pairs:
        off_i = np.array([i for i, _ in offdiag_pairs], dtype=int)
        off_j = np.array([j for _, j in offdiag_pairs], dtype=int)

        psd_offre_q05 = q05_real[:, off_i, off_j]
        psd_offre_q50 = q50_real[:, off_i, off_j]
        psd_offre_q95 = q95_real[:, off_i, off_j]

        psd_offim_q05 = q05_imag[:, off_i, off_j]
        psd_offim_q50 = q50_imag[:, off_i, off_j]
        psd_offim_q95 = q95_imag[:, off_i, off_j]
    else:
        psd_offre_q05 = np.empty((freq.size, 0), dtype=np.float64)
        psd_offre_q50 = np.empty((freq.size, 0), dtype=np.float64)
        psd_offre_q95 = np.empty((freq.size, 0), dtype=np.float64)
        psd_offim_q05 = np.empty((freq.size, 0), dtype=np.float64)
        psd_offim_q50 = np.empty((freq.size, 0), dtype=np.float64)
        psd_offim_q95 = np.empty((freq.size, 0), dtype=np.float64)

    periodogram_real = None
    periodogram_imag = None
    observed_group = getattr(idata, "observed_data", None)
    if observed_group is not None and "periodogram" in observed_group:
        periodogram = np.asarray(observed_group["periodogram"].values)
        periodogram_real = np.real(periodogram).astype(np.float64, copy=False)
        periodogram_imag = np.imag(periodogram).astype(np.float64, copy=False)

    true_psd = _calculate_true_var_psd_hz(
        freq,
        VAR_COEFFS,
        SIGMA,
        fs=DEFAULT_FS,
    )
    true_psd_real = np.real(true_psd).astype(np.float64, copy=False)
    true_psd_imag = np.imag(true_psd).astype(np.float64, copy=False)

    outpath = os.path.join(outdir, "compact_ci_curves.npz")
    save_payload = dict(
        freq=freq,
        psd_diag_q05=psd_diag_q05,
        psd_diag_q50=psd_diag_q50,
        psd_diag_q95=psd_diag_q95,
        psd_offre_q05=psd_offre_q05,
        psd_offre_q50=psd_offre_q50,
        psd_offre_q95=psd_offre_q95,
        psd_offim_q05=psd_offim_q05,
        psd_offim_q50=psd_offim_q50,
        psd_offim_q95=psd_offim_q95,
        offdiag_pairs=np.asarray(offdiag_pairs, dtype=int),
        # Full matrix quantiles (convenient for direct plotting)
        psd_real_q05=q05_real,
        psd_real_q50=q50_real,
        psd_real_q95=q95_real,
        psd_imag_q05=q05_imag,
        psd_imag_q50=q50_imag,
        psd_imag_q95=q95_imag,
        # Optional overlays for plotting without InferenceData.
        true_psd_real=true_psd_real,
        true_psd_imag=true_psd_imag,
    )
    if periodogram_real is not None and periodogram_imag is not None:
        save_payload["periodogram_real"] = periodogram_real
        save_payload["periodogram_imag"] = periodogram_imag

    np.savez_compressed(outpath, **save_payload)
    logger.info(f"Saved compact CI curves to {outpath}")


def _save_compact_run_summary(
    outdir: str,
    *,
    idata,
    seed: int,
    mode: str,
    N: int,
    Nb: int,
    coarse_Nh: int | None,
) -> None:
    """Save compact scalar summaries for later comparison tables."""
    metrics = _extract_run_metrics(
        idata,
        seed=seed,
        mode=mode,
        N=N,
        Nb=Nb,
        coarse_Nh=coarse_Nh,
    )

    lnz, lnz_err = _extract_lnz_summary(idata)
    metrics["lnz"] = lnz
    metrics["lnz_err"] = lnz_err

    out_json = os.path.join(outdir, "compact_run_summary.json")
    out_csv = os.path.join(outdir, "compact_run_summary.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"Saved compact run summary to {out_json} and {out_csv}")


def _prune_outputs_keep_psd_plot(outdir: str) -> None:
    """Delete heavy artifacts, keeping only PSD plot(s) and compact metrics."""
    image_ext = (".png", ".pdf", ".svg", ".jpg", ".jpeg")
    removed_files = 0

    for root, _, files in os.walk(outdir):
        for name in files:
            lower = name.lower()
            path = os.path.join(root, name)
            if lower.endswith(".nc"):
                os.remove(path)
                removed_files += 1
                continue
            if lower.endswith(image_ext):
                stem = os.path.splitext(lower)[0]
                if stem.startswith("psd_matrix"):
                    continue
                os.remove(path)
                removed_files += 1

    # Remove empty subdirectories left after pruning.
    for root, dirs, files in os.walk(outdir, topdown=False):
        if root == outdir:
            continue
        if not dirs and not files:
            try:
                os.rmdir(root)
            except OSError:
                pass

    logger.info(
        f"Pruned heavy outputs in {outdir}; removed {removed_files} files."
    )


def simulation_study(
    *,
    seed: int = 0,
    mode: Literal["large", "short"] = "short",
    outdir: str = OUT,
    K: int = 10,
) -> None:
    cfg = MODE_CONFIG[mode]
    N = int(cfg["N"])
    Nb = int(cfg["Nb"])
    coarse_Nh = cfg["coarse_Nh"]

    print(
        f">>>> Running simulation with mode={mode}, N={N}, Nb={Nb}, K={K}, seed={seed} <<<<"
    )
    _log_var_coefficients()
    outdir = f"{HERE}/{outdir}/seed_{seed}_{mode}_N{N}_K{K}"
    os.makedirs(outdir, exist_ok=True)

    spectral_radius = _companion_spectral_radius(VAR_COEFFS)
    is_stationary = bool(spectral_radius < 1.0)
    logger.info(
        f"Stationarity check (companion spectral radius): {spectral_radius:.6f}"
    )
    if not is_stationary:
        raise ValueError(
            f"Non-stationary VAR coefficients (spectral radius={spectral_radius:.6f})."
        )

    t, data = _simulate_var_process(
        n_samples=N,
        var_coeffs=VAR_COEFFS,
        sigma=SIGMA,
        seed=seed,
        fs=DEFAULT_FS,
        burn_in=DEFAULT_BURN_IN,
    )
    if not np.all(np.isfinite(data)):
        raise ValueError("Generated VAR samples contain non-finite values.")
    ts = MultivariateTimeseries(t=t, y=data)

    freq_true_hz = np.fft.rfftfreq(N, d=1.0 / DEFAULT_FS)[1:]
    true_psd = _calculate_true_var_psd_hz(
        freq_true_hz,
        VAR_COEFFS,
        SIGMA,
        fs=DEFAULT_FS,
    )

    coarse_grain_config = None
    if coarse_Nh is not None:
        coarse_Nh = int(coarse_Nh)
        if coarse_Nh <= 0:
            raise ValueError("coarse_Nh must be positive.")
        coarse_grain_config = dict(
            enabled=True,
            Nc=None,
            Nh=coarse_Nh,
        )

    idata = run_mcmc(
        data=ts,
        n_knots=K,
        degree=2,
        diffMatrixOrder=2,
        n_samples=DEFAULT_N_SAMPLES,
        n_warmup=DEFAULT_N_WARMUP,
        num_chains=DEFAULT_NUM_CHAINS,
        outdir=outdir,
        verbose=True,
        target_accept_prob=DEFAULT_TARGET_ACCEPT_PROB,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        init_from_vi=DEFAULT_INIT_FROM_VI,
        vi_steps=DEFAULT_VI_STEPS,
        vi_guide=DEFAULT_VI_GUIDE,
        vi_psd_max_draws=DEFAULT_VI_PSD_MAX_DRAWS,
        vi_lr=VI_LR,
        Nb=Nb,
        knot_kwargs=dict(method=DEFAULT_KNOT_METHOD),
        coarse_grain_config=coarse_grain_config,
        alpha_delta=DEFAULT_ALPHA_DELTA,
        beta_delta=DEFAULT_BETA_DELTA,
        compute_coherence_quantiles=True,
        true_psd=(freq_true_hz, true_psd),
        max_save_bytes=20_000_000,
    )
    metrics = _extract_run_metrics(
        idata,
        seed=seed,
        mode=mode,
        N=N,
        Nb=Nb,
        coarse_Nh=coarse_Nh,
    )
    _save_metrics_summary(outdir, metrics)
    _save_compact_ci_curves(outdir, idata)
    _save_compact_run_summary(
        outdir,
        idata=idata,
        seed=seed,
        mode=mode,
        N=N,
        Nb=Nb,
        coarse_Nh=coarse_Nh,
    )
    # _prune_outputs_keep_psd_plot(outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Multivariate PSD study with in-script VAR(2) data generation "
            "and mode presets."
        )
    )
    parser.add_argument(
        "seed",
        type=int,
        nargs="?",
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("large", "short"),
        default="short",
        help="Preset size: short=2K samples with Nb=2 (averaging only), large=16K samples with Nb=4 and Nh=4.",
    )

    args = parser.parse_args()
    simulation_study(
        seed=args.seed,
        mode=args.mode,
    )
