"""Publication-quality LISA PSD figures.

Produces three figures from saved ``inference_data.nc`` outputs when available,
falling back to ``compact_ci_curves.npz`` for older runs:

    Figure 1 — run_x XYZ posterior PSD
        3×3 matrix (diagonal = auto-spectra, lower-triangle = coherence).
        Posterior 90 % CI shaded, posterior median solid.
        Empirical data overlay (raw periodogram by default, Welch optional).
        True PSD overlaid.  No knot markers.

    Figure 2 — run_x XYZ posteriors transformed to AET
        Same layout but channels relabelled A, E, T.
        CI curves rotated via  S_AET = M @ S_XYZ @ M^H at each percentile.

    Figure 3 — run_y native AET posterior PSD
        Same layout using the compact_ci_curves.npz from the AET run.

Each figure is accompanied by a relative-error panel for each diagonal
element:  (median - truth) / truth vs. frequency.  Frequency bands where
the true PSD is near its minimum ("dip" regions) are hatched.

Usage
-----
    python paper_final_plots.py \\
        --run-x  runs/run_x_d2_k48_uniform_no_excision/.../seed_0 \\
        --run-y  runs/run_y_aet_d2_k48_uniform/.../seed_0 \\
        --outdir paper_figs

The script can overlay either the saved raw periodogram from ``inference_data.nc``
or a regenerated Welch estimate. Welch requires re-generating the LISA
timeseries and takes ~1–2 min locally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import arviz as az
import jax.numpy as jnp
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── project path setup ────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (SRC_ROOT, PROJECT_ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from log_psplines.arviz_utils.from_arviz import (  # noqa: E402
    get_multivar_ci_summary,
)
from log_psplines.diagnostics._utils import (  # noqa: E402
    compute_ci_coverage_multivar,
    compute_matrix_l2,
    compute_matrix_riae,
)
from log_psplines.plotting.base import setup_plot_style  # noqa: E402
from log_psplines.psplines import (  # noqa: E402
    LogPSplines,
    MultivariateLogPSplines,
)

setup_plot_style()

from utils.aet import (  # noqa: E402
    CHANNEL_LABELS_AET,
    CHANNEL_LABELS_XYZ,
    transform_ci_curves_to_aet,
)

# ── constants ─────────────────────────────────────────────────────────────────
FMIN = 1e-4
FMAX = 1e-1
DEFAULT_WELCH_BLOCK_DAYS = 14.0
SEC_IN_DAY = 86_400.0
DELTA_T = 5.0  # seconds

# PSD unit labels
PSD_UNIT_STRAIN = "1/Hz"
PSD_UNIT_FREQ = r"Hz$^2$/Hz"

# Posterior CI band style
CI_COLOR = "tab:blue"
CI_ALPHA = 0.25
MEDIAN_COLOR = "tab:blue"
MEDIAN_LW = 1.6

# True PSD style
TRUE_COLOR = "k"
TRUE_LW = 1.4
TRUE_LABEL = "True PSD"

# Welch style
WELCH_COLOR = "#e07b00"  # amber
WELCH_LW = 0.4
WELCH_ALPHA = 0.5
WELCH_LABEL = "Welch PSD"
WELCH_OVERLAP = 0.0
RAW_COLOR = "0.45"
RAW_LW = 0.4
RAW_ALPHA = 0.55
RAW_LABEL = "Raw periodogram"
RAW_MAX_POINTS = 1200
COMPARE_COLORS = {
    "30d": "#d95f02",
    "90d": "#1b9e77",
    "365d": "#1f78b4",
}
COMPARE_LABELS = {
    "30d": "1 month",
    "90d": "3 months",
    "365d": "1 year",
}

# Relative error style
RELERR_COLOR = "tab:blue"
RELERR_ZERO_COLOR = "k"
HATCH_ALPHA = 0.15
HATCH_COLOR = "0.5"
HATCH_PATTERN = "///"
# Dip mask: hatch bins where true_psd / max(true_psd_diag) < DIP_THRESHOLD
DIP_THRESHOLD = 0.05  # 5 % of peak — adjust as needed


def _strain_to_freq_factor(freq: np.ndarray) -> np.ndarray:
    """Compute strain → frequency-fluctuation PSD conversion factor per bin.

    S_freq(f) = S_strain(f) * (2π f ν₀ L / c)²
    Returns shape (Nf,) multiplicative factor.
    """
    from log_psplines.example_datasets.lisa_data import (
        C_LIGHT,
        L_ARM,
        LASER_FREQ,
    )

    return (2.0 * np.pi * freq * LASER_FREQ * L_ARM / C_LIGHT) ** 2


def _convert_ci_data_to_freq_units(ci_data: dict) -> dict:
    """Convert CI data from strain PSD to frequency-fluctuation PSD.

    Multiplies all PSD matrices (real, imag, truth) by the frequency-dependent
    strain_to_freq factor.  Coherence is unaffected (it's a ratio).
    """
    out = dict(ci_data)  # shallow copy
    freq = out["freq"]
    factor = _strain_to_freq_factor(freq)  # (Nf,)
    fac = factor[:, None, None]  # broadcast to (Nf, p, p)

    for key in (
        "psd_real_q05",
        "psd_real_q50",
        "psd_real_q95",
        "psd_imag_q05",
        "psd_imag_q50",
        "psd_imag_q95",
        "true_psd_real",
        "true_psd_imag",
    ):
        if key in out:
            out[key] = out[key] * fac

    return out


def _convert_ci_data_to_strain_units(ci_data: dict) -> dict:
    """Convert compact CI data from frequency-fluctuation PSD to strain PSD."""
    out = dict(ci_data)
    freq = out["freq"]
    factor = _strain_to_freq_factor(freq)
    safe_factor = np.where(factor > 0, factor, np.nan)
    fac = safe_factor[:, None, None]

    for key in (
        "psd_real_q05",
        "psd_real_q50",
        "psd_real_q95",
        "psd_imag_q05",
        "psd_imag_q50",
        "psd_imag_q95",
        "true_psd_real",
        "true_psd_imag",
    ):
        if key in out:
            out[key] = out[key] / fac

    return out


def _convert_welch_to_freq_units(
    welch_freq: np.ndarray, welch_S: np.ndarray
) -> np.ndarray:
    """Convert Welch spectral matrix from strain to freq-fluctuation units."""
    factor = _strain_to_freq_factor(welch_freq)
    return welch_S * factor[:, None, None]


def _convert_welch_to_strain_units(
    welch_freq: np.ndarray, welch_S: np.ndarray
) -> np.ndarray:
    """Convert Welch spectral matrix from freq-fluctuation to strain units."""
    factor = _strain_to_freq_factor(welch_freq)
    safe_factor = np.where(factor > 0, factor, np.nan)
    return welch_S / safe_factor[:, None, None]


# ── Welch helper ──────────────────────────────────────────────────────────────


def _welch_psd(
    y_xyz: np.ndarray,
    *,
    Lb: int,
    fs: float,
    overlap: float = WELCH_OVERLAP,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Blocked Welch spectral matrix using ``welch_spectral_matrix_xyz``.

    Returns
    -------
    freq : (Nf,) float
    S    : (Nf, 3, 3) complex spectral matrix
    """
    from log_psplines.example_datasets.lisa_data import (
        spectral_matrix_from_components,
        welch_spectral_matrix_xyz,
    )

    if not (0.0 <= float(overlap) < 1.0):
        raise ValueError("Welch overlap must be in [0, 1).")

    L = max(256, min(int(round(fs / fmin)), int(Lb)))
    n = int(y_xyz.shape[0])
    Nb = max(1, n // int(Lb))
    n_used = Nb * int(Lb)
    y_xyz = np.asarray(y_xyz[:n_used], dtype=np.float64)

    x_blocks = y_xyz[:, 0].reshape(Nb, int(Lb))
    y_blocks = y_xyz[:, 1].reshape(Nb, int(Lb))
    z_blocks = y_xyz[:, 2].reshape(Nb, int(Lb))

    Sxx = Syy = Szz = 0.0
    Sxy = Syz = Szx = 0.0
    freq_ref = None
    delta_t = 1.0 / float(fs)

    for idx in range(Nb):
        freq_block, Sxx_i, Syy_i, Szz_i, Sxy_i, Syz_i, Szx_i = (
            welch_spectral_matrix_xyz(
                x_blocks[idx],
                y_blocks[idx],
                z_blocks[idx],
                L=L,
                delta_t=delta_t,
                overlap=float(overlap),
            )
        )
        if freq_ref is None:
            freq_ref = np.asarray(freq_block, dtype=np.float64)
        Sxx += Sxx_i
        Syy += Syy_i
        Szz += Szz_i
        Sxy += Sxy_i
        Syz += Syz_i
        Szx += Szx_i

    Sxx /= float(Nb)
    Syy /= float(Nb)
    Szz /= float(Nb)
    Sxy /= float(Nb)
    Syz /= float(Nb)
    Szx /= float(Nb)

    mask = (freq_ref >= float(fmin)) & (freq_ref <= float(fmax))
    if not np.any(mask):
        raise ValueError("Welch frequency mask removed all bins.")

    S = spectral_matrix_from_components(
        Sxx[mask], Syy[mask], Szz[mask], Sxy[mask], Syz[mask], Szx[mask]
    )
    return freq_ref[mask], S


def _generate_xyz_for_welch(
    seed: int = 0,
    duration_days: float = 365.0,
    block_days: float = DEFAULT_WELCH_BLOCK_DAYS,
):
    """(Re-)generate XYZ timeseries; returns (y_xyz, Nb, Lb, fs)."""
    from log_psplines.example_datasets.lisatools_backend import (
        ensure_lisatools_backends,
    )

    ensure_lisatools_backends()
    from utils.data import generate_lisa_data

    ts, _freq_true, _S_true, Nb, Lb, dt = generate_lisa_data(
        seed=seed,
        duration_days=duration_days,
        block_days=block_days,
    )
    return ts.y, Nb, Lb, 1.0 / dt


def _generate_xyz_for_overlay(
    seed: int = 0, duration_days: float = 365.0
) -> tuple[np.ndarray, float]:
    """Regenerate XYZ timeseries for a raw empirical overlay."""
    from log_psplines.example_datasets.lisatools_backend import (
        ensure_lisatools_backends,
    )

    ensure_lisatools_backends()
    from utils.data import generate_lisa_data

    ts, _freq_true, _S_true, _Nb, _Lb, dt = generate_lisa_data(
        seed=seed,
        duration_days=duration_days,
        block_days=duration_days,
    )
    return ts.y, dt


def _raw_periodogram_psd(
    y_xyz: np.ndarray,
    *,
    dt: float,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-sided raw XYZ periodogram as a spectral matrix."""
    from log_psplines.example_datasets.lisa_data import (
        spectral_matrix_from_components,
    )

    n = int(y_xyz.shape[0])
    x = np.asarray(y_xyz[:, 0], dtype=np.float64)
    y = np.asarray(y_xyz[:, 1], dtype=np.float64)
    z = np.asarray(y_xyz[:, 2], dtype=np.float64)

    xf = np.fft.rfft(x)
    yf = np.fft.rfft(y)
    zf = np.fft.rfft(z)
    scale = float(dt) / float(n)

    sxx = scale * (np.abs(xf) ** 2)
    syy = scale * (np.abs(yf) ** 2)
    szz = scale * (np.abs(zf) ** 2)
    sxy = scale * (xf * np.conj(yf))
    syz = scale * (yf * np.conj(zf))
    szx = scale * (zf * np.conj(xf))

    if n > 2:
        sxx[1:-1] *= 2.0
        syy[1:-1] *= 2.0
        szz[1:-1] *= 2.0
        sxy[1:-1] *= 2.0
        syz[1:-1] *= 2.0
        szx[1:-1] *= 2.0

    freq = np.fft.rfftfreq(n, d=float(dt))
    if freq.size > 1 and np.isclose(freq[0], 0.0):
        freq = freq[1:]
        sxx = sxx[1:]
        syy = syy[1:]
        szz = szz[1:]
        sxy = sxy[1:]
        syz = syz[1:]
        szx = szx[1:]

    mask = (freq >= float(fmin)) & (freq <= float(fmax))
    if not np.any(mask):
        raise ValueError("Raw periodogram frequency mask removed all bins.")

    return freq[mask], spectral_matrix_from_components(
        sxx[mask], syy[mask], szz[mask], sxy[mask], syz[mask], szx[mask]
    )


# ── CI curve loaders ──────────────────────────────────────────────────────────


def _load_npz(npz_path: str | Path) -> dict:
    data = np.load(str(npz_path), allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files}


def _load_ci_data(run_dir: str | Path) -> dict:
    """Load CI curves from ``inference_data.nc`` when present, else NPZ."""
    run_dir = Path(run_dir)
    idata_path = run_dir / "inference_data.nc"
    npz_path = run_dir / "compact_ci_curves.npz"

    if idata_path.exists():
        try:
            idata = az.from_netcdf(str(idata_path))
            ci_data = get_multivar_ci_summary(idata)
            print(f"Loaded CI curves from {idata_path}")
            return ci_data
        except Exception as exc:
            if not npz_path.exists():
                raise RuntimeError(
                    f"Could not load CI curves from {idata_path}: {exc}"
                ) from exc
            print(
                f"WARNING: failed to read {idata_path} ({exc}); "
                f"falling back to {npz_path}."
            )

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Neither inference_data.nc nor compact_ci_curves.npz found in {run_dir}"
        )

    print(f"Loaded CI curves from {npz_path}")
    return _load_npz(npz_path)


def _load_component_model(
    dataset,
    prefix: str,
    *,
    degree: int,
    diff_matrix_order: int,
) -> LogPSplines:
    """Rehydrate one stored spline component from ``idata.spline_model``."""
    return LogPSplines.from_storage_dataset(
        dataset,
        prefix=prefix,
        degree=degree,
        diffMatrixOrder=diff_matrix_order,
    )


def _load_multivar_spline_model(idata) -> MultivariateLogPSplines:
    """Reconstruct the multivariate spline model from saved inference data."""
    dataset = getattr(idata, "spline_model", None)
    if dataset is None:
        raise KeyError("idata missing spline_model group.")

    degree = int(np.asarray(dataset["degree"].values).item())
    diff_matrix_order = int(
        np.asarray(dataset["diffMatrixOrder"].values).item()
    )
    N = int(np.asarray(dataset["N"].values).item())
    p = int(np.asarray(dataset["p"].values).item())

    diagonal_models = [
        _load_component_model(
            dataset,
            f"diag_{idx}",
            degree=degree,
            diff_matrix_order=diff_matrix_order,
        )
        for idx in range(p)
    ]

    offdiag_re_models = {}
    offdiag_im_models = {}
    if p > 1:
        for key in dataset.data_vars:
            match_re = re.fullmatch(
                r"theta_re_(\d+)_(\d+)_(?:knots|basis)", str(key)
            )
            if match_re is not None:
                j = int(match_re.group(1))
                l = int(match_re.group(2))
                prefix = f"theta_re_{j}_{l}"
                offdiag_re_models[(j, l)] = _load_component_model(
                    dataset,
                    prefix,
                    degree=degree,
                    diff_matrix_order=diff_matrix_order,
                )
                continue

            match_im = re.fullmatch(
                r"theta_im_(\d+)_(\d+)_(?:knots|basis)", str(key)
            )
            if match_im is not None:
                j = int(match_im.group(1))
                l = int(match_im.group(2))
                prefix = f"theta_im_{j}_{l}"
                offdiag_im_models[(j, l)] = _load_component_model(
                    dataset,
                    prefix,
                    degree=degree,
                    diff_matrix_order=diff_matrix_order,
                )

    return MultivariateLogPSplines(
        degree=degree,
        diffMatrixOrder=diff_matrix_order,
        N=N,
        p=p,
        diagonal_models=diagonal_models,
        offdiag_re_models=offdiag_re_models,
        offdiag_im_models=offdiag_im_models,
    )


def _rescale_multivar_ci_from_idata(
    idata,
    ci_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Match the multivariate posterior PSD rescaling used in ``to_arviz``."""
    attrs = getattr(idata, "attrs", {})
    channel_stds = attrs.get("channel_stds")
    if channel_stds is None:
        return ci_data

    stds = np.asarray(channel_stds, dtype=np.float64)
    n_channels = int(np.asarray(ci_data["psd_real_q50"]).shape[-1])
    if stds.shape != (n_channels,):
        raise ValueError(
            "Saved channel_stds shape does not match PSD channel dimension: "
            f"{stds.shape} vs {(n_channels,)}."
        )

    factor = np.outer(stds, stds).astype(np.float64)
    factor_3d = factor[None, :, :]
    rescaled = dict(ci_data)
    for key in (
        "psd_real_q05",
        "psd_real_q50",
        "psd_real_q95",
        "psd_imag_q05",
        "psd_imag_q50",
        "psd_imag_q95",
    ):
        rescaled[key] = np.asarray(ci_data[key], dtype=np.float64) * factor_3d
    return rescaled


def _flatten_chain_draw_samples(values: np.ndarray) -> np.ndarray:
    """Flatten ``(chain, draw, ...)`` arrays into ``(samples, ...)``."""
    arr = np.asarray(values)
    if arr.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions, got {arr.shape}.")
    if arr.ndim == 3:
        return arr
    return arr.reshape((-1,) + tuple(arr.shape[2:]))


def _recompute_ci_from_all_draws(
    run_dir: str | Path,
    *,
    max_draws: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Recompute PSD/coherence quantiles from all saved ``sample_stats`` draws."""
    idata = _load_idata(run_dir)
    if idata is None:
        raise FileNotFoundError(
            f"Missing inference_data.nc in {run_dir}; cannot recompute CI."
        )
    if not hasattr(idata, "sample_stats"):
        raise KeyError("idata missing sample_stats group.")
    sample_stats = idata.sample_stats
    for key in ("log_delta_sq", "theta_re", "theta_im"):
        if key not in sample_stats:
            raise KeyError(f"sample_stats missing '{key}'.")

    spline_model = _load_multivar_spline_model(idata)
    stored = get_multivar_ci_summary(idata)
    log_delta_sq = _flatten_chain_draw_samples(
        np.asarray(sample_stats["log_delta_sq"].values)
    )
    theta_re = _flatten_chain_draw_samples(
        np.asarray(sample_stats["theta_re"].values)
    )
    theta_im = _flatten_chain_draw_samples(
        np.asarray(sample_stats["theta_im"].values)
    )
    total_draws = int(log_delta_sq.shape[0])
    n_draws_max = (
        total_draws
        if not max_draws or max_draws <= 0
        else min(int(max_draws), total_draws)
    )
    psd_real_q, psd_imag_q, coh_q = spline_model.compute_psd_quantiles(
        jnp.asarray(log_delta_sq),
        jnp.asarray(theta_re),
        jnp.asarray(theta_im),
        percentiles=[5.0, 50.0, 95.0],
        n_samples_max=n_draws_max,
        compute_coherence=True,
    )
    recomputed = {
        "freq": np.asarray(stored["freq"], dtype=np.float64),
        "psd_real_q05": np.asarray(psd_real_q[0], dtype=np.float64),
        "psd_real_q50": np.asarray(psd_real_q[1], dtype=np.float64),
        "psd_real_q95": np.asarray(psd_real_q[2], dtype=np.float64),
        "psd_imag_q05": np.asarray(psd_imag_q[0], dtype=np.float64),
        "psd_imag_q50": np.asarray(psd_imag_q[1], dtype=np.float64),
        "psd_imag_q95": np.asarray(psd_imag_q[2], dtype=np.float64),
        "true_psd_real": np.asarray(stored["true_psd_real"], dtype=np.float64),
        "true_psd_imag": np.asarray(stored["true_psd_imag"], dtype=np.float64),
    }
    if coh_q is not None:
        recomputed["coh_q05"] = np.asarray(coh_q[0], dtype=np.float64)
        recomputed["coh_q50"] = np.asarray(coh_q[1], dtype=np.float64)
        recomputed["coh_q95"] = np.asarray(coh_q[2], dtype=np.float64)
    recomputed = _rescale_multivar_ci_from_idata(idata, recomputed)
    meta = {
        "posterior_psd_max_draws_stored": int(
            getattr(idata, "attrs", {}).get("posterior_psd_max_draws", -1)
        ),
        "recomputed_draws_used": int(n_draws_max),
        "recomputed_draws_total": int(total_draws),
    }
    return recomputed, meta


def _compare_ci_summaries(
    stored: dict[str, np.ndarray],
    recomputed: dict[str, np.ndarray],
    *,
    meta: dict[str, float | int] | None = None,
) -> dict[str, object]:
    """Return a compact numeric comparison of stored vs recomputed CI."""
    result: dict[str, object] = dict(meta or {})
    keys = [
        "psd_real_q05",
        "psd_real_q50",
        "psd_real_q95",
        "psd_imag_q05",
        "psd_imag_q50",
        "psd_imag_q95",
    ]
    if "coh_q05" in stored and "coh_q05" in recomputed:
        keys.extend(["coh_q05", "coh_q50", "coh_q95"])

    diffs: dict[str, dict[str, float]] = {}
    for key in keys:
        stored_arr = np.asarray(stored[key], dtype=np.float64)
        recomp_arr = np.asarray(recomputed[key], dtype=np.float64)
        abs_diff = np.abs(recomp_arr - stored_arr)
        denom = np.maximum(np.abs(recomp_arr), 1e-300)
        rel_diff = abs_diff / denom
        diffs[key] = {
            "max_abs_diff": float(np.max(abs_diff)),
            "median_abs_diff": float(np.median(abs_diff)),
            "max_rel_diff": float(np.max(rel_diff)),
            "median_rel_diff": float(np.median(rel_diff)),
        }
    result["array_diffs"] = diffs

    stored_metrics = _compute_compare_metrics(stored)
    full_metrics = _compute_compare_metrics(recomputed)
    result["stored_metrics"] = stored_metrics
    result["recomputed_metrics"] = full_metrics
    result["metric_deltas"] = {
        key: float(
            full_metrics.get(key, np.nan) - stored_metrics.get(key, np.nan)
        )
        for key in stored_metrics.keys()
        if key in full_metrics
    }
    return result


def _save_json(data: dict[str, object], outpath: str | Path) -> None:
    """Write JSON diagnostics to disk."""
    outpath = Path(outpath)
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    print(f"Saved JSON: {outpath}")


def _load_idata(run_dir: str | Path) -> az.InferenceData | None:
    """Load ``inference_data.nc`` from a run directory when available."""
    run_dir = Path(run_dir)
    idata_path = run_dir / "inference_data.nc"
    if not idata_path.exists():
        return None
    return az.from_netcdf(str(idata_path))


def _default_run_x_dir() -> Path:
    """Return the preferred default run_x directory for paper plotting."""
    preferred = (
        HERE
        / "runs"
        / "run_x_d2_k48_uniform_no_excision"
        / "k48_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff"
        / "seed_0"
    )
    if (preferred / "inference_data.nc").exists():
        return preferred

    base = HERE / "runs" / "run_x_d2_k48_uniform_no_excision"
    if base.exists():
        candidates = sorted(base.glob("*/seed_0"))
        with_idata = [
            path
            for path in candidates
            if (path / "inference_data.nc").exists()
        ]
        if with_idata:
            return with_idata[-1]
        if candidates:
            return candidates[-1]

    return preferred


def _load_raw_periodogram(
    run_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load raw observed periodogram from saved idata."""
    idata = _load_idata(run_dir)
    if idata is None:
        return None
    if (
        "observed_data" not in idata
        or "periodogram" not in idata["observed_data"]
    ):
        return None
    periodogram = idata["observed_data"]["periodogram"]
    freq = np.asarray(periodogram.coords["freq"].values, dtype=np.float64)
    psd = np.asarray(periodogram.values, dtype=np.complex128)
    return freq, psd


def _select_log_spaced_indices(
    freq: np.ndarray, max_points: int
) -> np.ndarray:
    """Select indices approximately evenly spaced in log-frequency."""
    freq = np.asarray(freq, dtype=np.float64)
    n = int(freq.size)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)

    pos_mask = freq > 0.0
    if not np.any(pos_mask):
        return np.unique(
            np.linspace(0, n - 1, num=max_points, dtype=int, endpoint=True)
        )

    pos_idx = np.flatnonzero(pos_mask)
    log_freq = np.log10(freq[pos_mask])
    targets = np.linspace(log_freq[0], log_freq[-1], num=max_points)
    nearest = np.searchsorted(log_freq, targets, side="left")
    nearest = np.clip(nearest, 0, log_freq.size - 1)
    return np.unique(pos_idx[nearest])


def _thin_overlay_for_plot(
    freq: np.ndarray,
    psd_matrix: np.ndarray,
    *,
    max_points: int = RAW_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin a dense overlay for visibility without changing the stored data."""
    idx = _select_log_spaced_indices(freq, max_points)
    return np.asarray(freq[idx], dtype=np.float64), np.asarray(
        psd_matrix[idx], dtype=np.complex128
    )


# ── dip mask helper ───────────────────────────────────────────────────────────


def _dip_mask(true_diag: np.ndarray) -> np.ndarray:
    """Boolean mask: True at frequency bins in the noise 'dip' region.

    true_diag : (Nf,) real, diagonal auto-spectrum (always positive).
    """
    peak = np.max(true_diag)
    return true_diag / peak < DIP_THRESHOLD


# ── single-panel plot helpers ─────────────────────────────────────────────────


def _plot_diag_panel(
    ax: plt.Axes,
    freq: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    true_psd: np.ndarray,
    overlay_freq: Optional[np.ndarray] = None,
    overlay_psd: Optional[np.ndarray] = None,
    overlay_color: str = WELCH_COLOR,
    overlay_lw: float = WELCH_LW,
    overlay_alpha: float = WELCH_ALPHA,
    overlay_label: str = WELCH_LABEL,
    overlay_marker: str | None = None,
    overlay_markersize: float | None = None,
    overlay_zorder: float = 4,
    show_posterior: bool = True,
    show_median: bool = True,
    show_truth: bool = True,
    *,
    channel_label: str = "",
) -> None:
    """Auto-spectrum (diagonal) panel: log-log scale."""
    if show_posterior:
        ax.fill_between(
            freq, q05, q95, color=CI_COLOR, alpha=CI_ALPHA, zorder=2
        )
        if show_median:
            ax.plot(freq, q50, color=MEDIAN_COLOR, lw=MEDIAN_LW, zorder=3)
    if show_truth:
        ax.plot(
            freq,
            true_psd,
            color=TRUE_COLOR,
            lw=TRUE_LW,
            zorder=5,
            label=TRUE_LABEL,
            ls="--",
        )
    if overlay_freq is not None and overlay_psd is not None:
        plot_kwargs = dict(
            color=overlay_color,
            lw=overlay_lw,
            alpha=overlay_alpha,
            zorder=overlay_zorder,
            label=overlay_label,
        )
        if overlay_marker is not None:
            plot_kwargs["marker"] = overlay_marker
            plot_kwargs["markersize"] = (
                RAW_MARKERSIZE
                if overlay_markersize is None
                else overlay_markersize
            )
            plot_kwargs["linestyle"] = "None"
        ax.plot(overlay_freq, overlay_psd, **plot_kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(FMIN, FMAX)
    # Set sensible y-limits: show the full posterior+truth range but don't let
    # transfer-null spikes (which can drop 20+ OOM) stretch the axis.
    pos_q = q50[q50 > 0]
    pos_t = true_psd[true_psd > 0]
    pos_overlay = None
    if overlay_psd is not None:
        pos_overlay = overlay_psd[overlay_psd > 0]
    if pos_q.size > 0 and pos_t.size > 0:
        ylo = min(pos_q.min(), pos_t.min()) * 0.3
        yhi = max(pos_q.max(), pos_t.max()) * 5.0
        if pos_overlay is not None and pos_overlay.size > 0:
            ylo = min(ylo, pos_overlay.min() * 0.8)
            yhi = max(yhi, np.percentile(pos_overlay, 99.5) * 1.2)
        ax.set_ylim(ylo, yhi)
    if channel_label:
        ax.text(
            0.04,
            0.93,
            f"$S_{{{channel_label}{channel_label}}}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            weight="bold",
        )


def _compute_coherence(
    psd_real: np.ndarray,
    psd_imag: np.ndarray,
    i: int,
    j: int,
) -> np.ndarray:
    """Compute coherence |S_ij|^2 / (S_ii * S_jj) from real+imag PSD matrices."""
    S_ij = psd_real[:, i, j] + 1j * psd_imag[:, i, j]
    S_ii = psd_real[:, i, i]
    S_jj = psd_real[:, j, j]
    denom = S_ii * S_jj
    safe_denom = np.where(denom > 0, denom, np.nan)
    return np.abs(S_ij) ** 2 / safe_denom


def _plot_coherence_panel(
    ax: plt.Axes,
    freq: np.ndarray,
    coh_q50: np.ndarray,
    coh_true: np.ndarray,
    overlay_freq: Optional[np.ndarray] = None,
    overlay_coh: Optional[np.ndarray] = None,
    overlay_color: str = WELCH_COLOR,
    overlay_lw: float = WELCH_LW,
    overlay_alpha: float = WELCH_ALPHA,
    overlay_marker: str | None = None,
    overlay_markersize: float | None = None,
    overlay_zorder: float = 4,
    show_posterior: bool = True,
    show_median: bool = True,
    show_truth: bool = True,
    *,
    ch_i: str = "",
    ch_j: str = "",
) -> None:
    """Off-diagonal coherence panel: |S_ij|^2 / (S_ii * S_jj)."""
    if overlay_freq is not None and overlay_coh is not None:
        plot_kwargs = dict(
            color=overlay_color,
            lw=overlay_lw,
            alpha=overlay_alpha,
            zorder=overlay_zorder,
        )
        if overlay_marker is not None:
            plot_kwargs["marker"] = overlay_marker
            plot_kwargs["markersize"] = (
                RAW_MARKERSIZE
                if overlay_markersize is None
                else overlay_markersize
            )
            plot_kwargs["linestyle"] = "None"
        ax.plot(overlay_freq, overlay_coh, **plot_kwargs)
    if show_posterior and show_median:
        ax.plot(freq, coh_q50, color=MEDIAN_COLOR, lw=MEDIAN_LW, zorder=3)
    if show_truth:
        ax.plot(
            freq,
            coh_true,
            color=TRUE_COLOR,
            lw=TRUE_LW,
            zorder=5,
            ls="--",
        )
    ax.set_xscale("log")
    ax.set_xlim(FMIN, FMAX)
    finite_arrays = [coh_true[np.isfinite(coh_true)]]
    if show_posterior:
        finite_arrays.append(coh_q50[np.isfinite(coh_q50)])
    if overlay_coh is not None:
        finite_arrays.append(overlay_coh[np.isfinite(overlay_coh)])

    finite_arrays = [arr for arr in finite_arrays if arr.size > 0]
    if finite_arrays:
        ymax = float(max(np.max(arr) for arr in finite_arrays))
        ymax = min(max(0.02, 1.15 * ymax), 1.05)
        ymin = -0.03 * ymax
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(-0.05, 1.05)
    if ch_i and ch_j:
        ax.text(
            0.04,
            0.93,
            f"$C_{{{ch_i}{ch_j}}}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            weight="bold",
        )


def _hide_upper_triangle(axes: np.ndarray) -> None:
    p = axes.shape[0]
    for i in range(p):
        for j in range(p):
            if j > i:
                axes[i, j].set_visible(False)


def _add_axis_labels(axes: np.ndarray, labels: list[str]) -> None:
    p = len(labels)
    for i in range(p):
        for j in range(p):
            if not axes[i, j].get_visible():
                continue
            if i == p - 1:
                axes[i, j].set_xlabel("Frequency [Hz]", fontsize=9)
            if j == 0:
                if i == j:
                    axes[i, j].set_ylabel("PSD [1/Hz]", fontsize=9)
                else:
                    axes[i, j].set_ylabel("Coherence", fontsize=9)


# ── relative error figure ─────────────────────────────────────────────────────


def _make_relative_error_figure(
    ci_data: dict,
    *,
    channel_labels: list[str],
    title: str = "",
    outpath: str | Path,
) -> None:
    """Diagonal-only relative error: (median - truth) / truth vs. frequency.

    Bins in the "dip" (low-noise) region are hatched.
    """
    freq = ci_data["freq"]
    p = 3
    fig, axes = plt.subplots(
        1, p, figsize=(10, 3.2), sharey=False, constrained_layout=True
    )
    if title:
        fig.suptitle(title, fontsize=12)

    for k in range(p):
        ax = axes[k]
        q50 = ci_data["psd_real_q50"][:, k, k]
        q05 = ci_data["psd_real_q05"][:, k, k]
        q95 = ci_data["psd_real_q95"][:, k, k]
        truth = ci_data["true_psd_real"][:, k, k]

        # guard against zero truth
        safe_truth = np.where(truth > 0, truth, np.nan)
        rel_med = (q50 - safe_truth) / safe_truth
        rel_lo = (q05 - safe_truth) / safe_truth
        rel_hi = (q95 - safe_truth) / safe_truth

        ax.fill_between(freq, rel_lo, rel_hi, color=CI_COLOR, alpha=CI_ALPHA)
        ax.plot(freq, rel_med, color=RELERR_COLOR, lw=1.4)
        ax.axhline(0, color=RELERR_ZERO_COLOR, lw=1.0, ls="--", zorder=5)

        # Hatch the dip region
        mask = _dip_mask(
            np.where(
                np.isnan(safe_truth),
                (
                    np.nanmin(safe_truth[safe_truth > 0])
                    if np.any(safe_truth > 0)
                    else 1.0
                ),
                safe_truth,
            )
        )
        if np.any(mask):
            ymin, ymax = ax.get_ylim() if ax.get_ylim() != (0, 1) else (-1, 1)
            # use axvspan for each contiguous masked region
            _shade_mask_regions(
                ax,
                freq,
                mask,
                color=HATCH_COLOR,
                alpha=HATCH_ALPHA,
                hatch=HATCH_PATTERN,
            )

        ax.set_xscale("log")
        ax.set_xlim(FMIN, FMAX)
        ax.set_xlabel("Frequency [Hz]", fontsize=9)
        ax.set_title(f"${channel_labels[k]}{channel_labels[k]}$", fontsize=11)
        if k == 0:
            ax.set_ylabel(
                r"$(S_{50} - S_\mathrm{true})\,/\,S_\mathrm{true}$", fontsize=9
            )
        ax.tick_params(labelsize=8)
        # symmetric y-limits around zero
        ylim = np.nanmax(
            np.abs([rel_lo[np.isfinite(rel_lo)], rel_hi[np.isfinite(rel_hi)]])
        )
        ylim = min(ylim * 1.1, 2.0)
        ax.set_ylim(-ylim, ylim)

    fig.savefig(str(outpath), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved relative error figure: {outpath}")


def _make_duration_residual_figure(
    compare_specs: list[dict],
    *,
    channel_labels: list[str],
    title: str = "",
    outpath: str | Path,
) -> None:
    """2×3 residual figure overlaying multiple durations.

    Each duration contributes one CI band (shaded) + median line, all centred
    near zero.  Top row: PSD relative error.  Bottom row: coherence residual.

    Args:
        compare_specs: list of dicts with keys ``label``, ``color``,
            ``ci_data`` (as returned by the duration-comparison loop).
    """
    offdiag = [(1, 0), (2, 0), (2, 1)]

    fig, axes = plt.subplots(
        2, 3, figsize=(11, 6), sharex=True, constrained_layout=True
    )
    if title:
        fig.suptitle(title, fontsize=12)

    # PSD y-limit: hard-cap at ±PSD_RELERR_YLIM so transfer-null spikes don't
    # compress the well-recovered smooth band.  Null spikes simply go off-axis.
    PSD_RELERR_YLIM = 0.15

    # Collect all relative-error values to set shared y-limits from the data.
    psd_rel_all: list[np.ndarray] = []
    coh_res_all: list[list[np.ndarray]] = [[] for _ in offdiag]

    for spec in compare_specs:
        ci = spec["ci_data"]
        col = spec["color"]
        lbl = spec["label"]
        freq = ci["freq"]

        # ── Top row: PSD relative error ───────────────────────────────────
        for k in range(3):
            ax = axes[0, k]
            q50 = ci["psd_real_q50"][:, k, k]
            q05 = ci["psd_real_q05"][:, k, k]
            q95 = ci["psd_real_q95"][:, k, k]
            truth = ci["true_psd_real"][:, k, k]
            safe = np.where(truth > 0, truth, np.nan)

            rel_lo = (q05 - safe) / safe
            rel_hi = (q95 - safe) / safe
            rel_med = (q50 - safe) / safe

            ax.fill_between(freq, rel_lo, rel_hi, color=col, alpha=0.35)
            ax.plot(freq, rel_med, color=col, lw=1.3, label=lbl)
            psd_rel_all.extend(
                [rel_lo[np.isfinite(rel_lo)], rel_hi[np.isfinite(rel_hi)]]
            )

        # ── Bottom row: coherence residual ────────────────────────────────
        for col_idx, (i, j) in enumerate(offdiag):
            ax = axes[1, col_idx]
            coh_q50 = _compute_coherence(
                ci["psd_real_q50"], ci["psd_imag_q50"], i, j
            )
            coh_q05 = _compute_coherence(
                ci["psd_real_q05"], ci["psd_imag_q05"], i, j
            )
            coh_q95 = _compute_coherence(
                ci["psd_real_q95"], ci["psd_imag_q95"], i, j
            )
            coh_true = _compute_coherence(
                ci["true_psd_real"], ci["true_psd_imag"], i, j
            )

            res_lo = coh_q05 - coh_true
            res_hi = coh_q95 - coh_true
            res_med = coh_q50 - coh_true

            ax.fill_between(
                freq, res_lo, res_hi, color=spec["color"], alpha=0.35
            )
            ax.plot(freq, res_med, color=spec["color"], lw=1.3)
            coh_res_all[col_idx].extend(
                [res_lo[np.isfinite(res_lo)], res_hi[np.isfinite(res_hi)]]
            )

    ylim_psd = PSD_RELERR_YLIM

    # Shared formatting
    for k in range(3):
        ax = axes[0, k]
        ax.axhline(0, color="k", lw=1.0, ls="--", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(FMIN, FMAX)
        ch = channel_labels[k]
        ax.set_title(f"$S_{{{ch}{ch}}}$", fontsize=11)
        if k == 0:
            ax.set_ylabel(
                r"$(q_{50} - S_\mathrm{true})\,/\,S_\mathrm{true}$", fontsize=9
            )
        ax.tick_params(labelsize=8)
        ax.set_ylim(-ylim_psd, ylim_psd)

    for col_idx, (i, j) in enumerate(offdiag):
        ax = axes[1, col_idx]
        ax.axhline(0, color="k", lw=1.0, ls="--", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(FMIN, FMAX)
        ax.set_xlabel("Frequency [Hz]", fontsize=9)
        ci_label = channel_labels[i]
        cj_label = channel_labels[j]
        ax.set_title(f"$C_{{{ci_label}{cj_label}}}$", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(r"$C_{50} - C_\mathrm{true}$", fontsize=9)
        ax.tick_params(labelsize=8)
        # Auto y-limits from data
        if coh_res_all[col_idx]:
            all_c = np.concatenate(coh_res_all[col_idx])
            lo_c, hi_c = np.nanpercentile(all_c, [1, 99])
            pad_c = max(abs(lo_c), abs(hi_c)) * 0.15
            ylim_c = min(max(abs(lo_c), abs(hi_c)) + pad_c, 0.3)
            ax.set_ylim(-ylim_c, ylim_c)

    # Legend from the top-left PSD panel (median lines carry labels)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, fontsize=8, loc="upper left")

    fig.savefig(str(outpath), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved duration residual figure: {outpath}")


def _make_combined_residual_figure(
    ci_data: dict,
    *,
    channel_labels: list[str],
    title: str = "",
    outpath: str | Path,
) -> None:
    """Combined 2×3 residual figure: PSD relative error (top) + coherence residual (bottom).

    Top row   : (S_ii,50 - S_ii,true) / S_ii,true  for i = X, Y, Z
    Bottom row : C_ij,50 - C_ij,true               for (j,i) = (Y,X), (Z,X), (Z,Y)

    The 90% CI band is shaded around each median residual.
    Transfer-null dip regions are hatched on each panel.
    """
    freq = ci_data["freq"]
    p = 3
    offdiag = [(1, 0), (2, 0), (2, 1)]  # (i, j) lower-triangle pairs

    fig, axes = plt.subplots(
        2, p, figsize=(11, 6), sharex=True, constrained_layout=True
    )
    if title:
        fig.suptitle(title, fontsize=12)

    # ── Top row: PSD relative error ───────────────────────────────────────────
    for k in range(p):
        ax = axes[0, k]
        q50 = ci_data["psd_real_q50"][:, k, k]
        q05 = ci_data["psd_real_q05"][:, k, k]
        q95 = ci_data["psd_real_q95"][:, k, k]
        truth = ci_data["true_psd_real"][:, k, k]

        safe_truth = np.where(truth > 0, truth, np.nan)
        rel_med = (q50 - safe_truth) / safe_truth
        rel_lo = (q05 - safe_truth) / safe_truth
        rel_hi = (q95 - safe_truth) / safe_truth

        ax.fill_between(freq, rel_lo, rel_hi, color=CI_COLOR, alpha=CI_ALPHA)
        ax.plot(freq, rel_med, color=RELERR_COLOR, lw=1.4)
        ax.axhline(0, color=RELERR_ZERO_COLOR, lw=1.0, ls="--", zorder=5)

        dip_mask = _dip_mask(
            np.where(
                np.isnan(safe_truth),
                (
                    np.nanmin(safe_truth[np.isfinite(safe_truth)])
                    if np.any(np.isfinite(safe_truth))
                    else 1.0
                ),
                safe_truth,
            )
        )
        if np.any(dip_mask):
            _shade_mask_regions(
                ax,
                freq,
                dip_mask,
                color=HATCH_COLOR,
                alpha=HATCH_ALPHA,
                hatch=HATCH_PATTERN,
            )

        ax.set_xscale("log")
        ax.set_xlim(FMIN, FMAX)
        ch = channel_labels[k]
        ax.set_title(f"$S_{{{ch}{ch}}}$", fontsize=11)
        if k == 0:
            ax.set_ylabel(
                r"$(q_{50} - S_\mathrm{true})\,/\,S_\mathrm{true}$", fontsize=9
            )
        ax.tick_params(labelsize=8)

        fin_lo = rel_lo[np.isfinite(rel_lo)]
        fin_hi = rel_hi[np.isfinite(rel_hi)]
        if fin_lo.size and fin_hi.size:
            ylim = min(
                max(np.abs(fin_lo).max(), np.abs(fin_hi).max()) * 1.1, 2.0
            )
            ax.set_ylim(-ylim, ylim)

    # ── Bottom row: coherence residual ────────────────────────────────────────
    for col, (i, j) in enumerate(offdiag):
        ax = axes[1, col]

        coh_q50 = _compute_coherence(
            ci_data["psd_real_q50"], ci_data["psd_imag_q50"], i, j
        )
        coh_q05 = _compute_coherence(
            ci_data["psd_real_q05"], ci_data["psd_imag_q05"], i, j
        )
        coh_q95 = _compute_coherence(
            ci_data["psd_real_q95"], ci_data["psd_imag_q95"], i, j
        )
        coh_true = _compute_coherence(
            ci_data["true_psd_real"], ci_data["true_psd_imag"], i, j
        )

        res_med = coh_q50 - coh_true
        res_lo = coh_q05 - coh_true
        res_hi = coh_q95 - coh_true

        ax.fill_between(freq, res_lo, res_hi, color=CI_COLOR, alpha=CI_ALPHA)
        ax.plot(freq, res_med, color=RELERR_COLOR, lw=1.4)
        ax.axhline(0, color=RELERR_ZERO_COLOR, lw=1.0, ls="--", zorder=5)

        # Hatch where true coherence is near zero (same dip concept)
        dip_mask = coh_true < 0.02
        if np.any(dip_mask):
            _shade_mask_regions(
                ax,
                freq,
                dip_mask,
                color=HATCH_COLOR,
                alpha=HATCH_ALPHA,
                hatch=HATCH_PATTERN,
            )

        ax.set_xscale("log")
        ax.set_xlim(FMIN, FMAX)
        ax.set_xlabel("Frequency [Hz]", fontsize=9)
        ci = channel_labels[i]
        cj = channel_labels[j]
        ax.set_title(f"$C_{{{ci}{cj}}}$", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"$C_{50} - C_\mathrm{true}$", fontsize=9)
        ax.tick_params(labelsize=8)

        fin_lo = res_lo[np.isfinite(res_lo)]
        fin_hi = res_hi[np.isfinite(res_hi)]
        if fin_lo.size and fin_hi.size:
            ylim = min(
                max(np.abs(fin_lo).max(), np.abs(fin_hi).max()) * 1.1, 0.5
            )
            ax.set_ylim(-ylim, ylim)

    fig.savefig(str(outpath), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined residual figure: {outpath}")


def _shade_mask_regions(
    ax: plt.Axes,
    freq: np.ndarray,
    mask: np.ndarray,
    color: str,
    alpha: float,
    hatch: str,
) -> None:
    """Add hatched axvspans for each contiguous True region in mask."""
    in_region = False
    f_start = None
    ymin, ymax = ax.get_ylim() if ax.get_ylim() != (0.0, 1.0) else (-2.0, 2.0)
    for idx, (f, m) in enumerate(zip(freq, mask)):
        if m and not in_region:
            f_start = f
            in_region = True
        elif not m and in_region:
            ax.axvspan(
                f_start,
                freq[idx - 1],
                color=color,
                alpha=alpha,
                hatch=hatch,
                linewidth=0,
                zorder=0,
            )
            in_region = False
    if in_region:
        ax.axvspan(
            f_start,
            freq[-1],
            color=color,
            alpha=alpha,
            hatch=hatch,
            linewidth=0,
            zorder=0,
        )


# ── main PSD matrix figure ────────────────────────────────────────────────────


def _make_psd_matrix_figure(
    ci_data: dict,
    *,
    channel_labels: list[str],
    overlay_freq: Optional[np.ndarray] = None,
    overlay_S: Optional[np.ndarray] = None,
    coherence_overlay_freq: Optional[np.ndarray] = None,
    coherence_overlay_S: Optional[np.ndarray] = None,
    overlay_label: str = WELCH_LABEL,
    overlay_color: str = WELCH_COLOR,
    overlay_lw: float = WELCH_LW,
    overlay_alpha: float = WELCH_ALPHA,
    overlay_marker: str | None = None,
    overlay_markersize: float | None = None,
    overlay_zorder: float = 4,
    show_overlay_on_coherence: bool = True,
    show_posterior: bool = True,
    show_median: bool = True,
    show_truth: bool = True,
    title: str = "",
    outpath: str | Path,
    psd_unit: str = PSD_UNIT_STRAIN,
) -> None:
    """3×3 PSD matrix figure.

    Layout:
        diagonal (i==j) : auto-spectrum, log-log
        lower triangle (i>j) : coherence, log-x lin-y
        upper triangle hidden
    """
    p = 3
    fig, axes = plt.subplots(
        p, p, figsize=(10, 8.5), constrained_layout=True, squeeze=False
    )
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    freq = ci_data["freq"]
    if coherence_overlay_freq is None:
        coherence_overlay_freq = overlay_freq
    if coherence_overlay_S is None:
        coherence_overlay_S = overlay_S

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                q05 = ci_data["psd_real_q05"][:, i, i]
                q50 = ci_data["psd_real_q50"][:, i, i]
                q95 = ci_data["psd_real_q95"][:, i, i]
                true = ci_data["true_psd_real"][:, i, i]

                overlay_f_diag = overlay_psd_diag = None
                if overlay_freq is not None and overlay_S is not None:
                    overlay_f_diag = overlay_freq
                    overlay_psd_diag = overlay_S[:, i, i].real

                _plot_diag_panel(
                    ax,
                    freq,
                    q05,
                    q50,
                    q95,
                    true,
                    overlay_f_diag,
                    overlay_psd_diag,
                    overlay_color=overlay_color,
                    overlay_lw=overlay_lw,
                    overlay_alpha=overlay_alpha,
                    overlay_label=overlay_label,
                    overlay_marker=overlay_marker,
                    overlay_markersize=overlay_markersize,
                    overlay_zorder=overlay_zorder,
                    show_posterior=show_posterior,
                    show_median=show_median,
                    show_truth=show_truth,
                    channel_label=channel_labels[i],
                )
            else:
                # lower triangle: coherence
                coh_q50 = _compute_coherence(
                    ci_data["psd_real_q50"], ci_data["psd_imag_q50"], i, j
                )
                coh_true = _compute_coherence(
                    ci_data["true_psd_real"], ci_data["true_psd_imag"], i, j
                )

                # Welch coherence
                overlay_f_coh = overlay_coh_ij = None
                if (
                    show_overlay_on_coherence
                    and coherence_overlay_freq is not None
                    and coherence_overlay_S is not None
                ):
                    overlay_f_coh = coherence_overlay_freq
                    S_ij_w = coherence_overlay_S[:, i, j]
                    S_ii_w = coherence_overlay_S[:, i, i].real
                    S_jj_w = coherence_overlay_S[:, j, j].real
                    denom_w = S_ii_w * S_jj_w
                    safe_w = np.where(denom_w > 0, denom_w, np.nan)
                    overlay_coh_ij = np.abs(S_ij_w) ** 2 / safe_w

                _plot_coherence_panel(
                    ax,
                    freq,
                    coh_q50,
                    coh_true,
                    overlay_f_coh,
                    overlay_coh_ij,
                    overlay_color=overlay_color,
                    overlay_lw=overlay_lw,
                    overlay_alpha=overlay_alpha,
                    overlay_marker=overlay_marker,
                    overlay_markersize=overlay_markersize,
                    overlay_zorder=overlay_zorder,
                    show_posterior=show_posterior,
                    show_median=show_median,
                    show_truth=show_truth,
                    ch_i=channel_labels[i],
                    ch_j=channel_labels[j],
                )

            # shared formatting
            ax.tick_params(labelsize=8)
            if i < p - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Frequency [Hz]", fontsize=9)
            if i == j:
                ax.set_ylabel(f"PSD [{psd_unit}]", fontsize=9)
            else:
                ax.set_ylabel("Coherence", fontsize=9)

    # Legend on top-left diagonal panel
    legend_elements = []
    if show_posterior:
        legend_elements.append(
            mpatches.Patch(facecolor=CI_COLOR, alpha=0.4, label="90% CI")
        )
        if show_median:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=MEDIAN_COLOR,
                    lw=MEDIAN_LW,
                    label="Posterior median",
                )
            )
    if show_truth:
        legend_elements.append(
            Line2D([0], [0], color=TRUE_COLOR, lw=TRUE_LW, label=TRUE_LABEL)
        )
    if overlay_freq is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=overlay_color,
                lw=overlay_lw,
                alpha=overlay_alpha,
                label=overlay_label,
            )
        )
    axes[0, 0].legend(handles=legend_elements, fontsize=7.5, loc="lower left")

    fig.savefig(str(outpath), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PSD matrix figure: {outpath}")


def _load_compact_run_summary(run_dir: str | Path) -> dict:
    """Load compact scalar metrics for a run when available."""
    summary_path = Path(run_dir) / "compact_run_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, encoding="utf-8") as fh:
        return json.load(fh)


def _compute_compare_metrics(ci_data: dict) -> dict[str, float]:
    """Compute metrics directly from compact CI curves on the plotting grid."""
    freq = np.asarray(ci_data["freq"], dtype=np.float64)
    q05_real = np.asarray(ci_data["psd_real_q05"], dtype=np.float64)
    q50_real = np.asarray(ci_data["psd_real_q50"], dtype=np.float64)
    q95_real = np.asarray(ci_data["psd_real_q95"], dtype=np.float64)
    q05_imag = np.asarray(ci_data["psd_imag_q05"], dtype=np.float64)
    q50_imag = np.asarray(ci_data["psd_imag_q50"], dtype=np.float64)
    q95_imag = np.asarray(ci_data["psd_imag_q95"], dtype=np.float64)
    true_real = np.asarray(ci_data["true_psd_real"], dtype=np.float64)
    true_imag = np.asarray(ci_data["true_psd_imag"], dtype=np.float64)

    diag_mask = np.eye(q05_real.shape[1], dtype=bool)
    true_psd = true_real + 1j * true_imag
    percentiles_stack = np.stack(
        [
            q05_real + 1j * q05_imag,
            q50_real + 1j * q50_imag,
            q95_real + 1j * q95_imag,
        ],
        axis=0,
    )

    return {
        "riae_matrix": float(compute_matrix_riae(q50_real, true_real, freq)),
        "l2_matrix": float(compute_matrix_l2(q50_real, true_real, freq)),
        "coverage": float(
            compute_ci_coverage_multivar(percentiles_stack, true_psd)
        ),
        "ciw_psd_diag_mean": float(
            np.mean((q95_real - q05_real)[:, diag_mask])
        ),
    }


def _compute_trimmed_sample_count(
    duration_days: float,
    *,
    block_days: float,
    dt: float = DELTA_T,
) -> tuple[int, int]:
    """Return (n_used, Nb) after block trimming for the run setup."""
    duration = float(duration_days) * SEC_IN_DAY
    n_total = int(duration / float(dt))
    Lb = int(round(float(block_days) * SEC_IN_DAY / float(dt)))
    Lb = min(Lb, n_total)
    Nb = max(1, n_total // Lb)
    return int(Nb * Lb), int(Nb)


def _format_runtime(seconds: float | None) -> str:
    """Format runtime in seconds for Markdown output."""
    if seconds is None or not np.isfinite(float(seconds)):
        return "-"
    return f"{float(seconds):.1f}"


def _format_metric(value: float | None, *, sci: bool = False) -> str:
    """Format scalar metrics for Markdown output."""
    if value is None or not np.isfinite(float(value)):
        return "-"
    if sci:
        return f"{float(value):.3e}"
    return f"{float(value):.4f}"


def _write_duration_comparison_readme(
    run_specs: list[dict],
    *,
    outpath: str | Path,
    figure_name: str,
    overlay_label: str,
    psd_unit: str,
    block_days: float,
) -> None:
    """Write a Markdown summary table for the duration comparison runs."""
    lines = [
        "# Run X Duration Comparison",
        "",
        f"Figure: `{figure_name}`",
        "",
        f"Data overlay: {overlay_label}",
        "",
        f"PSD unit: `{psd_unit}`",
        "",
        "Notes:",
        f"- `n` is the block-trimmed sample count actually used in inference with `{block_days:g}`-day blocks.",
        "- `ESS` and `runtime` are read from `compact_run_summary.json`.",
        "- `RIAE`, `L2`, `coverage`, and `CI width` are computed from the plotted CI curves on the saved frequency grid.",
        "- `CI width` is the mean diagonal 90% interval width.",
        "",
        "| Run | Nominal duration (days) | n | ESS median | Runtime (s) | RIAE | L2 | Coverage | CI width |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for spec in run_specs:
        summary = spec["summary"]
        metrics = spec["metrics"]
        duration_days = float(summary.get("duration_days", np.nan))
        n_used, _ = _compute_trimmed_sample_count(
            duration_days, block_days=block_days
        )
        ess_median = summary.get("ess_median")
        runtime = summary.get("runtime")
        coverage = metrics.get("coverage")
        coverage_str = (
            f"{100.0 * float(coverage):.1f}%"
            if coverage is not None and np.isfinite(float(coverage))
            else "-"
        )
        ess_str = (
            f"{float(ess_median):,.0f}"
            if ess_median is not None and np.isfinite(float(ess_median))
            else "-"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    spec["label"],
                    f"{duration_days:g}",
                    f"{n_used:,}",
                    ess_str,
                    _format_runtime(runtime),
                    _format_metric(metrics.get("riae_matrix")),
                    _format_metric(metrics.get("l2_matrix")),
                    coverage_str,
                    _format_metric(metrics.get("ciw_psd_diag_mean"), sci=True),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Run Paths",
            "",
        ]
    )
    for spec in run_specs:
        lines.append(f"- `{spec['label']}`: `{spec['run_dir']}`")

    outpath = Path(outpath)
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved duration comparison README: {outpath}")


def _make_multirun_psd_matrix_figure(
    run_specs: list[dict],
    *,
    channel_labels: list[str],
    overlay_freq: Optional[np.ndarray] = None,
    overlay_S: Optional[np.ndarray] = None,
    coherence_overlay_freq: Optional[np.ndarray] = None,
    coherence_overlay_S: Optional[np.ndarray] = None,
    overlay_label: str = WELCH_LABEL,
    overlay_color: str = WELCH_COLOR,
    overlay_lw: float = WELCH_LW,
    overlay_alpha: float = WELCH_ALPHA,
    overlay_marker: str | None = None,
    overlay_markersize: float | None = None,
    overlay_zorder: float = 4,
    show_overlay_on_coherence: bool = True,
    show_posterior: bool = True,
    show_median: bool = True,
    show_truth: bool = True,
    title: str = "",
    outpath: str | Path = "",
    psd_unit: str = PSD_UNIT_STRAIN,
) -> None:
    """Create a single XYZ PSD matrix figure with multiple posterior overlays."""
    if not run_specs:
        raise ValueError("run_specs must be non-empty.")

    p = 3
    fig, axes = plt.subplots(
        p, p, figsize=(10.5, 8.8), constrained_layout=True, squeeze=False
    )
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    truth_real = run_specs[0]["ci_data"]["true_psd_real"]
    truth_imag = run_specs[0]["ci_data"]["true_psd_imag"]

    if coherence_overlay_freq is None:
        coherence_overlay_freq = overlay_freq
    if coherence_overlay_S is None:
        coherence_overlay_S = overlay_S

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                true_diag = truth_real[:, i, i]
                pos_values = [true_diag[true_diag > 0]]
                if overlay_S is not None:
                    overlay_diag = overlay_S[:, i, i].real
                    pos_values.append(overlay_diag[overlay_diag > 0])
                else:
                    overlay_diag = None

                for spec in run_specs:
                    ci_data = spec["ci_data"]
                    q05 = ci_data["psd_real_q05"][:, i, i]
                    q50 = ci_data["psd_real_q50"][:, i, i]
                    q95 = ci_data["psd_real_q95"][:, i, i]
                    color = spec["color"]
                    if show_posterior:
                        ax.fill_between(
                            ci_data["freq"],
                            q05,
                            q95,
                            color=color,
                            alpha=0.16,
                            zorder=2,
                        )
                        if show_median:
                            ax.plot(
                                ci_data["freq"],
                                q50,
                                color=color,
                                lw=1.5,
                                zorder=4,
                            )
                    pos_values.extend(
                        [
                            q05[q05 > 0],
                            q50[q50 > 0],
                            q95[q95 > 0],
                        ]
                    )

                if show_truth:
                    ax.plot(
                        run_specs[0]["ci_data"]["freq"],
                        true_diag,
                        color=TRUE_COLOR,
                        lw=TRUE_LW,
                        zorder=5,
                        ls="--",
                        label=TRUE_LABEL,
                    )
                if overlay_freq is not None and overlay_diag is not None:
                    plot_kwargs = dict(
                        color=overlay_color,
                        lw=overlay_lw,
                        alpha=overlay_alpha,
                        zorder=overlay_zorder,
                        label=overlay_label,
                    )
                    if overlay_marker is not None:
                        plot_kwargs["marker"] = overlay_marker
                        plot_kwargs["markersize"] = (
                            2.0
                            if overlay_markersize is None
                            else overlay_markersize
                        )
                        plot_kwargs["linestyle"] = "None"
                    ax.plot(overlay_freq, overlay_diag, **plot_kwargs)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(FMIN, FMAX)
                finite_pos = [arr for arr in pos_values if arr.size > 0]
                if finite_pos:
                    ylo = min(float(np.min(arr)) for arr in finite_pos) * 0.3
                    yhi = max(float(np.max(arr)) for arr in finite_pos) * 5.0
                    ax.set_ylim(ylo, yhi)
                ax.text(
                    0.04,
                    0.93,
                    f"$S_{{{channel_labels[i]}{channel_labels[i]}}}$",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    weight="bold",
                )
            else:
                coh_true = _compute_coherence(truth_real, truth_imag, i, j)
                finite_arrays = []
                if show_truth:
                    finite_arrays.append(coh_true[np.isfinite(coh_true)])
                if (
                    show_overlay_on_coherence
                    and coherence_overlay_freq is not None
                    and coherence_overlay_S is not None
                ):
                    S_ij_w = coherence_overlay_S[:, i, j]
                    S_ii_w = coherence_overlay_S[:, i, i].real
                    S_jj_w = coherence_overlay_S[:, j, j].real
                    denom_w = S_ii_w * S_jj_w
                    safe_w = np.where(denom_w > 0, denom_w, np.nan)
                    overlay_coh = np.abs(S_ij_w) ** 2 / safe_w
                    finite_arrays.append(overlay_coh[np.isfinite(overlay_coh)])
                    plot_kwargs = dict(
                        color=overlay_color,
                        lw=overlay_lw,
                        alpha=overlay_alpha,
                        zorder=overlay_zorder,
                    )
                    if overlay_marker is not None:
                        plot_kwargs["marker"] = overlay_marker
                        plot_kwargs["markersize"] = (
                            2.0
                            if overlay_markersize is None
                            else overlay_markersize
                        )
                        plot_kwargs["linestyle"] = "None"
                    ax.plot(coherence_overlay_freq, overlay_coh, **plot_kwargs)

                for spec in run_specs:
                    ci_data = spec["ci_data"]
                    coh_q05 = ci_data.get("coh_q05")
                    coh_q50 = ci_data.get("coh_q50")
                    coh_q95 = ci_data.get("coh_q95")
                    if coh_q05 is None or coh_q50 is None or coh_q95 is None:
                        coh_q05 = coh_q50 = coh_q95 = _compute_coherence(
                            ci_data["psd_real_q50"],
                            ci_data["psd_imag_q50"],
                            i,
                            j,
                        )
                    else:
                        coh_q05 = np.asarray(coh_q05)[:, i, j]
                        coh_q50 = np.asarray(coh_q50)[:, i, j]
                        coh_q95 = np.asarray(coh_q95)[:, i, j]
                    finite_arrays.extend(
                        [
                            coh_q05[np.isfinite(coh_q05)],
                            coh_q50[np.isfinite(coh_q50)],
                            coh_q95[np.isfinite(coh_q95)],
                        ]
                    )
                    if show_posterior:
                        ax.fill_between(
                            ci_data["freq"],
                            coh_q05,
                            coh_q95,
                            color=spec["color"],
                            alpha=0.16,
                            zorder=2,
                        )
                        if show_median:
                            ax.plot(
                                ci_data["freq"],
                                coh_q50,
                                color=spec["color"],
                                lw=1.5,
                                zorder=4,
                            )

                if show_truth:
                    ax.plot(
                        run_specs[0]["ci_data"]["freq"],
                        coh_true,
                        color=TRUE_COLOR,
                        lw=TRUE_LW,
                        zorder=5,
                        ls="--",
                    )
                ax.set_xscale("log")
                ax.set_xlim(FMIN, FMAX)
                finite_arrays = [arr for arr in finite_arrays if arr.size > 0]
                if finite_arrays:
                    ymax = float(max(np.max(arr) for arr in finite_arrays))
                    ymax = min(max(0.02, 1.15 * ymax), 1.05)
                    ax.set_ylim(-0.03 * ymax, ymax)
                else:
                    ax.set_ylim(-0.05, 1.05)
                ax.text(
                    0.04,
                    0.93,
                    f"$C_{{{channel_labels[i]}{channel_labels[j]}}}$",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    weight="bold",
                )

            ax.tick_params(labelsize=8)
            if i < p - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Frequency [Hz]", fontsize=9)
            ax.set_ylabel(
                f"PSD [{psd_unit}]" if i == j else "Coherence", fontsize=9
            )

    legend_elements = []
    if show_posterior:
        for spec in run_specs:
            legend_elements.append(
                mpatches.Patch(
                    facecolor=spec["color"],
                    alpha=0.22,
                    label=f"{spec['label']} 90% CI",
                )
            )
            if show_median:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color=spec["color"],
                        lw=1.5,
                        label=f"{spec['label']} median",
                    )
                )
    if show_truth:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=TRUE_COLOR,
                lw=TRUE_LW,
                ls="--",
                label=TRUE_LABEL,
            )
        )
    if overlay_freq is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=overlay_color,
                lw=overlay_lw,
                alpha=overlay_alpha,
                label=overlay_label,
            )
        )
    fig.legend(
        handles=legend_elements,
        fontsize=7.5,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
    )

    fig.savefig(str(outpath), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-run PSD matrix figure: {outpath}")


# ── entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    # Try to auto-detect the preferred run_x seed_0 path.
    _default_run_x = _default_run_x_dir()
    parser = argparse.ArgumentParser(
        description="Generate final LISA paper figures (3 PSD + 3 relative-error).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-x",
        type=Path,
        default=_default_run_x,
        help=(
            "Path to run_x seed_0 directory. "
            "Uses inference_data.nc when present, else compact_ci_curves.npz."
        ),
    )
    parser.add_argument(
        "--run-y",
        type=Path,
        default=None,
        help=(
            "Path to run_y (native AET) seed_0 directory. "
            "Uses inference_data.nc when present, else compact_ci_curves.npz. "
            "If omitted, Figure 3 is skipped."
        ),
    )
    parser.add_argument(
        "--run-x-30d",
        type=Path,
        default=None,
        help="Optional 30-day run_x seed_0 directory for the multi-duration XYZ overlay figure.",
    )
    parser.add_argument(
        "--run-x-90d",
        type=Path,
        default=None,
        help="Optional 90-day run_x seed_0 directory for the multi-duration XYZ overlay figure.",
    )
    parser.add_argument(
        "--run-x-365d",
        type=Path,
        default=None,
        help=(
            "Optional 365-day run_x seed_0 directory for the multi-duration XYZ "
            "overlay figure. Defaults to --run-x when omitted."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=HERE / "paper_figs",
        help="Output directory for figures (default: paper_figs/).",
    )
    parser.add_argument(
        "--freq-units",
        action="store_true",
        default=False,
        help="Plot PSD in frequency-fluctuation units (Hz^2/Hz) instead of strain (1/Hz).",
    )
    parser.add_argument(
        "--data-overlay",
        type=str,
        choices=("raw", "welch", "mixed", "none"),
        default="raw",
        help=(
            "Empirical overlay for the paper plots. "
            "`raw` uses observed_data.periodogram from idata, "
            "`welch` regenerates a Welch estimate, "
            "`mixed` uses raw periodogram on diagonals and Welch for coherence, "
            "`none` disables the overlay."
        ),
    )
    parser.add_argument(
        "--welch-block-days",
        type=float,
        default=DEFAULT_WELCH_BLOCK_DAYS,
        help=(
            "Block length in days used when regenerating the Welch overlay. "
            "Larger values mean fewer averaged blocks. Default: 14."
        ),
    )
    parser.add_argument(
        "--overlay-duration-days",
        type=float,
        default=365.0,
        help=(
            "Duration of regenerated empirical overlay data in days. "
            "Used for `--data-overlay welch`, and also for regenerated raw overlays."
        ),
    )
    parser.add_argument(
        "--coherence-overlay-duration-days",
        type=float,
        default=None,
        help=(
            "Optional separate duration in days for the coherence overlay when "
            "using `--data-overlay welch`. If omitted, coherence uses the same "
            "duration as the diagonal overlay."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to regenerate timeseries for Welch (default: 0).",
    )
    parser.add_argument(
        "--raw-duration-days",
        type=float,
        default=None,
        help=(
            "When using `--data-overlay raw`, regenerate the raw periodogram "
            "from only the first N days instead of using the saved full-run "
            "observed_data.periodogram."
        ),
    )
    parser.add_argument(
        "--hide-posterior",
        action="store_true",
        default=False,
        help=(
            "Hide the posterior CI and median in the PSD figures. Useful for "
            "debugging the empirical data/truth overlays."
        ),
    )
    parser.add_argument(
        "--show-data",
        dest="show_data",
        action="store_true",
        default=True,
        help="Show empirical data overlays (default: on when available).",
    )
    parser.add_argument(
        "--hide-data",
        dest="show_data",
        action="store_false",
        help="Hide empirical data overlays regardless of --data-overlay.",
    )
    parser.add_argument(
        "--show-truth",
        dest="show_truth",
        action="store_true",
        default=True,
        help="Show analytical truth overlay (default: on).",
    )
    parser.add_argument(
        "--hide-truth",
        dest="show_truth",
        action="store_false",
        help="Hide analytical truth overlay.",
    )
    parser.add_argument(
        "--ci-only",
        action="store_true",
        default=False,
        help=(
            "Show posterior CI bands without posterior median lines in the paper plots."
        ),
    )
    parser.add_argument(
        "--show-median",
        action="store_true",
        default=False,
        help="Show posterior median lines in addition to CI bands (default: off).",
    )
    parser.add_argument(
        "--comparison-readme-name",
        type=str,
        default="README.md",
        help=(
            "Filename for the duration-comparison Markdown summary written to "
            "--outdir when the extra run_x durations are provided."
        ),
    )
    parser.add_argument(
        "--recompute-full-draw-ci",
        action="store_true",
        default=False,
        help=(
            "Recompute PSD/coherence quantiles from all saved sample_stats draws "
            "and use those quantiles for plotting instead of stored posterior_psd."
        ),
    )
    parser.add_argument(
        "--full-draw-ci-max-draws",
        type=int,
        default=0,
        help=(
            "Optional cap when recomputing CI from sample_stats. "
            "0 means use all saved draws."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    show_median = bool(args.show_median) and not args.ci_only
    compare_args = [args.run_x_30d, args.run_x_90d, args.run_x_365d]
    if any(path is not None for path in compare_args) and (
        args.run_x_30d is None or args.run_x_90d is None
    ):
        raise ValueError(
            "--run-x-30d and --run-x-90d must both be provided to build the "
            "multi-duration comparison figure."
        )

    recomputed_ci_cache: dict[str, dict[str, np.ndarray]] = {}

    def _maybe_recompute_ci(
        label: str, run_dir: Path, stored_ci: dict
    ) -> dict:
        if not args.recompute_full_draw_ci:
            return stored_ci
        recomputed_ci, meta = _recompute_ci_from_all_draws(
            run_dir,
            max_draws=(
                None
                if args.full_draw_ci_max_draws <= 0
                else args.full_draw_ci_max_draws
            ),
        )
        comparison = _compare_ci_summaries(
            stored_ci,
            recomputed_ci,
            meta=meta,
        )
        _save_json(
            comparison,
            outdir / f"{label}_stored_vs_full_draw_ci.json",
        )
        recomputed_ci_cache[str(run_dir)] = recomputed_ci
        return recomputed_ci

    # ── Load run_x XYZ CI curves ──────────────────────────────────────────────
    ci_xyz_stored = _load_ci_data(args.run_x)
    ci_xyz = _maybe_recompute_ci("run_x", Path(args.run_x), ci_xyz_stored)

    # ── Transform to AET ──────────────────────────────────────────────────────
    ci_aet_from_xyz = transform_ci_curves_to_aet(ci_xyz)
    print("Transformed XYZ CI curves to AET basis.")

    # ── Unit conversion ──────────────────────────────────────────────────────
    # Saved CI curves are stored in strain units.
    psd_unit = PSD_UNIT_FREQ if args.freq_units else PSD_UNIT_STRAIN
    unit_tag = " [Hz²/Hz]" if args.freq_units else " [1/Hz]"
    unit_slug = "freq_units" if args.freq_units else "strain_units"
    if args.freq_units:
        print("Converting CI curves from strain units to Hz^2/Hz.")
        ci_xyz = _convert_ci_data_to_freq_units(ci_xyz)
        ci_aet_from_xyz = _convert_ci_data_to_freq_units(ci_aet_from_xyz)
    else:
        print("Using CI curves in strain units (1/Hz).")

    # ── Empirical overlay ────────────────────────────────────────────────────
    overlay_freq_xyz = overlay_S_xyz = None
    overlay_freq_aet = overlay_S_aet = None
    coherence_overlay_freq_xyz = coherence_overlay_S_xyz = None
    coherence_overlay_freq_aet = coherence_overlay_S_aet = None
    overlay_label = None
    overlay_color = WELCH_COLOR
    overlay_lw = WELCH_LW
    overlay_alpha = WELCH_ALPHA
    overlay_marker = None
    overlay_markersize = None
    overlay_zorder = 4
    show_overlay_on_coherence = True

    if not args.show_data:
        print("Empirical overlay hidden (--hide-data).")
    elif args.data_overlay in {"raw", "mixed"}:
        from utils.aet import xyz_to_aet_matrix

        if args.raw_duration_days is not None:
            y_xyz, dt = _generate_xyz_for_overlay(
                seed=args.seed, duration_days=float(args.raw_duration_days)
            )
            overlay_freq_xyz, overlay_S_xyz = _raw_periodogram_psd(
                y_xyz, dt=dt
            )
            print(
                "Loaded regenerated raw periodogram overlay from the first "
                f"{args.raw_duration_days:g} days."
            )
        else:
            raw_xyz = _load_raw_periodogram(args.run_x)
            if raw_xyz is None:
                raise FileNotFoundError(
                    "Raw overlay requested but observed_data.periodogram is not "
                    f"available in {(Path(args.run_x) / 'inference_data.nc')}. "
                    "Point --run-x at a run directory with saved inference_data.nc."
                )
            overlay_freq_xyz, overlay_S_xyz = raw_xyz
            print(
                f"Loaded raw periodogram overlay from {Path(args.run_x) / 'inference_data.nc'}."
            )

        overlay_freq_xyz, overlay_S_xyz = _thin_overlay_for_plot(
            overlay_freq_xyz, overlay_S_xyz
        )
        overlay_freq_aet = overlay_freq_xyz
        overlay_S_aet = xyz_to_aet_matrix(overlay_S_xyz)
        overlay_label = RAW_LABEL
        overlay_color = RAW_COLOR
        overlay_lw = RAW_LW
        overlay_alpha = RAW_ALPHA
        overlay_marker = None
        overlay_markersize = None
        overlay_zorder = -1
        show_overlay_on_coherence = args.data_overlay == "mixed"
        if args.freq_units:
            overlay_S_xyz = _convert_welch_to_freq_units(
                overlay_freq_xyz, overlay_S_xyz
            )
            overlay_S_aet = _convert_welch_to_freq_units(
                overlay_freq_aet, overlay_S_aet
            )
        if args.data_overlay == "mixed":
            print("Generating Welch overlay for coherence panels...")
            y_xyz_c, Nb_c, Lb_c, fs_c = _generate_xyz_for_welch(
                seed=args.seed,
                duration_days=float(args.overlay_duration_days),
                block_days=args.welch_block_days,
            )
            coherence_overlay_freq_xyz, coherence_overlay_S_xyz = _welch_psd(
                y_xyz_c,
                Lb=Lb_c,
                fs=fs_c,
            )
            coherence_overlay_S_aet = xyz_to_aet_matrix(
                coherence_overlay_S_xyz
            )
            coherence_overlay_freq_aet = coherence_overlay_freq_xyz
            if args.freq_units:
                coherence_overlay_S_xyz = _convert_welch_to_freq_units(
                    coherence_overlay_freq_xyz, coherence_overlay_S_xyz
                )
                coherence_overlay_S_aet = _convert_welch_to_freq_units(
                    coherence_overlay_freq_aet, coherence_overlay_S_aet
                )
            print(
                f"  Coherence Welch PSD computed ({len(coherence_overlay_freq_xyz)} freq bins, "
                f"{Nb_c} blocks, {args.welch_block_days:g}-day blocks, "
                f"{args.overlay_duration_days:g} total days)."
            )
    elif args.data_overlay == "welch":
        print("Generating LISA timeseries for Welch overlay...")
        try:
            from utils.aet import xyz_to_aet_matrix

            y_xyz, Nb, Lb, fs = _generate_xyz_for_welch(
                seed=args.seed,
                duration_days=float(args.overlay_duration_days),
                block_days=args.welch_block_days,
            )
            overlay_freq_xyz, overlay_S_xyz = _welch_psd(
                y_xyz,
                Lb=Lb,
                fs=fs,
            )
            # Transform to AET
            overlay_S_aet = xyz_to_aet_matrix(overlay_S_xyz)
            overlay_freq_aet = overlay_freq_xyz
            coherence_overlay_freq_xyz = overlay_freq_xyz
            coherence_overlay_S_xyz = overlay_S_xyz
            coherence_overlay_freq_aet = overlay_freq_aet
            coherence_overlay_S_aet = overlay_S_aet
            overlay_label = WELCH_LABEL

            # Match the Welch overlay to the selected plot units.
            if args.freq_units:
                overlay_S_xyz = _convert_welch_to_freq_units(
                    overlay_freq_xyz, overlay_S_xyz
                )
                overlay_S_aet = _convert_welch_to_freq_units(
                    overlay_freq_aet, overlay_S_aet
                )
                coherence_overlay_S_xyz = overlay_S_xyz
                coherence_overlay_S_aet = overlay_S_aet

            print(
                f"  Welch PSD computed ({len(overlay_freq_xyz)} freq bins, "
                f"{Nb} blocks, {args.welch_block_days:g}-day blocks, "
                f"{args.overlay_duration_days:g} total days)."
            )

            coherence_duration = args.coherence_overlay_duration_days
            if coherence_duration is not None and float(
                coherence_duration
            ) != float(args.overlay_duration_days):
                print(
                    "Generating separate Welch overlay for coherence using "
                    f"{coherence_duration:g} days..."
                )
                y_xyz_c, Nb_c, Lb_c, fs_c = _generate_xyz_for_welch(
                    seed=args.seed,
                    duration_days=float(coherence_duration),
                    block_days=args.welch_block_days,
                )
                coherence_overlay_freq_xyz, coherence_overlay_S_xyz = (
                    _welch_psd(
                        y_xyz_c,
                        Lb=Lb_c,
                        fs=fs_c,
                    )
                )
                coherence_overlay_S_aet = xyz_to_aet_matrix(
                    coherence_overlay_S_xyz
                )
                coherence_overlay_freq_aet = coherence_overlay_freq_xyz
                if args.freq_units:
                    coherence_overlay_S_xyz = _convert_welch_to_freq_units(
                        coherence_overlay_freq_xyz, coherence_overlay_S_xyz
                    )
                    coherence_overlay_S_aet = _convert_welch_to_freq_units(
                        coherence_overlay_freq_aet, coherence_overlay_S_aet
                    )
                print(
                    f"  Coherence Welch PSD computed ({len(coherence_overlay_freq_xyz)} freq bins, "
                    f"{Nb_c} blocks, {args.welch_block_days:g}-day blocks, "
                    f"{coherence_duration:g} total days)."
                )
        except Exception as exc:
            print(
                f"  WARNING: Welch overlay failed ({exc}). Skipping overlay."
            )
    else:
        print("Empirical overlay disabled.")

    # ── Figure 1: run_x XYZ PSD ───────────────────────────────────────────────
    print("\n── Figure 1: run_x XYZ PSD ──")
    _make_psd_matrix_figure(
        ci_xyz,
        channel_labels=CHANNEL_LABELS_XYZ,
        overlay_freq=overlay_freq_xyz,
        overlay_S=overlay_S_xyz,
        coherence_overlay_freq=coherence_overlay_freq_xyz,
        coherence_overlay_S=coherence_overlay_S_xyz,
        overlay_label=overlay_label or WELCH_LABEL,
        overlay_color=overlay_color,
        overlay_lw=overlay_lw,
        overlay_alpha=overlay_alpha,
        overlay_marker=overlay_marker,
        overlay_markersize=overlay_markersize,
        overlay_zorder=overlay_zorder,
        show_overlay_on_coherence=show_overlay_on_coherence,
        show_posterior=not args.hide_posterior,
        show_median=show_median,
        show_truth=args.show_truth,
        title="",
        outpath=outdir / f"fig1_runx_xyz_psd_{unit_slug}.pdf",
        psd_unit=psd_unit,
    )
    if not args.hide_posterior and show_median and args.show_truth:
        _make_relative_error_figure(
            ci_xyz,
            channel_labels=CHANNEL_LABELS_XYZ,
            title=f"Relative Error — XYZ{unit_tag}",
            outpath=outdir / f"fig1_runx_xyz_relerr_{unit_slug}.pdf",
        )
    if args.show_truth:
        _make_combined_residual_figure(
            ci_xyz,
            channel_labels=CHANNEL_LABELS_XYZ,
            title=f"Residuals — XYZ{unit_tag}",
            outpath=outdir
            / f"fig1_runx_xyz_combined_residuals_{unit_slug}.pdf",
        )

    # ── Figure 1b: run_x XYZ duration comparison ──────────────────────────────
    compare_run_365d = args.run_x_365d or args.run_x
    if args.run_x_30d is not None and args.run_x_90d is not None:
        compare_dirs = {
            "30d": Path(args.run_x_30d),
            "90d": Path(args.run_x_90d),
            "365d": Path(compare_run_365d),
        }
        compare_ci_cache = {"365d": ci_xyz}
        compare_specs: list[dict] = []
        for key in ("30d", "90d", "365d"):
            run_dir = compare_dirs[key]
            if key in compare_ci_cache:
                ci_data = compare_ci_cache[key]
            else:
                stored_ci = _load_ci_data(run_dir)
                ci_data = _maybe_recompute_ci(
                    f"run_x_{key}", run_dir, stored_ci
                )
                if args.freq_units:
                    ci_data = _convert_ci_data_to_freq_units(ci_data)
            summary = _load_compact_run_summary(run_dir)
            compare_specs.append(
                {
                    "key": key,
                    "label": COMPARE_LABELS[key],
                    "color": COMPARE_COLORS[key],
                    "run_dir": run_dir,
                    "ci_data": ci_data,
                    "summary": summary,
                    "metrics": _compute_compare_metrics(ci_data),
                }
            )

        print("\n── Figure 1b: run_x XYZ duration comparison ──")
        compare_out = (
            outdir / f"fig1b_runx_xyz_duration_overlay_{unit_slug}.pdf"
        )
        _make_multirun_psd_matrix_figure(
            compare_specs,
            channel_labels=CHANNEL_LABELS_XYZ,
            overlay_freq=overlay_freq_xyz,
            overlay_S=overlay_S_xyz,
            coherence_overlay_freq=coherence_overlay_freq_xyz,
            coherence_overlay_S=coherence_overlay_S_xyz,
            overlay_label=overlay_label or WELCH_LABEL,
            overlay_color=overlay_color,
            overlay_lw=overlay_lw,
            overlay_alpha=overlay_alpha,
            overlay_marker=overlay_marker,
            overlay_markersize=overlay_markersize,
            overlay_zorder=overlay_zorder,
            show_overlay_on_coherence=show_overlay_on_coherence,
            show_posterior=not args.hide_posterior,
            show_median=show_median,
            show_truth=args.show_truth,
            title="",
            outpath=compare_out,
            psd_unit=psd_unit,
        )
        _write_duration_comparison_readme(
            compare_specs,
            outpath=outdir / args.comparison_readme_name,
            figure_name=compare_out.name,
            overlay_label=(
                (overlay_label or "None") if args.show_data else "Hidden"
            ),
            psd_unit=psd_unit,
            block_days=7.0,
        )

        print("\n── Figure 1c: duration comparison residuals ──")
        _make_duration_residual_figure(
            compare_specs,
            channel_labels=CHANNEL_LABELS_XYZ,
            title=f"Residuals by duration — XYZ{unit_tag}",
            outpath=outdir
            / f"fig1c_runx_xyz_duration_residuals_{unit_slug}.pdf",
        )

    # ── Figure 2: run_x posteriors in AET basis ───────────────────────────────
    print("\n── Figure 2: run_x XYZ posteriors → AET ──")
    _make_psd_matrix_figure(
        ci_aet_from_xyz,
        channel_labels=CHANNEL_LABELS_AET,
        overlay_freq=overlay_freq_aet,
        overlay_S=overlay_S_aet,
        coherence_overlay_freq=coherence_overlay_freq_aet,
        coherence_overlay_S=coherence_overlay_S_aet,
        overlay_label=overlay_label or WELCH_LABEL,
        overlay_color=overlay_color,
        overlay_lw=overlay_lw,
        overlay_alpha=overlay_alpha,
        overlay_marker=overlay_marker,
        overlay_markersize=overlay_markersize,
        overlay_zorder=overlay_zorder,
        show_overlay_on_coherence=show_overlay_on_coherence,
        show_posterior=not args.hide_posterior,
        show_median=show_median,
        show_truth=args.show_truth,
        title="",
        outpath=outdir / f"fig2_runx_aet_psd_{unit_slug}.pdf",
        psd_unit=psd_unit,
    )
    if not args.hide_posterior and show_median and args.show_truth:
        _make_relative_error_figure(
            ci_aet_from_xyz,
            channel_labels=CHANNEL_LABELS_AET,
            title=f"Relative Error — AET (from XYZ posterior){unit_tag}",
            outpath=outdir / f"fig2_runx_aet_relerr_{unit_slug}.pdf",
        )
    if args.show_truth:
        _make_combined_residual_figure(
            ci_aet_from_xyz,
            channel_labels=CHANNEL_LABELS_AET,
            title=f"Residuals — AET (from XYZ posterior){unit_tag}",
            outpath=outdir
            / f"fig2_runx_aet_combined_residuals_{unit_slug}.pdf",
        )

    # ── Figure 3: run_y native AET PSD ───────────────────────────────────────
    if args.run_y is not None:
        try:
            ci_aet_native = _load_ci_data(args.run_y)
            print("\n── Figure 3: run_y native AET PSD ──")
            if args.freq_units:
                ci_aet_native = _convert_ci_data_to_freq_units(ci_aet_native)

            overlay_freq_y = overlay_S_y = None
            coherence_overlay_freq_y = coherence_overlay_S_y = None
            if args.data_overlay == "raw":
                raw_aet = _load_raw_periodogram(args.run_y)
                if raw_aet is not None:
                    overlay_freq_y, overlay_S_y = raw_aet
                    if args.freq_units:
                        overlay_S_y = _convert_welch_to_freq_units(
                            overlay_freq_y, overlay_S_y
                        )
            elif args.data_overlay == "welch" and overlay_freq_xyz is not None:
                from utils.aet import xyz_to_aet_matrix

                overlay_S_y = xyz_to_aet_matrix(overlay_S_xyz)
                overlay_freq_y = overlay_freq_xyz
                coherence_overlay_freq_y = coherence_overlay_freq_aet
                coherence_overlay_S_y = coherence_overlay_S_aet

            _make_psd_matrix_figure(
                ci_aet_native,
                channel_labels=CHANNEL_LABELS_AET,
                overlay_freq=overlay_freq_y,
                overlay_S=overlay_S_y,
                coherence_overlay_freq=coherence_overlay_freq_y,
                coherence_overlay_S=coherence_overlay_S_y,
                overlay_label=overlay_label or WELCH_LABEL,
                overlay_color=overlay_color,
                overlay_lw=overlay_lw,
                overlay_alpha=overlay_alpha,
                overlay_marker=overlay_marker,
                overlay_markersize=overlay_markersize,
                overlay_zorder=overlay_zorder,
                show_overlay_on_coherence=show_overlay_on_coherence,
                show_posterior=not args.hide_posterior,
                show_median=show_median,
                show_truth=args.show_truth,
                title="",
                outpath=outdir / f"fig3_runy_aet_psd_{unit_slug}.pdf",
                psd_unit=psd_unit,
            )
            if not args.hide_posterior and show_median and args.show_truth:
                _make_relative_error_figure(
                    ci_aet_native,
                    channel_labels=CHANNEL_LABELS_AET,
                    title=f"Relative Error — native AET (run_y){unit_tag}",
                    outpath=outdir / f"fig3_runy_aet_relerr_{unit_slug}.pdf",
                )
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}. Skipping Figure 3.")
    else:
        print("\nrun_y not provided — skipping Figure 3.")

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
