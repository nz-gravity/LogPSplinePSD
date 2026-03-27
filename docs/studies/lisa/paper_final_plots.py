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
import os
import sys
from pathlib import Path
from typing import Optional

import arviz as az
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
from log_psplines.plotting.base import setup_plot_style  # noqa: E402

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
RAW_MARKER = "."
RAW_MARKERSIZE = 2.0

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
    show_posterior: bool = True,
    *,
    channel_label: str = "",
) -> None:
    """Auto-spectrum (diagonal) panel: log-log scale."""
    if show_posterior:
        ax.fill_between(
            freq, q05, q95, color=CI_COLOR, alpha=CI_ALPHA, zorder=2
        )
        ax.plot(freq, q50, color=MEDIAN_COLOR, lw=MEDIAN_LW, zorder=3)
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
            zorder=4,
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
            0.96,
            0.93,
            f"$S_{{{channel_label}{channel_label}}}$",
            transform=ax.transAxes,
            ha="right",
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
    show_posterior: bool = True,
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
            zorder=4,
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
    if show_posterior:
        ax.plot(freq, coh_q50, color=MEDIAN_COLOR, lw=MEDIAN_LW, zorder=3)
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
    ax.set_ylim(-0.05, 1.05)
    if ch_i and ch_j:
        ax.text(
            0.96,
            0.93,
            f"$C_{{{ch_i}{ch_j}}}$",
            transform=ax.transAxes,
            ha="right",
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
    overlay_label: str = WELCH_LABEL,
    overlay_color: str = WELCH_COLOR,
    overlay_lw: float = WELCH_LW,
    overlay_alpha: float = WELCH_ALPHA,
    overlay_marker: str | None = None,
    overlay_markersize: float | None = None,
    show_posterior: bool = True,
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
                    show_posterior=show_posterior,
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
                if overlay_freq is not None and overlay_S is not None:
                    overlay_f_coh = overlay_freq
                    S_ij_w = overlay_S[:, i, j]
                    S_ii_w = overlay_S[:, i, i].real
                    S_jj_w = overlay_S[:, j, j].real
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
                    show_posterior=show_posterior,
                    ch_i=channel_labels[i],
                    ch_j=channel_labels[j],
                )

            # shared formatting
            ax.tick_params(labelsize=8)
            if i < p - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Frequency [Hz]", fontsize=9)
            if j == 0:
                if i == j:
                    ax.set_ylabel(f"PSD [{psd_unit}]", fontsize=9)
                else:
                    ax.set_ylabel("Coherence", fontsize=9)

    # Legend on top-left diagonal panel
    legend_elements = []
    if show_posterior:
        legend_elements.extend(
            [
                mpatches.Patch(facecolor=CI_COLOR, alpha=0.4, label="90% CI"),
                Line2D(
                    [0],
                    [0],
                    color=MEDIAN_COLOR,
                    lw=MEDIAN_LW,
                    label="Posterior median",
                ),
            ]
        )
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
        choices=("raw", "welch", "none"),
        default="raw",
        help=(
            "Empirical overlay for the paper plots. "
            "`raw` uses observed_data.periodogram from idata, "
            "`welch` regenerates a Welch estimate, `none` disables the overlay."
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
        "--seed",
        type=int,
        default=0,
        help="Seed used to regenerate timeseries for Welch (default: 0).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load run_x XYZ CI curves ──────────────────────────────────────────────
    ci_xyz = _load_ci_data(args.run_x)

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
    overlay_label = None
    overlay_color = WELCH_COLOR
    overlay_lw = WELCH_LW
    overlay_alpha = WELCH_ALPHA
    overlay_marker = None
    overlay_markersize = None

    if args.data_overlay == "raw":
        raw_xyz = _load_raw_periodogram(args.run_x)
        if raw_xyz is None:
            raise FileNotFoundError(
                "Raw overlay requested but observed_data.periodogram is not "
                f"available in {(Path(args.run_x) / 'inference_data.nc')}. "
                "Point --run-x at a run directory with saved inference_data.nc."
            )
        from utils.aet import xyz_to_aet_matrix

        overlay_freq_xyz, overlay_S_xyz = raw_xyz
        overlay_freq_aet = overlay_freq_xyz
        overlay_S_aet = xyz_to_aet_matrix(overlay_S_xyz)
        overlay_label = RAW_LABEL
        overlay_color = RAW_COLOR
        overlay_lw = RAW_LW
        overlay_alpha = RAW_ALPHA
        overlay_marker = RAW_MARKER
        overlay_markersize = RAW_MARKERSIZE
        if args.freq_units:
            overlay_S_xyz = _convert_welch_to_freq_units(
                overlay_freq_xyz, overlay_S_xyz
            )
            overlay_S_aet = _convert_welch_to_freq_units(
                overlay_freq_aet, overlay_S_aet
            )
        print(
            f"Loaded raw periodogram overlay from {Path(args.run_x) / 'inference_data.nc'}."
        )
    elif args.data_overlay == "welch":
        print("Generating LISA timeseries for Welch overlay...")
        try:
            from utils.aet import xyz_to_aet_matrix

            y_xyz, Nb, Lb, fs = _generate_xyz_for_welch(
                seed=args.seed,
                duration_days=365.0,
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
            overlay_label = WELCH_LABEL

            # Match the Welch overlay to the selected plot units.
            if args.freq_units:
                overlay_S_xyz = _convert_welch_to_freq_units(
                    overlay_freq_xyz, overlay_S_xyz
                )
                overlay_S_aet = _convert_welch_to_freq_units(
                    overlay_freq_aet, overlay_S_aet
                )

            print(
                f"  Welch PSD computed ({len(overlay_freq_xyz)} freq bins, "
                f"{Nb} blocks, {args.welch_block_days:g}-day blocks)."
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
        overlay_label=overlay_label or WELCH_LABEL,
        overlay_color=overlay_color,
        overlay_lw=overlay_lw,
        overlay_alpha=overlay_alpha,
        overlay_marker=overlay_marker,
        overlay_markersize=overlay_markersize,
        show_posterior=not args.hide_posterior,
        title=f"LISA XYZ — Log-P-Spline PSD{unit_tag}",
        outpath=outdir / f"fig1_runx_xyz_psd_{unit_slug}.pdf",
        psd_unit=psd_unit,
    )
    if not args.hide_posterior:
        _make_relative_error_figure(
            ci_xyz,
            channel_labels=CHANNEL_LABELS_XYZ,
            title=f"Relative Error — XYZ{unit_tag}",
            outpath=outdir / f"fig1_runx_xyz_relerr_{unit_slug}.pdf",
        )

    # ── Figure 2: run_x posteriors in AET basis ───────────────────────────────
    print("\n── Figure 2: run_x XYZ posteriors → AET ──")
    _make_psd_matrix_figure(
        ci_aet_from_xyz,
        channel_labels=CHANNEL_LABELS_AET,
        overlay_freq=overlay_freq_aet,
        overlay_S=overlay_S_aet,
        overlay_label=overlay_label or WELCH_LABEL,
        overlay_color=overlay_color,
        overlay_lw=overlay_lw,
        overlay_alpha=overlay_alpha,
        overlay_marker=overlay_marker,
        overlay_markersize=overlay_markersize,
        show_posterior=not args.hide_posterior,
        title=f"LISA AET (from XYZ posterior){unit_tag}",
        outpath=outdir / f"fig2_runx_aet_psd_{unit_slug}.pdf",
        psd_unit=psd_unit,
    )
    if not args.hide_posterior:
        _make_relative_error_figure(
            ci_aet_from_xyz,
            channel_labels=CHANNEL_LABELS_AET,
            title=f"Relative Error — AET (from XYZ posterior){unit_tag}",
            outpath=outdir / f"fig2_runx_aet_relerr_{unit_slug}.pdf",
        )

    # ── Figure 3: run_y native AET PSD ───────────────────────────────────────
    if args.run_y is not None:
        try:
            ci_aet_native = _load_ci_data(args.run_y)
            print("\n── Figure 3: run_y native AET PSD ──")
            if args.freq_units:
                ci_aet_native = _convert_ci_data_to_freq_units(ci_aet_native)

            overlay_freq_y = overlay_S_y = None
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

            _make_psd_matrix_figure(
                ci_aet_native,
                channel_labels=CHANNEL_LABELS_AET,
                overlay_freq=overlay_freq_y,
                overlay_S=overlay_S_y,
                overlay_label=overlay_label or WELCH_LABEL,
                overlay_color=overlay_color,
                overlay_lw=overlay_lw,
                overlay_alpha=overlay_alpha,
                overlay_marker=overlay_marker,
                overlay_markersize=overlay_markersize,
                show_posterior=not args.hide_posterior,
                title=f"LISA AET — native analysis (run_y, K=48, uniform){unit_tag}",
                outpath=outdir / f"fig3_runy_aet_psd_{unit_slug}.pdf",
                psd_unit=psd_unit,
            )
            if not args.hide_posterior:
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
