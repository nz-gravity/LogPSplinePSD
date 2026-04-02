"""Compare VAR(2)-3D density-knot allocation for spectral vs cholesky scoring.

This script reproduces the knot-allocation stage used in the multivariate
study without running full MCMC. It generates:

1. Per-component score-vs-knot plots for both scoring methods.
2. Cholesky-component plots (log_delta_sq and |theta_{j,l}|) with both knot
   sets overlaid.
3. 3x3 spectral-matrix component plot with knot overlays.
4. CSV/JSON summaries of knot locations and score concentration metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib
import numpy as np

from log_psplines.datatypes import MultivariateTimeseries, Periodogram
from log_psplines.datatypes.multivar_utils import (
    U_to_Y,
    psd_to_cholesky_components,
)
from log_psplines.mcmc import run_mcmc
from log_psplines.psplines.knots_locator import (
    init_knots,
    multivar_psd_knot_scores,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FS = 1.0
DEFAULT_BURN_IN = 512
DEFAULT_WINDOW = "rect"
VI_LR = 5e-4

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

METHODS = ("spectral", "cholesky")


@dataclass(frozen=True)
class ScoreSummary:
    method: str
    component: str
    top10_mass: float
    effective_bins: float
    effective_fraction: float
    gini: float


def _parse_window(raw: str) -> str | tuple | None:
    raw = raw.strip().lower()
    if raw in ("rect", "none", ""):
        return None
    if raw.startswith("tukey_"):
        alpha = float(raw.split("_", 1)[1])
        return ("tukey", alpha)
    return raw


def _simulate_var_process(
    *,
    n_samples: int,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    seed: int,
    fs: float = DEFAULT_FS,
    burn_in: int = DEFAULT_BURN_IN,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate VAR(p): x_t = sum_k A_k x_{t-k} + eps_t."""
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
    """Compute one-sided theoretical PSD matrix S(f) on a Hz frequency grid."""
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    ar_order, n_channels, _ = var_coeffs.shape
    omega = 2.0 * np.pi * freqs_hz / float(fs)
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
    return 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))


def _as_distribution(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full_like(arr, 1.0 / max(arr.size, 1))
    return arr / total


def _gini(p: np.ndarray) -> float:
    """Return a simple Gini concentration index for non-negative weights."""
    if p.size == 0:
        return 0.0
    sorted_p = np.sort(p)
    n = sorted_p.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * sorted_p) / n - (n + 1) / n))


def _summarise_score(
    method: str, component: str, score: np.ndarray
) -> ScoreSummary:
    dist = _as_distribution(score)
    n = dist.size
    top_bins = max(1, int(np.ceil(0.10 * n)))
    top10_mass = float(np.sum(np.sort(dist)[-top_bins:]))
    positive = dist > 0.0
    entropy = -float(np.sum(dist[positive] * np.log(dist[positive])))
    effective_bins = float(np.exp(entropy))
    effective_fraction = effective_bins / float(max(n, 1))
    return ScoreSummary(
        method=method,
        component=component,
        top10_mass=top10_mass,
        effective_bins=effective_bins,
        effective_fraction=effective_fraction,
        gini=_gini(dist),
    )


def _compute_density_knots(
    *,
    freq: np.ndarray,
    score: np.ndarray,
    n_knots: int,
) -> tuple[np.ndarray, np.ndarray]:
    periodogram = Periodogram(
        freqs=np.asarray(freq, dtype=np.float64),
        power=np.maximum(np.asarray(score, dtype=np.float64), 1e-12),
    )
    knots_norm = init_knots(
        n_knots=n_knots,
        periodogram=periodogram,
        method="density",
    )
    fmin = float(freq[0])
    fmax = float(freq[-1])
    knots_hz = fmin + knots_norm * (fmax - fmin)
    return knots_norm, knots_hz


def _write_csv(path: str, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_score_panels(
    *,
    freq: np.ndarray,
    components: list[str],
    scores_by_method: dict[str, dict[str, np.ndarray]],
    knots_hz_by_method: dict[str, dict[str, np.ndarray]],
    summary_by_method: dict[str, dict[str, ScoreSummary]],
    out_path: str,
) -> None:
    n_rows = len(components)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(16, 2.8 * n_rows),
        sharex=True,
    )

    method_colors = {"spectral": "#1f77b4", "cholesky": "#ff7f0e"}

    for col, method in enumerate(METHODS):
        for row, component in enumerate(components):
            ax = axes[row, col]
            score = scores_by_method[method][component]
            score_norm = _as_distribution(score)
            metric = summary_by_method[method][component]
            color = method_colors[method]

            ax.plot(freq, score_norm, color=color, lw=1.4)
            for knot in knots_hz_by_method[method][component]:
                ax.axvline(knot, color=color, lw=0.8, alpha=0.28)

            ax.set_title(
                f"{method} | {component} | top10={metric.top10_mass:.2f}, "
                f"eff_frac={metric.effective_fraction:.2f}"
            )
            if col == 0:
                ax.set_ylabel("Normalized score")
            if row == n_rows - 1:
                ax.set_xlabel("Frequency (Hz)")
            ax.grid(alpha=0.15)

    fig.suptitle("Density Knot Scores and Knot Locations", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_cholesky_component_panels(
    *,
    freq: np.ndarray,
    log_delta_sq: np.ndarray,
    theta: np.ndarray,
    diag_knots_hz: dict[str, dict[str, np.ndarray]],
    offdiag_re_knots_hz: dict[str, dict[tuple[int, int], np.ndarray]],
    offdiag_im_knots_hz: dict[str, dict[tuple[int, int], np.ndarray]],
    out_path: str,
) -> None:
    p = log_delta_sq.shape[1]
    pairs = [(i, j) for i in range(1, p) for j in range(i)]
    n_rows = max(p, len(pairs))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(16, 3.0 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    method_colors = {"spectral": "#1f77b4", "cholesky": "#ff7f0e"}

    for row in range(n_rows):
        ax_left = axes[row, 0]
        if row < p:
            comp = f"diag_{row}"
            ax_left.plot(freq, log_delta_sq[:, row], color="black", lw=1.1)
            for method in METHODS:
                knots = diag_knots_hz[method][comp]
                for knot in knots:
                    ax_left.axvline(
                        knot,
                        color=method_colors[method],
                        lw=0.8,
                        alpha=0.24,
                    )
            ax_left.set_title(f"log_delta_sq[{row}] with knot overlays")
            if row == 0:
                handles = [
                    plt.Line2D([0], [0], color="black", lw=1.3, label="value"),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["spectral"],
                        lw=1.3,
                        label="spectral knots",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["cholesky"],
                        lw=1.3,
                        label="cholesky knots",
                    ),
                ]
                ax_left.legend(handles=handles, loc="upper right", fontsize=8)
            ax_left.grid(alpha=0.15)
            ax_left.set_ylabel("Value")
        else:
            ax_left.axis("off")

        ax_right = axes[row, 1]
        if row < len(pairs):
            i, j = pairs[row]
            theta_re_abs = np.abs(np.real(theta[:, i, j]))
            theta_im_abs = np.abs(np.imag(theta[:, i, j]))
            ax_right.plot(freq, theta_re_abs, color="black", lw=1.1)
            ax_right.plot(freq, theta_im_abs, color="0.35", lw=1.0, ls="--")
            for method in METHODS:
                for knot in offdiag_re_knots_hz[method][(i, j)]:
                    ax_right.axvline(
                        knot,
                        color=method_colors[method],
                        lw=0.8,
                        alpha=0.24,
                    )
                for knot in offdiag_im_knots_hz[method][(i, j)]:
                    ax_right.axvline(
                        knot,
                        color=method_colors[method],
                        lw=0.8,
                        alpha=0.24,
                        ls="--",
                    )
            ax_right.set_title(
                f"theta[{i},{j}] magnitudes with Re/Im knot overlays"
            )
            if row == 0:
                handles = [
                    plt.Line2D([0], [0], color="black", lw=1.3, label="|Re theta|"),
                    plt.Line2D(
                        [0], [0], color="0.35", lw=1.2, ls="--", label="|Im theta|"
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["spectral"],
                        lw=1.2,
                        label="spectral knots (Re)",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["spectral"],
                        lw=1.2,
                        ls="--",
                        label="spectral knots (Im)",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["cholesky"],
                        lw=1.2,
                        label="cholesky knots (Re)",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["cholesky"],
                        lw=1.2,
                        ls="--",
                        label="cholesky knots (Im)",
                    ),
                ]
                ax_right.legend(handles=handles, loc="upper right", fontsize=8)
            ax_right.grid(alpha=0.15)
            ax_right.set_ylabel("Magnitude")
        else:
            ax_right.axis("off")

        if row == n_rows - 1:
            if axes[row, 0].has_data():
                axes[row, 0].set_xlabel("Frequency (Hz)")
            if axes[row, 1].has_data():
                axes[row, 1].set_xlabel("Frequency (Hz)")

    fig.suptitle(
        "Cholesky Components with Spectral vs Cholesky Knot Sets", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_s_matrix_knots_grid(
    *,
    freq: np.ndarray,
    s_matrix: np.ndarray,
    diag_knots_hz: dict[str, dict[str, np.ndarray]],
    offdiag_re_knots_hz: dict[str, dict[tuple[int, int], np.ndarray]],
    offdiag_im_knots_hz: dict[str, dict[tuple[int, int], np.ndarray]],
    out_path: str,
) -> None:
    """Plot S(f) in a 3x3 panel layout with spectral/cholesky knot overlays."""
    if s_matrix.shape[1:] != (3, 3):
        raise ValueError(
            f"s_matrix must have shape (N, 3, 3), got {s_matrix.shape}"
        )

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    method_colors = {"spectral": "#1f77b4", "cholesky": "#ff7f0e"}

    def _panel_series(i: int, j: int) -> tuple[np.ndarray, str]:
        if i == j:
            return np.real(s_matrix[:, i, j]), f"S_{i+1}{j+1}"
        if i < j:
            return np.real(s_matrix[:, i, j]), f"Re S_{i+1}{j+1}"
        return np.imag(s_matrix[:, i, j]), f"Im S_{i+1}{j+1}"

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            y, label = _panel_series(i, j)
            ax.plot(freq, y, color="black", lw=1.0)

            if i == j:
                comp = f"diag_{i}"
                for method in METHODS:
                    for knot in diag_knots_hz[method][comp]:
                        ax.axvline(
                            knot,
                            color=method_colors[method],
                            lw=0.8,
                            alpha=0.25,
                        )
            else:
                pair = (i, j) if i > j else (j, i)
                for method in METHODS:
                    knot_map = (
                        offdiag_re_knots_hz if i < j else offdiag_im_knots_hz
                    )
                    for knot in knot_map[method][pair]:
                        ax.axvline(
                            knot,
                            color=method_colors[method],
                            lw=0.8,
                            alpha=0.20,
                        )
                ax.axhline(0.0, color="0.75", lw=0.8, ls="--")

            ax.set_title(label)
            ax.grid(alpha=0.15)

            if j == 0:
                ax.set_ylabel("Value")
            if i == 2:
                ax.set_xlabel("Frequency (Hz)")

            if i == 0 and j == 0:
                handles = [
                    plt.Line2D(
                        [0], [0], color="black", lw=1.2, label="component"
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["spectral"],
                        lw=1.2,
                        label="spectral knots",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=method_colors["cholesky"],
                        lw=1.2,
                        label="cholesky knots",
                    ),
                ]
                ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.suptitle("3x3 Spectral Matrix Components with Knot Overlays", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _extract_psd_quantiles(
    idata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (freq, q05, q50, q95) as complex arrays from posterior_psd."""
    if not hasattr(idata, "posterior_psd"):
        raise ValueError("idata has no posterior_psd group.")
    group = idata.posterior_psd
    if "psd_matrix_real" not in group:
        raise ValueError("posterior_psd missing psd_matrix_real.")

    pcts = np.asarray(
        group["psd_matrix_real"].coords["percentile"].values, dtype=float
    )
    freq = np.asarray(
        group["psd_matrix_real"].coords["freq"].values, dtype=float
    )
    real = np.asarray(group["psd_matrix_real"].values, dtype=np.float64)
    imag = (
        np.asarray(group["psd_matrix_imag"].values, dtype=np.float64)
        if "psd_matrix_imag" in group
        else np.zeros_like(real)
    )

    def _grab(target: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(pcts - target)))
        return real[idx] + 1j * imag[idx]

    return freq, _grab(5.0), _grab(50.0), _grab(95.0)


def _panel_component(
    matrix: np.ndarray,
    i: int,
    j: int,
) -> np.ndarray:
    """Map matrix entry to requested 3x3 display convention."""
    if i == j:
        return np.real(matrix[:, i, j])
    if i < j:
        return np.real(matrix[:, i, j])
    return np.imag(matrix[:, i, j])


def _panel_label(i: int, j: int) -> str:
    if i == j:
        return f"S_{i+1}{j+1}"
    if i < j:
        return f"Re S_{i+1}{j+1}"
    return f"Im S_{i+1}{j+1}"


def _plot_posterior_comparison_3x3(
    *,
    freq: np.ndarray,
    truth: np.ndarray,
    spectral_q05: np.ndarray,
    spectral_q50: np.ndarray,
    spectral_q95: np.ndarray,
    cholesky_q05: np.ndarray,
    cholesky_q50: np.ndarray,
    cholesky_q95: np.ndarray,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            y_true = _panel_component(truth, i, j)
            y_s05 = _panel_component(spectral_q05, i, j)
            y_s50 = _panel_component(spectral_q50, i, j)
            y_s95 = _panel_component(spectral_q95, i, j)
            y_c05 = _panel_component(cholesky_q05, i, j)
            y_c50 = _panel_component(cholesky_q50, i, j)
            y_c95 = _panel_component(cholesky_q95, i, j)

            ax.fill_between(freq, y_s05, y_s95, color="#1f77b4", alpha=0.20)
            ax.plot(freq, y_s50, color="#1f77b4", lw=1.2)
            ax.fill_between(freq, y_c05, y_c95, color="#ff7f0e", alpha=0.18)
            ax.plot(freq, y_c50, color="#ff7f0e", lw=1.2)
            ax.plot(freq, y_true, color="black", lw=1.0)

            if i != j:
                ax.axhline(0.0, color="0.75", lw=0.8, ls="--")

            ax.set_title(_panel_label(i, j))
            ax.grid(alpha=0.15)
            if j == 0:
                ax.set_ylabel("Value")
            if i == 2:
                ax.set_xlabel("Frequency (Hz)")

            if i == 0 and j == 0:
                handles = [
                    plt.Line2D(
                        [0], [0], color="#1f77b4", lw=1.5, label="spectral q50"
                    ),
                    plt.Line2D(
                        [0], [0], color="#ff7f0e", lw=1.5, label="cholesky q50"
                    ),
                    plt.Line2D(
                        [0], [0], color="black", lw=1.2, label="true PSD"
                    ),
                ]
                ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.suptitle(
        "Posterior PSD Comparison (5/50/95%) in 3x3 Matrix Layout", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_posterior_for_scoring(
    *,
    ts: MultivariateTimeseries,
    n_knots: int,
    nb: int,
    window: str,
    outdir: str,
    scoring: str,
    n_samples: int,
    n_warmup: int,
    num_chains: int,
    vi_steps: int,
    true_psd: tuple[np.ndarray, np.ndarray],
) -> object:
    run_dir = os.path.join(outdir, f"posterior_{scoring}")
    os.makedirs(run_dir, exist_ok=True)
    return run_mcmc(
        data=ts,
        n_knots=n_knots,
        degree=2,
        diffMatrixOrder=2,
        n_samples=n_samples,
        n_warmup=n_warmup,
        num_chains=num_chains,
        outdir=run_dir,
        verbose=True,
        target_accept_prob=0.9,
        max_tree_depth=12,
        init_from_vi=True,
        vi_steps=vi_steps,
        vi_guide="lowrank:16",
        vi_lr=VI_LR,
        vi_psd_max_draws=128,
        Nb=nb,
        wishart_window=_parse_window(window),
        knot_kwargs={"method": "density", "scoring": scoring},
        compute_coherence_quantiles=False,
        true_psd=true_psd,
        max_save_bytes=20_000_000,
    )


def run_comparison(
    *,
    n_samples: int,
    nb: int,
    n_knots: int,
    seed: int,
    window: str,
    outdir: str,
    run_posterior: bool = False,
    posterior_samples: int = 200,
    posterior_warmup: int = 200,
    posterior_chains: int = 1,
    posterior_vi_steps: int = 5000,
) -> dict[str, object]:
    os.makedirs(outdir, exist_ok=True)

    t, x = _simulate_var_process(
        n_samples=n_samples,
        var_coeffs=VAR_COEFFS,
        sigma=SIGMA,
        seed=seed,
    )
    ts = MultivariateTimeseries(t=t, y=x)
    ts_for_posterior = MultivariateTimeseries(
        t=np.asarray(t, dtype=np.float64),
        y=np.asarray(x, dtype=np.float64),
    )
    ts_std = ts.standardise_for_psd()
    fft_data = ts_std.to_wishart_stats(
        Nb=nb,
        window=_parse_window(window),
    )

    freq = np.asarray(fft_data.freq, dtype=np.float64)
    p = int(fft_data.p)
    u_re = np.asarray(fft_data.u_re, dtype=np.float64)
    u_im = np.asarray(fft_data.u_im, dtype=np.float64)
    u_complex = u_re + 1j * u_im
    # Channel-space Wishart matrices used by the multivariate likelihood.
    y_np = U_to_Y(u_complex)
    # For 3x3 PSD-style visualization we want the empirical PSD in channel space.
    if fft_data.raw_psd is not None:
        s_matrix_plot = np.asarray(fft_data.raw_psd, dtype=np.complex128)
    else:
        s_matrix_plot = np.asarray(
            fft_data.empirical_psd.psd, dtype=np.complex128
        )
    true_psd = _calculate_true_var_psd_hz(
        freq, VAR_COEFFS, SIGMA, fs=DEFAULT_FS
    )

    log_delta_sq, theta = psd_to_cholesky_components(y_np / float(nb))

    scores_by_method: dict[str, dict[str, np.ndarray]] = {}
    knots_norm_by_method: dict[str, dict[str, np.ndarray]] = {}
    knots_hz_by_method: dict[str, dict[str, np.ndarray]] = {}
    summary_by_method: dict[str, dict[str, ScoreSummary]] = {}

    for method in METHODS:
        diag_scores, offdiag_re_scores, offdiag_im_scores = multivar_psd_knot_scores(
            y_np,
            nb,
            p,
            scoring=method,
            u_re=u_re if method == "spectral" else None,
            u_im=u_im if method == "spectral" else None,
        )
        theta_pairs = [(j, l) for j in range(1, p) for l in range(j)]
        if len(offdiag_re_scores) != len(theta_pairs):
            raise ValueError(
                "Expected "
                + f"{len(theta_pairs)} offdiag re scores, got {len(offdiag_re_scores)}."
            )
        if len(offdiag_im_scores) != len(theta_pairs):
            raise ValueError(
                "Expected "
                + f"{len(theta_pairs)} offdiag im scores, got {len(offdiag_im_scores)}."
            )
        method_scores: dict[str, np.ndarray] = {}
        method_knots_norm: dict[str, np.ndarray] = {}
        method_knots_hz: dict[str, np.ndarray] = {}
        method_summary: dict[str, ScoreSummary] = {}

        for idx in range(p):
            component = f"diag_{idx}"
            score = np.asarray(diag_scores[idx], dtype=np.float64)
            knots_norm, knots_hz = _compute_density_knots(
                freq=freq,
                score=score,
                n_knots=n_knots,
            )
            method_scores[component] = score
            method_knots_norm[component] = knots_norm
            method_knots_hz[component] = knots_hz
            method_summary[component] = _summarise_score(
                method, component, score
            )

        for theta_idx, pair in enumerate(theta_pairs):
            j, l = pair
            component_re = f"theta_re_{j}_{l}"
            offdiag_re_score = np.asarray(
                offdiag_re_scores[theta_idx], dtype=np.float64
            )
            offdiag_re_knots_norm, offdiag_re_knots_hz = _compute_density_knots(
                freq=freq,
                score=offdiag_re_score,
                n_knots=n_knots,
            )
            method_scores[component_re] = offdiag_re_score
            method_knots_norm[component_re] = offdiag_re_knots_norm
            method_knots_hz[component_re] = offdiag_re_knots_hz
            method_summary[component_re] = _summarise_score(
                method, component_re, offdiag_re_score
            )

            component_im = f"theta_im_{j}_{l}"
            offdiag_im_score = np.asarray(
                offdiag_im_scores[theta_idx], dtype=np.float64
            )
            offdiag_im_knots_norm, offdiag_im_knots_hz = _compute_density_knots(
                freq=freq,
                score=offdiag_im_score,
                n_knots=n_knots,
            )
            method_scores[component_im] = offdiag_im_score
            method_knots_norm[component_im] = offdiag_im_knots_norm
            method_knots_hz[component_im] = offdiag_im_knots_hz
            method_summary[component_im] = _summarise_score(
                method, component_im, offdiag_im_score
            )

        scores_by_method[method] = method_scores
        knots_norm_by_method[method] = method_knots_norm
        knots_hz_by_method[method] = method_knots_hz
        summary_by_method[method] = method_summary

    score_plot_path = os.path.join(outdir, "knot_score_panels.png")
    components = [f"diag_{i}" for i in range(p)] + [
        f"theta_re_{j}_{l}" for j in range(1, p) for l in range(j)
    ] + [
        f"theta_im_{j}_{l}" for j in range(1, p) for l in range(j)
    ]
    _plot_score_panels(
        freq=freq,
        components=components,
        scores_by_method=scores_by_method,
        knots_hz_by_method=knots_hz_by_method,
        summary_by_method=summary_by_method,
        out_path=score_plot_path,
    )

    cholesky_plot_path = os.path.join(outdir, "cholesky_component_knots.png")
    _plot_cholesky_component_panels(
        freq=freq,
        log_delta_sq=log_delta_sq,
        theta=theta,
        diag_knots_hz={
            method: {
                f"diag_{i}": knots_hz_by_method[method][f"diag_{i}"]
                for i in range(p)
            }
            for method in METHODS
        },
        offdiag_re_knots_hz={
            method: {
                (j, l): knots_hz_by_method[method][f"theta_re_{j}_{l}"]
                for j in range(1, p)
                for l in range(j)
            }
            for method in METHODS
        },
        offdiag_im_knots_hz={
            method: {
                (j, l): knots_hz_by_method[method][f"theta_im_{j}_{l}"]
                for j in range(1, p)
                for l in range(j)
            }
            for method in METHODS
        },
        out_path=cholesky_plot_path,
    )

    s_matrix_plot_path = os.path.join(outdir, "s_matrix_knots_3x3.png")
    _plot_s_matrix_knots_grid(
        freq=freq,
        s_matrix=s_matrix_plot,
        diag_knots_hz={
            method: {
                f"diag_{i}": knots_hz_by_method[method][f"diag_{i}"]
                for i in range(p)
            }
            for method in METHODS
        },
        offdiag_re_knots_hz={
            method: {
                (j, l): knots_hz_by_method[method][f"theta_re_{j}_{l}"]
                for j in range(1, p)
                for l in range(j)
            }
            for method in METHODS
        },
        offdiag_im_knots_hz={
            method: {
                (j, l): knots_hz_by_method[method][f"theta_im_{j}_{l}"]
                for j in range(1, p)
                for l in range(j)
            }
            for method in METHODS
        },
        out_path=s_matrix_plot_path,
    )

    knot_rows: list[dict[str, object]] = []
    for method in METHODS:
        for component in components:
            knots_norm = knots_norm_by_method[method][component]
            knots_hz = knots_hz_by_method[method][component]
            for knot_idx, (k_norm, k_hz) in enumerate(
                zip(knots_norm, knots_hz)
            ):
                knot_rows.append(
                    {
                        "method": method,
                        "component": component,
                        "knot_index": knot_idx,
                        "knot_norm": float(k_norm),
                        "knot_hz": float(k_hz),
                    }
                )
    knots_csv_path = os.path.join(outdir, "knot_locations.csv")
    _write_csv(
        knots_csv_path,
        fieldnames=[
            "method",
            "component",
            "knot_index",
            "knot_norm",
            "knot_hz",
        ],
        rows=knot_rows,
    )

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        for component in components:
            metric = summary_by_method[method][component]
            summary_rows.append(
                {
                    "method": metric.method,
                    "component": metric.component,
                    "top10_mass": metric.top10_mass,
                    "effective_bins": metric.effective_bins,
                    "effective_fraction": metric.effective_fraction,
                    "gini": metric.gini,
                }
            )
    summary_csv_path = os.path.join(outdir, "score_concentration_summary.csv")
    _write_csv(
        summary_csv_path,
        fieldnames=[
            "method",
            "component",
            "top10_mass",
            "effective_bins",
            "effective_fraction",
            "gini",
        ],
        rows=summary_rows,
    )

    summary_json_path = os.path.join(outdir, "run_summary.json")
    posterior_plot_path = None
    if run_posterior:
        spectral_idata = _run_posterior_for_scoring(
            ts=ts_for_posterior,
            n_knots=n_knots,
            nb=nb,
            window=window,
            outdir=outdir,
            scoring="spectral",
            n_samples=posterior_samples,
            n_warmup=posterior_warmup,
            num_chains=posterior_chains,
            vi_steps=posterior_vi_steps,
            true_psd=(freq, true_psd),
        )
        cholesky_idata = _run_posterior_for_scoring(
            ts=ts_for_posterior,
            n_knots=n_knots,
            nb=nb,
            window=window,
            outdir=outdir,
            scoring="cholesky",
            n_samples=posterior_samples,
            n_warmup=posterior_warmup,
            num_chains=posterior_chains,
            vi_steps=posterior_vi_steps,
            true_psd=(freq, true_psd),
        )
        freq_s, s_q05, s_q50, s_q95 = _extract_psd_quantiles(spectral_idata)
        freq_c, c_q05, c_q50, c_q95 = _extract_psd_quantiles(cholesky_idata)
        if freq_s.shape != freq_c.shape or not np.allclose(freq_s, freq_c):
            raise ValueError(
                "Spectral/cholesky posterior frequency grids differ."
            )
        posterior_plot_path = os.path.join(
            outdir, "posterior_comparison_3x3.png"
        )
        _plot_posterior_comparison_3x3(
            freq=freq_s,
            truth=true_psd,
            spectral_q05=s_q05,
            spectral_q50=s_q50,
            spectral_q95=s_q95,
            cholesky_q05=c_q05,
            cholesky_q50=c_q50,
            cholesky_q95=c_q95,
            out_path=posterior_plot_path,
        )

    run_summary: dict[str, object] = {
        "seed": seed,
        "N": n_samples,
        "Nb": nb,
        "K": n_knots,
        "window": window,
        "freq_min_hz": float(freq[0]),
        "freq_max_hz": float(freq[-1]),
        "n_freq": int(freq.size),
        "plots": {
            "score_panels": score_plot_path,
            "cholesky_components": cholesky_plot_path,
            "s_matrix_knots_3x3": s_matrix_plot_path,
            "posterior_comparison_3x3": posterior_plot_path,
        },
        "tables": {
            "knot_locations_csv": knots_csv_path,
            "score_summary_csv": summary_csv_path,
        },
    }
    with open(summary_json_path, "w") as handle:
        json.dump(run_summary, handle, indent=2)

    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAR(2)-3D knot scoring diagnostics for spectral vs cholesky."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--N", type=int, default=2048, help="Number of simulated time samples."
    )
    parser.add_argument(
        "--Nb",
        type=int,
        default=4,
        help="Wishart blocks used in preprocessing.",
    )
    parser.add_argument("--K", type=int, default=20, help="Number of knots.")
    parser.add_argument(
        "--window",
        type=str,
        default=DEFAULT_WINDOW,
        help="Window name: rect|hann|tukey_<alpha>.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=(
            "docs/studies/multivar_psd/out_knot_diagnostics/"
            "var2_3d_seed0_N2048_Nb4_K20"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--with-posterior",
        action="store_true",
        help="Also run spectral/cholesky posterior fits and save 3x3 posterior comparison.",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=200,
        help="Posterior samples per chain for --with-posterior.",
    )
    parser.add_argument(
        "--posterior-warmup",
        type=int,
        default=200,
        help="Warmup samples per chain for --with-posterior.",
    )
    parser.add_argument(
        "--posterior-chains",
        type=int,
        default=1,
        help="Number of MCMC chains for --with-posterior.",
    )
    parser.add_argument(
        "--posterior-vi-steps",
        type=int,
        default=5000,
        help="VI init steps for --with-posterior.",
    )
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    summary = run_comparison(
        n_samples=int(args.N),
        nb=int(args.Nb),
        n_knots=int(args.K),
        seed=int(args.seed),
        window=str(args.window),
        outdir=outdir,
        run_posterior=bool(args.with_posterior),
        posterior_samples=int(args.posterior_samples),
        posterior_warmup=int(args.posterior_warmup),
        posterior_chains=int(args.posterior_chains),
        posterior_vi_steps=int(args.posterior_vi_steps),
    )
    print("Saved knot diagnostics.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
