"""Per-frequency diagnostic plots when a true PSD is available.

Provides two p×p matrix figures:
- ``plot_riae_vs_freq``: elementwise relative absolute error |q50 - true| / |true|
- ``plot_coverage_vs_freq``: rolling 90%-CI coverage fraction per element
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .base import interior_frequency_slice


def _complex_to_real(mat: np.ndarray) -> np.ndarray:
    """Mirror of diagnostics._utils._complex_to_real (avoids circular import).

    Upper triangle (i<=j) → real part; lower triangle (i>j) → imaginary part.
    """
    arr = np.asarray(mat)
    if not np.iscomplexobj(arr):
        return arr.copy()
    n = arr.shape[-1]
    upper = np.triu(np.ones((n, n), dtype=bool))
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    out = np.where(upper, arr.real, 0.0)
    out = np.where(lower, arr.imag, out)
    return out


def _extract_posterior_quantiles(
    idata,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (freq, q05, q50, q95) from idata.posterior_psd, or None."""
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is None or "psd_matrix_real" not in psd_group:
        return None

    real_arr = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    imag_arr = np.zeros_like(real_arr)
    if "psd_matrix_imag" in psd_group:
        imag_arr = np.asarray(
            psd_group["psd_matrix_imag"].values, dtype=np.float64
        )

    pcts = np.asarray(
        psd_group["psd_matrix_real"].coords.get("percentile", []), dtype=float
    )
    freq = np.asarray(
        psd_group["psd_matrix_real"].coords.get("freq", []), dtype=float
    )

    def _get(target: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(pcts - target)))
        return real_arr[idx] + 1j * imag_arr[idx]  # (N, p, p) complex

    if pcts.size < 3:
        return None
    return freq, _get(5.0), _get(50.0), _get(95.0)


def _panel_label(i: int, j: int, p: int) -> str:
    """Return axis label: 'Re S_{ij}' on upper, 'Im S_{ij}' on lower."""
    if i <= j:
        return rf"Re $S_{{{i}{j}}}$"
    else:
        return rf"Im $S_{{{j}{i}}}$"


def plot_riae_vs_freq(
    idata,
    true_psd: np.ndarray,
    freq: Optional[np.ndarray] = None,
    *,
    outdir: Optional[str] = None,
    fname: str = "riae_vs_freq.png",
    rolling_window: int = 1,
) -> plt.Figure:
    """Plot elementwise relative error |q50 - true| / |true| vs frequency.

    Parameters
    ----------
    idata:
        ArviZ ``InferenceData`` with a ``posterior_psd`` group.
    true_psd:
        True PSD matrix, shape ``(N, p, p)`` complex or real.
    freq:
        Frequency array ``(N,)``. Extracted from idata if None.
    outdir:
        Directory to save the figure. Skipped when None.
    fname:
        Filename for the saved figure.
    rolling_window:
        Optional smoothing window applied to the error curve.
    """
    result = _extract_posterior_quantiles(idata)
    if result is None:
        raise ValueError(
            "idata has no posterior_psd group with percentile quantiles."
        )
    freq_idata, q05, q50, q95 = result
    if freq is None:
        freq = freq_idata
    freq_idx = interior_frequency_slice(len(freq))
    freq = np.asarray(freq)[freq_idx]
    q05 = q05[freq_idx, ...]
    q50 = q50[freq_idx, ...]
    q95 = q95[freq_idx, ...]

    true_arr = np.asarray(true_psd, dtype=np.complex128)
    true_arr = true_arr[freq_idx, ...]
    N, p, _ = true_arr.shape

    # Convert everything to the real representation used by coverage
    def _c2r_stack(mat: np.ndarray) -> np.ndarray:
        """Apply _complex_to_real along the frequency axis → (N, p, p) float."""
        out = np.empty((mat.shape[0], p, p), dtype=np.float64)
        for k in range(mat.shape[0]):
            out[k] = _complex_to_real(mat[k])
        return out

    q50_r = _c2r_stack(q50)
    true_r = _c2r_stack(true_arr)

    eps = np.finfo(float).eps

    fig, axes = plt.subplots(p, p, figsize=(4 * p, 3 * p), squeeze=False)
    fig.suptitle(
        "Relative error vs frequency  |q50 − true| / |true|", fontsize=13
    )

    for i in range(p):
        for j in range(p):
            ax = axes[i][j]
            err = np.abs(q50_r[:, i, j] - true_r[:, i, j]) / (
                np.abs(true_r[:, i, j]) + eps
            )
            if rolling_window > 1:
                kernel = np.ones(rolling_window) / rolling_window
                err = np.convolve(err, kernel, mode="same")
            ax.plot(freq, err, lw=0.8, color="C3")
            ax.set_xlabel("Frequency" if i == p - 1 else "")
            ax.set_ylabel("Rel. error" if j == 0 else "")
            ax.set_title(_panel_label(i, j, p), fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(True, lw=0.3, alpha=0.5)

    fig.tight_layout()
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_coverage_vs_freq(
    idata,
    true_psd: np.ndarray,
    freq: Optional[np.ndarray] = None,
    *,
    outdir: Optional[str] = None,
    fname: str = "coverage_vs_freq.png",
    rolling_window: int = 20,
) -> plt.Figure:
    """Plot elementwise rolling 90%-CI coverage fraction vs frequency.

    For each (i,j) panel a binary coverage indicator (1 = covered, 0 = not)
    is smoothed with a rolling mean of ``rolling_window`` bins, so the curve
    lies in [0, 1] and the nominal level (0.90) is shown as a dashed line.

    Parameters
    ----------
    idata:
        ArviZ ``InferenceData`` with a ``posterior_psd`` group.
    true_psd:
        True PSD matrix, shape ``(N, p, p)`` complex or real.
    freq:
        Frequency array ``(N,)``. Extracted from idata if None.
    outdir:
        Directory to save the figure. Skipped when None.
    fname:
        Filename for the saved figure.
    rolling_window:
        Width (in frequency bins) of the rolling-mean smoothing window.
    """
    result = _extract_posterior_quantiles(idata)
    if result is None:
        raise ValueError(
            "idata has no posterior_psd group with percentile quantiles."
        )
    freq_idata, q05, q50, q95 = result
    if freq is None:
        freq = freq_idata
    freq_idx = interior_frequency_slice(len(freq))
    freq = np.asarray(freq)[freq_idx]
    q05 = q05[freq_idx, ...]
    q50 = q50[freq_idx, ...]
    q95 = q95[freq_idx, ...]

    true_arr = np.asarray(true_psd, dtype=np.complex128)
    true_arr = true_arr[freq_idx, ...]
    N, p, _ = true_arr.shape

    def _c2r_stack(mat: np.ndarray) -> np.ndarray:
        out = np.empty((mat.shape[0], p, p), dtype=np.float64)
        for k in range(mat.shape[0]):
            out[k] = _complex_to_real(mat[k])
        return out

    q05_r = _c2r_stack(q05)
    q95_r = _c2r_stack(q95)
    true_r = _c2r_stack(true_arr)

    kernel = np.ones(max(1, rolling_window)) / max(1, rolling_window)

    fig, axes = plt.subplots(p, p, figsize=(4 * p, 3 * p), squeeze=False)
    fig.suptitle(
        f"Rolling 90%-CI coverage vs frequency  (window={rolling_window} bins)",
        fontsize=13,
    )

    for i in range(p):
        for j in range(p):
            ax = axes[i][j]
            covered = (
                (true_r[:, i, j] >= q05_r[:, i, j])
                & (true_r[:, i, j] <= q95_r[:, i, j])
            ).astype(float)
            if rolling_window > 1:
                smoothed = np.convolve(covered, kernel, mode="same")
            else:
                smoothed = covered
            ax.plot(freq, smoothed, lw=0.9, color="C0", label="coverage")
            ax.axhline(0.90, color="k", lw=0.8, ls="--", label="nominal 90%")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Frequency" if i == p - 1 else "")
            ax.set_ylabel("Coverage" if j == 0 else "")
            ax.set_title(_panel_label(i, j, p), fontsize=9)
            ax.grid(True, lw=0.3, alpha=0.5)
            if i == 0 and j == p - 1:
                ax.legend(fontsize=7)

    fig.tight_layout()
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_true_psd_diagnostics(
    idata,
    true_psd: np.ndarray,
    freq: Optional[np.ndarray] = None,
    *,
    outdir: Optional[str] = None,
    rolling_window: int = 20,
) -> None:
    """Convenience wrapper: save both RIAE and coverage diagnostic figures.

    Parameters
    ----------
    idata:
        ArviZ ``InferenceData`` with a ``posterior_psd`` group.
    true_psd:
        True PSD matrix, shape ``(N, p, p)`` complex or real.
    freq:
        Frequency array ``(N,)``. Extracted from idata if None.
    outdir:
        Output directory. When None the figures are not saved.
    rolling_window:
        Rolling window for the coverage plot.
    """
    riae_fig = plot_riae_vs_freq(idata, true_psd, freq, outdir=outdir)
    cov_fig = plot_coverage_vs_freq(
        idata, true_psd, freq, outdir=outdir, rolling_window=rolling_window
    )
    plt.close(riae_fig)
    plt.close(cov_fig)
