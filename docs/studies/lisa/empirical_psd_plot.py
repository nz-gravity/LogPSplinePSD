"""Compute and plot the same empirical PSD matrix used in psd_matrix.png.

This script *recomputes* the empirical spectral matrix from the time series
using the same preprocessing pipeline as the multivariate sampler:

- Per-channel standardisation via ``MultivariateTimeseries.standardise_for_psd``.
- Blocked Wishart FFT statistics via ``to_wishart_stats(..., window="hann")``.
- Optional coarse-graining using ``compute_binning_structure`` +
  ``apply_coarse_grain_multivar_fft``.

The resulting empirical matrix is the same quantity stored as
``idata.observed_data["periodogram"]`` and plotted as the dashed "Empirical"
overlay in ``psd_matrix.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.coarse_grain import (  # noqa: E402
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.datatypes import MultivariateTimeseries  # noqa: E402

C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LASER_FREQ = 2.81e14  # Hz


def _strain_to_freq_scale(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=float)
    return (2.0 * np.pi * freq * LASER_FREQ * L_ARM / C_LIGHT) ** 2


def _interp_complex_matrix(
    freq_src: np.ndarray, matrix: np.ndarray, freq_tgt: np.ndarray
) -> np.ndarray:
    """Interpolate a complex matrix along frequency with duplicate-safe handling."""
    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    matrix = np.asarray(matrix)

    sort_idx = np.argsort(freq_src)
    freq_sorted = freq_src[sort_idx]
    matrix_sorted = matrix[sort_idx]
    freq_unique, uniq_idx = np.unique(freq_sorted, return_index=True)
    matrix_unique = matrix_sorted[uniq_idx]

    if freq_unique.shape == freq_tgt.shape and np.allclose(
        freq_unique, freq_tgt
    ):
        return np.asarray(matrix_unique)

    flat = matrix_unique.reshape(matrix_unique.shape[0], -1)
    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, idx].real)
            for idx in range(flat.shape[1])
        ]
    ).T
    imag_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, idx].imag)
            for idx in range(flat.shape[1])
        ]
    ).T
    return (real_interp + 1j * imag_interp).reshape(
        (freq_tgt.size,) + matrix_unique.shape[1:]
    )


def _coherence(matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    s_ii = np.asarray(matrix[:, i, i])
    s_jj = np.asarray(matrix[:, j, j])
    s_ij = np.asarray(matrix[:, i, j])
    denom = np.abs(s_ii) * np.abs(s_jj)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = (np.abs(s_ij) ** 2) / denom
    return np.clip(coh, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute and plot the empirical PSD/CSD matrix used in psd_matrix.png."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "lisa"
        / "lisatools_synth_data.npz",
        help="Path to lisatools_synth_data.npz (time series + true PSD matrix).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "lisa"
        / "empirical_psd_debug.png",
        help="Output PNG path.",
    )
    parser.add_argument("--fmin", type=float, default=1e-4)
    parser.add_argument("--fmax", type=float, default=1e-1)
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=0,
        help="Number of time blocks for Wishart averaging. When 0, infer from NPZ metadata if present.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help="Window applied per block (passed to scipy.signal.windows.get_window); use 'none' for rectangular.",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        default=True,
        help="Apply the same coarse-graining used in the sampler.",
    )
    parser.add_argument(
        "--no-coarse",
        action="store_false",
        dest="coarse",
        help="Disable coarse-graining (debug raw Wishart periodogram).",
    )
    parser.add_argument("--coarse_Nc", type=int, default=200)
    parser.add_argument(
        "--apply_strain_to_freq_scale",
        action="store_true",
        default=True,
        help="Apply the same strain→frequency-noise scale used in psd_matrix.png.",
    )
    parser.add_argument(
        "--no-apply_strain_to_freq_scale",
        action="store_false",
        dest="apply_strain_to_freq_scale",
        help="Plot in strain PSD units (1/Hz) without applying the strain→frequency-noise scaling.",
    )
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"{args.npz} not found.")

    with np.load(args.npz, allow_pickle=False) as synth:
        t_full = np.asarray(synth["time"], dtype=float)
        y_full = np.asarray(synth["data"], dtype=float)
        freq_true = np.asarray(synth["freq_true"], dtype=float)
        true_matrix = np.asarray(synth["true_matrix"])
        block_len_samples = (
            int(synth["block_len_samples"])
            if "block_len_samples" in synth.files
            else None
        )

    n_time = int(y_full.shape[0])
    if args.n_blocks and args.n_blocks > 0:
        n_blocks = int(args.n_blocks)
    elif block_len_samples is not None and block_len_samples > 0:
        n_blocks = max(1, n_time // int(block_len_samples))
    else:
        n_blocks = 1

    n_used = (n_time // n_blocks) * n_blocks
    if n_used != n_time:
        t_full = t_full[:n_used]
        y_full = y_full[:n_used]

    ts = MultivariateTimeseries(y=y_full, t=t_full).standardise_for_psd()
    window = (
        None
        if args.window.lower() in ("none", "rect", "boxcar")
        else args.window
    )
    fft = ts.to_wishart_stats(
        n_blocks=n_blocks,
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        window=window,
    )

    freq = np.asarray(fft.freq, dtype=float)
    fft_used = fft
    if args.coarse:
        cfg = CoarseGrainConfig(
            enabled=True,
            Nc=int(args.coarse_Nc),
            f_min=float(args.fmin),
            f_max=float(args.fmax),
        )
        spec = compute_binning_structure(
            fft.freq,
            Nc=cfg.Nc,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        fft_used, _ = apply_coarse_grain_multivar_fft(fft, spec)
        freq = np.asarray(fft_used.freq, dtype=float)

    if fft_used.raw_psd is None:
        raise RuntimeError("Expected raw_psd to be present on MultivarFFT.")

    emp_raw = np.asarray(fft_used.raw_psd, dtype=np.complex128)

    # Match the observed_data.periodogram scaling in to_arviz: remove the global
    # scaling_factor then apply per-channel std outer-product.
    sf = float(getattr(fft_used, "scaling_factor", 1.0) or 1.0)
    channel_stds = getattr(fft_used, "channel_stds", None)
    if channel_stds is not None:
        channel_stds = np.asarray(channel_stds, dtype=float)
        emp = (emp_raw / sf) * np.outer(channel_stds, channel_stds)[None, :, :]
    else:
        emp = emp_raw

    true = _interp_complex_matrix(freq_true, true_matrix, freq)

    scale = (
        _strain_to_freq_scale(freq)
        if args.apply_strain_to_freq_scale
        else None
    )
    if scale is not None:
        emp_plot = emp * scale[:, None, None]
        true_plot = true * scale[:, None, None] if true is not None else None
        unit = "Hz^2/Hz"
    else:
        emp_plot = emp
        true_plot = true
        unit = "1/Hz"

    def _summarize_diag(label: str, diag: np.ndarray) -> str:
        diag = np.asarray(diag, dtype=float)
        diag = diag[np.isfinite(diag)]
        if diag.size == 0:
            return f"{label}: (no finite values)"
        p05, p50, p95 = np.percentile(diag, [5.0, 50.0, 95.0])
        return (
            f"{label}: p05={p05:.3e}, p50={p50:.3e}, p95={p95:.3e}, "
            f"min={diag.min():.3e}, max={diag.max():.3e}"
        )

    emp_diag = np.abs(np.asarray(emp_plot[:, 0, 0].real))
    print(f"Diag PSD (ch0) [{unit}]")
    print(_summarize_diag("  Empirical", emp_diag))
    if true_plot is not None:
        true_diag = np.abs(np.asarray(true_plot[:, 0, 0].real))
        print(_summarize_diag("  True", true_diag))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    channels = ["X", "Y", "Z"]
    pairs = [(0, 1, "XY"), (0, 2, "ZX"), (1, 2, "YZ")]

    for idx, ax in enumerate(axes[0]):
        ax.loglog(
            freq,
            np.abs(emp_plot[:, idx, idx].real),
            "--",
            color="0.3",
            label="Empirical",
        )
        if true_plot is not None:
            ax.loglog(
                freq,
                np.abs(true_plot[:, idx, idx].real),
                color="k",
                lw=1.2,
                label="True",
            )
        ax.set_title(f"{channels[idx]} PSD")
        ax.set_ylabel(f"PSD [{unit}]")
        ax.grid(True, which="both", alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8)

    for (i, j, label), ax in zip(pairs, axes[1]):
        coh_emp = _coherence(emp, i, j)
        ax.semilogx(freq, coh_emp, color="0.3", alpha=0.7, label="Empirical")
        if true is not None:
            coh_true = _coherence(true, i, j)
            ax.semilogx(freq, coh_true, color="k", lw=1.2, label="True")
        ax.set_title(f"{label} coherence")
        ax.set_ylabel("Coherence")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, which="both", alpha=0.2)
        if label == "XY":
            ax.legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved empirical debug plot to {args.out}")


if __name__ == "__main__":
    main()
