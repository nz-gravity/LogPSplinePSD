"""Pre-flight consistency check for LISA preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.example_datasets.lisa_data import (
    LISAData,
    coherence,
    compute_periodograms,
    periodogram_covariance,
)
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FMIN, FMAX = 1e-4, 1e-1
COARSE_CFG = CoarseGrainConfig(
    enabled=True,
    Nc=200,
)


def infer_blocks(n: int) -> int:
    target = max(1, 2 ** int(np.round(np.log2(n / (24 * 7)))))
    while target > 1 and n % target != 0:
        target //= 2
    while target > 4:
        target //= 2
    return max(1, target)


def scale_processed_psd(
    psd: np.ndarray, scaling_factor: float, channel_stds: np.ndarray | None
) -> np.ndarray:
    sf = float(scaling_factor or 1.0)
    if channel_stds is None:
        return psd
    scale_matrix = np.outer(channel_stds, channel_stds)
    return psd * (scale_matrix / sf)


def interp_complex_matrix(
    freq_src: np.ndarray, freq_tgt: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    matrix = np.asarray(matrix)

    sort_idx = np.argsort(freq_src)
    freq_sorted = freq_src[sort_idx]
    matrix_sorted = matrix[sort_idx]
    freq_unique, uniq_idx = np.unique(freq_sorted, return_index=True)
    matrix_unique = matrix_sorted[uniq_idx]

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


def summarize_ratio(label: str, ratio: np.ndarray) -> None:
    clean = ratio[np.isfinite(ratio)]
    if clean.size == 0:
        print(f"{label}: no finite entries to summarize.")
        return
    pct = np.percentile(clean, [5, 50, 95])
    print(f"{label}: p05={pct[0]:.3g}, p50={pct[1]:.3g}, p95={pct[2]:.3g}")


def main() -> None:
    lisa = LISAData.load()
    freq_raw, auto_raw, cross_raw, _ = compute_periodograms(
        lisa.time, lisa.data[:, 0], lisa.data[:, 1], lisa.data[:, 2]
    )
    raw_matrix = periodogram_covariance(auto_raw, cross_raw)

    ts = MultivariateTimeseries(y=lisa.data, t=lisa.time)
    ts_std = ts.standardise_for_psd()
    Nb = infer_blocks(ts_std.y.shape[0])
    fft = ts_std.to_wishart_stats(Nb=Nb, fmin=FMIN, fmax=FMAX)

    spec = compute_binning_structure(
        fft.freq,
        Nc=COARSE_CFG.Nc,
    )
    fft = apply_coarse_grain_multivar_fft(fft, spec)

    psd_phys = scale_processed_psd(
        fft.raw_psd, fft.scaling_factor, fft.channel_stds
    )
    freq_proc = np.asarray(fft.freq)
    raw_interp = interp_complex_matrix(freq_raw, freq_proc, raw_matrix)

    diag_ratio = np.abs(psd_phys[:, 0, 0]) / np.abs(raw_interp[:, 0, 0])
    print("Median diag ratio (X):", np.median(diag_ratio))
    print("Max abs ratio (X):", np.max(diag_ratio))

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    channels = ["X", "Y", "Z"]
    combos = [(1, 0, "XY"), (2, 0, "ZX"), (2, 1, "YZ")]

    for idx, ax in enumerate(axes[0]):
        ax.loglog(freq_proc, psd_phys[:, idx, idx].real, label="Processed")
        ax.loglog(
            freq_proc,
            np.abs(raw_interp[:, idx, idx]),
            "--",
            label="Raw interp",
        )
        ax.set_title(f"{channels[idx]} PSD")
        ax.set_ylabel("PSD [1/Hz]")

    for (i, j, label), ax in zip(combos, axes[1]):
        ax.loglog(freq_proc, np.abs(psd_phys[:, i, j]), label="Processed")
        ax.loglog(
            freq_proc, np.abs(raw_interp[:, i, j]), "--", label="Raw interp"
        )
        ax.set_title(f"|{label}| CSD")
        ax.set_ylabel("|CSD| [1/Hz]")

    for ax in axes[-1]:
        ax.set_xlabel("Frequency [Hz]")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = RESULTS_DIR / "preprocessing_check.png"
    fig.savefig(out, dpi=200)
    print(f"Saved comparison plot to {out}")

    true_matrix = np.asarray(lisa.true_matrix)
    welch_matrix = np.asarray(lisa.matrix)
    freq_true = np.asarray(lisa.freq)

    # Quick ratio checks between analytic truth and Welch estimates.
    for idx, label in enumerate(channels):
        ratio = welch_matrix[:, idx, idx].real / true_matrix[:, idx, idx].real
        summarize_ratio(f"{label} PSD Welch/True", ratio)
    for i, j, label in combos:
        ratio = np.abs(welch_matrix[:, i, j]) / np.abs(true_matrix[:, i, j])
        summarize_ratio(f"{label} |CSD| Welch/True", ratio)

    # Compare full-length periodogram vs true/Welch on the periodogram grid.
    true_on_raw = interp_complex_matrix(freq_true, freq_raw, true_matrix)
    welch_on_raw = interp_complex_matrix(freq_true, freq_raw, welch_matrix)
    for idx, label in enumerate(channels):
        ratio = raw_matrix[:, idx, idx].real / true_on_raw[:, idx, idx].real
        summarize_ratio(f"{label} PSD periodogram/True", ratio)
        ratio = raw_matrix[:, idx, idx].real / welch_on_raw[:, idx, idx].real
        summarize_ratio(f"{label} PSD periodogram/Welch", ratio)
    for i, j, label in combos:
        ratio = np.abs(raw_matrix[:, i, j]) / np.abs(true_on_raw[:, i, j])
        summarize_ratio(f"{label} |CSD| periodogram/True", ratio)
        ratio = np.abs(raw_matrix[:, i, j]) / np.abs(welch_on_raw[:, i, j])
        summarize_ratio(f"{label} |CSD| periodogram/Welch", ratio)

    fig2, axes = plt.subplots(3, 3, figsize=(11, 7), sharex=True)

    # PSD_ii along the diagonal
    for idx, ax in enumerate(axes[0]):
        ax.loglog(freq_true, true_matrix[:, idx, idx].real, label="True")
        ax.loglog(
            freq_true,
            welch_matrix[:, idx, idx].real,
            "--",
            label="Welch",
        )
        ax.set_title(f"{channels[idx]} PSD")
        ax.set_ylabel("PSD [1/Hz]")
        if idx == 0:
            ax.legend(fontsize=8)

    # |CSD_ij| off-diagonal
    for (i, j, label), ax in zip(combos, axes[1]):
        ax.loglog(freq_true, np.abs(true_matrix[:, i, j]), label="True")
        ax.loglog(
            freq_true,
            np.abs(welch_matrix[:, i, j]),
            "--",
            label="Welch",
        )
        ax.set_title(f"|{label}| CSD")
        ax.set_ylabel("|CSD| [1/Hz]")
        if label == "XY":
            ax.legend(fontsize=8)

    # Coherences
    for (i, j, label), ax in zip(combos, axes[2]):
        coh_true = coherence(
            true_matrix[:, i, i].real,
            true_matrix[:, j, j].real,
            true_matrix[:, i, j],
        )
        coh_welch = coherence(
            welch_matrix[:, i, i].real,
            welch_matrix[:, j, j].real,
            welch_matrix[:, i, j],
        )
        ax.semilogx(freq_true, coh_true, label="True")
        ax.semilogx(freq_true, coh_welch, "--", label="Welch")
        ax.set_title(f"{label} coherence")
        ax.set_ylabel("Coherence")
        ax.set_ylim(0.0, 1.05)
        if label == "XY":
            ax.legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Frequency [Hz]")

    fig2.tight_layout()
    out2 = RESULTS_DIR / "preprocessing_true_psd_coherence.png"
    fig2.savefig(out2, dpi=200)
    print(f"Saved true PSD/CSD/coherence plot to {out2}")


if __name__ == "__main__":
    main()
