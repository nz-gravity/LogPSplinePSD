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

from log_psplines.coarse_grain import (
    CoarseGrainConfig,
    coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.example_datasets.lisa_data import (
    LISAData,
    compute_periodograms,
    periodogram_covariance,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FMIN, FMAX = 1e-4, 1e-1
COARSE_CFG = CoarseGrainConfig(
    enabled=True,
    f_transition=5e-3,
    n_log_bins=200,
    f_min=FMIN,
    f_max=FMAX,
)


def infer_blocks(n_time: int) -> int:
    target = max(1, 2 ** int(np.round(np.log2(n_time / (24 * 7)))))
    while target > 1 and n_time % target != 0:
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
    flat = matrix.reshape(matrix.shape[0], -1)
    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_src, flat[:, idx].real)
            for idx in range(flat.shape[1])
        ]
    ).T
    imag_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_src, flat[:, idx].imag)
            for idx in range(flat.shape[1])
        ]
    ).T
    return (real_interp + 1j * imag_interp).reshape(
        (freq_tgt.size,) + matrix.shape[1:]
    )


def main() -> None:
    lisa = LISAData.load()
    freq_raw, auto_raw, cross_raw, _ = compute_periodograms(
        lisa.time, lisa.data[:, 0], lisa.data[:, 1], lisa.data[:, 2]
    )
    raw_matrix = periodogram_covariance(auto_raw, cross_raw)

    ts = MultivariateTimeseries(y=lisa.data, t=lisa.time)
    ts_std = ts.standardise_for_psd()
    n_blocks = infer_blocks(ts_std.y.shape[0])
    fft = ts_std.to_wishart_stats(n_blocks=n_blocks, fmin=FMIN, fmax=FMAX)

    spec = compute_binning_structure(
        fft.freq,
        f_transition=COARSE_CFG.f_transition,
        n_log_bins=COARSE_CFG.n_log_bins,
        f_min=COARSE_CFG.f_min,
        f_max=COARSE_CFG.f_max,
    )
    fft, _ = coarse_grain_multivar_fft(fft, spec)

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


if __name__ == "__main__":
    main()
