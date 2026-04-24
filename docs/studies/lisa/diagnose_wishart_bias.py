"""Diagnose low-frequency bias in the Wishart statistics.

Computes the actual Wishart empirical PSD (the same data the likelihood sees)
and compares it to the analytical truth.  Produces a ratio plot S_wishart/S_true
that reveals any systematic bias.

Usage:
    python diagnose_wishart_bias.py [--seed 0] [--wishart-window tukey]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.example_datasets.lisatools_backend import (
    ensure_lisatools_backends,
)

ensure_lisatools_backends()

from utils.data import generate_lisa_data

from log_psplines.datatypes.multivar import MultivarFFT
from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.logger import logger, set_level

set_level("INFO")

FMIN = 1e-4
FMAX = 1e-1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration-days", type=float, default=365.0)
    p.add_argument("--block-days", type=float, default=7.0)
    p.add_argument(
        "--wishart-window",
        choices=("none", "hann", "tukey"),
        default="tukey",
    )
    p.add_argument("--wishart-tukey-alpha", type=float, default=0.1)
    p.add_argument(
        "--wishart-detrend",
        choices=("none", "constant", "linear"),
        default="constant",
    )
    p.add_argument("--wishart-floor-fraction", type=float, default=None)
    p.add_argument("--absolute-freq-units", action="store_true", default=True)
    p.add_argument(
        "--no-absolute-freq-units",
        action="store_false",
        dest="absolute_freq_units",
    )
    p.add_argument("--outdir", type=str, default="runs/diagnostics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Window spec
    if args.wishart_window == "none":
        window = None
    elif args.wishart_window == "hann":
        window = "hann"
    elif args.wishart_window == "tukey":
        window = ("tukey", args.wishart_tukey_alpha)
    else:
        window = None

    detrend = False if args.wishart_detrend == "none" else args.wishart_detrend

    # Generate data (same as main.py)
    ts, freq_true, S_true, Nb, Lb, dt = generate_lisa_data(
        seed=args.seed,
        duration_days=args.duration_days,
        block_days=args.block_days,
        fmin_generate=min(FMIN, 1e-5),
        fmax_generate=max(FMAX, 1e-1),
        absolute_freq_units=args.absolute_freq_units,
    )
    fs = 1.0 / dt
    logger.info(f"Data: Nb={Nb}, Lb={Lb}, dt={dt}, fs={fs}")

    # Compute Wishart (same path as inference)
    wishart = MultivarFFT.compute_wishart(
        x=ts.y,
        fs=fs,
        Nb=Nb,
        fmin=FMIN,
        fmax=FMAX,
        window=window,
        detrend=detrend,
        wishart_floor_fraction=args.wishart_floor_fraction,
    )

    # Extract empirical PSD from Wishart
    emp = wishart.empirical_psd
    freq_w = np.asarray(emp.freq, dtype=np.float64)
    psd_w = np.asarray(emp.psd, dtype=np.complex128)
    p = psd_w.shape[1]

    logger.info(
        f"Wishart empirical PSD: {freq_w.shape[0]} freq bins, "
        f"range [{freq_w[0]:.2e}, {freq_w[-1]:.2e}] Hz"
    )

    # Interpolate truth to Wishart freq grid
    S_interp = interp_matrix(
        np.asarray(freq_true, dtype=np.float64),
        np.asarray(S_true, dtype=np.complex128),
        freq_w,
    )

    # Compute ratio: S_wishart / S_true (diagonal elements)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    channel_labels = ["X", "Y", "Z"]

    # Left column: diagonal PSD comparison
    for ch in range(p):
        ax = axes[ch, 0]
        s_emp = psd_w[:, ch, ch].real
        s_tru = S_interp[:, ch, ch].real

        ax.loglog(freq_w, s_tru, "k-", lw=1.5, label="Analytical", zorder=5)
        ax.loglog(
            freq_w, s_emp, "-", color="C0", lw=0.3, alpha=0.5, label="Wishart"
        )

        # Smoothed Wishart (log-frequency moving average, no edge artefacts)
        n_smooth = max(1, len(freq_w) // 500)
        if n_smooth > 1:
            # Use uniform_filter1d (reflects at borders, no zero-padding bias)
            from scipy.ndimage import uniform_filter1d

            s_smooth = uniform_filter1d(s_emp, size=n_smooth, mode="mirror")
            ax.loglog(
                freq_w,
                s_smooth,
                "-",
                color="C1",
                lw=1.2,
                label=f"Wishart (smoothed, n={n_smooth})",
            )

        ax.set_ylabel(f"S_{channel_labels[ch]}{channel_labels[ch]} [1/Hz]")
        ax.set_xlim(FMIN, FMAX)
        ax.legend(fontsize=8)
        if ch == 0:
            ax.set_title("PSD comparison")

    # Right column: ratio S_wishart / S_true
    for ch in range(p):
        ax = axes[ch, 1]
        s_emp = psd_w[:, ch, ch].real
        s_tru = S_interp[:, ch, ch].real

        safe = s_tru > 0
        ratio = np.full_like(s_emp, np.nan)
        ratio[safe] = s_emp[safe] / s_tru[safe]

        ax.semilogx(
            freq_w[safe], ratio[safe], ".", ms=0.3, alpha=0.3, color="C0"
        )

        # Smoothed ratio (mirror padding — no edge artefacts)
        if n_smooth > 1:
            from scipy.ndimage import uniform_filter1d

            r_smooth = uniform_filter1d(
                np.where(np.isfinite(ratio), ratio, 1.0),
                size=n_smooth,
                mode="mirror",
            )
            ax.semilogx(
                freq_w, r_smooth, "-", color="C1", lw=1.2, label="Smoothed"
            )

        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_ylabel(
            f"S_{channel_labels[ch]}{channel_labels[ch]} (Wishart / True)"
        )
        ax.set_ylim(0.0, 3.0)
        ax.set_xlim(FMIN, FMAX)
        ax.legend(fontsize=8)
        if ch == 0:
            ax.set_title("Ratio (Wishart / Analytical)")

    for ax in axes[-1]:
        ax.set_xlabel("Frequency [Hz]")

    wlabel = args.wishart_window
    if wlabel == "tukey":
        wlabel = f"tukey({args.wishart_tukey_alpha})"
    fig.suptitle(
        f"Wishart empirical PSD vs Truth\n"
        f"seed={args.seed}, Nb={Nb}, Lb={Lb}, "
        f"window={wlabel}, detrend={args.wishart_detrend}",
        fontsize=12,
    )
    fig.tight_layout()

    outpath = os.path.join(args.outdir, "wishart_vs_truth.png")
    fig.savefig(outpath, dpi=150)
    logger.info(f"Saved: {outpath}")

    # Print summary statistics by frequency decade
    print("\n=== Ratio summary (S_wishart / S_true), diagonal mean ===")
    decades = [
        (1e-4, 1e-3),
        (1e-3, 1e-2),
        (1e-2, 3e-2),
        (3e-2, 1e-1),
    ]
    for flo, fhi in decades:
        mask = (freq_w >= flo) & (freq_w < fhi)
        if not np.any(mask):
            continue
        ratios = []
        for ch in range(p):
            s_emp = psd_w[mask, ch, ch].real
            s_tru = S_interp[mask, ch, ch].real
            safe = s_tru > 0
            if np.any(safe):
                ratios.append(np.median(s_emp[safe] / s_tru[safe]))
        if ratios:
            mean_ratio = np.mean(ratios)
            print(
                f"  [{flo:.0e}, {fhi:.0e}) Hz: "
                f"median ratio = {mean_ratio:.4f} "
                f"(bias = {(mean_ratio - 1) * 100:+.2f}%)"
            )

    # Also test without window for comparison
    if window is not None:
        print("\n=== Re-running with NO window for comparison ===")
        wishart_rect = MultivarFFT.compute_wishart(
            x=ts.y,
            fs=fs,
            Nb=Nb,
            fmin=FMIN,
            fmax=FMAX,
            window=None,
            detrend=detrend,
            wishart_floor_fraction=args.wishart_floor_fraction,
        )
        emp_rect = wishart_rect.empirical_psd
        psd_rect = np.asarray(emp_rect.psd, dtype=np.complex128)

        print("=== Ratio summary (NO window), diagonal mean ===")
        for flo, fhi in decades:
            mask = (freq_w >= flo) & (freq_w < fhi)
            if not np.any(mask):
                continue
            ratios = []
            for ch in range(p):
                s_emp = psd_rect[mask, ch, ch].real
                s_tru = S_interp[mask, ch, ch].real
                safe = s_tru > 0
                if np.any(safe):
                    ratios.append(np.median(s_emp[safe] / s_tru[safe]))
            if ratios:
                mean_ratio = np.mean(ratios)
                print(
                    f"  [{flo:.0e}, {fhi:.0e}) Hz: "
                    f"median ratio = {mean_ratio:.4f} "
                    f"(bias = {(mean_ratio - 1) * 100:+.2f}%)"
                )


if __name__ == "__main__":
    main()
