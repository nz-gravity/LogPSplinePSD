"""
Create a small GIF/MP4 animation of univariate posterior PSD draws.

This is intentionally lightweight and only depends on the objects already
stored in this project’s ArviZ ``InferenceData``:

- ``posterior['weights']`` (draw-level samples)
- ``spline_model`` (packed basis + penalty matrix)
- ``observed_data['periodogram']`` (already rescaled)

Example:

  .venv/bin/python docs/animation_psd_draws.py run.nc psd_draws.gif --thin 10 --max-frames 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))

from log_psplines.arviz_utils.from_arviz import (
    get_periodogram,
    get_spline_model,
    get_weights,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate univariate posterior PSD draws from an ArviZ netCDF."
    )
    parser.add_argument(
        "idata_path", help="Path to ArviZ netCDF file (e.g. run.nc)."
    )
    parser.add_argument("out_path", help="Output path (.gif or .mp4).")
    parser.add_argument(
        "--thin", type=int, default=10, help="Thin posterior draws."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=250,
        help="Maximum number of frames (draws) to render.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=60,
        help="Frame interval in milliseconds.",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    idata = az.from_netcdf(args.idata_path)
    pdgrm = get_periodogram(idata)
    spline_model = get_spline_model(idata)
    weights = get_weights(idata, thin=max(1, args.thin))

    scaling_factor = float(idata.attrs.get("scaling_factor", 1.0) or 1.0)

    if weights.ndim != 2:
        raise ValueError(
            f"Expected weights with shape (n_draws, n_weights), got {weights.shape}."
        )

    if weights.shape[0] == 0:
        raise ValueError("No posterior draws found after thinning.")

    max_frames = min(int(args.max_frames), weights.shape[0])
    idx = np.linspace(0, weights.shape[0] - 1, max_frames, dtype=int)
    weights = weights[idx]

    freqs = np.asarray(pdgrm.freqs, dtype=np.float64)
    psd_draws = np.exp(
        np.asarray([spline_model(w) for w in weights], dtype=np.float64)
    )
    psd_draws = psd_draws * scaling_factor

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(freqs, pdgrm.power, color="0.7", lw=1.0, label="Periodogram")

    try:
        psd_q = idata.posterior_psd["psd"]
        percentiles = np.asarray(
            psd_q.coords["percentile"].values, dtype=float
        )
        psd_q_vals = np.asarray(psd_q.values, dtype=np.float64)

        def _grab(p: float) -> np.ndarray:
            j = int(np.argmin(np.abs(percentiles - p)))
            return psd_q_vals[j]

        q05 = _grab(5.0)
        q50 = _grab(50.0)
        q95 = _grab(95.0)
        ax.fill_between(freqs, q05, q95, color="C0", alpha=0.18, lw=0)
        ax.loglog(
            freqs, q50, color="C0", lw=1.2, alpha=0.9, label="Median (5–95%)"
        )
    except Exception:
        pass

    (draw_line,) = ax.loglog(
        freqs, psd_draws[0], color="C1", lw=1.6, alpha=0.9, label="PSD draw"
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.legend(frameon=False, loc="best")
    ax.set_title(f"Posterior draw 1/{psd_draws.shape[0]}")
    fig.tight_layout()

    def _update(i: int):
        draw_line.set_ydata(psd_draws[i])
        ax.set_title(f"Posterior draw {i+1}/{psd_draws.shape[0]}")
        return (draw_line,)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=psd_draws.shape[0],
        interval=int(args.interval_ms),
        blit=True,
    )

    out_path = args.out_path
    if out_path.endswith(".gif"):
        ani.save(out_path, writer="pillow", dpi=int(args.dpi))
    elif out_path.endswith(".mp4"):
        ani.save(out_path, writer="ffmpeg", dpi=int(args.dpi))
    else:
        raise ValueError("out_path must end with .gif or .mp4")


if __name__ == "__main__":
    main()
