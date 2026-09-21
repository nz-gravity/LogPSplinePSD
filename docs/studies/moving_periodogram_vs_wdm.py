"""Make the moving-periodogram/WDM geometry comparison figure.

The transforms themselves are package code.  This script only plots their
returned coordinates and powers, so it is intended as a small documentation
companion rather than a second implementation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.data import TimeSeries
from log_psplines.preprocessing.moving_periodogram import (
    tang_moving_periodogram,
)
from log_psplines.preprocessing.wdm import wdm_periodogram

DOCS = Path(__file__).resolve().parents[1]
STATIC = DOCS / "_static"
OUT_GEOMETRY = STATIC / "moving-periodogram-vs-wdm.png"
OUT_WINDOW = STATIC / "moving-periodogram-vs-wdm-window-size.png"
OUT_THIN = STATIC / "moving-periodogram-thinning.png"


def make_series(n: int = 256, dt: float = 0.1, seed: int = 7):
    x = np.random.default_rng(seed).standard_normal(n)
    return x, TimeSeries(x, np.arange(n) * dt), dt


def add_window_block_guides(ax, u: np.ndarray, freq: np.ndarray, m: int) -> None:
    """Shade each retained window-block and connect its points with a line.

    ``u`` and ``freq`` must be in the block-major, frequency-minor order
    returned by :func:`tang_moving_periodogram`, i.e. reshape to
    ``(n_blocks, m)`` recovers one block per row.
    """
    n_blocks = u.size // m
    u2 = u.reshape(n_blocks, m)
    freq2 = freq.reshape(n_blocks, m)
    step = np.median(np.diff(u2[:, 0])) if n_blocks > 1 else u2[0, -1] - u2[0, 0]
    for b in range(n_blocks):
        pad = 0.2 * (step if step > 0 else 1.0 / u.size)
        ax.axvspan(
            u2[b].min() - pad,
            u2[b].max() + pad,
            color=("0.88" if b % 2 == 0 else "0.95"),
            zorder=0,
            lw=0,
        )
        ax.plot(u2[b], freq2[b], "-", color="0.4", lw=0.8, alpha=0.6, zorder=1)


def plot_geometry() -> None:
    """Same series, WDM grid vs. moving-periodogram zig-zag scatter."""
    x, series, dt = make_series()
    m, thin = 8, 2

    wdm = wdm_periodogram(series, nt=32, trim_low=1, trim_high=1)
    raw = tang_moving_periodogram(x, m=m, thin=thin)
    raw_time = raw["u"]
    raw_frequency = raw["omega"] / (2.0 * np.pi * dt)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.16, top=0.72, wspace=0.22
    )

    mesh = axes[0].pcolormesh(
        wdm.time,
        wdm.frequency,
        np.log10(np.maximum(wdm.power.T, 1e-12)),
        shading="nearest",
        cmap="magma",
    )
    axes[0].set_title("WDM\nwhole time × frequency grid")
    axes[0].set_ylabel("frequency [Hz]")
    fig.colorbar(mesh, ax=axes[0], pad=0.02, label="log power")

    points = axes[1].scatter(
        raw_time,
        raw_frequency,
        c=np.log10(np.maximum(raw["mi"], 1e-12)),
        s=13,
        cmap="magma",
        edgecolors="none",
        zorder=2,
    )
    add_window_block_guides(axes[1], raw_time, raw_frequency, m)
    axes[1].set_title(
        f"moving periodogram\none frequency per window (thin={thin})"
    )
    fig.colorbar(points, ax=axes[1], pad=0.02, label="log power")

    for ax in axes:
        ax.set_xlabel("time / record length")
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Same series, different time\u2013frequency sampling geometry",
        fontsize=10,
        y=0.99,
    )
    STATIC.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GEOMETRY, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_GEOMETRY}")


def plot_window_size_tradeoff() -> None:
    """Resolution trade-off: moving-periodogram ``m`` vs. WDM ``nt``.

    A bigger half-window ``m`` (equivalently, fewer WDM time blocks ``nt``)
    averages over more samples per window, so each estimate is less noisy but
    can only resolve coarser time structure -- the usual time/frequency
    resolution trade-off.
    """
    x, series, dt = make_series(n=512, seed=3)

    m_values = (4, 8, 16)
    nt_values = (64, 32, 16)  # smaller nt <-> larger time window, like larger m

    fig, axes = plt.subplots(
        2, 3, figsize=(9.5, 6.2), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.08, top=0.8, hspace=0.4, wspace=0.15
    )

    for ax, m in zip(axes[0], m_values):
        raw = tang_moving_periodogram(x, m=m, thin=2)
        freq = raw["omega"] / (2.0 * np.pi * dt)
        points = ax.scatter(
            raw["u"],
            freq,
            c=np.log10(np.maximum(raw["mi"], 1e-12)),
            s=10,
            cmap="magma",
            edgecolors="none",
            zorder=2,
        )
        add_window_block_guides(ax, raw["u"], freq, m)
        ax.set_title(
            f"m={m} (window = {2 * m + 1} samples)", fontsize=10
        )
    axes[0, 0].set_ylabel("frequency [Hz]")
    axes[0, 1].annotate(
        "moving periodogram",
        xy=(0.5, 1.32),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    for ax, nt in zip(axes[1], nt_values):
        wdm = wdm_periodogram(series, nt=nt, trim_low=1, trim_high=1)
        mesh = ax.pcolormesh(
            wdm.time,
            wdm.frequency,
            np.log10(np.maximum(wdm.power.T, 1e-12)),
            shading="nearest",
            cmap="magma",
        )
        ax.set_title(
            f"nt={nt} ({512 // nt} time samples/block)", fontsize=10
        )
        ax.set_xlabel("time / record length")
    axes[1, 0].set_ylabel("frequency [Hz]")
    axes[1, 1].annotate(
        "WDM",
        xy=(0.5, 1.32),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)

    fig.colorbar(
        points, ax=axes[0], pad=0.02, label="log power", shrink=0.85
    )
    fig.colorbar(mesh, ax=axes[1], pad=0.02, label="log power", shrink=0.85)

    fig.suptitle(
        "Window size controls the time/frequency resolution trade-off:\n"
        "fewer, wider windows/blocks -> coarser time but finer frequency",
        fontsize=10,
        y=0.97,
    )
    STATIC.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_WINDOW, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_WINDOW}")


def plot_thinning() -> None:
    """Same ``m``, increasing ``thin``: blocks stay the same size but spread out.

    Thinning does not change the window width or the frequency rungs; it
    only discards whole blocks, widening the gaps in the figure below.
    """
    x, _, dt = make_series(n=384, seed=11)
    m = 8
    thin_values = (1, 2, 4)

    fig, axes = plt.subplots(
        1, 3, figsize=(9.5, 3.2), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.2, top=0.72, wspace=0.12
    )

    for ax, thin in zip(axes, thin_values):
        raw = tang_moving_periodogram(x, m=m, thin=thin)
        freq = raw["omega"] / (2.0 * np.pi * dt)
        points = ax.scatter(
            raw["u"],
            freq,
            c=np.log10(np.maximum(raw["mi"], 1e-12)),
            s=13,
            cmap="magma",
            edgecolors="none",
            zorder=2,
        )
        add_window_block_guides(ax, raw["u"], freq, m)
        n_blocks = raw["u"].size // m
        ax.set_title(f"thin={thin} ({n_blocks} retained blocks)", fontsize=10)
        ax.set_xlabel("time / record length")

    axes[0].set_ylabel("frequency [Hz]")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(points, ax=axes, pad=0.02, label="log power", shrink=0.85)

    fig.suptitle(
        f"Thinning drops whole window-blocks (m={m} fixed): "
        "wider gaps, same window width and frequency rungs",
        fontsize=10,
        y=0.95,
    )
    STATIC.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_THIN, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_THIN}")


def main() -> None:
    plot_geometry()
    plot_window_size_tradeoff()
    plot_thinning()


if __name__ == "__main__":
    main()
