#!/usr/bin/env python3
"""
Generate a LogPsplinePSD logo with
- black wordmark
- orange 'P'
- faint noisy PSD / periodogram background
- smooth fitted PSD curve
- transparent PNG and SVG export

Usage
-----
python make_logo.py

Outputs
-------
logo_pspline_psd.png
logo_pspline_psd.svg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


# ---------------------------
# Style / palette
# ---------------------------
BLACK = "#000000"
ORANGE = "#ff8c1a"
NOISE = "#d9cfc2"      # faint noisy periodogram
FIT = "#e8c59e"        # smooth fitted PSD
MARKER = "#c99595"     # optional soft pink markers
BG = (1, 1, 1, 0)      # fully transparent

FONT_FAMILY = "DejaVu Sans"
FONT_WEIGHT = "bold"


# ---------------------------
# Synthetic PSD-like shapes
# ---------------------------
def gaussian(x, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_psd_shape(x):
    """
    Create a plausible PSD-like curve:
    broad low-frequency structure + peaks + high-frequency decay.
    """
    baseline = 0.12 + 0.55 / (x + 0.18) ** 0.45
    baseline = baseline / baseline.max()

    fit = (
        0.55 * baseline
        + gaussian(x, 0.34, 0.085, 0.80)
        + gaussian(x, 0.57, 0.10, 0.35)
        + gaussian(x, 0.78, 0.12, 0.15)
    )

    fit = fit / fit.max()
    return fit


def make_noisy_periodogram(x, fit, rng, n_lines=140):
    """
    Sample a sparse noisy periodogram-like set of line segments.
    """
    xs = np.linspace(0.02, 0.98, n_lines)

    # evaluate fit at sparse locations
    fit_s = np.interp(xs, x, fit)

    # multiplicative noise with a bit of heavy-tailed behaviour
    scale = rng.lognormal(mean=0.0, sigma=0.45, size=n_lines)
    chisq_like = rng.chisquare(df=2, size=n_lines) / 2.0
    noisy = fit_s * (0.55 * scale + 0.45 * chisq_like)

    # keep within a visually reasonable range
    noisy = np.clip(noisy, 0.08, 1.55)

    return xs, fit_s, noisy


# ---------------------------
# Text helpers
# ---------------------------
def text_width_axes(fig, ax, text, fontsize, family=FONT_FAMILY, weight=FONT_WEIGHT):
    """
    Estimate text width in axes coordinates using TextPath.
    """
    fp = FontProperties(family=family, weight=weight, size=fontsize)
    path = TextPath((0, 0), text, prop=fp)
    width_points = path.get_extents().width
    width_inches = width_points / 72.0
    axes_width_inches = fig.get_size_inches()[0] * ax.get_position().width
    return width_inches / axes_width_inches


def add_wordmark(ax, fig):
    """
    Draw:
        LogPspline
        PSD
    with orange P and black remaining letters.
    """
    # Main wordmark position in axes coordinates
    x0 = 0.03
    y0 = 0.60
    fs_main = 56
    fs_sub = 42

    # Split to color only the P
    parts = [("Log", BLACK), ("P", ORANGE), ("spline", BLACK)]
    x = x0

    for txt, color in parts:
        ax.text(
            x, y0, txt,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=fs_main,
            fontweight=FONT_WEIGHT,
            fontfamily=FONT_FAMILY,
            color=color,
            zorder=10,
        )
        x += text_width_axes(fig, ax, txt, fs_main)

    # Sub-label
    ax.text(
        x0, 0.38, "PSD",
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=fs_sub,
        fontweight=FONT_WEIGHT,
        fontfamily=FONT_FAMILY,
        color=BLACK,
        zorder=10,
    )


# ---------------------------
# Main logo drawing
# ---------------------------
def make_logo(outbase="logo_pspline_psd", seed=4):
    rng = np.random.default_rng(seed)

    # Canvas similar to your current aspect ratio
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background PSD curve region
    x = np.linspace(0.02, 0.98, 700)
    fit = make_psd_shape(x)

    # Position / scale the background shape so it sits behind the wordmark
    # and slopes down to the right, similar to your current logo.
    x_plot = 0.10 + 0.90 * x
    y_fit = 0.22 + 0.48 * fit

    # Noisy periodogram lines
    xs, fit_s, noisy_s = make_noisy_periodogram(x, fit, rng, n_lines=155)
    xs_plot = 0.10 + 0.90 * xs
    y_fit_s = 0.22 + 0.48 * fit_s
    y_noisy_s = 0.22 + 0.48 * noisy_s / noisy_s.max() * 1.05

    # Center noise roughly around the fit for a more "periodogram around fit" feel
    y_top = np.maximum(y_fit_s, y_noisy_s)
    y_bot = np.minimum(y_fit_s, y_fit_s - 0.65 * (y_noisy_s - y_fit_s))

    # Clip within canvas
    y_top = np.clip(y_top, 0.08, 0.92)
    y_bot = np.clip(y_bot, 0.08, 0.92)

    # Draw faint noisy background
    ax.vlines(
        xs_plot, y_bot, y_top,
        color=NOISE,
        linewidth=2.3,
        alpha=0.70,
        zorder=1,
        capstyle="round",
    )

    # Smooth fitted curve
    ax.plot(
        x_plot, y_fit,
        color=FIT,
        linewidth=6.0,
        alpha=0.95,
        solid_capstyle="round",
        zorder=2,
    )

    # Optional marker points on the fit
    marker_x = np.array([0.17, 0.30, 0.58, 0.77, 0.90])
    marker_y = np.interp(marker_x, x_plot, y_fit)
    ax.scatter(
        marker_x, marker_y,
        s=110,
        color=MARKER,
        edgecolor=NOISE,
        linewidth=1.6,
        zorder=3,
        alpha=0.95,
    )

    # Add wordmark on top
    add_wordmark(ax, fig)

    # Save
    out_png = Path(f"{outbase}.png")
    out_svg = Path(f"{outbase}.svg")

    fig.savefig(
        out_png,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        out_svg,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    make_logo()