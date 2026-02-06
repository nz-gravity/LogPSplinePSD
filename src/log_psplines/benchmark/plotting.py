import json
import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch

from ..logger import logger

NUTS_COLOR = "tab:orange"
GPU_MARKER = "o--"
CPU_MARKER = "s-"
CPU_ALPHA = 0.75
GPU_ALPHA = 1.0

CPU_KWGS = dict(alpha=0.75, filled=False)
GPU_KWGS = dict(alpha=1.0, filled=True)
NUTS_KWGS = dict(color=NUTS_COLOR)


def logspace_widths(xs, log_width=0.1):
    xs = np.array(xs)
    return 10 ** (np.log10(xs) + log_width / 2) - 10 ** (
        np.log10(xs) - log_width / 2
    )


def plot_box(ax, xs, ys, color="C0", alpha=0.7, filled=True):
    xscale = ax.get_xscale()

    if xscale == "log":
        widths = logspace_widths(xs, log_width=0.1)
    else:
        widths = None

    bp = ax.boxplot(
        ys,
        positions=xs,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        label=None,
    )

    for box in bp["boxes"]:
        if filled:
            box.set_facecolor(color)
            box.set_alpha(alpha)
            box.set_linewidth(0)
        else:
            box.set_facecolor("none")
            box.set_edgecolor(color)
            box.set_alpha(alpha)
            box.set_linewidth(3)

    for median in bp["medians"]:
        median.set_color(color)
        median.set_alpha(alpha)

    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color(color)
            line.set_alpha(alpha)


def plot_ess(*args, **kwargs):
    plot_box(*args, **kwargs)
    args[0].set_ylabel("ESS")


def plot_runtimes(*args, **kwargs):
    plot_box(*args, **kwargs)
    args[0].set_ylabel("Runtime (seconds)")
    args[0].set_yscale("log")


def plot_data_size_results(filepaths: List[str]) -> None:
    """Plot data size analysis results."""

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_xscale("log")

    for filepath in filepaths:
        if not os.path.exists(filepath):
            logger.info(f"Data file {filepath} not found")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        kwgs = _get_kwgs(filepath)
        plot_ess(axes[0], data["ns"], data["ess"], **kwgs)
        plot_runtimes(axes[1], data["ns"], data["runtimes"], **kwgs)

    axes[1].set_xlabel(r"$N$")

    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    axes[1].xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs="auto", numticks=10)
    )
    axes[1].xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    axes[1].xaxis.set_minor_formatter(ticker.NullFormatter())

    # remove vertical space between subplots
    _add_legend(axes[0], [os.path.basename(f) for f in filepaths])
    plt.subplots_adjust(hspace=0.0)
    fdir = os.path.dirname(filepaths[0])

    plt.savefig(f"{fdir}/N_vs_runtime.png", dpi=150)
    plt.close()


def plot_knots_results(filepaths: List[str]) -> None:
    """Plot knots analysis results."""

    fig, axes = plt.subplots(2, 1, sharex=True)

    for filepath in filepaths:
        if not os.path.exists(filepath):
            logger.info(f"Data file {filepath} not found")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        kwgs = {
            **NUTS_KWGS,
            **(CPU_KWGS if data["device"] == "cpu" else GPU_KWGS),
        }
        plot_ess(axes[0], data["ks"], data["ess"], **kwgs)
        plot_runtimes(axes[1], data["ks"], data["runtimes"], **kwgs)

    axes[1].set_xlabel(r"$K$")

    # use autoformatter for x ticks
    axes[1].xaxis.set_major_locator(ticker.AutoLocator())
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())

    _add_legend(axes[0], [os.path.basename(f) for f in filepaths])
    plt.subplots_adjust(hspace=0.0)
    fdir = os.path.dirname(filepaths[0])

    plt.savefig(f"{fdir}/K_vs_runtime.png", dpi=150)
    plt.close()


def _get_kwgs(fname: str):
    return {
        **NUTS_KWGS,
        **(CPU_KWGS if "cpu" in fname else GPU_KWGS),
    }


def _add_legend(ax, fnames: List[str]) -> None:
    """Add legend to the axes."""

    patches, labels = [], []
    for fname in fnames:
        kwgs = _get_kwgs(fname)
        if kwgs["filled"]:
            p = Patch(color=kwgs["color"], alpha=kwgs["alpha"])
        else:
            p = Patch(
                edgecolor=kwgs["color"], alpha=kwgs["alpha"], facecolor="none"
            )
        patches.append(p)

        sampler = "NUTS"
        device = "CPU" if "cpu" in fname else "GPU"
        labels.append(f"{sampler} ({device})")

    ax.legend(
        handles=patches,
        labels=labels,
        frameon=True,
        fontsize="small",
        loc="upper right",
    )
