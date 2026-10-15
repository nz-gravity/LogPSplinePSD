"""Compact manuscript figures from saved Hann-study draws; no inference."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.dw_wdm_small import hann_paper as study

from log_psplines import PSDResult

MP_COLOR = "#1f6f9c"
WDM_COLOR = "#d98014"
TRUTH_COLOR = "#1a1a1a"


def paper_style() -> None:
    """Match the companion paper's overleaf/scripts/lisa_style.py."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.3,
            "legend.borderaxespad": 0.4,
            "savefig.dpi": 400,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    directory = study.ROOT / "figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            directory / f"{name}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


def smooth_surfaces() -> dict:
    """Reload only the four preselected seed-6101 illustrative fits."""
    u, f, _ = study.extension.grid()
    model = study.b.master_model()
    truth = np.broadcast_to(2 * study.b.variance(u)[:, None], (len(u), len(f)))
    surfaces = {}
    for cell in study.read(study.ROOT / "inventory.json"):
        if cell["dataset"] != "smooth" or cell["seed"] != 6101:
            continue
        posterior = PSDResult.from_netcdf(
            Path(cell["path"]) / "posterior.nc"
        ).posterior
        surfaces[cell["id"]] = (
            study.quantiles(posterior, model, u, f),
            truth,
            u,
            f,
        )
    assert len(surfaces) == 4
    return surfaces


def plot_smooth(surfaces: dict | None = None) -> None:
    paper_style()
    surfaces = smooth_surfaces() if surfaces is None else surfaces
    fig, axes = plt.subplots(
        1, 2, figsize=(7, 1.9), sharex=True, sharey=True, layout="constrained"
    )
    for ax, method, color, panel in zip(
        axes,
        ("MP", "WDM"),
        (MP_COLOR, WDM_COLOR),
        ("(a) Hann MP", "(b) WDM"),
        strict=True,
    ):
        ax.axvspan(3686 / 8192, 4505 / 8192, color="0.90", zorder=0)
        for mask, ls, label in (
            ("complete", "--", "Complete"),
            ("single", "-", "Gapped"),
        ):
            q, truth, u, f = surfaces[f"smooth_s6101_{mask}_{method}"]
            # Evaluate the stated frequency exactly rather than snapping to a nearby grid point.
            j = np.flatnonzero(np.isclose(f, 0.25))
            assert len(j) == 1, (
                "The fixed illustration grid must include 0.25 Hz"
            )
            low, med, high = q[:, :, j[0]]
            if mask == "complete":
                ax.plot(u, low, color=color, ls="--", lw=0.55, alpha=0.5)
                ax.plot(u, high, color=color, ls="--", lw=0.55, alpha=0.5)
            else:
                ax.fill_between(u, low, high, color=color, alpha=0.18, lw=0)
            ax.plot(u, med, color=color, ls=ls, label=label)
        ax.plot(u, truth[:, 0], ":", color=TRUTH_COLOR, label="Injected PSD")
        ax.text(0.025, 0.96, panel, transform=ax.transAxes, va="top")
        ax.legend(loc="upper right")
        ax.set(
            xlabel="Rescaled physical time", xlim=(0.05, 0.95), ylim=(0.3, 4.9)
        )
    axes[0].set_ylabel(r"One-sided PSD (unit$^2$/Hz)")
    save(fig, "smooth_seed6101")


def plot_lisa(saved: dict | None = None) -> None:
    paper_style()
    saved = (
        np.load(study.ROOT / "lisa_summary/summaries.npz")
        if saved is None
        else saved
    )
    f = saved["frequency_hz"] * 1000
    truth = saved["truth_time_average"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7, 3.2),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [1.5, 1]},
        layout="constrained",
    )
    for col, condition, panel in (
        (0, "complete", "(a) Complete"),
        (1, "single", "(b) Gapped"),
    ):
        top, bottom = axes[:, col]
        for method, color, label in (
            ("MP", MP_COLOR, "Hann MP"),
            ("WDM", WDM_COLOR, "WDM"),
        ):
            q = saved[f"{condition}_{method}_time_average_quantiles"]
            assert (
                q.shape == (3, len(f)) and np.all(q > 0) and np.all(truth > 0)
            )
            top.loglog(f, q[1], color=color, label=label)
            top.fill_between(f, q[0], q[2], color=color, alpha=0.18, lw=0)
            # Quantiles commute with this monotone transform of each draw's linear average.
            error = np.log(q / truth[None, :])
            bottom.semilogx(f, error[1], color=color)
            bottom.fill_between(
                f, error[0], error[2], color=color, alpha=0.18, lw=0
            )
        top.loglog(f, truth, ":", color=TRUTH_COLOR, label="Injected PSD")
        top.text(0.025, 0.96, panel, transform=top.transAxes, va="top")
        top.legend(loc="lower right")
        bottom.axhline(0, color="0.35", ls=":", lw=0.7, zorder=0)
        bottom.set(xlabel="Frequency (mHz)", xlim=(f[0], f[-1]))
    axes[0, 0].set_ylabel(r"$\overline{S}_A$ (Hz$^2$/Hz)")
    axes[1, 0].set_ylabel(r"$\ln(\overline{S}_A/\overline{S}^{\rm true}_A)$")
    save(fig, "lisa_time_average")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-to-paper", action="store_true")
    args = parser.parse_args()
    plot_smooth()
    plot_lisa()
    if args.copy_to_paper:
        for name in ("smooth_seed6101", "lisa_time_average"):
            for suffix in ("pdf", "png"):
                shutil.copy2(
                    study.ROOT / f"figures/{name}.{suffix}",
                    study.PAPER / f"figures/hann_{name}.{suffix}",
                )


if __name__ == "__main__":
    main()
