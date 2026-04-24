"""Plot RIAE and coverage vs eta for different LISA observation durations.

Two figures are produced:

Figure 1 — Coverage and accuracy vs eta (one curve per duration)
  Shows that the optimal eta shifts left (toward zero) as duration grows,
  and that RIAE is insensitive to eta while coverage is not.

Figure 2 — Recommended eta vs Nb / duration
  For each duration, finds the smallest eta with coverage_diag >= 0.88
  (within 2 pp of 90% nominal) and plots it against Nb and duration.
  Overlays the formula eta = 2/Nb to validate the scaling.

Usage
-----
python docs/studies/lisa/plot_duration_eta_study.py
python docs/studies/lisa/plot_duration_eta_study.py --runs-dir /path/to/runs
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLOCK_DAYS = 7
COVERAGE_TARGET = 0.88  # "near-nominal" threshold for eta_opt selection

DURATION_STYLES: dict[int, dict] = {
    7: {"color": "#1f77b4", "marker": "o", "label": "7 d  (Nb=1)"},
    30: {"color": "#ff7f0e", "marker": "s", "label": "30 d  (Nb=4)"},
    91: {"color": "#2ca02c", "marker": "^", "label": "91 d  (Nb=13)"},
    182: {"color": "#9467bd", "marker": "D", "label": "182 d  (Nb=26)"},
    365: {"color": "#d62728", "marker": "P", "label": "365 d  (Nb=52)"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_summaries(runs_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(
        glob.glob(
            str(runs_dir / "**/compact_run_summary.json"), recursive=True
        )
    ):
        try:
            with open(p) as f:
                d = json.load(f)
            if "eta" not in d or "duration_days" not in d:
                continue
            if "coverage_diag" not in d and "riae_diag_mean" not in d:
                continue
            rows.append(d)
        except Exception:
            continue
    return rows


def _group_by_duration(rows: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for r in rows:
        dur = int(round(float(r["duration_days"])))
        groups.setdefault(dur, []).append(r)
    for dur in groups:
        groups[dur].sort(key=lambda r: float(r["eta"]))
    return groups


def _nb(duration_days: int) -> int:
    return max(1, duration_days // BLOCK_DAYS)


def _arr(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r.get(key, np.nan)) for r in rows])


# ---------------------------------------------------------------------------
# Figure 1: coverage + RIAE + sampler diagnostics vs eta
# ---------------------------------------------------------------------------


def plot_metrics_vs_eta(groups: dict[int, list[dict]], outpath: Path) -> None:
    """Four-panel figure: key metrics vs raw eta, one curve per duration."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ax_cov, ax_riae, ax_rhat = axes

    for dur, rows in sorted(groups.items()):
        style = DURATION_STYLES.get(
            dur, {"color": "gray", "marker": "x", "label": f"{dur}d"}
        )
        etas = _arr(rows, "eta")
        cov_diag = _arr(rows, "coverage_diag")
        cov_coh = _arr(rows, "coverage_coherence")
        riae = _arr(rows, "riae_diag_mean")
        rhat = _arr(rows, "rhat_max")

        kw = dict(
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=1.5,
            markersize=6,
        )
        ax_cov.plot(etas, cov_diag, **kw)
        ax_cov.plot(
            etas,
            cov_coh,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.5,
            markersize=6,
            linestyle="--",
            alpha=0.5,
        )
        ax_riae.plot(etas, riae, **kw)
        ax_rhat.semilogy(etas, rhat, **kw)

    # Reference lines
    ax_cov.axhline(0.90, color="k", lw=1, ls=":", label="90% nominal")
    ax_cov.axhline(
        COVERAGE_TARGET,
        color="k",
        lw=0.8,
        ls="--",
        alpha=0.5,
        label=f"{int(COVERAGE_TARGET * 100)}% threshold",
    )
    ax_rhat.axhline(1.01, color="k", lw=1, ls="--", label="R-hat = 1.01")

    ax_cov.set(
        xlabel=r"$\eta$",
        ylabel="Coverage",
        title="Coverage vs η\n(solid = diagonal, dashed = coherence)",
        ylim=(0.4, 1.05),
    )
    ax_riae.set(
        xlabel=r"$\eta$",
        ylabel="RIAE (diagonal, mean)",
        title="Point accuracy vs η\n(flat → η only affects CI width)",
    )
    ax_rhat.set(
        xlabel=r"$\eta$",
        ylabel="R-hat (max)",
        title="Sampler convergence vs η",
    )
    ax_rhat.set_ylim(0.99, 1.02)

    handles, labels = ax_cov.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("LISA duration × η sweep", fontsize=13)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Figure 2: recommended eta vs Nb, with eta = 2/Nb overlay
# ---------------------------------------------------------------------------


def plot_eta_vs_nb(groups: dict[int, list[dict]], outpath: Path) -> None:
    """Scatter of eta_opt vs Nb, overlaid with the formula eta = 2/Nb."""
    # For each duration, find the smallest eta with coverage_diag >= threshold.
    records: list[
        tuple[int, int, float, float]
    ] = []  # (dur, Nb, eta_opt, cov)
    for dur, rows in sorted(groups.items()):
        nb = _nb(dur)
        for r in rows:
            eta = float(r.get("eta", np.nan))
            cov = float(r.get("coverage_diag", np.nan))
            rhat = float(r.get("rhat_max", np.nan))
            # Only consider runs where sampler converged.
            if np.isnan(cov) or np.isnan(rhat) or rhat > 1.1:
                continue
            if cov >= COVERAGE_TARGET:
                records.append((dur, nb, eta, cov))
                break  # rows sorted by eta ascending; first hit is the smallest viable eta

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    ax_nb, ax_days = axes

    # Scatter: eta_opt vs Nb
    for dur, nb, eta_opt, cov in records:
        style = DURATION_STYLES.get(dur, {"color": "gray", "marker": "x"})
        ax_nb.scatter(
            nb,
            eta_opt,
            color=style["color"],
            marker=style["marker"],
            s=80,
            zorder=3,
            label=DURATION_STYLES.get(dur, {}).get("label", f"{dur}d"),
        )
        ax_days.scatter(
            dur,
            eta_opt,
            color=style["color"],
            marker=style["marker"],
            s=80,
            zorder=3,
        )

    # Overlay formula eta = 2/Nb
    nb_range = np.linspace(0.8, max(_nb(d) for d in groups) * 1.2, 200)
    eta_formula = np.minimum(1.0, 2.0 / nb_range)
    ax_nb.plot(
        nb_range,
        eta_formula,
        "k--",
        lw=1.5,
        label=r"$\eta = 2 / N_b$  (formula)",
    )
    ax_nb.set(
        xlabel=r"$N_b$ (Bartlett segments)",
        ylabel=r"$\eta_{\rm opt}$  (smallest η with coverage ≥ 88%)",
        title=r"Recommended $\eta$ vs $N_b$",
        xscale="log",
        yscale="log",
    )

    days_range = np.linspace(5, max(groups) * 1.2, 200)
    nb_from_days = np.maximum(1, days_range / BLOCK_DAYS)
    ax_days.plot(
        days_range,
        np.minimum(1.0, 2.0 / nb_from_days),
        "k--",
        lw=1.5,
        label=r"$\eta = 2 \cdot T_{\rm block} / T_{\rm obs}$",
    )
    ax_days.set(
        xlabel="Observation duration (days)",
        ylabel=r"$\eta_{\rm opt}$",
        title=r"Recommended $\eta$ vs observation duration",
        xscale="log",
        yscale="log",
    )

    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        r"Empirical $\eta_{\rm opt}$ follows $\eta = 2/N_b$ across all durations",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Text summary table
# ---------------------------------------------------------------------------


def print_table(groups: dict[int, list[dict]]) -> None:
    header = (
        f"{'dur':>5} {'Nb':>3} {'eta':>8} "
        f"{'riae_diag':>10} {'cov_diag':>9} {'cov_coh':>8} {'rhat':>7} {'ch2_steps':>10}"
    )
    print(header)
    print("-" * len(header))
    for dur, rows in sorted(groups.items()):
        nb = _nb(dur)
        eta_formula = min(1.0, 2.0 / nb)
        for r in rows:
            eta = float(r.get("eta", np.nan))
            riae = float(r.get("riae_diag_mean", np.nan))
            cov = float(r.get("coverage_diag", np.nan))
            coh = float(r.get("coverage_coherence", np.nan))
            rhat = float(r.get("rhat_max", np.nan))
            steps = float(r.get("num_steps_channel_2_median", np.nan))
            # Mark the eta closest to the formula recommendation.
            marker = (
                " ← η=2/Nb"
                if abs(eta - eta_formula) < eta_formula * 0.3
                else ""
            )
            print(
                f"{dur:>5} {nb:>3} {eta:>8.4f} "
                f"{riae:>10.5f} {cov:>9.4f} {coh:>8.4f} {rhat:>7.4f} {steps:>10.0f}"
                f"{marker}"
            )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
    )
    p.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "paper_figs"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _load_summaries(runs_dir)
    if not rows:
        print(f"No usable compact_run_summary.json files found in {runs_dir}")
        sys.exit(1)

    groups = _group_by_duration(rows)
    target_durations = set(DURATION_STYLES.keys())
    groups = {
        d: rs
        for d, rs in groups.items()
        if d in target_durations and any("coverage_diag" in r for r in rs)
    }

    if not groups:
        print(
            f"No runs with coverage_diag found for durations {sorted(target_durations)}"
        )
        sys.exit(1)

    print(f"Durations found: {sorted(groups.keys())}")
    print(
        f"Runs per duration: {', '.join(f'{d}d={len(r)}' for d, r in sorted(groups.items()))}"
    )
    print()
    print_table(groups)

    plot_metrics_vs_eta(groups, outdir / "duration_eta_metrics.pdf")
    plot_eta_vs_nb(groups, outdir / "duration_eta_formula.pdf")
    print("\nDone.")


if __name__ == "__main__":
    main()
