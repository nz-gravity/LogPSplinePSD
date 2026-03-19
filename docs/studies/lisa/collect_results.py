"""
Collect results from LISA simulation study.

Scans {outdir}/seed_*/compact_run_summary.json, aggregates into:
  - results_lisa_per_seed.csv   : one row per seed
  - results_lisa_summary.csv    : mean +/- std
  - printed summary table

Usage:
    python collect_results.py                          # defaults to out_lisa_sim/
    python collect_results.py --outdir out_lisa_sim
    python collect_results.py --outdir out_lisa_test --min-seeds 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

KEY_METRICS = [
    "coverage",
    "riae_matrix",
    "riae_diag_mean",
    "riae_offdiag",
    "coherence_riae",
    "ess_median",
    "rhat_max",
    "n_divergences",
    "runtime",
    "ciw_psd_diag_mean",
    "ciw_psd_offdiag_mean",
]

# Config fields that identify a condition (for grouping in summaries).
CONFIG_COLS = ["run_slug", "K", "knot_method", "diff_order", "duration_days"]


def collect(outdir: str) -> pd.DataFrame:
    """Read all compact_run_summary.json files under outdir/**/seed_*/."""
    # Supports both flat (outdir/seed_N/) and nested (outdir/slug/seed_N/) layouts.
    search_nested = os.path.join(
        outdir, "*", "seed_*", "compact_run_summary.json"
    )
    search_flat = os.path.join(outdir, "seed_*", "compact_run_summary.json")
    files = sorted(set(glob.glob(search_nested) + glob.glob(search_flat)))
    if not files:
        print(f"No files found under: {outdir}")
        return pd.DataFrame()

    rows = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            data["seed_dir"] = os.path.basename(os.path.dirname(f))
            rows.append(data)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} seeds from {outdir}/")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std for key metrics."""
    if df.empty:
        return df

    for col in KEY_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = {}
    for m in KEY_METRICS:
        if m in df.columns:
            agg[f"{m}_mean"] = (m, "mean")
            agg[f"{m}_std"] = (m, "std")
            agg[f"{m}_min"] = (m, "min")
            agg[f"{m}_max"] = (m, "max")
    agg["n_seeds"] = ("seed", "count")

    summary = df.agg(
        {
            m: ["mean", "std", "min", "max"]
            for m in KEY_METRICS
            if m in df.columns
        }
    )
    return summary


def print_table(df: pd.DataFrame) -> None:
    """Print a human-readable summary."""
    if df.empty:
        return

    n = len(df)
    print("\n" + "=" * 72)
    print(f"  LISA SIMULATION STUDY RESULTS  ({n} seeds)")
    print("=" * 72)

    for m in KEY_METRICS:
        if m not in df.columns:
            continue
        vals = pd.to_numeric(df[m], errors="coerce").dropna()
        if vals.empty:
            continue
        mean = vals.mean()
        std = vals.std()
        lo = vals.min()
        hi = vals.max()
        print(f"  {m:<28s}  {mean:>10.4f} +/- {std:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(HERE, "out_lisa_sim"),
        help="Directory containing seed_* subdirectories (default: out_lisa_sim/).",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=0,
        help="Warn if fewer than this many seeds completed.",
    )
    parser.add_argument(
        "--out-prefix",
        default="results_lisa",
        help="Output CSV prefix (default: results_lisa).",
    )
    args = parser.parse_args()

    df = collect(args.outdir)
    if df.empty:
        return

    if args.min_seeds > 0 and len(df) < args.min_seeds:
        print(
            f"  WARNING: Only {len(df)} seeds completed "
            f"(expected >= {args.min_seeds})."
        )

    # Per-seed CSV
    per_seed_csv = os.path.join(HERE, f"{args.out_prefix}_per_seed.csv")
    df.to_csv(per_seed_csv, index=False)
    print(f"Saved per-seed results -> {per_seed_csv}")

    # Summary CSV
    summary_csv = os.path.join(HERE, f"{args.out_prefix}_summary.csv")
    summary_data = {}
    for m in KEY_METRICS:
        if m in df.columns:
            vals = pd.to_numeric(df[m], errors="coerce")
            summary_data[f"{m}_mean"] = [vals.mean()]
            summary_data[f"{m}_std"] = [vals.std()]
    summary_data["n_seeds"] = [len(df)]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary          -> {summary_csv}")

    print_table(df)

    # Per-seed coverage for quick sanity check
    if "coverage" in df.columns and len(df) <= 120:
        sort_cols = [c for c in ["run_slug", "seed"] if c in df.columns]
        print("  Per-seed coverage:")
        for _, row in df.sort_values(sort_cols).iterrows():
            slug = str(row.get("run_slug", ""))
            slug_str = f"  [{slug}]" if slug else ""
            print(
                f"    seed={int(row['seed']):3d}{slug_str}  "
                f"coverage={float(row['coverage']):.4f}  "
                f"riae={float(row.get('riae_matrix', float('nan'))):.4f}"
            )


if __name__ == "__main__":
    main()
