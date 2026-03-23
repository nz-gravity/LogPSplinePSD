"""
Collect results from 3D VAR simulation studies.

Scans out_var3/ for compact_run_summary.json files, aggregates into:
  - results_per_seed.csv   : one row per seed
  - results_summary.csv    : mean ± std per condition
  - results_summary.txt    : human-readable table

Usage:
    python collect_results.py                        # all results in out_var3/
    python collect_results.py --glob "*rect*"        # filter by slug pattern
    python collect_results.py --glob "*Nh*"          # coarse-grain study only
    python collect_results.py --glob "*large*"       # large study only
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out_var3")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect(pattern: str = "*") -> pd.DataFrame:
    """Read all compact_run_summary.json files matching the glob pattern."""
    search = os.path.join(OUT_DIR, pattern, "compact_run_summary.json")
    files = sorted(glob.glob(search))

    if not files:
        print(f"No files found matching: {search}")
        return pd.DataFrame()

    rows = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            data["run_dir"] = os.path.basename(os.path.dirname(f))
            rows.append(data)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} runs from out_var3/{pattern}/")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

KEY_METRICS = ["coverage", "riae_matrix", "l2_matrix", "ess_median", "runtime"]
GROUP_COLS = ["mode", "N", "Nb", "Nh"]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per condition (mode × N × Nb × Nh)."""
    if df.empty:
        return df

    # Coerce numeric
    for col in KEY_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    groups = [c for c in GROUP_COLS if c in df.columns]
    agg = {}
    for m in KEY_METRICS:
        if m in df.columns:
            agg[f"{m}_mean"] = (m, "mean")
            agg[f"{m}_std"] = (m, "std")
            agg[f"{m}_min"] = (m, "min")
            agg[f"{m}_max"] = (m, "max")
    agg["n_seeds"] = ("seed", "count")

    summary = df.groupby(groups).agg(**agg).reset_index()
    return summary


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def print_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(
        f"  {'Mode':<12} {'N':>6} {'Nb':>4} {'Nh':>5} {'Seeds':>6}  "
        f"{'Coverage':>10}  {'RIAE':>10}  {'L2':>10}  {'ESS':>8}"
    )
    print("  " + "-" * 80)

    for _, row in summary.iterrows():
        nh = str(row.get("Nh", "?"))
        cov = row.get("coverage_mean", float("nan"))
        cov_s = row.get("coverage_std", float("nan"))
        riae = row.get("riae_matrix_mean", float("nan"))
        riae_s = row.get("riae_matrix_std", float("nan"))
        l2 = row.get("l2_matrix_mean", float("nan"))
        l2_s = row.get("l2_matrix_std", float("nan"))
        ess = row.get("ess_median_mean", float("nan"))
        n_s = int(row.get("n_seeds", 0))

        print(
            f"  {str(row.get('mode','?')):<12} "
            f"{int(row.get('N', 0)):>6} "
            f"{int(row.get('Nb', 0)):>4} "
            f"{nh:>5} "
            f"{n_s:>6}  "
            f"{cov:>8.4f}±{cov_s:.3f}  "
            f"{riae:>8.4f}±{riae_s:.3f}  "
            f"{l2:>8.4f}±{l2_s:.3f}  "
            f"{ess:>8.0f}"
        )

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob pattern for subdirs in out_var3/ (default: '*'). "
        "Examples: '*rect*', '*Nh*', '*large*'",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV prefix (default: 'results_<glob>')",
    )
    args = parser.parse_args()

    label = args.glob.strip("*").strip("_") or "all"
    out_prefix = args.out or f"results_{label}"

    df = collect(args.glob)
    if df.empty:
        return

    # Per-seed CSV
    per_seed_csv = os.path.join(HERE, f"{out_prefix}_per_seed.csv")
    df.to_csv(per_seed_csv, index=False)
    print(f"Saved per-seed results → {per_seed_csv}")

    # Summary CSV + table
    summary = summarise(df)
    summary_csv = os.path.join(HERE, f"{out_prefix}_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary          → {summary_csv}")

    print_table(summary)

    # Also print raw per-seed coverage for quick sanity check
    if "coverage" in df.columns and len(df) <= 120:
        print("  Per-seed coverage:")
        for _, row in df.sort_values(
            GROUP_COLS + ["seed"], key=lambda x: x.astype(str)
        ).iterrows():
            print(
                f"    seed={int(row['seed']):3d}  "
                f"mode={str(row.get('mode','?')):<12}  "
                f"Nh={str(row.get('Nh','?')):>5}  "
                f"coverage={float(row['coverage']):.4f}"
            )


if __name__ == "__main__":
    main()
