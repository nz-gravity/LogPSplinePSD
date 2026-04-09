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
DEFAULT_OUT_DIR = os.path.join(HERE, "out_var3")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect(
    pattern: str = "*",
    results_dir: str = DEFAULT_OUT_DIR,
) -> pd.DataFrame:
    """Read all compact_run_summary.json files matching the glob pattern."""
    search = os.path.join(results_dir, pattern, "compact_run_summary.json")
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
    if not df.empty:
        geometry = df.apply(_derive_geometry, axis=1, result_type="expand")
        geometry.columns = ["Lb", "N_ell", "Nc"]
        df = pd.concat([df, geometry], axis=1)
    print(f"Collected {len(df)} runs from {results_dir}/{pattern}/")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

KEY_METRICS = [
    "coverage",
    "riae_matrix",
    "l2_matrix",
    # Use the diagonal PSD CI width as the single NUTS CIW summary so it
    # remains directly comparable to the scalar VI CI width.
    "ciw_psd_diag_mean",
    "vi_coverage",
    "vi_riae_matrix",
    "vi_l2_matrix",
    "vi_ci_width",
    "ess_median",
    "runtime",
]
GROUP_COLS = ["mode", "N", "Nb", "Lb", "Nh", "N_ell", "Nc"]


def _derive_geometry(row: pd.Series) -> tuple[float, float, float]:
    """Derive (Lb, N_ell, Nc) from stored run metadata."""
    try:
        n_time = int(row["N"])
        nb = int(row["Nb"])
        if n_time <= 0 or nb <= 0 or (n_time % nb) != 0:
            return np.nan, np.nan, np.nan
        lb = n_time // nb
        n_ell = lb // 2
        nh_val = row.get("Nh", "OFF")
        if isinstance(nh_val, str) and nh_val.upper() == "OFF":
            return float(lb), float(n_ell), float(n_ell)
        nh = int(nh_val)
        if nh <= 0 or (n_ell % nh) != 0:
            return float(lb), float(n_ell), np.nan
        return float(lb), float(n_ell), float(n_ell // nh)
    except Exception:
        return np.nan, np.nan, np.nan


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per condition (mode × N × Nb × Nh)."""
    if df.empty:
        return df

    if not {"Lb", "N_ell", "Nc"}.issubset(df.columns):
        df = df.copy()
        geometry = df.apply(_derive_geometry, axis=1, result_type="expand")
        geometry.columns = ["Lb", "N_ell", "Nc"]
        df = pd.concat([df, geometry], axis=1)

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
    has_vi = any(
        col in summary.columns
        for col in (
            "vi_coverage_mean",
            "vi_riae_matrix_mean",
            "vi_l2_matrix_mean",
            "vi_ci_width_mean",
        )
    )

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    header = (
        f"  {'Mode':<12} {'N':>6} {'Nb':>4} {'Lb':>6} {'Nh':>5} {'N_ell':>6} {'Nc':>6} {'Seeds':>6}  "
        f"{'Coverage':>10}  {'RIAE':>10}  {'L2':>10}  {'CIW':>10}"
    )
    if has_vi:
        header += (
            f"  {'VI Cov':>10}  {'VI RIAE':>10}  {'VI L2':>10}  {'VI CIW':>10}"
        )
    header += f"  {'ESS':>8}"
    print(header)
    print("  " + "-" * (142 if has_vi else 94))

    for _, row in summary.iterrows():
        lb_val = row.get("Lb", np.nan)
        nh = str(row.get("Nh", "?"))
        n_ell_val = row.get("N_ell", np.nan)
        nc_val = row.get("Nc", np.nan)
        lb = (
            str(int(lb_val))
            if pd.notna(lb_val) and np.isfinite(float(lb_val))
            else "?"
        )
        n_ell = (
            str(int(n_ell_val))
            if pd.notna(n_ell_val) and np.isfinite(float(n_ell_val))
            else "?"
        )
        nc = (
            str(int(nc_val))
            if pd.notna(nc_val) and np.isfinite(float(nc_val))
            else "?"
        )
        cov = row.get("coverage_mean", float("nan"))
        cov_s = row.get("coverage_std", float("nan"))
        riae = row.get("riae_matrix_mean", float("nan"))
        riae_s = row.get("riae_matrix_std", float("nan"))
        l2 = row.get("l2_matrix_mean", float("nan"))
        l2_s = row.get("l2_matrix_std", float("nan"))
        ciw = row.get("ciw_psd_diag_mean_mean", float("nan"))
        ciw_s = row.get("ciw_psd_diag_mean_std", float("nan"))
        vi_cov = row.get("vi_coverage_mean", float("nan"))
        vi_cov_s = row.get("vi_coverage_std", float("nan"))
        vi_riae = row.get("vi_riae_matrix_mean", float("nan"))
        vi_riae_s = row.get("vi_riae_matrix_std", float("nan"))
        vi_l2 = row.get("vi_l2_matrix_mean", float("nan"))
        vi_l2_s = row.get("vi_l2_matrix_std", float("nan"))
        vi_ciw = row.get("vi_ci_width_mean", float("nan"))
        vi_ciw_s = row.get("vi_ci_width_std", float("nan"))
        ess = row.get("ess_median_mean", float("nan"))
        n_s = int(row.get("n_seeds", 0))

        line = (
            f"  {str(row.get('mode','?')):<12} "
            f"{int(row.get('N', 0)):>6} "
            f"{int(row.get('Nb', 0)):>4} "
            f"{lb:>6} "
            f"{nh:>5} "
            f"{n_ell:>6} "
            f"{nc:>6} "
            f"{n_s:>6}  "
            f"{cov:>8.4f}±{cov_s:.3f}  "
            f"{riae:>8.4f}±{riae_s:.3f}  "
            f"{l2:>8.4f}±{l2_s:.3f}  "
            f"{ciw:>8.4f}±{ciw_s:.3f}"
        )
        if has_vi:
            line += (
                f"  {vi_cov:>8.4f}±{vi_cov_s:.3f}  "
                f"{vi_riae:>8.4f}±{vi_riae_s:.3f}  "
                f"{vi_l2:>8.4f}±{vi_l2_s:.3f}  "
                f"{vi_ciw:>8.4f}±{vi_ciw_s:.3f}"
            )
        line += f"  {ess:>8.0f}"
        print(line)

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
        help="Glob pattern for subdirs in the results root (default: '*'). "
        "Examples: '*rect*', '*Nh*', '*large*'",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_OUT_DIR,
        help=(
            "Root directory containing per-run subdirectories "
            f"(default: '{DEFAULT_OUT_DIR}')."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV prefix (default: 'results_<glob>')",
    )
    args = parser.parse_args()

    label = args.glob.strip("*").strip("_") or "all"
    out_prefix = args.out or f"results_{label}"

    df = collect(args.glob, results_dir=args.results_dir)
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
                f"Nb={int(row.get('Nb', 0)):>3d}  "
                f"Lb={int(row.get('Lb', 0)):>5d}  "
                f"Nh={str(row.get('Nh','?')):>5}  "
                f"N_ell={int(row.get('N_ell', 0)):>5d}  "
                f"Nc={int(row.get('Nc', 0)):>5d}  "
                f"coverage={float(row['coverage']):.4f}"
            )


if __name__ == "__main__":
    main()
