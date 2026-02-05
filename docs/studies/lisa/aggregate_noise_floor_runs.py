#!/usr/bin/env python
"""Aggregate diagnostics from LISA run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


def _tree_depth_hits(sample_stats, p: int, max_tree_depth: int) -> dict:
    max_steps = 2**max_tree_depth - 1
    hits = {}
    hit_fracs = []
    for ch in range(p):
        key = f"num_steps_channel_{ch}"
        if key not in sample_stats:
            hits[f"td_hit_c{ch}"] = np.nan
            continue
        steps = np.asarray(sample_stats[key].values)
        frac = float(np.mean(steps >= max_steps))
        hits[f"td_hit_c{ch}"] = frac
        hit_fracs.append(frac)
    hits["td_hit_max"] = float(np.max(hit_fracs)) if hit_fracs else np.nan
    hits["td_hit_mean"] = float(np.mean(hit_fracs)) if hit_fracs else np.nan
    return hits


def _summarize_csv(summary_path: Path) -> dict:
    df = pd.read_csv(summary_path)
    if "r_hat" not in df.columns or "ess_bulk" not in df.columns:
        raise ValueError(f"Missing r_hat/ess_bulk in {summary_path}")
    rhat = pd.to_numeric(df["r_hat"], errors="coerce")
    ess_bulk = pd.to_numeric(df["ess_bulk"], errors="coerce")
    ess_tail = (
        pd.to_numeric(df["ess_tail"], errors="coerce")
        if "ess_tail" in df.columns
        else None
    )
    stats = {
        "rhat_max": float(np.nanmax(rhat)),
        "rhat_mean": float(np.nanmean(rhat)),
        "ess_bulk_min": float(np.nanmin(ess_bulk)),
        "ess_bulk_med": float(np.nanmedian(ess_bulk)),
    }
    if ess_tail is not None:
        stats["ess_tail_min"] = float(np.nanmin(ess_tail))
        stats["ess_tail_med"] = float(np.nanmedian(ess_tail))
    return stats


def _collect_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary_statistics.csv"
    idata_path = run_dir / "inference_data.nc"
    if not summary_path.exists() or not idata_path.exists():
        return None

    summary_stats = _summarize_csv(summary_path)
    idata = az.from_netcdf(str(idata_path))

    p = int(idata.attrs.get("p", 0))
    max_tree_depth = int(idata.attrs.get("max_tree_depth", 10))
    td_stats = _tree_depth_hits(idata.sample_stats, p, max_tree_depth)

    noise_mode = idata.attrs.get("noise_floor_mode", "")
    noise_scale = idata.attrs.get("noise_floor_scale", np.nan)
    noise_const = idata.attrs.get("noise_floor_constant", np.nan)
    noise_blocks = idata.attrs.get("noise_floor_blocks", "")
    use_floor = bool(idata.attrs.get("use_noise_floor", False))

    return {
        "run": run_dir.name,
        "use_floor": use_floor,
        "mode": str(noise_mode),
        "scale": float(noise_scale) if noise_scale is not None else np.nan,
        "constant": float(noise_const) if noise_const is not None else np.nan,
        "blocks": str(noise_blocks),
        "max_tree_depth": max_tree_depth,
        **summary_stats,
        **td_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate LISA run diagnostics from summary_statistics.csv and inference_data.nc."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Root folder containing lisa* run directories.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format.",
    )
    args = parser.parse_args()

    root = args.results_root
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    runs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name != "lisa" and not entry.name.startswith("lisa_"):
            continue
        row = _collect_run(entry)
        if row is not None:
            runs.append(row)

    if not runs:
        print(
            "No runs found with summary_statistics.csv and inference_data.nc"
        )
        return 0

    df = pd.DataFrame(runs)
    df = df.sort_values("run")

    if args.format == "csv":
        print(df.to_csv(index=False))
    else:
        with pd.option_context("display.max_columns", None):
            print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
