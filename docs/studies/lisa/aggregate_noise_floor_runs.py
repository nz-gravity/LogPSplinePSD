#!/usr/bin/env python
"""Aggregate diagnostics from LISA run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


def _tree_depth_hits(
    sample_stats, n_channels: int, max_tree_depth: int
) -> dict:
    max_steps = 2**max_tree_depth - 1
    hits = {}
    hit_fracs = []
    for ch in range(n_channels):
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


def _noise_floor_metrics(idata: az.InferenceData, channel: int = 2) -> dict:
    """Compute noise-floor diagnostic metrics for a given channel.

    Metrics (per user notes):
    A) frac_theory_active = mean( scale*theory(f) > constant )
    B) frac_floor = mean( delta_sq_median(f) < 0.1*eps2(f) )
    C) r(f) = eps2(f) / (delta_eff_sq_median(f)) where delta_eff = delta_sq + eps2
    """
    if not hasattr(idata, "noise_floor"):
        return {}
    if (
        not hasattr(idata, "sample_stats")
        or "log_delta_sq" not in idata.sample_stats
    ):
        return {}

    try:
        eps2 = np.asarray(
            idata.noise_floor["eps2_floor_std"].sel(channels=channel).values,
            dtype=np.float64,
        )
    except Exception:
        return {}

    log_delta_sq = np.asarray(
        idata.sample_stats["log_delta_sq"].values, dtype=np.float64
    )
    if log_delta_sq.ndim != 4 or channel >= log_delta_sq.shape[-1]:
        return {}

    delta_sq = np.exp(log_delta_sq[..., channel])  # (chain, draw, freq)
    delta_sq_med = np.median(delta_sq, axis=(0, 1))

    constant = float(idata.attrs.get("noise_floor_constant", np.nan))
    scale = float(idata.attrs.get("noise_floor_scale", np.nan))

    # A: fraction where theory_scaled beats constant (hard indicator).
    frac_theory_active = np.nan
    try:
        theory = np.asarray(
            idata.noise_floor["theory_psd_std"].sel(channels=channel).values,
            dtype=np.float64,
        )
        mask = np.isfinite(theory) & np.isfinite(eps2)
        if np.any(mask) and np.isfinite(constant) and np.isfinite(scale):
            frac_theory_active = float(
                np.mean((scale * theory[mask]) > constant)
            )
    except Exception:
        frac_theory_active = np.nan

    # B: floor dominance over posterior median delta^2.
    mask2 = np.isfinite(delta_sq_med) & np.isfinite(eps2)
    frac_floor = (
        float(np.mean(delta_sq_med[mask2] < (0.1 * eps2[mask2])))
        if np.any(mask2)
        else np.nan
    )

    # C: r(f) summaries.
    delta_eff_med = delta_sq_med + eps2
    mask3 = (
        np.isfinite(delta_eff_med)
        & (delta_eff_med > 0)
        & np.isfinite(eps2)
        & (eps2 >= 0)
    )
    if np.any(mask3):
        r = eps2[mask3] / delta_eff_med[mask3]
        r_med = float(np.median(r))
        r_p95 = float(np.quantile(r, 0.95))
        r_frac_gt_05 = float(np.mean(r > 0.5))
        r_frac_gt_09 = float(np.mean(r > 0.9))
    else:
        r_med = r_p95 = r_frac_gt_05 = r_frac_gt_09 = np.nan

    return {
        "frac_theory_active": frac_theory_active,
        "frac_floor_dom": frac_floor,
        "r_med": r_med,
        "r_p95": r_p95,
        "r_frac_gt_0p5": r_frac_gt_05,
        "r_frac_gt_0p9": r_frac_gt_09,
    }


def _collect_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary_statistics.csv"
    idata_path = run_dir / "inference_data.nc"
    if not summary_path.exists() or not idata_path.exists():
        return None

    summary_stats = _summarize_csv(summary_path)
    idata = az.from_netcdf(str(idata_path))

    n_channels = int(idata.attrs.get("n_channels", 0))
    max_tree_depth = int(idata.attrs.get("max_tree_depth", 10))
    td_stats = _tree_depth_hits(idata.sample_stats, n_channels, max_tree_depth)

    noise_mode = idata.attrs.get("noise_floor_mode", "")
    noise_scale = idata.attrs.get("noise_floor_scale", np.nan)
    noise_const = idata.attrs.get("noise_floor_constant", np.nan)
    noise_tau = idata.attrs.get("noise_floor_tau", np.nan)
    noise_blocks = idata.attrs.get("noise_floor_blocks", "")
    use_floor = bool(idata.attrs.get("use_noise_floor", False))

    nf_metrics = _noise_floor_metrics(idata, channel=2)

    return {
        "run": run_dir.name,
        "use_floor": use_floor,
        "mode": str(noise_mode),
        "scale": float(noise_scale) if noise_scale is not None else np.nan,
        "constant": float(noise_const) if noise_const is not None else np.nan,
        "tau": float(noise_tau) if noise_tau is not None else np.nan,
        "blocks": str(noise_blocks),
        "max_tree_depth": max_tree_depth,
        **nf_metrics,
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
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Include the base 'lisa' directory in addition to 'lisa_*' runs.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default="",
        help="Only include run directories whose name contains this substring (e.g. 'hybrid_').",
    )
    args = parser.parse_args()

    root = args.results_root
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    runs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "lisa":
            if not args.include_base:
                continue
        elif not entry.name.startswith("lisa_"):
            continue
        if args.name_contains and args.name_contains not in entry.name:
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
