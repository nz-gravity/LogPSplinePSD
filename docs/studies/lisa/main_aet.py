"""LISA multivariate PSD study — native AET channel analysis (run_y).

Generates LISA XYZ noise, transforms timeseries and true PSD to the AET basis,
then runs the same P-spline MCMC pipeline as main.py but in the AET frame.

Usage:
    python main_aet.py <seed> [options]

Examples:
    python main_aet.py 0 --outdir runs/run_y_aet --K 48 --knot-method uniform
    python main_aet.py 0 --outdir runs/run_y_aet --K 48 --no-vi --keep-nc
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.example_datasets.lisatools_backend import (  # noqa: E402
    ensure_lisatools_backends,
)

ensure_lisatools_backends()

import jax  # noqa: E402

from log_psplines.datatypes import MultivariateTimeseries  # noqa: E402
from log_psplines.logger import logger, set_level  # noqa: E402

set_level("INFO")
logger.info(f"JAX devices: {jax.devices()}")

from utils.aet import (  # noqa: E402
    CHANNEL_LABELS_AET,
    M_AET,
    xyz_to_aet_matrix,
    xyz_to_aet_timeseries,
)
from utils.data import generate_lisa_data  # noqa: E402
from utils.inference import run_lisa_mcmc, save_inference_data  # noqa: E402
from utils.metrics import extract_and_save_metrics  # noqa: E402
from utils.plotting import make_psd_plot  # noqa: E402
from utils.preprocessing import (  # noqa: E402
    compute_Nl_analysis,
    setup_coarse_grain,
)
from utils.windows import window_slug, window_spec  # noqa: E402


def _build_run_slug(args) -> str:
    wishart_ws = window_spec(
        args.wishart_window, tukey_alpha=args.wishart_tukey_alpha
    )
    welch_ws = window_spec(
        args.welch_window, tukey_alpha=args.welch_tukey_alpha
    )
    vi_tag = "viOn" if args.vi else "viOff"
    tau_tag = (
        f"tau{args.tau:g}".replace(".", "p")
        if args.tau is not None
        else "tauOff"
    )
    return (
        f"k{args.K}"
        f"_d{args.diff_order}"
        f"_km{args.knot_method}"
        f"_ww{window_slug(wishart_ws)}"
        f"_ew{window_slug(welch_ws)}"
        f"_nc{args.coarse_Nc}"
        f"_bd{args.block_days:g}d"
        f"_ta{args.target_accept:g}"
        f"_td{args.max_tree_depth}"
        f"_{vi_tag}"
        f"_{tau_tag}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LISA AET-channel PSD study (per-seed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("seed", type=int)
    parser.add_argument("--outdir", type=str, default="runs/run_y_aet")

    data_g = parser.add_argument_group("data")
    data_g.add_argument("--duration-days", type=float, default=365.0)
    data_g.add_argument("--block-days", type=float, default=7.0)

    model_g = parser.add_argument_group("model")
    model_g.add_argument("--K", type=int, default=48)
    model_g.add_argument(
        "--knot-method",
        type=str,
        choices=("density", "log", "uniform"),
        default="uniform",
    )
    model_g.add_argument("--diff-order", type=int, choices=(1, 2), default=2)
    model_g.add_argument("--coarse-Nc", type=int, default=8192)
    model_g.add_argument("--alpha-delta", type=float, default=3.0)
    model_g.add_argument("--beta-delta", type=float, default=3.0)

    win_g = parser.add_argument_group("windows")
    win_g.add_argument(
        "--wishart-window",
        type=str,
        choices=("none", "hann", "tukey"),
        default="tukey",
    )
    win_g.add_argument("--wishart-tukey-alpha", type=float, default=0.1)
    win_g.add_argument(
        "--welch-window",
        type=str,
        choices=("none", "hann", "tukey"),
        default="hann",
    )
    win_g.add_argument("--welch-tukey-alpha", type=float, default=0.1)

    mcmc_g = parser.add_argument_group("mcmc")
    mcmc_g.add_argument("--n-samples", type=int, default=1000)
    mcmc_g.add_argument("--n-warmup", type=int, default=1500)
    mcmc_g.add_argument("--num-chains", type=int, default=4)
    mcmc_g.add_argument("--target-accept", type=float, default=0.7)
    mcmc_g.add_argument("--max-tree-depth", type=int, default=10)

    vi_g = parser.add_argument_group("vi")
    vi_g.add_argument("--vi", action="store_true", default=False, dest="vi")
    vi_g.add_argument("--no-vi", action="store_false", dest="vi")
    vi_g.add_argument("--vi-steps", type=int, default=500_000)

    prior_g = parser.add_argument_group("prior")
    prior_g.add_argument("--tau", type=float, default=None)

    out_g = parser.add_argument_group("output")
    out_g.add_argument("--no-plot", action="store_true", default=False)
    out_g.add_argument(
        "--keep-nc",
        dest="keep_nc",
        action="store_true",
        default=True,
        help="Keep inference_data.nc after run (default: enabled).",
    )
    out_g.add_argument(
        "--delete-nc",
        dest="keep_nc",
        action="store_false",
        help="Delete inference_data.nc after run to save disk.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    run_slug = _build_run_slug(args)
    seed_dir = os.path.join(args.outdir, run_slug, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)

    logger.info(f"=== LISA AET sim study: seed={args.seed} ===")
    logger.info(f"Run slug: {run_slug}")

    wishart_ws = window_spec(
        args.wishart_window, tukey_alpha=args.wishart_tukey_alpha
    )
    welch_ws = window_spec(
        args.welch_window, tukey_alpha=args.welch_tukey_alpha
    )

    # 1. Generate XYZ data then transform to AET
    ts_xyz, freq_true, S_true_xyz, Nb, Lb, dt = generate_lisa_data(
        seed=args.seed,
        duration_days=args.duration_days,
        block_days=args.block_days,
    )

    # Transform timeseries: (N, 3) XYZ -> (N, 3) AET
    y_aet = xyz_to_aet_timeseries(ts_xyz.y)
    import numpy as np

    ts_aet = MultivariateTimeseries(y=y_aet, t=ts_xyz.t)

    # Transform true PSD: (Nf, 3, 3) XYZ -> (Nf, 3, 3) AET
    S_true_aet = xyz_to_aet_matrix(np.asarray(S_true_xyz))

    logger.info(
        f"AET transform applied. y_aet shape: {y_aet.shape}, "
        f"S_true_aet shape: {S_true_aet.shape}"
    )

    # 2. Setup coarse graining (same as XYZ run)
    Nl = compute_Nl_analysis(Lb, dt)
    coarse_cfg = setup_coarse_grain(Nl, args.coarse_Nc)

    # 3. Run MCMC in AET basis
    idata = run_lisa_mcmc(
        ts_aet,
        Nb=Nb,
        coarse_cfg=coarse_cfg,
        freq_true=freq_true,
        S_true=S_true_aet,
        K=args.K,
        knot_method=args.knot_method,
        diff_order=args.diff_order,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        num_chains=args.num_chains,
        target_accept=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        alpha_delta=args.alpha_delta,
        beta_delta=args.beta_delta,
        vi=args.vi,
        vi_steps=args.vi_steps,
        wishart_window=wishart_ws,
        tau=args.tau,
        outdir=seed_dir,
    )

    runtime = time.time() - t0
    if not hasattr(idata, "attrs"):
        idata.attrs = {}
    idata.attrs["runtime"] = runtime
    idata.attrs["K"] = args.K
    idata.attrs["knot_method"] = args.knot_method
    idata.attrs["diff_order"] = args.diff_order
    idata.attrs["duration_days"] = args.duration_days
    idata.attrs["run_slug"] = run_slug
    idata.attrs["channel_basis"] = "AET"
    save_inference_data(idata, outdir=seed_dir)

    # 4. Extract metrics
    import csv
    import json

    metrics = extract_and_save_metrics(
        idata,
        seed=args.seed,
        freq_true=freq_true,
        S_true=S_true_aet,
        outdir=seed_dir,
    )
    metrics["K"] = args.K
    metrics["knot_method"] = args.knot_method
    metrics["diff_order"] = args.diff_order
    metrics["duration_days"] = args.duration_days
    metrics["run_slug"] = run_slug
    metrics["channel_basis"] = "AET"

    out_json = os.path.join(seed_dir, "compact_run_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    out_csv = os.path.join(seed_dir, "compact_run_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # 5. Plot (AET channel labels)
    if not args.no_plot:
        try:
            make_psd_plot(
                idata,
                y_full=y_aet,
                fs=1.0 / dt,
                Lb=Lb,
                Nb=Nb,
                freq_true=freq_true,
                S_true=S_true_aet,
                outdir=seed_dir,
                welch_window=welch_ws,
            )
        except Exception as exc:
            logger.warning(f"Plot generation failed: {exc}")

    # 6. Cleanup
    nc_path = Path(seed_dir) / "inference_data.nc"
    if nc_path.exists() and not args.keep_nc:
        nc_path.unlink()
        logger.info(f"Removed {nc_path}.")
    elif nc_path.exists() and args.keep_nc:
        logger.info(f"Keeping {nc_path} (--keep-nc set).")

    logger.info(
        f"=== Seed {args.seed} (AET) complete in {runtime:.0f}s | "
        f"coverage={metrics.get('coverage', 'N/A')} | "
        f"riae={metrics.get('riae_matrix', 'N/A')} ==="
    )


if __name__ == "__main__":
    main()
