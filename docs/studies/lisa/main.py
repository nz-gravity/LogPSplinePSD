"""LISA multivariate PSD simulation study — single-seed entry point.

Usage:
    python main.py <seed> [options]

Examples:
    # Quick local test (7 days, no VI, no coarse graining)
    python main.py 0 --outdir out_test --duration-days 7 --K 10 --coarse-Nc 0 --no-vi

    # Full 365-day run with defaults
    python main.py 42 --outdir out_lisa_sim

    # Sweep knot method and window
    python main.py 0 --knot-method log --wishart-window hann --K 30

    # Test different knot counts (density placement)
    python main.py 5 --K 24 --knot-method density
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

from log_psplines.logger import logger, set_level  # noqa: E402

set_level("INFO")
logger.info(f"JAX devices: {jax.devices()}")

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
    """Build a human-readable slug encoding all key run settings."""
    from utils.windows import window_slug, window_spec

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
        description="LISA multivariate PSD simulation study (per-seed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "seed", type=int, help="Random seed for noise realization."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out_lisa_sim",
        help="Output root directory (default: out_lisa_sim).",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data_g = parser.add_argument_group("data")
    data_g.add_argument(
        "--duration-days",
        type=float,
        default=365.0,
        help="Data duration in days. Try 7, 30, 90, 180, 365 (default: 365).",
    )
    data_g.add_argument(
        "--block-days",
        type=float,
        default=7.0,
        help="Block duration in days (default: 7.0 → Nb=52 for 365 days).",
    )

    # ── Spline / model ────────────────────────────────────────────────────────
    model_g = parser.add_argument_group("model")
    model_g.add_argument(
        "--K",
        type=int,
        default=20,
        help="Number of P-spline knots (default: 20).",
    )
    model_g.add_argument(
        "--knot-method",
        type=str,
        choices=("density", "log", "uniform"),
        default="density",
        help=(
            "Knot placement method (default: density). "
            "density=quantile-based (adapts to spectral shape), "
            "log=logarithmically spaced, uniform=linearly spaced."
        ),
    )
    model_g.add_argument(
        "--diff-order",
        type=int,
        choices=(1, 2),
        default=2,
        help="P-spline difference penalty order (default: 2).",
    )
    model_g.add_argument(
        "--coarse-Nc",
        type=int,
        default=8192,
        help="Target coarse-grained freq bins; 0 = disabled (default: 8192).",
    )
    model_g.add_argument(
        "--null-excision",
        nargs="*",
        metavar="CENTER:HW",
        default=None,
        help=(
            "Null-band excision: remove frequency bins around LISA transfer nulls. "
            "Pass zero or more 'center:halfwidth' pairs in Hz "
            "(e.g. --null-excision 0.030:0.001 0.060:0.001 0.090:0.001). "
            "With no values, uses the three standard LISA TDI-2 nulls at "
            "0.030, 0.060, 0.090 Hz with ±1 mHz half-width."
        ),
    )
    model_g.add_argument(
        "--alpha-delta",
        type=float,
        default=3.0,
        help="Smoothing prior alpha (default: 3.0).",
    )
    model_g.add_argument(
        "--beta-delta",
        type=float,
        default=3.0,
        help="Smoothing prior beta (default: 3.0).",
    )

    # ── Windows ───────────────────────────────────────────────────────────────
    win_g = parser.add_argument_group("windows")
    win_g.add_argument(
        "--wishart-window",
        type=str,
        choices=("none", "hann", "tukey"),
        default="none",
        help="Taper applied to each block before Wishart likelihood FFT (default: none).",
    )
    win_g.add_argument(
        "--wishart-tukey-alpha",
        type=float,
        default=0.1,
        help="Tukey alpha for --wishart-window tukey (default: 0.1).",
    )
    win_g.add_argument(
        "--welch-window",
        type=str,
        choices=("none", "hann", "tukey"),
        default="hann",
        help="Window for Welch diagnostic overlay (default: hann).",
    )
    win_g.add_argument(
        "--welch-tukey-alpha",
        type=float,
        default=0.1,
        help="Tukey alpha for --welch-window tukey (default: 0.1).",
    )

    # ── MCMC ──────────────────────────────────────────────────────────────────
    mcmc_g = parser.add_argument_group("mcmc")
    mcmc_g.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Posterior draws per chain (default: 1000).",
    )
    mcmc_g.add_argument(
        "--n-warmup",
        type=int,
        default=1500,
        help="Warmup iterations per chain (default: 1500).",
    )
    mcmc_g.add_argument(
        "--num-chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4).",
    )
    mcmc_g.add_argument(
        "--target-accept",
        type=float,
        default=0.7,
        help="NUTS target acceptance (default: 0.7).",
    )
    mcmc_g.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help="NUTS max tree depth (default: 10).",
    )

    # ── VI ────────────────────────────────────────────────────────────────────
    vi_g = parser.add_argument_group("vi")
    vi_g.add_argument(
        "--vi",
        action="store_true",
        default=False,
        dest="vi",
        help="Enable VI initialization (default: off).",
    )
    vi_g.add_argument(
        "--no-vi",
        action="store_false",
        dest="vi",
        help="Disable VI initialization (default).",
    )
    vi_g.add_argument(
        "--vi-steps",
        type=int,
        default=500_000,
        help="VI optimization steps (default: 500000).",
    )

    # ── Design prior ──────────────────────────────────────────────────────────
    prior_g = parser.add_argument_group("prior")
    prior_g.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Design PSD tau for soft shrinkage (default: None = disabled).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out_g = parser.add_argument_group("output")
    out_g.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Skip PSD matrix plot generation.",
    )
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

    logger.info(f"=== LISA sim study: seed={args.seed} ===")
    logger.info(f"Run slug: {run_slug}")
    logger.info(
        f"Config: duration={args.duration_days}d, K={args.K}, "
        f"knot_method={args.knot_method}, diff_order={args.diff_order}, "
        f"block_days={args.block_days}, coarse_Nc={args.coarse_Nc}, "
        f"wishart_window={args.wishart_window}, "
        f"vi={'on' if args.vi else 'off'}, tau={args.tau}"
    )

    # Resolve window specs
    wishart_ws = window_spec(
        args.wishart_window, tukey_alpha=args.wishart_tukey_alpha
    )
    welch_ws = window_spec(
        args.welch_window, tukey_alpha=args.welch_tukey_alpha
    )

    # 1. Generate data
    ts, freq_true, S_true, Nb, Lb, dt = generate_lisa_data(
        seed=args.seed,
        duration_days=args.duration_days,
        block_days=args.block_days,
    )

    # 2. Setup coarse graining
    Nl = compute_Nl_analysis(Lb, dt)
    coarse_cfg = setup_coarse_grain(Nl, args.coarse_Nc)

    # 2b. Parse null-band excision
    _LISA_NULLS_DEFAULT = ((0.030, 0.001), (0.060, 0.001), (0.090, 0.001))
    if args.null_excision is None:
        freq_excl_bands = ()
    elif len(args.null_excision) == 0:
        # --null-excision with no values → use standard LISA nulls
        freq_excl_bands = tuple(
            (c - hw, c + hw) for c, hw in _LISA_NULLS_DEFAULT
        )
    else:
        freq_excl_bands = tuple(
            (
                float(p.split(":")[0]) - float(p.split(":")[1]),
                float(p.split(":")[0]) + float(p.split(":")[1]),
            )
            for p in args.null_excision
        )
    if freq_excl_bands:
        logger.info(f"Null-band excision bands: {freq_excl_bands}")

    # 3. Run MCMC
    idata = run_lisa_mcmc(
        ts,
        Nb=Nb,
        coarse_cfg=coarse_cfg,
        freq_true=freq_true,
        S_true=S_true,
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
        freq_excl_bands=freq_excl_bands,
        outdir=seed_dir,
    )

    runtime = time.time() - t0
    if not hasattr(idata, "attrs"):
        idata.attrs = {}
    idata.attrs["runtime"] = runtime
    # Stash config in attrs for collect_results.py
    idata.attrs["K"] = args.K
    idata.attrs["knot_method"] = args.knot_method
    idata.attrs["diff_order"] = args.diff_order
    idata.attrs["duration_days"] = args.duration_days
    idata.attrs["run_slug"] = run_slug
    idata.attrs["channel_basis"] = "XYZ"
    save_inference_data(idata, outdir=seed_dir)

    # 4. Extract metrics + save compact outputs
    metrics = extract_and_save_metrics(
        idata,
        seed=args.seed,
        freq_true=freq_true,
        S_true=S_true,
        outdir=seed_dir,
    )
    # Also record config fields in the metrics JSON
    metrics["K"] = args.K
    metrics["knot_method"] = args.knot_method
    metrics["diff_order"] = args.diff_order
    metrics["duration_days"] = args.duration_days
    metrics["run_slug"] = run_slug

    # Re-save with config fields appended
    import csv
    import json

    out_json = os.path.join(seed_dir, "compact_run_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    out_csv = os.path.join(seed_dir, "compact_run_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # 5. Plot (optional)
    if not args.no_plot:
        try:
            make_psd_plot(
                idata,
                y_full=ts.y,
                fs=1.0 / dt,
                Lb=Lb,
                Nb=Nb,
                freq_true=freq_true,
                S_true=S_true,
                outdir=seed_dir,
                welch_window=welch_ws,
            )
        except Exception as exc:
            logger.warning(f"Plot generation failed: {exc}")

    # 6. Cleanup large files (unless explicitly disabled)
    nc_path = Path(seed_dir) / "inference_data.nc"
    if nc_path.exists() and not args.keep_nc:
        nc_path.unlink()
        logger.info(f"Removed {nc_path} to save disk space.")
    elif nc_path.exists() and args.keep_nc:
        logger.info(f"Keeping {nc_path} (--keep-nc set).")

    logger.info(
        f"=== Seed {args.seed} complete in {runtime:.0f}s | "
        f"coverage={metrics.get('coverage', 'N/A')} | "
        f"riae={metrics.get('riae_matrix', 'N/A')} ==="
    )


if __name__ == "__main__":
    main()
