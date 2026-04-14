"""Sweep explicit eta values for long-duration LISA blocked-NUTS runs.

This helper targets the specific failure mode where the third blocked-NUTS
channel collapses to a tiny step size on long-duration LISA analyses.

Each sweep run:
1. Reuses the same simulated LISA dataset for all eta values within a seed.
2. Runs the standard LISA inference path with an explicit ``eta``.
3. Saves compact run metrics plus per-channel NUTS diagnostics.
4. Writes one aggregate CSV/JSON at the sweep root for quick comparison.

Example
-------
python docs/studies/lisa/eta_sweep.py \
  --seeds 0 1 2 \
  --etas 1.0 0.25 0.125 0.08 0.06 0.04
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

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
from log_psplines.preprocessing.coarse_grain import (  # noqa: E402
    CoarseGrainConfig,
)

set_level("INFO")
logger.info(f"JAX devices: {jax.devices()}")

from utils.data import generate_lisa_data  # noqa: E402
from utils.inference import run_lisa_mcmc, save_inference_data  # noqa: E402
from utils.metrics import extract_and_save_metrics  # noqa: E402
from utils.preprocessing import (  # noqa: E402
    build_transfer_null_exclusion_bands,
    compute_analysis_frequencies,
    compute_Nl_analysis,
    setup_coarse_grain,
)
from utils.windows import window_slug, window_spec  # noqa: E402


def _format_knot_counts(knot_counts: int | Mapping[str, int]) -> str:
    if isinstance(knot_counts, Mapping):
        return (
            f"delta={int(knot_counts['delta'])}, "
            f"theta_re={int(knot_counts['theta_re'])}, "
            f"theta_im={int(knot_counts['theta_im'])}"
        )
    return str(int(knot_counts))


def _resolve_family_knot_counts(
    args: argparse.Namespace,
) -> int | dict[str, int]:
    overrides = {
        "delta": args.K_delta,
        "theta_re": args.K_theta_re,
        "theta_im": args.K_theta_im,
    }
    if all(value is None for value in overrides.values()):
        return int(args.K)
    base = int(args.K)
    return {
        key: int(base if value is None else value)
        for key, value in overrides.items()
    }


def _eta_slug(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


def _label_to_index(label: str) -> int:
    value = 0
    for char in label.upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid run label '{label}'.")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value


def _index_to_label(index: int) -> str:
    if index <= 0:
        raise ValueError("index must be positive.")
    chars: list[str] = []
    current = index
    while current > 0:
        current -= 1
        chars.append(chr(ord("A") + (current % 26)))
        current //= 26
    return "".join(reversed(chars))


def _default_labels(n_labels: int, *, start: str = "AQ") -> list[str]:
    start_index = _label_to_index(start)
    return [
        _index_to_label(start_index + offset) for offset in range(n_labels)
    ]


def _resolve_run_labels(args: argparse.Namespace) -> list[str]:
    labels = (
        [label.upper() for label in args.labels]
        if args.labels is not None
        else _default_labels(len(args.etas))
    )
    if len(labels) != len(args.etas):
        raise ValueError(
            "Number of labels must match number of eta values: "
            f"{len(labels)} labels for {len(args.etas)} etas."
        )
    return labels


def _build_run_dir_name(
    *,
    label: str,
    args: argparse.Namespace,
    eta: float,
    knot_counts: int | Mapping[str, int],
) -> str:
    knot_token = (
        f"k{int(knot_counts)}"
        if not isinstance(knot_counts, Mapping)
        else (
            f"kdelta{int(knot_counts['delta'])}"
            f"_ktre{int(knot_counts['theta_re'])}"
            f"_ktim{int(knot_counts['theta_im'])}"
        )
    )
    null_tag = (
        f"nullexc{int(args.exclude_bins_per_side)}"
        if args.exclude_transfer_nulls
        else "nonullexc"
    )
    vi_tag = "vi" if args.vi else "novi"
    return (
        f"run_{label}_{int(args.duration_days)}d_eta{_eta_slug(eta)}_"
        f"{knot_token}_{args.knot_method}_{null_tag}_{vi_tag}"
    )


def _resolve_vi_coarse_cfg(
    *,
    target_nfreq: int,
    explicit_nc: int,
    auto_enabled: bool,
    auto_target_nfreq: int,
) -> CoarseGrainConfig | None:
    """Build a coarse-VI config request for the final analysis grid.

    The core preprocessing now derives and applies coarse-VI on the final
    post-exclusion analysis grid, so this helper only needs to express the
    target coarse bin count.
    """
    if explicit_nc > 0:
        return CoarseGrainConfig(enabled=True, Nc=int(explicit_nc))
    if not auto_enabled:
        return None
    return CoarseGrainConfig(
        enabled=True,
        Nc=min(int(auto_target_nfreq), int(target_nfreq)),
    )


def _safe_stat(values: np.ndarray, fn, *, default: float = np.nan) -> float:
    if values.size == 0:
        return default
    return float(fn(values))


def _count_retained_after_exclusion(
    freq: np.ndarray,
    bands: Sequence[tuple[float, float]],
) -> int:
    mask = np.ones(len(freq), dtype=bool)
    for low, high in bands:
        mask &= ~((freq >= float(low)) & (freq <= float(high)))
    return int(np.count_nonzero(mask))


def _extract_channel_diagnostics(idata) -> dict[str, float | int]:
    """Extract compact per-channel NUTS diagnostics from an InferenceData."""
    metrics: dict[str, float | int] = {}
    attrs = getattr(idata, "attrs", {}) or {}
    p = int(attrs.get("p", 0) or 0)
    sample_stats = getattr(idata, "sample_stats", None)

    for channel_index in range(p):
        eta_attr = attrs.get(f"sampling_eta_channel_{channel_index}")
        if eta_attr is not None:
            metrics[f"sampling_eta_channel_{channel_index}"] = float(eta_attr)

        if sample_stats is None:
            continue

        for field in ("step_size", "accept_prob", "num_steps", "diverging"):
            key = f"{field}_channel_{channel_index}"
            if key not in sample_stats:
                continue
            values = np.asarray(sample_stats[key].values)
            flat = values.reshape(-1)

            if field == "diverging":
                metrics[f"n_divergences_channel_{channel_index}"] = int(
                    np.sum(flat.astype(np.int64))
                )
                continue

            flat = flat.astype(np.float64, copy=False)
            metrics[f"{field}_channel_{channel_index}_mean"] = _safe_stat(
                flat, np.nanmean
            )
            metrics[f"{field}_channel_{channel_index}_median"] = _safe_stat(
                flat, np.nanmedian
            )
            metrics[f"{field}_channel_{channel_index}_min"] = _safe_stat(
                flat, np.nanmin
            )
            metrics[f"{field}_channel_{channel_index}_max"] = _safe_stat(
                flat, np.nanmax
            )

    return metrics


def _write_summary_rows(
    outdir: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    if not rows:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    summary_json = outdir / "eta_sweep_summary.json"
    summary_csv = outdir / "eta_sweep_summary.csv"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2, sort_keys=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"Wrote aggregate eta sweep summary to {summary_csv} ({len(rows)} rows)."
    )


def _load_existing_summary(seed_dir: Path) -> dict[str, Any] | None:
    summary_path = seed_dir / "compact_run_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep explicit eta values on long-duration LISA runs."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "studies" / "lisa" / "runs"),
        help="Base runs directory for the sweep.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional run labels, one per eta value (e.g. AQ AR AS).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Noise realization seeds to run.",
    )
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=[1.0, 0.25, 0.125, 0.08, 0.06, 0.04],
        help="Explicit eta values to sweep.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cases even if compact summaries already exist.",
    )
    parser.add_argument(
        "--keep-nc",
        action="store_true",
        help="Keep inference_data.nc for each run.",
    )

    parser.add_argument("--duration-days", type=float, default=365.0)
    parser.add_argument("--block-days", type=float, default=7.0)
    parser.add_argument(
        "--absolute-freq-units", action="store_true", default=True
    )
    parser.add_argument(
        "--no-absolute-freq-units",
        action="store_false",
        dest="absolute_freq_units",
    )

    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--K-delta", type=int, default=None)
    parser.add_argument("--K-theta-re", type=int, default=None)
    parser.add_argument("--K-theta-im", type=int, default=4)
    parser.add_argument(
        "--knot-method",
        type=str,
        choices=("density", "log", "uniform"),
        default="density",
    )
    parser.add_argument("--diff-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--fmin", type=float, default=1e-4)
    parser.add_argument("--fmax", type=float, default=1e-1)
    parser.add_argument("--coarse-Nc", type=int, default=8192)

    parser.add_argument(
        "--wishart-window",
        type=str,
        choices=("none", "hann", "tukey"),
        default="tukey",
    )
    parser.add_argument("--wishart-tukey-alpha", type=float, default=0.1)
    parser.add_argument(
        "--wishart-detrend",
        type=str,
        choices=("none", "constant", "linear"),
        default="constant",
    )
    parser.add_argument("--wishart-floor-fraction", type=float, default=1e-6)

    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-warmup", type=int, default=1500)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.7)
    parser.add_argument("--max-tree-depth", type=int, default=10)

    parser.add_argument("--alpha-delta", type=float, default=3.0)
    parser.add_argument("--beta-delta", type=float, default=3.0)

    parser.add_argument("--vi", action="store_true", default=True)
    parser.add_argument("--no-vi", action="store_false", dest="vi")
    parser.add_argument("--vi-steps", type=int, default=500_000)
    parser.add_argument("--vi-guide", type=str, default="diag")
    parser.add_argument("--vi-posterior-draws", type=int, default=1024)
    parser.add_argument("--vi-coarse-Nc", type=int, default=0)
    parser.add_argument("--auto-coarse-vi", action="store_true", default=True)
    parser.add_argument(
        "--no-auto-coarse-vi", action="store_false", dest="auto_coarse_vi"
    )
    parser.add_argument("--auto-coarse-vi-target-nfreq", type=int, default=192)

    parser.add_argument(
        "--exclude-transfer-nulls", action="store_true", default=True
    )
    parser.add_argument(
        "--no-exclude-transfer-nulls",
        action="store_false",
        dest="exclude_transfer_nulls",
    )
    parser.add_argument("--exclude-bins-per-side", type=int, default=3)
    parser.add_argument("--exclude-half-width", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    knot_counts = _resolve_family_knot_counts(args)
    labels = _resolve_run_labels(args)

    wishart_ws = window_spec(
        args.wishart_window, tukey_alpha=args.wishart_tukey_alpha
    )
    wishart_detrend = (
        False if args.wishart_detrend == "none" else args.wishart_detrend
    )

    logger.info(
        f"Starting LISA eta sweep: seeds={args.seeds}, etas={args.etas}, "
        f"labels={labels}, "
        f"duration={args.duration_days}d, K={_format_knot_counts(knot_counts)}, "
        f"knot_method={args.knot_method}, diff_order={args.diff_order}, "
        f"window={window_slug(wishart_ws)}, coarse_Nc={args.coarse_Nc}, "
        f"vi={'on' if args.vi else 'off'}"
    )

    rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        ts, freq_true, S_true, Nb, Lb, dt = generate_lisa_data(
            seed=seed,
            duration_days=args.duration_days,
            block_days=args.block_days,
            fmin_generate=min(args.fmin, 1e-5),
            fmax_generate=max(args.fmax, 1e-1),
            absolute_freq_units=args.absolute_freq_units,
        )
        n_freq_analysis = compute_Nl_analysis(
            Lb, dt, fmin=args.fmin, fmax=args.fmax
        )
        coarse_cfg = setup_coarse_grain(n_freq_analysis, args.coarse_Nc)
        if args.exclude_transfer_nulls:
            retained_freq = compute_analysis_frequencies(
                Lb,
                dt,
                fmin=args.fmin,
                fmax=args.fmax,
                coarse_cfg=coarse_cfg,
            )
            exclude_freq_bands = build_transfer_null_exclusion_bands(
                retained_freq,
                bins_per_side=args.exclude_bins_per_side,
                half_width=args.exclude_half_width,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            processed_nfreq = _count_retained_after_exclusion(
                retained_freq, exclude_freq_bands
            )
        else:
            exclude_freq_bands = ()
            retained_freq = compute_analysis_frequencies(
                Lb,
                dt,
                fmin=args.fmin,
                fmax=args.fmax,
                coarse_cfg=coarse_cfg,
            )
            processed_nfreq = int(len(retained_freq))
        coarse_vi_cfg = _resolve_vi_coarse_cfg(
            target_nfreq=processed_nfreq,
            explicit_nc=int(args.vi_coarse_Nc),
            auto_enabled=bool(args.auto_coarse_vi),
            auto_target_nfreq=int(args.auto_coarse_vi_target_nfreq),
        )
        if coarse_vi_cfg is not None:
            logger.info(
                "Eta sweep coarse VI config: "
                f"processed_nfreq={processed_nfreq}, target={args.auto_coarse_vi_target_nfreq}, "
                f"Nc={coarse_vi_cfg.Nc}"
            )
        use_downstream_auto_coarse_vi = (
            bool(args.auto_coarse_vi) if coarse_vi_cfg is None else False
        )

        for label, eta in zip(labels, args.etas, strict=True):
            eta_value = float(eta)
            run_dir = outdir / _build_run_dir_name(
                label=label,
                args=args,
                eta=eta_value,
                knot_counts=knot_counts,
            )
            eta_dir = run_dir / f"seed_{seed}"
            existing = None if args.force else _load_existing_summary(eta_dir)
            if existing is not None:
                logger.info(
                    f"Skipping existing eta sweep run: label={label}, "
                    f"seed={seed}, eta={eta_value:g}"
                )
                rows.append(existing)
                continue

            eta_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Running eta sweep case: label={label}, seed={seed}, eta={eta_value:g}, "
                f"Nb={Nb}, coarse_enabled={coarse_cfg.enabled}"
            )

            run_start = time.time()
            idata = run_lisa_mcmc(
                ts,
                Nb=Nb,
                coarse_cfg=coarse_cfg,
                freq_true=freq_true,
                S_true=S_true,
                K=knot_counts,
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
                vi_guide=args.vi_guide,
                vi_posterior_draws=args.vi_posterior_draws,
                coarse_grain_config_vi=coarse_vi_cfg,
                auto_coarse_vi=use_downstream_auto_coarse_vi,
                auto_coarse_vi_target_nfreq=args.auto_coarse_vi_target_nfreq,
                fmin=args.fmin,
                fmax=args.fmax,
                wishart_window=wishart_ws,
                wishart_detrend=wishart_detrend,
                wishart_floor_fraction=args.wishart_floor_fraction,
                exclude_freq_bands=exclude_freq_bands,
                tau=args.tau,
                outdir=str(eta_dir),
                eta=eta_value,
            )

            runtime = time.time() - run_start
            idata.attrs["runtime"] = runtime
            idata.attrs["duration_days"] = args.duration_days
            idata.attrs["eta"] = eta_value
            idata.attrs["block_days"] = args.block_days
            idata.attrs["coarse_Nc"] = args.coarse_Nc
            idata.attrs["knot_method"] = args.knot_method
            idata.attrs["diff_order"] = args.diff_order
            idata.attrs["K"] = int(args.K)
            idata.attrs["K_delta"] = (
                knot_counts["delta"]
                if isinstance(knot_counts, Mapping)
                else int(knot_counts)
            )
            idata.attrs["K_theta_re"] = (
                knot_counts["theta_re"]
                if isinstance(knot_counts, Mapping)
                else int(knot_counts)
            )
            idata.attrs["K_theta_im"] = (
                knot_counts["theta_im"]
                if isinstance(knot_counts, Mapping)
                else int(knot_counts)
            )

            if args.keep_nc:
                save_inference_data(idata, outdir=str(eta_dir))

            metrics = extract_and_save_metrics(
                idata,
                seed=seed,
                freq_true=freq_true,
                S_true=S_true,
                outdir=str(eta_dir),
                excluded_bands=exclude_freq_bands,
            )
            metrics.update(_extract_channel_diagnostics(idata))
            metrics.update(
                {
                    "run_label": label,
                    "run_dir": run_dir.name,
                    "eta": eta_value,
                    "seed": int(seed),
                    "duration_days": float(args.duration_days),
                    "block_days": float(args.block_days),
                    "Nb": int(Nb),
                    "K": int(args.K),
                    "K_delta": (
                        knot_counts["delta"]
                        if isinstance(knot_counts, Mapping)
                        else int(knot_counts)
                    ),
                    "K_theta_re": (
                        knot_counts["theta_re"]
                        if isinstance(knot_counts, Mapping)
                        else int(knot_counts)
                    ),
                    "K_theta_im": (
                        knot_counts["theta_im"]
                        if isinstance(knot_counts, Mapping)
                        else int(knot_counts)
                    ),
                    "knot_method": args.knot_method,
                    "diff_order": int(args.diff_order),
                    "coarse_Nc": int(args.coarse_Nc),
                    "wishart_window": window_slug(wishart_ws),
                    "wishart_detrend": str(args.wishart_detrend),
                    "wishart_floor_fraction": (
                        float(args.wishart_floor_fraction)
                        if args.wishart_floor_fraction is not None
                        else np.nan
                    ),
                    "exclude_transfer_nulls": int(args.exclude_transfer_nulls),
                    "exclude_bins_per_side": int(args.exclude_bins_per_side),
                    "vi": int(args.vi),
                    "target_accept": float(args.target_accept),
                    "max_tree_depth": int(args.max_tree_depth),
                    "summary_path": str(eta_dir / "compact_run_summary.json"),
                }
            )

            summary_path = eta_dir / "compact_run_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            with (eta_dir / "compact_run_summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)

            nc_path = eta_dir / "inference_data.nc"
            if nc_path.exists() and not args.keep_nc:
                nc_path.unlink()
                logger.info(f"Removed {nc_path} to save disk.")

            rows.append(metrics)

    rows.sort(
        key=lambda row: (
            str(row.get("run_label", "")),
            int(row["seed"]),
            float(row["eta"]),
        )
    )
    _write_summary_rows(outdir, rows)
    logger.info(f"Eta sweep complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
