#!/usr/bin/env python
"""Paired study: VI-initialized NUTS versus cold-start NUTS on VAR(2), 3D.

This script is aimed at the specific question:
    "How much does VI initialization affect the downstream MCMC run?"

For each seed it runs two multivariate blocked-NUTS fits on the same synthetic
VAR(2) 3-channel dataset:
1. ``init_from_vi=True``  ("vi_init")
2. ``init_from_vi=False`` ("cold_start")

It records:
- downstream NUTS runtime (from ``idata.attrs['runtime']``),
- end-to-end wall time,
- ESS / divergence / acceptance diagnostics,
- PSD accuracy versus the known truth,
- posterior shift between the two NUTS runs for the same seed.

Outputs
-------
``comparison_results.csv``
    One row per (seed, mode).
``paired_deltas.csv``
    One row per seed with vi_init - cold_start deltas.
``summary.json``
    Compact aggregate summary across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.diagnostics._utils import (
    compute_matrix_riae,
    interior_frequency_slice,
)
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    NUTSConfigOverride,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)

FS = 1.0
BURN_IN = 512

A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [
        [-0.2, 0.5, 0.0],
        [0.4, -0.1, 0.0],
        [0.0, 0.0, -0.1],
    ],
    dtype=np.float64,
)
VAR_COEFFS = np.array([A1, A2], dtype=np.float64)

SIGMA_VAL = 0.25
OFF_DIAG = 0.08
SIGMA = np.array(
    [
        [SIGMA_VAL, 0.0, OFF_DIAG],
        [0.0, SIGMA_VAL, OFF_DIAG],
        [OFF_DIAG, OFF_DIAG, SIGMA_VAL],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class StudyConfig:
    n: int = 4096
    nb: int = 4
    n_knots: int = 10
    n_samples: int = 250
    n_warmup: int = 250
    num_chains: int = 2
    vi_steps: int = 20_000
    vi_lr: float = 1e-3
    vi_guide: str | None = "lowrank:16"
    target_accept_prob: float = 0.9
    max_tree_depth: int = 12
    wishart_window: str | tuple | None = "hann"
    verbose: bool = False


def _simulate_var(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ar_order, n_ch, _ = VAR_COEFFS.shape
    n_total = int(n_samples) + BURN_IN
    rng = np.random.default_rng(int(seed))
    noise = rng.multivariate_normal(np.zeros(n_ch), SIGMA, size=n_total)
    x = np.zeros((n_total, n_ch), dtype=np.float64)
    for t_idx in range(ar_order, n_total):
        state = noise[t_idx].copy()
        for lag in range(1, ar_order + 1):
            state += VAR_COEFFS[lag - 1] @ x[t_idx - lag]
        x[t_idx] = state
    x = x[BURN_IN:]
    t = np.arange(x.shape[0], dtype=np.float64) / FS
    return t, x


def _true_psd(freqs_hz: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / FS
    n_ch = VAR_COEFFS.shape[1]
    ident = np.eye(n_ch, dtype=np.complex128)
    psd = np.empty((omega.size, n_ch, n_ch), dtype=np.complex128)
    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, VAR_COEFFS.shape[0] + 1):
            a_f -= VAR_COEFFS[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        psd[idx] = (2.0 / FS) * h_f @ SIGMA @ h_f.conj().T
    return 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))


def _safe_float(attrs: dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = float(attrs.get(key))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def _posterior_q50_complex(idata) -> tuple[np.ndarray, np.ndarray]:
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is None or "psd_matrix_real" not in psd_group:
        raise ValueError("posterior_psd.psd_matrix_real is required.")

    psd_real = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    percentiles = np.asarray(
        psd_group["psd_matrix_real"].coords.get(
            "percentile", np.arange(psd_real.shape[0], dtype=float)
        ),
        dtype=np.float64,
    )
    q50_idx = int(np.argmin(np.abs(percentiles - 50.0)))
    q50_real = psd_real[q50_idx]

    if "psd_matrix_imag" in psd_group:
        psd_imag = np.asarray(
            psd_group["psd_matrix_imag"].values, dtype=np.float64
        )
        q50_imag = psd_imag[q50_idx]
    else:
        q50_imag = np.zeros_like(q50_real)

    freqs = np.asarray(
        psd_group["psd_matrix_real"].coords.get("freq"), dtype=np.float64
    )
    freq_idx = interior_frequency_slice(freqs.size)
    freqs = freqs[freq_idx]
    q50 = q50_real[freq_idx] + 1j * q50_imag[freq_idx]
    return freqs, q50


def _posterior_shift_metrics(idata_a, idata_b) -> dict[str, float]:
    freqs_a, q50_a = _posterior_q50_complex(idata_a)
    freqs_b, q50_b = _posterior_q50_complex(idata_b)
    if freqs_a.shape != freqs_b.shape or not np.allclose(freqs_a, freqs_b):
        raise ValueError("Posterior PSD frequency grids do not match.")

    diff = q50_a - q50_b
    diag_idx = np.arange(q50_a.shape[1])
    offdiag_mask = ~np.eye(q50_a.shape[1], dtype=bool)

    diag_abs = np.abs(diff[:, diag_idx, diag_idx])
    offdiag_abs = np.abs(diff[:, offdiag_mask])

    return {
        "posterior_shift_riae_matrix": float(
            compute_matrix_riae(q50_a, q50_b, freqs_a)
        ),
        "posterior_shift_diag_abs_mean": float(np.mean(diag_abs)),
        "posterior_shift_diag_abs_max": float(np.max(diag_abs)),
        "posterior_shift_offdiag_abs_mean": float(np.mean(offdiag_abs)),
        "posterior_shift_offdiag_abs_max": float(np.max(offdiag_abs)),
    }


def _build_run_config(
    *,
    cfg: StudyConfig,
    freqs: np.ndarray,
    true_psd: np.ndarray,
    rng_key: int,
    init_from_vi: bool,
    outdir: str | None,
) -> RunMCMCConfig:
    return RunMCMCConfig(
        n_samples=cfg.n_samples,
        n_warmup=cfg.n_warmup,
        num_chains=cfg.num_chains,
        rng_key=int(rng_key),
        Nb=cfg.nb,
        wishart_window=cfg.wishart_window,
        model=ModelConfig(
            n_knots=cfg.n_knots,
            degree=2,
            diffMatrixOrder=2,
            true_psd=(freqs, true_psd),
        ),
        diagnostics=DiagnosticsConfig(
            verbose=cfg.verbose,
            outdir=outdir,
            compute_lnz=False,
        ),
        vi=VIConfig(
            only_vi=False,
            init_from_vi=init_from_vi,
            vi_steps=cfg.vi_steps,
            vi_lr=cfg.vi_lr,
            vi_guide=cfg.vi_guide,
        ),
        nuts=NUTSConfigOverride(
            target_accept_prob=cfg.target_accept_prob,
            max_tree_depth=cfg.max_tree_depth,
            dense_mass=True,
        ),
        extra_kwargs={"compute_coherence_quantiles": True},
    )


def _run_one(
    *,
    seed: int,
    cfg: StudyConfig,
    mode: str,
    init_from_vi: bool,
    base_outdir: Path,
) -> tuple[dict[str, Any], Any]:
    t, y = _simulate_var(cfg.n, seed)
    ts = MultivariateTimeseries(t=t, y=y)
    freqs = np.fft.rfftfreq(cfg.n, d=1.0 / FS)[1:]
    true_psd = _true_psd(freqs)

    run_dir = base_outdir / f"seed_{seed}" / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _build_run_config(
        cfg=cfg,
        freqs=freqs,
        true_psd=true_psd,
        rng_key=10_000 + int(seed),
        init_from_vi=init_from_vi,
        outdir=None,
    )

    start = time.perf_counter()
    idata = run_mcmc(data=ts, config=run_cfg)
    wall_seconds = time.perf_counter() - start

    attrs = getattr(idata, "attrs", {})
    row: dict[str, Any] = {
        "seed": int(seed),
        "mode": mode,
        "init_from_vi": int(init_from_vi),
        "n": int(cfg.n),
        "nb": int(cfg.nb),
        "n_knots": int(cfg.n_knots),
        "n_samples": int(cfg.n_samples),
        "n_warmup": int(cfg.n_warmup),
        "num_chains": int(cfg.num_chains),
        "vi_steps": int(cfg.vi_steps),
        "wall_seconds_total": float(wall_seconds),
        "nuts_runtime_seconds": _safe_float(attrs, "runtime"),
        "riae_matrix_vs_truth": _safe_float(attrs, "riae_matrix", "riae"),
        "coverage_vs_truth": _safe_float(attrs, "coverage"),
        "ci_width_vs_truth": _safe_float(
            attrs, "ci_width", "ci_width_diag_mean"
        ),
        "ess_bulk_min": _safe_float(attrs, "mcmc_ess_bulk_min"),
        "ess_bulk_median": _safe_float(attrs, "mcmc_ess_bulk_median"),
        "ess_tail_min": _safe_float(attrs, "mcmc_ess_tail_min"),
        "ess_tail_median": _safe_float(attrs, "mcmc_ess_tail_median"),
        "rhat_max": _safe_float(attrs, "mcmc_rhat_max"),
        "rhat_mean": _safe_float(attrs, "mcmc_rhat_mean"),
        "divergence_fraction": _safe_float(attrs, "mcmc_divergence_fraction"),
        "divergence_total": _safe_float(attrs, "mcmc_divergence_total"),
        "acceptance_rate_mean": _safe_float(
            attrs, "mcmc_acceptance_rate_mean"
        ),
        "psis_khat_max": _safe_float(attrs, "mcmc_psis_khat_max"),
        "vi_variance_ratio_vs_mcmc": _safe_float(
            attrs, "vi_variance_ratio_vs_mcmc"
        ),
        "vi_psis_khat_max": _safe_float(attrs, "vi_psis_khat_max"),
        "vi_moment_bias_pct": _safe_float(attrs, "vi_moment_bias_pct"),
    }

    with open(run_dir / "run_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)

    return row, idata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_delta_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "n": int(finite.size),
        "n_negative": int(np.sum(finite < 0.0)),
        "n_positive": int(np.sum(finite > 0.0)),
        "n_zero": int(np.sum(finite == 0.0)),
    }


def _aggregate_summary(
    rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    cfg: StudyConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "study_config": asdict(cfg),
        "n_runs": len(rows),
        "n_pairs": len(paired_rows),
        "by_mode": {},
        "paired_deltas": {},
    }

    for mode in ("vi_init", "cold_start"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        mode_summary: dict[str, Any] = {"n": len(mode_rows)}
        for metric in (
            "wall_seconds_total",
            "nuts_runtime_seconds",
            "riae_matrix_vs_truth",
            "coverage_vs_truth",
            "ci_width_vs_truth",
            "ess_bulk_min",
            "ess_bulk_median",
            "divergence_fraction",
            "acceptance_rate_mean",
            "rhat_max",
        ):
            vals = np.asarray([row.get(metric, np.nan) for row in mode_rows])
            finite = vals[np.isfinite(vals)]
            if finite.size:
                mode_summary[metric] = {
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                    "std": float(np.std(finite)),
                }
        summary["by_mode"][mode] = mode_summary

    for metric in (
        "delta_wall_seconds_total",
        "delta_nuts_runtime_seconds",
        "delta_riae_matrix_vs_truth",
        "delta_coverage_vs_truth",
        "delta_ci_width_vs_truth",
        "delta_ess_bulk_min",
        "delta_ess_bulk_median",
        "delta_divergence_fraction",
        "delta_divergence_total",
        "delta_acceptance_rate_mean",
        "delta_rhat_max",
        "posterior_shift_riae_matrix",
        "posterior_shift_diag_abs_mean",
        "posterior_shift_offdiag_abs_mean",
    ):
        vals = np.asarray([row.get(metric, np.nan) for row in paired_rows])
        stats = _paired_delta_summary(vals)
        if stats:
            summary["paired_deltas"][metric] = stats

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("VI-init versus cold-start NUTS")
    print("=" * 72)
    print(f"Pairs: {summary.get('n_pairs', 0)}")

    paired = summary.get("paired_deltas", {})
    print("\nPaired deltas (vi_init - cold_start):")
    for metric in (
        "delta_nuts_runtime_seconds",
        "delta_wall_seconds_total",
        "delta_ess_bulk_min",
        "delta_divergence_fraction",
        "delta_riae_matrix_vs_truth",
        "posterior_shift_riae_matrix",
    ):
        stats = paired.get(metric)
        if not stats:
            continue
        print(
            f"  {metric:28s} mean={stats['mean']:.4g}  "
            f"median={stats['median']:.4g}  std={stats['std']:.4g}  n={stats['n']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run one seed only. If omitted, run a range of seeds.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of seeds to run when --seed is not provided.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out_vi_init_nuts_comparison",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--n", type=int, default=4096, help="Time-series length."
    )
    parser.add_argument(
        "--nb", type=int, default=4, help="Wishart block count."
    )
    parser.add_argument(
        "--n-knots", type=int, default=10, help="Number of spline knots."
    )
    parser.add_argument(
        "--n-samples", type=int, default=250, help="Posterior draws per chain."
    )
    parser.add_argument(
        "--n-warmup", type=int, default=250, help="Warmup draws per chain."
    )
    parser.add_argument(
        "--num-chains", type=int, default=2, help="Number of chains."
    )
    parser.add_argument(
        "--vi-steps", type=int, default=20_000, help="VI optimization steps."
    )
    parser.add_argument(
        "--vi-lr", type=float, default=1e-3, help="VI learning rate."
    )
    parser.add_argument(
        "--vi-guide",
        type=str,
        default="lowrank:16",
        help="VI guide specifier.",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=0.9,
        help="NUTS target acceptance probability.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=12,
        help="NUTS max tree depth.",
    )
    parser.add_argument(
        "--wishart-window",
        type=str,
        default="hann",
        help="Window applied before the Wishart FFT.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose run_mcmc logging.",
    )
    args = parser.parse_args()

    cfg = StudyConfig(
        n=int(args.n),
        nb=int(args.nb),
        n_knots=int(args.n_knots),
        n_samples=int(args.n_samples),
        n_warmup=int(args.n_warmup),
        num_chains=int(args.num_chains),
        vi_steps=int(args.vi_steps),
        vi_lr=float(args.vi_lr),
        vi_guide=None if args.vi_guide.lower() == "none" else args.vi_guide,
        target_accept_prob=float(args.target_accept_prob),
        max_tree_depth=int(args.max_tree_depth),
        wishart_window=(
            None
            if args.wishart_window.lower() == "none"
            else args.wishart_window
        ),
        verbose=bool(args.verbose),
    )

    seeds = (
        [int(args.seed)]
        if args.seed is not None
        else list(range(args.n_seeds))
    )
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        vi_row, vi_idata = _run_one(
            seed=seed,
            cfg=cfg,
            mode="vi_init",
            init_from_vi=True,
            base_outdir=outdir,
        )
        cold_row, cold_idata = _run_one(
            seed=seed,
            cfg=cfg,
            mode="cold_start",
            init_from_vi=False,
            base_outdir=outdir,
        )
        rows.extend([vi_row, cold_row])

        paired = {
            "seed": int(seed),
            "delta_wall_seconds_total": float(
                vi_row["wall_seconds_total"] - cold_row["wall_seconds_total"]
            ),
            "delta_nuts_runtime_seconds": float(
                vi_row["nuts_runtime_seconds"]
                - cold_row["nuts_runtime_seconds"]
            ),
            "delta_riae_matrix_vs_truth": float(
                vi_row["riae_matrix_vs_truth"]
                - cold_row["riae_matrix_vs_truth"]
            ),
            "delta_coverage_vs_truth": float(
                vi_row["coverage_vs_truth"] - cold_row["coverage_vs_truth"]
            ),
            "delta_ci_width_vs_truth": float(
                vi_row["ci_width_vs_truth"] - cold_row["ci_width_vs_truth"]
            ),
            "delta_ess_bulk_min": float(
                vi_row["ess_bulk_min"] - cold_row["ess_bulk_min"]
            ),
            "delta_ess_bulk_median": float(
                vi_row["ess_bulk_median"] - cold_row["ess_bulk_median"]
            ),
            "delta_divergence_fraction": float(
                vi_row["divergence_fraction"] - cold_row["divergence_fraction"]
            ),
            "delta_divergence_total": float(
                vi_row["divergence_total"] - cold_row["divergence_total"]
            ),
            "delta_acceptance_rate_mean": float(
                vi_row["acceptance_rate_mean"]
                - cold_row["acceptance_rate_mean"]
            ),
            "delta_rhat_max": float(vi_row["rhat_max"] - cold_row["rhat_max"]),
        }
        paired.update(_posterior_shift_metrics(vi_idata, cold_idata))
        paired_rows.append(paired)

        print(
            "  vi_init:    "
            f"NUTS={vi_row['nuts_runtime_seconds']:.2f}s  "
            f"wall={vi_row['wall_seconds_total']:.2f}s  "
            f"ESSmin={vi_row['ess_bulk_min']:.1f}  "
            f"div={vi_row['divergence_fraction']:.4f}"
        )
        print(
            "  cold_start: "
            f"NUTS={cold_row['nuts_runtime_seconds']:.2f}s  "
            f"wall={cold_row['wall_seconds_total']:.2f}s  "
            f"ESSmin={cold_row['ess_bulk_min']:.1f}  "
            f"div={cold_row['divergence_fraction']:.4f}"
        )
        print(
            "  shift:      "
            f"posterior_riae={paired['posterior_shift_riae_matrix']:.4f}  "
            f"delta_NUTS={paired['delta_nuts_runtime_seconds']:.2f}s"
        )

    _write_csv(outdir / "comparison_results.csv", rows)
    _write_csv(outdir / "paired_deltas.csv", paired_rows)

    summary = _aggregate_summary(rows, paired_rows, cfg)
    with open(outdir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    _print_summary(summary)
    print(f"\nWrote results to {outdir}")


if __name__ == "__main__":
    main()
