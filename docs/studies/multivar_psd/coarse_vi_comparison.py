#!/usr/bin/env python
"""Compare three VI strategies (no NUTS).

Runs the 3-channel VAR(2) simulation 10 times under three conditions:
  A) coarse_only:  VI-coarse → transfer weights (skip fine VI)
  B) coarse+fine:  VI-coarse → transfer → VI-fine
  C) fine_only:    VI-fine on full grid

Reports RIAE, coverage, and wall time for each seed.

Usage:
    python coarse_vi_comparison.py          # run all 10 seeds
    python coarse_vi_comparison.py --seed 0 # run a single seed (all modes)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)

# ─── VAR(2) 3-channel setup (same as 3d_study.py) ────────────────────────────

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

# ─── Study parameters ────────────────────────────────────────────────────────

NB = 4
N_SEEDS = 10
K = 10

VI_STEPS = 20_000
VI_LR = 1e-3


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _simulate_var(n_samples: int, seed: int):
    ar_order, n_ch, _ = VAR_COEFFS.shape
    n_total = n_samples + BURN_IN
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(np.zeros(n_ch), SIGMA, size=n_total)
    x = np.zeros((n_total, n_ch), dtype=np.float64)
    for t in range(ar_order, n_total):
        state = noise[t].copy()
        for lag in range(1, ar_order + 1):
            state += VAR_COEFFS[lag - 1] @ x[t - lag]
        x[t] = state
    x = x[BURN_IN:]
    t = np.arange(x.shape[0], dtype=np.float64) / FS
    return t, x


def _true_psd(freqs_hz: np.ndarray):
    omega = 2.0 * np.pi * freqs_hz / FS
    n_ch = VAR_COEFFS.shape[1]
    ident = np.eye(n_ch, dtype=np.complex128)
    psd = np.empty((freqs_hz.size, n_ch, n_ch), dtype=np.complex128)
    for i, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, VAR_COEFFS.shape[0] + 1):
            a_f -= VAR_COEFFS[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        psd[i] = (2.0 / FS) * h_f @ SIGMA @ h_f.conj().T
    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    return psd


MODES = [
    # (label, auto_coarse_vi, vi_coarse_only)
    ("coarse_only", True, True),
    ("coarse+fine", True, False),
    ("fine_only", False, False),
]


def _run_one(seed: int, mode: tuple, outdir: str, *, n: int) -> dict:
    label, auto_coarse, coarse_only = mode

    t, data = _simulate_var(n, seed)
    ts = MultivariateTimeseries(t=t, y=data)

    freqs = np.fft.rfftfreq(n, d=1.0 / FS)[1:]
    true = _true_psd(freqs)

    run_dir = os.path.join(outdir, f"seed_{seed}", label)

    cfg = RunMCMCConfig(
        n_samples=4,
        n_warmup=4,
        num_chains=1,
        Nb=NB,
        model=ModelConfig(
            n_knots=K,
            degree=2,
            diffMatrixOrder=2,
            true_psd=(freqs, true),
        ),
        diagnostics=DiagnosticsConfig(verbose=True, outdir=run_dir),
        vi=VIConfig(
            only_vi=True,
            vi_steps=VI_STEPS,
            vi_lr=VI_LR,
            auto_coarse_vi=auto_coarse,
            vi_coarse_only=coarse_only,
        ),
        extra_kwargs={"compute_coherence_quantiles": True},
    )

    t0 = time.perf_counter()
    idata = run_mcmc(data=ts, config=cfg)
    wall = time.perf_counter() - t0

    attrs = idata.attrs
    vi_psd_attrs = {}
    if hasattr(idata, "vi_psd"):
        vi_psd_attrs = idata.vi_psd.attrs

    return {
        "seed": seed,
        "mode": label,
        "auto_coarse": int(auto_coarse),
        "coarse_only": int(coarse_only),
        "riae": float(
            attrs.get(
                "riae_matrix",
                vi_psd_attrs.get("riae_matrix", attrs.get("riae", np.nan)),
            )
        ),
        "coverage": float(
            attrs.get("coverage", vi_psd_attrs.get("coverage", np.nan))
        ),
        "wall_time": round(wall, 2),
        "coarse_vi_attempted": int(attrs.get("coarse_vi_attempted", 0)),
        "coarse_vi_success": int(attrs.get("coarse_vi_success", 0)),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed (all modes). If omitted, run all 10.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        nargs="+",
        help="Dataset size(s) to sweep (default: 4096 16384 65536).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out_coarse_vi_comparison",
        help="Output directory.",
    )
    args = parser.parse_args()

    seeds = list(range(N_SEEDS)) if args.seed is None else [args.seed]
    n_values = args.N if args.N else [4096, 16384, 65536]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    results = []
    for n in n_values:
        for seed in seeds:
            for mode in MODES:
                label = mode[0]
                print(f"\n{'='*60}")
                print(f"  N={n} | Seed {seed} | mode={label}")
                print(f"{'='*60}\n")
                try:
                    row = _run_one(seed, mode, outdir, n=n)
                    row["N"] = n
                    results.append(row)
                    print(
                        f"\n  -> RIAE={row['riae']:.4f}  coverage={row['coverage']:.4f}  "
                        f"wall={row['wall_time']}s"
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    results.append(
                        {
                            "N": n,
                            "seed": seed,
                            "mode": label,
                            "auto_coarse": int(mode[1]),
                            "coarse_only": int(mode[2]),
                            "riae": np.nan,
                            "coverage": np.nan,
                            "wall_time": np.nan,
                            "coarse_vi_attempted": 0,
                            "coarse_vi_success": 0,
                        }
                    )

    # ─── Summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = os.path.join(outdir, "comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for n in sorted(df["N"].unique()):
        print(f"\n  N={n}")
        for mode in ["coarse_only", "coarse+fine", "fine_only"]:
            sub = df[(df["N"] == n) & (df["mode"] == mode)]
            if sub.empty:
                continue
            parts = []
            for col in ["riae", "coverage", "wall_time"]:
                vals = sub[col].dropna()
                if vals.empty:
                    continue
                parts.append(f"{col}={vals.mean():.4f}±{vals.std():.4f}")
            print(f"    {mode:12s}:  {' | '.join(parts)}")
    print()


if __name__ == "__main__":
    main()
