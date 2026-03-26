"""VI-only smoke run comparing baseline vs transfer-null excision for LISA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from utils.data import generate_lisa_data
from utils.inference import FMAX, FMIN, run_lisa_mcmc
from utils.preprocessing import (
    build_transfer_null_exclusion_bands,
    compute_Nl_analysis,
    setup_coarse_grain,
)

from log_psplines.mcmc import ModelConfig, RunMCMCConfig
from log_psplines.preprocessing.preprocessing import _preprocess_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="docs/studies/lisa/smoke_out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-days", type=float, default=8.0)
    parser.add_argument("--block-days", type=float, default=2.0)
    parser.add_argument("--target-nc", type=int, default=256)
    parser.add_argument("--knots", type=int, default=12)
    parser.add_argument("--diff-order", type=int, default=2)
    parser.add_argument("--vi-steps", type=int, default=10_000)
    parser.add_argument("--vi-guide", default="diag")
    parser.add_argument("--vi-posterior-draws", type=int, default=128)
    parser.add_argument("--bins-per-side", type=int, default=1)
    parser.add_argument("--half-width", type=float, default=None)
    parser.add_argument("--wishart-window", default="hann")
    parser.add_argument("--wishart-floor-fraction", type=float, default=1e-6)
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ts, freq_true, s_true, nb, lb, dt = generate_lisa_data(
        seed=int(args.seed),
        duration_days=float(args.duration_days),
        block_days=float(args.block_days),
    )
    coarse_cfg = setup_coarse_grain(
        compute_Nl_analysis(lb, dt),
        int(args.target_nc),
    )

    preproc_cfg = RunMCMCConfig(
        Nb=nb,
        coarse_grain_config=coarse_cfg,
        wishart_window=args.wishart_window,
        wishart_floor_fraction=float(args.wishart_floor_fraction),
        model=ModelConfig(
            n_knots=int(args.knots),
            degree=2,
            diffMatrixOrder=int(args.diff_order),
            fmin=FMIN,
            fmax=FMAX,
        ),
    )
    preproc = _preprocess_data(ts, config=preproc_cfg)
    retained_freq = np.asarray(preproc.processed_data.freq, dtype=np.float64)
    exclude_freq_bands = build_transfer_null_exclusion_bands(
        retained_freq,
        bins_per_side=int(args.bins_per_side),
        half_width=args.half_width,
        fmin=FMIN,
        fmax=FMAX,
    )

    baseline_dir = outdir / "baseline"
    excised_dir = outdir / "excised"

    run_lisa_mcmc(
        ts,
        Nb=nb,
        coarse_cfg=coarse_cfg,
        freq_true=freq_true,
        S_true=s_true,
        K=int(args.knots),
        diff_order=int(args.diff_order),
        n_samples=32,
        n_warmup=0,
        num_chains=1,
        vi=True,
        only_vi=True,
        vi_steps=int(args.vi_steps),
        vi_guide=str(args.vi_guide),
        vi_posterior_draws=int(args.vi_posterior_draws),
        wishart_window=args.wishart_window,
        wishart_floor_fraction=float(args.wishart_floor_fraction),
        exclude_freq_bands=(),
        outdir=str(baseline_dir),
    )

    run_lisa_mcmc(
        ts,
        Nb=nb,
        coarse_cfg=coarse_cfg,
        freq_true=freq_true,
        S_true=s_true,
        K=int(args.knots),
        diff_order=int(args.diff_order),
        n_samples=32,
        n_warmup=0,
        num_chains=1,
        vi=True,
        only_vi=True,
        vi_steps=int(args.vi_steps),
        vi_guide=str(args.vi_guide),
        vi_posterior_draws=int(args.vi_posterior_draws),
        wishart_window=args.wishart_window,
        wishart_floor_fraction=float(args.wishart_floor_fraction),
        exclude_freq_bands=exclude_freq_bands,
        outdir=str(excised_dir),
    )

    print(f"Baseline output: {baseline_dir}")
    print(f"Excised output: {excised_dir}")
    print(f"Excluded bands ({len(exclude_freq_bands)}): {exclude_freq_bands}")


if __name__ == "__main__":
    main()
