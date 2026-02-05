"""VAR(3) paper job runner.

This script is a small CLI wrapper around ``log_psplines.mcmc.run_mcmc`` that
generates multivariate VAR(3) time series and runs the multivariate blocked NUTS
pipeline (optionally with frequency coarse graining).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import numpy as np

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
jax.config.update("jax_enable_x64", True)

set_level("INFO")


SIGMA = np.array([[1.0, 0.10, 0.45], [0.10, 1.0, 0.00], [0.45, 0.00, 1.0]])
VMA_COEFFS = np.array(
    [
        np.eye(3),
        [[0.00, 0.00, 0.15], [0.00, 0.00, 0.00], [0.12, 0.00, 0.00]],
    ]
)
VAR_COEFFS = np.array(
    [
        [
            [0.30, 0.03, 0.12],
            [0.02, 0.25, 0.04],
            [0.10, 0.00, 0.94],
        ],
        [
            [0.10, 0.01, 0.05],
            [0.00, 0.08, 0.03],
            [0.06, 0.00, -0.64],
        ],
        [
            [0.05, 0.00, 0.00],
            [0.00, 0.03, 0.01],
            [0.00, 0.00, 0.00],
        ],
    ]
)


def _resolve_blocks(n_time_target: int, block_size: int) -> tuple[int, int]:
    n_time_target = int(n_time_target)
    block_size = int(block_size)
    if n_time_target <= 0:
        raise ValueError("--n-time must be positive.")
    if block_size <= 0:
        raise ValueError("--block-size must be positive.")

    Nb = max(1, n_time_target // block_size)
    n_used = Nb * block_size
    return Nb, n_used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--n-time", type=int, required=True, help="Target N.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=5000,
        help="Samples per Wishart block (used to pick n_time_blocks).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knots", type=int, default=15)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--diff-order", type=int, default=2)

    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)

    parser.add_argument("--alpha-delta", type=float, default=1.0)
    parser.add_argument("--beta-delta", type=float, default=1.0)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=12)

    parser.add_argument(
        "--init-from-vi", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--vi-steps", type=int, default=20_000)
    parser.add_argument("--vi-guide", type=str, default="lowrank:16")

    parser.add_argument(
        "--coarse-bins",
        type=int,
        default=0,
        help="Enable coarse graining with Nc=coarse_bins (0 disables).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    idata_path = outdir / "inference_data.nc"
    if idata_path.exists() and not args.overwrite:
        logger.info(
            f"Found {idata_path}; skipping (pass --overwrite to rerun)."
        )
        return

    Nb, n_used = _resolve_blocks(args.n, args.block_size)
    if n_used != int(args.n):
        logger.info(
            f"Trimming target N={int(args.n)} -> N_used={n_used} to fit {Nb} blocks of {int(args.block_size)}."
        )

    varma = VARMAData(
        n_samples=n_used,
        seed=int(args.seed),
        var_coeffs=VAR_COEFFS,
        vma_coeffs=VMA_COEFFS,
        sigma=SIGMA,
    )
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)

    coarse_cfg: CoarseGrainConfig | None = None
    coarse_bins = int(args.coarse_bins)
    if coarse_bins > 0:
        coarse_cfg = CoarseGrainConfig(enabled=True, Nc=coarse_bins)

    logger.info(
        f"Running VAR(3) job: N={n_used}, blocks={Nb}, coarse_bins={coarse_bins or 'off'}, outdir={outdir}"
    )

    run_mcmc(
        data=ts,
        sampler="multivar_blocked_nuts",
        n_knots=int(args.knots),
        degree=int(args.degree),
        diffMatrixOrder=int(args.diff_order),
        n_samples=int(args.samples),
        n_warmup=int(args.warmup),
        num_chains=int(args.chains),
        rng_key=int(args.seed),
        outdir=str(outdir),
        verbose=True,
        compute_psis=False,
        skip_plot_diagnostics=False,
        target_accept_prob=float(args.target_accept),
        max_tree_depth=int(args.max_tree_depth),
        init_from_vi=bool(args.init_from_vi),
        vi_steps=int(args.vi_steps),
        vi_guide=str(args.vi_guide),
        vi_psd_max_draws=64,
        n_time_blocks=int(Nb),
        knot_kwargs=dict(method="log"),
        coarse_grain_config=coarse_cfg,
        alpha_delta=float(args.alpha_delta),
        beta_delta=float(args.beta_delta),
        compute_coherence_quantiles=True,
        save_preprocessing_plots=True,
        true_psd=varma.get_true_psd(),
    )


if __name__ == "__main__":
    main()
