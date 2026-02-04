"""Multivariate PSD simulation study with VARMA data.

Inputs: N (data size), K (number of knots), SEED (random seed to start)
Outputs: Estimated PSDs, coverage probabilities, and performance metrics
"""

import os

import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
jax.config.update("jax_enable_x64", True)

import argparse

import numpy as np

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

set_level("DEBUG")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("out_var3")

SIGMA = np.array([[1.0, 0.10, 0.45], [0.10, 1.0, 0.00], [0.45, 0.00, 1.0]])
# 3x3 identity MA(0) to match VAR dimension
VMA_COEFFS = np.array(
    [
        np.eye(3),
        # MA(1) mixing only between channels 1 and 3
        [[0.00, 0.00, 0.15], [0.00, 0.00, 0.00], [0.12, 0.00, 0.00]],
    ]
)
VAR_COEFFS = np.array(
    [
        # Lag 1 coefficients (A1)
        [
            [0.30, 0.03, 0.12],  # add 1←3 coupling
            [0.02, 0.25, 0.04],
            [0.10, 0.00, 0.94],
        ],  # add 3←1 coupling; ch3 φ1≈0.94
        # Lag 2 coefficients (A2)
        [
            [0.10, 0.01, 0.05],  # small 1←3 lag-2 influence
            [0.00, 0.08, 0.03],
            [0.06, 0.00, -0.64],
        ],  # small 3←1 lag-2 influence
        # Lag 3 coefficients (A3)
        [
            [0.05, 0.00, 0.00],
            [0.00, 0.03, 0.01],
            [0.00, 0.00, 0.00],
        ],  # ch3: φ3 ≈ 0 (AR(2)-like)
    ]
)


def simulation_study(
    outdir: str = OUT,
    N: int = 5024,
    K: int = 15,
    SEED: int = 42,
    *,
    skip_diagnostics: bool = False,
    n_time_blocks: int = 4,
    knot_method: str = "log",
    coarse_n_freqs_per_bin: int | None = 5,
    coarse_f_min: float | None = None,
    coarse_f_max: float | None = None,
    alpha_delta: float = 1.0,
    beta_delta: float = 1.0,
    target_accept_prob: float = 0.9,
    max_tree_depth: int = 12,
    init_from_vi: bool = False,
    vi_steps: int = 5000,
    vi_guide: str | None = "diag",
    posterior_psd_max_draws: int = 200,
    vi_psd_max_draws: int = 64,
    n_samples: int = 1000,
    n_warmup: int = 1000,
    num_chains: int = 4,
):
    print(f">>>> Running simulation with N={N}, K={K}, SEED={SEED} <<<<")
    outdir = f"{HERE}/{outdir}/seed_{SEED}_N{N}_K{K}"
    os.makedirs(outdir, exist_ok=True)

    # Generate VARMA data
    np.random.seed(SEED)
    varma = VARMAData(
        n_samples=N,
        seed=SEED,
        var_coeffs=VAR_COEFFS,
        vma_coeffs=VMA_COEFFS,
        sigma=SIGMA,
    )
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)

    if knot_method not in {"linear", "log"}:
        raise ValueError(
            f"knot_method must be 'linear' or 'log', got {knot_method!r}."
        )

    coarse_grain_config = None
    if coarse_n_freqs_per_bin is not None:
        coarse_n_freqs_per_bin = int(coarse_n_freqs_per_bin)
        if coarse_n_freqs_per_bin <= 0:
            raise ValueError("coarse_n_freqs_per_bin must be positive.")
        if coarse_n_freqs_per_bin % 2 == 0:
            raise ValueError("coarse_n_freqs_per_bin must be odd.")
        coarse_grain_config = dict(
            enabled=True,
            n_bins=None,
            n_freqs_per_bin=coarse_n_freqs_per_bin,
            f_min=coarse_f_min,
            f_max=coarse_f_max,
        )

    run_mcmc(
        data=ts,
        sampler="multivar_blocked_nuts",
        n_knots=K,
        degree=2,
        diffMatrixOrder=2,
        n_samples=n_samples,
        n_warmup=n_warmup,
        num_chains=num_chains,
        outdir=outdir,
        verbose=True,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_from_vi=init_from_vi,
        vi_steps=vi_steps,
        vi_guide=vi_guide,
        vi_psd_max_draws=vi_psd_max_draws,
        posterior_psd_max_draws=posterior_psd_max_draws,
        n_time_blocks=n_time_blocks,
        knot_kwargs=dict(method=knot_method),
        coarse_grain_config=coarse_grain_config,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        compute_psis=False,
        compute_coherence_quantiles=True,
        skip_plot_diagnostics=skip_diagnostics,
        true_psd=varma.get_true_psd(),
        save_preprocessing_plots=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multivariate PSD simulation study with VARMA data"
    )
    parser.add_argument(
        "--N", type=int, default=5024, help="Number of time points"
    )
    parser.add_argument(
        "--K", type=int, default=15, help="Number of spline knots"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-time-blocks",
        type=int,
        default=4,
        help="Number of non-overlapping time blocks used to form Wishart-averaged FFT stats.",
    )
    parser.add_argument(
        "--knot-method",
        type=str,
        default="log",
        choices=["linear", "log"],
        help="Knot placement method for the spline basis.",
    )
    parser.add_argument(
        "--coarse-n-freqs-per-bin",
        type=int,
        default=5,
        help="If set, enable coarse graining with this odd bin size (set to 0 to disable).",
    )
    parser.add_argument(
        "--alpha-delta",
        type=float,
        default=1.0,
        help="Gamma prior concentration for delta hyperparameters.",
    )
    parser.add_argument(
        "--beta-delta",
        type=float,
        default=1.0,
        help="Gamma prior rate for delta hyperparameters.",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="NUTS target acceptance probability.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=12,
        help="NUTS maximum tree depth.",
    )
    parser.add_argument(
        "--init-from-vi",
        action="store_true",
        help="Use VI-based initialisation for each block.",
    )
    parser.add_argument(
        "--vi-steps",
        type=int,
        default=5000,
        help="Number of VI optimisation steps per block (if --init-from-vi).",
    )
    parser.add_argument(
        "--vi-guide",
        type=str,
        default="diag",
        help="VI guide spec passed to the blocked initialiser (if --init-from-vi).",
    )
    parser.add_argument(
        "--posterior-psd-max-draws",
        type=int,
        default=200,
        help="Max posterior draws used to compute PSD quantiles for plots/coverage.",
    )
    parser.add_argument(
        "--vi-psd-max-draws",
        type=int,
        default=64,
        help="Max VI posterior draws used to compute VI PSD quantiles for overlay.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of posterior samples per chain.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        help="Number of warmup samples per chain.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip MCMC diagnostics plots/summaries (still saves main PSD plots).",
    )

    args = parser.parse_args()
    coarse_n_freqs_per_bin = args.coarse_n_freqs_per_bin
    if coarse_n_freqs_per_bin is not None and int(coarse_n_freqs_per_bin) <= 0:
        coarse_n_freqs_per_bin = None
    simulation_study(
        N=args.N,
        K=args.K,
        SEED=args.seed,
        skip_diagnostics=args.skip_diagnostics,
        n_time_blocks=args.n_time_blocks,
        knot_method=args.knot_method,
        coarse_n_freqs_per_bin=coarse_n_freqs_per_bin,
        alpha_delta=args.alpha_delta,
        beta_delta=args.beta_delta,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        init_from_vi=bool(args.init_from_vi),
        vi_steps=args.vi_steps,
        vi_guide=args.vi_guide,
        posterior_psd_max_draws=args.posterior_psd_max_draws,
        vi_psd_max_draws=args.vi_psd_max_draws,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        num_chains=args.chains,
    )
