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
from log_psplines.logger import set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

set_level("DEBUG")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("out_var3")

DEFAULT_KNOT_METHOD = "log"
DEFAULT_N_TIME_BLOCKS = 4
DEFAULT_TARGET_ACCEPT_PROB = 0.9
DEFAULT_MAX_TREE_DEPTH = 12
DEFAULT_INIT_FROM_VI = True
DEFAULT_VI_STEPS = 20000
DEFAULT_VI_GUIDE = "lowrank:16"
DEFAULT_VI_PSD_MAX_DRAWS = 64
DEFAULT_POSTERIOR_PSD_MAX_DRAWS = 200
DEFAULT_ALPHA_DELTA = 1.0
DEFAULT_BETA_DELTA = 1.0
DEFAULT_N_SAMPLES = 1000
DEFAULT_N_WARMUP = 1000
DEFAULT_NUM_CHAINS = 4

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
    coarse_n_freqs_per_bin: int | None = 5,
    coarse_f_min: float | None = None,
    coarse_f_max: float | None = None,
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

    coarse_grain_config = None
    knot_method = DEFAULT_KNOT_METHOD
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
        n_samples=DEFAULT_N_SAMPLES,
        n_warmup=DEFAULT_N_WARMUP,
        num_chains=DEFAULT_NUM_CHAINS,
        outdir=outdir,
        verbose=True,
        target_accept_prob=DEFAULT_TARGET_ACCEPT_PROB,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        init_from_vi=DEFAULT_INIT_FROM_VI,
        vi_steps=DEFAULT_VI_STEPS,
        vi_guide=DEFAULT_VI_GUIDE,
        vi_psd_max_draws=DEFAULT_VI_PSD_MAX_DRAWS,
        posterior_psd_max_draws=DEFAULT_POSTERIOR_PSD_MAX_DRAWS,
        n_time_blocks=DEFAULT_N_TIME_BLOCKS,
        knot_kwargs=dict(method=knot_method),
        coarse_grain_config=coarse_grain_config,
        alpha_delta=DEFAULT_ALPHA_DELTA,
        beta_delta=DEFAULT_BETA_DELTA,
        compute_psis=False,
        compute_coherence_quantiles=True,
        skip_plot_diagnostics=False,
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--coarse-n-freqs-per-bin",
        type=int,
        default=5,
        help="If set, enable coarse graining with this odd bin size (set to 0 to disable).",
    )

    args = parser.parse_args()
    coarse_n_freqs_per_bin = args.coarse_n_freqs_per_bin
    if coarse_n_freqs_per_bin is not None and coarse_n_freqs_per_bin <= 0:
        coarse_n_freqs_per_bin = None
    simulation_study(
        N=args.N,
        SEED=args.seed,
        coarse_n_freqs_per_bin=coarse_n_freqs_per_bin,
    )
