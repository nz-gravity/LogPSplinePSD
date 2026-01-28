"""Multivariate PSD simulation study with VARMA data

Inputs: N (data size), K (number of knots), SEED (random seed to start)
Outputs: Estimated PSDs, coverage probabilities, and performance metrics
"""

import os

import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
jax.config.update("jax_enable_x64", True)

import argparse
import os

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
    run_mcmc(
        data=ts,
        sampler="multivar_blocked_nuts",
        n_knots=K,
        degree=2,
        diffMatrixOrder=2,
        n_samples=1000,
        n_warmup=1000,
        num_chains=4,
        outdir=outdir,
        verbose=True,
        target_accept_prob=0.8,
        vi_psd_max_draws=16,
        vi_steps=20000,
        posterior_psd_max_draws=20,
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
        "--K", type=int, default=7, help="Number of spline knots"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip MCMC diagnostics plots/summaries (still saves main PSD plots).",
    )

    args = parser.parse_args()
    simulation_study(
        N=args.N,
        K=args.K,
        SEED=args.seed,
        skip_diagnostics=args.skip_diagnostics,
    )
