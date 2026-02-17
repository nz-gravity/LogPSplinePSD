"""Multivariate PSD simulation study with VARMA data

Inputs: N (data size), K (number of knots), SEED (random seed to start)
Outputs: Estimated PSDs, coverage probabilities, and performance metrics
"""

import argparse
import os

import numpy as np

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

HERE = os.path.dirname(os.path.abspath(__file__))


def simulation_study(outdir: str = "out", N=1024, K=7, SEED=42):
    print(f">>>> Running simulation with N={N}, K={K}, SEED={SEED} <<<<")
    outdir = f"{HERE}/{outdir}/seed_{SEED}_N{N}_K{K}"
    os.makedirs(outdir, exist_ok=True)

    # Generate VARMA data
    np.random.seed(SEED)
    varma = VARMAData(n_samples=N, seed=SEED)
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)
    run_mcmc(
        data=ts,
        n_knots=K,
        degree=3,
        diffMatrixOrder=2,
        n_samples=1000,
        n_warmup=1000,
        outdir=outdir,
        verbose=True,
        target_accept_prob=0.8,
        true_psd=varma.get_true_psd(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multivariate PSD simulation study with VARMA data"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--N", type=int, default=1024, help="Number of time points"
    )
    parser.add_argument(
        "--K", type=int, default=7, help="Number of spline knots"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    simulation_study(args.outdir, N=args.N, K=args.K, SEED=args.seed)
