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

DEFAULT_KNOT_METHOD = "log"
DEFAULT_N_TIME_BLOCKS = 4
DEFAULT_TARGET_ACCEPT_PROB = 0.9
DEFAULT_MAX_TREE_DEPTH = 12
DEFAULT_INIT_FROM_VI = True
DEFAULT_VI_STEPS = 100_000
DEFAULT_VI_GUIDE = "flow:2"
VI_LR = 5e-4
DEFAULT_VI_PSD_MAX_DRAWS = 256
DEFAULT_POSTERIOR_PSD_MAX_DRAWS = 256
DEFAULT_ALPHA_DELTA = 1.0
DEFAULT_BETA_DELTA = 1.0
DEFAULT_N_SAMPLES = 1000
DEFAULT_N_WARMUP = 3000
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


def _assert_valid_var3_dataset(varma: VARMAData) -> None:
    """Fail fast if the generated VAR3 dataset is invalid/non-stationary."""
    if varma.var_companion_spectral_radius is None:
        raise RuntimeError(
            "VAR3 dataset check failed: companion spectral radius was not computed."
        )

    spectral_radius = float(varma.var_companion_spectral_radius)
    is_stationary = bool(varma.is_var_stationary)
    is_valid = bool(varma.is_valid_var_dataset)

    logger.info(
        f"VAR3 dataset check: valid={is_valid}, stationary={is_stationary}, "
        f"companion spectral radius={spectral_radius:.6f}"
    )
    if spectral_radius > 0.98:
        logger.warning(
            f"VAR3 AR dynamics are close to a unit root "
            f"(spectral radius={spectral_radius:.6f}); "
            "long-memory effects may be pronounced."
        )

    if not is_valid:
        raise ValueError(
            f"Invalid VAR3 dataset: stationary={is_stationary}, "
            f"spectral_radius={spectral_radius:.6f}. "
            "Adjust VAR coefficients to enforce stability (< 1)."
        )


def simulation_study(
    outdir: str = OUT,
    N: int = 5024,
    K: int = 10,
    SEED: int = 42,
    *,
    coarse_Nh: int | None = 4,
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
    _assert_valid_var3_dataset(varma)
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)

    coarse_grain_config = None
    knot_method = DEFAULT_KNOT_METHOD
    if coarse_Nh is not None:
        coarse_Nh = int(coarse_Nh)
        if coarse_Nh <= 0:
            raise ValueError("coarse_Nh must be positive.")
        coarse_grain_config = dict(
            enabled=True,
            Nc=None,
            Nh=coarse_Nh,
        )

    run_mcmc(
        data=ts,
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
        vi_lr=VI_LR,
        Nb=DEFAULT_N_TIME_BLOCKS,
        knot_kwargs=dict(method=knot_method),
        coarse_grain_config=coarse_grain_config,
        fmin=coarse_f_min,
        fmax=coarse_f_max,
        alpha_delta=DEFAULT_ALPHA_DELTA,
        beta_delta=DEFAULT_BETA_DELTA,
        compute_coherence_quantiles=True,
        true_psd=varma.get_true_psd(),
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
        "--coarse-Nh",
        type=int,
        default=4,
        help="If set, enable coarse graining with this bin size (set to 0 to disable).",
    )

    args = parser.parse_args()
    coarse_Nh = args.coarse_Nh
    if coarse_Nh is not None and coarse_Nh <= 0:
        coarse_Nh = None
    simulation_study(
        N=args.N,
        SEED=args.seed,
        coarse_Nh=coarse_Nh,
    )
