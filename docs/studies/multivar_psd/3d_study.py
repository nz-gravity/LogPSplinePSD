"""Multivariate PSD simulation study with in-script VAR data generation.

CLI args:
1) seed (default 0)
2) mode: "large" or "short"
"""

import argparse
import os
from typing import Literal

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import numpy as np

from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

jax.config.update("jax_enable_x64", True)

set_level("DEBUG")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("out_var3")

DEFAULT_KNOT_METHOD = "density"
DEFAULT_N_TIME_BLOCKS = 4  # keep averaging enabled in both short/large modes
DEFAULT_TARGET_ACCEPT_PROB = 0.95
DEFAULT_MAX_TREE_DEPTH = 14
DEFAULT_INIT_FROM_VI = True
DEFAULT_VI_STEPS = 100_000
DEFAULT_VI_GUIDE = "lowrank:16"
VI_LR = 5e-4
DEFAULT_VI_PSD_MAX_DRAWS = 256
DEFAULT_POSTERIOR_PSD_MAX_DRAWS = 256
DEFAULT_ALPHA_DELTA = 1.0
DEFAULT_BETA_DELTA = 1.0
DEFAULT_N_SAMPLES = 1000
DEFAULT_N_WARMUP = 3000
DEFAULT_NUM_CHAINS = 4

DEFAULT_FS = 1.0  # Hz
DEFAULT_BURN_IN = 512
EPS = 1e-12

# VAR(2) setup (3 channels) embedded directly in this script.
A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [
        [-0.2, 0.5, 0.0],  # var2 -> var1 at lag 2
        [0.4, -0.1, 0.0],  # var1 -> var2 at lag 2
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

MODE_CONFIG = {
    "short": {"N": 1024, "coarse_Nh": None},
    "large": {"N": 16 * 1024, "coarse_Nh": 4},
}


def _log_var_coefficients() -> None:
    """Log the VAR(p) coefficient matrices used by this study."""
    logger.info("Using VAR coefficients:")
    for lag, coeff in enumerate(VAR_COEFFS, start=1):
        coeff_str = np.array2string(coeff, precision=4, suppress_small=False)
        logger.info(f"A{lag} =\n{coeff_str}")


def _companion_spectral_radius(var_coeffs: np.ndarray) -> float:
    """Return companion-matrix spectral radius for VAR(p) coefficients."""
    ar_order, n_channels, _ = var_coeffs.shape
    companion = np.zeros(
        (n_channels * ar_order, n_channels * ar_order), dtype=np.float64
    )
    companion[:n_channels, : (n_channels * ar_order)] = np.hstack(var_coeffs)
    if ar_order > 1:
        companion[n_channels:, :-n_channels] = np.eye(
            n_channels * (ar_order - 1), dtype=np.float64
        )
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def _simulate_var_process(
    n_samples: int,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    seed: int,
    *,
    fs: float = DEFAULT_FS,
    burn_in: int = DEFAULT_BURN_IN,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate VAR(p): x_t = sum_k A_k x_{t-k} + eps_t.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time grid in seconds.
    x : np.ndarray, shape (N, C)
        Simulated channels.
    """
    ar_order, n_channels, _ = var_coeffs.shape
    n_total = int(n_samples) + int(burn_in)
    rng = np.random.default_rng(int(seed))
    noise = rng.multivariate_normal(np.zeros(n_channels), sigma, size=n_total)
    x = np.zeros((n_total, n_channels), dtype=np.float64)

    for t_idx in range(ar_order, n_total):
        state = noise[t_idx].copy()
        for lag in range(1, ar_order + 1):
            state = state + var_coeffs[lag - 1] @ x[t_idx - lag]
        x[t_idx] = state

    x = x[burn_in:]
    t = np.arange(x.shape[0], dtype=np.float64) / float(fs)
    return t, x


def _calculate_true_var_psd_hz(
    freqs_hz: np.ndarray,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    *,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """Compute one-sided theoretical PSD matrix S(f) on a Hz frequency grid.

    Parameters
    ----------
    freqs_hz : (F,)
        Frequencies in Hz (not angular frequency).
    var_coeffs : (P, C, C)
        VAR coefficient matrices.
    sigma : (C, C)
        Innovation covariance.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    if freqs_hz.ndim != 1:
        raise ValueError("freqs_hz must be one-dimensional.")
    if freqs_hz.size and (
        np.min(freqs_hz) < 0.0 or np.max(freqs_hz) > (fs / 2.0 + 1e-12)
    ):
        raise ValueError("freqs_hz must lie in [0, fs/2].")

    ar_order, n_channels, _ = var_coeffs.shape
    omega = 2.0 * np.pi * freqs_hz / float(fs)  # rad/sample, derived from Hz
    psd = np.empty(
        (freqs_hz.shape[0], n_channels, n_channels), dtype=np.complex128
    )
    ident = np.eye(n_channels, dtype=np.complex128)

    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, ar_order + 1):
            a_f = a_f - var_coeffs[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        s_f = h_f @ sigma @ h_f.conj().T
        psd[idx] = (2.0 / float(fs)) * s_f

    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]

    # Keep matrices numerically Hermitian to protect downstream coherence math.
    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    psd = np.where(np.abs(psd) < EPS, EPS, psd)
    return psd


def simulation_study(
    *,
    seed: int = 0,
    mode: Literal["large", "short"] = "short",
    outdir: str = OUT,
    K: int = 10,
) -> None:
    cfg = MODE_CONFIG[mode]
    N = int(cfg["N"])
    coarse_Nh = cfg["coarse_Nh"]

    print(
        f">>>> Running simulation with mode={mode}, N={N}, K={K}, seed={seed} <<<<"
    )
    _log_var_coefficients()
    outdir = f"{HERE}/{outdir}/seed_{seed}_{mode}_N{N}_K{K}"
    os.makedirs(outdir, exist_ok=True)

    spectral_radius = _companion_spectral_radius(VAR_COEFFS)
    is_stationary = bool(spectral_radius < 1.0)
    logger.info(
        f"Stationarity check (companion spectral radius): {spectral_radius:.6f}"
    )
    if not is_stationary:
        raise ValueError(
            f"Non-stationary VAR coefficients (spectral radius={spectral_radius:.6f})."
        )

    t, data = _simulate_var_process(
        n_samples=N,
        var_coeffs=VAR_COEFFS,
        sigma=SIGMA,
        seed=seed,
        fs=DEFAULT_FS,
        burn_in=DEFAULT_BURN_IN,
    )
    if not np.all(np.isfinite(data)):
        raise ValueError("Generated VAR samples contain non-finite values.")
    ts = MultivariateTimeseries(t=t, y=data)

    freq_true_hz = np.fft.rfftfreq(N, d=1.0 / DEFAULT_FS)[1:]
    true_psd = _calculate_true_var_psd_hz(
        freq_true_hz,
        VAR_COEFFS,
        SIGMA,
        fs=DEFAULT_FS,
    )

    coarse_grain_config = None
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
        knot_kwargs=dict(method=DEFAULT_KNOT_METHOD),
        coarse_grain_config=coarse_grain_config,
        alpha_delta=DEFAULT_ALPHA_DELTA,
        beta_delta=DEFAULT_BETA_DELTA,
        compute_coherence_quantiles=True,
        true_psd=(freq_true_hz, true_psd),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Multivariate PSD study with in-script VAR(2) data generation "
            "and mode presets."
        )
    )
    parser.add_argument(
        "seed",
        type=int,
        nargs="?",
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("large", "short"),
        default="short",
        help="Preset size: short=1K samples (averaging only), large=16K samples (averaging + coarse graining).",
    )

    args = parser.parse_args()
    simulation_study(
        seed=args.seed,
        mode=args.mode,
    )
