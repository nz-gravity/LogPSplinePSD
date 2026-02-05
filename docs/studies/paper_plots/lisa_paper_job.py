"""LISA paper job runner.

Loads a 3-channel XYZ time series from a paper-sized lisatools-synth NPZ (see
``generate_lisa_paper_data.py``), trims to a target length with a desired block
structure, then runs multivariate blocked NUTS with optional frequency coarse
graining and frequency truncation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import numpy as np

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
jax.config.update("jax_enable_x64", True)
set_level("INFO")


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


def _load_paper_synth_npz(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Load the paper-sized lisatools-synth NPZ."""
    with np.load(path, allow_pickle=False) as npz:
        required = {"time", "data", "freq_true", "true_matrix"}
        missing = required.difference(npz.files)
        if missing:
            raise ValueError(
                f"Synth NPZ is missing keys {sorted(missing)} (has {npz.files})."
            )
        t = np.asarray(npz["time"], dtype=float)
        y = np.asarray(npz["data"], dtype=float)
        freq_true = np.asarray(npz["freq_true"], dtype=float)
        true_matrix = np.asarray(npz["true_matrix"])
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"Expected synth data shape (N,3); got {y.shape}.")
    return t, y, (freq_true, true_matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--synth-npz",
        type=str,
        required=True,
        help="Path to paper-sized lisatools synth NPZ (time/data/freq_true/true_matrix).",
    )
    parser.add_argument("--n-time", type=int, required=True, help="Target N.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=5000,
        help="Samples per Wishart block (used to pick Nb).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knots", type=int, default=30)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--diff-order", type=int, default=2)

    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)

    parser.add_argument("--alpha-delta", type=float, default=3.0)
    parser.add_argument("--beta-delta", type=float, default=3.0)
    parser.add_argument("--target-accept", type=float, default=0.7)
    parser.add_argument("--max-tree-depth", type=int, default=10)

    parser.add_argument(
        "--init-from-vi", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--vi-steps", type=int, default=50_000)
    parser.add_argument("--vi-guide", type=str, default="lowrank:16")
    parser.add_argument("--vi-lr", type=float, default=1e-4)
    parser.add_argument("--vi-posterior-draws", type=int, default=512)

    parser.add_argument("--fmin", type=float, default=1e-4)
    parser.add_argument("--fmax", type=float, default=1e-1)

    parser.add_argument(
        "--coarse-bins",
        type=int,
        default=0,
        help="Enable coarse graining with Nc=coarse_bins (0 disables).",
    )
    parser.add_argument(
        "--coarse-n-freqs-per-bin",
        type=int,
        default=0,
        help="Enable coarse graining with this odd bin size (0 disables). If set, overrides --coarse-bins.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents.",
    )
    parser.add_argument(
        "--no-true-psd",
        action="store_true",
        help="Do not attach an analytic true PSD/CSD matrix for overlays.",
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

    synth_path = Path(args.synth_npz).expanduser().resolve()
    t, y, true_psd_source = _load_paper_synth_npz(synth_path)

    if t.size < 2:
        raise ValueError("Need at least 2 time samples.")
    dt = float(t[1] - t[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Invalid dt={dt}.")

    Nb, n_used = _resolve_blocks(args.n, args.block_size)
    if y.shape[0] < n_used:
        raise ValueError(
            f"Input series too short: have {y.shape[0]}, need {n_used}."
        )

    y = y[:n_used]
    t = t[:n_used]
    duration_days = float((t[-1] - t[0]) / 86_400.0)

    fmin = float(args.fmin) if args.fmin is not None else None
    fmax = float(args.fmax) if args.fmax is not None else None
    nyq = 0.5 / float(dt)
    if fmax is not None and float(fmax) > nyq:
        logger.warning(
            f"Requested fmax={float(fmax):g} exceeds Nyquist={nyq:g} (dt={dt:g}); clamping to Nyquist."
        )
        fmax = nyq

    coarse_cfg: CoarseGrainConfig | None = None
    coarse_n_freqs = int(args.coarse_Nh)
    coarse_bins = int(args.coarse_bins)
    if coarse_n_freqs > 0:
        coarse_cfg = CoarseGrainConfig(
            enabled=True,
            Nc=None,
            Nh=coarse_n_freqs,
            f_min=fmin,
            f_max=fmax,
        )
    elif coarse_bins > 0:
        coarse_cfg = CoarseGrainConfig(
            enabled=True,
            Nc=coarse_bins,
            Nh=None,
            f_min=fmin,
            f_max=fmax,
        )

    if true_psd_source is None and not args.no_true_psd:
        raise RuntimeError(
            "Synth NPZ did not provide true PSD keys; rerun the synth generator."
        )

    logger.info(
        f"Running LISA job: data={synth_path.name}, N={n_used}, dt={dt:g}, duration_days={duration_days:.2f}, "
        f"blocks={Nb}, f=[{fmin:g},{fmax:g}], coarse={'Nh='+str(coarse_n_freqs) if coarse_n_freqs>0 else ('Nc='+str(coarse_bins) if coarse_bins>0 else 'off')}, outdir={outdir}"
    )

    ts = MultivariateTimeseries(t=t, y=y)
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
        Nb=int(Nb),
        coarse_grain_config=coarse_cfg,
        fmin=fmin,
        fmax=fmax,
        alpha_delta=float(args.alpha_delta),
        beta_delta=float(args.beta_delta),
        init_from_vi=bool(args.init_from_vi),
        vi_steps=int(args.vi_steps),
        vi_lr=float(args.vi_lr),
        vi_guide=str(args.vi_guide),
        vi_posterior_draws=int(args.vi_posterior_draws),
        vi_progress_bar=True,
        target_accept_prob=float(args.target_accept),
        max_tree_depth=int(args.max_tree_depth),
        dense_mass=True,
        knot_kwargs=dict(method="log"),
        true_psd=true_psd_source,
    )


if __name__ == "__main__":
    main()
