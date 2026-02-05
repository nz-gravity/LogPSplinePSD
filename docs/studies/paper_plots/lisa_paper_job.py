"""LISA paper job runner.

Loads a 3-channel XYZ time series (from ``.h5`` or ``.npz``), trims to a target
length with a desired block structure, then runs multivariate blocked NUTS with
optional frequency coarse graining and frequency truncation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

import h5py
import jax
import numpy as np

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.example_datasets.lisa_data import (
    analytic_covariance_from_model,
)
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

    n_blocks = max(1, n_time_target // block_size)
    n_used = n_blocks * block_size
    return n_blocks, n_used


def _load_xyz_h5(path: Path, *, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("--downsample must be >= 1.")
    with h5py.File(path, "r") as handle:
        t = np.asarray(handle["t"][::stride], dtype=float)
        x = np.asarray(handle["X2"][::stride], dtype=float)
        y = np.asarray(handle["Y2"][::stride], dtype=float)
        z = np.asarray(handle["Z2"][::stride], dtype=float)
    data = np.stack([x, y, z], axis=1)
    return t, data


def _load_xyz_npz(
    path: Path, *, stride: int
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    if stride < 1:
        raise ValueError("--downsample must be >= 1.")
    with np.load(path, allow_pickle=False) as npz:
        if "time" not in npz.files or "data" not in npz.files:
            raise ValueError("NPZ must contain 'time' and 'data' arrays.")
        t = np.asarray(npz["time"], dtype=float)[::stride]
        data = np.asarray(npz["data"], dtype=float)[::stride]
        true_psd = None
        if "freq_true" in npz.files and "true_matrix" in npz.files:
            freq_true = np.asarray(npz["freq_true"], dtype=float)
            true_matrix = np.asarray(npz["true_matrix"])
            true_psd = (freq_true, true_matrix)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected data shape (N,3); got {data.shape}.")
    return t, data, true_psd


def _infer_source(path: Path) -> Literal["h5", "npz"]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    if suffix == ".npz":
        return "npz"
    raise ValueError("Unsupported input; expected .h5/.hdf5 or .npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/tdi.h5",
        help="Input file path (.h5/.npz). Defaults to data/tdi.h5.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Keep every k-th sample from the input.",
    )
    parser.add_argument("--n-time", type=int, required=True, help="Target N.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=5000,
        help="Samples per Wishart block (used to pick n_time_blocks).",
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
    parser.add_argument("--vi-guide", type=str, default="diag")
    parser.add_argument("--vi-lr", type=float, default=1e-4)
    parser.add_argument("--vi-posterior-draws", type=int, default=512)

    parser.add_argument("--fmin", type=float, default=1e-4)
    parser.add_argument("--fmax", type=float, default=1e-1)

    parser.add_argument(
        "--coarse-bins",
        type=int,
        default=0,
        help="Enable coarse graining with n_bins=coarse_bins (0 disables).",
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

    data_path = Path(args.data).expanduser().resolve()
    source = _infer_source(data_path)
    stride = int(args.downsample)

    if source == "h5":
        t, y = _load_xyz_h5(data_path, stride=stride)
        true_psd_source = None
    else:
        t, y, true_psd_source = _load_xyz_npz(data_path, stride=stride)

    if t.size < 2:
        raise ValueError("Need at least 2 time samples.")
    dt = float(t[1] - t[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Invalid dt={dt}.")

    n_blocks, n_used = _resolve_blocks(args.n_time, args.block_size)
    if y.shape[0] < n_used:
        raise ValueError(
            f"Input series too short after downsampling: have {y.shape[0]}, need {n_used}."
        )

    y = y[:n_used]
    t = t[:n_used]

    fmin = float(args.fmin) if args.fmin is not None else None
    fmax = float(args.fmax) if args.fmax is not None else None

    coarse_cfg: CoarseGrainConfig | None = None
    coarse_bins = int(args.coarse_bins)
    if coarse_bins > 0:
        coarse_cfg = CoarseGrainConfig(
            enabled=True,
            n_bins=coarse_bins,
            f_min=fmin,
            f_max=fmax,
        )

    block_len = int(args.block_size)
    freq_eval = np.fft.rfftfreq(block_len, d=dt)[1:]
    if freq_eval.size == 0:
        raise ValueError(
            "block-size too small to retain positive frequencies."
        )

    if true_psd_source is None and not args.no_true_psd:
        true_matrix = analytic_covariance_from_model(
            freq_eval,
            dt=dt,
            n_time=int(n_used),
            model="scirdv1",
            central_freq=None,
        )
        true_psd_source = (freq_eval, true_matrix)

    logger.info(
        f"Running LISA job: data={data_path.name}, N={n_used}, dt={dt:g}, blocks={n_blocks}, "
        f"f=[{fmin:g},{fmax:g}], coarse_bins={coarse_bins or 'off'}, outdir={outdir}"
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
        outdir=str(outdir),
        verbose=True,
        compute_psis=False,
        skip_plot_diagnostics=False,
        n_time_blocks=int(n_blocks),
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
