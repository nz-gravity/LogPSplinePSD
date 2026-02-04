from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure local package imports work even when running outside the repo .venv.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.coarse_grain import compute_binning_structure
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.diagnostics.coarse_grain_checks import (
    bin_doubling_stiffness_check,
)
from log_psplines.logger import logger, set_level


def _posterior_median(idata, name: str) -> np.ndarray | None:
    if not hasattr(idata, "sample_stats"):
        return None
    if name not in idata.sample_stats:
        return None
    array = np.asarray(idata.sample_stats[name].values)
    if array.ndim < 2:
        return array
    return np.median(array, axis=(0, 1))


def _parse_eps(text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("eps list is empty.")
    return tuple(float(p) for p in parts)


def _direction_vector(
    *,
    direction: str,
    n_dim: int,
    n_theta: int,
) -> np.ndarray:
    direction = direction.strip().lower()
    total = int(n_dim + 2 * n_theta)

    if direction.startswith("random"):
        seed = 0
        if ":" in direction:
            seed = int(direction.split(":", 1)[1])
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(total,)).astype(np.float64)
        v /= float(np.linalg.norm(v))
        return v

    if ":" not in direction:
        raise ValueError(
            "direction must be like 'log_delta_sq:2', 'theta_re:0', 'theta_im:0', or 'random:0'."
        )

    kind, idx_s = direction.split(":", 1)
    idx = int(idx_s)
    if kind == "log_delta_sq":
        if idx < 0 or idx >= n_dim:
            raise ValueError(f"log_delta_sq index out of range: {idx}.")
        offset = 0
    elif kind == "theta_re":
        if idx < 0 or idx >= n_theta:
            raise ValueError(f"theta_re index out of range: {idx}.")
        offset = n_dim
    elif kind == "theta_im":
        if idx < 0 or idx >= n_theta:
            raise ValueError(f"theta_im index out of range: {idx}.")
        offset = n_dim + n_theta
    else:
        raise ValueError(f"Unknown direction kind: {kind}.")

    v = np.zeros((total,), dtype=np.float64)
    v[offset + idx] = 1.0
    return v


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bin-doubling stiffness proxy check for LISA multivariate coarse graining."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Results directory containing inference_data.nc and lisatools_synth_data.npz.",
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=100,
        help="High-bin index (0-based) to check; requires an adjacent bin.",
    )
    parser.add_argument(
        "--eps",
        type=str,
        default="1e-3,1e-2,5e-2",
        help="Comma-separated finite-difference epsilons.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="log_delta_sq:2",
        help="Direction: log_delta_sq:i | theta_re:i | theta_im:i | random:seed",
    )
    parser.add_argument("--fmin", type=float, default=1e-4)
    parser.add_argument("--fmax", type=float, default=1e-1)
    parser.add_argument("--n-log-bins", type=int, default=512)
    parser.add_argument("--keep-low", action="store_true")
    parser.add_argument("--binning", type=str, default="linear")
    parser.add_argument("--representative", type=str, default="middle")
    parser.add_argument(
        "--n-freqs-per-bin",
        type=int,
        default=0,
        help="If >0, use fixed-size bins with this many (odd) frequencies each.",
    )
    parser.add_argument("--loglevel", type=str, default="INFO")
    args = parser.parse_args()

    set_level(args.loglevel.upper())

    results_dir = args.results
    idata_path = results_dir / "inference_data.nc"
    npz_path = results_dir / "lisatools_synth_data.npz"
    if not idata_path.exists():
        raise FileNotFoundError(f"Missing {idata_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    import arviz as az

    logger.info(f"Loading {idata_path}...")
    idata = az.from_netcdf(str(idata_path))

    logger.info(f"Loading {npz_path}...")
    with np.load(npz_path, allow_pickle=False) as synth:
        t = synth["time"]
        y = synth["data"]
        block_len_samples = (
            int(synth["block_len_samples"])
            if "block_len_samples" in synth
            else None
        )

    dt = float(t[1] - t[0])
    fs = 1.0 / dt
    n_time = int(y.shape[0])
    if block_len_samples is None:
        raise ValueError(
            "Expected block_len_samples in synthetic NPZ for this check."
        )

    n_blocks = max(1, int(n_time // block_len_samples))
    n_used = int(n_blocks * block_len_samples)
    if n_used != n_time:
        t = t[:n_used]
        y = y[:n_used]
    logger.info(
        f"Using n_blocks={n_blocks}, block_len_samples={block_len_samples}, n_time={y.shape[0]}."
    )

    raw_series = MultivariateTimeseries(y=y, t=t)
    standardized_ts = raw_series.standardise_for_psd()
    fft_fine = standardized_ts.to_wishart_stats(
        n_blocks=n_blocks, fmin=float(args.fmin), fmax=float(args.fmax)
    )

    n_dim = int(fft_fine.n_dim)
    n_theta = int(n_dim * (n_dim - 1) / 2)
    epsilons = _parse_eps(args.eps)

    spec = compute_binning_structure(
        np.asarray(fft_fine.freq, dtype=float),
        f_transition=1e-6,
        n_log_bins=int(args.n_log_bins),
        binning=str(args.binning),
        representative=str(args.representative),
        keep_low=bool(args.keep_low),
        n_freqs_per_bin=(
            None
            if int(args.n_freqs_per_bin) <= 0
            else int(args.n_freqs_per_bin)
        ),
        f_min=float(args.fmin),
        f_max=float(args.fmax),
    )

    log_delta_sq = _posterior_median(idata, "log_delta_sq")
    theta_re = _posterior_median(idata, "theta_re")
    theta_im = _posterior_median(idata, "theta_im")
    if log_delta_sq is None:
        raise ValueError("log_delta_sq not found in sample_stats.")
    log_delta_sq = np.asarray(log_delta_sq, dtype=np.float64)

    param_idx = (
        int(spec.n_low + args.bin) if bool(args.keep_low) else int(args.bin)
    )
    if param_idx < 0 or param_idx >= log_delta_sq.shape[0]:
        raise ValueError(
            f"Requested param_idx={param_idx} out of range for log_delta_sq[0]={log_delta_sq.shape[0]}."
        )

    log_delta_sq_bin = np.asarray(log_delta_sq[param_idx], dtype=np.float64)
    if log_delta_sq_bin.shape != (n_dim,):
        raise ValueError("log_delta_sq bin shape mismatch.")

    if theta_re is None or theta_im is None:
        theta_re_bin = None
        theta_im_bin = None
        logger.warning(
            "theta_re/theta_im not found in sample_stats; running stiffness check with theta=0."
        )
    else:
        theta_re = np.asarray(theta_re, dtype=np.float64)
        theta_im = np.asarray(theta_im, dtype=np.float64)
        theta_re_bin = np.asarray(theta_re[param_idx], dtype=np.float64)
        theta_im_bin = np.asarray(theta_im[param_idx], dtype=np.float64)
        if theta_re_bin.shape != (n_theta,) or theta_im_bin.shape != (
            n_theta,
        ):
            raise ValueError("theta bin shape mismatch.")

    v = _direction_vector(
        direction=args.direction, n_dim=n_dim, n_theta=n_theta
    )
    res = bin_doubling_stiffness_check(
        fft_fine=fft_fine,
        spec=spec,
        bin_index=int(args.bin),
        nu=int(fft_fine.nu),
        log_delta_sq_bin=log_delta_sq_bin,
        theta_re_bin=theta_re_bin,
        theta_im_bin=theta_im_bin,
        direction=v,
        epsilons=epsilons,
    )

    logger.info(
        f"Bin-doubling stiffness check bin={res.bin_index}: "
        f"N_small={res.n_members_small}, N_large={res.n_members_large} "
        f"(expected ratio ~ {res.n_members_large / res.n_members_small:.3g})."
    )
    for entry in res.eps_results:
        logger.info(
            f"  eps={entry.eps:g}: kappa_small={entry.kappa_small:.6e}, "
            f"kappa_large={entry.kappa_large:.6e}, ratio={entry.ratio:.6g} "
            f"(expected {entry.expected_ratio:.6g})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
