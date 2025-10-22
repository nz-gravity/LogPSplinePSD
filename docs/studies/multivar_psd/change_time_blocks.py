"""
Simulation study: effect of Wishart block averaging (`n_time_blocks`) on multivariate PSD inference.

This script reuses a fixed VAR(2) process, runs the multivariate blocked NUTS
sampler for a sweep of ``n_time_blocks`` values, and records the Relative
Integrated Absolute Error (RIAE) of the posterior median PSD as well as the
coverage of the 90% posterior intervals on the auto-spectra.

Outputs are written to ``docs/studies/multivar_psd/out_change_time_blocks/`` by
default:

* ``results.csv`` – summary table with RIAE / coverage / runtime per setting
* ``riae_vs_blocks.png`` – quick visualisation of RIAE vs. block count

Usage
-----
Run from the project root:

    python docs/studies/multivar_psd/change_time_blocks.py \
        --outdir docs/studies/multivar_psd/out_change_time_blocks

"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from log_psplines.example_datasets.varma_data import (
    VARMAData,
    _calculate_spec_matrix_helper,
)
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

# ArviZ emits a warning when fewer than 2 chains are present; suppress for this study.
warnings.filterwarnings(
    "ignore", message="Shape validation failed", module="arviz"
)

N = 8192 * 2  # length of time series


@dataclass
class StudyConfig:
    """Configuration for the time-block sweep."""

    n_time_blocks: Sequence[int]
    n_samples: int
    n_warmup: int
    rng_key: int
    sampler: str = "multivar_blocked_nuts"
    outdir: Path = Path("docs/studies/multivar_psd/out_change_time_blocks")
    save_idata: bool = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _compute_riae(
    freq: np.ndarray, estimate: np.ndarray, truth: np.ndarray
) -> float:
    """Relative integrated absolute error based on Frobenius norm."""
    diff = estimate - truth
    err = np.linalg.norm(diff, axis=(1, 2))  # Frobenius per frequency
    integral = np.trapezoid(err, freq)
    denom = freq[-1] - freq[0]
    return float(integral / denom)


def _extract_psd_quantiles(idata):
    """Return (freq, median_psd, q05, q95) from InferenceData."""
    psd_real = idata.posterior_psd["psd_matrix_real"]
    psd_imag = idata.posterior_psd["psd_matrix_imag"]
    percentiles = np.asarray(psd_real.coords["percentile"].values)
    freq = np.asarray(psd_real.coords["freq"].values)

    def grab(arr, p):
        idx = int(np.argmin(np.abs(percentiles - p)))
        return np.asarray(arr[idx])

    real_q05 = grab(psd_real, 5.0)
    real_q50 = grab(psd_real, 50.0)
    real_q95 = grab(psd_real, 95.0)

    imag_q50 = grab(psd_imag, 50.0)

    median = real_q50 + 1j * imag_q50
    return freq, median, real_q05, real_q95


def _compute_diag_coverage(
    real_q05: np.ndarray, real_q95: np.ndarray, truth: np.ndarray
) -> float:
    """90% coverage across diagonal elements."""
    diag_truth = truth.real[:, range(truth.shape[1]), range(truth.shape[2])]
    diag_q05 = real_q05[:, range(real_q05.shape[1]), range(real_q05.shape[2])]
    diag_q95 = real_q95[:, range(real_q95.shape[1]), range(real_q95.shape[2])]
    inside = (diag_truth >= diag_q05) & (diag_truth <= diag_q95)
    return float(np.mean(inside))


def _true_psd_on_grid(
    freq: np.ndarray,
    var_coeffs: np.ndarray,
    vma_coeffs: np.ndarray,
    sigma: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Evaluate the closed-form VARMA PSD on the supplied frequency grid."""
    # convert to normalised frequency in [0, 0.5)
    norm_freq = (freq / fs).reshape(-1, 1)
    spec = np.apply_along_axis(
        lambda f: _calculate_spec_matrix_helper(
            f[0], vma_coeffs.shape[1], var_coeffs, vma_coeffs, sigma
        ),
        axis=1,
        arr=norm_freq,
    )
    return spec / (2 * np.pi)


def run_study(config: StudyConfig) -> pd.DataFrame:
    """Execute the sweep and return a dataframe with summary statistics."""
    _ensure_dir(config.outdir)

    # Generate synthetic VAR(2) data once
    varma = VARMAData(n_samples=N, seed=1234)
    timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)
    results = []

    for n_blocks in config.n_time_blocks:
        print(f"Running sampler with n_time_blocks={n_blocks}...")
        block_outdir = config.outdir / f"blocks_{n_blocks}"
        if config.save_idata:
            _ensure_dir(block_outdir)
            outdir = block_outdir
        else:
            outdir = None

        idata = run_mcmc(
            timeseries,
            sampler=config.sampler,
            n_samples=config.n_samples,
            n_warmup=config.n_warmup,
            n_time_blocks=n_blocks,
            rng_key=config.rng_key,
            verbose=False,
            compute_lnz=False,
            outdir=str(outdir) if outdir is not None else None,
        )

        freq, median_psd, q05_real, q95_real = _extract_psd_quantiles(idata)

        true_psd_aligned = _true_psd_on_grid(
            freq,
            varma.var_coeffs,
            varma.vma_coeffs,
            varma.sigma,
            varma.fs,
        )

        riae = _compute_riae(freq, median_psd, true_psd_aligned)
        coverage = _compute_diag_coverage(q05_real, q95_real, true_psd_aligned)
        runtime = float(idata.attrs.get("runtime", np.nan))

        results.append(
            dict(
                n_time_blocks=n_blocks,
                riae=riae,
                diag_coverage=coverage,
                runtime_seconds=runtime,
                sampler_type=idata.attrs.get("sampler_type", "unknown"),
            )
        )

        if config.save_idata and outdir is not None:
            idata.to_netcdf(outdir / "inference_data.nc")

    df = (
        pd.DataFrame(results)
        .sort_values("n_time_blocks")
        .reset_index(drop=True)
    )
    return df


def plot_results(df: pd.DataFrame, outdir: Path) -> None:
    """Create a simple RIAE plot."""
    _ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    blocks = df["n_time_blocks"].to_numpy()
    riae = df["riae"].to_numpy()
    runtime = df["runtime_seconds"].to_numpy()

    ax.plot(
        blocks, riae, marker="o", linestyle="-", color="tab:blue", label="RIAE"
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of time blocks")
    ax.set_ylabel("RIAE (PSD median)")
    ax.set_title("Blocked averaging sensitivity")
    ax.grid(True, alpha=0.3, which="both")

    ax_runtime = ax.twinx()
    ax_runtime.plot(
        blocks,
        runtime,
        marker="s",
        linestyle="--",
        color="tab:orange",
        label="Runtime",
    )
    ax_runtime.set_ylabel("Runtime (seconds)")

    # Secondary x-axis for block length (n_time / n_blocks); assumes equal n_time for all runs.
    n_time = N  # matches VARMAData(n_samples=N)
    block_lengths = n_time // blocks

    def blocks_to_length(x):
        return n_time / x

    def length_to_blocks(x):
        return n_time / x

    secax = ax.secondary_xaxis(
        "top", functions=(blocks_to_length, length_to_blocks)
    )
    secax.set_xscale("log", base=2)
    secax.set_xlabel("Block length (samples)")
    secax.invert_xaxis()

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_runtime.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "riae_vs_blocks.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("docs/studies/multivar_psd/out_change_time_blocks"),
        help="Directory to store results and plots.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of n_time_blocks values to evaluate.",
    )
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-warmup", type=int, default=400)
    parser.add_argument("--rng-key", type=int, default=2024)
    parser.add_argument(
        "--save-idata",
        action="store_true",
        help="Persist InferenceData for each configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        n_time_blocks=args.blocks,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        rng_key=args.rng_key,
        outdir=args.outdir,
        save_idata=args.save_idata,
    )
    df = run_study(config)
    print(df)
    df.to_csv(config.outdir / "results.csv", index=False)
    plot_results(df, config.outdir)
    print(f"Results written to {config.outdir}")


if __name__ == "__main__":
    main()
