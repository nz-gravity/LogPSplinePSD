"""
Simulation study: effect of Wishart block averaging (`Nb`) on multivariate PSD inference.

This script reuses a fixed VAR(2) process, runs the multivariate blocked NUTS
sampler for a sweep of ``Nb`` values, and records the Relative
Integrated Absolute Error (RIAE) of the posterior median PSD as well as the
coverage of the 90% posterior intervals on the auto-spectra.

Outputs are written to ``docs/studies/multivar_psd/out_change_time_blocks/`` by
default:

* ``results.csv`` – per-run table with RIAE / coverage / runtime per seed
* ``summary.csv`` – aggregated means / standard deviations across seeds
* ``riae_vs_blocks.png`` – visualisation of RIAE, runtime, and coverage vs. block count
* (optional) PSD quantile plots when ``--save-psd-plots`` is provided

Usage
-----
Run from the project root:

    python docs/studies/multivar_psd/change_time_blocks.py \
        --outdir docs/studies/multivar_psd/out_change_time_blocks \
        --seeds 0 1 2

"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from log_psplines.example_datasets.varma_data import (
    VARMAData,
    _calculate_spec_matrix_helper,
)
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc
from log_psplines.plotting import plot_psd_matrix

# ArviZ emits a warning when fewer than 2 chains are present; suppress for this study.
warnings.filterwarnings(
    "ignore", message="Shape validation failed", module="arviz"
)

N = 2048  # length of time series for VARMA simulations


@dataclass
class StudyConfig:
    """Configuration for the time-block sweep."""

    Nb: Sequence[int]
    n_samples: int
    n_warmup: int
    seeds: Sequence[int] = (0, 1, 2)
    sampler: str = "multivar_blocked_nuts"
    outdir: Path = Path("docs/studies/multivar_psd/out_change_time_blocks")
    save_idata: bool = False
    save_psd_plots: bool = False


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

    results: list[dict[str, float | int | str]] = []

    for seed in config.seeds:
        varma = VARMAData(n_samples=N, seed=seed)
        timeseries = MultivariateTimeseries(t=varma.time, y=varma.data)

        for Nb in config.Nb:
            print(f"Running sampler with Nb={Nb}, seed={seed}...")

            block_dir = config.outdir / f"blocks_{Nb}" / f"seed_{seed}"
            if config.save_idata or config.save_psd_plots:
                _ensure_dir(block_dir)
            outdir = block_dir if config.save_idata else None

            idata = run_mcmc(
                timeseries,
                sampler=config.sampler,
                n_samples=config.n_samples,
                n_warmup=config.n_warmup,
                Nb=Nb,
                rng_key=seed,
                verbose=False,
                compute_lnz=False,
                outdir=str(outdir) if outdir is not None else None,
            )

            freq, median_psd, q05_real, q95_real = _extract_psd_quantiles(
                idata
            )

            true_psd_aligned = _true_psd_on_grid(
                freq,
                varma.var_coeffs,
                varma.vma_coeffs,
                varma.sigma,
                varma.fs,
            )

            riae = _compute_riae(freq, median_psd, true_psd_aligned)
            coverage = _compute_diag_coverage(
                q05_real, q95_real, true_psd_aligned
            )
            runtime = float(idata.attrs.get("runtime", np.nan))

            results.append(
                dict(
                    Nb=Nb,
                    seed=seed,
                    riae=riae,
                    diag_coverage=coverage,
                    runtime_seconds=runtime,
                    sampler_type=idata.attrs.get("sampler_type", "unknown"),
                )
            )

            if config.save_psd_plots:
                plot_dir = block_dir
                try:
                    plot_psd_matrix(
                        idata=idata,
                        freq=freq,
                        true_psd=true_psd_aligned,
                        outdir=str(plot_dir),
                        filename=f"psd_quantiles_blocks_{Nb}_seed_{seed}.png",
                        save=True,
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - plotting fallback
                    print(
                        f"Warning: could not generate PSD plot for blocks={Nb}, seed={seed}: {exc}"
                    )

            if config.save_idata:
                idata.to_netcdf(block_dir / "inference_data.nc")

    df = (
        pd.DataFrame(results)
        .sort_values(["Nb", "seed"])
        .reset_index(drop=True)
    )
    return df


def plot_results(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Plot summary statistics and return aggregated dataframe."""
    _ensure_dir(outdir)

    summary = (
        df.groupby("Nb")
        .agg(
            riae_mean=("riae", "mean"),
            riae_std=("riae", "std"),
            runtime_mean=("runtime_seconds", "mean"),
            runtime_std=("runtime_seconds", "std"),
            coverage_mean=("diag_coverage", "mean"),
            coverage_std=("diag_coverage", "std"),
            n_runs=("seed", "count"),
        )
        .reset_index()
    )

    summary.fillna(0.0, inplace=True)
    summary["n_runs"] = summary["n_runs"].astype(int)

    blocks = summary["Nb"].to_numpy()
    riae_mean = summary["riae_mean"].to_numpy()
    riae_std = summary["riae_std"].to_numpy()
    runtime_mean = summary["runtime_mean"].to_numpy()
    runtime_std = summary["runtime_std"].to_numpy()
    coverage_mean = summary["coverage_mean"].to_numpy()
    coverage_std = summary["coverage_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(
        blocks,
        riae_mean,
        yerr=riae_std,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="RIAE",
        capsize=3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(blocks)
    ax.set_xlabel("Number of time blocks")
    ax.set_ylabel("RIAE (PSD median)")
    ax.set_title("Blocked averaging sensitivity")
    ax.grid(True, alpha=0.3, which="both")

    ax_runtime = ax.twinx()
    ax_runtime.errorbar(
        blocks,
        runtime_mean,
        yerr=runtime_std,
        marker="s",
        linestyle="--",
        color="tab:orange",
        label="Runtime",
        capsize=3,
    )
    ax_runtime.set_ylabel("Runtime (seconds)")
    ax_runtime.tick_params(axis="y", colors="tab:orange")

    ax_cov = ax.twinx()
    ax_cov.spines["right"].set_position(("axes", 1.12))
    ax_cov.set_frame_on(True)
    ax_cov.patch.set_visible(False)
    for spine in ax_cov.spines.values():
        spine.set_visible(False)
    ax_cov.spines["right"].set_visible(True)
    ax_cov.errorbar(
        blocks,
        coverage_mean,
        yerr=coverage_std,
        marker="^",
        linestyle="-.",
        color="tab:green",
        label="Diag coverage",
        capsize=3,
    )
    ax_cov.set_ylabel("Diagonal coverage", color="tab:green")
    ax_cov.set_ylim(0.0, 1.05)
    ax_cov.tick_params(axis="y", colors="tab:green")

    def blocks_to_length(x):
        return N / x

    def length_to_blocks(x):
        return N / x

    secax = ax.secondary_xaxis(
        "top", functions=(blocks_to_length, length_to_blocks)
    )
    secax.set_xlabel("Block length (samples)")
    lengths = blocks_to_length(blocks)
    unique_lengths = np.unique(lengths)[::-1]
    if unique_lengths.size >= 2:
        secax.set_xlim(unique_lengths[0], unique_lengths[-1])
    secax.set_xticks(unique_lengths)
    length_labels = []
    for val in unique_lengths:
        exponent = np.log2(val)
        if np.isclose(exponent, np.round(exponent), atol=1e-6):
            length_labels.append(f"$2^{int(np.round(exponent))}$")
        else:
            length_labels.append(f"{int(val)}")
    secax.set_xticklabels(length_labels)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_runtime.get_legend_handles_labels()
    lines3, labels3 = ax_cov.get_legend_handles_labels()
    ax.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "riae_vs_blocks.png", dpi=150)
    plt.close(fig)

    return summary


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
        help="List of Nb values to evaluate.",
    )
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-warmup", type=int, default=400)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for data regeneration / sampler RNG.",
    )
    parser.add_argument(
        "--save-idata",
        action="store_true",
        help="Persist InferenceData for each configuration.",
    )
    parser.add_argument(
        "--save-psd-plots",
        action="store_true",
        help="Store PSD quantile plots for each run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        Nb=args.blocks,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seeds=args.seeds,
        outdir=args.outdir,
        save_idata=args.save_idata,
        save_psd_plots=args.save_psd_plots,
    )
    df = run_study(config)
    print(df)
    df.to_csv(config.outdir / "results.csv", index=False)
    summary = plot_results(df, config.outdir)
    summary.to_csv(config.outdir / "summary.csv", index=False)
    config_json = asdict(config)
    config_json["outdir"] = str(config_json["outdir"])
    config_json["seeds"] = list(config_json["seeds"])
    with open(config.outdir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config_json, fh, indent=2, default=list)
    print(summary)
    print(f"Results written to {config.outdir}")


if __name__ == "__main__":
    main()
