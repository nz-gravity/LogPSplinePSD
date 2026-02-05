#!/usr/bin/env python3
"""
Multivariate PSD + CSD estimation for hourly air quality / meteorology data
from the Beijing Multi-Site Air Quality dataset using log-P-splines.

Default variables: PM2.5, NO2, CO, TEMP, WSPM (wind speed).
Default window: 2013-03-01 to 2017-02-28.

Dataset: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.base import extract_plotting_data
from log_psplines.plotting.psd_matrix import plot_psd_matrix

HERE = Path(__file__).resolve().parent
BASE_RESULTS_DIR = HERE / "results" / "airquality"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_RESULTS_DIR
IDATA_PATH = RESULTS_DIR / "airquality_pspline_inference.nc"
RUN_VI_ONLY = True


# ---------------------------------------------------------------------------
# CLI / helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log-P-spline PSD analysis for the Beijing air quality dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to PRSA2017 CSV file (e.g., PRSA_Data_20130301-20170228.csv).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["PM2.5", "NO2", "CO", "TEMP", "WSPM"],
        help="Variables to analyse.",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Optional station names to keep (e.g., Aotizhongxin).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2013-03-01",
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2017-03-01",
        help="Exclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for output directory/cache slug.",
    )
    parser.add_argument(
        "--full-mcmc",
        dest="vi_only",
        action="store_false",
        help="Run full blocked NUTS instead of VI-only shortcut.",
    )
    parser.set_defaults(vi_only=True)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached inference and rerun estimation.",
    )
    return parser.parse_args()


def _slugify_run(data_path: str, variables: list[str], stations, start, end):
    source = Path(data_path).stem
    var_slug = "-".join(v.replace(" ", "").upper() for v in variables)
    station_slug = (
        "ALL"
        if not stations
        else "-".join(s.replace(" ", "") for s in stations)
    )
    start_slug = start.replace("-", "")
    end_slug = end.replace("-", "")
    return (
        f"{source}_{station_slug}_{var_slug}_{start_slug}_{end_slug}".lower()
    )


# ---------------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------------
def load_airquality_data(filepath: str, stations=None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    required_cols = {"year", "month", "day", "hour"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns {sorted(required_cols)}. Found {df.columns.tolist()}"
        )
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]], errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    if stations is not None and "station" in df.columns:
        df = df[df["station"].isin(stations)]
    return df


def select_vars_and_preprocess(
    df: pd.DataFrame, variables: list[str]
) -> pd.DataFrame:
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Variables not found in CSV: {missing}")

    df_vars = df[variables].copy()
    df_vars = df_vars.dropna()
    df_vars = df_vars - df_vars.mean()
    return df_vars


def _infer_sampling_interval_hours(index: pd.Index) -> float:
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 1.0
    return float(np.median(diffs) / 3600.0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_timeseries(df: pd.DataFrame, save_path: Path | None = None) -> None:
    plt.figure(figsize=(14, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column, alpha=0.8)
    plt.title("Hourly Anomalies (demeaned)")
    plt.xlabel("Date")
    plt.ylabel("Anomaly")
    plt.grid(ls=":", lw=1)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_psds_period(
    freqs: np.ndarray,
    psd_matrix: np.ndarray,
    vars_list: list[str],
    save_path: Path | None = None,
) -> None:
    freqs_nz = freqs[1:]
    periods = 1.0 / freqs_nz  # hours
    plt.figure(figsize=(12, 6))
    for i, var in enumerate(vars_list):
        plt.semilogy(periods, psd_matrix[1:, i, i], label=var)
    plt.title("PSD of Hourly Anomalies (period domain)")
    plt.xlabel("Period (hours / days / weeks)")
    plt.ylabel("PSD")
    plt.xscale("log")
    xticks = np.array(
        [
            1,
            3,
            6,
            12,
            24,
            24 * 3,
            24 * 7,
            24 * 14,
            24 * 30,
        ],
        dtype=float,
    )
    xtick_labels = np.array(
        ["1h", "3h", "6h", "12h", "1d", "3d", "1w", "2w", "1mo"]
    )
    valid = (xticks >= periods.min()) & (xticks <= periods.max())
    plt.xticks(xticks[valid], xtick_labels[valid])
    plt.grid(ls=":", lw=1)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def _get_percentile_slice(
    arr: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - target)))
    return arr[idx]


def plot_coherence_matrix(
    freqs: np.ndarray,
    coherence_quantiles: np.ndarray,
    percentiles: np.ndarray,
    vars_list: list[str],
    save_path: Path,
) -> None:
    q05 = _get_percentile_slice(coherence_quantiles, percentiles, 5.0)
    q50 = _get_percentile_slice(coherence_quantiles, percentiles, 50.0)
    q95 = _get_percentile_slice(coherence_quantiles, percentiles, 95.0)

    freq_nz = freqs[1:]
    if freq_nz.size == 0:
        raise ValueError(
            "Need at least two frequency bins for coherence plot."
        )
    periods = 1.0 / freq_nz

    n = len(vars_list)
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.2 * n))
    axes = np.asarray(axes)

    xticks = np.array(
        [
            1,
            3,
            6,
            12,
            24,
            24 * 3,
            24 * 7,
            24 * 14,
            24 * 30,
        ],
        dtype=float,
    )
    xtick_labels = np.array(
        ["1h", "3h", "6h", "12h", "1d", "3d", "1w", "2w", "1mo"]
    )
    valid = (xticks >= periods.min()) & (xticks <= periods.max())
    xticks = xticks[valid]
    xtick_labels = xtick_labels[valid]

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i < j:
                ax.set_axis_off()
                continue
            if i == j:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    vars_list[i],
                    ha="center",
                    va="center",
                    fontsize=13,
                    weight="bold",
                    transform=ax.transAxes,
                )
                continue

            ax.fill_between(
                periods,
                q05[1:, i, j],
                q95[1:, i, j],
                color="tab:blue",
                alpha=0.3,
            )
            ax.plot(
                periods,
                q50[1:, i, j],
                color="tab:blue",
                lw=1.5,
            )
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.grid(ls=":", lw=0.8)

            if i == n - 1:
                ax.set_xlabel(vars_list[j])
            if j == 0:
                ax.set_ylabel(vars_list[i])

            if xticks.size:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=40)

    fig.suptitle("Posterior Coherence Quantiles (period domain)", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PSD estimation helpers
# ---------------------------------------------------------------------------
def _select_time_blocks(n: int, min_block_len: int = 128) -> int:
    if n <= min_block_len:
        return 1
    max_blocks = max(1, n // min_block_len)
    Nb = 1
    while (Nb * 2) <= max_blocks:
        Nb *= 2
    while Nb > 1 and n % Nb != 0:
        Nb //= 2
    return max(1, Nb)


def _build_timeseries(
    dataframe: pd.DataFrame, dt_hours: float
) -> MultivariateTimeseries:
    y = dataframe.values.astype(np.float64)
    t = np.arange(len(dataframe), dtype=np.float64) * dt_hours
    return MultivariateTimeseries(y=y, t=t)


def _extract_median_psd(idata) -> tuple[np.ndarray, np.ndarray]:
    group = None
    if (
        hasattr(idata, "posterior_psd")
        and "psd_matrix_real" in idata.posterior_psd
    ):
        group = idata.posterior_psd
    elif (
        hasattr(idata, "vi_posterior_psd")
        and "psd_matrix_real" in idata.vi_posterior_psd
    ):
        group = idata.vi_posterior_psd
    else:
        raise RuntimeError("InferenceData missing psd_matrix_real entries.")

    real_ds = group["psd_matrix_real"]
    imag_ds = group["psd_matrix_imag"]
    percentiles = np.asarray(real_ds.coords["percentile"].values, dtype=float)
    idx = int(np.argmin(np.abs(percentiles - 50.0)))
    freq = np.asarray(real_ds.coords["freq"].values, dtype=float)
    psd_real = np.asarray(real_ds.isel(percentile=idx))
    psd_imag = np.asarray(imag_ds.isel(percentile=idx))
    return freq, psd_real + 1j * psd_imag


def _extract_coherence_quantiles(idata):
    extracted = extract_plotting_data(idata)

    quantiles = extracted.get("posterior_psd_matrix_quantiles")
    if quantiles is None:
        return None, None

    coherence = quantiles.get("coherence")
    if coherence is None:
        return None, None

    percentiles = np.asarray(quantiles["percentile"], dtype=float)
    freqs = extracted.get("frequencies")
    if freqs is None:
        freqs = np.asarray(
            idata.posterior_psd.coords["freq"].values, dtype=float
        )
    else:
        freqs = np.asarray(freqs, dtype=float)
    return freqs, (percentiles, np.asarray(coherence, dtype=float))


def estimate_spectral_matrix(
    df: pd.DataFrame, dt_hours: float, *, cache_results: bool = True
) -> tuple[np.ndarray, np.ndarray, az.InferenceData, EmpiricalPSD]:
    timeseries = _build_timeseries(df, dt_hours)
    n = timeseries.y.shape[0]
    Nb = _select_time_blocks(n)
    if Nb > 1:
        print(f"Using {Nb} Wishart blocks (Lb={n // Nb} samples)")
    else:
        print("Using full-length periodogram (1 block)")

    fs = 1.0 / dt_hours
    fmin = fs / (n * 1.0)
    fmax = 0.5 * fs

    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        Nc=200,
        f_min=fmin,
        f_max=fmax,
    )

    idata = None
    if cache_results and IDATA_PATH.exists():
        print(f"Loading cached inference from {IDATA_PATH}")
        idata = az.from_netcdf(IDATA_PATH)

    if idata is None:
        print("Running blocked multivariate NUTS (log-P-splines)...")
        idata = run_mcmc(
            data=timeseries,
            sampler="multivar_blocked_nuts",
            n_samples=750,
            n_warmup=750,
            n_knots=15,
            degree=3,
            diffMatrixOrder=2,
            n_time_blocks=Nb,
            coarse_grain_config=coarse_cfg,
            only_vi=RUN_VI_ONLY,
            vi_steps=15_000,
            vi_lr=5e-4,
            vi_progress_bar=True,
            rng_key=0,
            verbose=True,
            outdir=str(RESULTS_DIR),
            fmin=fmin,
            fmax=fmax,
        )
        if cache_results:
            idata.to_netcdf(IDATA_PATH)
            print(f"Saved inference to {IDATA_PATH}")

    freqs, psd_matrix = _extract_median_psd(idata)
    empirical_psd = timeseries.get_empirical_psd()
    return freqs, psd_matrix, idata, empirical_psd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    slug = args.label or _slugify_run(
        args.data_path, args.variables, args.stations, args.start, args.end
    )

    global RESULTS_DIR, IDATA_PATH, RUN_VI_ONLY
    RESULTS_DIR = BASE_RESULTS_DIR / slug
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IDATA_PATH = RESULTS_DIR / "airquality_pspline_inference.nc"
    RUN_VI_ONLY = args.vi_only

    print(f"Writing outputs to {RESULTS_DIR}")
    print(f"Cache path: {IDATA_PATH}")

    df_raw = load_airquality_data(args.data_path, args.stations)
    if args.start:
        df_raw = df_raw[df_raw.index >= pd.Timestamp(args.start)]
    if args.end:
        df_raw = df_raw[df_raw.index < pd.Timestamp(args.end)]
    if df_raw.empty:
        raise RuntimeError(
            "Filtered dataframe is empty. Check date range / stations."
        )

    df_sel = select_vars_and_preprocess(df_raw, args.variables)
    plot_timeseries(df_sel, RESULTS_DIR / "airquality_timeseries.png")

    dt_hours = _infer_sampling_interval_hours(df_sel.index)
    freqs, psd_matrix, idata, empirical_psd = estimate_spectral_matrix(
        df_sel, dt_hours, cache_results=not args.no_cache
    )

    plot_psds_period(
        freqs,
        psd_matrix,
        args.variables,
        save_path=RESULTS_DIR / "airquality_psd_period.png",
    )

    plot_psd_matrix(
        idata=idata,
        freq=freqs,
        empirical_psd=empirical_psd,
        outdir=str(RESULTS_DIR),
        filename="airquality_psd_matrix.png",
        diag_yscale="log",
        offdiag_yscale="linear",
        xscale="log",
        show_coherence=True,
        overlay_vi=True,
    )

    coh_freqs, coh_payload = _extract_coherence_quantiles(idata)
    if coh_payload is not None and coh_freqs is not None:
        percentiles, coherence_quantiles = coh_payload
        plot_coherence_matrix(
            coh_freqs,
            coherence_quantiles,
            percentiles,
            args.variables,
            RESULTS_DIR / "airquality_coherences.png",
        )

    print("\nSpectral matrix shape:", psd_matrix.shape)
    print("Example S(f0) matrix:")
    print(psd_matrix[0])


if __name__ == "__main__":
    main()
