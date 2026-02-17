#!/usr/bin/env python3
"""
Multivariate PSD + CSD estimation for daily log-return stock data using
log-P-spline inference.

Default tickers: AAPL, MSFT, JPM, XOM, WMT
Period : 2010–2020
"""

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from highlight_text import ax_text
from matplotlib import colors as mcolors

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.base import extract_plotting_data
from log_psplines.plotting.psd_matrix import plot_psd_matrix
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

HERE = Path(__file__).resolve().parent
BASE_RESULTS_DIR = HERE / "results" / "finance"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICKER_SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "WMT": "Industrial/Retail",
    "CAT": "Industrial/Retail",
    "SPY": "Benchmark",
}

SECTOR_COLORS = {
    "Technology": "#3B73B9",
    "Financials": "#9C4F96",
    "Energy": "#EF8A17",
    "Industrial/Retail": "#4DA167",
    "Benchmark": "#7F7F7F",
    "Other": "#999999",
}


def _sector_color(ticker: str) -> str:
    return SECTOR_COLORS.get(
        TICKER_SECTORS.get(ticker, "Other"), SECTOR_COLORS["Other"]
    )


def _with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgba = mcolors.to_rgba(color)
    return (rgba[0], rgba[1], rgba[2], alpha)


# Globals that get updated per CLI run
RESULTS_DIR = BASE_RESULTS_DIR
IDATA_PATH = RESULTS_DIR / "finance_pspline_inference.nc"
RUN_VI_ONLY = True


def _setup_presentation_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f7",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": ":",
            "grid.linewidth": 0.8,
            "axes.titlesize": 25,
            "axes.labelsize": 18,
            "legend.frameon": False,
            "font.size": 20,
        }
    )


_setup_presentation_style()


def _slugify_run(tickers, start, end):
    tick_str = "-".join(t.upper() for t in tickers)
    start_slug = start.replace("-", "")
    end_slug = end.replace("-", "")
    return f"{start_slug}_{end_slug}_{tick_str}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multivariate log-P-spline PSD demo for finance data."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=[
            "AAPL",
            "MSFT",
            # "JPM",  "BAC",
            "XOM",
            #  "CVX",
            # "WMT", "CAT",
            # "SPY",
        ],
        help="List of ticker symbols to download.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2020-01-01",
        help="Exclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for the output directory/cache.",
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
        help="Ignore existing cached inference and rerun.",
    )
    return parser.parse_args()


# ---------------------------------------------------------
# 1. DOWNLOAD DATA
# ---------------------------------------------------------


def download_prices(tickers, start, end):
    print("Downloading data...")
    raw = yf.download(tickers, start=start, end=end)

    # Extract price column robustly
    if "Adj Close" in raw.columns.get_level_values(0):
        prices = raw["Adj Close"]
        print("Using 'Adj Close'")
    else:
        prices = raw["Close"]
        print("Using 'Close'")

    return prices


# ---------------------------------------------------------
# 2. CONVERT TO STATIONARY SERIES (LOG-RETURNS)
# ---------------------------------------------------------


def make_log_returns(prices):
    # log-return r_t = log(P_t / P_{t-1})
    logr = np.log(prices / prices.shift(1)).dropna()

    # Demean
    logr = logr - logr.mean()

    return logr


# ---------------------------------------------------------
# 3. PLOT TIME SERIES
# ---------------------------------------------------------


def plot_timeseries(
    log_returns: pd.DataFrame, save_path: Path | None = None
) -> None:
    plt.figure(figsize=(14, 7.5))
    unique_sectors = []
    for idx, t in enumerate(log_returns.columns):
        color = _sector_color(t)
        linestyle = "-" if idx % 2 == 0 else "--"
        plt.plot(
            log_returns.index,
            log_returns[t],
            label=t,
            alpha=0.95,
            lw=1.8,
            color=color,
            ls=linestyle,
        )
        sect = TICKER_SECTORS.get(t, "Other")
        if sect not in unique_sectors:
            unique_sectors.append(sect)

    plt.title("Daily Log-Returns (demeaned)", pad=14)
    plt.xlabel("Date")
    plt.ylabel("Log-return")

    legend_lines = []
    legend_labels = []
    for sector in unique_sectors:
        legend_lines.append(
            plt.Line2D(
                [0],
                [0],
                color=SECTOR_COLORS.get(sector, "#777777"),
                lw=2.5,
            )
        )
        legend_labels.append(sector)

    leg1 = plt.legend(
        handles=legend_lines,
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        title="Sectors",
    )
    plt.gca().add_artist(leg1)
    plt.legend(loc="upper left")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# 4. ESTIMATE PSD + CSD MATRIX
# ---------------------------------------------------------


def _build_timeseries(log_returns: pd.DataFrame) -> MultivariateTimeseries:
    """Convert demeaned log returns to a MultivariateTimeseries object."""
    y = log_returns.values.astype(np.float64)
    t = np.arange(len(log_returns), dtype=np.float64)
    return MultivariateTimeseries(y=y, t=t)


def _select_time_blocks(n: int, min_block_len: int = 128) -> int:
    """
    Choose a block count for Wishart averaging that divides n
    and keeps block length ≥ `min_block_len`.
    """
    if n <= min_block_len:
        return 1

    max_blocks = max(1, n // min_block_len)
    # Prefer powers of two for FFT efficiency
    Nb = 1
    while (Nb * 2) <= max_blocks:
        Nb *= 2

    # Ensure divisibility
    while Nb > 1 and n % Nb != 0:
        Nb //= 2

    return max(1, Nb)


def _extract_median_psd(idata) -> tuple[np.ndarray, np.ndarray]:
    """Grab the 50th percentile PSD matrix from either posterior or VI groups."""
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
        raise RuntimeError("InferenceData missing PSD matrix summaries.")

    real_ds = group["psd_matrix_real"]
    imag_ds = group["psd_matrix_imag"]
    percentiles = np.asarray(real_ds.coords["percentile"].values, dtype=float)
    idx = int(np.argmin(np.abs(percentiles - 50.0)))

    freq = np.asarray(real_ds.coords["freq"].values, dtype=float)
    psd_real = np.asarray(real_ds.isel(percentile=idx))
    psd_imag = np.asarray(imag_ds.isel(percentile=idx))
    return freq, psd_real + 1j * psd_imag


def estimate_spectral_matrix(
    log_returns: pd.DataFrame,
    *,
    cache_results: bool = True,
) -> tuple[np.ndarray, np.ndarray, az.InferenceData, EmpiricalPSD]:
    """
    Estimate the multivariate spectral matrix using the log-P-spline sampler.

    Returns:
        freqs: np.ndarray of shape (N,)
        S:     complex ndarray of shape (N, p, p)
        idata: ArviZ InferenceData with full posterior/VI diagnostics
        empirical_psd: Welch estimate for overlay/reference
    """

    timeseries = _build_timeseries(log_returns)
    n = timeseries.y.shape[0]
    Nb = _select_time_blocks(n)
    if Nb > 1:
        print(f"Using {Nb} Wishart blocks " f"(Lb={n // Nb} samples)")
    else:
        print("Using full-length periodogram (1 block)")

    dt = timeseries.t[1] - timeseries.t[0]
    fs = 1.0 / dt
    fmin = fs / (n * 1.0)
    fmax = 0.5 * fs

    coarse_cfg = CoarseGrainConfig(
        enabled=False,
        Nc=160,
    )

    idata = None
    if cache_results and IDATA_PATH.exists():
        print(f"Loading cached inference from {IDATA_PATH}")
        idata = az.from_netcdf(IDATA_PATH)

    if idata is None:
        print("Running blocked multivariate NUTS (log-P-splines)...")
        idata = run_mcmc(
            data=timeseries,
            n_samples=750,
            n_warmup=750,
            n_knots=10,
            degree=3,
            diffMatrixOrder=2,
            Nb=Nb,
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


# ---------------------------------------------------------
# 5. PSD PLOT — PERIOD DOMAIN (DAYS → MONTHS → YEARS)
# ---------------------------------------------------------


def plot_psds_period(
    freqs: np.ndarray,
    S: np.ndarray,
    tickers: list[str],
    save_path: Path | None = None,
):
    """
    Plot PSD vs period instead of cycles/day.
    Period = 1 / frequency.
    """
    # Remove frequency = 0 (period = ∞)
    freqs_nz = freqs[1:]
    periods = 1 / freqs_nz  # in days

    plt.figure(figsize=(12, 6))

    for i, t in enumerate(tickers):
        plt.semilogy(
            periods,
            S[1:, i, i],
            label=t,
            lw=2.0,
            alpha=0.95,
        )

    plt.title("PSD of Log-Returns (period domain)", pad=14)
    plt.xlabel("Period (days / months / years)")
    plt.ylabel("PSD")
    plt.xscale("log")

    # Custom ticks for intuitive interpretation
    xticks = [
        1,
        2,
        5,  # days
        10,
        20,
        30,
        60,
        90,  # weeks/months
        180,
        365,
        365 * 2,
        365 * 5,  # half-year, 1y, 2y, 5y
    ]
    xtick_labels = [
        "1d",
        "2d",
        "5d",
        "10d",
        "20d",
        "1mo",
        "2mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
    ]

    plt.xticks(xticks, xtick_labels)
    plt.grid(ls=":", lw=1)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# 6. OPTIONAL: CSD + COHERENCE PLOTS
# ---------------------------------------------------------


def plot_csd(
    freqs: np.ndarray,
    S: np.ndarray,
    tickers: list[str],
    a: str = "AAPL",
    b: str = "MSFT",
    save_path: Path | None = None,
) -> None:
    i = tickers.index(a)
    j = tickers.index(b)
    Pxy = S[:, i, j]

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, np.abs(Pxy))
    plt.title(f"CSD Magnitude |S_{a},{b}(f)|")
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("|CSD|")
    plt.grid(ls=":", lw=1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_coherence(
    freqs: np.ndarray,
    S: np.ndarray,
    tickers: list[str],
    a: str = "AAPL",
    b: str = "MSFT",
    save_path: Path | None = None,
) -> None:
    i = tickers.index(a)
    j = tickers.index(b)

    S_ii = S[:, i, i]
    S_jj = S[:, j, j]
    S_ij = S[:, i, j]

    coh = np.abs(S_ij) ** 2 / (S_ii * S_jj)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, coh)
    plt.title(f"Coherence Between {a} and {b}")
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Coherence")
    plt.ylim(0, 1.05)
    plt.grid(ls=":", lw=1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_prices(prices: pd.DataFrame, save_path: Path | None = None) -> None:
    plt.figure(figsize=(14, 7.5))
    unique_sectors = []
    for idx, t in enumerate(prices.columns):
        color = _sector_color(t)
        linestyle = "-" if idx % 2 == 0 else "--"
        plt.plot(
            prices.index,
            prices[t],
            label=t,
            alpha=0.95,
            lw=1.8,
            color=color,
            ls=linestyle,
        )
        sector = TICKER_SECTORS.get(t, "Other")
        if sector not in unique_sectors:
            unique_sectors.append(sector)

    plt.title("Daily Closing Prices", pad=14)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.yscale("log")
    plt.xlim(prices.index.min(), prices.index.max())

    plt.legend(
        loc="upper left", fontsize=18, ncols=2, frameon=True, framealpha=0.5
    )

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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


def _get_percentile_slice(
    arr: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - target)))
    return arr[idx]


def plot_coherence_matrix(
    freqs: np.ndarray,
    coherence_quantiles: np.ndarray,
    percentiles: np.ndarray,
    tickers: list[str],
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

    n = len(tickers)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(3.6 * n, 3.2 * n),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    axes = np.asarray(axes)

    xticks = np.array([5, 30, 180, 365 * 2], dtype=float)
    xtick_labels = np.array(["5d", "1mo", "6mo", "2y"])
    valid = (xticks >= periods.min()) & (xticks <= periods.max())
    xticks = xticks[valid]
    xtick_labels = xtick_labels[valid]

    unique_sectors: list[str] = []

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#4a4a4a")
                spine.set_linewidth(1.2)
            if i < j:
                ax.set_axis_off()
                continue
            if i == j:
                ax.set_axis_off()
                sector = TICKER_SECTORS.get(tickers[i], "Other")
                if sector not in unique_sectors:
                    unique_sectors.append(sector)
                ax.text(
                    0.05,
                    0.15,
                    tickers[i],
                    ha="left",
                    va="top",
                    fontsize=25,
                    weight="bold",
                    color=_sector_color(tickers[i]),
                    transform=ax.transAxes,
                )
                continue

            row_color = _sector_color(tickers[i])
            col_color = _sector_color(tickers[j])
            if TICKER_SECTORS.get(tickers[j], "Other") not in unique_sectors:
                unique_sectors.append(TICKER_SECTORS.get(tickers[j], "Other"))

            ax.set_facecolor(_with_alpha(row_color, 0.14))

            ax.fill_between(
                periods,
                q05[1:, i, j],
                q95[1:, i, j],
                color=_with_alpha(col_color, 0.3),
            )
            ax.plot(
                periods,
                q50[1:, i, j],
                color=col_color,
                lw=1.7,
            )
            ax.axhline(
                0.5, color="#2f2f2f", linestyle=":", linewidth=0.9, zorder=1
            )
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.grid(ls=":", lw=0.7)
            ax.set_xlim(periods.min(), periods.max())

            # define coordinates inside cell
            label_x = 0.05
            label_y = 0.92
            font_size = 18

            # ticker colours
            color_i = _sector_color(tickers[i])
            color_j = _sector_color(tickers[j])

            # draw first ticker (row) in color_i
            ax.text(
                label_x,
                label_y,
                tickers[i],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=font_size,
                fontweight="bold",
                color=color_i,
                clip_on=False,
            )

            # draw “× + ticker_j” slightly to the right in color_j
            ax.text(
                label_x,
                label_y,  # adjusted offset
                f"\n× {tickers[j]}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=font_size,
                fontweight="bold",
                color=color_j,
                clip_on=False,
            )

            if i != n - 1:
                ax.set_xticklabels([])
            else:
                if xticks.size:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

            if j != 0:
                ax.set_yticklabels([])

    # fig.text(0.50, 0.08, "Period", ha="center", fontsize=30)
    # fig.text(0.08, 0.50, "Coherence", va="center", rotation="vertical", fontsize=30)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    slug = args.label or _slugify_run(args.tickers, args.start, args.end)

    global RESULTS_DIR, IDATA_PATH, RUN_VI_ONLY
    RESULTS_DIR = BASE_RESULTS_DIR / slug
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IDATA_PATH = RESULTS_DIR / "finance_pspline_inference.nc"
    RUN_VI_ONLY = args.vi_only

    print(f"Using output directory: {RESULTS_DIR}")
    print(f"Cache file: {IDATA_PATH}")

    # 1. Download prices
    prices = download_prices(args.tickers, args.start, args.end)
    plot_prices(prices, RESULTS_DIR / "finance_demo_prices.png")

    # 2. Convert to stationary series
    log_returns = make_log_returns(prices)

    # 3. Plot time-domain log-returns
    plot_timeseries(
        log_returns,
        RESULTS_DIR / "finance_demo_log_returns.png",
    )

    # 4. Estimate spectral matrix via log-P-splines
    freqs, S, idata, empirical_psd = estimate_spectral_matrix(
        log_returns, cache_results=not args.no_cache
    )

    # 5. Plot PSD in period domain
    plot_psds_period(
        freqs,
        S,
        args.tickers,
        save_path=RESULTS_DIR / "finance_demo_psd_period.png",
    )

    # 6. Multivariate PSD matrix summary (with coherence)
    plot_psd_matrix(
        idata=idata,
        freq=freqs,
        empirical_psd=empirical_psd,
        outdir=str(RESULTS_DIR),
        filename="finance_psd_matrix.png",
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
            args.tickers,
            RESULTS_DIR / "finance_demo_coherences.png",
        )

    # Optional pairwise overlays for specific tickers:
    # plot_csd(freqs, S, args.tickers, "AAPL", "MSFT", RESULTS_DIR / "csd.png")
    # plot_coherence(freqs, S, args.tickers, "AAPL", "MSFT", RESULTS_DIR / "coh.png")

    print("\nSpectral matrix shape:", S.shape)
    print("Example S(f0) matrix:")
    print(S[0])


if __name__ == "__main__":
    main()
