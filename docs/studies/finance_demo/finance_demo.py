#!/usr/bin/env python3
"""
Multivariate PSD + CSD estimation for daily log-return stock data using
log-P-spline inference.
X-axis is in PERIOD (days/months/years), not frequency.

Tickers: AAPL, MSFT, KO, PEP
Period : 2019–2024
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.base import extract_plotting_data
from log_psplines.plotting.psd_matrix import plot_psd_matrix

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "finance"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IDATA_PATH = RESULTS_DIR / "finance_pspline_inference.nc"
RUN_VI_ONLY = True

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
    plt.figure(figsize=(14, 6))
    for t in log_returns.columns:
        plt.plot(log_returns.index, log_returns[t], label=t, alpha=0.8)
    plt.title("Daily Log-Returns (demeaned)")
    plt.xlabel("Date")
    plt.ylabel("Log-return")
    plt.grid(ls=":", lw=1)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
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


def _select_time_blocks(n_time: int, min_block_len: int = 128) -> int:
    """
    Choose a block count for Wishart averaging that divides n_time
    and keeps block length ≥ `min_block_len`.
    """
    if n_time <= min_block_len:
        return 1

    max_blocks = max(1, n_time // min_block_len)
    # Prefer powers of two for FFT efficiency
    n_blocks = 1
    while (n_blocks * 2) <= max_blocks:
        n_blocks *= 2

    # Ensure divisibility
    while n_blocks > 1 and n_time % n_blocks != 0:
        n_blocks //= 2

    return max(1, n_blocks)


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
        freqs: np.ndarray of shape (n_freq,)
        S:     complex ndarray of shape (n_freq, n_channels, n_channels)
        idata: ArviZ InferenceData with full posterior/VI diagnostics
        empirical_psd: Welch estimate for overlay/reference
    """

    timeseries = _build_timeseries(log_returns)
    n_time = timeseries.y.shape[0]
    n_blocks = _select_time_blocks(n_time)
    if n_blocks > 1:
        print(
            f"Using {n_blocks} Wishart blocks "
            f"(block_len={n_time // n_blocks} samples)"
        )
    else:
        print("Using full-length periodogram (1 block)")

    dt = timeseries.t[1] - timeseries.t[0]
    fs = 1.0 / dt
    fmin = fs / (n_time * 1.0)
    fmax = 0.5 * fs

    coarse_cfg = CoarseGrainConfig(
        enabled=True,
        f_transition=5e-2,
        n_log_bins=160,
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
            n_knots=10,
            degree=3,
            diffMatrixOrder=2,
            n_time_blocks=n_blocks,
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
        plt.semilogy(periods, S[1:, i, i], label=t)

    plt.title("PSD of Log-Returns (period domain)")
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
    plt.figure(figsize=(14, 6))
    for t in prices.columns:
        plt.plot(prices.index, prices[t], label=t, alpha=0.8)
    plt.title("Daily Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.yscale("log")
    plt.grid(ls=":", lw=1)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
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
    fig, axes = plt.subplots(n, n, figsize=(3.6 * n, 3.2 * n))
    axes = np.asarray(axes)

    xticks = np.array(
        [
            1,
            2,
            5,
            10,
            20,
            30,
            60,
            90,
            180,
            365,
            365 * 2,
            365 * 5,
        ],
        dtype=float,
    )
    xtick_labels = np.array(
        [
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
                    tickers[i],
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                    transform=ax.transAxes,
                )
                continue

            ax.fill_between(
                periods,
                q05[1:, i, j],
                q95[1:, i, j],
                color="tab:blue",
                alpha=0.25,
            )
            ax.plot(
                periods,
                q50[1:, i, j],
                color="tab:blue",
                lw=1.6,
            )
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.grid(ls=":", lw=0.8)

            if i == n - 1:
                ax.set_xlabel(tickers[j])
            if j == 0:
                ax.set_ylabel(tickers[i])

            if xticks.size:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=45)

    fig.suptitle("Posterior Coherence Quantiles (period domain)", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "KO", "PEP"]
    start = "2008-01-01"
    end = "2020-01-01"

    # 1. Download prices
    prices = download_prices(tickers, start, end)
    plot_prices(prices, RESULTS_DIR / "finance_demo_prices.png")

    # 2. Convert to stationary series
    log_returns = make_log_returns(prices)

    # 3. Plot time-domain
    plot_timeseries(
        log_returns,
        RESULTS_DIR / "finance_demo_log_returns.png",
    )

    # 4. Estimate spectral matrix via log-P-splines
    freqs, S, idata, empirical_psd = estimate_spectral_matrix(log_returns)

    # 5. Plot PSD in period domain
    plot_psds_period(
        freqs,
        S,
        tickers,
        save_path=RESULTS_DIR / "finance_demo_psd_period.png",
    )

    # 6. Multivariate PSD matrix summary
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
            tickers,
            RESULTS_DIR / "finance_demo_coherences.png",
        )

    # Additional examples:
    # plot_csd(freqs, S, tickers, "AAPL", "MSFT", RESULTS_DIR / "csd.png")
    # plot_coherence(freqs, S, tickers, "AAPL", "MSFT", RESULTS_DIR / "coh.png")

    print("\nSpectral matrix shape:", S.shape)
    print("Example S(f0) matrix:")
    print(S[0])
