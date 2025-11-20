#!/usr/bin/env python3
"""
Multivariate PSD + CSD estimation for daily log-return stock data.
X-axis is in PERIOD (days/months/years), not frequency.

Tickers: AAPL, MSFT, KO, PEP
Period : 2019–2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import signal

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


def plot_timeseries(log_returns):
    plt.figure(figsize=(14, 6))
    for t in log_returns.columns:
        plt.plot(log_returns.index, log_returns[t], label=t, alpha=0.8)
    plt.title("Daily Log-Returns (demeaned)")
    plt.xlabel("Date")
    plt.ylabel("Log-return")
    plt.grid(ls=":", lw=1)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4. ESTIMATE PSD + CSD MATRIX
# ---------------------------------------------------------


def estimate_spectral_matrix(log_returns, fs=1.0, nperseg=256):
    """
    Returns:
        freqs: (n_freq,)
        S:     (n_freq, N, N) complex spectral matrix
    """
    tickers = list(log_returns.columns)
    N = len(tickers)

    # Frequency grid
    f, _ = signal.welch(
        log_returns[tickers[0]].values,
        fs=fs,
        nperseg=nperseg,
        detrend="constant",
        window="hann",
    )

    S = np.zeros((len(f), N, N), dtype=complex)

    for i, ti in enumerate(tickers):
        xi = log_returns[ti].values

        for j, tj in enumerate(tickers):
            if j < i:
                S[:, i, j] = np.conj(S[:, j, i])
                continue

            xj = log_returns[tj].values

            if i == j:
                _, Pxx = signal.welch(
                    xi,
                    fs=fs,
                    nperseg=nperseg,
                    detrend="constant",
                    window="hann",
                )
                S[:, i, j] = Pxx
            else:
                _, Pxy = signal.csd(
                    xi,
                    xj,
                    fs=fs,
                    nperseg=nperseg,
                    detrend="constant",
                    window="hann",
                )
                S[:, i, j] = Pxy

    return f, S


# ---------------------------------------------------------
# 5. PSD PLOT — PERIOD DOMAIN (DAYS → MONTHS → YEARS)
# ---------------------------------------------------------


def plot_psds_period(freqs, S, tickers):
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
    plt.show()


# ---------------------------------------------------------
# 6. OPTIONAL: CSD + COHERENCE PLOTS
# ---------------------------------------------------------


def plot_csd(freqs, S, tickers, a="AAPL", b="MSFT"):
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
    plt.show()


def plot_coherence(freqs, S, tickers, a="AAPL", b="MSFT"):
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
    plt.show()


def plot_all_coherences(freqs, S, tickers):
    """
    Plot coherence between every pair of tickers in a matrix grid.
    """

    N = len(tickers)
    fig, axes = plt.subplots(
        N, N, figsize=(3 * N, 3 * N), sharex=True, sharey=True
    )

    for i in range(N):
        for j in range(N):
            ax = axes[i, j]

            if i == j:
                # Diagonal: label self
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
                ax.set_axis_off()
                continue

            # Coherence formula
            S_ii = S[:, i, i]
            S_jj = S[:, j, j]
            S_ij = S[:, i, j]

            coherence = np.abs(S_ij) ** 2 / (S_ii * S_jj)

            ax.plot(freqs, coherence, lw=1.2)
            ax.set_ylim(0, 1.05)
            ax.grid(ls=":", lw=0.8)

            if i == N - 1:
                ax.set_xlabel(tickers[j])
            if j == 0:
                ax.set_ylabel(tickers[i])

    fig.suptitle("Coherence Between All Pairs of Stocks", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "KO", "PEP"]
    start = "2010-01-01"
    end = "2020-01-01"

    # 1. Download prices
    prices = download_prices(tickers, start, end)

    # 2. Convert to stationary series
    log_returns = make_log_returns(prices)

    # 3. Plot time-domain
    plot_timeseries(log_returns)

    # 4. Estimate spectral matrix
    freqs, S = estimate_spectral_matrix(log_returns, fs=1.0, nperseg=256)

    # 5. Plot PSD in period domain
    plot_psds_period(freqs, S, tickers)

    # Additional examples:
    # plot_csd(freqs, S, tickers, "AAPL", "MSFT")
    # plot_coherence(freqs, S, tickers, "AAPL", "MSFT")

    print("\nSpectral matrix shape:", S.shape)
    print("Example S(f0) matrix:")
    print(S[0])

    plot_all_coherences(freqs, S, tickers)
