"""Utilities to compare PSD-implied moments against empirical time-domain moments."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _validate_freqs(freqs: np.ndarray) -> np.ndarray:
    freqs_arr = np.asarray(freqs, dtype=float)
    if freqs_arr.ndim != 1:
        raise ValueError("freqs must be one-dimensional")
    if np.any(np.diff(freqs_arr) <= 0):
        raise ValueError("freqs must be strictly increasing")
    return freqs_arr


def compute_psd_variances(
    psd_samples: np.ndarray, freqs: Sequence[float]
) -> np.ndarray:
    """Integrate diagonal PSD samples to obtain variance per channel.

    Parameters
    ----------
    psd_samples:
        Array with shape ``(n_samples, n_channels, n_freqs)`` containing one-sided
        PSD samples for each channel.
    freqs:
        Monotonically increasing frequency grid corresponding to the last axis of
        ``psd_samples``.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_samples, n_channels)`` containing PSD-implied
        variances for each posterior draw and channel.
    """

    psd_arr = np.asarray(psd_samples, dtype=float)
    if psd_arr.ndim != 3:
        raise ValueError(
            "psd_samples must have shape (n_samples, n_channels, n_freqs)"
        )

    freqs_arr = _validate_freqs(np.asarray(freqs, dtype=float))
    if freqs_arr.shape[0] != psd_arr.shape[-1]:
        raise ValueError("freqs length must match the PSD frequency dimension")

    variances = np.trapz(psd_arr, freqs_arr, axis=-1)
    return variances


def compute_psd_covariances(
    psd_samples: np.ndarray,
    freqs: Sequence[float],
    channel_pairs: Iterable[Tuple[int, int]],
) -> np.ndarray:
    """Integrate cross-PSDs to obtain covariance samples for channel pairs.

    Parameters
    ----------
    psd_samples:
        Array with shape ``(n_samples, n_channels, n_channels, n_freqs)`` containing
        one-sided cross-PSDs. Only the real part contributes to the covariance.
    freqs:
        Monotonically increasing frequency grid corresponding to the last axis of
        ``psd_samples``.
    channel_pairs:
        Iterable of ``(i, j)`` tuples indicating which covariances to extract.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_samples, n_pairs)`` containing PSD-implied covariances
        for each requested pair and posterior draw.
    """

    psd_arr = np.asarray(psd_samples)
    if psd_arr.ndim != 4:
        raise ValueError(
            "psd_samples must have shape (n_samples, n_channels, n_channels, n_freqs)"
        )

    freqs_arr = _validate_freqs(np.asarray(freqs, dtype=float))
    if freqs_arr.shape[0] != psd_arr.shape[-1]:
        raise ValueError("freqs length must match the PSD frequency dimension")

    pairs = list(channel_pairs)
    covariances = []
    for i, j in pairs:
        cross_psd = np.real(psd_arr[:, i, j, :])
        covariances.append(np.trapezoid(cross_psd, freqs_arr, axis=-1))

    return np.stack(covariances, axis=-1)


def compute_empirical_variances(data: np.ndarray) -> np.ndarray:
    """Sample variances for each channel from raw time-series data."""

    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2:
        raise ValueError("data must have shape (n_timesteps, n_channels)")

    return np.var(data_arr, axis=0, ddof=1)


def compute_empirical_covariances(
    data: np.ndarray, channel_pairs: Iterable[Tuple[int, int]]
) -> np.ndarray:
    """Sample covariances for specified channel pairs from raw time-series data."""

    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2:
        raise ValueError("data must have shape (n_timesteps, n_channels)")

    cov_matrix = np.cov(data_arr, rowvar=False, ddof=1)
    pairs = list(channel_pairs)
    return np.array([cov_matrix[i, j] for i, j in pairs], dtype=float)


def plot_psd_vs_empirical_moments(
    posterior_samples: np.ndarray,
    empirical_values: np.ndarray | float,
    credible_fraction: float,
) -> plt.Figure:
    """Plot posterior moment samples against empirical estimates.

    The posterior is visualised as a histogram with a shaded credible interval, and
    a vertical line indicates the empirical value.
    """

    samples = np.asarray(posterior_samples, dtype=float).ravel()
    empirical = float(np.asarray(empirical_values).squeeze())

    if not (0.0 < credible_fraction < 1.0):
        raise ValueError("credible_fraction must be in (0, 1)")

    lower_q = (1.0 - credible_fraction) / 2.0
    upper_q = 1.0 - lower_q
    lower, upper = np.quantile(samples, [lower_q, upper_q])

    fig, ax = plt.subplots()
    ax.hist(
        samples,
        bins=30,
        density=True,
        alpha=0.6,
        color="C0",
        label="Posterior",
    )
    ax.axvspan(
        lower,
        upper,
        color="C0",
        alpha=0.2,
        label=f"{credible_fraction:.0%} CI",
    )
    ax.axvline(empirical, color="C1", linestyle="--", label="Empirical")

    ax.set_xlabel("Moment")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


__all__ = [
    "compute_psd_variances",
    "compute_psd_covariances",
    "compute_empirical_variances",
    "compute_empirical_covariances",
    "plot_psd_vs_empirical_moments",
]
