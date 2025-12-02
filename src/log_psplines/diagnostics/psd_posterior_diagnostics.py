"""Frequency-domain diagnostics for PSD posteriors."""

from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _validate_psd_array(psd: np.ndarray) -> np.ndarray:
    arr = np.asarray(psd, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            "PSD arrays must have shape (n_channels, n_freqs) or (n_freqs,)"
        )
    return arr


def compute_psd_credible_bands(
    psd_samples: np.ndarray, credible_fraction: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute posterior median PSD and central credible interval bands.

    Parameters
    ----------
    psd_samples : np.ndarray
        Array of posterior PSD samples with shape ``(n_samples, n_channels, n_freqs)``.
    credible_fraction : float
        Credible mass to include inside the interval (e.g. ``0.95`` for a 95% band).

    Returns
    -------
    tuple of np.ndarray
        ``(median, lower, upper)`` arrays each with shape ``(n_channels, n_freqs)``.
    """

    samples = np.asarray(psd_samples, dtype=float)
    if samples.ndim != 3:
        raise ValueError(
            "psd_samples must have shape (n_samples, n_channels, n_freqs)"
        )

    lower_q = (1.0 - credible_fraction) / 2.0 * 100.0
    upper_q = (1.0 + credible_fraction) / 2.0 * 100.0

    median = np.median(samples, axis=0)
    lower = np.percentile(samples, lower_q, axis=0)
    upper = np.percentile(samples, upper_q, axis=0)

    return median, lower, upper


def compute_welch_coverage(
    welch_psd: np.ndarray, bands: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Compute the per-channel coverage of Welch PSD within posterior bands."""

    welch_arr = _validate_psd_array(welch_psd)
    try:
        _, lower, upper = bands
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            "bands must be a (median, lower, upper) tuple"
        ) from exc

    lower_arr = _validate_psd_array(lower)
    upper_arr = _validate_psd_array(upper)

    if not (welch_arr.shape == lower_arr.shape == upper_arr.shape):
        raise ValueError("Welch PSD and band arrays must share the same shape")

    within = (welch_arr >= lower_arr) & (welch_arr <= upper_arr)
    return np.mean(within, axis=1)


def compute_riae(
    psd_est: np.ndarray, psd_ref: np.ndarray, freqs: Iterable[float]
) -> np.ndarray:
    """Compute relative integrated absolute error (RIAE).

    The RIAE is defined as ``∫|psd_est - psd_ref| df / ∫ psd_ref df`` computed
    independently for each channel.
    """

    est = _validate_psd_array(psd_est)
    ref = _validate_psd_array(psd_ref)
    if est.shape != ref.shape:
        raise ValueError("psd_est and psd_ref must share the same shape")

    freqs_arr = np.asarray(freqs, dtype=float)
    if freqs_arr.ndim != 1 or freqs_arr.shape[0] != est.shape[-1]:
        raise ValueError(
            "freqs must be 1-D with length matching the PSD frequency axis"
        )

    numerator = np.trapz(np.abs(est - ref), freqs_arr, axis=-1)
    denominator = np.trapz(ref, freqs_arr, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        riae = numerator / denominator
    return riae


def plot_psd_with_bands_and_welch(
    freqs: Iterable[float],
    psd_median: np.ndarray,
    psd_lower: np.ndarray,
    psd_upper: np.ndarray,
    welch_psd: np.ndarray,
    *,
    channel_names: Iterable[str] | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot posterior PSD bands alongside Welch estimates."""

    freqs_arr = np.asarray(freqs, dtype=float)
    median_arr = _validate_psd_array(psd_median)
    lower_arr = _validate_psd_array(psd_lower)
    upper_arr = _validate_psd_array(psd_upper)
    welch_arr = _validate_psd_array(welch_psd)

    if not (
        median_arr.shape
        == lower_arr.shape
        == upper_arr.shape
        == welch_arr.shape
        == (median_arr.shape[0], freqs_arr.shape[0])
    ):
        raise ValueError(
            "All PSD inputs must share shape (n_channels, n_freqs)"
        )

    n_channels = median_arr.shape[0]
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(n_channels)]
    else:
        channel_names = list(channel_names)
        if len(channel_names) != n_channels:
            raise ValueError(
                "channel_names length must match number of channels"
            )

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)

    for idx, ax in enumerate(axes):
        ax.loglog(
            freqs_arr, welch_arr[idx], label="Welch PSD", color="C1", ls="--"
        )
        ax.loglog(
            freqs_arr, median_arr[idx], label="Posterior median", color="C0"
        )
        ax.fill_between(
            freqs_arr,
            lower_arr[idx],
            upper_arr[idx],
            color="C0",
            alpha=0.2,
            label="Credible band" if idx == 0 else None,
        )
        ax.set_ylabel(channel_names[idx])
        ax.grid(True, which="both", ls=":", alpha=0.6)

    axes[-1].set_xlabel("Frequency [Hz]")
    axes[0].legend()
    fig.tight_layout()
    return fig


__all__ = [
    "compute_psd_credible_bands",
    "compute_welch_coverage",
    "compute_riae",
    "plot_psd_with_bands_and_welch",
]
