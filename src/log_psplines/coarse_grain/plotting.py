"""Plotting helpers for coarse-grained periodograms."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .preprocess import CoarseGrainSpec, compute_gaussian_bin_statistics


def plot_coarse_vs_original(
    freqs: np.ndarray,
    power: np.ndarray,
    spec: CoarseGrainSpec,
    transition_freq: float = None,
    scaling_factor: float = 1.0,
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot original and coarse-grained periodograms for visual comparison."""

    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    _, coarse_power, _ = compute_gaussian_bin_statistics(freqs, power, spec)

    n_orig = freqs.size
    n_coarse = spec.f_coarse.size

    ax.loglog(
        freqs,
        power * scaling_factor,
        color="#1f77b4",
        alpha=0.6,
        label=f"Original [n={n_orig}]",
    )
    ax.loglog(
        spec.f_coarse,
        coarse_power * scaling_factor,
        linestyle="-",
        color="#d62728",
        label=f"Coarse-grained [n={n_coarse}]",
    )
    if transition_freq is not None:
        ax.axvline(
            transition_freq,
            color="gray",
            linestyle="--",
            label=f"Transition freq. ({transition_freq:.2e} Hz)",
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.legend()
    fig.tight_layout()
    return fig, ax
