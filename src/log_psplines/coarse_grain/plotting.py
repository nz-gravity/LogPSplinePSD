"""Plotting helpers for coarse-grained periodograms."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .preprocess import CoarseGrainSpec, apply_coarse_graining_univar


def plot_coarse_vs_original(
    freqs: np.ndarray,
    power: np.ndarray,
    spec: CoarseGrainSpec,
    transition_freq: float = None,
    scaling_factor: float = 1.0,
    *,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Plot original and coarse-grained periodograms for visual comparison."""

    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    selected_power = power[spec.selection_mask]
    selected_freqs = freqs[spec.selection_mask]
    coarse_power, weights = apply_coarse_graining_univar(
        selected_power, spec, selected_freqs
    )

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
    return fig, ax, weights


def plot_coarse_grain_weights(
    spec: CoarseGrainSpec,
    weights: np.ndarray,
    transition_freq: float = None,
    *,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the frequency weights used in coarse-graining for diagnostics."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot weights vs coarse-grained frequencies
    ax.semilogx(
        spec.f_coarse,
        weights,
        "-",
        color="#2ca02c",
        linewidth=2,
        markersize=4,
        label="Coarse-grain weights",
    )

    # Add reference line for uniform weights
    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Uniform weights (reference)",
    )

    if transition_freq is not None:
        ax.axvline(
            transition_freq,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Transition freq. ({transition_freq:.2e} Hz)",
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Weight")
    ax.set_title("Coarse-Graining Frequency Weights")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax
