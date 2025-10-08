"""Plotting helpers for coarse-grained periodograms."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..datatypes import MultivarFFT
from ..datatypes.multivar import EmpiricalPSD
from ..plotting import plot_psd_matrix
from .multivar import apply_coarse_graining_multivar_fft
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


def _empirical_to_ci(emp: EmpiricalPSD, show_coherence: bool) -> dict:
    n_channels = emp.psd.shape[1]
    ci: dict = {"psd": {}}
    for i in range(n_channels):
        diag = emp.psd[:, i, i].real
        ci["psd"][(i, i)] = (diag, diag, diag)

    if show_coherence:
        coh_dict = {}
        for i in range(1, n_channels):
            for j in range(i):
                coh_vals = emp.coherence[:, i, j]
                coh_dict[(i, j)] = (coh_vals, coh_vals, coh_vals)
        ci["coh"] = coh_dict
    else:
        re_dict = {}
        im_dict = {}
        for i in range(n_channels):
            for j in range(i):
                re_vals = emp.psd[:, i, j].real
                im_vals = emp.psd[:, i, j].imag
                re_dict[(i, j)] = (re_vals, re_vals, re_vals)
                im_dict[(i, j)] = (im_vals, im_vals, im_vals)
        ci["re"] = re_dict
        ci["im"] = im_dict
    return ci


def plot_coarse_vs_original_multivar(
    fft: MultivarFFT,
    spec: CoarseGrainSpec,
    *,
    show_coherence: bool = True,
    transition_freq: float | None = None,
    channel_labels: Sequence[str] | None = None,
) -> tuple[plt.Figure, Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Compare multivariate FFT data before/after coarse graining.

    Produces a two-panel figure using :func:`plot_psd_matrix` for the original
    and coarse-grained empirical PSD matrices. The coarse-grained FFT is
    generated via :func:`apply_coarse_graining_multivar_fft`, and per-bin weights
    are returned for downstream use (e.g. likelihood weighting).
    """

    coarse_result = apply_coarse_graining_multivar_fft(fft, spec)

    empirical_full = fft.empirical_psd
    empirical_coarse = coarse_result.fft.empirical_psd

    ci_full = _empirical_to_ci(empirical_full, show_coherence)
    ci_coarse = _empirical_to_ci(empirical_coarse, show_coherence)

    n_channels = empirical_full.psd.shape[1]
    fig = plt.figure(figsize=(7.8 * n_channels, 3.9 * n_channels))
    subfigs = fig.subfigures(1, 2, wspace=0.08)

    axes_full = subfigs[0].subplots(n_channels, n_channels)
    axes_coarse = subfigs[1].subplots(n_channels, n_channels)

    plot_psd_matrix(
        ci_dict=ci_full,
        freq=empirical_full.freq,
        empirical_psd=None,
        true_psd=None,
        channel_labels=channel_labels,
        show_coherence=show_coherence,
        diag_yscale="log",
        xscale="linear",
        fig=subfigs[0],
        axes=axes_full,
        save=False,
    )

    plot_psd_matrix(
        ci_dict=ci_coarse,
        freq=empirical_coarse.freq,
        empirical_psd=None,
        true_psd=None,
        channel_labels=channel_labels,
        show_coherence=show_coherence,
        diag_yscale="log",
        xscale="linear",
        fig=subfigs[1],
        axes=axes_coarse,
        save=False,
    )

    if transition_freq is not None:
        for ax_grid in (axes_full, axes_coarse):
            for ax_row in np.atleast_2d(ax_grid):
                for ax in np.atleast_1d(ax_row):
                    if ax.axison:
                        ax.axvline(
                            transition_freq,
                            color="gray",
                            linestyle="--",
                            alpha=0.6,
                        )

    subfigs[0].suptitle("Original (full resolution)", fontsize=14)
    subfigs[1].suptitle("Coarse-grained", fontsize=14)
    fig.tight_layout()
    return fig, (axes_full, axes_coarse), coarse_result.weights
