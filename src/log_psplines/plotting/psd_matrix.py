"""Stationary PSD matrix plots from the native fitted result."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from log_psplines.data.spectral import EmpiricalPSD
from log_psplines.models.matrix import SpectralMatrix

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


@dataclass(frozen=True)
class PSDMatrixPlotSpec:
    """Rendering options for a stationary ``PSDResult``."""

    result: PSDResult
    true_psd: np.ndarray | None = None  # (frequency, channel, channel_aux)
    extra_empirical_psd: tuple[EmpiricalPSD, ...] = ()
    extra_empirical_labels: tuple[str, ...] = ()
    extra_empirical_styles: tuple[dict[str, Any], ...] = ()
    knot_frequencies: np.ndarray | None = None  # explicit frequencies for S_00
    outdir: str | Path = "."
    filename: str = "psd_matrix.png"
    dpi: int = 150
    show_coherence: bool = True
    show_csd_magnitude: bool = False
    channel_labels: list[str] | str | None = None
    diag_yscale: str = "log"
    offdiag_yscale: str = "linear"
    xscale: str = "linear"
    label: str | None = None
    model_color: str = "tab:blue"
    show_empirical: bool = True
    fig: plt.Figure | None = None
    ax: np.ndarray | plt.Axes | None = None
    save: bool = True
    close: bool | None = None
    overlay_vi: bool = False
    vi_color: str = "tab:orange"
    vi_label: str = "VI median"
    vi_alpha: float = 0.2
    freq_range: tuple[float, float] | None = None
    excluded_bands: tuple[tuple[float, float], ...] = ()
    psd_scale: (
        np.ndarray | float | Callable[[np.ndarray], np.ndarray] | None
    ) = None
    psd_unit_label: str = "1/Hz"


def _spectral_quantiles(
    samples: np.ndarray, kind: str, i: int, j: int
) -> np.ndarray:
    """Return 5/50/95 curves for one panel from (chain, draw, F, C, C)."""
    if kind == "coherence":
        values = np.asarray(SpectralMatrix.coherence(samples))[..., i, j]
    else:
        values = samples[..., i, j]
        if kind == "real":
            values = values.real
        elif kind == "imag":
            values = values.imag
        elif kind == "magnitude":
            values = np.abs(values)
        else:
            raise ValueError(f"Unknown panel kind: {kind}")
    return np.percentile(
        values.reshape(-1, values.shape[-1]), (5, 50, 95), axis=0
    )


def _panel_kind(i: int, j: int, spec: PSDMatrixPlotSpec) -> str:
    if i == j:
        return "real"
    if spec.show_coherence:
        return "coherence"
    if spec.show_csd_magnitude:
        return "magnitude"
    return "real" if i > j else "imag"


def _resolve_scale(
    spec: PSDMatrixPlotSpec, frequency: np.ndarray
) -> np.ndarray:
    source = spec.psd_scale
    if source is None:
        return np.ones_like(frequency)
    values = source(frequency) if callable(source) else source
    scale = np.asarray(values, dtype=float)
    if scale.ndim == 0:
        scale = np.full_like(frequency, float(scale))
    if scale.shape != frequency.shape:
        raise ValueError("psd_scale must match frequency shape.")
    if np.any(scale < 0):
        raise ValueError("psd_scale must be non-negative.")
    if np.any(scale == 0):
        if np.any(frequency[scale == 0] != 0):
            raise ValueError("psd_scale has zeros at nonzero frequencies.")
        scale = scale.copy()
        scale[scale == 0] = np.nan
    return scale


def _plot_band(
    ax: plt.Axes,
    frequency: np.ndarray,
    curves: np.ndarray,
    *,
    color: str,
    label: str | None,
    alpha: float = 0.25,
    style: str = "-",
) -> None:
    ax.fill_between(frequency, curves[0], curves[2], color=color, alpha=alpha)
    ax.plot(frequency, curves[1], color=color, lw=1.5, ls=style, label=label)


def _observed_periodogram(result: PSDResult) -> EmpiricalPSD | None:
    observed = result.observed_data
    if observed is None or "periodogram" not in observed:
        return None
    periodogram = observed["periodogram"]
    if periodogram.dims != ("frequency", "channel", "channel_aux"):
        raise ValueError(
            "observed periodogram must have frequency and matrix dimensions"
        )
    values = np.asarray(periodogram.values)
    return EmpiricalPSD(
        freq=np.asarray(periodogram.coords["frequency"].values),
        psd=values,
        coherence=np.asarray(SpectralMatrix.coherence(values)),
        channels=np.asarray(periodogram.coords["channel"].values),
    )


def _empirical_curve(
    empirical: EmpiricalPSD, kind: str, i: int, j: int
) -> np.ndarray:
    if kind == "coherence":
        return np.asarray(empirical.coherence[:, i, j])
    values = empirical.psd[:, i, j]
    if kind == "real":
        return np.asarray(values.real)
    if kind == "imag":
        return np.asarray(values.imag)
    return np.abs(values)


def _render_empirical_panel(
    ax: plt.Axes,
    spec: PSDMatrixPlotSpec,
    empirical: EmpiricalPSD | None,
    kind: str,
    i: int,
    j: int,
    scale: np.ndarray,
) -> None:
    if not spec.show_empirical:
        return
    series = (() if empirical is None else (empirical,)) + tuple(
        spec.extra_empirical_psd
    )
    for index, item in enumerate(series):
        item_freq = np.asarray(item.freq)
        curve = _empirical_curve(item, kind, i, j)
        if kind != "coherence":
            item_scale = np.interp(item_freq, spec.result.frequency, scale)
            curve = curve * item_scale
        style: dict[str, Any] = {
            "color": "0.4",
            "lw": 1.0,
            "alpha": 0.3,
            "ls": "--",
            "zorder": -5,
            "label": "Empirical",
        }
        extra_index = index - (empirical is not None)
        if extra_index >= 0:
            if extra_index < len(spec.extra_empirical_styles):
                style.update(spec.extra_empirical_styles[extra_index])
            style["label"] = (
                spec.extra_empirical_labels[extra_index]
                if extra_index < len(spec.extra_empirical_labels)
                else f"Empirical {extra_index + 2}"
            )
        ax.plot(item_freq, curve, **style)


def _truth_curve(truth: np.ndarray, kind: str, i: int, j: int) -> np.ndarray:
    if kind == "coherence":
        return np.abs(truth[:, i, j]) ** 2 / (
            truth[:, i, i].real * truth[:, j, j].real
        )
    values = truth[:, i, j]
    if kind == "real":
        return values.real
    if kind == "imag":
        return values.imag
    return np.abs(values)


def _render_panel(
    ax: plt.Axes,
    spec: PSDMatrixPlotSpec,
    frequency: np.ndarray,
    posterior: np.ndarray,
    vi: np.ndarray | None,
    empirical: EmpiricalPSD | None,
    truth: np.ndarray | None,
    scale: np.ndarray,
    kind: str,
    i: int,
    j: int,
) -> None:
    for band_index, (low, high) in enumerate(spec.excluded_bands):
        ax.axvspan(
            min(low, high),
            max(low, high),
            color="#d8b365",
            alpha=0.24,
            zorder=-20,
            label="Excluded band" if i == j == band_index == 0 else None,
        )
    curves = _spectral_quantiles(posterior, kind, i, j)
    if kind != "coherence":
        curves = curves * scale
    _render_empirical_panel(ax, spec, empirical, kind, i, j, scale)
    _plot_band(
        ax,
        frequency,
        curves,
        color=spec.model_color,
        label=(
            spec.label or ("Posterior median" if vi is not None else "Median")
        )
        if i == j == 0
        else None,
    )
    if truth is not None:
        truth_values = _truth_curve(truth, kind, i, j)
        if kind != "coherence":
            truth_values = truth_values * scale
        ax.plot(
            frequency,
            truth_values,
            color="k",
            lw=1.6,
            label="Analytical" if i == j == 0 else None,
            zorder=6,
        )
    if vi is not None:
        vi_curves = _spectral_quantiles(vi, kind, i, j)
        if kind != "coherence":
            vi_curves = vi_curves * scale
        _plot_band(
            ax,
            frequency,
            vi_curves,
            color=spec.vi_color,
            label=spec.vi_label if i == j == 0 else None,
            alpha=spec.vi_alpha,
            style="--",
        )
    if i == j == 0 and spec.knot_frequencies is not None:
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.vlines(
            spec.knot_frequencies,
            0.02,
            0.065,
            transform=transform,
            colors="tab:green",
            linewidth=1.7,
            zorder=9,
        )
    ax.set_yscale(spec.diag_yscale if i == j else spec.offdiag_yscale)
    if kind == "coherence":
        ax.set_ylim(0, 1)
    if i == j == 0:
        ax.legend(frameon=False, fontsize=9)


def _panel_label(kind: str, i: int, j: int, labels: list[str]) -> str:
    left, right = labels[min(i, j)], labels[max(i, j)]
    name = f"S_{{{left}{right}}}"
    if kind == "coherence":
        return f"$C_{{{left}{right}}}$"
    if kind == "magnitude":
        return f"$|{name}|$"
    if i == j:
        return f"${name}$"
    return f"${'Re' if kind == 'real' else 'Im'}({name})$"


def plot_psd_matrix(spec: PSDMatrixPlotSpec) -> tuple[plt.Figure, np.ndarray]:
    """Plot a stationary scalar PSD or spectral matrix from ``spec.result``."""
    if spec.show_coherence and spec.show_csd_magnitude:
        raise ValueError(
            "Choose either coherence display or |CSD| magnitude, not both."
        )
    result = spec.result
    if result.time is not None:
        raise ValueError("plot_psd_matrix requires a stationary PSDResult")
    spectrum = result.spectrum
    expected = ("chain", "draw", "frequency", "channel", "channel_aux")
    if spectrum.dims != expected:
        raise ValueError(f"spectrum dimensions must be {expected}")
    posterior = np.asarray(spectrum.values)
    frequency = result.frequency
    channels = posterior.shape[-1]
    if posterior.shape[-2] != channels:
        raise ValueError("spectrum matrix must be square")
    vi = None
    if spec.overlay_vi and result.vi_spectrum is not None:
        if (
            result.vi_spectrum.dims != expected
            or result.vi_spectrum.shape[2:] != posterior.shape[2:]
        ):
            raise ValueError(
                "VI spectrum must match posterior spectral dimensions"
            )
        vi = np.asarray(result.vi_spectrum.values)
    scale = _resolve_scale(spec, frequency)
    truth = None
    if spec.true_psd is not None:
        truth = np.asarray(spec.true_psd)
        if truth.ndim == 1 and channels == 1:
            truth = truth[:, None, None]
        if truth.shape != posterior.shape[2:]:
            raise ValueError(
                "true_psd must have shape (frequency, channel, channel_aux)"
            )
    empirical = _observed_periodogram(result)
    labels = spec.channel_labels
    if labels is None:
        labels = [str(i + 1) for i in range(channels)]
    elif isinstance(labels, str):
        labels = list(labels)
    if len(labels) != channels:
        raise ValueError("channel_labels must match the number of channels")

    created = spec.fig is None and spec.ax is None
    if created:
        fig, axes_value = plt.subplots(
            channels,
            channels,
            figsize=(3.9 * channels, 3.9 * channels),
            dpi=spec.dpi,
        )
    elif spec.fig is not None and spec.ax is not None:
        fig, axes_value = spec.fig, spec.ax
    else:
        raise ValueError("fig and ax must be supplied together")
    axes = np.asarray(axes_value, dtype=object).reshape(channels, channels)
    if axes.shape != (channels, channels):
        raise ValueError("Provided axes must match the channel matrix shape")
    for i in range(channels):
        for j in range(channels):
            ax = axes[i, j]
            if i < j and (spec.show_coherence or spec.show_csd_magnitude):
                ax.axis("off")
                continue
            kind = _panel_kind(i, j, spec)
            ax.set_xscale(spec.xscale)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            _render_panel(
                ax,
                spec,
                frequency,
                posterior,
                vi,
                empirical,
                truth,
                scale,
                kind,
                i,
                j,
            )
            if i == j:
                ylabel = f"PSD [{spec.psd_unit_label}]"
            elif kind == "coherence":
                ylabel = "Coherence"
            elif kind == "magnitude":
                ylabel = f"|CSD| [{spec.psd_unit_label}]"
            else:
                ylabel = f"{kind.title()}[CSD] [{spec.psd_unit_label}]"
            ax.set_ylabel(ylabel)
            if i == channels - 1:
                ax.set_xlabel("Frequency [Hz]")
            ax.text(
                0.96,
                0.95,
                _panel_label(kind, i, j, labels),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10.5,
                weight="bold",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "0.0",
                    "alpha": 0.92,
                },
            )
            if spec.freq_range is not None:
                ax.set_xlim(spec.freq_range)
    if created:
        fig.subplots_adjust(
            left=0.12,
            right=0.98,
            top=0.98,
            bottom=0.10,
            wspace=0.30,
            hspace=0.30,
        )
        if spec.save:
            destination = Path(spec.outdir) / spec.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=spec.dpi, bbox_inches="tight")
    if spec.close if spec.close is not None else (created and spec.save):
        plt.close(fig)
    return fig, axes
