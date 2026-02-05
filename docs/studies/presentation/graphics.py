from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc
from log_psplines.plotting.base import extract_plotting_data

# -------------------------------------------------------------
# SETUP
# -------------------------------------------------------------
plt.style.use("seaborn-v0_8-poster")

OUTDIR = Path("out_figs")
OUTDIR.mkdir(exist_ok=True)

varma_data = VARMAData(fs=1, n_samples=2048)
pdgrm = varma_data.get_periodogram()

# Convert time → years
t_years = varma_data.time / 365.25

# Convert frequency → cycles/year (copy to avoid later mutations)
freq_hz = np.asarray(varma_data.freq, dtype=float).copy()
freq_cpy = freq_hz * 365.25


# -------------------------------------------------------------
# STYLE HELPERS
# -------------------------------------------------------------
def clean_axes(ax, color=None):
    ax.grid(which="major", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.minorticks_off()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12, width=1.5)
    if color is not None:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)


def _closest_percentile(percentiles: np.ndarray, target: float) -> int:
    percentiles = np.asarray(percentiles, dtype=float)
    return int(np.argmin(np.abs(percentiles - target)))


def _make_psd_panel(
    freq_axis: np.ndarray,
    pdgrm: np.ndarray,
    *,
    data_alpha: float = 0.85,
    linewidth: float = 1.5,
):
    fig, ax = plt.subplots(2, 1, figsize=(4, 5), sharex=True)
    for i in range(2):
        ax[i].semilogy(
            freq_axis,
            pdgrm[:, i, i].real,
            color=f"C{i}",
            lw=linewidth,
            alpha=data_alpha,
        )
        ax[i].set_ylabel("PSD")
        ax[i].text(
            0.03,
            0.9,
            f"Device {i+1}",
            transform=ax[i].transAxes,
            fontsize=16,
            fontweight="bold",
            color=f"C{i}",
        )
        clean_axes(ax[i])
    ax[-1].set_xlabel("Frequency [cycles/year]")
    ax[-1].set_xlim(0, freq_axis.max())
    fig.subplots_adjust(hspace=0.05)
    return fig, ax


def _make_psd_csd_panel(
    freq_axis: np.ndarray,
    pdgrm: np.ndarray,
    *,
    data_alpha: float = 0.85,
    linewidth: float = 1.5,
):
    psd1 = pdgrm[:, 0, 0].real
    psd2 = pdgrm[:, 1, 1].real
    csd = pdgrm[:, 0, 1]

    fig, ax = plt.subplots(3, 1, figsize=(4, 7), sharex=True)
    ax[0].semilogy(freq_axis, psd1, color="C0", lw=linewidth, alpha=data_alpha)
    ax[0].set_ylabel("PSD")
    ax[0].text(
        0.03,
        0.9,
        "Device 1",
        transform=ax[0].transAxes,
        color="C0",
        fontsize=16,
        fontweight="bold",
    )
    clean_axes(ax[0])

    ax[1].semilogy(freq_axis, psd2, color="C1", lw=linewidth, alpha=data_alpha)
    ax[1].set_ylabel("PSD")
    ax[1].text(
        0.03,
        0.9,
        "Device 2",
        transform=ax[1].transAxes,
        color="C1",
        fontsize=16,
        fontweight="bold",
    )
    clean_axes(ax[1])

    ax[2].plot(
        freq_axis,
        np.real(csd),
        color="C2",
        lw=linewidth,
        alpha=data_alpha,
        label="Re",
    )
    ax[2].plot(
        freq_axis,
        np.imag(csd),
        color="C3",
        lw=linewidth,
        alpha=data_alpha,
        linestyle="--",
        label="Im",
    )
    ax[2].set_ylabel("CSD")
    ax[2].text(
        0.03,
        0.9,
        "CSD(1,2)",
        transform=ax[2].transAxes,
        color="C2",
        fontsize=16,
        fontweight="bold",
    )
    ax[2].legend(frameon=False, fontsize=12)
    clean_axes(ax[2])
    ax[2].set_xlabel("Frequency [cycles/year]")
    ax[2].set_xlim(0, freq_axis.max())
    fig.subplots_adjust(hspace=0.05)
    return fig, ax


def _make_coherence_axis(
    freq_axis: np.ndarray,
    coherence: np.ndarray,
    *,
    data_alpha: float = 0.9,
    linewidth: float = 1.8,
):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(freq_axis, coherence, color="C4", lw=linewidth, alpha=data_alpha)
    ax.set_xlabel("Frequency [cycles/year]")
    ax.set_ylabel("Coherence")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, freq_axis.max())
    ax.text(
        0.03,
        0.1,
        "Device 1 ↔ Device 2",
        transform=ax.transAxes,
        fontsize=15,
        color="C4",
        fontweight="bold",
    )
    clean_axes(ax)
    return fig, ax


# -------------------------------------------------------------
# 1. TIMESERIES PLOT (2 rows)
# -------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

for i in range(2):
    ax[i].plot(
        t_years,
        varma_data.data[:, i],
        color=f"C{i}",
        lw=1.5,
        alpha=0.85,
    )

    ax[i].set_ylabel("y(t)")
    ax[i].text(
        0.03,
        0.9,
        f"Device {i+1}",
        transform=ax[i].transAxes,
        fontsize=16,
        fontweight="bold",
        color=f"C{i}",
    )

    clean_axes(ax[i])

ax[-1].set_xlabel("Time [years]")
ax[-1].set_xlim(0, t_years[-1])

fig.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig(OUTDIR / "varma_timeseries.png", dpi=150)


# -------------------------------------------------------------
# 2. PSDs ONLY (2 rows)
# -------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

for i in range(2):
    ax[i].semilogy(
        freq_cpy,
        pdgrm[:, i, i].real,
        color=f"C{i}",
        lw=1.5,
        alpha=0.85,
    )

    ax[i].set_ylabel("PSD")
    ax[i].text(
        0.03,
        0.9,
        f"Device {i+1}",
        transform=ax[i].transAxes,
        fontsize=16,
        fontweight="bold",
        color=f"C{i}",
    )

    clean_axes(ax[i])

ax[-1].set_xlabel("Frequency [cycles/year]")
ax[-1].set_xlim(0, freq_cpy.max())

fig.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig(OUTDIR / "varma_psds.png", dpi=150)


# -------------------------------------------------------------
# 3. PSD + CSD (3 rows)
# -------------------------------------------------------------
psd1 = pdgrm[:, 0, 0].real
psd2 = pdgrm[:, 1, 1].real
csd = pdgrm[:, 0, 1]

fig, ax = plt.subplots(3, 1, figsize=(4, 7), sharex=True)

# Row 1: PSD 1
ax[0].semilogy(freq_cpy, psd1, color="C0", lw=1.5, alpha=0.85)
ax[0].set_ylabel("PSD")
ax[0].text(
    0.03,
    0.9,
    "Device 1",
    transform=ax[0].transAxes,
    color="C0",
    fontsize=16,
    fontweight="bold",
)
clean_axes(ax[0])

# Row 2: PSD 2
ax[1].semilogy(freq_cpy, psd2, color="C1", lw=1.5, alpha=0.85)
ax[1].set_ylabel("PSD")
ax[1].text(
    0.03,
    0.9,
    "Device 2",
    transform=ax[1].transAxes,
    color="C1",
    fontsize=16,
    fontweight="bold",
)
clean_axes(ax[1])

# Row 3: CSD (real + imag)
ax[2].plot(
    freq_cpy,
    np.real(csd),
    color="C2",
    lw=1.5,
    alpha=0.9,
    label="Re",
)
ax[2].plot(
    freq_cpy,
    np.imag(csd),
    color="C3",
    lw=1.5,
    alpha=0.9,
    linestyle="--",
    label="Im",
)
ax[2].set_ylabel("CSD")
ax[2].text(
    0.03,
    0.9,
    "CSD(1,2)",
    transform=ax[2].transAxes,
    color="C2",
    fontsize=16,
    fontweight="bold",
)
ax[2].legend(frameon=False, fontsize=12)
clean_axes(ax[2])

ax[2].set_xlabel("Frequency [cycles/year]")
ax[2].set_xlim(0, freq_cpy.max())

fig.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig(OUTDIR / "varma_psd_csd.png", dpi=150)


# -------------------------------------------------------------
# 4. COHERENCE (1 row)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3))

f_hz, Cxy = sig.coherence(
    varma_data.data[:, 0],
    varma_data.data[:, 1],
    fs=1,
    nperseg=512,
)

f_cpy = f_hz * 365.25

ax.plot(f_cpy, Cxy, color="C4", lw=1.8, alpha=0.9)

ax.set_xlabel("Frequency [cycles/year]")
ax.set_ylabel("Coherence")
ax.set_ylim(0, 1)
ax.set_xlim(0, f_cpy.max())
ax.text(
    0.03,
    0.1,
    "Device 1 ↔ Device 2",
    transform=ax.transAxes,
    fontsize=15,
    color="C4",
    fontweight="bold",
)

clean_axes(ax)

plt.tight_layout()
plt.savefig(OUTDIR / "varma_coherence.png", dpi=150)


# -------------------------------------------------------------
# 5. MULTIVARIATE LOG P-SPLINE FIT (VI) + PSD/CSD/COHENCE
# -------------------------------------------------------------
def run_or_load_pspline_fit(
    data: VARMAData, cache_dir: Path
) -> az.InferenceData:
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "inference_data.nc"
    if cache_file.exists():
        return az.from_netcdf(cache_file)

    ts = MultivariateTimeseries(t=data.time, y=data.data)
    idata = run_mcmc(
        data=ts,
        sampler="multivar_blocked_nuts",
        n_knots=20,
        degree=3,
        diffMatrixOrder=2,
        n_samples=1000,
        n_warmup=1000,
        Nb=1,
        rng_key=321,
        only_vi=False,
        vi_steps=2000,
        vi_lr=3e-3,
        outdir=str(cache_dir),
        true_psd={"freq": data.freq, "psd": data.get_true_psd()},
        verbose=True,
    )
    idata.to_netcdf(cache_file)
    return idata


def extract_pspline_quantiles(
    idata: az.InferenceData,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    extracted = extract_plotting_data(idata)
    quant = extracted.get("posterior_psd_matrix_quantiles")
    if quant is None:
        raise RuntimeError("Inference data missing PSD/CSD quantiles.")

    freq = np.asarray(extracted.get("frequencies"), dtype=float)
    if freq.size == 0:
        raise RuntimeError("No frequency grid found in inference data.")

    coherence = quant.get("coherence")
    if coherence is not None:
        coherence = np.asarray(coherence)

    payload = {
        "percentiles": np.asarray(quant["percentile"], dtype=float),
        "psd_real": np.asarray(quant["real"], dtype=float),
        "psd_imag": np.asarray(quant["imag"], dtype=float),
        "coherence": coherence,
    }
    return freq, payload


def plot_pspline_psds(
    freq_hz_fit: np.ndarray,
    quant_data: Dict[str, np.ndarray],
    periodogram: np.ndarray,
    freq_cpy_data: np.ndarray,
    out_path: Path,
):
    freq_cpy_fit = freq_hz_fit * 365.25
    percentiles = quant_data["percentiles"]
    q_real = quant_data["psd_real"]
    idx_lo = _closest_percentile(percentiles, 5.0)
    idx_med = _closest_percentile(percentiles, 50.0)
    idx_hi = _closest_percentile(percentiles, 95.0)

    fig, axes = _make_psd_panel(
        freq_cpy_data,
        periodogram,
        data_alpha=0.3,
        linewidth=1.2,
    )
    axes[0].lines[0].set_label("Periodogram")
    axes[0].lines[0].set_label("Periodogram")
    colors = ["C0", "C1"]
    for ch in range(2):
        ax = axes[ch]
        base_color = colors[ch]
        lo = np.clip(q_real[idx_lo, :, ch, ch], 1e-12, None)
        med = np.clip(q_real[idx_med, :, ch, ch], 1e-12, None)
        hi = np.clip(q_real[idx_hi, :, ch, ch], 1e-12, None)
        ax.fill_between(
            freq_cpy_fit,
            lo,
            hi,
            color=base_color,
            alpha=0.18,
            lw=0,
            label="90% CI" if ch == 0 else None,
            zorder=1,
        )
        ax.semilogy(
            freq_cpy_fit,
            med,
            color=base_color,
            lw=2.4,
            label="P-spline median" if ch == 0 else None,
            zorder=2,
        )
        ax.set_ylabel(f"PSD (channel {ch+1})")
        clean_axes(ax, color=base_color)

    axes[-1].set_xlabel("Frequency [cycles/year]")
    axes[-1].set_xlim(0, freq_cpy_fit.max())
    axes[0].legend(frameon=False, fontsize=10)
    fig.subplots_adjust(hspace=0.07)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pspline_csd(
    freq_hz_fit: np.ndarray,
    quant_data: Dict[str, np.ndarray],
    freq_cpy_data: np.ndarray,
    periodogram: np.ndarray,
    out_path: Path,
):
    freq_cpy_fit = freq_hz_fit * 365.25
    percentiles = quant_data["percentiles"]
    q_real = quant_data["psd_real"]
    q_imag = quant_data["psd_imag"]
    idx_lo = _closest_percentile(percentiles, 5.0)
    idx_med = _closest_percentile(percentiles, 50.0)
    idx_hi = _closest_percentile(percentiles, 95.0)

    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    labels = ["Re CSD(1,2)", "Im CSD(1,2)"]
    colors = ["C2", "C3"]
    csd_data = periodogram[:, 0, 1]
    data_components = [csd_data.real, csd_data.imag]
    for row, (component, label, data_component, color) in enumerate(
        zip((q_real, q_imag), labels, data_components, colors)
    ):
        lo = component[idx_lo, :, 0, 1]
        med = component[idx_med, :, 0, 1]
        hi = component[idx_hi, :, 0, 1]

        ax = axes[row]
        ax.plot(
            freq_cpy_data,
            data_component,
            color=color,
            lw=1.0,
            alpha=0.3,
            linestyle="--" if row == 1 else "-",
            label="Periodogram" if row == 0 else None,
            zorder=0,
        )
        ax.fill_between(
            freq_cpy_fit,
            lo,
            hi,
            color=color,
            alpha=0.2,
            lw=0,
            zorder=1,
        )
        ax.plot(
            freq_cpy_fit,
            med,
            color=color,
            lw=2.2,
            linestyle="--" if row == 1 else "-",
            label="P-spline median" if row == 0 else None,
            zorder=2,
        )
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4, zorder=0)
        ax.set_ylabel(label)
        clean_axes(ax, color=color)

    axes[-1].set_xlabel("Frequency [cycles/year]")
    axes[-1].set_xlim(0, freq_cpy_fit.max())
    axes[0].legend(frameon=False, fontsize=10)
    fig.subplots_adjust(hspace=0.08)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pspline_coherence(
    freq_hz_fit: np.ndarray,
    quant_data: Dict[str, np.ndarray],
    freq_cpy_data: np.ndarray,
    coherence_data: np.ndarray,
    out_path: Path,
):
    coh = quant_data.get("coherence")
    if coh is None:
        return

    freq_cpy_fit = freq_hz_fit * 365.25
    percentiles = quant_data["percentiles"]
    idx_lo = _closest_percentile(percentiles, 5.0)
    idx_med = _closest_percentile(percentiles, 50.0)
    idx_hi = _closest_percentile(percentiles, 95.0)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(
        freq_cpy_data,
        coherence_data,
        color="C4",
        lw=1.2,
        alpha=0.35,
        label="Empirical coherence",
        zorder=0,
    )

    lo = np.clip(coh[idx_lo, :, 0, 1], 0.0, 1.0)
    med = np.clip(coh[idx_med, :, 0, 1], 0.0, 1.0)
    hi = np.clip(coh[idx_hi, :, 0, 1], 0.0, 1.0)

    ax.fill_between(
        freq_cpy_fit,
        lo,
        hi,
        color="C4",
        alpha=0.2,
        lw=0,
        label="90% CI",
        zorder=1,
    )
    ax.plot(
        freq_cpy_fit,
        med,
        color="C4",
        lw=2.5,
        alpha=0.9,
        label="P-spline median",
        zorder=2,
    )

    ax.set_xlabel("Frequency [cycles/year]")
    ax.set_ylabel("Coherence (1↔2)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, freq_cpy_fit.max())
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


pspline_cache = OUTDIR / "pspline_vi"
idata = run_or_load_pspline_fit(varma_data, pspline_cache)
freq_fit_hz, quant_payload = extract_pspline_quantiles(idata)
plot_pspline_psds(
    freq_fit_hz,
    quant_payload,
    pdgrm,
    freq_cpy,
    OUTDIR / "varma_pspline_psds.png",
)
plot_pspline_csd(
    freq_fit_hz,
    quant_payload,
    freq_cpy,
    pdgrm,
    OUTDIR / "varma_pspline_csd.png",
)
plot_pspline_coherence(
    freq_fit_hz,
    quant_payload,
    f_cpy,
    Cxy,
    OUTDIR / "varma_pspline_coherence.png",
)

# def plot_background_panel(
#     freq_hz_fit: np.ndarray,
#     quant_data: Dict[str, np.ndarray],
#     freq_cpy_data: np.ndarray,
#     periodogram: np.ndarray,
#     freq_coh_data: np.ndarray,
#     coherence_data: np.ndarray,
#     out_path: Path,
# ) -> None:
#     """Create minimal multi-panel figure (PSD + CSD + coherence) for slides."""
#     freq_fit = freq_hz_fit * 365.25
#     percentiles = quant_data["percentiles"]
#     idx_lo = _closest_percentile(percentiles, 5.0)
#     idx_hi = _closest_percentile(percentiles, 95.0)
#     q_psd = quant_data["psd_real"]
#     q_csd_real = quant_data["psd_real"]
#     q_csd_im = quant_data["psd_imag"]
#     coh_q = quant_data.get("coherence")

#     fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

#     # --- PSD panel (both channels) ---
#     ax_psd = axes[0]
#     colors = ["#1f77b4", "#ff7f0e"]
#     for ch, color in enumerate(colors):
#         data = periodogram[:, ch, ch].real
#         ax_psd.loglog(freq_cpy_data, data, color=color, alpha=0.25, lw=1.0)
#         lo = np.clip(q_psd[idx_lo, :, ch, ch], 1e-12, None)
#         hi = np.clip(q_psd[idx_hi, :, ch, ch], 1e-12, None)
#         ax_psd.fill_between(
#             freq_fit,
#             lo,
#             hi,
#             color=color,
#             alpha=0.45,
#             lw=0,
#         )

#     # --- CSD panel (real & imag) ---
#     ax_csd = axes[1]
#     csd_data = periodogram[:, 0, 1]
#     components = [
#         (csd_data.real, q_csd_real, "C2"),
#         (csd_data.imag, q_csd_im, "C3"),
#     ]
#     for idx, (data_component, quant_array, color) in enumerate(components):
#         lo = quant_array[idx_lo, :, 0, 1]
#         hi = quant_array[idx_hi, :, 0, 1]
#         ax_csd.plot(
#             freq_cpy_data,
#             data_component,
#             color=color,
#             alpha=0.25,
#             lw=1.0,
#             linestyle="--" if idx == 1 else "-",
#         )
#         ax_csd.fill_between(
#             freq_fit,
#             lo,
#             hi,
#             color=color,
#             alpha=0.45,
#             lw=0,
#         )
#     ax_csd.axhline(0.0, color="k", alpha=0.15, lw=0.8)

#     # --- Coherence panel ---
#     ax_coh = axes[2]
#     if coh_q is not None and coherence_data is not None:
#         lo = np.clip(coh_q[idx_lo, :, 0, 1], 0.0, 1.0)
#         hi = np.clip(coh_q[idx_hi, :, 0, 1], 0.0, 1.0)
#         ax_coh.plot(
#             freq_coh_data,
#             coherence_data,
#             color="C4",
#             alpha=0.25,
#             lw=1.0,
#         )
#         ax_coh.fill_between(
#             freq_fit,
#             lo,
#             hi,
#             color="C4",
#             alpha=0.45,
#             lw=0,
#         )
#     else:
#         ax_coh.axis("off")

#     for ax in axes:
#         ax.set_axis_off()

#     fig.savefig(out_path, dpi=500, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)


def _format_background_axes(
    ax,
    with_axes: bool,
    freq_range: tuple[float, float] | None = None,
) -> None:
    if not with_axes:
        ax.set_axis_off()
        return

    ax.set_axis_on()
    ax.grid(False)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#999999")
    ax.tick_params(labelsize=10, width=1.0, colors="#555555")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if freq_range is not None:
        ax.set_xlim(freq_range)


def plot_background_panel(
    freq_hz_fit: np.ndarray,
    quant_data: Dict[str, np.ndarray],
    freq_cpy_data: np.ndarray,
    periodogram: np.ndarray,
    freq_coh_data: np.ndarray,
    coherence_data: np.ndarray,
    out_prefix: Path,
) -> None:
    freq_fit = freq_hz_fit * 365.25
    percentiles = quant_data["percentiles"]
    idx_lo = _closest_percentile(percentiles, 5.0)
    idx_hi = _closest_percentile(percentiles, 95.0)
    q_psd = quant_data["psd_real"]
    q_csd_real = quant_data["psd_real"]
    q_csd_im = quant_data["psd_imag"]
    coh_q = quant_data.get("coherence")
    variants = [(False, ""), (True, "_axes")]

    # PSD backgrounds
    for ch in range(2):
        color = f"C{ch}"
        data = periodogram[:, ch, ch].real
        lo = np.clip(q_psd[idx_lo, :, ch, ch], 1e-12, None)
        hi = np.clip(q_psd[idx_hi, :, ch, ch], 1e-12, None)
        for show_axes, suffix in variants:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(freq_cpy_data, data, color=color, alpha=0.25, lw=1.0)
            ax.fill_between(freq_fit, lo, hi, color=color, alpha=0.45, lw=0)
            _format_background_axes(
                ax,
                show_axes,
                (float(freq_cpy_data[0]), float(freq_cpy_data[-1])),
            )
            fig.savefig(
                out_prefix / f"background_psd_ch{ch+1}{suffix}.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)

    csd_data = periodogram[:, 0, 1]
    components = [
        (csd_data.real, q_csd_real, "background_csd_real", "C2", False),
        (csd_data.imag, q_csd_im, "background_csd_imag", "C3", True),
    ]
    for data_component, quant_arr, name, color, dashed in components:
        lo = quant_arr[idx_lo, :, 0, 1]
        hi = quant_arr[idx_hi, :, 0, 1]
        for show_axes, suffix in variants:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(
                freq_cpy_data,
                data_component,
                color=color,
                alpha=0.25,
                lw=1.0,
                linestyle="--" if dashed else "-",
            )
            ax.fill_between(freq_fit, lo, hi, color=color, alpha=0.45, lw=0)
            ax.axhline(0.0, color="k", alpha=0.1, lw=0.8)
            _format_background_axes(
                ax,
                show_axes,
                (float(freq_cpy_data[0]), float(freq_cpy_data[-1])),
            )
            fig.savefig(
                out_prefix / f"{name}{suffix}.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)

    if coh_q is not None and coherence_data is not None:
        lo = np.clip(coh_q[idx_lo, :, 0, 1], 0.0, 1.0)
        hi = np.clip(coh_q[idx_hi, :, 0, 1], 0.0, 1.0)
        for show_axes, suffix in variants:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(
                freq_coh_data, coherence_data, color="C4", alpha=0.25, lw=1.0
            )
            ax.fill_between(freq_fit, lo, hi, color="C4", alpha=0.45, lw=0)
            _format_background_axes(
                ax,
                show_axes,
                (float(freq_coh_data[0]), float(freq_coh_data[-1])),
            )
            fig.savefig(
                out_prefix / f"background_coherence{suffix}.png",
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)


plot_background_panel(
    freq_fit_hz,
    quant_payload,
    freq_cpy,
    pdgrm,
    f_cpy,
    Cxy,
    OUTDIR,
)

print("Saved all figures (periodogram + multivar P-spline) to:", OUTDIR)
