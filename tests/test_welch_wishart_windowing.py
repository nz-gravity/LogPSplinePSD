import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import welch

from log_psplines.coarse_grain import (
    apply_coarse_grain_multivar_fft,
    apply_coarse_graining_univar,
    compute_binning_structure,
)
from log_psplines.datatypes.multivar import MultivarFFT
from log_psplines.example_datasets.lisa_data import LISAData


def _make_test_signal(n: int, fs: float) -> np.ndarray:
    rng = np.random.default_rng(123)
    t = np.arange(n, dtype=float) / fs
    low = 0.7 * np.sin(2 * np.pi * 0.002 * t)
    mid = 0.4 * np.sin(2 * np.pi * 0.08 * t)
    noise = rng.normal(scale=0.05, size=n)
    return (low + mid + noise)[:, None]


def _wishart_psd(
    data: np.ndarray,
    fs: float,
    Nb: int,
    window: str | tuple | None,
) -> tuple[np.ndarray, np.ndarray]:
    fft = MultivarFFT.compute_wishart(data, fs=fs, Nb=Nb, window=window)
    return fft.freq, np.real(fft.raw_psd[:, 0, 0])


def _log_rms_difference(reference: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-18
    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    return float(
        np.sqrt(np.mean((np.log10(tgt + eps) - np.log10(ref + eps)) ** 2))
    )


def _load_lisa_x_segment(n: int, offset: int = 0) -> np.ndarray:
    lisa = LISAData.load(data_path=Path("data/tdi.h5"))
    x_channel = np.asarray(lisa.data[:, 0], dtype=float)
    segment = x_channel[offset : offset + n]
    if segment.shape[0] != n:
        raise ValueError(
            f"Requested {n} samples from LISA TDI data but got {segment.shape[0]}"
        )
    return segment


def _coarse_log_rms(
    psd_fine: np.ndarray,
    freq_fine: np.ndarray,
    wishart_fft: MultivarFFT,
    spec,
) -> float:
    selection = np.asarray(spec.selection_mask, dtype=bool)
    psd_coarse = apply_coarse_graining_univar(
        psd_fine[selection], spec, freq_fine[selection]
    )
    wishart_coarse = apply_coarse_grain_multivar_fft(wishart_fft, spec)
    psd_wishart_coarse = np.real(wishart_coarse.raw_psd[:, 0, 0])
    return _log_rms_difference(psd_coarse, psd_wishart_coarse)


def _coarse_apply(
    psd_fine: np.ndarray, freq_fine: np.ndarray, spec
) -> tuple[np.ndarray, np.ndarray]:
    selection = np.asarray(spec.selection_mask, dtype=bool)
    psd_coarse = apply_coarse_graining_univar(
        psd_fine[selection], spec, freq_fine[selection]
    )
    return np.asarray(spec.f_coarse, dtype=float), psd_coarse


def _coarse_psd_wishart(
    wishart_fft: MultivarFFT, spec
) -> tuple[np.ndarray, np.ndarray]:
    wishart_coarse = apply_coarse_grain_multivar_fft(wishart_fft, spec)
    return wishart_coarse.freq, np.real(wishart_coarse.raw_psd[:, 0, 0])


def test_windowing_reduces_welch_wishart_gap(outdir):
    outdir = f"{outdir}/welch_wishart_windowing"
    os.makedirs(outdir, exist_ok=True)

    fs = 1.0
    n = 4096
    Nb = 4
    Lb = n // Nb
    data = _make_test_signal(n, fs)
    trans_freq = 10**-2

    f_welch, psd_welch = welch(
        data[:, 0],
        fs=fs,
        window="hann",
        nperseg=Lb,
        noverlap=Lb // 2,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    # Drop DC to match compute_wishart behavior
    f_welch = f_welch[1:]
    psd_welch = psd_welch[1:]

    wishart_rect = MultivarFFT.compute_wishart(data, fs=fs, Nb=Nb, window=None)
    np.testing.assert_allclose(wishart_rect.freq, f_welch)
    psd_rect = np.real(wishart_rect.raw_psd[:, 0, 0])

    baseline_rms = _log_rms_difference(psd_welch, psd_rect)

    window_cfgs = {
        "hann": "hann",
        "tukey": ("tukey", 0.5),
        "hamming": "hamming",
        "kaiser_beta5": ("kaiser", 5.0),
    }

    window_rms = {}
    for name, cfg in window_cfgs.items():
        freq_win, psd_win = _wishart_psd(data, fs=fs, Nb=Nb, window=cfg)
        np.testing.assert_allclose(freq_win, f_welch)
        window_rms[name] = _log_rms_difference(psd_welch, psd_win)
        print(f"Window {name}: log-RMS diff={window_rms[name]:.4f}")

    assert window_rms["hann"] < baseline_rms * 0.8
    best_label = min(window_rms, key=window_rms.get)
    assert window_rms[best_label] < baseline_rms * 0.75

    Nc = min(512, f_welch.size)
    if (Nc % 2) != (f_welch.size % 2):
        Nc = max(1, Nc - 1)
    spec = compute_binning_structure(
        f_welch,
        Nc=Nc,
        f_min=f_welch[0],
        f_max=f_welch[-1],
    )
    coarse_baseline = _coarse_log_rms(psd_welch, f_welch, wishart_rect, spec)
    coarse_rms = {}
    for name, cfg in window_cfgs.items():
        fft_win = MultivarFFT.compute_wishart(data, fs=fs, Nb=Nb, window=cfg)
        coarse_rms[name] = _coarse_log_rms(psd_welch, f_welch, fft_win, spec)
        print(f"Coarse window {name}: log-RMS diff={coarse_rms[name]:.4f}")
    assert coarse_rms["hann"] < coarse_baseline * 0.8
    best_coarse = min(coarse_rms, key=coarse_rms.get)
    assert coarse_rms[best_coarse] < coarse_baseline * 0.75

    _make_plot(
        f_welch,
        psd_welch,
        psd_rect,
        window_cfgs,
        data,
        fs,
        Nb,
        spec,
        trans_freq,
        f"{outdir}/simulated.png",
    )


@pytest.mark.skipif(
    not Path("data/tdi.h5").exists(), reason="LISA TDI data not available"
)
def test_lisa_x_channel_windowing_improves_match(outdir):
    outdir = f"{outdir}/welch_wishart_windowing"
    os.makedirs(outdir, exist_ok=True)

    fs = 0.5  # dt = 2 s in the provided TDI file
    n = 65536  # keep runtime modest while matching block sizes
    Nb = 4
    Lb = n // Nb
    trans_freq = 10**-4

    data = _load_lisa_x_segment(n)[:, None]

    f_welch, psd_welch = welch(
        data[:, 0],
        fs=fs,
        window="hann",
        nperseg=Lb,
        noverlap=Lb // 2,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    f_welch = f_welch[1:]
    psd_welch = psd_welch[1:]

    wishart_rect = MultivarFFT.compute_wishart(data, fs=fs, Nb=Nb, window=None)
    np.testing.assert_allclose(wishart_rect.freq, f_welch)
    psd_rect = np.real(wishart_rect.raw_psd[:, 0, 0])
    baseline_rms = _log_rms_difference(psd_welch, psd_rect)

    window_cfgs = {
        "hann": "hann",
        "hamming": "hamming",
        "tukey": ("tukey", 0.5),
        "kaiser_beta5": ("kaiser", 5.0),
    }

    window_rms = {}
    for name, cfg in window_cfgs.items():
        freq_win, psd_win = _wishart_psd(data, fs=fs, Nb=Nb, window=cfg)
        np.testing.assert_allclose(freq_win, f_welch)
        window_rms[name] = _log_rms_difference(psd_welch, psd_win)
        print(f"LISA window {name}: log-RMS diff={window_rms[name]:.4f}")

    assert window_rms["hann"] < baseline_rms * 0.8
    best_label = min(window_rms, key=window_rms.get)
    assert window_rms[best_label] < baseline_rms * 0.7

    Nc = min(f_welch.size // 2, f_welch.size)
    if (Nc % 2) != (f_welch.size % 2):
        Nc = max(1, Nc - 1)
    spec = compute_binning_structure(
        f_welch,
        Nc=Nc,
        f_min=f_welch[0],
        f_max=f_welch[-1],
    )
    coarse_baseline = _coarse_log_rms(psd_welch, f_welch, wishart_rect, spec)
    coarse_rms = {}
    for name, cfg in window_cfgs.items():
        fft_win = MultivarFFT.compute_wishart(data, fs=fs, Nb=Nb, window=cfg)
        coarse_rms[name] = _coarse_log_rms(psd_welch, f_welch, fft_win, spec)
        print(
            f"LISA coarse window {name}: log-RMS diff={coarse_rms[name]:.4f}"
        )
    _make_plot(
        f_welch,
        psd_welch,
        psd_rect,
        window_cfgs,
        data,
        fs,
        Nb,
        spec,
        trans_freq,
        f"{outdir}/lisa.png",
    )
    assert (
        coarse_rms["hamming"] < coarse_baseline
    ), f"Hamming window should improve coarse-grained match for LISA data (compared to rectangular) {coarse_rms}"
    best_coarse = min(coarse_rms, key=coarse_rms.get)
    assert (
        coarse_rms[best_coarse] < coarse_baseline
    ), f"{best_coarse} window should improve coarse-grained match"


def _make_plot(
    f_welch,
    psd_welch,
    psd_rect,
    window_cfgs,
    data,
    fs,
    Nb,
    spec,
    trans_freq,
    fname,
):
    # make some plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    axes[0].loglog(f_welch, psd_welch, label="Welch", color="black")
    axes[0].loglog(
        f_welch,
        psd_rect,
        label=f"Averaged (rectangular) [rms={_log_rms_difference(psd_welch, psd_rect):.4f}]",
        linestyle="--",
    )

    axes[1].loglog(f_welch, psd_welch, label="Welch", color="black")
    for name, cfg in window_cfgs.items():
        _, psd_win = _wishart_psd(data, fs=fs, Nb=Nb, window=cfg)
        axes[1].loglog(
            f_welch,
            psd_win,
            label=f"Averaged {name} [rms={_log_rms_difference(psd_welch, psd_win):.4f}]",
        )
    axes[0].set_title("Welch vs Averaged (no window)")
    axes[1].set_title("Welch vs Averaged (windowed)")

    f_coarse, psd_coarse_welch = _coarse_apply(psd_welch, f_welch, spec)
    _, psd_coarse_rect = _coarse_psd_wishart(
        MultivarFFT.compute_wishart(data, fs=fs, Nb=Nb, window=None),
        spec,
    )
    axes[2].loglog(
        f_welch,
        psd_welch,
        label="Welch",
        color="black",
    )
    axes[2].loglog(
        f_coarse,
        psd_coarse_rect,
        label=f"Averaged rectangular [rms={_log_rms_difference(psd_coarse_welch, psd_coarse_rect):.4f}]",
        linestyle="--",
    )
    for name, cfg in window_cfgs.items():
        fft_win = MultivarFFT.compute_wishart(data, fs=fs, Nb=Nb, window=cfg)
        _, psd_coarse_win = _coarse_psd_wishart(fft_win, spec)
        axes[2].loglog(
            f_coarse,
            psd_coarse_win,
            label=f"Averaged {name} coarse [rms={_log_rms_difference(psd_coarse_welch, psd_coarse_win):.4f}]",
        )
    axes[2].axvline(
        trans_freq, color="black", linestyle=":", label="Transition freq"
    )
    axes[2].set_title("Coarse-grained (binned) PSDs")
    for ax in axes:
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.set_xlabel("Frequency [Hz]")
    # Place a shared legend to the right
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    legend = fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.5,
        fontsize="small",
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(
        fname, dpi=300, bbox_inches="tight", bbox_extra_artists=[legend]
    )
