"""PSD matrix plotting with Welch overlay for LISA study."""

from __future__ import annotations

import numpy as np

from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.logger import logger
from log_psplines.plotting.psd_matrix import PSDMatrixPlotSpec, plot_psd_matrix

from .windows import welch_window_arg

FMIN = 1e-4
FMAX = 1e-1


def _drop_dc(emp: EmpiricalPSD) -> EmpiricalPSD:
    if emp.freq.size > 0 and np.isclose(emp.freq[0], 0.0):
        return EmpiricalPSD(
            freq=emp.freq[1:],
            psd=emp.psd[1:],
            coherence=emp.coherence[1:],
            channels=emp.channels,
        )
    return emp


def _restrict_freq_range(
    emp: EmpiricalPSD, *, fmin: float, fmax: float
) -> EmpiricalPSD:
    freq = np.asarray(emp.freq, dtype=float)
    mask = (freq >= float(fmin)) & (freq <= float(fmax))
    if not np.any(mask):
        raise ValueError("Welch frequency mask removed all bins.")
    return EmpiricalPSD(
        freq=freq[mask],
        psd=emp.psd[mask],
        coherence=emp.coherence[mask],
        channels=emp.channels,
    )


def _blocked_welch(
    data: np.ndarray,
    *,
    fs: float,
    Lb: int,
    nperseg: int,
    noverlap: int,
    window: str | tuple[str, float],
    detrend: str | bool,
) -> EmpiricalPSD:
    """Welch PSD computed within each block then averaged."""
    n, p = data.shape
    Nb = n // Lb
    if Nb < 1:
        raise ValueError("Not enough samples for even one Welch block.")
    n_used = Nb * Lb
    if n_used != n:
        data = data[:n_used]

    psd_sum = None
    freq_ref = None
    for idx in range(Nb):
        seg = data[idx * Lb : (idx + 1) * Lb]
        seg_nperseg = min(nperseg, Lb)
        seg_noverlap = min(noverlap, seg_nperseg - 1)
        emp = EmpiricalPSD.from_timeseries_data(
            data=seg,
            fs=fs,
            nperseg=seg_nperseg,
            noverlap=seg_noverlap,
            window=window,
            detrend=detrend,
        )
        if freq_ref is None:
            freq_ref = emp.freq
            psd_sum = np.zeros_like(emp.psd)
        if emp.freq.shape != freq_ref.shape or not np.allclose(
            emp.freq, freq_ref
        ):
            raise ValueError("Blocked Welch produced inconsistent freq grids.")
        psd_sum += emp.psd

    psd_avg = psd_sum / float(Nb)
    diag = np.abs(np.diagonal(psd_avg, axis1=1, axis2=2))
    denom = diag[:, :, None] * diag[:, None, :]
    coh = np.abs(psd_avg) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(denom > 0, coh.real / denom, np.nan)
    for ch in range(p):
        coherence[:, ch, ch] = 1.0

    return EmpiricalPSD(freq=freq_ref, psd=psd_avg, coherence=coherence)


def _empirical_ci_dict(empirical: EmpiricalPSD) -> dict:
    """Build a degenerate CI dictionary from a single empirical estimate."""
    psd = np.asarray(empirical.psd)
    coh = np.asarray(empirical.coherence)
    _, p, _ = psd.shape
    ci_dict: dict[str, dict[tuple[int, int], tuple[np.ndarray, ...]]] = {
        "psd": {},
        "coh": {},
        "re": {},
        "im": {},
        "mag": {},
    }
    for i in range(p):
        ci_dict["psd"][(i, i)] = (
            psd[:, i, i].real,
            psd[:, i, i].real,
            psd[:, i, i].real,
        )
        for j in range(p):
            if i > j:
                ci_dict["coh"][(i, j)] = (
                    coh[:, i, j],
                    coh[:, i, j],
                    coh[:, i, j],
                )
            elif i != j:
                ci_dict["re"][(i, j)] = (
                    psd[:, i, j].real,
                    psd[:, i, j].real,
                    psd[:, i, j].real,
                )
                ci_dict["im"][(i, j)] = (
                    psd[:, i, j].imag,
                    psd[:, i, j].imag,
                    psd[:, i, j].imag,
                )
    return ci_dict


def make_preprocessing_psd_plot(
    *,
    y_full: np.ndarray,
    fs: float,
    Lb: int,
    freq_true: np.ndarray,
    S_true: np.ndarray,
    outdir: str,
    filename: str = "preprocessing_psd_matrix.png",
    welch_window: str | tuple[str, float] | None = None,
    fmin: float = FMIN,
    fmax: float = FMAX,
    excluded_bands: tuple[tuple[float, float], ...] = (),
) -> None:
    """Plot true PSD vs raw empirical PSD vs blocked Welch before inference."""
    n = y_full.shape[0]
    if n < 2:
        raise ValueError(
            "Need at least two samples to build preprocessing PSD."
        )

    freq_emp = np.fft.rfftfreq(n, d=1.0 / fs)[1:]
    fft = np.fft.rfft(y_full, axis=0)[1:, :]
    scale = np.full(freq_emp.shape, 2.0 / (n * fs), dtype=np.float64)
    if (n % 2) == 0 and scale.size > 0:
        scale[-1] = 1.0 / (n * fs)
    psd_emp = np.einsum("ni,nj->nij", fft, np.conj(fft)) * scale[:, None, None]
    diag = np.abs(np.diagonal(psd_emp, axis1=1, axis2=2))
    denom = diag[:, :, None] * diag[:, None, :]
    coh = np.abs(psd_emp) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence_emp = np.where(denom > 0, coh.real / denom, np.nan)
    for ch in range(y_full.shape[1]):
        coherence_emp[:, ch, ch] = 1.0

    empirical = EmpiricalPSD(
        freq=freq_emp,
        psd=psd_emp,
        coherence=coherence_emp,
    )
    empirical = _restrict_freq_range(_drop_dc(empirical), fmin=fmin, fmax=fmax)

    welch_nperseg = int(round(10.0 * fs / float(fmin)))
    welch_nperseg = max(256, min(welch_nperseg, Lb))
    welch_noverlap = min(int(round(0.5 * welch_nperseg)), welch_nperseg - 1)
    empirical_welch = _blocked_welch(
        y_full,
        fs=fs,
        Lb=Lb,
        nperseg=welch_nperseg,
        noverlap=welch_noverlap,
        window=welch_window_arg(welch_window),
        detrend="constant",
    )
    empirical_welch = _restrict_freq_range(
        _drop_dc(empirical_welch),
        fmin=fmin,
        fmax=fmax,
    )

    plot_psd_matrix(
        PSDMatrixPlotSpec(
            ci_dict=_empirical_ci_dict(empirical),
            freq=np.asarray(empirical.freq, dtype=float),
            empirical_psd=None,
            extra_empirical_psd=[empirical_welch],
            extra_empirical_labels=["Welch (block-avg)"],
            extra_empirical_styles=[
                dict(color="0.5", lw=1.3, alpha=0.9, ls="-", zorder=-4),
            ],
            true_psd=interp_matrix(
                np.asarray(freq_true),
                np.asarray(S_true),
                np.asarray(empirical.freq, dtype=float),
            ),
            outdir=outdir,
            filename=filename,
            diag_yscale="log",
            offdiag_yscale="linear",
            xscale="log",
            show_coherence=True,
            show_knots=False,
            label="Empirical",
            model_color="0.75",
            freq_range=(fmin, fmax),
            excluded_bands=excluded_bands,
        )
    )
    logger.info(f"Saved preprocessing PSD plot to {outdir}/{filename}")


def make_psd_plot(
    idata,
    *,
    y_full: np.ndarray,
    fs: float,
    Lb: int,
    Nb: int,
    freq_true: np.ndarray,
    S_true: np.ndarray,
    outdir: str,
    filename: str = "psd_matrix.png",
    welch_window: str | tuple[str, float] | None = None,
    fmin: float = FMIN,
    fmax: float = FMAX,
    excluded_bands: tuple[tuple[float, float], ...] = (),
) -> None:
    """Generate PSD matrix plot with Welch overlay.

    Parameters
    ----------
    welch_window : str or tuple or None
        Window spec for the Welch diagnostic (None = rectangular/"boxcar").
        Passed through ``welch_window_arg()`` before use.
    """
    freq_plot = np.asarray(idata["posterior_psd"]["freq"].values)
    true_psd_physical = interp_matrix(
        np.asarray(freq_true), np.asarray(S_true), freq_plot
    )

    # Blocked Welch diagnostic — targets df ≈ FMIN.
    welch_nperseg = int(round(10.0 * fs / float(fmin)))
    welch_nperseg = max(256, min(welch_nperseg, Lb))
    welch_noverlap = int(round(0.5 * welch_nperseg))
    welch_noverlap = min(welch_noverlap, welch_nperseg - 1)

    empirical_welch = _blocked_welch(
        y_full,
        fs=fs,
        Lb=Lb,
        nperseg=welch_nperseg,
        noverlap=welch_noverlap,
        window=welch_window_arg(welch_window),
        detrend="constant",
    )
    empirical_welch = _drop_dc(empirical_welch)
    empirical_welch = _restrict_freq_range(
        empirical_welch, fmin=fmin, fmax=fmax
    )

    plot_psd_matrix(
        PSDMatrixPlotSpec(
            idata=idata,
            freq=freq_plot,
            empirical_psd=None,
            extra_empirical_psd=[empirical_welch],
            extra_empirical_labels=["Welch (block-avg)"],
            extra_empirical_styles=[
                dict(color="0.5", lw=1.3, alpha=0.9, ls="-", zorder=-4),
            ],
            outdir=outdir,
            filename=filename,
            diag_yscale="log",
            offdiag_yscale="linear",
            xscale="log",
            show_csd_magnitude=False,
            show_coherence=True,
            overlay_vi=True,
            freq_range=(fmin, fmax),
            true_psd=true_psd_physical,
            excluded_bands=excluded_bands,
        )
    )
    logger.info(f"Saved PSD plot to {outdir}/{filename}")

    # Also save a linear-x version for inspecting null regions.
    linear_filename = filename.replace(".png", "_linx.png")
    plot_psd_matrix(
        PSDMatrixPlotSpec(
            idata=idata,
            freq=freq_plot,
            empirical_psd=None,
            extra_empirical_psd=[empirical_welch],
            extra_empirical_labels=["Welch (block-avg)"],
            extra_empirical_styles=[
                dict(color="0.5", lw=1.3, alpha=0.9, ls="-", zorder=-4),
            ],
            outdir=outdir,
            filename=linear_filename,
            diag_yscale="log",
            offdiag_yscale="linear",
            xscale="linear",
            show_csd_magnitude=False,
            show_coherence=True,
            overlay_vi=True,
            freq_range=(fmin, fmax),
            true_psd=true_psd_physical,
            excluded_bands=excluded_bands,
        )
    )
    logger.info(f"Saved linear-x PSD plot to {outdir}/{linear_filename}")
