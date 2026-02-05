#!/usr/bin/env python3
"""
Generate and analyse example LISA X/Y/Z noise data.

Two data sources are supported:
* Synthetic XYZ noise drawn from lisatools' sensitivity matrix (default).
* The legacy LDC tdi.h5 sample (downloaded on demand).

In both cases, the code builds the analytic PSD/CSD matrix and compares it
against Welch estimates from the time-domain data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests

try:
    from .lisatools_backend import ensure_lisatools_backends

    ensure_lisatools_backends()
    from lisatools import detector as lisa_models
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    _HAS_LISATOOLS = True
except Exception:
    lisa_models = None
    XYZ2SensitivityMatrix = None
    _HAS_LISATOOLS = False

# --- Constants and default paths -------------------------------------------------
C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LIGHT_TRAVEL_TIME = L_ARM / C_LIGHT  # ≈ 8.33 s

OMS_ASD = 1.5e-11
OMS_FKNEE = 2e-3
PM_ASD = 3e-15
PM_LOW_FKNEE = 4e-4
PM_HIGH_FKNEE = 8e-3
LASER_FREQ = 2.81e14  # Hz

DATA_PATH = Path("data/tdi.h5")
TRIANGLE_PNG = Path("spectra_triangle.png")
TEN_DAYS = "https://raw.githubusercontent.com/nz-gravity/test_data/main/lisa_noise/noise_4a_truncated/data/tdi.h5"


def _oms_disp_psd(freq: np.ndarray) -> np.ndarray:
    """OMS displacement PSD [m^2 / Hz]."""
    return OMS_ASD**2 * (1.0 + (OMS_FKNEE / freq) ** 4)


def _tm_acc_psd(freq: np.ndarray) -> np.ndarray:
    """Proof-mass acceleration PSD [(m/s^2)^2 / Hz]."""
    return (
        PM_ASD**2
        * (1.0 + (PM_LOW_FKNEE / freq) ** 2)
        * (1.0 + (freq / PM_HIGH_FKNEE) ** 4)
    )


def _acc_to_disp(acc_psd: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Acceleration → displacement PSD."""
    return acc_psd / (2.0 * np.pi * freq) ** 4


def _disp_to_freq_psd(disp_psd: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Displacement → absolute frequency noise PSD."""
    scale = (2.0 * np.pi * freq * LASER_FREQ / C_LIGHT) ** 2
    return disp_psd * scale


def lisa_base_noises(freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Sop, Spm) frequency-noise PSDs in Hz^2/Hz for OMS and PM links."""
    Sop = _disp_to_freq_psd(_oms_disp_psd(freq), freq)
    Sa = _tm_acc_psd(freq)
    Sd = _acc_to_disp(Sa, freq)
    Spm = _disp_to_freq_psd(Sd, freq)
    return Sop, Spm


def _x_arg(freq: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * freq * LIGHT_TRAVEL_TIME


def _Cxx(freq: np.ndarray) -> np.ndarray:
    x = _x_arg(freq)
    return 16.0 * np.sin(x) ** 2 * np.sin(2.0 * x) ** 2


def _Cxy(freq: np.ndarray) -> np.ndarray:
    x = _x_arg(freq)
    return -16.0 * np.sin(x) * np.sin(2.0 * x) ** 3


def _S_xx(freq: np.ndarray, Sop: np.ndarray, Spm: np.ndarray) -> np.ndarray:
    x = _x_arg(freq)
    C = _Cxx(freq)
    sop_term = 4.0 * C * Sop
    spm_term = 4.0 * C * (3.0 + np.cos(2.0 * x)) * Spm
    return sop_term + spm_term


def _S_xy(freq: np.ndarray, Sop: np.ndarray, Spm: np.ndarray) -> np.ndarray:
    C = _Cxy(freq)
    sop_term = C * Sop
    spm_term = 4.0 * C * Spm
    return sop_term + spm_term


def tdi_xyz_psd_matrix(freq: np.ndarray) -> np.ndarray:
    """Analytic TDI-2 XYZ spectral matrix (absolute frequency noise), Hz^2/Hz."""
    Sop, Spm = lisa_base_noises(freq)
    Sxx = _S_xx(freq, Sop, Spm)
    Sxy = _S_xy(freq, Sop, Spm)

    S = np.zeros((len(freq), 3, 3), dtype=np.complex128)
    S[:, 0, 0] = S[:, 1, 1] = S[:, 2, 2] = Sxx
    S[:, 0, 1] = S[:, 1, 0] = Sxy
    S[:, 1, 2] = S[:, 2, 1] = Sxy
    S[:, 2, 0] = S[:, 0, 2] = Sxy
    return S


def ensure_lisa_tdi_data(
    data_url: str = TEN_DAYS, destination: Path = DATA_PATH
) -> float:
    """Download the default LISA TDI dataset if it is not present locally."""
    start_download = time.perf_counter()
    if isinstance(destination, str):
        destination = Path(destination)
    if destination.exists():
        print("Data file exists; download skipped.")
        return 0.0

    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading data...")
    response = requests.get(data_url, stream=True)
    if not response.ok:
        raise RuntimeError(f"Download failed ({response.status_code})")

    with destination.open("wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    download_time = time.perf_counter() - start_download
    print(f"Download complete: {download_time:.3f} s")
    return download_time


def strain_to_freq_psd(S_strain, f, L=L_ARM, nu0=LASER_FREQ):
    """
    Convert strain PSD matrix S_strain(f) -> absolute frequency PSD S_{Δν}(f).
    S_strain : shape (N, 3, 3), units 1/Hz
    returns  : shape (N, 3, 3), units Hz^2/Hz
    """
    factor = (2 * np.pi * f * nu0 * L / C_LIGHT) ** 2  # shape (N,)
    return S_strain * factor[:, None, None]


# --- lisatools-based helpers ----------------------------------------------------
def generate_lisa_xyz_noise_timeseries(
    duration: float,
    delta_t: float,
    model: str = "scirdv1",
    central_freq: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate correlated X, Y, Z TDI noise without lisatools.

    The Fourier coefficients are drawn such that Var[FFT[x]] = (N/(2Δt)) S(f)
    where S(f) is the XYZ spectral matrix from ``tdi_xyz_psd_matrix``.
    """
    if seed is not None:
        np.random.seed(seed)

    n = int(duration / delta_t)
    freq = np.fft.rfftfreq(n, d=delta_t)
    if freq[0] == 0.0:
        freq[0] = freq[1]

    S_true = tdi_xyz_psd_matrix(freq)  # (N, 3, 3) absolute frequency noise

    cov_fft = (n / (2.0 * delta_t)) * S_true
    chol = np.linalg.cholesky(cov_fft)

    N = len(freq)
    eps = np.random.normal(
        0.0, 1.0 / np.sqrt(2.0), (3, N)
    ) + 1j * np.random.normal(0.0, 1.0 / np.sqrt(2.0), (3, N))
    eps[:, 0] = np.random.normal(0.0, 1.0, 3)
    eps[:, -1] = np.random.normal(0.0, 1.0, 3)

    noise_fft = np.einsum("fij,jf->if", chol, eps)  # (3, N)

    x_t = np.fft.irfft(noise_fft[0], n=n)
    y_t = np.fft.irfft(noise_fft[1], n=n)
    z_t = np.fft.irfft(noise_fft[2], n=n)
    return x_t, y_t, z_t, freq, S_true


def welch_spectral_matrix_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    L: int,
    delta_t: float,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, ...]:
    """Welch PSD/CSD estimator for the XYZ channels (Hann window + overlap)."""
    n = len(x)
    step = int(L * (1 - overlap))
    w = np.hanning(L)
    U = np.mean(w**2)

    Sxx = Syy = Szz = 0.0
    Sxy = Syz = Szx = 0.0
    count = 0

    for start in range(0, n - L + 1, step):
        xs = x[start : start + L] * w
        ys = y[start : start + L] * w
        zs = z[start : start + L] * w

        Xf = np.fft.rfft(xs)
        Yf = np.fft.rfft(ys)
        Zf = np.fft.rfft(zs)

        scale = 2.0 * delta_t / (L * U)

        Sxx += scale * (np.abs(Xf) ** 2)
        Syy += scale * (np.abs(Yf) ** 2)
        Szz += scale * (np.abs(Zf) ** 2)

        Sxy += scale * (Xf * np.conj(Yf))
        Syz += scale * (Yf * np.conj(Zf))
        Szx += scale * (Zf * np.conj(Xf))

        count += 1

    Sxx /= count
    Syy /= count
    Szz /= count
    Sxy /= count
    Syz /= count
    Szx /= count

    freq = np.fft.rfftfreq(L, d=delta_t)
    if freq[0] == 0.0:
        freq[0] = freq[1]
    return freq, Sxx, Syy, Szz, Sxy, Syz, Szx


def spectral_matrix_from_components(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Szz: np.ndarray,
    Sxy: np.ndarray,
    Syz: np.ndarray,
    Szx: np.ndarray,
) -> np.ndarray:
    """Assemble a 3×3 spectral matrix Σ(f) from auto- and cross-spectra."""
    nf = len(Sxx)
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = Sxx
    cov[:, 1, 1] = Syy
    cov[:, 2, 2] = Szz

    cov[:, 0, 1] = Sxy
    cov[:, 1, 0] = np.conj(Sxy)

    cov[:, 1, 2] = Syz
    cov[:, 2, 1] = np.conj(Syz)

    cov[:, 2, 0] = Szx
    cov[:, 0, 2] = np.conj(Szx)
    return cov


def coherence(Sii: np.ndarray, Sjj: np.ndarray, Sij: np.ndarray) -> np.ndarray:
    return np.abs(Sij) / np.sqrt(Sii * Sjj)


def analytic_covariance_from_model(
    freq: np.ndarray,
    dt: float,
    n: int,
    model: str = "scirdv1",
    central_freq: Optional[float] = None,
) -> np.ndarray:
    """
    Build the analytic XYZ spectral matrix for the requested model.

    If lisatools is installed, we use its sensitivity matrix directly.
    Otherwise we fall back to the legacy LDC formulation.
    """
    if _HAS_LISATOOLS:
        model_checked = lisa_models.check_lisa_model(model)
        sens = XYZ2SensitivityMatrix(freq, model=model_checked)
        return np.transpose(sens.sens_mat, (2, 0, 1))

    fs = 1.0 / dt
    fmin = 1.0 / (n * dt)
    Spm, Sop = lisa_link_noises_ldc(freq, fs=fs, fmin=fmin)
    diag, csd = tdi2_psd_and_csd(freq, Spm, Sop)
    return covariance_matrix(diag, csd)


def plot_psd_coherence(
    freq: np.ndarray,
    S_true: np.ndarray,
    S_emp: Dict[str, np.ndarray],
    fname: Optional[Union[Path, str]] = None,
    *,
    psd_unit_label: str = "1/Hz",
) -> None:
    """
    Plot PSDs on the diagonal and coherences on the lower triangle.

    freq: frequency array
    S_true: (N, 3, 3) analytic spectral matrix
    S_emp: dict with keys "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"
    """

    Sxx_true = S_true[:, 0, 0]
    Syy_true = S_true[:, 1, 1]
    Szz_true = S_true[:, 2, 2]

    Sxy_true = S_true[:, 0, 1]
    Syz_true = S_true[:, 1, 2]
    Szx_true = S_true[:, 2, 0]

    Sxx_emp = S_emp["Sxx"]
    Syy_emp = S_emp["Syy"]
    Szz_emp = S_emp["Szz"]

    Sxy_emp = S_emp["Sxy"]
    Syz_emp = S_emp["Syz"]
    Szx_emp = S_emp["Szx"]

    coh_xy_true = coherence(Sxx_true, Syy_true, Sxy_true)
    coh_yz_true = coherence(Syy_true, Szz_true, Syz_true)
    coh_zx_true = coherence(Szz_true, Sxx_true, Szx_true)

    coh_xy_emp = coherence(Sxx_emp, Syy_emp, Sxy_emp)
    coh_yz_emp = coherence(Syy_emp, Szz_emp, Syz_emp)
    coh_zx_emp = coherence(Szz_emp, Sxx_emp, Szx_emp)

    channels = ["X", "Y", "Z"]
    true_psd = [Sxx_true, Syy_true, Szz_true]
    emp_psd = [Sxx_emp, Syy_emp, Szz_emp]

    true_coh = [
        [None, coh_xy_true, coh_zx_true],
        [coh_xy_true, None, coh_yz_true],
        [coh_zx_true, coh_yz_true, None],
    ]

    emp_coh = [
        [None, coh_xy_emp, coh_zx_emp],
        [coh_xy_emp, None, coh_yz_emp],
        [coh_zx_emp, coh_yz_emp, None],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            if i == j:
                ax.loglog(freq, true_psd[i], label="True PSD")
                ax.loglog(freq, emp_psd[i], alpha=0.5, label="Welch PSD")
                ax.set_title(f"{channels[i]} PSD")
                ax.set_ylabel(f"PSD [{psd_unit_label}]")
                ax.grid(True, which="both", ls="--", alpha=0.3)
                if i == 0:
                    ax.legend()
                continue

            ax.semilogx(freq, true_coh[i][j], label="True coh")
            ax.semilogx(freq, emp_coh[i][j], alpha=0.5, label="Welch coh")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Coherence")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_title(f"{channels[i]}–{channels[j]}")
            if i == 1 and j == 0:
                ax.legend()

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    if fname is not None:
        out_path = Path(fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _interp_complex(
    freq_src: np.ndarray, vals: np.ndarray, freq_tgt: np.ndarray
) -> np.ndarray:
    """Interpolate complex-valued spectra by interpolating real/imag parts."""
    real = np.interp(freq_tgt, freq_src, vals.real)
    imag = np.interp(freq_tgt, freq_src, vals.imag)
    return real + 1j * imag


# --- Noise and transfer helpers --------------------------------------------------
def lisa_link_noises_ldc(
    freq: np.ndarray,
    fs: float,
    fmin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the single-link proof-mass (Spm) and optical-path (Sop) PSDs
    used in the LDC noise realizations.

    See https://zenodo.org/doi/10.5281/zenodo.15698080
    """
    exp_term = np.exp(-2.0 * np.pi * fmin / fs) * np.exp(
        -2j * np.pi * freq / fs
    )
    denom_mag2 = np.abs(1.0 - exp_term) ** 2

    psd_tm_high = (
        (2.0 * PM_ASD * LASER_FREQ / (2.0 * np.pi * C_LIGHT)) ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
    )
    psd_tm_low = (
        (2.0 * PM_ASD * LASER_FREQ * PM_LOW_FKNEE / (2.0 * np.pi * C_LIGHT))
        ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
        * np.abs(1.0 / (1.0 - np.exp(-2j * np.pi * freq / fs))) ** 2
        * (2.0 * np.pi / fs) ** 2
    )
    Spm = psd_tm_high + psd_tm_low

    psd_oms_high = (OMS_ASD * fs * LASER_FREQ / C_LIGHT) ** 2 * np.sin(
        2.0 * np.pi * freq / fs
    ) ** 2
    psd_oms_low = (
        (2.0 * np.pi * OMS_ASD * LASER_FREQ * OMS_FKNEE**2 / C_LIGHT) ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
    )
    Sop = psd_oms_high + psd_oms_low
    return Spm, Sop


def tdi2_psd_and_csd(
    freq: np.ndarray, Spm: np.ndarray, Sop: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute diagonal PSD (X2) and cross-term CSD (XY) for TDI2 combinations.

    Because of the symmetries of the equal-arm constellation:
        S_X2 = S_Y2 = S_Z2
        S_XY = S_YZ = S_ZX

    See e.g. Eq. (54-55) of https://arxiv.org/pdf/2211.02539

    Also see
    https://github.com/mikekatz04/LISAanalysistools/blob/bf2ea28e2c62e9f0be0b8f74884b601b9bf2097d/src/lisatools/sensitivity.py#L347C10-L347C39
    """
    x = 2.0 * np.pi * LIGHT_TRAVEL_TIME * freq
    sinx = np.sin(x)
    sin2x = np.sin(2.0 * x)
    cosx = np.cos(x)
    cos2x = np.cos(2.0 * x)

    diag = 64.0 * sinx**2 * sin2x**2 * Sop
    diag += 256.0 * (3.0 + cos2x) * cosx**2 * sinx**4 * Spm

    csd = -16.0 * sinx * (sin2x**3) * (4.0 * Spm + Sop)
    return diag, csd


def covariance_matrix(diag: np.ndarray, csd: np.ndarray) -> np.ndarray:
    """Assemble the 3×3 covariance matrix Σ(f) for each frequency."""
    nf = diag.size
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = cov[:, 1, 1] = cov[:, 2, 2] = diag
    cov[:, 0, 1] = cov[:, 1, 0] = csd
    cov[:, 1, 2] = cov[:, 2, 1] = csd
    cov[:, 0, 2] = cov[:, 2, 0] = csd
    return cov


def periodogram_covariance(
    auto_psd: Dict[str, np.ndarray], cross_csd: Dict[str, np.ndarray]
) -> np.ndarray:
    """Build the empirical 3×3 spectral matrix from auto/cross periodograms."""
    nf = len(next(iter(auto_psd.values())))
    cov = np.zeros((nf, 3, 3), dtype=np.complex128)
    cov[:, 0, 0] = auto_psd["X"]
    cov[:, 1, 1] = auto_psd["Y"]
    cov[:, 2, 2] = auto_psd["Z"]

    cov[:, 0, 1] = cross_csd["XY"]
    cov[:, 1, 0] = np.conj(cov[:, 0, 1])

    cov[:, 1, 2] = cross_csd["YZ"]
    cov[:, 2, 1] = np.conj(cov[:, 1, 2])

    cov[:, 2, 0] = cross_csd["ZX"]
    cov[:, 0, 2] = np.conj(cov[:, 2, 0])
    return cov


# --- Data handling + spectral estimates -----------------------------------------
def load_tdi_timeseries(
    h5_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read t, X2, Y2, Z2 arrays from the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        t = np.array(f["t"])
        X2 = np.array(f["X2"])
        Y2 = np.array(f["Y2"])
        Z2 = np.array(f["Z2"])
    return t, X2, Y2, Z2


def compute_periodograms(
    t: np.ndarray,
    X2: np.ndarray,
    Y2: np.ndarray,
    Z2: np.ndarray,
) -> Tuple[
    np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]
]:
    """Return (freq, auto-PSD dict, cross-CSD dict, metadata) from the time-domain data."""
    dt = t[1] - t[0]
    n = len(t)
    freq_full = np.fft.rfftfreq(n, dt)
    fft_x = np.fft.rfft(X2)
    fft_y = np.fft.rfft(Y2)
    fft_z = np.fft.rfft(Z2)

    scale = dt / n
    auto = {
        "X": scale * np.abs(fft_x) ** 2,
        "Y": scale * np.abs(fft_y) ** 2,
        "Z": scale * np.abs(fft_z) ** 2,
    }
    cross = {
        "XY": scale * fft_x * np.conj(fft_y),
        "YZ": scale * fft_y * np.conj(fft_z),
        "ZX": scale * fft_z * np.conj(fft_x),
    }

    # Double the positive-frequency interior to account for two-sided FFT.
    for arr in list(auto.values()) + list(cross.values()):
        arr[1:-1] *= 2.0

    # Drop the DC bin for plotting (log axis incompatible with zero frequency).
    freq = freq_full[1:]
    for key in auto:
        auto[key] = auto[key][1:]
    for key in cross:
        cross[key] = cross[key][1:]

    meta = {"dt": dt, "fs": 1.0 / dt, "n": n, "fmin": 1.0 / (n * dt)}
    return freq, auto, cross, meta


@dataclass
class LISAData:
    """
    Container for frequency- and time-domain LISA spectra and helpers.

    Use ``prefer_simulated=True`` (default) to synthesise XYZ noise directly
    from the lisatools sensitivity matrix. If lisatools is unavailable, or you
    prefer to use the LDC sample, set ``prefer_simulated=False`` or provide a
    ``data_path`` pointing to the HDF5 file.
    """

    time: np.ndarray
    data: np.ndarray
    freq: np.ndarray
    matrix: np.ndarray
    true_matrix: np.ndarray
    delta_t: float

    @classmethod
    def load(
        cls,
        data_path: Optional[Union[Path, str]] = DATA_PATH,
        data_url: str = TEN_DAYS,
        duration: float = 4 * 86_400.0,
        delta_t: float = 5.0,
        model: str = "scirdv1",
        central_freq: Optional[float] = None,
        seed: Optional[int] = 123,
        welch_length: int = 4096,
        welch_overlap: float = 0.5,
        prefer_simulated: bool = True,
    ) -> "LISAData":
        """
        Build a LISAData instance either from simulation or a stored dataset.

        prefer_simulated:
            If True and lisatools is installed, synthesise XYZ noise using its
            sensitivity matrix. Otherwise fall back to the legacy HDF5 sample.
            When True, any provided data_path is ignored unless lisatools is
            unavailable; set to False to force loading from disk.
        """
        if prefer_simulated and _HAS_LISATOOLS:
            return cls._from_simulation(
                duration=duration,
                delta_t=delta_t,
                model=model,
                central_freq=central_freq,
                seed=seed,
                welch_length=welch_length,
                welch_overlap=welch_overlap,
            )

        if prefer_simulated and not _HAS_LISATOOLS:
            print(
                "lisatools not installed; falling back to the legacy LDC "
                "sample at data/tdi.h5."
            )

        if data_path is None:
            raise ValueError(
                "data_path must be provided when prefer_simulated is False."
            )

        return cls._from_hdf5(
            data_path=data_path,
            data_url=data_url,
            model=model,
            welch_length=welch_length,
            welch_overlap=welch_overlap,
            central_freq=central_freq,
        )

    @classmethod
    def _from_simulation(
        cls,
        duration: float,
        delta_t: float,
        model: str,
        central_freq: Optional[float],
        seed: Optional[int],
        welch_length: int,
        welch_overlap: float,
    ) -> "LISAData":
        x, y, z, _, _ = generate_lisa_xyz_noise_timeseries(
            duration=duration,
            delta_t=delta_t,
            model=model,
            central_freq=central_freq,
            seed=seed,
        )

        freq_est, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
            x, y, z, L=welch_length, delta_t=delta_t, overlap=welch_overlap
        )
        true_cov = tdi_xyz_psd_matrix(freq_est)
        empirical_cov = spectral_matrix_from_components(
            Sxx, Syy, Szz, Sxy, Syz, Szx
        )

        t = np.arange(len(x)) * delta_t
        data = np.vstack((x, y, z)).T
        return cls(
            time=t,
            freq=freq_est,
            matrix=empirical_cov,
            true_matrix=true_cov,
            data=data,
            delta_t=delta_t,
        )

    @classmethod
    def _from_hdf5(
        cls,
        data_path: Union[Path, str],
        data_url: str,
        model: str,
        welch_length: int,
        welch_overlap: float,
        central_freq: Optional[float],
    ) -> "LISAData":
        path = Path(data_path)
        if not path.exists():
            ensure_lisa_tdi_data(data_url=data_url, destination=path)

        t, X2, Y2, Z2 = load_tdi_timeseries(path)
        dt = t[1] - t[0]
        data = np.vstack((X2, Y2, Z2)).T
        freq_est, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
            X2, Y2, Z2, L=welch_length, delta_t=dt, overlap=welch_overlap
        )

        true_cov = analytic_covariance_from_model(
            freq_est,
            dt=dt,
            n=len(t),
            model=model,
            central_freq=central_freq,
        )
        empirical_cov = spectral_matrix_from_components(
            Sxx, Syy, Szz, Sxy, Syz, Szx
        )
        print(f"Loaded {len(t)} samples from {path} (fs={1.0 / dt:.6f} Hz).")

        return cls(
            time=t,
            freq=freq_est,
            matrix=empirical_cov,
            true_matrix=true_cov,
            data=data,
            delta_t=dt,
        )

    def plot(
        self,
        fname: Union[Path, str] = TRIANGLE_PNG,
    ) -> None:
        """Produce the diagnostic PSD/CSD plots for the stored spectra."""
        S_emp = {
            "Sxx": self.matrix[:, 0, 0].real,
            "Syy": self.matrix[:, 1, 1].real,
            "Szz": self.matrix[:, 2, 2].real,
            "Sxy": self.matrix[:, 0, 1],
            "Syz": self.matrix[:, 1, 2],
            "Szx": self.matrix[:, 2, 0],
        }

        plot_psd_coherence(self.freq, self.true_matrix, S_emp, fname=fname)
        print(f"Wrote PSD/coherence plot to {Path(fname).resolve()}")


def plot_true_psd_comparison(
    out_path: Union[Path, str] = "lisa_psd_compare.png",
    model: str = "scirdv1",
) -> None:
    """
    Overplot the analytic PSD from the simulated lisatools model and the
    analytic PSD computed on the HDF5 dataset grid.
    """
    sim = LISAData.load(prefer_simulated=True, model=model)
    hdf5 = LISAData.load(prefer_simulated=False, model=model)

    freq_target = hdf5.freq
    sim_diag = sim.true_matrix[:, 0, 0]
    sim_interp = _interp_complex(sim.freq, sim_diag, freq_target)

    hdf5_diag = hdf5.true_matrix[:, 0, 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(freq_target, hdf5_diag.real, label="HDF5 analytic PSD")
    ax.loglog(
        freq_target, sim_interp.real, "--", label="Simulated analytic PSD"
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote comparison plot to {out_path.resolve()}")


# --- Main entry point ------------------------------------------------------------
def main() -> None:
    lisa_data = LISAData.load()
    lisa_data.plot(TRIANGLE_PNG)
    plot_true_psd_comparison()


if __name__ == "__main__":
    main()
