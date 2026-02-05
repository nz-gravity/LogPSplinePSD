"""Generate lisatools-based XYZ noise and compare Welch vs true PSD/CSD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from log_psplines.example_datasets.lisatools_backend import (
    ensure_lisatools_backends,
)

ensure_lisatools_backends()

try:
    from lisatools import detector as lisa_models
    from lisatools.sensitivity import XYZ2SensitivityMatrix
except Exception as exc:
    raise SystemExit(
        "lisatools is required for this script. " "Install it and re-run."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.example_datasets.lisa_data import (  # noqa: E402
    plot_psd_coherence,
    spectral_matrix_from_components,
    strain_to_freq_psd,
    welch_spectral_matrix_xyz,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

USE_FREQ_UNITS = True
SAVE_NPZ = True
NPZ_PATH = RESULTS_DIR / "lisatools_synth_data.npz"
SEC_IN_DAY = 86_400.0

def summarize_ratio(label: str, ratio: np.ndarray) -> None:
    clean = ratio[np.isfinite(ratio)]
    if clean.size == 0:
        print(f"{label}: no finite entries to summarize.")
        return
    pct = np.percentile(clean, [5, 50, 95])
    print(f"{label}: p05={pct[0]:.3g}, p50={pct[1]:.3g}, p95={pct[2]:.3g}")


def generate_lisatools_xyz_noise_timeseries(
    duration: float,
    delta_t: float,
    model: str = "scirdv1",
    seed: int | None = None,
    return_freq_noise: bool = False,
    chunk_seconds: float | None = None,
):
    """
    Generate X,Y,Z TDI noise with PSD matrix from lisatools.

    Var[FFT[x]] = (N / (2 * delta_t)) * S(f)
    """
    if chunk_seconds is not None:
        if return_freq_noise:
            raise ValueError("return_freq_noise not supported with chunking.")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")

        n_total = int(duration / delta_t)
        n_chunk = int(chunk_seconds / delta_t)
        if n_chunk <= 1:
            raise ValueError("chunk length too short for FFT.")
        n_chunks = n_total // n_chunk
        if n_chunks < 1:
            raise ValueError("duration shorter than one chunk.")
        n_used = n_chunks * n_chunk
        if n_used != n_total:
            print(
                f"Truncating duration to {n_chunks} chunks "
                f"({n_used * delta_t:.0f} s total)."
            )

        rng = np.random.default_rng(seed)
        freq_chunk = np.fft.rfftfreq(n_chunk, d=delta_t)
        if freq_chunk[0] == 0.0:
            freq_chunk[0] = freq_chunk[1]

        model_checked = lisa_models.check_lisa_model(model)
        sens_chunk = XYZ2SensitivityMatrix(freq_chunk, model=model_checked)
        S_true_chunk = np.transpose(sens_chunk.sens_mat, (2, 0, 1))
        cov_fft = (n_chunk / (2.0 * delta_t)) * S_true_chunk
        chol = np.linalg.cholesky(cov_fft)

        x_t = np.empty(n_used, dtype=np.float64)
        y_t = np.empty_like(x_t)
        z_t = np.empty_like(x_t)
        n_freq = len(freq_chunk)
        for idx in range(n_chunks):
            eps = rng.normal(
                0.0, 1.0 / np.sqrt(2.0), (3, n_freq)
            ) + 1j * rng.normal(0.0, 1.0 / np.sqrt(2.0), (3, n_freq))
            eps[:, 0] = rng.normal(0.0, 1.0, 3)
            eps[:, -1] = rng.normal(0.0, 1.0, 3)
            noise_fft = np.einsum("fij,jf->if", chol, eps)
            start = idx * n_chunk
            end = start + n_chunk
            x_t[start:end] = np.fft.irfft(noise_fft[0], n=n_chunk)
            y_t[start:end] = np.fft.irfft(noise_fft[1], n=n_chunk)
            z_t[start:end] = np.fft.irfft(noise_fft[2], n=n_chunk)

        freq = np.fft.rfftfreq(n_used, d=delta_t)
        if freq[0] == 0.0:
            freq[0] = freq[1]
        sens = XYZ2SensitivityMatrix(freq, model=model_checked)
        S_true = np.transpose(sens.sens_mat, (2, 0, 1))
        return x_t, y_t, z_t, freq, S_true

    if seed is not None:
        np.random.seed(seed)

    n = int(duration / delta_t)
    freq = np.fft.rfftfreq(n, d=delta_t)
    if freq[0] == 0.0:
        freq[0] = freq[1]

    model_checked = lisa_models.check_lisa_model(model)
    sens = XYZ2SensitivityMatrix(freq, model=model_checked)
    S_true = np.transpose(sens.sens_mat, (2, 0, 1))

    cov_fft = (n / (2.0 * delta_t)) * S_true
    chol = np.linalg.cholesky(cov_fft)

    n_freq = len(freq)
    eps = np.random.normal(
        0.0, 1.0 / np.sqrt(2.0), (3, n_freq)
    ) + 1j * np.random.normal(0.0, 1.0 / np.sqrt(2.0), (3, n_freq))
    eps[:, 0] = np.random.normal(0.0, 1.0, 3)
    eps[:, -1] = np.random.normal(0.0, 1.0, 3)

    noise_fft = np.einsum("fij,jf->if", chol, eps)

    x_t = np.fft.irfft(noise_fft[0], n=n)
    y_t = np.fft.irfft(noise_fft[1], n=n)
    z_t = np.fft.irfft(noise_fft[2], n=n)

    if return_freq_noise:
        return x_t, y_t, z_t, freq, S_true, noise_fft
    return x_t, y_t, z_t, freq, S_true


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate lisatools XYZ noise and diagnostics."
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        default=7 * 8,
        help="Total duration in days (default: 365).",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=1e-4,
        help="Minimum frequency for diagnostics (default: 1e-4).",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=1e-1,
        help="Maximum frequency for diagnostics (default: 1e-1).",
    )
    parser.add_argument(
        "--welch-days",
        type=float,
        default=None,
        help="Welch segment length in days (default: derived from fmin).",
    )
    parser.add_argument(
        "--welch-overlap",
        type=float,
        default=0.0,
        help="Welch overlap fraction (default: 0.0).",
    )
    args = parser.parse_args()

    duration_days = float(args.duration_days)
    fmin_diag = float(args.fmin)
    fmax_diag = float(args.fmax)
    duration = duration_days * SEC_IN_DAY
    delta_t = 1
    model = "scirdv1"
    unit_label = "freq" if USE_FREQ_UNITS else "strain"

    target_block_seconds = 7.0 * SEC_IN_DAY
    block_len_samples = int(round(target_block_seconds / delta_t))
    chunk_seconds = block_len_samples * delta_t
    total_samples = int(duration / delta_t)
    print(
        f"Total duration: {duration_days:.2f} days "
        f"({total_samples} samples @ {1.0 / delta_t:.1f} Hz)."
    )
    print(
        "Using chunk length = 7 days: "
        f"{block_len_samples} samples ({chunk_seconds:.0f} s)."
    )

    x_t, y_t, z_t, freq_true, S_true = generate_lisatools_xyz_noise_timeseries(
        duration=duration,
        delta_t=delta_t,
        model=model,
        seed=123,
        chunk_seconds=chunk_seconds,
    )

    if args.welch_days is None:
        L = int(round(1.0 / (delta_t * fmin_diag)))
    else:
        L = int(round(float(args.welch_days) * 86_400.0 / delta_t))
    if L > len(x_t):
        L = len(x_t)
    overlap = float(args.welch_overlap)
    if not (0.0 <= overlap < 1.0):
        raise ValueError("welch-overlap must be in [0, 1).")
    n_chunks = len(x_t) // block_len_samples
    if n_chunks < 1:
        n_chunks = 1
        block_len_samples = len(x_t)
    if L > block_len_samples:
        L = block_len_samples
    print(
        "Welch segment length: "
        f"{L} samples ({L * delta_t:.0f} s), overlap={overlap:.2f}."
    )
    print(
        f"Welch averaging across {n_chunks} chunk(s) of {block_len_samples} samples."
    )

    x_chunks = x_t[: n_chunks * block_len_samples].reshape(
        n_chunks, block_len_samples
    )
    y_chunks = y_t[: n_chunks * block_len_samples].reshape(
        n_chunks, block_len_samples
    )
    z_chunks = z_t[: n_chunks * block_len_samples].reshape(
        n_chunks, block_len_samples
    )

    Sxx = Syy = Szz = 0.0
    Sxy = Syz = Szx = 0.0
    freq_est = None
    for idx in range(n_chunks):
        freq_chunk, Sxx_i, Syy_i, Szz_i, Sxy_i, Syz_i, Szx_i = (
            welch_spectral_matrix_xyz(
                x_chunks[idx],
                y_chunks[idx],
                z_chunks[idx],
                L=L,
                delta_t=delta_t,
                overlap=overlap,
            )
        )
        if freq_est is None:
            freq_est = freq_chunk
        Sxx += Sxx_i
        Syy += Syy_i
        Szz += Szz_i
        Sxy += Sxy_i
        Syz += Syz_i
        Szx += Szx_i

    Sxx /= n_chunks
    Syy /= n_chunks
    Szz /= n_chunks
    Sxy /= n_chunks
    Syz /= n_chunks
    Szx /= n_chunks

    freq_mask = (freq_est >= fmin_diag) & (freq_est <= fmax_diag)
    if not np.any(freq_mask):
        raise ValueError("Frequency mask removed all bins; check fmin/fmax.")
    freq_est = freq_est[freq_mask]
    Sxx = Sxx[freq_mask]
    Syy = Syy[freq_mask]
    Szz = Szz[freq_mask]
    Sxy = Sxy[freq_mask]
    Syz = Syz[freq_mask]
    Szx = Szx[freq_mask]

    model_checked = lisa_models.check_lisa_model(model)
    sens_est = XYZ2SensitivityMatrix(freq_est, model=model_checked)
    S_true_est = np.transpose(sens_est.sens_mat, (2, 0, 1))

    emp_matrix = spectral_matrix_from_components(Sxx, Syy, Szz, Sxy, Syz, Szx)
    true_matrix = S_true
    true_matrix_freq = None
    if USE_FREQ_UNITS:
        S_true_est = strain_to_freq_psd(S_true_est, freq_est)
        emp_matrix = strain_to_freq_psd(emp_matrix, freq_est)
        true_matrix_freq = strain_to_freq_psd(true_matrix, freq_true)

    Sxx = emp_matrix[:, 0, 0].real
    Syy = emp_matrix[:, 1, 1].real
    Szz = emp_matrix[:, 2, 2].real
    Sxy = emp_matrix[:, 0, 1]
    Syz = emp_matrix[:, 1, 2]
    Szx = emp_matrix[:, 2, 0]

    summarize_ratio(
        f"X PSD Welch/True ({unit_label})",
        Sxx / S_true_est[:, 0, 0].real,
    )
    summarize_ratio(
        f"Y PSD Welch/True ({unit_label})",
        Syy / S_true_est[:, 1, 1].real,
    )
    summarize_ratio(
        f"Z PSD Welch/True ({unit_label})",
        Szz / S_true_est[:, 2, 2].real,
    )
    summarize_ratio(
        f"XY |CSD| Welch/True ({unit_label})",
        np.abs(Sxy) / np.abs(S_true_est[:, 0, 1]),
    )
    summarize_ratio(
        f"YZ |CSD| Welch/True ({unit_label})",
        np.abs(Syz) / np.abs(S_true_est[:, 1, 2]),
    )
    summarize_ratio(
        f"ZX |CSD| Welch/True ({unit_label})",
        np.abs(Szx) / np.abs(S_true_est[:, 2, 0]),
    )

    S_emp = {
        "Sxx": Sxx,
        "Syy": Syy,
        "Szz": Szz,
        "Sxy": Sxy,
        "Syz": Syz,
        "Szx": Szx,
    }
    out = RESULTS_DIR / "lisatools_synth_psd_coherence.png"
    plot_psd_coherence(
        freq_est,
        S_true_est,
        S_emp,
        fname=out,
        psd_unit_label=("Hz^2/Hz" if USE_FREQ_UNITS else "1/Hz"),
    )
    print(f"Saved lisatools PSD/coherence plot to {out}")

    if SAVE_NPZ:
        time = np.arange(len(x_t)) * delta_t
        data = np.vstack((x_t, y_t, z_t)).T.astype(np.float32)
        np.savez_compressed(
            NPZ_PATH,
            time=time,
            data=data,
            freq_true=freq_true,
            true_matrix=true_matrix,
            true_matrix_freq=true_matrix_freq,
            delta_t=delta_t,
            model=model,
            use_freq_units=USE_FREQ_UNITS,
            block_len_samples=block_len_samples,
            block_seconds=chunk_seconds,
        )
        print(f"Saved synthetic lisatools data to {NPZ_PATH}")


if __name__ == "__main__":
    main()
