"""Generate lisatools-based XYZ noise and compare Welch vs true PSD/CSD."""

from __future__ import annotations

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
        "lisatools is required for this script. Install it and re-run."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.datatypes import MultivariateTimeseries  # noqa: E402
from log_psplines.example_datasets.lisa_data import (  # noqa: E402
    plot_psd_coherence,
    spectral_matrix_from_components,
    strain_to_freq_psd,
    welch_spectral_matrix_xyz,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "lisa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

USE_FREQ_UNITS = False  # True => PSD/CSD in "Hz^2/Hz" instead of "1/Hz" for better dynamic range visualization.
SAVE_NPZ = True
NPZ_PATH = RESULTS_DIR / "lisa_data.npz"
SEC_IN_DAY = 86_400.0

# Study configuration (no CLI args by design).
DURATION_DAYS = 365.0
FMIN_DIAG = 1e-4
FMAX_DIAG = 1e-1
DELTA_T = 5.0
MODEL = "scirdv1"
SEED = 123

# Frequency band used when drawing FFT coefficients.
GENERATION_FMIN = 1e-5
GENERATION_FMAX = FMAX_DIAG

# Cholesky regularization for numerical robustness.
CHOLESKY_FLOOR_REL = 1e-12
CHOLESKY_FLOOR_ABS = 0.0

# Internal frequency-chunk processing only controls memory footprint.
# It does not create/disconnect time-domain chunks.
FREQ_CHUNK_SIZE = 200_000

# Welch diagnostics settings.
WELCH_SEGMENT_DAYS: float | None = None  # None => derive from FMIN_DIAG
WELCH_OVERLAP = 0.0
POSTPROCESS_BLOCK_DAYS = 7.0


def _bytes_to_gib(n_bytes: float) -> float:
    return float(n_bytes) / float(1024**3)


def estimate_h5_filesize_gib(
    *,
    duration_days: float,
    delta_t: float,
    fmin_generate: float,
    fmax_generate: float,
    use_freq_units: bool,
) -> dict[str, float]:
    """Estimate uncompressed HDF5 file size for the generated payload."""
    n = int(float(duration_days) * SEC_IN_DAY / float(delta_t))
    if n <= 1:
        raise ValueError("Need at least 2 samples to estimate output size.")
    n_freq = int(n // 2 + 1)
    nyq = 0.5 / float(delta_t)
    fmin_eff = max(float(fmin_generate), 0.0)
    fmax_eff = min(float(fmax_generate), nyq)
    if fmax_eff <= fmin_eff:
        n_active = 0
    else:
        df = 1.0 / (float(n) * float(delta_t))
        k_lo = max(1, int(np.ceil(fmin_eff / df)))
        k_hi = min(n_freq - 1, int(np.floor(fmax_eff / df)))
        n_active = max(0, k_hi - k_lo + 1)

    # Arrays stored in the output payload:
    # time(float64), data(float32, N x 3), freq_true(float64),
    # true_matrix(complex128, N_active x 3 x 3), true_matrix_freq (optional).
    payload_bytes = 8 * n + 4 * 3 * n + 8 * n_active + 16 * 9 * n_active
    if use_freq_units:
        payload_bytes += 16 * 9 * n_active

    # Modest metadata allowance for HDF5 object headers.
    metadata_bytes = float(2 * 1024**2)
    total_bytes = float(payload_bytes) + metadata_bytes

    return {
        "n_samples": float(n),
        "n_active": float(n_active),
        "payload_gib": _bytes_to_gib(payload_bytes),
        "metadata_gib": _bytes_to_gib(metadata_bytes),
        "total_gib": _bytes_to_gib(total_bytes),
    }


def estimate_generation_memory_gib(
    *,
    duration_days: float,
    delta_t: float,
    fmin_generate: float,
    fmax_generate: float,
    freq_chunk_size: int,
    use_freq_units: bool,
) -> dict[str, float]:
    """Approximate generation memory footprint (GiB).

    This is an estimate from array shapes/dtypes in this script, not a strict
    upper bound. Real usage depends on NumPy/lisatools temporary allocations.
    """
    n = int(float(duration_days) * SEC_IN_DAY / float(delta_t))
    if n <= 1:
        raise ValueError("Need at least 2 samples to estimate memory.")
    n_freq = int(n // 2 + 1)
    nyq = 0.5 / float(delta_t)
    fmin_eff = max(float(fmin_generate), 0.0)
    fmax_eff = min(float(fmax_generate), nyq)
    if fmax_eff <= fmin_eff:
        n_active = 0
    else:
        df = 1.0 / (float(n) * float(delta_t))
        k_lo = max(1, int(np.ceil(fmin_eff / df)))
        k_hi = min(n_freq - 1, int(np.floor(fmax_eff / df)))
        n_active = max(0, k_hi - k_lo + 1)

    chunk = max(1, min(int(freq_chunk_size), max(n_active, 1)))

    # Phase A: chunked covariance/Cholesky loop with persistent FFT storage.
    phase_a = (
        8 * n_freq  # freq
        + 8 * n_freq  # freq_eval
        + 1 * n_freq  # active mask
        + 8 * n_active  # active indices
        + 8 * 3 * n_freq  # noise_fft complex64
        + 8 * chunk  # idx
        + 8 * chunk  # freq_chunk
        + 16 * 9 * chunk  # s_chunk complex128
        + 16 * 9 * chunk  # cov_chunk complex128
        + 16 * 9 * chunk  # chol_chunk complex128
        + 8 * 3 * chunk  # eigvals float64
        + 16 * 3 * chunk  # eps complex128
        + 16 * 3 * chunk  # coeff_chunk complex128
    )

    # Phase B: materialized time series + stored true spectral matrix.
    phase_b = (
        8 * n_freq  # freq
        + 8 * n_freq  # freq_eval
        + 1 * n_freq  # active mask
        + 8 * n_active  # active indices
        + 8 * 3 * n_freq  # noise_fft
        + 8 * 3 * n  # x_t, y_t, z_t float64
        + 8 * n_active  # freq_true
        + 16 * 9 * n_active  # true_matrix complex128
    )
    if use_freq_units:
        phase_b += 16 * 9 * n_active  # true_matrix_freq

    peak = max(phase_a, phase_b)
    # Safety headroom for hidden temporaries and allocator overhead.
    peak_with_headroom = 1.25 * peak
    return {
        "n_samples": float(n),
        "n_freq": float(n_freq),
        "n_active": float(n_active),
        "phase_a_gib": _bytes_to_gib(phase_a),
        "phase_b_gib": _bytes_to_gib(phase_b),
        "peak_gib": _bytes_to_gib(peak),
        "peak_with_headroom_gib": _bytes_to_gib(peak_with_headroom),
    }


def summarize_ratio(label: str, ratio: np.ndarray) -> None:
    clean = ratio[np.isfinite(ratio)]
    if clean.size == 0:
        print(f"{label}: no finite entries to summarize.")
        return
    pct = np.percentile(clean, [5, 50, 95])
    print(f"{label}: p05={pct[0]:.3g}, p50={pct[1]:.3g}, p95={pct[2]:.3g}")


def _interp_real(
    freq_src: np.ndarray, vals: np.ndarray, freq_tgt: np.ndarray
) -> np.ndarray:
    return np.interp(
        np.asarray(freq_tgt, dtype=np.float64),
        np.asarray(freq_src, dtype=np.float64),
        np.asarray(vals, dtype=np.float64),
    )


def _interp_complex(
    freq_src: np.ndarray, vals: np.ndarray, freq_tgt: np.ndarray
) -> np.ndarray:
    vals_arr = np.asarray(vals, dtype=np.complex128)
    real = np.interp(
        np.asarray(freq_tgt, dtype=np.float64),
        np.asarray(freq_src, dtype=np.float64),
        vals_arr.real,
    )
    imag = np.interp(
        np.asarray(freq_tgt, dtype=np.float64),
        np.asarray(freq_src, dtype=np.float64),
        vals_arr.imag,
    )
    return real + 1j * imag


def _one_sided_periodogram_components(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    delta_t: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return one-sided periodogram PSD/CSD components for XYZ."""
    n = int(x.shape[0])
    if n <= 1:
        raise ValueError("Need at least 2 samples for periodogram.")
    Xf = np.fft.rfft(x)
    Yf = np.fft.rfft(y)
    Zf = np.fft.rfft(z)
    scale = float(delta_t) / float(n)

    Sxx = scale * (np.abs(Xf) ** 2)
    Syy = scale * (np.abs(Yf) ** 2)
    Szz = scale * (np.abs(Zf) ** 2)
    Sxy = scale * (Xf * np.conj(Yf))
    Syz = scale * (Yf * np.conj(Zf))
    Szx = scale * (Zf * np.conj(Xf))

    if n > 2:
        Sxx[1:-1] *= 2.0
        Syy[1:-1] *= 2.0
        Szz[1:-1] *= 2.0
        Sxy[1:-1] *= 2.0
        Syz[1:-1] *= 2.0
        Szx[1:-1] *= 2.0

    freq = np.fft.rfftfreq(n, d=float(delta_t))
    return freq[1:], Sxx[1:], Syy[1:], Szz[1:], Sxy[1:], Syz[1:], Szx[1:]


def generate_lisatools_xyz_noise_timeseries(
    duration: float,
    delta_t: float,
    model: str = "scirdv1",
    seed: int | None = None,
    return_freq_noise: bool = False,
    fmin_generate: float = 1e-4,
    fmax_generate: float = 1e-1,
    cholesky_floor_rel: float = 1e-12,
    cholesky_floor_abs: float = 0.0,
    freq_chunk_size: int = 200_000,
):
    """
    Generate X,Y,Z TDI noise with PSD matrix from lisatools.

    Var[FFT[x]] = (N / (2 * delta_t)) * S(f)
    """
    n = int(duration / delta_t)
    if n <= 1:
        raise ValueError("Duration too short for the requested delta_t.")
    if freq_chunk_size <= 0:
        raise ValueError("freq_chunk_size must be positive.")

    freq = np.fft.rfftfreq(n, d=delta_t)
    if freq.size < 2:
        raise ValueError("Duration too short to retain positive frequencies.")
    freq_eval = freq.copy()
    if freq_eval[0] == 0.0:
        freq_eval[0] = freq_eval[1]

    nyq = 0.5 / float(delta_t)
    fmin_eff = max(float(fmin_generate), 0.0)
    fmax_eff = min(float(fmax_generate), nyq)
    if fmax_eff <= fmin_eff:
        raise ValueError(
            f"Invalid generation band [{fmin_generate}, {fmax_generate}] for delta_t={delta_t:g} (Nyquist={nyq:g})."
        )
    active_mask = (freq >= fmin_eff) & (freq <= fmax_eff)
    active_mask[0] = False
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        raise ValueError(
            "No active FFT bins in generation band. Increase duration or widen [fmin, fmax]."
        )

    model_checked = lisa_models.check_lisa_model(model)
    floor_rel = float(cholesky_floor_rel)
    floor_abs = float(cholesky_floor_abs)
    eye = np.eye(3, dtype=np.complex128)
    fft_scale = n / (2.0 * float(delta_t))

    rng = np.random.default_rng(seed)
    noise_fft = np.zeros((3, freq.size), dtype=np.complex64)
    for start in range(0, active_idx.size, int(freq_chunk_size)):
        stop = min(start + int(freq_chunk_size), active_idx.size)
        idx = active_idx[start:stop]
        freq_chunk = np.asarray(freq_eval[idx], dtype=np.float64)

        sens_chunk = XYZ2SensitivityMatrix(freq_chunk, model=model_checked)
        s_chunk = np.transpose(sens_chunk.sens_mat, (2, 0, 1)).astype(
            np.complex128
        )
        cov_chunk = fft_scale * s_chunk
        cov_chunk = 0.5 * (cov_chunk + np.conj(np.swapaxes(cov_chunk, 1, 2)))

        if floor_rel > 0.0 or floor_abs > 0.0:
            min_eig = np.linalg.eigvalsh(cov_chunk)[:, 0].real
            diag_scale = np.real(np.trace(cov_chunk, axis1=1, axis2=2)) / 3.0
            target_floor = np.maximum(
                floor_abs, floor_rel * np.maximum(diag_scale, 0.0)
            )
            shift = np.maximum(target_floor - min_eig, 0.0)
            if np.any(shift > 0.0):
                cov_chunk = cov_chunk + shift[:, None, None] * eye[None, :, :]

        try:
            chol_chunk = np.linalg.cholesky(cov_chunk)
        except np.linalg.LinAlgError:
            min_eig = np.linalg.eigvalsh(cov_chunk)[:, 0].real
            diag_scale = np.real(np.trace(cov_chunk, axis1=1, axis2=2)) / 3.0
            fallback_floor = 1e-15 * np.maximum(diag_scale, 1.0)
            shift = np.maximum(fallback_floor - min_eig, 0.0)
            cov_chunk = cov_chunk + shift[:, None, None] * eye[None, :, :]
            chol_chunk = np.linalg.cholesky(cov_chunk)
        n_chunk = idx.size
        eps = rng.normal(0.0, 1.0 / np.sqrt(2.0), (3, n_chunk)) + 1j * (
            rng.normal(0.0, 1.0 / np.sqrt(2.0), (3, n_chunk))
        )
        if n % 2 == 0:
            nyquist_idx = freq.size - 1
            nyq_loc = np.where(idx == nyquist_idx)[0]
            if nyq_loc.size:
                eps[:, int(nyq_loc[0])] = rng.normal(0.0, 1.0, 3)
        coeff_chunk = np.einsum("fij,jf->if", chol_chunk, eps)
        noise_fft[:, idx] = coeff_chunk.astype(np.complex64)

    x_t = np.fft.irfft(noise_fft[0], n=n)
    y_t = np.fft.irfft(noise_fft[1], n=n)
    z_t = np.fft.irfft(noise_fft[2], n=n)

    freq_true = np.asarray(freq[active_mask], dtype=np.float64)
    sens_true = XYZ2SensitivityMatrix(freq_true, model=model_checked)
    S_true = np.transpose(sens_true.sens_mat, (2, 0, 1))

    if return_freq_noise:
        return x_t, y_t, z_t, freq_true, S_true, noise_fft
    return x_t, y_t, z_t, freq_true, S_true


def main() -> None:
    duration_days = float(DURATION_DAYS)
    fmin_diag = float(FMIN_DIAG)
    fmax_diag = float(FMAX_DIAG)
    duration = duration_days * SEC_IN_DAY
    delta_t = float(DELTA_T)
    model = str(MODEL)
    unit_label = "freq" if USE_FREQ_UNITS else "strain"
    gen_fmin = float(GENERATION_FMIN)
    gen_fmax = float(GENERATION_FMAX)

    target_block_seconds = float(POSTPROCESS_BLOCK_DAYS) * SEC_IN_DAY
    Lb = int(round(target_block_seconds / delta_t))
    block_seconds = Lb * delta_t
    total_samples = int(duration / delta_t)
    mem_est = estimate_generation_memory_gib(
        duration_days=duration_days,
        delta_t=delta_t,
        fmin_generate=gen_fmin,
        fmax_generate=gen_fmax,
        freq_chunk_size=int(FREQ_CHUNK_SIZE),
        use_freq_units=bool(USE_FREQ_UNITS),
    )
    h5_est = estimate_h5_filesize_gib(
        duration_days=duration_days,
        delta_t=delta_t,
        fmin_generate=gen_fmin,
        fmax_generate=gen_fmax,
        use_freq_units=bool(USE_FREQ_UNITS),
    )
    print(
        f"Total duration: {duration_days:.2f} days "
        f"({total_samples} samples @ {1.0 / delta_t:.1f} Hz)."
    )
    print(
        "Estimated RAM (rough worst-case, GiB): "
        f"~{mem_est['peak_with_headroom_gib']:.2f}."
    )
    print(
        "Estimated output file size (rough worst-case, GiB): "
        f"~{h5_est['total_gib']:.2f}."
    )
    print(
        f"Postprocessing block length = {float(POSTPROCESS_BLOCK_DAYS):g} days: "
        f"{Lb} samples ({block_seconds:.0f} s)."
    )

    x_t, y_t, z_t, freq_true, S_true = generate_lisatools_xyz_noise_timeseries(
        duration=duration,
        delta_t=delta_t,
        model=model,
        seed=int(SEED),
        fmin_generate=gen_fmin,
        fmax_generate=gen_fmax,
        cholesky_floor_rel=float(CHOLESKY_FLOOR_REL),
        cholesky_floor_abs=float(CHOLESKY_FLOOR_ABS),
        freq_chunk_size=int(FREQ_CHUNK_SIZE),
    )

    if WELCH_SEGMENT_DAYS is None:
        L = int(round(1.0 / (delta_t * fmin_diag)))
    else:
        L = int(round(float(WELCH_SEGMENT_DAYS) * 86_400.0 / delta_t))
    if len(x_t) < L:
        L = len(x_t)
    overlap = float(WELCH_OVERLAP)
    if not (0.0 <= overlap < 1.0):
        raise ValueError("welch-overlap must be in [0, 1).")
    Nb = len(x_t) // Lb
    if Nb < 1:
        Nb = 1
        Lb = len(x_t)
    n_used = Nb * Lb
    if n_used != len(x_t):
        n_trim = len(x_t) - n_used
        print(
            f"Trimming {n_trim} samples for block-consistent analysis "
            f"(Nb={Nb}, Lb={Lb})."
        )
    block_seconds = Lb * delta_t
    if Lb < L:
        L = Lb
    print(
        "Welch segment length: "
        f"{L} samples ({L * delta_t:.0f} s), overlap={overlap:.2f}."
    )
    print(f"Welch averaging across {Nb} block(s) of {Lb} samples.")

    x_blocks = x_t[: Nb * Lb].reshape(Nb, Lb)
    y_blocks = y_t[: Nb * Lb].reshape(Nb, Lb)
    z_blocks = z_t[: Nb * Lb].reshape(Nb, Lb)

    Sxx = Syy = Szz = 0.0
    Sxy = Syz = Szx = 0.0
    freq_est = None
    for idx in range(Nb):
        freq_block, Sxx_i, Syy_i, Szz_i, Sxy_i, Syz_i, Szx_i = (
            welch_spectral_matrix_xyz(
                x_blocks[idx],
                y_blocks[idx],
                z_blocks[idx],
                L=L,
                delta_t=delta_t,
                overlap=overlap,
            )
        )
        if freq_est is None:
            freq_est = freq_block
        Sxx += Sxx_i
        Syy += Syy_i
        Szz += Szz_i
        Sxy += Sxy_i
        Syz += Syz_i
        Szx += Szx_i

    Sxx /= Nb
    Syy /= Nb
    Szz /= Nb
    Sxy /= Nb
    Syz /= Nb
    Szx /= Nb

    freq_raw, Sxx_raw, Syy_raw, Szz_raw, Sxy_raw, Syz_raw, Szx_raw = (
        _one_sided_periodogram_components(
            x_t, y_t, z_t, delta_t=float(delta_t)
        )
    )

    model_checked = lisa_models.check_lisa_model(model)
    freq_ref = np.geomspace(float(fmin_diag), float(fmax_diag), num=2000)
    sens_ref = XYZ2SensitivityMatrix(freq_ref, model=model_checked)
    S_true_ref = np.transpose(sens_ref.sens_mat, (2, 0, 1))
    if USE_FREQ_UNITS:
        S_true_ref = strain_to_freq_psd(S_true_ref, freq_ref)

    def _summarize_and_plot(
        *,
        slug: str,
        display: str,
        freq_emp: np.ndarray,
        Sxx_emp: np.ndarray,
        Syy_emp: np.ndarray,
        Szz_emp: np.ndarray,
        Sxy_emp: np.ndarray,
        Syz_emp: np.ndarray,
        Szx_emp: np.ndarray,
    ) -> None:
        mask = (freq_emp >= fmin_diag) & (freq_emp <= fmax_diag)
        if not np.any(mask):
            raise ValueError(
                f"Frequency mask removed all bins for {display}; check fmin/fmax."
            )

        freq_use = np.asarray(freq_emp[mask], dtype=float)
        Sxx_use = Sxx_emp[mask]
        Syy_use = Syy_emp[mask]
        Szz_use = Szz_emp[mask]
        Sxy_use = Sxy_emp[mask]
        Syz_use = Syz_emp[mask]
        Szx_use = Szx_emp[mask]

        emp_matrix = spectral_matrix_from_components(
            Sxx_use, Syy_use, Szz_use, Sxy_use, Syz_use, Szx_use
        )
        if USE_FREQ_UNITS:
            emp_matrix = strain_to_freq_psd(emp_matrix, freq_use)

        Sxx_plot = _interp_real(freq_use, emp_matrix[:, 0, 0].real, freq_ref)
        Syy_plot = _interp_real(freq_use, emp_matrix[:, 1, 1].real, freq_ref)
        Szz_plot = _interp_real(freq_use, emp_matrix[:, 2, 2].real, freq_ref)
        Sxy_plot = _interp_complex(freq_use, emp_matrix[:, 0, 1], freq_ref)
        Syz_plot = _interp_complex(freq_use, emp_matrix[:, 1, 2], freq_ref)
        Szx_plot = _interp_complex(freq_use, emp_matrix[:, 2, 0], freq_ref)

        summarize_ratio(
            f"X PSD {display}/True ({unit_label})",
            Sxx_plot / S_true_ref[:, 0, 0].real,
        )
        summarize_ratio(
            f"Y PSD {display}/True ({unit_label})",
            Syy_plot / S_true_ref[:, 1, 1].real,
        )
        summarize_ratio(
            f"Z PSD {display}/True ({unit_label})",
            Szz_plot / S_true_ref[:, 2, 2].real,
        )
        summarize_ratio(
            f"XY |CSD| {display}/True ({unit_label})",
            np.abs(Sxy_plot) / np.abs(S_true_ref[:, 0, 1]),
        )
        summarize_ratio(
            f"YZ |CSD| {display}/True ({unit_label})",
            np.abs(Syz_plot) / np.abs(S_true_ref[:, 1, 2]),
        )
        summarize_ratio(
            f"ZX |CSD| {display}/True ({unit_label})",
            np.abs(Szx_plot) / np.abs(S_true_ref[:, 2, 0]),
        )

        S_emp = {
            "Sxx": Sxx_plot,
            "Syy": Syy_plot,
            "Szz": Szz_plot,
            "Sxy": Sxy_plot,
            "Syz": Syz_plot,
            "Szx": Szx_plot,
        }
        out = RESULTS_DIR / f"lisa_psd_coherence_{slug}.png"
        plot_psd_coherence(
            freq_ref,
            S_true_ref,
            S_emp,
            fname=out,
            psd_unit_label=("Hz^2/Hz" if USE_FREQ_UNITS else "1/Hz"),
            empirical_label=display,
        )
        print(f"Saved {display} PSD/coherence plot to {out}")

    _summarize_and_plot(
        slug="welch",
        display="Welch",
        freq_emp=np.asarray(freq_est, dtype=float),
        Sxx_emp=np.asarray(Sxx),
        Syy_emp=np.asarray(Syy),
        Szz_emp=np.asarray(Szz),
        Sxy_emp=np.asarray(Sxy),
        Syz_emp=np.asarray(Syz),
        Szx_emp=np.asarray(Szx),
    )
    _summarize_and_plot(
        slug="raw_periodogram",
        display="Raw periodogram",
        freq_emp=freq_raw,
        Sxx_emp=Sxx_raw,
        Syy_emp=Syy_raw,
        Szz_emp=Szz_raw,
        Sxy_emp=Sxy_raw,
        Syz_emp=Syz_raw,
        Szx_emp=Szx_raw,
    )
    ts = MultivariateTimeseries(
        y=np.vstack((x_t[:n_used], y_t[:n_used], z_t[:n_used])).T.astype(
            np.float64
        ),
        t=(np.arange(n_used, dtype=np.float64) * float(delta_t)),
    )
    ts_std = ts.standardise_for_psd()
    fft_analysis = ts_std.to_wishart_stats(
        Nb=int(Nb),
        fmin=float(fmin_diag),
        fmax=float(fmax_diag),
        window="hann",
    )
    psd_analysis = fft_analysis.empirical_psd
    _summarize_and_plot(
        slug="analysis_nocg",
        display="Analysis no-CG (Wishart)",
        freq_emp=np.asarray(psd_analysis.freq, dtype=float),
        Sxx_emp=np.asarray(psd_analysis.psd[:, 0, 0]),
        Syy_emp=np.asarray(psd_analysis.psd[:, 1, 1]),
        Szz_emp=np.asarray(psd_analysis.psd[:, 2, 2]),
        Sxy_emp=np.asarray(psd_analysis.psd[:, 0, 1]),
        Syz_emp=np.asarray(psd_analysis.psd[:, 1, 2]),
        Szx_emp=np.asarray(psd_analysis.psd[:, 2, 0]),
    )

    true_matrix = S_true
    true_matrix_freq = (
        strain_to_freq_psd(true_matrix, freq_true) if USE_FREQ_UNITS else None
    )

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
            Nb=Nb,
            Lb=Lb,
            block_seconds=block_seconds,
        )
        print(f"Saved synthetic lisatools data to {NPZ_PATH}")


if __name__ == "__main__":
    main()
