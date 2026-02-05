"""Generate paper-sized LISA XYZ synthetic data with lisatools.

This script generates a 3-channel XYZ time series directly at the requested
cadence (no naive downsampling), and writes an ``.npz`` bundle suitable for
``paper_plots``:

- ``time``: (N,)
- ``data``: (N, 3)
- ``freq_true`` / ``true_matrix``: analytic spectral matrix over a user-chosen
  frequency band (used for plotting overlays).

Internally it uses lisatools' ``XYZ2SensitivityMatrix`` to construct the
frequency-domain covariance for a configurable FFT block size, then draws
Fourier coefficients with that covariance and transforms back to the time
domain. This mirrors the approach in ``docs/studies/lisa/lisatools_synth_check.py``
but is focused on producing paper-ready datasets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from log_psplines.example_datasets.lisa_data import (
    plot_psd_coherence,
    spectral_matrix_from_components,
    welch_spectral_matrix_xyz,
)
from log_psplines.example_datasets.lisatools_backend import (
    ensure_lisatools_backends,
)

ensure_lisatools_backends()

from lisatools import detector as lisa_models  # noqa: E402
from lisatools.sensitivity import XYZ2SensitivityMatrix  # noqa: E402

WEEK_SECONDS = 7.0 * 86_400.0


def _resolve_duration_seconds(
    *,
    duration_weeks: float | None,
    duration_days: float | None,
    n: int | None,
    dt: float,
) -> tuple[int, float]:
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("--delta-t must be positive.")

    if n is not None:
        n_time_i = int(n)
        if n_time_i <= 0:
            raise ValueError("--n-time must be positive.")
        return n_time_i, n_time_i * dt

    if duration_weeks is not None:
        dur = float(duration_weeks) * WEEK_SECONDS
    elif duration_days is not None:
        dur = float(duration_days) * 86_400.0
    else:
        raise ValueError(
            "Provide --duration-weeks, --duration-days, or --n-time."
        )

    if not np.isfinite(dur) or dur <= 0.0:
        raise ValueError("Duration must be positive.")

    n_time_i = int(dur / dt)
    if n_time_i <= 1:
        raise ValueError("Duration too short for the requested delta-t.")
    return n_time_i, n_time_i * dt


def _trim_to_multiple(n: int, multiple: int) -> int:
    n = int(n)
    multiple = int(multiple)
    if multiple <= 0:
        raise ValueError("multiple must be positive.")
    n_used = n - (n % multiple)
    return n_used if n_used > 0 else multiple


def _draw_fft_noise(chol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = int(chol.shape[0])
    eps = rng.normal(0.0, 1.0 / math.sqrt(2.0), (3, N)) + 1j * rng.normal(
        0.0, 1.0 / math.sqrt(2.0), (3, N)
    )
    eps[:, 0] = rng.normal(0.0, 1.0, 3)
    eps[:, -1] = rng.normal(0.0, 1.0, 3)
    return np.einsum("fij,jf->if", chol, eps)  # (3, N)


def generate_xyz_chunks(
    *,
    n_total: int,
    dt: float,
    chunk_len: int,
    model: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate XYZ time series by concatenating independent FFT chunks."""
    n_total = int(n_total)
    chunk_len = int(chunk_len)
    dt = float(dt)

    if n_total <= 1:
        raise ValueError("n_total must be > 1.")
    if chunk_len <= 2:
        raise ValueError("chunk_len must be > 2.")
    n_chunks = n_total // chunk_len
    if n_chunks < 1:
        raise ValueError("Total length shorter than one chunk.")
    n_used = n_chunks * chunk_len

    freq_chunk = np.fft.rfftfreq(chunk_len, d=dt)
    if freq_chunk.size < 2:
        raise ValueError("Chunk length too short for rFFT.")
    freq_eval = freq_chunk.copy()
    if freq_eval[0] == 0.0:
        freq_eval[0] = freq_eval[1]

    model_checked = lisa_models.check_lisa_model(model)
    sens_chunk = XYZ2SensitivityMatrix(freq_eval, model=model_checked)
    S_true_chunk = np.transpose(sens_chunk.sens_mat, (2, 0, 1))

    cov_fft = (chunk_len / (2.0 * dt)) * np.asarray(
        S_true_chunk, dtype=np.complex128
    )
    chol = np.linalg.cholesky(cov_fft)

    rng = np.random.default_rng(int(seed))
    data = np.empty((n_used, 3), dtype=np.float32)

    for idx in range(n_chunks):
        noise_fft = _draw_fft_noise(chol, rng)
        start = idx * chunk_len
        end = start + chunk_len
        data[start:end, 0] = np.fft.irfft(noise_fft[0], n=chunk_len).astype(
            np.float32
        )
        data[start:end, 1] = np.fft.irfft(noise_fft[1], n=chunk_len).astype(
            np.float32
        )
        data[start:end, 2] = np.fft.irfft(noise_fft[2], n=chunk_len).astype(
            np.float32
        )

    time = (np.arange(n_used) * dt).astype(np.float64)
    return time, data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument(
        "--duration-weeks",
        type=float,
        default=None,
        help="Duration in weeks (alternative to --n-time).",
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        default=None,
        help="Duration in days (alternative to --n-time).",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=None,
        help="Explicit number of samples (overrides duration args).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5000,
        help="Chunk length in samples (also stored as Lb).",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=5.0,
        help="Sample spacing in seconds (default: 5.0).",
    )
    parser.add_argument("--fmin", type=float, default=1e-4)
    parser.add_argument("--fmax", type=float, default=1e-1)
    parser.add_argument(
        "--write-check-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a Welch vs analytic PSD/coherence triangle plot.",
    )
    parser.add_argument(
        "--welch-overlap",
        type=float,
        default=0.0,
        help="Welch overlap fraction in [0,1).",
    )
    parser.add_argument(
        "--welch-seg-len",
        type=int,
        default=None,
        help="Welch segment length in samples (default: ceil(1/(dt*fmin))).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", type=str, default="scirdv1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file.",
    )

    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        print(f"{out} exists; skipping (pass --overwrite to regenerate).")
        return

    dt = float(args.delta_t)
    chunk_len = int(args.block_size)
    n_target, duration_seconds = _resolve_duration_seconds(
        duration_weeks=args.duration_weeks,
        duration_days=args.duration_days,
        n=args.n,
        dt=dt,
    )
    n_used = _trim_to_multiple(n_target, chunk_len)
    duration_days = duration_seconds / 86_400.0

    time, data = generate_xyz_chunks(
        n_total=int(n_used),
        dt=float(dt),
        chunk_len=int(chunk_len),
        model=str(args.model),
        seed=int(args.seed),
    )

    # Store analytic PSD/CSD matrix only in the requested band on the chunk FFT grid.
    freq_chunk = np.fft.rfftfreq(int(chunk_len), d=float(dt))[1:]
    if freq_chunk.size == 0:
        raise ValueError(
            "Chunk length too short to retain positive frequencies."
        )
    fmin = float(args.fmin)
    fmax = float(args.fmax)
    if fmax <= fmin:
        raise ValueError("--fmax must exceed --fmin.")

    nyq = 0.5 / float(dt)
    if fmax > nyq:
        fmax = nyq

    mask = (freq_chunk >= fmin) & (freq_chunk <= fmax)
    freq_true = freq_chunk[mask]
    if freq_true.size == 0:
        raise ValueError(
            "No frequencies remain in [fmin, fmax] for the chosen delta-t/chunk length."
        )

    model_checked = lisa_models.check_lisa_model(str(args.model))
    sens_true = XYZ2SensitivityMatrix(freq_true, model=model_checked)
    true_matrix = np.transpose(sens_true.sens_mat, (2, 0, 1))

    if bool(args.write_check_plot):
        overlap = float(args.welch_overlap)
        if not (0.0 <= overlap < 1.0):
            raise ValueError("--welch-overlap must be in [0,1).")
        if args.welch_seg_len is None:
            seg_len = int(math.ceil(1.0 / (float(dt) * float(args.fmin))))
        else:
            seg_len = int(args.welch_seg_len)
        seg_len = max(2, min(seg_len, int(data.shape[0])))

        x = np.asarray(data[:, 0], dtype=float)
        y = np.asarray(data[:, 1], dtype=float)
        z = np.asarray(data[:, 2], dtype=float)
        freq_est, Sxx, Syy, Szz, Sxy, Syz, Szx = welch_spectral_matrix_xyz(
            x, y, z, L=seg_len, delta_t=float(dt), overlap=overlap
        )
        freq_mask = (freq_est >= float(args.fmin)) & (
            freq_est <= float(args.fmax)
        )
        if not np.any(freq_mask):
            raise ValueError("Welch frequency mask removed all bins.")
        freq_plot = np.asarray(freq_est[freq_mask], dtype=float)
        Sxx = Sxx[freq_mask]
        Syy = Syy[freq_mask]
        Szz = Szz[freq_mask]
        Sxy = Sxy[freq_mask]
        Syz = Syz[freq_mask]
        Szx = Szx[freq_mask]

        sens_plot = XYZ2SensitivityMatrix(freq_plot, model=model_checked)
        S_true_plot = np.transpose(sens_plot.sens_mat, (2, 0, 1))

        emp_matrix = spectral_matrix_from_components(
            Sxx, Syy, Szz, Sxy, Syz, Szx
        )
        S_emp = {
            "Sxx": np.real(emp_matrix[:, 0, 0]),
            "Syy": np.real(emp_matrix[:, 1, 1]),
            "Szz": np.real(emp_matrix[:, 2, 2]),
            "Sxy": emp_matrix[:, 0, 1],
            "Syz": emp_matrix[:, 1, 2],
            "Szx": emp_matrix[:, 2, 0],
        }
        plot_path = out.with_suffix("").with_name(
            out.with_suffix("").name + "_welch_vs_true.png"
        )
        plot_psd_coherence(
            freq_plot,
            S_true_plot,
            S_emp,
            fname=plot_path,
            psd_unit_label="1/Hz",
        )
        print(f"Wrote Welch-vs-true plot to {plot_path}.")

    np.savez_compressed(
        out,
        time=time,
        data=data,
        freq_true=freq_true,
        true_matrix=true_matrix,
        delta_t=float(dt),
        model=str(args.model),
        Lb=int(chunk_len),
        n_chunks=int(n_used // chunk_len),
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        duration_days=float(duration_days),
    )
    print(
        f"Wrote {out} (N={n_used}, dt={dt:g}s, duration_days={duration_days:.2f}, chunk_len={chunk_len})."
    )


if __name__ == "__main__":
    main()
