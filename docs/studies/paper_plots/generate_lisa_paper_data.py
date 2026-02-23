"""Generate paper-sized LISA XYZ synthetic data with lisatools.

This script generates one continuous 3-channel XYZ time series at the requested
cadence (no chunk-and-concatenate), and writes an ``.npz`` bundle suitable for
``paper_plots``:

- ``time``: (N,)
- ``data``: (N, 3)
- ``freq_true`` / ``true_matrix``: analytic spectral matrix over a user-chosen
  frequency band (used for plotting overlays).

Internally it uses lisatools' ``XYZ2SensitivityMatrix`` on the full-length FFT
grid, draws correlated Fourier coefficients once, and transforms back to time
domain.
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
CHOLESKY_FLOOR_REL = 1e-12
CHOLESKY_FLOOR_ABS = 0.0
FREQ_CHUNK_SIZE = 200_000
GENERATION_FMIN_FLOOR = 1e-5


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


def generate_xyz_continuous(
    *,
    n_total: int,
    dt: float,
    model: str,
    seed: int,
    fmin_generate: float,
    fmax_generate: float,
    cholesky_floor_rel: float,
    cholesky_floor_abs: float,
    freq_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one continuous XYZ time series from a band-limited FFT draw."""
    n_total = int(n_total)
    dt = float(dt)

    if n_total <= 1:
        raise ValueError("n_total must be > 1.")
    if freq_chunk_size <= 0:
        raise ValueError("freq_chunk_size must be positive.")
    freq = np.fft.rfftfreq(n_total, d=dt)
    if freq.size < 2:
        raise ValueError("n_total too short for rFFT.")

    nyq = 0.5 / dt
    fmin_eff = max(float(fmin_generate), 0.0)
    fmax_eff = min(float(fmax_generate), nyq)
    if fmax_eff <= fmin_eff:
        raise ValueError(
            f"Invalid generation band [{fmin_generate}, {fmax_generate}] for dt={dt:g} (Nyquist={nyq:g})."
        )
    active_mask = (freq >= fmin_eff) & (freq <= fmax_eff)
    if active_mask.size:
        active_mask[0] = False
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        raise ValueError(
            "No active FFT bins in the generation band. Increase duration or widen [fmin, fmax]."
        )

    model_checked = lisa_models.check_lisa_model(model)
    floor_rel = float(cholesky_floor_rel)
    floor_abs = float(cholesky_floor_abs)
    eye = np.eye(3, dtype=np.complex128)
    fft_scale = n_total / (2.0 * dt)

    rng = np.random.default_rng(int(seed))
    noise_fft = np.zeros((3, freq.size), dtype=np.complex64)

    for start in range(0, active_idx.size, int(freq_chunk_size)):
        stop = min(start + int(freq_chunk_size), active_idx.size)
        idx = active_idx[start:stop]
        freq_chunk = np.asarray(freq[idx], dtype=np.float64)

        sens_chunk = XYZ2SensitivityMatrix(freq_chunk, model=model_checked)
        s_chunk = np.transpose(sens_chunk.sens_mat, (2, 0, 1)).astype(
            np.complex128
        )
        cov_chunk = fft_scale * s_chunk
        cov_chunk = 0.5 * (
            cov_chunk + np.conj(np.swapaxes(cov_chunk, 1, 2))
        )

        if floor_rel > 0.0 or floor_abs > 0.0:
            min_eig = np.linalg.eigvalsh(cov_chunk)[:, 0].real
            diag_scale = (
                np.real(np.trace(cov_chunk, axis1=1, axis2=2)) / 3.0
            )
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
        eps = rng.normal(0.0, 1.0 / math.sqrt(2.0), (3, n_chunk)) + 1j * (
            rng.normal(0.0, 1.0 / math.sqrt(2.0), (3, n_chunk))
        )

        if n_total % 2 == 0:
            nyquist_idx = freq.size - 1
            nyq_loc = np.where(idx == nyquist_idx)[0]
            if nyq_loc.size:
                eps[:, int(nyq_loc[0])] = rng.normal(0.0, 1.0, 3)

        coeff_chunk = np.einsum("fij,jf->if", chol_chunk, eps)
        noise_fft[:, idx] = coeff_chunk.astype(np.complex64)

    data = np.empty((n_total, 3), dtype=np.float32)
    data[:, 0] = np.fft.irfft(noise_fft[0], n=n_total).astype(np.float32)
    data[:, 1] = np.fft.irfft(noise_fft[1], n=n_total).astype(np.float32)
    data[:, 2] = np.fft.irfft(noise_fft[2], n=n_total).astype(np.float32)

    time = (np.arange(n_total) * dt).astype(np.float64)
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
        help=(
            "Postprocessing block length in samples (stored as Lb metadata and "
            "used for freq_true grid). Does not affect time-series generation."
        ),
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
    block_len = int(args.block_size)
    if block_len <= 2:
        raise ValueError("--block-size must be > 2.")
    n_target, _ = _resolve_duration_seconds(
        duration_weeks=args.duration_weeks,
        duration_days=args.duration_days,
        n=args.n_time,
        dt=dt,
    )
    n_used = int(n_target)
    duration_days = (n_used * dt) / 86_400.0
    gen_fmin = min(float(args.fmin), float(GENERATION_FMIN_FLOOR))
    gen_fmax = float(args.fmax)

    time, data = generate_xyz_continuous(
        n_total=int(n_used),
        dt=float(dt),
        model=str(args.model),
        seed=int(args.seed),
        fmin_generate=gen_fmin,
        fmax_generate=gen_fmax,
        cholesky_floor_rel=float(CHOLESKY_FLOOR_REL),
        cholesky_floor_abs=float(CHOLESKY_FLOOR_ABS),
        freq_chunk_size=int(FREQ_CHUNK_SIZE),
    )

    # Store analytic PSD/CSD matrix only in the requested band on the
    # postprocessing block FFT grid.
    freq_block = np.fft.rfftfreq(int(block_len), d=float(dt))[1:]
    if freq_block.size == 0:
        raise ValueError(
            "Block length too short to retain positive frequencies."
        )
    fmin = float(args.fmin)
    fmax = float(args.fmax)
    if fmax <= fmin:
        raise ValueError("--fmax must exceed --fmin.")

    nyq = 0.5 / float(dt)
    if fmax > nyq:
        fmax = nyq

    mask = (freq_block >= fmin) & (freq_block <= fmax)
    freq_true = freq_block[mask]
    if freq_true.size == 0:
        raise ValueError(
            "No frequencies remain in [fmin, fmax] for the chosen delta-t/block length."
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

    nb_blocks = max(1, int(n_used // block_len))
    np.savez_compressed(
        out,
        time=time,
        data=data,
        freq_true=freq_true,
        true_matrix=true_matrix,
        delta_t=float(dt),
        model=str(args.model),
        Lb=int(block_len),
        Nb=int(nb_blocks),
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        duration_days=float(duration_days),
    )
    print(
        f"Wrote {out} (N={n_used}, dt={dt:g}s, duration_days={duration_days:.2f}, block_len={block_len})."
    )


if __name__ == "__main__":
    main()
