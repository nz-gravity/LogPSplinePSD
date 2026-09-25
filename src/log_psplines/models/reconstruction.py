"""Memory-bounded spectral matrix reconstruction utilities."""

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from log_psplines.models.matrix import SpectralMatrix


def _psd_chunk_iterator(
    log_delta_sq_samples: np.ndarray,
    theta_re_samples: np.ndarray | None,
    theta_im_samples: np.ndarray | None,
    *,
    n_samps: int,
    chunk_size: int,
):
    """Yield reconstructed PSD chunks with shape (n_samps, chunk, n, n)."""

    N = log_delta_sq_samples.shape[1]
    p = log_delta_sq_samples.shape[2]

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        log_chunk = log_delta_sq_samples[:n_samps, start:end, :]
        theta_re_chunk = (
            theta_re_samples[:n_samps, start:end, :]
            if theta_re_samples is not None
            else None
        )
        theta_im_chunk = (
            theta_im_samples[:n_samps, start:end, :]
            if theta_im_samples is not None
            else None
        )

        psd_chunk = SpectralMatrix(p)(
            log_chunk, theta_re_chunk, theta_im_chunk
        )

        yield start, end, psd_chunk


def reconstruct_psd_matrix(
    log_delta_sq_samples: jnp.ndarray,
    theta_re_samples: jnp.ndarray,
    theta_im_samples: jnp.ndarray,
    n_samples_max: int = 50,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Reconstruct PSD matrices from Cholesky components using NumPy.

    The computation streams over frequency chunks (default 2048 bins) so the
    peak memory stays modest even for very long spectra. Results are returned
    as a ``complex128`` NumPy array of shape
    ``(n_samps, N, p, p)``.
    """
    log_delta_sq_arr = np.asarray(log_delta_sq_samples)
    theta_re_arr = np.asarray(theta_re_samples)
    theta_im_arr = np.asarray(theta_im_samples)

    if log_delta_sq_arr.ndim == 4:
        log_delta_sq_arr = log_delta_sq_arr[0]
    if theta_re_arr.ndim == 4:
        theta_re_arr = theta_re_arr[0]
    if theta_im_arr.ndim == 4:
        theta_im_arr = theta_im_arr[0]

    n_samples, N, p = log_delta_sq_arr.shape
    n_theta = theta_re_arr.shape[2] if theta_re_arr.ndim > 2 else 0
    n_samps = min(int(n_samples_max), int(n_samples))

    if chunk_size is None or chunk_size <= 0:
        chunk_size = N

    log_delta_sq_arr = log_delta_sq_arr[:n_samps]
    theta_re_arr = theta_re_arr[:n_samps]
    theta_im_arr = theta_im_arr[:n_samps]

    psd = np.empty((n_samps, N, p, p), dtype=np.complex128)

    for start, end, psd_chunk in _psd_chunk_iterator(
        log_delta_sq_arr,
        theta_re_arr if n_theta > 0 else None,
        theta_im_arr if n_theta > 0 else None,
        n_samps=n_samps,
        chunk_size=chunk_size,
    ):
        psd[:, start:end] = psd_chunk

    return psd


def compute_psd_quantiles(
    log_delta_sq_samples: jnp.ndarray,
    theta_re_samples: jnp.ndarray,
    theta_im_samples: jnp.ndarray,
    *,
    percentiles: Sequence[float] | None = None,
    n_samples_max: int = 50,
    chunk_size: int = 2048,
    compute_coherence: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Compute PSD (and optional coherence) percentiles without storing all draws.

    Returns
    -------
    psd_real_percentiles : np.ndarray
        Percentiles of the real part of the PSD matrix with shape
        ``(n_percentiles, N, p, p)``.
    psd_imag_percentiles : np.ndarray
        Percentiles of the imaginary part of the PSD matrix with matching shape.
    coherence_percentiles : Optional[np.ndarray]
        When ``compute_coherence`` is ``True`` and ``p > 1``, contains
        percentiles of the coherence matrix; otherwise ``None``.
    """

    if percentiles is None:
        percentiles = [5.0, 50.0, 95.0]

    log_delta_sq_arr = np.asarray(log_delta_sq_samples)
    theta_re_arr = np.asarray(theta_re_samples)
    theta_im_arr = np.asarray(theta_im_samples)

    if log_delta_sq_arr.ndim == 4:
        log_delta_sq_arr = log_delta_sq_arr[0]
    if theta_re_arr.ndim == 4:
        theta_re_arr = theta_re_arr[0]
    if theta_im_arr.ndim == 4:
        theta_im_arr = theta_im_arr[0]

    n_samples, N, p = log_delta_sq_arr.shape
    n_theta = theta_re_arr.shape[2] if theta_re_arr.ndim > 2 else 0
    n_samps = min(int(n_samples_max), int(n_samples))

    if chunk_size is None or chunk_size <= 0:
        chunk_size = N

    log_delta_sq_arr = log_delta_sq_arr[:n_samps]
    theta_re_arr = theta_re_arr[:n_samps]
    theta_im_arr = theta_im_arr[:n_samps]

    n_percentiles = len(percentiles)
    psd_percentiles = np.empty((n_percentiles, N, p, p), dtype=np.float64)
    psd_imag_percentiles = np.empty_like(psd_percentiles)

    coherence_percentiles = (
        np.empty(
            (n_percentiles, N, p, p),
            dtype=np.float64,
        )
        if compute_coherence and p > 1
        else None
    )

    for start, end, psd_chunk in _psd_chunk_iterator(
        log_delta_sq_arr,
        theta_re_arr if n_theta > 0 else None,
        theta_im_arr if n_theta > 0 else None,
        n_samps=n_samps,
        chunk_size=chunk_size,
    ):
        psd_real = psd_chunk.real
        psd_imag = psd_chunk.imag

        real_q = np.percentile(psd_real, percentiles, axis=0)
        imag_q = np.percentile(psd_imag, percentiles, axis=0)

        psd_percentiles[:, start:end] = real_q
        psd_imag_percentiles[:, start:end] = imag_q

        if coherence_percentiles is not None:
            diag = np.abs(
                np.diagonal(psd_chunk, axis1=2, axis2=3)
            )  # (samples, chunk, channels)
            denom = diag[..., :, None] * diag[..., None, :]
            denom = np.where(denom > 0.0, denom, np.nan)
            coh_samples = (np.abs(psd_chunk) ** 2) / denom
            coh_samples = np.nan_to_num(coh_samples, nan=0.0, posinf=0.0)
            coh_q = np.percentile(coh_samples, percentiles, axis=0)

            # enforce exact ones on diagonal to avoid numerical drift
            for idx in range(n_percentiles):
                for c in range(p):
                    coh_q[idx, :, c, c] = 1.0

            coherence_percentiles[:, start:end] = coh_q

    return psd_percentiles, psd_imag_percentiles, coherence_percentiles
