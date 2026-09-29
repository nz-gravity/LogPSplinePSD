"""Memory-bounded spectral matrix reconstruction utilities."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import Array
from jaxtyping import Float

from log_psplines.models.anova import ANOVALogPSpline
from log_psplines.models.matrix import SpectralMatrix

if TYPE_CHECKING:
    from log_psplines.data.spectral import WishartData
    from log_psplines.inference.components import SpectralComponents
    from log_psplines.models.spectrum import LogPSpline


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
    log_delta_sq_samples: Float[Array | np.ndarray, "*samples N p"],
    theta_re_samples: Float[Array | np.ndarray, "*samples N _"],
    theta_im_samples: Float[Array | np.ndarray, "*samples N _"],
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
    log_delta_sq_samples: Float[Array | np.ndarray, "*samples N p"],
    theta_re_samples: Float[Array | np.ndarray, "*samples N _"],
    theta_im_samples: Float[Array | np.ndarray, "*samples N _"],
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


def _flatten(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    return arr.reshape((-1,) + arr.shape[2:])


def _batch_spline_eval(
    basis: Float[Array | np.ndarray, "N K_f"],
    weights: Float[Array | np.ndarray, "*samples K_f"],
) -> Float[np.ndarray, "*samples N"]:
    return np.einsum("fk,sk->sf", np.asarray(basis), np.asarray(weights))


def reconstruct_stationary_spectrum(
    posterior: xr.Dataset,
    spline_model: "SpectralComponents",
    data: "WishartData",
) -> xr.DataArray:
    """Reconstruct stationary spectral matrices from posterior draws."""
    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    n_sample = n_chain * n_draw

    log_delta = []
    for j in range(int(data.p)):
        weights = _flatten(posterior[f"weights_delta_{j}"].values)
        log_delta.append(
            _batch_spline_eval(spline_model.diagonal_models[j].basis, weights)
        )
    log_delta_sq = np.stack(log_delta, axis=-1)

    n_theta = int(spline_model.n_theta)
    theta_re = np.zeros((n_sample, int(data.N), n_theta))
    theta_im = np.zeros_like(theta_re)
    for theta_idx, (j, previous_channel) in enumerate(
        spline_model.theta_pairs
    ):
        for part, target in (("re", theta_re), ("im", theta_im)):
            name = f"weights_theta_{part}_{j}_{previous_channel}"
            if name not in posterior:
                continue
            weights = _flatten(posterior[name].values)
            model = spline_model.get_theta_model(part, j, previous_channel)
            target[..., theta_idx] = _batch_spline_eval(model.basis, weights)

    spectrum = reconstruct_psd_matrix(
        jnp.asarray(log_delta_sq),
        jnp.asarray(theta_re),
        jnp.asarray(theta_im),
        n_samples_max=n_sample,
    ).reshape(n_chain, n_draw, int(data.N), int(data.p), int(data.p))

    if data.channel_stds is not None:
        scale = np.outer(data.channel_stds, data.channel_stds)
        spectrum = spectrum * scale[None, None, None, :, :]

    return xr.DataArray(
        np.asarray(spectrum, dtype=np.complex128),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "frequency": np.asarray(data.freq, dtype=float),
            "channel": np.arange(int(data.p)),
            "channel_aux": np.arange(int(data.p)),
        },
        name="spectral_density",
    )


def power_draws_from_basis(posterior, model_data, frequency_slice=slice(None)):
    """Reconstruct all scalar power draws from saved bases, in a frequency chunk.

    Returns (chain, draw, time, frequency, 1), in the original data units.
    Works after a PSDResult NetCDF round trip without the original model.
    """
    bt = np.asarray(model_data["basis_time"])
    bf = np.asarray(model_data["basis_frequency"])[frequency_slice]
    if "weights_eta" in posterior:
        g = np.einsum("fj,cdj->cdf", bf, posterior["weights_g"].values)
        logs = g[:, :, None, :] + np.einsum(
            "ti,cdij,fj->cdtf",
            bt,
            posterior["weights_eta"].values,
            bf,
            optimize=True,
        )
    else:
        logs = np.einsum(
            "ti,cdij,fj->cdtf",
            bt,
            posterior["weights"].values,
            bf,
            optimize=True,
        )
    values = np.exp(logs)
    if "reference" in model_data:
        values *= np.asarray(model_data["reference"])[
            None, None, :, frequency_slice
        ]
    return values[..., None]


def reconstruct_power_spectrum(
    posterior: xr.Dataset,
    spline: "LogPSpline | ANOVALogPSpline",
    *,
    time: np.ndarray | None = None,
    frequency: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> xr.DataArray:
    """Reconstruct a scalar time-frequency spectrum from coefficient draws."""
    if spline.time is None:
        raise ValueError("Power results require a time basis")
    if isinstance(spline, ANOVALogPSpline):
        bf = np.asarray(spline.frequency.basis)
        g = np.einsum("fj,cdj->cdf", bf, posterior["weights_g"].values)
        eta = np.einsum(
            "ti,cdij,fj->cdtf",
            spline.time_basis,
            posterior["weights_eta"].values,
            bf,
            optimize=True,
        )
        log_psd = g[:, :, None, :] + eta
    else:
        log_psd = np.einsum(
            "ti,cdij,fj->cdtf",
            np.asarray(spline.time.basis),
            np.asarray(posterior["weights"].values),
            np.asarray(spline.basis),
            optimize=True,
        )
    spectrum = np.exp(log_psd)
    if reference is not None:
        spectrum = spectrum * np.asarray(reference)[None, None, :, :]
    spectrum = spectrum[..., None, None]
    return xr.DataArray(
        spectrum.astype(np.complex128),
        dims=("chain", "draw", "time", "frequency", "channel", "channel_aux"),
        coords={
            "chain": posterior.coords["chain"],
            "draw": posterior.coords["draw"],
            "time": np.asarray(
                spline.time.grid if time is None else time, dtype=float
            ),
            "frequency": np.asarray(
                spline.frequency.grid if frequency is None else frequency,
                dtype=float,
            ),
            "channel": [0],
            "channel_aux": [0],
        },
        name="spectral_density",
    )
