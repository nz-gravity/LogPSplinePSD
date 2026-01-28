from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..datatypes import MultivarFFT
from ..logger import logger
from .initialisation import init_basis_and_penalty, init_knots, init_weights
from .psplines import LogPSplines


@dataclass
class MultivariateLogPSplines:
    """
    Multivariate log P-splines using Cholesky parameterization for cross-spectral density matrices.

    Uses Cholesky decomposition: S(f) = T^(-1) D T^(-H) where:
    - D is diagonal matrix with exp(log_delta_sq) elements (one P-spline per channel)
    - T is lower triangular with -theta terms (separate P-splines for real/imaginary parts)

    This enables flexible modeling of both auto-spectra and cross-spectra while
    ensuring positive definiteness of the estimated PSD matrix.

    Parameters
    ----------
    degree : int
        Polynomial degree of B-spline basis functions
    diffMatrixOrder : int
        Order of finite difference penalty matrix
    n_freq : int
        Number of frequency bins
    n_channels : int
        Number of channels in multivariate data
    diagonal_models : List[LogPSplines]
        P-spline models for diagonal PSD components (one per channel)
    offdiag_re_model : LogPSplines, optional
        P-spline model for real parts of off-diagonal terms
    offdiag_im_model : LogPSplines, optional
        P-spline model for imaginary parts of off-diagonal terms
    """

    degree: int
    diffMatrixOrder: int
    n_freq: int
    n_channels: int

    # P-spline components for each Cholesky parameter
    diagonal_models: List[LogPSplines]  # One per channel
    offdiag_re_model: Optional[LogPSplines] = None  # Real off-diagonal terms
    offdiag_im_model: Optional[LogPSplines] = (
        None  # Imaginary off-diagonal terms
    )

    def __post_init__(self):
        """Validate multivariate model parameters."""
        if len(self.diagonal_models) != self.n_channels:
            raise ValueError(
                f"Number of diagonal models ({len(self.diagonal_models)}) "
                f"must match number of channels ({self.n_channels})"
            )

        # For multivariate case (n_channels > 1), we need off-diagonal models
        if self.n_channels > 1:
            if self.offdiag_re_model is None or self.offdiag_im_model is None:
                raise ValueError(
                    "Off-diagonal models required for multivariate case (n_channels > 1)"
                )

    @classmethod
    def from_multivar_fft(
        cls,
        fft_data: MultivarFFT,
        n_knots: int,
        degree: int = 3,
        diffMatrixOrder: int = 2,
        knot_kwargs: dict = {},
    ) -> "MultivariateLogPSplines":
        """
        Factory method to construct multivariate P-spline model from FFT data.

        Parameters
        ----------
        fft_data : MultivarFFT
            Multivariate FFT data with real/imaginary components and design matrices
        n_knots : int
            Number of interior knots for P-spline basis
        degree : int, default=3
            Polynomial degree of B-spline basis
        diffMatrixOrder : int, default=2
            Order of difference penalty matrix
        knot_kwargs : dict, default={}
            Additional arguments for knot placement

        Returns
        -------
        MultivariateLogPSplines
            Fully initialized multivariate model
        """
        n_freq = fft_data.n_freq
        n_channels = fft_data.n_dim

        # Create frequency grid for knot placement (normalized to [0,1])
        freq = np.asarray(fft_data.freq, dtype=np.float64)
        finite_mask = np.isfinite(freq)
        if not finite_mask.any():
            freq_norm = np.zeros_like(freq)
        else:
            freq_finite = freq[finite_mask]
            freq_min = float(freq_finite.min())
            freq_max = float(freq_finite.max())
            denom = freq_max - freq_min
            if denom <= 0:
                freq_norm = np.zeros_like(freq)
            else:
                freq_norm = (freq - freq_min) / denom
                freq_norm = np.where(finite_mask, freq_norm, 0.0)

        # Initialize knots (same for all components for now, linear spacing)
        knot_method = knot_kwargs.get("method", "linear")
        if knot_method == "linear":
            knots = np.linspace(0, 1, n_knots)
        elif knot_method == "log":
            if n_freq < 2:
                knots = np.linspace(0, 1, n_knots)
            else:
                knots_raw = np.geomspace(
                    fft_data.freq[1], fft_data.freq[-1], n_knots
                )
                knots = (knots_raw - knots_raw.min()) / (
                    knots_raw.max() - knots_raw.min()
                )
        else:
            raise ValueError(f"Unknown knot placement method: {knot_method}")

        # Create basis and penalty matrices (same for all components)
        basis, penalty = init_basis_and_penalty(
            knots, degree, n_freq, diffMatrixOrder, grid_points=freq_norm
        )

        use_wishart = fft_data.u_re is not None and fft_data.u_im is not None
        if use_wishart:
            u_re = jnp.asarray(fft_data.u_re)
            u_im = jnp.asarray(fft_data.u_im)
            u_complex = u_re + 1j * u_im
            Y = jnp.einsum("fkc,fkd->fcd", u_complex, jnp.conj(u_complex))
            nu_scale = float(max(int(fft_data.nu), 1))
        else:
            Y = None
            nu_scale = 1.0

        # Create diagonal models (one per channel)
        diagonal_models = []
        for i in range(n_channels):
            if use_wishart:
                empirical_diag_power = jnp.real(Y[:, i, i]) / nu_scale
            else:
                empirical_diag_power = (
                    fft_data.y_re[:, i] ** 2 + fft_data.y_im[:, i] ** 2
                )
            empirical_diag_power = jnp.maximum(
                empirical_diag_power, 1e-12
            )  # Avoid log(0)

            # Create dummy LogPSplines object for init_weights function
            dummy_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=jnp.zeros(basis.shape[1]),
                parametric_model=jnp.ones(n_freq),
            )

            # Initialize weights for this diagonal component
            initial_weights = init_weights(
                jnp.log(empirical_diag_power), dummy_model
            )

            diagonal_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=initial_weights,
                parametric_model=jnp.ones(
                    n_freq
                ),  # No parametric component for now
            )
            diagonal_models.append(diagonal_model)

        # Create off-diagonal models if needed
        offdiag_re_model = None
        offdiag_im_model = None

        if n_channels > 1:
            # Simple empirical cross-spectra initialization for sanity checking
            n_theta = int(
                n_channels * (n_channels - 1) / 2
            )  # Number of off-diagonal parameters
            empirical_csd = jnp.zeros(n_freq)
            theta_idx = 0
            for i in range(1, n_channels):
                for j in range(i):
                    if use_wishart:
                        csd_ij = jnp.abs(Y[:, i, j]) / nu_scale
                    else:
                        csd_ij = (
                            fft_data.y_re[:, i] * fft_data.y_re[:, j]
                            + fft_data.y_im[:, i] * fft_data.y_im[:, j]
                        )
                    empirical_csd = empirical_csd.at[:].add(jnp.abs(csd_ij))
                    theta_idx += 1
            empirical_csd = (
                empirical_csd / n_theta
            )  # Average across theta components

            # Initialize with small values based on empirical estimates
            small_init = jnp.log(jnp.maximum(empirical_csd, 1e-8))

            # Create dummy LogPSplines object for init_weights function
            dummy_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=jnp.zeros(basis.shape[1]),
                parametric_model=jnp.ones(n_freq),
            )
            initial_weights_offdiag = init_weights(small_init, dummy_model)

            offdiag_re_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=initial_weights_offdiag,
                parametric_model=jnp.ones(n_freq),
            )

            offdiag_im_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=initial_weights_offdiag,  # Same initialization for real and imaginary
                parametric_model=jnp.ones(n_freq),
            )

        return cls(
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            n_freq=n_freq,
            n_channels=n_channels,
            diagonal_models=diagonal_models,
            offdiag_re_model=offdiag_re_model,
            offdiag_im_model=offdiag_im_model,
        )

    @property
    def n_knots(self) -> int:
        """Number of knots (same for all components)."""
        return len(self.diagonal_models[0].knots)

    @property
    def n_basis(self) -> int:
        """Number of basis functions per component."""
        return self.diagonal_models[0].n_basis

    @property
    def n_theta(self) -> int:
        """Number of off-diagonal parameters."""
        return int(self.n_channels * (self.n_channels - 1) / 2)

    @property
    def total_components(self) -> int:
        """Total number of P-spline components."""
        return self.n_channels + (2 if self.n_theta > 0 else 0)

    def get_all_bases_and_penalties(
        self,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Get basis and penalty matrices for all components (for NumPyro model).

        Returns
        -------
        Tuple[List[jnp.ndarray], List[jnp.ndarray]]
            Lists of basis and penalty matrices for all components
        """
        all_bases = []
        all_penalties = []

        # Add diagonal components
        for model in self.diagonal_models:
            all_bases.append(model.basis)
            all_penalties.append(model.penalty_matrix)

        # Add off-diagonal components if they exist
        if self.offdiag_re_model is not None:
            all_bases.append(self.offdiag_re_model.basis)
            all_penalties.append(self.offdiag_re_model.penalty_matrix)

        if self.offdiag_im_model is not None:
            all_bases.append(self.offdiag_im_model.basis)
            all_penalties.append(self.offdiag_im_model.penalty_matrix)

        return all_bases, all_penalties

    def _psd_chunk_iterator(
        self,
        log_delta_sq_samples: np.ndarray,
        theta_re_samples: Optional[np.ndarray],
        theta_im_samples: Optional[np.ndarray],
        *,
        n_samps: int,
        chunk_size: int,
    ):
        """Yield reconstructed PSD chunks with shape (n_samps, chunk, n, n)."""

        n_freq = log_delta_sq_samples.shape[1]
        n_channels = log_delta_sq_samples.shape[2]
        n_theta = (
            theta_re_samples.shape[2] if theta_re_samples is not None else 0
        )
        tril_row, tril_col = np.tril_indices(n_channels, k=-1)
        n_lower = len(tril_row)

        for start in range(0, n_freq, chunk_size):
            end = min(start + chunk_size, n_freq)

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

            chunk_len = end - start
            psd_chunk = np.empty(
                (n_samps, chunk_len, n_channels, n_channels),
                dtype=np.complex64,
            )

            for s in range(n_samps):
                for local_f in range(chunk_len):
                    diag_vals = np.exp(log_chunk[s, local_f]).astype(
                        np.float32
                    )
                    T = np.eye(n_channels, dtype=np.complex64)

                    if n_theta > 0:
                        theta_complex = (
                            theta_re_chunk[s, local_f]
                            + 1j * theta_im_chunk[s, local_f]
                        )
                        n_use = min(theta_complex.shape[0], n_lower)
                        if n_use:
                            T[tril_row[:n_use], tril_col[:n_use]] = (
                                -theta_complex[:n_use]
                            )

                    Tinverse = np.linalg.inv(T)
                    D = np.diag(diag_vals)
                    psd_chunk[s, local_f] = (
                        Tinverse @ D @ Tinverse.conj().T
                    ).astype(np.complex64)

            yield start, end, psd_chunk

    def reconstruct_psd_matrix(
        self,
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
        as a ``complex64`` NumPy array of shape
        ``(n_samps, n_freq, n_channels, n_channels)``.
        """
        log_delta_sq_arr = np.asarray(log_delta_sq_samples)
        theta_re_arr = np.asarray(theta_re_samples)
        theta_im_arr = np.asarray(theta_im_samples)

        # Accept both flattened samples (draw, freq, ...) and full MCMC output
        # with an explicit chain dimension (chain, draw, freq, ...).
        if log_delta_sq_arr.ndim == 4:
            log_delta_sq_arr = log_delta_sq_arr.reshape(
                (log_delta_sq_arr.shape[0] * log_delta_sq_arr.shape[1],)
                + tuple(log_delta_sq_arr.shape[2:])
            )
        if theta_re_arr.ndim == 4:
            theta_re_arr = theta_re_arr.reshape(
                (theta_re_arr.shape[0] * theta_re_arr.shape[1],)
                + tuple(theta_re_arr.shape[2:])
            )
        if theta_im_arr.ndim == 4:
            theta_im_arr = theta_im_arr.reshape(
                (theta_im_arr.shape[0] * theta_im_arr.shape[1],)
                + tuple(theta_im_arr.shape[2:])
            )

        n_samples, n_freq, n_channels = log_delta_sq_arr.shape
        n_theta = theta_re_arr.shape[2] if theta_re_arr.ndim > 2 else 0
        n_samps = min(int(n_samples_max), int(n_samples))

        if chunk_size is None or chunk_size <= 0:
            chunk_size = n_freq

        if n_samps < n_samples:
            # Avoid using only the first draws (can be highly autocorrelated).
            idx = np.linspace(0, n_samples - 1, n_samps, dtype=int)
            log_delta_sq_arr = log_delta_sq_arr[idx]
            theta_re_arr = theta_re_arr[idx]
            theta_im_arr = theta_im_arr[idx]
        else:
            log_delta_sq_arr = log_delta_sq_arr[:n_samps]
            theta_re_arr = theta_re_arr[:n_samps]
            theta_im_arr = theta_im_arr[:n_samps]

        psd = np.empty(
            (n_samps, n_freq, n_channels, n_channels), dtype=np.complex64
        )

        for start, end, psd_chunk in self._psd_chunk_iterator(
            log_delta_sq_arr,
            theta_re_arr if n_theta > 0 else None,
            theta_im_arr if n_theta > 0 else None,
            n_samps=n_samps,
            chunk_size=chunk_size,
        ):
            psd[:, start:end] = psd_chunk

        return psd

    def compute_psd_quantiles(
        self,
        log_delta_sq_samples: jnp.ndarray,
        theta_re_samples: jnp.ndarray,
        theta_im_samples: jnp.ndarray,
        *,
        percentiles: Optional[Sequence[float]] = None,
        n_samples_max: int = 50,
        chunk_size: int = 2048,
        compute_coherence: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute PSD (and optional coherence) percentiles without storing all draws.

        Returns
        -------
        psd_real_percentiles : np.ndarray
            Percentiles of the real part of the PSD matrix with shape
            ``(n_percentiles, n_freq, n_channels, n_channels)``.
        psd_imag_percentiles : np.ndarray
            Percentiles of the imaginary part of the PSD matrix with matching shape.
        coherence_percentiles : Optional[np.ndarray]
            When ``compute_coherence`` is ``True`` and ``n_channels > 1``, contains
            percentiles of the coherence matrix; otherwise ``None``.
        """

        if percentiles is None:
            percentiles = [5.0, 50.0, 95.0]

        log_delta_sq_arr = np.asarray(log_delta_sq_samples)
        theta_re_arr = np.asarray(theta_re_samples)
        theta_im_arr = np.asarray(theta_im_samples)

        # Accept both flattened samples (draw, freq, ...) and full MCMC output
        # with an explicit chain dimension (chain, draw, freq, ...).
        if log_delta_sq_arr.ndim == 4:
            log_delta_sq_arr = log_delta_sq_arr.reshape(
                (log_delta_sq_arr.shape[0] * log_delta_sq_arr.shape[1],)
                + tuple(log_delta_sq_arr.shape[2:])
            )
        if theta_re_arr.ndim == 4:
            theta_re_arr = theta_re_arr.reshape(
                (theta_re_arr.shape[0] * theta_re_arr.shape[1],)
                + tuple(theta_re_arr.shape[2:])
            )
        if theta_im_arr.ndim == 4:
            theta_im_arr = theta_im_arr.reshape(
                (theta_im_arr.shape[0] * theta_im_arr.shape[1],)
                + tuple(theta_im_arr.shape[2:])
            )

        n_samples, n_freq, n_channels = log_delta_sq_arr.shape
        n_theta = theta_re_arr.shape[2] if theta_re_arr.ndim > 2 else 0
        n_samps = min(int(n_samples_max), int(n_samples))

        if chunk_size is None or chunk_size <= 0:
            chunk_size = n_freq

        if n_samps < n_samples:
            # Avoid using only the first draws (can be highly autocorrelated).
            idx = np.linspace(0, n_samples - 1, n_samps, dtype=int)
            log_delta_sq_arr = log_delta_sq_arr[idx]
            theta_re_arr = theta_re_arr[idx]
            theta_im_arr = theta_im_arr[idx]
        else:
            log_delta_sq_arr = log_delta_sq_arr[:n_samps]
            theta_re_arr = theta_re_arr[:n_samps]
            theta_im_arr = theta_im_arr[:n_samps]

        n_percentiles = len(percentiles)
        psd_percentiles = np.empty(
            (n_percentiles, n_freq, n_channels, n_channels), dtype=np.float64
        )
        psd_imag_percentiles = np.empty_like(psd_percentiles)

        coherence_percentiles = (
            np.empty(
                (n_percentiles, n_freq, n_channels, n_channels),
                dtype=np.float64,
            )
            if compute_coherence and n_channels > 1
            else None
        )

        for start, end, psd_chunk in self._psd_chunk_iterator(
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
                    for c in range(n_channels):
                        coh_q[idx, :, c, c] = 1.0

                coherence_percentiles[:, start:end] = coh_q

        return psd_percentiles, psd_imag_percentiles, coherence_percentiles

    def __repr__(self):
        return (
            f"MultivariateLogPSplines(channels={self.n_channels}, "
            f"knots={self.n_knots}, degree={self.degree}, "
            f"penaltyOrder={self.diffMatrixOrder}, n_freq={self.n_freq})"
        )

    def get_psd_matrix_percentiles(
        self, psd_matrix_samples: jnp.ndarray, percentiles=[2.5, 50, 97.5]
    ) -> np.ndarray:
        arr = np.asarray(psd_matrix_samples)
        if arr.ndim == 4 and arr.shape[0] == len(percentiles):
            return arr.astype(np.float64, copy=False)

        if arr.ndim == 3:
            arr = arr[None, ...]
        elif arr.ndim != 4:
            raise ValueError(
                f"Expected 4D array (samples, freqs, n, n), got {arr.shape}"
            )

        psd_matrix_real = _complex_to_real_batch(arr)
        posterior_percentiles = np.percentile(
            psd_matrix_real, percentiles, axis=0
        )
        return posterior_percentiles.astype(np.float64, copy=False)

    def get_psd_matrix_coverage(
        self, psd_matrix_samples: jnp.ndarray, empirical_psd: jnp.ndarray
    ) -> float:
        empirical_psd_real = _complex_to_real_batch(empirical_psd)

        psd_percentiles = self.get_psd_matrix_percentiles(psd_matrix_samples)
        coverage = np.mean(
            (empirical_psd_real >= psd_percentiles[0])
            & (empirical_psd_real <= psd_percentiles[-1])
        )
        return float(coverage)


def _complex_to_real_batch(mats):
    """
    Safe, vectorized transform:
      - Upper triangle (incl diag) -> real part
      - Strict lower triangle      -> imag part
    mats: (..., n, n) complex
    returns float32 with same leading dims
    """
    mats = np.asarray(mats)
    n = mats.shape[-1]
    # boolean masks that broadcast over leading dims
    upper = np.triu(np.ones((n, n), dtype=bool))
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)

    out = np.where(upper, mats.real, 0.0)
    out = np.where(lower, mats.imag, out)
    return out.astype(np.float64, copy=False)
