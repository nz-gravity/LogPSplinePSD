from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.scipy.linalg import solve_triangular

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
        freq_norm = (fft_data.freq - fft_data.freq.min()) / (
            fft_data.freq.max() - fft_data.freq.min()
        )

        # Initialize knots (same for all components for now, linear spacing)
        knot_method = knot_kwargs.get("method", "linear")
        if knot_method == "linear":
            knots = np.linspace(0, 1, n_knots)
        elif knot_method == "log":
            knots = np.geomspace(fft_data.freq[1], fft_data.freq[-1], n_knots)
            knots = (knots - knots.min()) / (
                knots.max() - knots.min()
            )  # Normalize to [0,1]
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

    def reconstruct_psd_matrix(
        self,
        log_delta_sq_samples: jnp.ndarray,
        theta_re_samples: jnp.ndarray,
        theta_im_samples: jnp.ndarray,
        n_samples_max: int = 50,
    ) -> jnp.ndarray:
        """
        Reconstruct PSD matrix from Cholesky components using JAX for efficiency.
        JIT-compiled for speed. Handles batch processing over samples and frequencies.
        Returns:
            psd_matrices: shape (n_samps, n_freq, n_channels, n_channels)
        """
        # Handle arrays that may have chain dimension (1, n_draws, n_freq, n_channels)
        if (
            log_delta_sq_samples.ndim == 4
        ):  # (n_chains, n_draws, n_freq, n_channels)
            log_delta_sq_samples = log_delta_sq_samples[0]  # Take first chain
        if theta_re_samples.ndim == 4:
            theta_re_samples = theta_re_samples[0]
        if theta_im_samples.ndim == 4:
            theta_im_samples = theta_im_samples[0]

        n_samples, n_freq, n_channels = log_delta_sq_samples.shape
        n_theta = theta_re_samples.shape[2] if theta_re_samples.ndim > 2 else 0
        n_samps = min(n_samples_max, n_samples)

        # Subsample if needed
        log_delta_sq = log_delta_sq_samples[:n_samps]
        theta_re = theta_re_samples[:n_samps]
        theta_im = theta_im_samples[:n_samps]

        @jax.jit
        def build_single_psd(log_delta_sq_sf, theta_re_sf, theta_im_sf):
            """Process single sample at single frequency.
            Args:
                log_delta_sq_sf: shape (n_channels,)
                theta_re_sf: shape (n_theta,)
                theta_im_sf: shape (n_theta,)
            """
            D = jnp.diag(jnp.exp(log_delta_sq_sf))
            T = jnp.eye(n_channels, dtype=jnp.complex64)
            if n_theta > 0 and n_channels > 1:
                theta_complex = theta_re_sf + 1j * theta_im_sf
                tril_row, tril_col = jnp.tril_indices(n_channels, k=-1)
                n_lower = len(tril_row)
                n_use = min(n_theta, n_lower)
                if n_use > 0:
                    T = T.at[tril_row[:n_use], tril_col[:n_use]].set(
                        -theta_complex[:n_use]
                    )
            temp = solve_triangular(T, D, lower=True)
            T_inv_H = (
                solve_triangular(
                    T, jnp.eye(n_channels, dtype=jnp.complex64), lower=True
                )
                .conj()
                .T
            )
            return temp @ T_inv_H

        @jax.jit
        def process_single_sample(log_delta_sq_s, theta_re_s, theta_im_s):
            """Process all frequencies for a single sample.
            Args:
                log_delta_sq_s: shape (n_freq, n_channels)
                theta_re_s: shape (n_freq, n_theta)
                theta_im_s: shape (n_freq, n_theta)
            """
            return vmap(build_single_psd, in_axes=(0, 0, 0))(
                log_delta_sq_s, theta_re_s, theta_im_s
            )

        # vmap over samples (axis 0)
        psd_matrices = vmap(process_single_sample, in_axes=(0, 0, 0))(
            log_delta_sq, theta_re, theta_im
        )
        # Apply consistent scaling factor (divide by 2*pi) to match empirical PSD convention
        return psd_matrices / (2 * jnp.pi)

    def __repr__(self):
        return (
            f"MultivariateLogPSplines(channels={self.n_channels}, "
            f"knots={self.n_knots}, degree={self.degree}, "
            f"penaltyOrder={self.diffMatrixOrder}, n_freq={self.n_freq})"
        )

    def get_psd_matrix_percentiles(
        self, psd_matrix_samples: jnp.ndarray, percentiles=[2.5, 50, 97.5]
    ) -> dict:
        # ensure shape is (samples, freqs, n, n)
        if (
            psd_matrix_samples.ndim == 3
        ):  # maybe missing the "samples" dimension
            psd_matrix_samples = psd_matrix_samples[None, ...]
        elif psd_matrix_samples.ndim != 4:
            raise ValueError(
                f"Expected 4D array (samples, freqs, n, n), got {psd_matrix_samples.shape}"
            )
        logger.debug(f"psd_matrix_samples shape: {psd_matrix_samples.shape}")

        # transform each sample to real-valued representation
        psd_matrix_real = _complex_to_real_batch(psd_matrix_samples)

        posterior_percentiles = np.percentile(
            psd_matrix_real, percentiles, axis=0
        )  # (percentile, freq, channels, channels_out)
        return posterior_percentiles

    def get_psd_matrix_coverage(
        self, psd_matrix_samples: jnp.ndarray, empirical_psd: jnp.ndarray
    ) -> float:
        # Transform empirical_psd to real-valued representation
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
    returns float64 with same leading dims
    """
    mats = np.asarray(mats)
    n = mats.shape[-1]
    # boolean masks that broadcast over leading dims
    upper = np.triu(np.ones((n, n), dtype=bool))
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)

    out = np.where(upper, mats.real, 0.0)
    out = np.where(lower, mats.imag, out)
    return out.astype(np.float64, copy=False)
