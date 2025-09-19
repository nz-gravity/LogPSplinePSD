from .psplines import LogPSplines
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..datatypes import MultivarFFT
import jax.numpy as jnp
import numpy as np
from .initialisation import init_basis_and_penalty, init_knots, init_weights
from tqdm.auto import trange


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
    offdiag_im_model: Optional[LogPSplines] = None  # Imaginary off-diagonal terms

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
    ) -> 'MultivariateLogPSplines':
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
        freq_norm = (fft_data.freq - fft_data.freq.min()) / (fft_data.freq.max() - fft_data.freq.min())

        # Initialize knots (same for all components for now, linear spacing)
        knots = np.linspace(0, 1, n_knots)

        # Create basis and penalty matrices (same for all components)
        basis, penalty = init_basis_and_penalty(knots, degree, n_freq, diffMatrixOrder)

        # Create diagonal models (one per channel)
        diagonal_models = []
        for i in range(n_channels):
            # Create pseudo-periodogram for initialization (using diagonal of empirical PSD)
            empirical_diag_power = fft_data.y_re[:, i] ** 2 + fft_data.y_im[:, i] ** 2
            empirical_diag_power = jnp.maximum(empirical_diag_power, 1e-12)  # Avoid log(0)

            # Create dummy LogPSplines object for init_weights function
            dummy_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=jnp.zeros(basis.shape[1]),
                parametric_model=jnp.ones(n_freq)
            )

            # Initialize weights for this diagonal component
            initial_weights = init_weights(jnp.log(empirical_diag_power), dummy_model)

            diagonal_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=initial_weights,
                parametric_model=jnp.ones(n_freq)  # No parametric component for now
            )
            diagonal_models.append(diagonal_model)

        # Create off-diagonal models if needed
        offdiag_re_model = None
        offdiag_im_model = None

        if n_channels > 1:
            # Simple empirical cross-spectra initialization for sanity checking
            n_theta = int(n_channels * (n_channels - 1) / 2)  # Number of off-diagonal parameters
            empirical_csd = jnp.zeros(n_freq)
            theta_idx = 0
            for i in range(1, n_channels):
                for j in range(i):
                    # Average cross-spectrum magnitude across all channel pairs and frequencies
                    csd_ij = fft_data.y_re[:, i] * fft_data.y_re[:, j] + fft_data.y_im[:, i] * fft_data.y_im[:, j]
                    empirical_csd = empirical_csd.at[:].add(jnp.abs(csd_ij))
                    theta_idx += 1
            empirical_csd = empirical_csd / n_theta  # Average across theta components

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
                parametric_model=jnp.ones(n_freq)
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
                parametric_model=jnp.ones(n_freq)
            )

            offdiag_im_model = LogPSplines(
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                n=n_freq,
                basis=basis,
                penalty_matrix=penalty,
                knots=knots,
                weights=initial_weights_offdiag,  # Same initialization for real and imaginary
                parametric_model=jnp.ones(n_freq)
            )

        return cls(
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            n_freq=n_freq,
            n_channels=n_channels,
            diagonal_models=diagonal_models,
            offdiag_re_model=offdiag_re_model,
            offdiag_im_model=offdiag_im_model
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

    def get_all_bases_and_penalties(self) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
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
            n_samples_max: int = 50
    ) -> jnp.ndarray:
        """
        Reconstruct PSD matrix from Cholesky components.

        Parameters
        ----------
        log_delta_sq_samples : jnp.ndarray, shape (n_samples, n_freq, n_channels)
            Samples of log diagonal elements
        theta_re_samples : jnp.ndarray, shape (n_samples, n_freq, n_theta)
            Samples of real off-diagonal elements  
        theta_im_samples : jnp.ndarray, shape (n_samples, n_freq, n_theta)
            Samples of imaginary off-diagonal elements
        n_samples_max : int, default=50
            Maximum number of samples to reconstruct (for computational efficiency)

        Returns
        -------
        jnp.ndarray, shape (n_samples_used, n_freq, n_channels, n_channels)
            Reconstructed PSD matrices

        Notes
        -----
        Implements: S(f) = T^(-1) D T^(-H) where:
        - D = diag(exp(log_delta_sq))  
        - T is lower triangular with -theta in lower triangle
        """
        n_samples, n_freq, n_channels = log_delta_sq_samples.shape
        n_theta = theta_re_samples.shape[2] if theta_re_samples.ndim > 2 else 0
        n_samps = min(n_samples_max, n_samples)

        psd_samples = np.zeros((n_samps, n_freq, n_channels, n_channels), dtype=complex)

        for i in trange(n_samps, desc="Reconstructing PSD matrices"):
            for k in range(n_freq):
                # Diagonal matrix D
                D = np.diag(np.exp(log_delta_sq_samples[i, k, :]))

                # Lower triangular matrix T  
                T = np.eye(n_channels, dtype=complex)

                if n_theta > 0 and n_channels > 1:
                    theta_idx = 0
                    for row in range(1, n_channels):
                        for col in range(row):
                            if theta_idx < n_theta:
                                theta_val = (theta_re_samples[i, k, theta_idx] +
                                             1j * theta_im_samples[i, k, theta_idx])
                                T[row, col] = -theta_val
                                theta_idx += 1

                # Reconstruct PSD: S = T^(-1) D T^(-H)
                try:
                    T_inv = np.linalg.inv(T)
                    S = T_inv @ D @ T_inv.conj().T
                    psd_samples[i, k] = S
                except np.linalg.LinAlgError:
                    # Fallback to diagonal if inversion fails
                    psd_samples[i, k] = D

        return psd_samples

    def __repr__(self):
        return (f"MultivariateLogPSplines(channels={self.n_channels}, "
                f"knots={self.n_knots}, degree={self.degree}, "
                f"penaltyOrder={self.diffMatrixOrder}, n_freq={self.n_freq})")


