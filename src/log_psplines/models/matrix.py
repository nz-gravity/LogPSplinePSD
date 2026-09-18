"""Spectral matrices from scalar modified-Cholesky components."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralMatrix:
    """Construct S = inv(T) D inv(T)^H with T[j,l] = -theta[j,l].

    log_variance (..., C) and theta (..., C*(C-1)//2) use row-major
    strict-lower-triangle ordering. Leading axes are untouched: frequency,
    posterior draws, or a future (time, frequency) grid. This algebra does
    not implement time-varying inference. Reconstruction uses NumPy complex128
    as before, independently of JAX's inference precision configuration.
    """

    channels: int

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError("channels must be positive")

    def __call__(
        self,
        log_variance: np.ndarray,
        theta_re: np.ndarray | None = None,
        theta_im: np.ndarray | None = None,
    ) -> np.ndarray:
        logs = np.asarray(log_variance)
        if logs.shape[-1] != self.channels:
            raise ValueError(
                "log_variance trailing dimension must match channels"
            )
        shape = logs.shape[:-1]
        row, col = np.tril_indices(self.channels, k=-1)
        triangular = np.broadcast_to(
            np.eye(self.channels, dtype=np.complex128),
            (*shape, self.channels, self.channels),
        ).copy()
        if len(row):
            if theta_re is None or theta_im is None:
                raise ValueError(
                    "Both real and imaginary theta components are required"
                )
            theta = np.asarray(theta_re) + 1j * np.asarray(theta_im)
            if theta.shape != (*shape, len(row)):
                raise ValueError(
                    "theta must match leading dimensions and lower triangle size"
                )
            triangular[..., row, col] = -theta
        inverse = np.linalg.inv(triangular)
        diagonal = np.exp(logs).astype(np.float64)
        return (inverse * diagonal[..., None, :]) @ inverse.conj().swapaxes(
            -1, -2
        )

    @staticmethod
    def coherence(spectrum: np.ndarray) -> np.ndarray:
        """Squared coherence (..., C, C), including unit diagonal."""
        diagonal = np.diagonal(spectrum, axis1=-2, axis2=-1).real
        return np.abs(spectrum) ** 2 / (
            diagonal[..., :, None] * diagonal[..., None, :]
        )
