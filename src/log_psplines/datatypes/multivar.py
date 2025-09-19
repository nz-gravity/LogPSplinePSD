import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class MultivarFFT:
    """
    Discrete FFTs for multivariate time series.
    Stores real/imaginary parts and Cholesky design matrices.

    Attributes:
        y_re: Real part of FFT (n_freq, n_dim)
        y_im: Imag part of FFT (n_freq, n_dim)
        Z_re: Real part of Cholesky design matrix (n_freq, n_dim, n_theta)
        Z_im: Imag part of Cholesky design matrix (n_freq, n_dim, n_theta)
        freq: Frequency grid (n_freq,)
        n_freq: Number of frequencies
        n_dim: Number of channels
    """
    y_re: jnp.ndarray
    y_im: jnp.ndarray
    Z_re: jnp.ndarray
    Z_im: jnp.ndarray
    freq: jnp.ndarray
    n_freq: int
    n_dim: int

    @classmethod
    def compute_fft(cls, x: jnp.ndarray, fs: float = 1.0) -> 'MultivarFFT':
        """
        Compute FFT and Cholesky design matrices for multivariate time series.
        FFT is normalized by sqrt(n_time).
        """
        n_time, n_dim = x.shape
        assert n_time > n_dim, f"N of time {n_time} must be greater than dim {n_dim}"
        x_fft = jnp.fft.fft(x, axis=0) / jnp.sqrt(n_time)
        freqs = jnp.fft.fftfreq(n_time, 1 / fs)
        pos_freq_idx = freqs > 0
        freqs = freqs[pos_freq_idx]
        x_fft = x_fft[pos_freq_idx, :]
        y_re = jnp.real(x_fft)
        y_im = jnp.imag(x_fft)
        Z_re, Z_im = cls.compute_cholesky_design(x_fft)
        return cls(
            y_re=y_re,
            y_im=y_im,
            Z_re=Z_re,
            Z_im=Z_im,
            freq=freqs,
            n_freq=len(freqs),
            n_dim=n_dim
        )

    @staticmethod
    def compute_cholesky_design(x_fft: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Cholesky design matrices Z_re, Z_im for multivariate PSD.
        For each frequency, Z_k[j, i, l] = FFT of previous components for Cholesky off-diagonal.

        Args:
            x_fft: Array of shape (n_freq, n_channels), complex FFT values.
        Returns:
            Z_re: Real part of Cholesky design matrix (n_freq, n_channels, n_theta)
            Z_im: Imag part of Cholesky design matrix (n_freq, n_channels, n_theta)

        Example (for n_channels=3):
            n_theta = 3
            For each frequency j:
                Z_k[j, 1, 0] = x_fft[j, 0]
                Z_k[j, 2, 1] = x_fft[j, 0]
                Z_k[j, 2, 2] = x_fft[j, 1]
            All other entries are zero.
            So for p=3, Z_k[j] looks like:
                [[0, 0, 0],
                 [x_fft[j,0], 0, 0],
                 [0, x_fft[j,0], x_fft[j,1]]]
        """
        n, p = x_fft.shape
        if p <= 1:
            return jnp.zeros((n, p, 0)), jnp.zeros((n, p, 0))
        n_theta = int(p * (p - 1) / 2)
        Z_k = np.zeros((n, p, n_theta), dtype=np.complex64)
        for j in range(n):
            count = 0
            for i in range(1, p):
                Z_k[j, i, count:count + i] = np.array(x_fft[j, :i])
                count += i
        return jnp.real(jnp.array(Z_k)), jnp.imag(jnp.array(Z_k))


@dataclass
class MultivariateTimeseries:
    y: jnp.ndarray  # Shape: (n_time, n_channels)
    t: jnp.ndarray = None
    std: jnp.ndarray = None  # Per-channel std

    def __post_init__(self):
        if self.t is None:
            self.t = jnp.arange(self.y.shape[0])
        if self.std is None:
            self.std = jnp.std(self.y, axis=0)
        assert self.y.shape[0] == self.t.shape[0], "y and t must have the same length"
        if jnp.isnan(self.y).any() or jnp.isnan(self.t).any():
            raise ValueError("y or t contains NaN values.")

    @property
    def n_channels(self):
        return self.y.shape[1] if self.y.ndim > 1 else 1

    @property
    def fs(self) -> float:
        return float(1 / (self.t[1] - self.t[0]))

    def to_cross_spectral_density(self) -> "MultivarFFT":
        """Convert to frequency domain using your DiscreteFFT logic"""
        return MultivarFFT.compute_fft(self.y, fs=self.fs)
