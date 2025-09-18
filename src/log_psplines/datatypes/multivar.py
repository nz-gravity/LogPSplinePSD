import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DiscreteFFT:
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
    def compute_fft(cls, x: np.ndarray, fs: float = 1.0) -> 'DiscreteFFT':
        """
        Compute FFT and Cholesky design matrices for multivariate time series.
        FFT is normalized by sqrt(n_time).
        """
        n_time, n_dim = x.shape
        assert n_time > n_dim, f"N of time {n_time} must be greater than dim {n_dim}"
        x_fft = np.fft.fft(x, axis=0) / np.sqrt(n_time)
        freqs = np.fft.fftfreq(n_time, 1 / fs)
        pos_freq_idx = freqs > 0
        freqs = freqs[pos_freq_idx]
        x_fft = x_fft[pos_freq_idx, :]
        y_re = np.real(x_fft)
        y_im = np.imag(x_fft)
        Z_re, Z_im = cls.compute_cholesky_design(x_fft)
        return cls(
            y_re=jnp.array(y_re),
            y_im=jnp.array(y_im),
            Z_re=jnp.array(Z_re),
            Z_im=jnp.array(Z_im),
            freq=jnp.array(freqs),
            n_freq=len(freqs),
            n_dim=n_dim
        )

    @staticmethod
    def compute_cholesky_design(x_fft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Cholesky design matrices Z_re, Z_im for multivariate PSD.
        For each frequency, Z_k[j, i, l] = FFT of previous components for Cholesky off-diagonal.
        """
        n, p = x_fft.shape
        if p <= 1:
            return np.zeros((n, p, 0)), np.zeros((n, p, 0))
        n_theta = int(p * (p - 1) / 2)
        Z_k = np.zeros((n, p, n_theta), dtype=np.complex64)
        for j in range(n):
            count = 0
            for i in range(1, p):
                Z_k[j, i, count:count + i] = x_fft[j, :i]
                count += i
        return np.real(Z_k), np.imag(Z_k)
