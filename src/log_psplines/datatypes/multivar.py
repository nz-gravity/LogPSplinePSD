import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


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
    y_re: np.ndarray
    y_im: np.ndarray
    Z_re: np.ndarray
    Z_im: np.ndarray
    freq: np.ndarray
    n_freq: int
    n_dim: int
    scaling_factor: Optional[float] = 1.0  # Track the PSD scaling factor

    @classmethod
    def compute_fft(
            cls,
            x: np.ndarray,
            fs: float = 1.0,
            fmin: float = None,
            fmax: float = None,
            scaling_factor: Optional[float] = 1.0
    ) -> 'MultivarFFT':
        """
        Compute FFT and Cholesky design matrices for multivariate time series.
        FFT is normalized by sqrt(n_time).

        Parameters
        ----------
        x : np.ndarray
            Input time series (n_time, n_channels)
        fs : float
            Sampling frequency
        fmin, fmax : float, optional
            Frequency range for filtering. If None, uses all positive frequencies.
        """
        n_time, n_dim = x.shape
        assert n_time > n_dim, f"N of time {n_time} must be greater than dim {n_dim}"

        # standardise
        x = (x - np.mean(x, axis=0))

        x_fft = np.fft.fft(x, axis=0) / np.sqrt(n_time)
        freqs = np.fft.fftfreq(n_time, 1 / fs)

        # Get positive frequencies only
        pos_freq_idx = freqs > 0 # skip zero freq
        freqs = freqs[pos_freq_idx]
        x_fft = x_fft[pos_freq_idx, :]

        # Apply frequency range filtering if specified
        if fmin is not None or fmax is not None:
            fmin = fmin if fmin is not None else freqs[0] # skip zero freq
            fmax = fmax if fmax is not None else freqs[-1]
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            freqs = freqs[freq_mask]
            x_fft = x_fft[freq_mask, :]

        y_re = np.real(x_fft)
        y_im = np.imag(x_fft)
        Z_re, Z_im = cls.compute_cholesky_design(x_fft)

        return cls(
            y_re=y_re,
            y_im=y_im,
            Z_re=Z_re,
            Z_im=Z_im,
            freq=freqs,
            n_freq=len(freqs),
            n_dim=n_dim,
            scaling_factor=scaling_factor
        )

    def cut(self, fmin: float, fmax: float) -> 'MultivarFFT':
        """Return a new MultivarFFT within frequency range [fmin, fmax]."""
        mask = (self.freq >= fmin) & (self.freq <= fmax)
        return MultivarFFT(
            y_re=self.y_re[mask],
            y_im=self.y_im[mask],
            Z_re=self.Z_re[mask],
            Z_im=self.Z_im[mask],
            freq=self.freq[mask],
            n_freq=int(np.sum(mask)),
            n_dim=self.n_dim
        )

    @staticmethod
    def compute_cholesky_design(x_fft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            return np.zeros((n, p, 0)), np.zeros((n, p, 0))
        n_theta = int(p * (p - 1) / 2)
        Z_k = np.zeros((n, p, n_theta), dtype=np.complex64)
        for j in range(n):
            count = 0
            for i in range(1, p):
                Z_k[j, i, count:count + i] = np.array(x_fft[j, :i])
                count += i
        return np.real(Z_k), np.imag(Z_k)

    @property
    def amplitude_range(self) -> Tuple[float, float]:
        amps = (np.min(self.y_re), np.max(self.y_re))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))


    def __repr__(self):
        return f"MultivarFFT(n_freq={self.n_freq}, n_dim={self.n_dim}, amplitudes={self.amplitude_range})"

    @property
    def empirical_psd(self):
        return self.get_empirical_psd(self.y_re, self.y_im) * self.scaling_factor

    @staticmethod
    def get_empirical_psd(y_re, y_im) -> np.ndarray:
        y_re = np.array(y_re, dtype=np.float64)
        y_im = np.array(y_im, dtype=np.float64)
        n_freq, n_dim = y_re.shape
        _y = y_re + 1j * y_im
        empirical_psd = np.zeros((n_freq, n_dim, n_dim), dtype=np.complex128)
        norm_factor = 2 * np.pi
        for i in range(n_dim):
            for j in range(n_dim):
                empirical_psd[:, i, j] = 2 * (_y[:, i] * np.conj(_y[:, j])) / norm_factor
        return empirical_psd


@dataclass
class MultivariateTimeseries:
    y: np.ndarray  # numpy array (n_time, n_channels) for numerical stability
    t: np.ndarray = None  # numpy array
    std: np.ndarray = None  # numpy array, per-channel std
    scaling_factor: Optional[float] = 1.0  # numpy array for per-channel scaling
    original_stds: Optional[np.ndarray] = None    # numpy array for original per-channel stds

    def __post_init__(self):
        if self.t is None:
            self.t = np.arange(self.y.shape[0])
        if self.std is None:
            self.std = np.std(self.y, axis=0)
        if self.y.shape[0] != self.t.shape[0]:
            raise ValueError("y and t must have the same length")
        if np.isnan(self.y).any() or np.isnan(self.t).any():
            raise ValueError("y or t contains NaN values.")

    @property
    def n_channels(self):
        return self.y.shape[1] if self.y.ndim > 1 else 1

    @property
    def fs(self) -> float:
        return float(1 / (self.t[1] - self.t[0]))

    def standardise(self):
        self.std = np.std(self.y, axis=0)
        y = (self.y - np.mean(self.y, axis=0)) / self.std
        return MultivariateTimeseries(y, self.t, self.std)

    def standardise_for_psd(self):
        if self.original_stds is None:
            self.original_stds = np.std(self.y, axis=0)
        y_standardized = (self.y - np.mean(self.y, axis=0)) / self.original_stds
        psd_scaling_factor = np.std(self.y) ** 2.0
        return MultivariateTimeseries(
            y=y_standardized,
            t=self.t,
            std=np.ones_like(self.original_stds),
            scaling_factor=psd_scaling_factor,
            original_stds=self.original_stds
        )

    def to_cross_spectral_density(
            self,
            fmin: float = None,
            fmax: float = None
    ) -> "MultivarFFT":
        return MultivarFFT.compute_fft(self.y, fs=self.fs, fmin=fmin, fmax=fmax,  scaling_factor=self.scaling_factor)

    @property
    def amplitude_range(self) -> Tuple[float, float]:
        amps = (np.min(self.y), np.max(self.y))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))

    def __repr__(self):
        return f"MultivariateTimeseries(n_time={self.y.shape[0]}, n_channels={self.n_channels}, fs={self.fs:.3f}, amplitudes={self.amplitude_range})"
