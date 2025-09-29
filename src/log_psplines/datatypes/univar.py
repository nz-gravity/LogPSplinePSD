import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class Timeseries:
    t: np.ndarray  # numpy array for numerical stability with extreme scales
    y: np.ndarray  # numpy array for numerical stability with extreme scales
    std: float = 1.0
    scaling_factor: float = 1.0  # Track the PSD scaling factor
    original_std: Optional[float] = None  # Store original standard deviation

    @property
    def n(self):
        return len(self.t)

    @property
    def fs(self) -> float:
        """Sampling frequency computed from the time array."""
        return float(1 / (self.t[1] - self.t[0]))

    def to_periodogram(self) -> "Periodogram":
        """Compute the one-sided periodogram of the timeseries."""
        freq = np.fft.rfftfreq(len(self.y), d=1 / self.fs)
        power = 2 * np.abs(np.fft.rfft(self.y)) ** 2 / len(self.y) / self.fs
        return Periodogram(
            freq[1:], power[1:], scaling_factor=self.scaling_factor
        )

    def standardise(self):
        """Standardise the timeseries to have zero mean and unit variance."""
        self.std = float(np.std(self.y))
        y = (self.y - np.mean(self.y)) / self.std
        return Timeseries(self.t, y, self.std)

    def standardise_for_psd(self):
        """Standardise specifically for PSD estimation, keeping track of scaling."""
        if self.original_std is None:
            self.original_std = float(np.std(self.y))

        # Standardize to unit variance
        y_standardized = (self.y - np.mean(self.y)) / self.original_std
        # The PSD scaling factor is the square of the amplitude scaling
        psd_scaling_factor = self.original_std**2
        return Timeseries(
            t=self.t,
            y=y_standardized,
            std=1.0,
            scaling_factor=psd_scaling_factor,
            original_std=self.original_std,
        )

    @property
    def amplitude_range(self):
        amps = (np.min(self.y), np.max(self.y))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))

    def __repr__(self):
        return f"Timeseries(n={len(self.t)}, std={self.std:.3f}, fs={self.fs:.3f}, amplitudes={self.amplitude_range})"


@dataclasses.dataclass
class Periodogram:
    freqs: np.ndarray
    power: np.ndarray
    filtered: bool = False
    scaling_factor: float = 1.0  # New: scaling factor for rescaling PSD

    def __post_init__(self):
        # assert no nans
        if np.isnan(self.freqs).any() or np.isnan(self.power).any():
            raise ValueError("Frequency or power contains NaN values.")

    @property
    def n(self):
        return len(self.freqs)

    @property
    def fs(self) -> float:
        """Sampling frequency computed from the frequency array."""
        return float(2 * self.freqs[-1])

    def highpass(self, min_freq: float) -> "Periodogram":
        """Return a new Periodogram with frequencies above a threshold."""
        mask = self.freqs > min_freq
        return Periodogram(
            self.freqs[mask],
            self.power[mask],
            filtered=True,
            scaling_factor=self.scaling_factor,
        )

    def to_timeseries(self) -> "Timeseries":
        """Compute the inverse FFT of the periodogram."""
        y = np.fft.irfft(self.power, n=2 * (self.n - 1))
        t = np.linspace(0, 1 / self.fs, len(y))
        return Timeseries(np.array(t), np.array(y))

    def __mul__(self, other):
        return Periodogram(
            self.freqs, self.power * other, scaling_factor=self.scaling_factor
        )

    def __truediv__(self, other):
        return Periodogram(
            self.freqs, self.power / other, scaling_factor=self.scaling_factor
        )

    def __repr__(self):
        return f"Periodogram(n={self.n}, fs={self.fs:.3f}, filtered={self.filtered}, amplitudes={self.amplitude_range})"

    def cut(self, fmin, fmax):
        """Return a new Periodogram with frequencies within [fmin, fmax]."""
        mask = (self.freqs >= fmin) & (self.freqs <= fmax)
        return Periodogram(
            self.freqs[mask],
            self.power[mask],
            filtered=True,
            scaling_factor=self.scaling_factor,
        )

    @property
    def amplitude_range(self):
        amps = (np.min(self.power), np.max(self.power))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))
