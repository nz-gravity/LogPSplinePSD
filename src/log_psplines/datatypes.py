import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass
class Timeseries:
    t: jnp.ndarray
    y: jnp.ndarray
    std: float = 1.0

    @property
    def n(self):
        return len(self.t)

    @property
    def fs(self) -> float:
        """Sampling frequency computed from the time array."""
        return float(1 / (self.t[1] - self.t[0]))

    def to_periodogram(self) -> "Periodogram":
        """Compute the one-sided periodogram of the timeseries."""
        freq = jnp.fft.rfftfreq(len(self.y), d=1 / self.fs)
        power = 2 * jnp.abs(jnp.fft.rfft(self.y)) ** 2 / len(self.y) / self.fs
        return Periodogram(freq[1:], power[1:])

    def standardise(self):
        """Standardise the timeseries to have zero mean and unit variance."""
        self.std = float(jnp.std(self.y))
        y = (self.y - jnp.mean(self.y)) / self.std
        return Timeseries(self.t, y, self.std)

    def __repr__(self):
        return f"Timeseries(n={len(self.t)}, std={self.std:.3f}, fs={self.fs:.3f})"


@dataclasses.dataclass
class Periodogram:
    freqs: jnp.ndarray
    power: jnp.ndarray
    filtered: bool = False

    def __post_init__(self):
        # assert no nans
        if jnp.isnan(self.freqs).any() or jnp.isnan(self.power).any():

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
        return Periodogram(self.freqs[mask], self.power[mask], filtered=True)

    def to_timeseries(self) -> "Timeseries":
        """Compute the inverse FFT of the periodogram."""
        y = jnp.fft.irfft(self.power, n=2 * (self.n - 1))
        t = jnp.linspace(0, 1 / self.fs, len(y))
        return Timeseries(t, y)

    def __mul__(self, other):
        return Periodogram(self.freqs, self.power * other)

    def __truediv__(self, other):
        return Periodogram(self.freqs, self.power / other)

    def __repr__(self):
        return f"Periodogram(n={self.n}, fs={self.fs:.3f}, filtered={self.filtered})"
