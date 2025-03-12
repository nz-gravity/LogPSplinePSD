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
        power = jnp.abs(jnp.fft.rfft(self.y)) ** 2 / len(self.y)
        return Periodogram(freq[1:], power[1:])

    def standardise(self):
        """Standardise the timeseries to have zero mean and unit variance."""
        self.std = float(jnp.std(self.y))
        y = (self.y - jnp.mean(self.y)) / self.std
        return Timeseries(self.t, y, self.std)


@dataclasses.dataclass
class Periodogram:
    freqs: jnp.ndarray
    power: jnp.ndarray

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
        return Periodogram(self.freqs[mask], self.power[mask])

