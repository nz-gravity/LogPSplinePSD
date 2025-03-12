import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass
class Timeseries:
    t: jnp.ndarray
    y: jnp.ndarray

    @property
    def fs(self) -> float:
        """Sampling frequency computed from the time array."""
        return float(1 / (self.t[1] - self.t[0]))

    def to_periodogram(self) -> "Periodogram":
        """Compute the one-sided periodogram of the timeseries."""
        freq = jnp.fft.rfftfreq(len(self.y), d=1 / self.fs)
        power = jnp.abs(jnp.fft.rfft(self.y)) ** 2 / len(self.y)
        return Periodogram(freq[1:], power[1:])



@dataclasses.dataclass
class Periodogram:
    freqs: jnp.ndarray
    power: jnp.ndarray

    @property
    def fs(self) -> float:
        """Sampling frequency computed from the frequency array."""
        return float(2 * self.freqs[-1])

    def highpass(self, min_freq: float) -> "Periodogram":
        """Return a new Periodogram with frequencies above a threshold."""
        mask = self.freqs > min_freq
        return Periodogram(self.freqs[mask], self.power[mask])

