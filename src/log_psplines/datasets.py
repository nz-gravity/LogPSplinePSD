import dataclasses
import jax.numpy as jnp
import warnings


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

    def downsample(self, factor: int) -> "Timeseries":
        """Downsample the timeseries by a factor."""
        t, y = self.t[::factor], self.y[::factor]
        return Timeseries(t, y, self.std)


    def to_periodogram(self) -> "Periodogram":
        """Compute the one-sided periodogram of the timeseries."""
        freq = jnp.fft.rfftfreq(len(self.y), d=1 / self.fs)
        power = jnp.abs(jnp.fft.rfft(self.y)) ** 2 / len(self.y)
        return Periodogram(freq[1:], power[1:])

    def standardise(self):
        """Standardise the timeseries to have zero mean and unit variance."""
        std = float(jnp.std(self.y))
        if std == 1:
            warnings.warn(
                "Standard deviation is already 1. No standardisation applied."
            )
            return self

        y = (self.y - jnp.mean(self.y)) / std
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

    def downsample(self, factor):
        f, p = _downsample(self.freqs, self.power, factor)
        return Periodogram(f, p)


def _downsample(x,y, factor: int) -> "Timeseries":
    init_n = len(x)
    if factor <= 1:
        raise ValueError("Downsampling factor must be greater than 1.")
    if factor >= init_n:
        raise ValueError("Downsampling factor must be less than the length of the dataset.")


    y = y[::factor]
    x = x[::factor]

    percent = f"{100 * (1 - len(y) / init_n):.2f}%"
    print(f"Downsampling from {init_n:,} to {len(y):,} points ({percent})")
    return x, y