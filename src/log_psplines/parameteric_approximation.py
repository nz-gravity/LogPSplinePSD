from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .datasets import Timeseries


@dataclass
class PSDApprox:
    freq: jnp.array
    power: jnp.array

    @classmethod
    def fit(cls, ts: Timeseries, window_size: int = 101):
        """
        Calculates a running median approximation of the Power Spectral Density (PSD)
        from a periodogram.

        Args:
          periodogram: A JAX array representing the periodogram.
          window_size: An integer representing the size of the running median window.
                       Must be odd.

        Returns:
          A JAX array representing the running median PSD approximation.
        """

        std = np.std(ts.y)
        mean = np.mean(ts.y)
        ynorm = (ts.y - mean) / std
        sampling_freq = float(1 / (ts.t[1] - ts.t[0]))
        freq = np.fft.rfftfreq(len(ynorm), d=1 / sampling_freq)
        power = np.abs(jnp.fft.rfft(ynorm)) ** 2 / len(ynorm)

        freq = freq[1::5]
        periodogram = power[1::5]
        periodogram = jnp.array(periodogram, dtype=jnp.float64)

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd for running median.")

        padding = window_size // 2
        padded_periodogram = jnp.pad(
            periodogram, (padding, padding), mode="reflect"
        )

        def median_window(i):
            window = jax.lax.dynamic_slice(
                padded_periodogram, (i,), (window_size,)
            )
            return jnp.median(jnp.sort(window))

        n = periodogram.shape[0]
        indices = jnp.arange(n)
        running_median = jax.vmap(median_window)(indices)

        # rescale the running median to have the same power as the original
        running_median = running_median * std**2

        return cls(freq=freq, power=running_median)

    def plot(self, ax, scaling=1):
        p = np.array(self.power) * scaling
        ax.loglog(
            self.freq,
            p,
            label="running median approx",
            linestyle="--",
            marker="o",
        )
