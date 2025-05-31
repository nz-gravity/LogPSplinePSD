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
    def fit(
        cls,
        ts: Timeseries,
        window_size: int = 101,
        warm_up_factor: float = 0.0,
        downsample_factor: int = 5,
        warmup_strategy: str = "growing",  # options: "growing", "nan"
    ):
        """
        Calculates a running median approximation of the Power Spectral Density (PSD)
        from a periodogram, with improved handling of the initial phase.

        Args:
          ts: A Timeseries object with fields t (time) and y (signal).
          window_size: Size of the running median window (must be odd).
          warm_up_factor: Fraction of window_size used for warm-up period.
          downsample_factor: Downsampling factor for FFT bins.
          warmup_strategy: How to handle the warm-up zone: "growing" window or fill with NaNs.

        Returns:
          An instance of PSDApprox containing frequency and power arrays.
        """

        std = jnp.std(ts.y)
        mean = jnp.mean(ts.y)
        ynorm = (ts.y - mean) / std
        sampling_freq = float(1 / (ts.t[1] - ts.t[0]))
        freq = jnp.fft.rfftfreq(len(ynorm), d=1 / sampling_freq)
        power = jnp.abs(jnp.fft.rfft(ynorm)) ** 2 / len(ynorm)

        # original n
        original_n = power.shape[0]

        freq = freq[1::downsample_factor]
        periodogram = power[1::downsample_factor]

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd for running median.")

        n = periodogram.shape[0]
        running_median = jnp.full(n, jnp.nan, dtype=jnp.float64)
        padding = window_size // 2
        padded = jnp.pad(periodogram, (padding, padding), mode="edge")

        def median_window(i):
            window = lax.dynamic_slice(padded, (i,), (window_size,))
            return jnp.median(window)

        # Compute full medians after warm-up
        start_idx = padding
        end_idx = n - padding
        indices = jnp.arange(start_idx, end_idx)
        medians = jax.vmap(median_window)(indices)
        running_median = running_median.at[start_idx:end_idx].set(medians)

        # Handle warm-up region
        if warmup_strategy == "growing":
            for i in range(padding):
                win = jnp.sort(periodogram[: 2 * i + 1])
                val = jnp.median(win)
                running_median = running_median.at[i].set(val)
        elif warmup_strategy == "nan":
            pass  # keep initial region as NaN

        # Rescale PSD to match original power
        running_median = running_median * std**2

        n = running_median.shape[0]

        # Now, fill the datapoints between to ensure same frequency resolution
        if original_n > n:
            # Create a new frequency array with the original resolution
            new_freq = jnp.linspace(freq[0], freq[-1], original_n)
            # linear interpolation to fill the gaps
            running_median = jnp.interp(new_freq, freq, running_median)

            # Update freq to match the new resolution
            freq = new_freq

        return cls(freq=freq, power=running_median)

    def plot(self, ax, scaling=1):
        p = np.array(self.power) * scaling
        valid = ~np.isnan(p)
        ax.loglog(
            self.freq[valid],
            p[valid],
            label="running median approx",
            linestyle="--",
            marker="o",
            markersize=1,
        )

    def __repr__(self):
        return f"PSDApprox(n={len(self.freq)})"
