from dataclasses import dataclass

import numpy as np

from .datasets import Timeseries


@dataclass
class PSDApprox:
    freq: np.ndarray
    power: np.ndarray

    @classmethod
    def fit(
        cls,
        ts: Timeseries,
        window_size: int = 101,
        downsample_factor: int = 5,
    ):
        """
        Calculates a running‐median approximation of the Power Spectral Density (PSD)
        from a periodogram, using a “growing” window at the front, a centered window
        in the middle, and a “shrinking” window at the back (no NaNs anywhere).

        Args:
          ts: A Timeseries object with fields t (time) and y (signal).
          window_size: Size of the running‐median window (must be odd).
          downsample_factor: Downsampling factor for FFT bins.

        Returns:
          An instance of PSDApprox containing frequency and power arrays.
        """

        # 1) STANDARDIZE the time series (zero‐mean, unit‐variance)
        std = np.std(ts.y)
        mean = np.mean(ts.y)
        ynorm = (ts.y - mean) / std

        # 2) COMPUTE the periodogram (using rFFT)
        sampling_freq = float(1.0 / (ts.t[1] - ts.t[0]))
        freq_all = np.fft.rfftfreq(len(ynorm), d=1.0 / sampling_freq)
        power_all = np.abs(np.fft.rfft(ynorm)) ** 2 / len(ynorm)

        original_n = power_all.shape[0]

        # 3) DOWN‐SAMPLE (skip the zero‐frequency bin first)
        freq = freq_all[1::downsample_factor]
        periodogram = power_all[1::downsample_factor]
        n = len(periodogram)

        # 4) CHECK window_size is odd
        if window_size % 2 == 0:
            raise ValueError(
                "window_size must be odd for a symmetric running‐median."
            )

        padding = window_size // 2
        running_median = np.full(n, np.nan)

        # 5) PAD the periodogram with edge‐values so that every index has a full window
        #    (we'll use this for the “centered” block)
        padded = np.pad(periodogram, (padding, padding), mode="edge")

        # 6) MIDDLE region: for i = padding .. (n - padding - 1), we can take a full window_size slice
        for i in range(padding, n - padding):
            window = padded[i : i + window_size]  # exactly window_size points
            running_median[i] = np.median(window)

        # 7) GROWING region at the FRONT (i = 0 .. padding - 1):
        #    use periodogram[0 : (2*i + 1)] to “grow” from size 1 up to window_size
        for i in range(padding):
            front_window = periodogram[0 : (2 * i + 1)]
            running_median[i] = np.median(front_window)

        # 8) SHRINKING region at the BACK (i = n - padding .. n - 1):
        #    use periodogram[(i - padding) : n] to shrink from window_size down to 1
        for i in range(n - padding, n):
            back_window = periodogram[(i - padding) : n]
            running_median[i] = np.median(back_window)

        # 9) RESCALE back to original variance units (we divided by std^2 before)
        running_median *= std**2

        # 10) If we down‐sampled, INTERPOLATE back up so freq/power arrays match original length
        if original_n > len(running_median):
            new_freq = np.linspace(freq[0], freq[-1], original_n - 1)
            # We dropped the zero‐freq bin at the top; so we re‐insert it manually as zero‐power.
            # Alternatively, interpolate from freq[0]..freq[-1] onto an array of length (original_n-1).
            interp_vals = np.interp(new_freq, freq, running_median)
            freq = np.concatenate(([0.0], new_freq))
            running_median = np.concatenate(([0.0], interp_vals))
        else:
            # If no interpolation needed, we still want to “re‐attach” the zero‐freq bin
            freq = np.concatenate(([0.0], freq))
            running_median = np.concatenate(([0.0], running_median))

        # 11) FINAL sanity‐check: there should be no NaNs left
        if np.any(np.isnan(running_median)):
            pct = 100.0 * np.mean(np.isnan(running_median))
            raise ValueError(f"Running median still contains {pct:.1f}% NaNs")

        return cls(freq=freq, power=running_median)

    def plot(self, ax, scaling=1):
        """
        Simple log–log plot of freq vs. PSD‐estimate.
        """
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
