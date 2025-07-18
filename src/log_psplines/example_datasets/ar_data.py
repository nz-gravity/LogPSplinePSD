from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from ..datatypes import Periodogram, Timeseries


class ARData:
    """
    A class to simulate an AR(p) process (for p up to 4, or higher) and
    compute its theoretical PSD as well as the raw periodogram.

    Attributes
    ----------
    ar_coefs : np.ndarray
        1D array of AR coefficients [a1, a2, ..., ap].
    order : int
        Order p of the AR process.
    sigma : float
        Standard deviation of the white‐noise driving the AR process.
    fs : float
        Sampling frequency [Hz].
    duration : float
        Total duration of the time series [s].
    n : int
        Number of samples = int(duration * fs).
    seed : Optional[int]
        Seed for the random number generator (if given).
    ts : np.ndarray
        Simulated time‐domain AR(p) series of length n.
    freqs : np.ndarray
        One‐sided frequency axis (length n//2 + 1).
    psd_theoretical : np.ndarray
        Theoretical one‐sided PSD (power per Hz) sampled at freqs.
    periodogram : np.ndarray
        One‐sided raw periodogram (power per Hz) from the simulated ts.
    """

    def __init__(
        self,
        order: int,
        duration: float,
        fs: float,
        sigma: float = 1.0,
        seed: Optional[int] = None,
        ar_coefs: Sequence[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        ar_coefs : Sequence[float]
            Coefficients [a1, a2, ..., ap] for an AR(p) model.
            For example, for AR(2) with x[t] = a1 x[t-1] + a2 x[t-2] + noise,
            pass ar_coefs=[a1, a2].
        duration : float
            Total length of the time series in seconds.
        fs : float
            Sampling frequency in Hz.
        sigma : float, default=1.0
            Standard deviation of the white‐noise innovations.
        seed : Optional[int], default=None
            Seed for the random number generator (if you want reproducible draws).
        """
        self.order = order

        if ar_coefs is None:
            if order == 1:
                ar_coefs = [0.9]
            elif order == 2:
                ar_coefs = [1.45, -0.9025]
            elif order == 3:
                ar_coefs = [0.9, -0.8, 0.7]
            elif order == 4:
                ar_coefs = [0.9, -0.8, 0.7, -0.6]
            elif order == 5:
                ar_coefs = [1, -2.2137, 2.9403, -2.1697, 0.9606]

        else:
            assert len(self.ar_coefs) == order

        self.ar_coefs = np.asarray(ar_coefs, dtype=float)
        self.order = len(self.ar_coefs)
        self.sigma = float(sigma)
        self.fs = float(fs)
        self.duration = float(duration)
        self.n = int(self.duration * self.fs)
        self.seed = seed

        self.ts = self._generate_timeseries()
        self.freqs = np.fft.rfftfreq(self.n, d=1.0 / self.fs)
        self.times = np.arange(self.n) / self.fs
        self.psd_theoretical = self._compute_theoretical_psd()
        self.periodogram = self._compute_periodogram()

        # convert to Timeseries and Periodogram datatypes
        self.ts = Timeseries(t=self.times, y=self.ts, std=self.sigma)
        self.periodogram = Periodogram(
            freqs=self.freqs, power=self.periodogram, filtered=False
        )

    def __repr__(self):
        return f"ARData(order={self.order}, n={self.n})"

    def _generate_timeseries(self) -> np.ndarray:
        """
        Generate an AR(p) time series of length n using the recursion

            x[t] = a1*x[t-1] + a2*x[t-2] + ... + ap*x[t-p] + noise[t],

        where noise[t] ~ Normal(0, sigma^2).  For t < 0, we assume x[t] = 0.

        Returns
        -------
        ts : np.ndarray
            Simulated AR(p) time series of length n.
        """
        n = self.n + 100  # Generate extra samples to avoid edge effects
        rng = np.random.default_rng(self.seed)
        x = np.zeros(n, dtype=float)
        noise = rng.normal(loc=0.0, scale=self.sigma, size=n)

        # Iterate from t = p .. n-1
        for t in range(self.order, self.n):
            past_terms = 0.0
            # sum over a_k * x[t-k-1]
            for k, a_k in enumerate(self.ar_coefs, start=1):
                past_terms += a_k * x[t - k]
            x[t] = past_terms + noise[t]

        # Return only the last n samples
        return x[self.order : self.order + self.n]

    def _compute_theoretical_psd(self) -> np.ndarray:
        """
        Compute the theoretical one‐sided PSD (power per Hz) of the AR(p) process:

            S_theory(f) = (sigma^2 / fs) / | 1 - a1*e^{-i*2πf/fs} - a2*e^{-i*2πf*2/fs} - ... - ap*e^{-i*2πf*p/fs} |^2

        evaluated at freqs = [0, 1, 2, ..., fs/2].

        Returns
        -------
        psd_th : np.ndarray
            One‐sided theoretical PSD of length n//2 + 1.
        """
        # digital‐frequency omega = 2π (f / fs)
        omega = 2 * np.pi * self.freqs / self.fs

        # Form the denominator polynomial: 1 - sum_{k=1}^p a_k e^{-i k omega}
        # We compute numerator = sigma^2 / fs, denominator=|...|^2
        denom = np.ones_like(omega, dtype=complex)
        for k, a_k in enumerate(self.ar_coefs, start=1):
            denom -= a_k * np.exp(-1j * k * omega)
        denom_mag2 = np.abs(denom) ** 2

        psd_th = (self.sigma**2 / self.fs) / denom_mag2
        return psd_th.real * 2  # should already be float

    def _compute_periodogram(self) -> np.ndarray:
        """
        Compute the one‐sided raw periodogram of the simulated time series:

            Pxx(f_k) = (1 / (n * fs)) * |H(f_k)|^2,
            then double all bins except DC (k=0) and Nyquist (k=n/2) if n is even.

        Returns
        -------
        pxx : np.ndarray
            One‐sided periodogram (power per Hz) of length n//2 + 1.
        """
        # 1) Full FFT
        H_full = np.fft.fft(self.ts)

        # 2) Compute |H|^2 and normalize by (n * fs) → gives power per Hz
        Pxx_full = (1.0 / (self.n * self.fs)) * np.abs(H_full) ** 2

        # 3) Keep only the first (n//2 + 1) bins for real‐input one‐sided PSD
        Pxx_one = Pxx_full[: self.n // 2 + 1]

        # 4) Double all interior bins (1 .. n//2-1) to account for negative frequencies
        if self.n % 2 == 0:
            # n even → Nyquist is index n/2 and should NOT be doubled
            Pxx_one[1:-1] *= 2.0
        else:
            # n odd → last index is floor(n/2), which is still not doubled
            Pxx_one[1:] *= 2.0

        return Pxx_one

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        *,
        show_legend: bool = True,
        periodogram_kwargs: Optional[dict] = None,
        theoretical_kwargs: Optional[dict] = None,
    ) -> plt.Axes:
        """
        Plot the one‐sided raw periodogram and the theoretical PSD
        on the same axes (log–log).

        Parameters
        ----------
        ax : Optional[plt.Axes]
            If provided, plot onto this Axes object. Otherwise, create a new figure/axes.
        show_legend : bool, default=True
            Whether to display a legend.
        periodogram_kwargs : Optional[dict], default=None
            Additional kwargs to pass to plt.semilogy when plotting the periodogram.
        theoretical_kwargs : Optional[dict], default=None
            Additional kwargs to pass to plt.semilogy when plotting the theoretical PSD.

        Returns
        -------
        ax : plt.Axes
            The Axes object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        # Default plotting styles
        p_kwargs = {"label": "Raw Periodogram", "alpha": 0.6, "linewidth": 1.0}
        t_kwargs = {
            "label": "Theoretical PSD",
            "linestyle": "--",
            "color": "C1",
            "linewidth": 2.0,
        }

        if periodogram_kwargs is not None:
            p_kwargs.update(periodogram_kwargs)
        if theoretical_kwargs is not None:
            t_kwargs.update(theoretical_kwargs)

        # Plot raw periodogram
        ax.semilogy(self.freqs, self.periodogram.power, **p_kwargs)

        # Plot theoretical PSD
        ax.semilogy(self.freqs, self.psd_theoretical, **t_kwargs)

        try:
            f, Pxx = welch(
                self.ts.y,
                fs=self.fs,
                nperseg=min(256, len(self.ts.y) // 4),
                scaling="density",
            )
            ax.semilogy(f, Pxx, label="Welch PSD", color="C3", linestyle=":")
        except Exception as e:
            pass

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [power/Hz]")
        ax.set_title(
            f"AR({self.order}) Process: Periodogram vs Theoretical PSD"
        )

        if show_legend:
            ax.legend()

        ax.grid(True, which="both", ls=":", alpha=0.5)
        return ax


# Example usage:
if __name__ == "__main__":
    # --- Simulate AR(2) over 8 seconds at 1024 Hz ---
    ar2 = ARData(
        ar_coefs=[0.9, -0.5], duration=8.0, fs=1024.0, sigma=1.0, seed=42
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ar2.plot(ax=ax)
    plt.show()

    # --- Simulate AR(4) over 4 seconds at 2048 Hz ---
    # e.g. coefficients [0.5, -0.3, 0.1, -0.05]
    ar4 = ARData(
        ar_coefs=[0.5, -0.3, 0.1, -0.05], duration=4.0, fs=2048.0, sigma=1.0
    )
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ar4.plot(
        ax=ax2,
        periodogram_kwargs={"color": "C2"},
        theoretical_kwargs={"color": "k", "linestyle": "-."},
    )
    plt.show()
