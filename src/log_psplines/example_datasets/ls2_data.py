"""LS2 locally stationary time-varying MA(1) process.

The process is

    X_t = w_t + b(t / T) w_{t-1},

with

    b(u) = 1.1 cos(1.5 - cos(4 pi u)),

and i.i.d. Gaussian innovations

    w_t ~ N(0, sigma^2).

The default parameters reproduce the LS2 process from Tang et al. (2026).

The analytic pointwise PSD is returned in the Oppenheim-Schafer digital
convention. For sigma=1, unit-variance white noise therefore has PSD 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LS2Data:
    """Locally stationary LS2 time-varying MA(1) process.

    Parameters
    ----------
    n_samples:
        Number of time-domain samples.
    fs:
        Sampling frequency in Hz.
    sigma:
        Standard deviation of the Gaussian innovations.
    seed:
        Random seed.
    amplitude:
        Amplitude of the time-varying MA coefficient.
    offset:
        Offset inside the cosine defining the MA coefficient.
    modulation:
        Temporal modulation frequency in units of pi.

    Notes
    -----
    The process is

        X_t = w_t + b(t / T) w_{t-1},

    where

        b(u) = amplitude * cos(offset - cos(modulation * pi * u)).

    The default values give

        b(u) = 1.1 cos(1.5 - cos(4 pi u)).
    """

    n_samples: int = 512
    fs: float = 64.0
    sigma: float = 1.0
    seed: int | None = None

    amplitude: float = 1.1
    offset: float = 1.5
    modulation: float = 4.0

    data: np.ndarray = field(init=False)
    time: np.ndarray = field(init=False)
    freq: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2.")

        if self.fs <= 0:
            raise ValueError("fs must be positive.")

        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

        self.rng = np.random.default_rng(self.seed)

        self.time = np.arange(self.n_samples) / self.fs

        # Positive Fourier frequencies, including Nyquist but excluding DC.
        self.freq = np.fft.rfftfreq(
            self.n_samples,
            d=self.dt,
        )[1:]

        self.data = self.simulate()

    @property
    def dt(self) -> float:
        """Sampling interval in seconds."""
        return 1.0 / self.fs

    @property
    def p(self) -> int:
        """Number of data channels."""
        return 1

    @property
    def duration(self) -> float:
        """Total nominal duration in seconds."""
        return self.n_samples / self.fs

    @property
    def rescaled_time(self) -> np.ndarray:
        """Sample positions on the locally stationary time coordinate."""
        return np.arange(self.n_samples) / self.n_samples

    def coefficient(
        self,
        u: np.ndarray | float,
    ) -> np.ndarray:
        """Return the time-varying MA(1) coefficient b(u).

        Parameters
        ----------
        u:
            Rescaled time coordinate. Typically in [0, 1).

        Returns
        -------
        ndarray
            Time-varying MA coefficient.
        """
        u = np.asarray(u)

        return self.amplitude * np.cos(
            self.offset - np.cos(self.modulation * np.pi * u)
        )

    def simulate(self) -> np.ndarray:
        """Generate one realization of the LS2 process.

        Returns
        -------
        ndarray
            Array with shape ``(n_samples, 1)``.
        """
        w = self.rng.normal(
            loc=0.0,
            scale=self.sigma,
            size=self.n_samples + 1,
        )

        b = self.coefficient(self.rescaled_time)

        x = w[1:] + b * w[:-1]

        return x[:, None]

    def get_true_psd(
        self,
        *,
        time_grid: np.ndarray | None = None,
        freq_grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the analytic pointwise time-varying PSD.

        Parameters
        ----------
        time_grid:
            Rescaled time coordinates ``u``. If omitted, the sample locations
            ``t / T`` are used.
        freq_grid:
            Frequencies in Hz. If omitted, ``self.freq`` is used.

        Returns
        -------
        ndarray
            PSD array with shape ``(n_time, n_freq)``.

        Notes
        -----
        For

            X_t = w_t + b(u) w_{t-1},

        the pointwise spectrum is

            S(u, omega)
                = sigma^2 [
                    1
                    + b(u)^2
                    + 2 b(u) cos(omega)
                ],

        where

            omega = 2 pi f / fs.
        """
        if time_grid is None:
            time_grid = self.rescaled_time

        if freq_grid is None:
            freq_grid = self.freq

        time_grid = np.asarray(time_grid)
        freq_grid = np.asarray(freq_grid)

        b = self.coefficient(time_grid)

        omega = 2.0 * np.pi * freq_grid / self.fs

        psd = self.sigma**2 * (
            1.0
            + b[:, None] ** 2
            + 2.0 * b[:, None] * np.cos(omega[None, :])
        )

        return psd
    
    def plot(
        self,
        *,
        fname: str | Path | None = None,
        show: bool = False,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (10.0, 6.0),
        nperseg: int = 64,
        noverlap: int | None = None,
    ) -> None:
        """Plot the LS2 realization and its local spectrum.

        Layout
        ------
        - top: time series with true local standard deviation envelope
        - left: time-averaged empirical PSD and time-averaged true PSD
        - centre: empirical spectrogram with true PSD contours overlaid
        """
        from pathlib import Path

        from scipy.signal import spectrogram

        x = self.data[:, 0]

        if noverlap is None:
            noverlap = int(0.875 * nperseg)

        # ------------------------------------------------------------------
        # Empirical local PSD from the data
        # ------------------------------------------------------------------
        freq, time_stft, empirical_psd = spectrogram(
            x,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            mode="psd",
        )

        # Drop DC to match the convention used elsewhere in the class.
        freq = freq[1:]
        empirical_psd = empirical_psd[1:, :]

        # ------------------------------------------------------------------
        # Analytic truth on the same STFT grid
        # ------------------------------------------------------------------
        u_stft = time_stft / self.duration

        true_psd = self.get_true_psd(
            time_grid=u_stft,
            freq_grid=freq,
        )

        # Convert from digital PSD convention to one-sided PSD-per-Hz convention
        # used by scipy.signal.spectrogram(..., scaling="density").
        true_psd_hz = 2.0 * true_psd / self.fs

        empirical_mean = empirical_psd.mean(axis=1)
        true_mean = true_psd_hz.mean(axis=0)

        # Local standard deviation for the top panel
        b = self.coefficient(self.rescaled_time)
        local_std = self.sigma * np.sqrt(1.0 + b**2)

        # ------------------------------------------------------------------
        # Figure layout
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=(1.35, 6.0, 0.22),
            height_ratios=(1.25, 5.0),
            left=0.08,
            right=0.95,
            bottom=0.10,
            top=0.94,
            wspace=0.12,
            hspace=0.20,
        )

        ax_time = fig.add_subplot(gs[0, 1])
        ax_spec = fig.add_subplot(gs[1, 0])
        ax_tf = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[1, 2])

        # ------------------------------------------------------------------
        # Top panel: time series
        # ------------------------------------------------------------------
        ax_time.plot(
            self.time,
            x,
            lw=1.0,
            color="C0",
            label="Data",
        )

        ax_time.plot(
            self.time,
            local_std,
            lw=1.2,
            ls="--",
            color="k",
            alpha=0.8,
            label=r"True local $\sigma_t$",
        )

        ax_time.plot(
            self.time,
            -local_std,
            lw=1.2,
            ls="--",
            color="k",
            alpha=0.5,
        )

        ax_time.set_ylabel("Amplitude")
        ax_time.set_xlim(self.time[0], self.time[-1])
        ax_time.tick_params(axis="x", labelbottom=False)
        ax_time.spines["top"].set_visible(False)
        ax_time.spines["right"].set_visible(False)
        ax_time.legend(frameon=False, fontsize=8, loc="upper right")

        # ------------------------------------------------------------------
        # Middle panel: empirical spectrogram
        # ------------------------------------------------------------------
        mesh = ax_tf.pcolormesh(
            time_stft,
            freq,
            empirical_psd,
            shading="auto",
            cmap=cmap,
        )

        truth_min = np.min(true_psd_hz)
        truth_max = np.max(true_psd_hz)
        levels = np.linspace(truth_min, truth_max, 5)[1:-1]

        ax_tf.contour(
            time_stft,
            freq,
            true_psd_hz.T,
            levels=levels,
            colors="white",
            linewidths=1.2,
            alpha=0.95,
        )

        ax_tf.set_xlabel("Time [s]")
        ax_tf.set_ylabel("Frequency [Hz]")
        ax_tf.set_xlim(time_stft[0], time_stft[-1])
        ax_tf.set_ylim(freq[0], freq[-1])

        # ------------------------------------------------------------------
        # Left panel: time-averaged PSD
        # ------------------------------------------------------------------
        ax_spec.plot(
            empirical_mean,
            freq,
            lw=1.4,
            color="C0",
            label="Data",
        )

        ax_spec.plot(
            true_mean,
            freq,
            lw=1.4,
            ls="--",
            color="k",
            label="Truth",
        )

        ax_spec.set_xlabel("Time-averaged PSD")
        ax_spec.set_ylim(freq[0], freq[-1])
        ax_spec.tick_params(axis="y", labelleft=False)
        ax_spec.spines["top"].set_visible(False)
        ax_spec.spines["right"].set_visible(False)
        ax_spec.legend(frameon=False, fontsize=8, loc="upper right")

        # ------------------------------------------------------------------
        # Colorbar
        # ------------------------------------------------------------------
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("PSD")

        if fname is not None:
            fname = Path(fname)
            fname.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                fname,
                dpi=200,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close(fig)