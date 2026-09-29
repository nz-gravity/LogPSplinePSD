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
            1.0 + b[:, None] ** 2 + 2.0 * b[:, None] * np.cos(omega[None, :])
        )

        return psd

    def plot(
        self,
        *,
        fname: str | Path | None = None,
        show: bool = False,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (8.0, 5.0),
    ) -> None:
        """Plot the analytic local PSD on the time-frequency grid.

        The colour values use the digital PSD convention of
        :meth:`get_true_psd`.
        """
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        mesh = ax.pcolormesh(
            self.time,
            self.freq,
            self.get_true_psd().T,
            shading="auto",
            cmap=cmap,
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        fig.colorbar(mesh, ax=ax, label="PSD [digital]")

        if fname is not None:
            path = Path(fname)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200)
        if show:
            plt.show()
        plt.close(fig)
