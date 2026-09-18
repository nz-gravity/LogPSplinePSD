from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from log_psplines.data.spectral import EmpiricalPSD, WishartData
from log_psplines.preprocessing.periodogram import (
    compute_fft,
    compute_wishart,
    empirical_spectrum,
)

from ..logger import logger


@dataclass
class TimeSeries:
    data: np.ndarray  # numpy array (n, p) for numerical stability
    t: np.ndarray | None = None  # numpy array
    std: np.ndarray | None = None  # numpy array, per-channel std
    scaling_factor: float | None = 1.0  # numpy array for per-channel scaling
    original_stds: np.ndarray | None = (
        None  # numpy array for original per-channel stds
    )

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim == 1:
            self.data = self.data[:, np.newaxis]
        if self.data.ndim != 2:
            raise ValueError("y must have shape (n,) or (n, p).")
        if self.t is None:
            self.t = np.arange(self.data.shape[0])
        else:
            self.t = np.asarray(self.t, dtype=np.float64)
        if self.std is None:
            self.std = np.std(self.data, axis=0)
        else:
            self.std = np.asarray(self.std, dtype=np.float64)
        if self.data.shape[0] != self.t.shape[0]:
            raise ValueError("y and t must have the same length")
        if np.isnan(self.data).any() or np.isnan(self.t).any():
            raise ValueError("y or t contains NaN values.")

    @property
    def p(self):
        return self.data.shape[1] if self.data.ndim > 1 else 1

    @property
    def fs(self) -> float:
        if self.t is None:
            raise ValueError(
                "t must be set before accessing sampling frequency."
            )
        return float(1 / (self.t[1] - self.t[0]))

    def standardise(self):
        self.std = np.std(self.data, axis=0)
        y = (self.data - np.mean(self.data, axis=0)) / self.std
        return TimeSeries(y, self.t, self.std)

    def standardise_for_psd(self):
        if self.original_stds is None:
            self.original_stds = np.std(self.data, axis=0)
        y_standardized = (
            self.data - np.mean(self.data, axis=0)
        ) / self.original_stds
        psd_scaling_factor = np.std(self.data) ** 2.0
        return TimeSeries(
            data=y_standardized,
            t=self.t,
            std=np.ones_like(self.original_stds),
            scaling_factor=psd_scaling_factor,
            original_stds=self.original_stds,
        )

    def to_cross_spectral_density(
        self,
        fmin: float | None = None,
        fmax: float | None = None,
    ) -> "WishartData":
        return compute_fft(
            self.data,
            fs=self.fs,
            fmin=fmin,
            fmax=fmax,
            scaling_factor=self.scaling_factor,
            channel_stds=self.original_stds,
        )

    def to_wishart_stats(
        self,
        Nb: int,
        fmin: float | None = None,
        fmax: float | None = None,
        window: str | tuple | None = None,
        detrend: str | bool = "constant",
        wishart_floor_fraction: float | None = None,
    ) -> "WishartData":
        n = self.data.shape[0]
        if isinstance(Nb, bool) or not isinstance(Nb, (int, np.integer)):
            raise TypeError("Nb must be a positive integer.")
        Nb = int(Nb)
        if Nb <= 0:
            raise ValueError("Nb must be positive.")
        if n % Nb != 0:
            raise ValueError(f"n={n} must be divisible by Nb={Nb}.")
        Lb = n // Nb

        wishart_fft = compute_wishart(
            self.data,
            fs=self.fs,
            Nb=Nb,
            fmin=fmin,
            fmax=fmax,
            scaling_factor=self.scaling_factor,
            channel_stds=self.original_stds,
            window=window,
            detrend=detrend,
            wishart_floor_fraction=wishart_floor_fraction,
        )
        log_msg = (
            f"Wishart averaging (blocks={Nb}): "
            f"n={n} -> Lb={Lb}, "
            f"N={wishart_fft.N}, p={wishart_fft.p}"
        )
        logger.info(log_msg)
        return wishart_fft

    @property
    def amplitude_range(self) -> tuple[float, float]:
        min_amp = float(np.min(self.data))
        max_amp = float(np.max(self.data))
        return (float(f"{min_amp:.3g}"), float(f"{max_amp:.3g}"))

    def __repr__(self):
        return f"TimeSeries(n={self.data.shape[0]}, p={self.p}, fs={self.fs:.3f}, amplitudes={self.amplitude_range})"

    def get_empirical_psd(self, **kwargs) -> "EmpiricalPSD":
        return empirical_spectrum(
            self.data,
            fs=self.fs,
            **kwargs,
        )
