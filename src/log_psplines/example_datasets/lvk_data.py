import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from gwpy.timeseries import TimeSeries


@dataclass
class LVKData:
    """
    A dataclass for downloading, loading, and computing PSDs of gravitational-wave strain data.

    Upon initialization, the PSDs for all overlapping segments are computed immediately.
    """

    strain: TimeSeries
    duration: int
    segment_duration: int
    segment_overlap: float
    min_freq: Optional[float] = None
    max_freq: Optional[float] = None

    # Fields computed in __post_init__; not passed at construction
    fs: float = field(init=False)
    n: int = field(init=False)
    nperseg: int = field(init=False)
    noverlap: int = field(init=False)
    step: int = field(init=False)
    n_segments: int = field(init=False)
    freqs: np.ndarray = field(init=False)
    psds: np.ndarray = field(init=False)
    median_psd: np.ndarray = field(init=False)

    def __post_init__(self):
        # Sampling info
        self.fs = float(self.strain.sample_rate.value)
        self.n = len(self.strain)

        # Number of samples per segment and overlap in samples
        self.nperseg = int(self.fs * self.segment_duration)
        self.noverlap = int(self.nperseg * self.segment_overlap)
        self.step = self.nperseg - self.noverlap

        # Compute number of segments
        self.n_segments = (self.n - self.noverlap) // self.step

        # Extract raw numpy array of strain values
        data = self.strain.value

        # Build strided array of shape (n_segments, nperseg)
        shape = (self.n_segments, self.nperseg)
        strides = (self.step * data.strides[-1], data.strides[-1])
        segments = np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides
        )

        # Compute one-sided PSD for each segment
        freqs_full, psd_full = signal.welch(
            segments,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            axis=-1,
            return_onesided=True,
            scaling="density",
        )

        # Apply frequency mask if requested
        freq_mask = np.ones_like(freqs_full, dtype=bool)
        if self.min_freq is not None:
            freq_mask &= freqs_full >= self.min_freq
        if self.max_freq is not None:
            freq_mask &= freqs_full <= self.max_freq

        self.freqs = freqs_full[freq_mask]
        self.psds = psd_full[:, freq_mask]
        self.median_psd = np.median(self.psds, axis=0)

    @classmethod
    def download(
        cls,
        detector: str = "H1",
        gps_start: int = 1126259462,
        duration: int = 1024,
        channel: Optional[str] = None,
    ) -> TimeSeries:
        """
        Download open strain data from GWOSC for a given detector and GPS range.
        """
        print(
            f"Downloading {detector} data [{gps_start} - {gps_start + duration}]..."
        )
        strain = TimeSeries.fetch_open_data(
            detector, gps_start, gps_start + duration
        )
        if channel:
            strain.channel = channel
        return strain

    @classmethod
    def load(
        cls,
        detector: str = "H1",
        gps_start: int = 1126259462,
        duration: int = 1024,
        segment_duration: int = 4,
        segment_overlap: float = 0.5,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        cache_file: str = "strain_cache.gwf",
        channel: str = "H1:GWOSC-STRAIN",
    ) -> "LVKData":
        """
        Load strain data from cache or download if needed, then compute PSDs.
        """
        if os.path.exists(cache_file):
            try:
                strain = TimeSeries.read(cache_file)
                print(f"Loaded cached strain from '{cache_file}'")
            except Exception as e:
                print(
                    f"Failed to read cache '{cache_file}': {e}. Redownloading..."
                )
                os.remove(cache_file)
                strain = cls.download(
                    detector, gps_start, duration, channel=channel
                )
                strain.write(cache_file)
                print(f"Cached new strain to '{cache_file}'")
        else:
            strain = cls.download(
                detector, gps_start, duration, channel=channel
            )
            strain.write(cache_file)
            print(f"Cached strain to '{cache_file}'")

        return cls(
            strain=strain,
            duration=duration,
            segment_duration=segment_duration,
            segment_overlap=segment_overlap,
            min_freq=min_freq,
            max_freq=max_freq,
        )

    def compute_median_psd(
        self, n_segments: Optional[int] = None
    ) -> np.ndarray:
        """
        Return the median PSD computed over the first `n_segments` segments.
        """
        if n_segments is None:
            n_segments = self.n_segments
        if n_segments > self.n_segments:
            raise ValueError(
                "n_segments exceeds available number of segments."
            )
        return np.median(self.psds[:n_segments, :], axis=0)

    def plot_psd(self) -> plt.Figure:
        """
        Plot all individual-segment PSDs in gray and the median PSD in red.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(self.freqs, self.psds.T, color="gray", alpha=0.3)
        ax.loglog(
            self.freqs, self.median_psd, color="r", lw=2, label="Median PSD"
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD [strain^2/Hz]")
        ax.set_title(f"PSD: {self.strain.channel}")
        ax.grid(True, which="both", ls=":")
        ax.legend()
        return fig
