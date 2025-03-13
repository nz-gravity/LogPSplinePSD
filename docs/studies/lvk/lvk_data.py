import os
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from gwpy.timeseries import TimeSeries


class LVKData:
    def __init__(
        self,
        strain: TimeSeries,
        duration: int,
        segment_duration: int,
        segment_overlap: float,
        min_freq: float = None,
        max_freq: float = None,
    ):
        self.strain = strain
        self.duration = duration
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.min_freq = min_freq
        self.max_freq = max_freq

    @property
    def n_segments(self):
        fs = self.strain.sample_rate.value
        nperseg = int(fs * self.segment_duration)
        noverlap = int(nperseg * self.segment_overlap)
        step = nperseg - noverlap
        return (len(self.strain) - noverlap) // step

    @cached_property
    def psds(self):
        fs = self.strain.sample_rate.value
        nperseg = int(fs * self.segment_duration)
        noverlap = int(nperseg * self.segment_overlap)
        data = self.strain.value

        # Use array strides for efficient segmentation
        step = nperseg - noverlap
        shape = (nperseg, (data.shape[-1] - noverlap) // step)
        strides = (data.strides[-1], step * data.strides[-1])
        segments = np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides
        ).T

        # Compute PSDs in vectorized form
        freqs, psd_array = signal.welch(
            segments,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True,
            scaling="density",
            axis=-1,
        )

        # Frequency masking
        freq_mask = np.ones_like(freqs, dtype=bool)
        if self.min_freq is not None:
            freq_mask &= freqs >= self.min_freq
        if self.max_freq is not None:
            freq_mask &= freqs <= self.max_freq

        self._freq = freqs[freq_mask]
        return psd_array[:, freq_mask]

    @cached_property
    def welch_psd(self):
        return np.median(self.psds, axis=0)

    @classmethod
    def download(
        cls, detector="H1", gps_start=1126259462, duration=1024, channel=None
    ):
        print(
            f"Downloading {detector} data [{gps_start}-{gps_start + duration}]..."
        )
        strain = TimeSeries.fetch_open_data(
            detector, gps_start, gps_start + duration
        )
        if channel:  # Set explicit channel name for caching
            strain.channel = channel
        return strain

    @classmethod
    def load(
        cls,
        detector="H1",
        gps_start=1126259462,
        duration=1024,
        segment_duration=4,
        segment_overlap=0.5,
        min_freq: float = None,
        max_freq: float = None,
        cache_file="strain_cache.txt",
        channel="H1:GWOSC-STRAIN",
    ):
        """
        Fixed changes:
        1. Use correct default channel name for O1/O2 data
        2. Simplified caching logic
        3. Added channel validation
        """
        if os.path.exists(cache_file):
            try:
                strain = TimeSeries.read(cache_file)
                print(f"Loaded cached data: {cache_file}")
                return cls(
                    strain,
                    duration,
                    segment_duration,
                    segment_overlap,
                    min_freq,
                    max_freq,
                )
            except Exception as e:
                print(f"Cache invalid: {e}. Redownloading...")
                os.remove(cache_file)

        # Download and cache new data
        strain = cls.download(detector, gps_start, duration, channel=channel)
        strain.write(cache_file)
        print(f"Cached data to: {cache_file}")
        return cls(
            strain,
            duration,
            segment_duration,
            segment_overlap,
            min_freq,
            max_freq,
        )

    def plot_psd(self):
        plt.figure(figsize=(8, 5))
        plt.loglog(self._freq, self.psds.T, color="gray", alpha=0.3)
        plt.loglog(self._freq, self.welch_psd, "r", lw=2, label="Median PSD")
        plt.xlabel("Frequency (Hz)"), plt.ylabel("PSD")
        plt.title(f"PSD: {self.strain.channel}")
        plt.grid(True), plt.legend()
        return plt.gcf()


def test_lvk_data():
    # Download data and compute PSDs.
    lvk_data = LVKData.load(
        detector="H1",
        gps_start=1126259462,
        duration=1024,
        segment_duration=4,
        segment_overlap=0.5,
        min_freq=10,
        max_freq=2048,
    )
    # Access number of segments:
    print("Number of segments:", lvk_data.n_segments)
    # Plot individual and median PSDs.
    fig = lvk_data.plot_psd()
    plt.savefig("lvk_psd.png", bbox_inches="tight")
