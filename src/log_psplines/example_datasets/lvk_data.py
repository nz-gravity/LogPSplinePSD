import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from pycbc import noise as pycbc_noise
from pycbc import psd as pycbc_psd


@dataclass
class LineDetail:
    """Information about a detected spectral line."""
    f_start: float
    f_end: float
    f_peak: float
    bandwidth: float
    max_ratio: float
    peak_power: float

    def __repr__(self) -> str:
        return (f"LineDetail(f_peak={self.f_peak:.1f} Hz, "
                f"bandwidth={self.bandwidth:.2f} Hz, "
                f"max_ratio={self.max_ratio:.1f})")


@dataclass
class PSDEstimate:
    """Results from PSD estimation with line detection."""
    frequencies: np.ndarray
    periodogram: np.ndarray
    running_median: np.ndarray
    psd_model: np.ndarray
    is_line_bin: np.ndarray
    line_details: List[LineDetail]

    @property
    def n_lines(self) -> int:
        return len(self.line_details)


def _estimate_frequency_resolution(frequencies: np.ndarray) -> float:
    """Estimate the frequency resolution."""
    return np.median(np.diff(frequencies))


def _compute_running_median(periodogram: np.ndarray, window_width_hz: float, freq_resolution: float) -> np.ndarray:
    """Compute running median background estimate."""
    from scipy.ndimage import median_filter
    window_bins = max(1, int(np.round(window_width_hz / freq_resolution)))
    kernel_size = window_bins + (1 - window_bins % 2)
    return median_filter(periodogram, size=kernel_size, mode='nearest')


def _detect_line_candidates(frequencies: np.ndarray, periodogram: np.ndarray,
                            running_median: np.ndarray, threshold: float,
                            fmin: Optional[float], fmax: Optional[float]) -> np.ndarray:
    """Detect frequency bins that may contain spectral lines."""
    power_ratio = periodogram / (running_median + np.finfo(float).eps)
    freq_mask = np.ones_like(frequencies, dtype=bool)
    if fmin is not None:
        freq_mask &= frequencies >= fmin
    if fmax is not None:
        freq_mask &= frequencies <= fmax
    return (power_ratio > threshold) & freq_mask


def _find_contiguous_regions(boolean_array: np.ndarray) -> List[Tuple[int, int]]:
    """Find start and end indices of contiguous True regions."""
    if not boolean_array.any():
        return []
    padded = np.concatenate([[False], boolean_array, [False]])
    transitions = np.diff(padded.astype(int))
    start_indices = np.where(transitions == 1)[0]
    end_indices = np.where(transitions == -1)[0] - 1
    return list(zip(start_indices, end_indices))


def _extend_line_boundaries(power_ratio: np.ndarray, start_idx: int, end_idx: int, threshold: float) -> Tuple[int, int]:
    """Extend line boundaries to include transition points."""
    n_bins = len(power_ratio)

    # Extend left boundary
    extended_start = start_idx
    for i in range(start_idx - 1, -1, -1):
        if power_ratio[i] < threshold:
            extended_start = i
            break
    else:
        extended_start = 0

    # Extend right boundary
    extended_end = end_idx
    for i in range(end_idx + 1, n_bins):
        if power_ratio[i] < threshold:
            extended_end = i
            break
    else:
        extended_end = n_bins - 1

    return extended_start, extended_end


def _extract_line_details(frequencies: np.ndarray, periodogram: np.ndarray,
                          running_median: np.ndarray, is_line_bin: np.ndarray,
                          threshold: float) -> List[LineDetail]:
    """Extract details for each detected spectral line."""
    if not is_line_bin.any():
        return []

    line_regions = _find_contiguous_regions(is_line_bin)
    power_ratio = periodogram / (running_median + np.finfo(float).eps)
    line_details = []

    for start_idx, end_idx in line_regions:
        # Extend boundaries
        extended_start, extended_end = _extend_line_boundaries(power_ratio, start_idx, end_idx, threshold)

        # Calculate line properties
        f_start = frequencies[extended_start]
        f_end = frequencies[extended_end]
        bandwidth = f_end - f_start

        # Find peak within core region
        core_slice = slice(start_idx, end_idx + 1)
        peak_idx_rel = np.argmax(power_ratio[core_slice])
        peak_idx = start_idx + peak_idx_rel

        f_peak = frequencies[peak_idx]
        max_ratio = power_ratio[peak_idx]
        peak_power = periodogram[peak_idx]

        line_details.append(LineDetail(
            f_start=f_start,
            f_end=f_end,
            f_peak=f_peak,
            bandwidth=bandwidth,
            max_ratio=max_ratio,
            peak_power=peak_power
        ))

    return line_details


class LVKData:
    def __init__(self, strain: np.ndarray, psd: np.ndarray, freqs: np.ndarray,
                 fmin: float = 20, fmax: float = 2048.0, threshold:float=10.0, window_width_hz: float = 128.0):
        self.strain = strain

        self.freqs = freqs
        self.fs = 4096
        self.psd = psd
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        self.window_width_hz = window_width_hz
        self.line_analysis_result: Optional[PSDEstimate] = None
        self.knots_locations: Optional[np.ndarray] = None

        self._identify_lines()

    @classmethod
    def download_data(cls, detector: str = "H1", gps_start: int = 1126259462,
                      duration: int = 1024, fmin: float = 20, fmax: float = 2048):
        gps_end = gps_start + duration
        print(f"Downloading {detector} data [{gps_start} - {gps_end}]")
        strain = TimeSeries.fetch_open_data(detector, gps_start, gps_end)
        # rescale strain for standardization
        mean = strain.mean().value  # gwpy Quantity → strip units with .value
        std = strain.std().value
        strain = (strain - mean) / std

        psd = strain.psd()
        print("Data download and validation successful.")
        return cls(strain=strain.value, psd=psd.value, freqs=psd.frequencies.value,
                   fmin=fmin, fmax=fmax)

    # @classmethod
    # def simulate_data(cls,
    #                     duration: float = 4.0, fs: float = 2048.0, fmin: float = 20.0,
    #                   ):
    #
    #     psd_series = pycbc_psd.aLIGOAPlusDesignSensitivityT1800042(PSD_N, DF, FLOW)
    #     ts = pycbc_noise.noise_from_psd(10 * N, DT, psd_series, seed=0)
    #     psd = jnp.interp(
    #         FREQS,
    #         xp=psd_series.sample_frequencies.numpy(),
    #         fp=psd_series.data
    #     ) * (FS * N / 2.0)  # Scale PSD to match FFT normalization
    #     noise = np.array(ts.data[:N])  # Use first N samples of noise
    #


    def _identify_lines(self) -> None:
        """Identify spectral lines using single-scale approach."""
        print(f"Detecting lines with {self.window_width_hz} Hz window and threshold {self.threshold}")

        # Rescale for numerical stability
        periodogram = self.psd
        freq_resolution = _estimate_frequency_resolution(self.freqs)

        # Compute background and detect lines
        running_median = _compute_running_median(periodogram, self.window_width_hz, freq_resolution)
        is_line_bin = _detect_line_candidates(self.freqs, periodogram, running_median,
                                              self.threshold, self.fmin, self.fmax)
        line_details = _extract_line_details(self.freqs, periodogram, running_median,
                                             is_line_bin, self.threshold)

        # Create PSD model
        psd_model = np.where(is_line_bin, periodogram, running_median)

        self.line_analysis_result = PSDEstimate(
            frequencies=self.freqs,
            periodogram=periodogram,
            running_median=running_median,
            psd_model=psd_model,
            is_line_bin=is_line_bin,
            line_details=line_details
        )

        # Calculate knots
        self.knots_locations = self._calculate_knots()

        print(f"Found {len(line_details)} spectral lines")
        if line_details:
            bandwidths = [line.bandwidth for line in line_details]
            print(f"Bandwidth range: {min(bandwidths):.2f} - {max(bandwidths):.2f} Hz")
        print(f"Generated {len(self.knots_locations)} knots")

    def _calculate_knots(self, min_separation=3) -> np.ndarray:
        """Calculate knot locations with proper handling of broad lines."""
        knots = []

        # Start with background grid
        background_knots = np.logspace(np.log10(self.fmin), np.log10(self.fmax), num=10)

        # Line-based knots
        line_regions = []
        if self.line_analysis_result and self.line_analysis_result.line_details:
            for line in self.line_analysis_result.line_details:
                line_center = (line.f_start + line.f_end) / 2
                knots.extend([line.f_start, line.f_end, line_center])
                knots.extend([line.f_start-min_separation, line.f_end+min_separation])
                line_regions.append((line.f_start, line.f_end))

        # Only add background knots that DON'T fall within line regions
        for bg_knot in background_knots:
            is_inside_line = False
            for f_start, f_end in line_regions:
                if f_start <= bg_knot <= f_end:
                    is_inside_line = True
                    break
            if not is_inside_line:
                knots.append(bg_knot)

        # Remove duplicates and enforce minimum separation
        unique_knots = np.unique(np.array(knots))
        if len(unique_knots) <= 1:
            return unique_knots

        filtered_knots = [unique_knots[0]]
        for knot in unique_knots[1:]:
            if knot - filtered_knots[-1] >= 1.0:  # Increase minimum separation to 1.0 Hz
                filtered_knots.append(knot)

        return np.array(knots)

    def plot_psd_analysis(self, figsize: Tuple[float, float] = (12, 8),
                          fname: str = 'psd_analysis.png') -> None:
        """Create simplified plot of PSD analysis results."""
        if not self.line_analysis_result:
            print("No analysis results available.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        result = self.line_analysis_result
        freqs = result.frequencies

        # Top panel: PSD data
        ax1.loglog(freqs, result.periodogram, 'lightgray', alpha=0.7, label='Periodogram', linewidth=1)
        ax1.loglog(freqs, result.psd_model, 'red', label='PSD model', linewidth=2)
        ax1.loglog(freqs, result.running_median, 'blue', label='Background', linewidth=2, zorder=10)

        # Knot markers on top panel
        if self.knots_locations is not None and len(self.knots_locations) > 0:
            knot_psd_values = []
            for knot_freq in self.knots_locations:
                idx = np.argmin(np.abs(freqs - knot_freq))
                knot_psd_values.append(result.psd_model[idx])
            ax1.scatter(self.knots_locations, knot_psd_values, c='orange', s=50,
                        marker='o', edgecolors='black', label=f'Knots ({len(self.knots_locations)})', zorder=6)

        ax1.set_ylabel('Power Spectral Density')
        ax1.set_title(f'PSD Analysis ({self.fmin}-{self.fmax} Hz)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(self.fmin, self.fmax)

        # Bottom panel: Power ratio
        power_ratio = result.periodogram / (result.running_median + np.finfo(float).eps)
        ax2.loglog(freqs, power_ratio, color='gray', linewidth=1, label='Power ratio')
        ax2.loglog(freqs, np.where(power_ratio >= self.threshold, power_ratio, np.nan), color='red', linewidth=2,
                   label='Power ratio (above threshold)')

        # Knot markers on bottom panel
        if self.knots_locations is not None and len(self.knots_locations) > 0:
            knot_ratio_values = []
            for knot_freq in self.knots_locations:
                idx = np.argmin(np.abs(freqs - knot_freq))
                knot_ratio_values.append(power_ratio[idx])
            ax2.scatter(self.knots_locations, knot_ratio_values, c='orange', s=30,
                        marker='o', edgecolors='black', zorder=6)

        ax2.axhline(self.threshold, color='red', linestyle=':', alpha=0.7, label=f'Threshold={self.threshold}')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Ratio')
        ax2.set_title('Detection Criterion')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(self.fmin, self.fmax)
        ax2.set_xscale('linear')

        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f"Plot saved as {fname}")

