import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from gwpy.timeseries import TimeSeries, StateVector
import matplotlib.pyplot as plt


# ====================================================================
# Data structures for spectral line detection
# ====================================================================

@dataclass
class LineDetail:
    """Information about a detected spectral line."""
    f_start: float  # Start frequency of the line (extended to wings)
    f_end: float  # End frequency of the line (extended to wings)
    f_peak: float  # Frequency of maximum power
    bandwidth: float  # Line bandwidth (f_end - f_start)
    max_ratio: float  # Maximum ratio of Pxx/running_median
    peak_power: float  # Power at the peak frequency
    core_start: float  # Start of core region (above threshold)
    core_end: float  # End of core region (above threshold)

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
        """Number of detected lines."""
        return len(self.line_details)

    @property
    def line_frequencies(self) -> np.ndarray:
        """Peak frequencies of all detected lines."""
        return np.array([line.f_peak for line in self.line_details])

    @property
    def line_bandwidths(self) -> np.ndarray:
        """Bandwidths of all detected lines."""
        return np.array([line.bandwidth for line in self.line_details])


# ====================================================================
# Helper functions for spectral line detection
# ====================================================================

def _estimate_frequency_resolution(frequencies: np.ndarray) -> float:
    """Estimate the frequency resolution, handling non-uniform spacing."""
    freq_diffs = np.diff(frequencies)
    freq_resolution = np.median(freq_diffs)
    if not np.allclose(freq_diffs, freq_resolution, rtol=1e-3, atol=1e-6 * freq_resolution):
        warnings.warn(
            f"Frequency spacing not uniform. Using median spacing "
            f"df = {freq_resolution:.3e} Hz for window sizing."
        )
    return freq_resolution


def _compute_running_median(
        periodogram: np.ndarray,
        window_width_hz: float,
        freq_resolution: float
) -> np.ndarray:
    """Compute running median background estimate using a median filter."""
    from scipy.signal import medfilt
    window_bins = max(1, int(np.round(window_width_hz / freq_resolution)))
    kernel_size = window_bins + (1 - window_bins % 2)
    return medfilt(periodogram, kernel_size=kernel_size)


def _detect_line_candidates(
        frequencies: np.ndarray,
        periodogram: np.ndarray,
        running_median: np.ndarray,
        threshold: float,
        fmin: Optional[float],
        fmax: Optional[float]
) -> np.ndarray:
    """Detect frequency bins that may contain spectral lines."""
    power_ratio = periodogram / (running_median + np.finfo(float).eps)
    freq_mask = np.ones_like(frequencies, dtype=bool)
    if fmin is not None:
        freq_mask &= frequencies >= fmin
    if fmax is not None:
        freq_mask &= frequencies <= fmax
    is_line_bin = (power_ratio > threshold) & freq_mask
    return is_line_bin


def _find_contiguous_regions(boolean_array: np.ndarray) -> List[Tuple[int, int]]:
    """Find start and end indices of contiguous True regions."""
    if not boolean_array.any():
        return []
    padded = np.concatenate([[False], boolean_array, [False]])
    transitions = np.diff(padded.astype(int))
    start_indices = np.where(transitions == 1)[0]
    end_indices = np.where(transitions == -1)[0] - 1
    return list(zip(start_indices, end_indices))


def _extend_line_boundaries(
        power_ratio: np.ndarray,
        start_idx: int,
        end_idx: int,
        threshold: float
) -> Tuple[int, int]:
    """
    Extend line boundaries to include the transition points:
    - Last point before ratio goes above threshold
    - First point after ratio goes below threshold
    """
    n_bins = len(power_ratio)

    # Extend left boundary - find last point before going above threshold
    extended_start = start_idx
    for i in range(start_idx - 1, -1, -1):
        if power_ratio[i] < threshold:
            extended_start = i  # Include this point (last one below threshold)
            break
        # If we reach here, power_ratio[i] >= threshold, so continue looking
    else:
        # We reached the beginning without finding a point below threshold
        extended_start = 0

    # Extend right boundary - find first point after going below threshold
    extended_end = end_idx
    for i in range(end_idx + 1, n_bins):
        if power_ratio[i] < threshold:
            extended_end = i  # Include this point (first one below threshold)
            break
        # If we reach here, power_ratio[i] >= threshold, so continue looking
    else:
        # We reached the end without finding a point below threshold
        extended_end = n_bins - 1

    return extended_start, extended_end


def _extract_line_details(
        frequencies: np.ndarray,
        periodogram: np.ndarray,
        running_median: np.ndarray,
        is_line_bin: np.ndarray,
        threshold: float
) -> List[LineDetail]:
    """Extract details for each detected spectral line with extended bandwidth."""
    if not is_line_bin.any():
        return []

    line_regions = _find_contiguous_regions(is_line_bin)
    line_details = []
    power_ratio = periodogram / (running_median + np.finfo(float).eps)

    processed_regions = []  # Track extended regions to avoid overlaps

    for start_idx, end_idx in line_regions:
        # Store core region (above threshold)
        core_start_freq = frequencies[start_idx]
        core_end_freq = frequencies[end_idx]

        # Extend boundaries to include wings
        extended_start, extended_end = _extend_line_boundaries(
            power_ratio, start_idx, end_idx, threshold
        )

        # Check for overlap with previously processed regions and merge if needed
        merged = False
        for i, (prev_start, prev_end) in enumerate(processed_regions):
            if not (extended_end < prev_start or extended_start > prev_end):
                # Merge overlapping regions
                processed_regions[i] = (min(extended_start, prev_start),
                                        max(extended_end, prev_end))
                merged = True
                break

        if not merged:
            processed_regions.append((extended_start, extended_end))

            # Calculate line properties using extended boundaries
            f_start = frequencies[extended_start]
            f_end = frequencies[extended_end]
            bandwidth = f_end - f_start

            # Find peak within the original core region (above threshold)
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
                peak_power=peak_power,
                core_start=core_start_freq,
                core_end=core_end_freq
            ))

    return line_details

class LVKData:

    def __init__(
            self,
            strain: np.ndarray,
            psd: np.ndarray,
            freqs: np.ndarray,
            fmin: Optional[float] = 20.0,
            fmax: Optional[float] = 2048.0
    ):
        self.strain = strain
        self.freqs = freqs
        self.df = freqs[1] - freqs[0]
        self.fs = freqs[-1] * 2  # Nyquist frequency
        self.dt = 1.0 / self.fs
        self.time = np.arange(len(strain)) * self.dt

        self.psd = psd
        self.line_analysis_result: Optional[PSDEstimate] = None
        self.knots_locations: Optional[np.ndarray] = None
        self.fmin = fmin
        self.fmax = fmax

        self._identify_lines(fmin=fmin, fmax=fmax)

    @classmethod
    def download_data(cls,
                      detector: str = "H1",
                      gps_start: int = 1126259462,
                      duration: int = 1024,
                      fmin: Optional[float] = 20.0,
                        fmax: Optional[float] = 2048.0
                      ):
        gps_end = gps_start + duration
        print(f"Downloading {detector} data [{gps_start} - {gps_end}]")
        strain = TimeSeries.fetch_open_data(detector, gps_start, gps_end, )
        psd = strain.psd()
        print("Data download and validation successful.")
        return cls(
            strain=strain.value,
            psd=psd.value,
            freqs=psd.frequencies.value,
            fmin=fmin,
            fmax=fmax
        )

    def _identify_lines(
            self,
            window_width_hz: float = 16.0,
            threshold: float = 30.0,
            fmin: Optional[float] = 20.0,
            fmax: Optional[float] = 2048.0,
    ) -> Optional[PSDEstimate]:
        """Identify spectral lines with improved bandwidth calculation."""

        # Estimate PSD with lines
        result = self._estimate_psd_with_lines(
            frequencies=self.freqs,
            periodogram=self.psd,
            window_width_hz=window_width_hz,
            threshold=threshold,
            fmin=fmin,
            fmax=fmax
        )
        self.line_analysis_result = result

        # Calculate knot locations with improved algorithm
        self.knots_locations = self._calculate_knots(result.line_details, fmin, fmax)

        print(f"Found {result.n_lines} spectral lines with extended bandwidth calculation.")
        if result.n_lines > 0:
            total_bandwidth = sum(line.bandwidth for line in result.line_details)
            avg_bandwidth = total_bandwidth / result.n_lines
            print(f"Average line bandwidth: {avg_bandwidth:.2f} Hz")
            print(f"Total spectral content in lines: {total_bandwidth:.2f} Hz")

        print(f"Generated {len(self.knots_locations)} knot locations for spline fitting.")

        return result

    def update_frequency_range(self, fmin: float, fmax: float) -> None:
        """Update the frequency range and re-run the analysis."""
        self.fmin = fmin
        self.fmax = fmax
        print(f"Updated frequency range to [{fmin:.1f}, {fmax:.1f}] Hz")
        self._identify_lines()

    def print_knot_info(self) -> None:
        """Print detailed information about knot placement."""
        if self.knots_locations is None or len(self.knots_locations) == 0:
            print("No knots available. Run analysis first.")
            return

        print(f"\n=== KNOT PLACEMENT SUMMARY ===")
        print(f"Total knots: {len(self.knots_locations)}")
        print(f"Frequency range: {self.knots_locations.min():.1f} - {self.knots_locations.max():.1f} Hz")
        print(f"Knot locations: {', '.join([f'{k:.1f}' for k in self.knots_locations])}")

        if self.line_analysis_result and self.line_analysis_result.n_lines > 0:
            print(f"\n=== LINE-BASED KNOTS ===")
            for i, line in enumerate(self.line_analysis_result.line_details):
                print(f"Line {i + 1}: Peak={line.f_peak:.1f} Hz, "
                      f"Boundaries=[{line.f_start:.1f}, {line.f_end:.1f}] Hz, "
                      f"Bandwidth={line.bandwidth:.2f} Hz")

            # Identify which knots are at boundaries vs midpoints
            line_boundaries = []
            line_peaks = []
            for line in self.line_analysis_result.line_details:
                line_boundaries.extend([line.f_start, line.f_end])
                line_peaks.append(line.f_peak)

            boundary_knots = [k for k in self.knots_locations if any(abs(k - b) < 0.1 for b in line_boundaries)]
            peak_knots = [k for k in self.knots_locations if any(abs(k - p) < 0.1 for p in line_peaks)]
            other_knots = [k for k in self.knots_locations if k not in boundary_knots and k not in peak_knots]

            print(f"\nBoundary knots (where power drops below threshold): {len(boundary_knots)}")
            print(f"Peak knots: {len(peak_knots)}")
            print(f"Midpoint/gap knots: {len(other_knots)}")

    def _calculate_knots(self, line_details: List[LineDetail], fmin: float = 20.0, fmax: float = 2048.0) -> np.ndarray:
        """Calculate knot locations using improved bandwidth estimates."""
        knots = []

        if len(line_details) > 0:
            # 1. Extended line boundaries and center locations
            for line in line_details:
                # Use geometric center instead of peak
                line_center = (line.f_start + line.f_end) / 2
                knots.extend([line.f_start, line.f_end, line_center])

                # Add quarter points for better resolution within broad lines
                if line.bandwidth > 2.0:  # Only for lines broader than 2 Hz
                    quarter_1 = line.f_start + 0.25 * line.bandwidth
                    quarter_3 = line.f_start + 0.75 * line.bandwidth
                    knots.extend([quarter_1, quarter_3])

            # 2. Midpoints between line centers (not peaks)
            sorted_line_centers = sorted([(line.f_start + line.f_end) / 2 for line in line_details])
            if len(sorted_line_centers) > 1:
                midpoints = (np.array(sorted_line_centers[:-1]) + np.array(sorted_line_centers[1:])) / 2
                knots.extend(midpoints)

                # 3. Additional knots between the midpoints for better coverage
                if len(midpoints) > 1:
                    between_midpoints = (midpoints[:-1] + midpoints[1:]) / 2
                    knots.extend(between_midpoints)

            # 4. Add boundary knots if lines don't extend to frequency limits
            all_line_freqs = [line.f_start for line in line_details] + [line.f_end for line in line_details]
            min_line_freq = min(all_line_freqs)
            max_line_freq = max(all_line_freqs)

            # Add knots in gaps at the beginning and end
            if min_line_freq > fmin:
                gap_knots = np.linspace(fmin, min_line_freq, num=3)[:-1]  # Exclude endpoint
                knots.extend(gap_knots)

            if max_line_freq < fmax:
                gap_knots = np.linspace(max_line_freq, fmax, num=3)[1:]  # Exclude startpoint
                knots.extend(gap_knots)

        else:
            # If no lines detected, add some basic knots across the frequency range
            knots = np.logspace(np.log10(fmin), np.log10(fmax), num=10)

        return np.unique(np.array(knots))

    def _enforce_minimum_separation(self, knots: np.ndarray, min_sep_hz: float = 0.5) -> np.ndarray:
        """Remove knots that are too close together."""
        if len(knots) <= 1:
            return knots

        filtered_knots = [knots[0]]  # Always keep first knot

        for knot in knots[1:]:
            # Only add if sufficiently separated from the last added knot
            if knot - filtered_knots[-1] >= min_sep_hz:
                filtered_knots.append(knot)

        return np.array(filtered_knots)

    def _estimate_psd_with_lines(
            self,
            frequencies: np.ndarray,
            periodogram: np.ndarray,
            window_width_hz: float,
            threshold: float,
            fmin: Optional[float],
            fmax: Optional[float],
    ) -> PSDEstimate:
        """
        Estimate a PSD model by detecting and modeling narrow spectral lines.
        """
        frequencies = np.asarray(frequencies, dtype=float)
        periodogram = np.asarray(periodogram, dtype=float)

        # Rescale periodogram to avoid working with very small scale values
        periodogram = periodogram / np.nanmax(periodogram) * 1e-3

        if len(frequencies) != len(periodogram):
            raise ValueError("frequencies and periodogram must have the same length")

        if len(frequencies) < 3:
            raise ValueError("Need at least 3 frequency bins")

        freq_resolution = _estimate_frequency_resolution(frequencies)
        running_median = _compute_running_median(
            periodogram, window_width_hz, freq_resolution
        )
        is_line_bin = _detect_line_candidates(
            frequencies, periodogram, running_median, threshold, fmin, fmax
        )
        line_details = _extract_line_details(
            frequencies, periodogram, running_median, is_line_bin, threshold
        )
        psd_model = np.where(is_line_bin, periodogram, running_median)

        return PSDEstimate(
            frequencies=frequencies,
            periodogram=periodogram,
            running_median=running_median,
            psd_model=psd_model,
            is_line_bin=is_line_bin,
            line_details=line_details
        )

    def plot_psd_analysis(
            self,
            include_lines: bool = True,
            figsize: Tuple[float, float] = (12, 8),
            show_legend: bool = True,
            line_color: str = 'red',
            knot_color: str = 'orange',
            fname:str = 'psd_analysis_plot.png'
    ) -> None:
        """
        Create a comprehensive plot of the PSD analysis results.
        """
        if self.psd is None or self.freqs is None or self.line_analysis_result is None:
            print("No analysis results available to plot. Please run analysis first.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        self._plot_psd_main(ax1, self.line_analysis_result, include_lines, line_color, knot_color, show_legend)
        self._plot_power_ratio(ax2, self.line_analysis_result, line_color)

        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight', dpi=300)

    def _plot_psd_main(
            self,
            ax: plt.Axes,
            result: PSDEstimate,
            include_lines: bool,
            line_color: str,
            knot_color: str,
            show_legend: bool
    ) -> None:
        """Plot the main PSD analysis panel."""
        freqs = result.frequencies
        ax.loglog(freqs, result.periodogram, 'lightgray', alpha=0.7,
                  label='Raw periodogram', linewidth=0.8)
        ax.loglog(freqs, result.running_median, 'blue',
                  label='Background estimate', linewidth=2)

        # Plot lines and PSD model
        if include_lines:
            ax.loglog(freqs, result.psd_model, 'black',
                      label='PSD model', linewidth=2)
            if result.n_lines > 0:
                # Plot extended line regions
                for i, line in enumerate(result.line_details):
                    mask = (freqs >= line.f_start) & (freqs <= line.f_end)
                    if mask.any():
                        ax.fill_between(freqs[mask], result.periodogram[mask],
                                        result.running_median[mask],
                                        color=line_color, alpha=0.3,
                                        label='Extended lines' if i == 0 else "")

                # Mark peaks
                peak_freqs = [line.f_peak for line in result.line_details]
                peak_powers = [result.periodogram[np.argmin(np.abs(freqs - f))] for f in peak_freqs]
                ax.scatter(peak_freqs, peak_powers, c='black', s=30,
                           label=f'Peaks ({result.n_lines})', zorder=5, marker='x')

        # Plot knots as circle markers on the PSD model
        if self.knots_locations is not None and self.knots_locations.size > 0:
            freq_min, freq_max = freqs.min(), freqs.max()
            valid_knots = self.knots_locations[(self.knots_locations >= freq_min) &
                                               (self.knots_locations <= freq_max)]

            if len(valid_knots) > 0 and include_lines:
                # Find PSD model values at knot locations
                knot_psd_values = []
                knot_freqs_plot = []
                for knot_freq in valid_knots:
                    # Find closest frequency bin
                    idx = np.argmin(np.abs(freqs - knot_freq))
                    knot_psd_values.append(result.psd_model[idx])
                    knot_freqs_plot.append(freqs[idx])

                ax.scatter(knot_freqs_plot, knot_psd_values,
                           c=knot_color, s=50, marker='o',
                           edgecolors='black', linewidth=1,
                           label=f'Knots ({len(valid_knots)})', zorder=6)

        ax.set_ylabel('Power Spectral Density')
        ax.set_title('PSD Analysis with Extended Line Detection')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(self.fmin, self.fmax)
        if show_legend:
            ax.legend(loc='upper left', fontsize=10, frameon=False)

    def _plot_power_ratio(self, ax: plt.Axes, result: PSDEstimate, line_color: str) -> None:
        """Plot the power ratio in the bottom panel."""
        freqs = result.frequencies
        eps = np.finfo(float).eps
        power_ratio = result.periodogram / (result.running_median + eps)

        ax.loglog(freqs, power_ratio, 'gray', linewidth=1, label='Power ratio')

        if result.n_lines > 0:
            # Highlight extended line regions
            for line in result.line_details:
                mask = (freqs >= line.f_start) & (freqs <= line.f_end)
                if mask.any():
                    ax.loglog(freqs[mask], power_ratio[mask], color=line_color,
                              linewidth=2, alpha=0.8)

        ax.axhline(30, color='red', linestyle=':', alpha=0.7, label='Threshold=30')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Ratio')
        ax.set_title('Detection Criterion (Periodogram / Background)')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(self.fmin, self.fmax)
        ax.legend(loc='upper left', fontsize=10, frameon=False)

