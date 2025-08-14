import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt


@dataclass
class LineDetail:
    """Information about a detected spectral line."""
    f_start: float  # Start frequency of the line
    f_end: float  # End frequency of the line
    f_peak: float  # Frequency of maximum power
    bandwidth: float  # Line bandwidth (f_end - f_start)
    max_ratio: float  # Maximum ratio of Pxx/running_median
    peak_power: float  # Power at the peak frequency

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


def estimate_psd_with_lines(
        frequencies: np.ndarray,
        periodogram: np.ndarray,
        window_width_hz: float = 8.0,
        threshold: float = 10.0,
        fmin: Optional[float] = 20.0,
        fmax: Optional[float] = 2048.0,
) -> PSDEstimate:
    """
    Estimate a PSD model by detecting and modeling narrow spectral lines.

    This function identifies narrow spectral lines that stand out above the
    broadband noise background, then creates a PSD model that preserves the
    lines while smoothing the background.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency bins of the periodogram (Hz), assumed roughly uniform
    periodogram : np.ndarray
        Periodogram power values
    window_width_hz : float, default=8.0
        Width of the median filter window in Hz. Should be larger than
        expected line widths but smaller than background variation scales
    threshold : float, default=10.0
        Line detection threshold. Bins with power > threshold × running_median
        are marked as line candidates
    fmin : float, optional, default=20.0
        Minimum frequency for line detection. If None, no lower bound
    fmax : float, optional, default=2048.0
        Maximum frequency for line detection. If None, no upper bound

    Returns
    -------
    PSDEstimate
        Complete results including detected lines and PSD model

    Notes
    -----
    The algorithm works by:
    1. Computing a running median to estimate the background PSD
    2. Finding bins where power significantly exceeds the background
    3. Grouping contiguous line bins into discrete line features
    4. Creating a hybrid model: lines where detected, background elsewhere
    """
    # Input validation and preprocessing
    frequencies = np.asarray(frequencies, dtype=float)
    periodogram = np.asarray(periodogram, dtype=float)

    if len(frequencies) != len(periodogram):
        raise ValueError("frequencies and periodogram must have the same length")

    if len(frequencies) < 3:
        raise ValueError("Need at least 3 frequency bins")

    # Estimate frequency resolution
    freq_resolution = _estimate_frequency_resolution(frequencies)

    # Compute running median background estimate
    running_median = _compute_running_median(
        periodogram, window_width_hz, freq_resolution
    )

    # Detect line candidates
    is_line_bin = _detect_line_candidates(
        frequencies, periodogram, running_median, threshold, fmin, fmax
    )

    # Group contiguous line bins and extract line details
    line_details = _extract_line_details(
        frequencies, periodogram, running_median, is_line_bin
    )

    # Create final PSD model
    psd_model = np.where(is_line_bin, periodogram, running_median)

    return PSDEstimate(
        frequencies=frequencies,
        periodogram=periodogram,
        running_median=running_median,
        psd_model=psd_model,
        is_line_bin=is_line_bin,
        line_details=line_details
    )


def _estimate_frequency_resolution(frequencies: np.ndarray) -> float:
    """Estimate the frequency resolution, handling non-uniform spacing."""
    freq_diffs = np.diff(frequencies)
    freq_resolution = np.median(freq_diffs)

    # Check for roughly uniform spacing
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
    """Compute running median background estimate."""
    # Convert window width to number of bins
    window_bins = max(1, int(np.round(window_width_hz / freq_resolution)))
    # Ensure odd kernel size
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
    # Compute power ratio (with protection against division by zero)
    eps = np.finfo(float).eps
    power_ratio = periodogram / (running_median + eps)

    # Apply frequency range mask
    freq_mask = np.ones_like(frequencies, dtype=bool)
    if fmin is not None:
        freq_mask &= frequencies >= fmin
    if fmax is not None:
        freq_mask &= frequencies <= fmax

    # Identify line candidates
    is_line_bin = (power_ratio > threshold) & freq_mask

    return is_line_bin


def _extract_line_details(
        frequencies: np.ndarray,
        periodogram: np.ndarray,
        running_median: np.ndarray,
        is_line_bin: np.ndarray
) -> List[LineDetail]:
    """Extract details for each detected spectral line."""
    if not is_line_bin.any():
        return []

    # Find contiguous line regions
    line_regions = _find_contiguous_regions(is_line_bin)

    # Extract details for each line
    line_details = []
    power_ratio = periodogram / (running_median + np.finfo(float).eps)

    for start_idx, end_idx in line_regions:
        # Get line boundaries
        f_start = frequencies[start_idx]
        f_end = frequencies[end_idx]
        bandwidth = f_end - f_start

        # Find peak within the line
        line_slice = slice(start_idx, end_idx + 1)
        peak_idx_rel = np.argmax(power_ratio[line_slice])
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


def _find_contiguous_regions(boolean_array: np.ndarray) -> List[Tuple[int, int]]:
    """Find start and end indices of contiguous True regions."""
    if not boolean_array.any():
        return []

    # Pad with False to catch boundary transitions
    padded = np.concatenate([[False], boolean_array, [False]])
    transitions = np.diff(padded.astype(int))

    start_indices = np.where(transitions == 1)[0]  # False -> True
    end_indices = np.where(transitions == -1)[0] - 1  # True -> False

    return list(zip(start_indices, end_indices))


def plot_psd_analysis(
        result: PSDEstimate,
        knots: Optional[np.ndarray] = None,
        knot_values: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (12, 8),
        show_legend: bool = True,
        line_color: str = 'red',
        knot_color: str = 'orange'
) -> plt.Figure:
    """
    Create a comprehensive plot of the PSD analysis results.

    Parameters
    ----------
    result : PSDEstimate
        Results from estimate_psd_with_lines
    knots : np.ndarray, optional
        Knot frequencies to plot as vertical lines
    knot_values : np.ndarray, optional
        Values at knot locations for plotting points
    figsize : tuple, default=(12, 8)
        Figure size
    show_legend : bool, default=True
        Whether to show legend
    line_color : str, default='red'
        Color for detected lines
    knot_color : str, default='orange'
        Color for knot markers

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Main PSD plot
    _plot_psd_main(ax1, result, knots, knot_values, line_color, knot_color, show_legend)

    # Power ratio plot
    _plot_power_ratio(ax2, result, line_color)

    plt.tight_layout()
    return fig


def _plot_psd_main(
        ax: plt.Axes,
        result: PSDEstimate,
        knots: Optional[np.ndarray],
        knot_values: Optional[np.ndarray],
        line_color: str,
        knot_color: str,
        show_legend: bool
) -> None:
    """Plot the main PSD analysis panel."""
    freqs = result.frequencies

    # Plot periodogram and background
    ax.loglog(freqs, result.periodogram, 'lightgray', alpha=0.7,
              label='Raw periodogram', linewidth=0.8)
    ax.loglog(freqs, result.running_median, 'blue',
              label='Background estimate', linewidth=2)
    ax.loglog(freqs, result.psd_model, 'black',
              label='PSD model', linewidth=2)

    # Highlight detected lines
    if result.n_lines > 0:
        line_freqs = freqs[result.is_line_bin]
        line_powers = result.periodogram[result.is_line_bin]
        ax.scatter(line_freqs, line_powers, c=line_color, s=15,
                   label=f'Lines ({result.n_lines})', alpha=0.8, zorder=5)

    # Plot knots if provided
    if knots is not None:
        for knot_freq in knots:
            if result.frequencies[0] <= knot_freq <= result.frequencies[-1]:
                ax.axvline(knot_freq, color=knot_color, alpha=0.6,
                           linestyle='--', linewidth=1)

        if knot_values is not None and len(knot_values) == len(knots):
            valid_mask = (knots >= result.frequencies[0]) & (knots <= result.frequencies[-1])
            ax.scatter(knots[valid_mask], knot_values[valid_mask],
                       c=knot_color, s=30, marker='v',
                       label=f'Knots ({np.sum(valid_mask)})', zorder=6)

    ax.set_ylabel('Power Spectral Density')
    ax.set_title('PSD Analysis with Line Detection')
    ax.grid(True, which='both', alpha=0.3)

    if show_legend:
        ax.legend(loc='upper right')


def _plot_power_ratio(ax: plt.Axes, result: PSDEstimate, line_color: str) -> None:
    """Plot the power ratio in the bottom panel."""
    freqs = result.frequencies
    eps = np.finfo(float).eps
    power_ratio = result.periodogram / (result.running_median + eps)

    ax.semilogx(freqs, power_ratio, 'gray', linewidth=1, label='Power ratio')

    # Highlight line regions
    if result.n_lines > 0:
        line_freqs = freqs[result.is_line_bin]
        line_ratios = power_ratio[result.is_line_bin]
        ax.scatter(line_freqs, line_ratios, c=line_color, s=10, alpha=0.8)

    ax.axhline(10, color='red', linestyle=':', alpha=0.7, label='Threshold=10')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Ratio')
    ax.set_title('Detection Criterion (Periodogram / Background)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()


def plot_line_summary(result: PSDEstimate, figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Create a summary plot showing line properties.

    Parameters
    ----------
    result : PSDEstimate
        Results from estimate_psd_with_lines
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    plt.Figure
        The created figure
    """
    if result.n_lines == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No lines detected',
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Line Detection Summary')
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Line frequencies vs strength
    line_freqs = result.line_frequencies
    line_ratios = [line.max_ratio for line in result.line_details]
    line_bandwidths = result.line_bandwidths

    scatter = ax1.scatter(line_freqs, line_ratios, c=line_bandwidths,
                          s=50, alpha=0.7, cmap='viridis')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Max Power Ratio')
    ax1.set_title(f'Detected Lines (n={result.n_lines})')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Bandwidth (Hz)')

    # Histogram of line properties
    ax2.hist(line_bandwidths, bins=min(10, result.n_lines), alpha=0.7,
             color='skyblue', edgecolor='black')
    ax2.set_xlabel('Line Bandwidth (Hz)')
    ax2.set_ylabel('Count')
    ax2.set_title('Bandwidth Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Convenience function for quick analysis
def analyze_psd_lines(
        frequencies: np.ndarray,
        periodogram: np.ndarray,
        plot: bool = True,
        **kwargs
) -> Union[PSDEstimate, Tuple[PSDEstimate, plt.Figure]]:
    """
    Convenience function to analyze PSD and optionally create plots.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency bins
    periodogram : np.ndarray
        Periodogram values
    plot : bool, default=True
        Whether to create plots
    **kwargs
        Additional arguments passed to estimate_psd_with_lines

    Returns
    -------
    PSDEstimate or (PSDEstimate, Figure)
        Analysis results, optionally with plot
    """
    result = estimate_psd_with_lines(frequencies, periodogram, **kwargs)

    if plot:
        fig = plot_psd_analysis(result)
        return result, fig
    else:
        return result