import numpy as np
from typing import List, Optional, Tuple
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt


class LvkKnotAllocator:
    def __init__(self,
                 freqs: np.ndarray,
                 psd: np.ndarray,
                 fmin: float = 20.0,
                 fmax: float = 2048.0,
                 window_width_hz: float = 8.0,
                 iqr_factor: float = 4.0,
                 d: int = 25,
                 extra_thresh_multiplier: float = 2.0,
                 max_extra_per_peak: int = 8,
                 degree=3,
                 knots_plotfn:str=None
                 ):
        self.freqs = np.asarray(freqs)
        self.psd = np.asarray(psd)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.window_width_hz = float(window_width_hz)
        self.iqr_factor = float(iqr_factor)
        self.d = int(d)
        self.extra_thresh_multiplier = float(extra_thresh_multiplier)
        self.max_extra_per_peak = int(max_extra_per_peak)

        self.knots_locations: Optional[np.ndarray] = None  # normalized [0,1]
        self.threshold: Optional[float] = None
        self.running_median: Optional[np.ndarray] = None
        self.is_line_bin: Optional[np.ndarray] = None
        self.peaks: Optional[np.ndarray] = None
        self.smoothed_peaks: Optional[np.ndarray] = None
        self.bin_regions: Optional[List[Tuple[int, int, str]]] = None

        self._identify_lines()
        self._process_peaks()
        self.knots = self.calculate_knots(degree=degree)
        self.knots_hz = self.knots * (self.fmax - self.fmin) + self.fmin
        print(f"Generated {len(self.knots)} adaptive knots ({self.knots_hz[0]:.1f}-{self.knots_hz[-1]:.1f} Hz)")

        if knots_plotfn is not None:
            self.plot_analysis(fname=knots_plotfn)

    # --- core analysis ---
    def _identify_lines(self) -> None:
        freq_resolution = np.median(np.diff(self.freqs))
        window_bins = max(1, int(np.round(self.window_width_hz / max(freq_resolution, 1e-12))))
        kernel_size = window_bins + (1 - window_bins % 2)
        self.running_median = median_filter(self.psd, size=kernel_size, mode='nearest')

        power_ratio = self.psd / (self.running_median + np.finfo(float).eps)
        q1, q3 = np.percentile(power_ratio, [25, 75])
        iqr = q3 - q1
        self.threshold = q3 + self.iqr_factor * iqr

        freq_mask = (self.freqs >= self.fmin) & (self.freqs <= self.fmax)
        self.is_line_bin = (power_ratio > self.threshold) & freq_mask

        n_lines = int(np.sum(np.diff(np.concatenate([[False], self.is_line_bin, [False]])) == 1))
        print(f"Found {n_lines} spectral line regions using threshold = {self.threshold:.2f}")

    def _extract_peaks(self) -> np.ndarray:
        power_ratio = self.psd / (self.running_median + np.finfo(float).eps)
        threshold = self.threshold
        freq_mask = (self.freqs >= self.fmin) & (self.freqs <= self.fmax)
        index = (power_ratio > threshold) & freq_mask
        out = np.zeros_like(self.psd)
        out[index] = np.log(power_ratio[index] + np.finfo(float).eps)
        return out

    def _smooth_peaks(self, peaks: np.ndarray, d: int = 10) -> np.ndarray:
        out = peaks.copy()
        n = len(peaks)
        gauss_win = gaussian(2 * d, 2.5)
        dec = gauss_win[:d][::-1]
        for i in range(d):
            i_r = i + 1
            aux1 = np.concatenate([peaks[i_r:], np.zeros(i_r)])
            aux2 = np.concatenate([np.zeros(i_r), peaks[:n - i_r]])
            out = np.maximum(out, aux1 * dec[i])
            out = np.maximum(out, aux2 * dec[i])
        sigma = max(1.0, d / 4.0)
        out = gaussian_filter1d(out, sigma=sigma, mode='nearest')
        band_mask = (self.freqs >= self.fmin) & (self.freqs <= self.fmax)
        out[~band_mask] = 0.0
        return out

    def _find_regions(self, binary_array: np.ndarray) -> List[Tuple[int, int]]:
        if len(binary_array) == 0:
            return []
        changes = np.diff(np.concatenate([[False], binary_array, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1
        max_idx = len(binary_array) - 1
        starts = np.clip(starts, 0, max_idx)
        ends = np.clip(ends, 0, max_idx)
        return [(int(s), int(e)) for s, e in zip(starts, ends) if s <= e]

    def _process_peaks(self) -> None:
        power_ratio = self.psd / (self.running_median + np.finfo(float).eps)
        log_pr = np.log(np.clip(power_ratio, a_min=1e-12, a_max=None))
        band_mask = (self.freqs >= self.fmin) & (self.freqs <= self.fmax)
        log_pr[~band_mask] = 0.0
        peaks_binary = (power_ratio > self.threshold) & band_mask
        sigma = max(1.0, self.d / 4.0)
        smoothed = gaussian_filter1d(np.where(peaks_binary, np.maximum(log_pr, 0.0), 0.0),
                                     sigma=sigma, mode='nearest')
        self.peaks = np.where(peaks_binary, np.maximum(log_pr, 0.0), 0.0)
        self.smoothed_peaks = smoothed

        if np.any(smoothed > 0.0):
            region_thresh = max(1e-6, 0.05 * np.nanmax(smoothed))
            has_peaks = smoothed >= region_thresh
            peak_regions = self._find_regions(has_peaks)
            zero_regions = self._find_regions(~has_peaks & band_mask)
            all_regions: List[Tuple[int, int, str]] = []
            for s, e in peak_regions:
                all_regions.append((s, e, 'peak'))
            for s, e in zero_regions:
                all_regions.append((s, e, 'zero'))
            self.bin_regions = sorted(all_regions, key=lambda x: x[0])
            peak_count = sum(1 for _, _, t in self.bin_regions if t == 'peak')
            zero_count = len(self.bin_regions) - peak_count
            print(f"Adaptive binning (d={self.d}): {peak_count} peak regions, {zero_count} zero regions")
        else:
            band_idxs = np.where(band_mask)[0]
            self.bin_regions = [(int(band_idxs[0]), int(band_idxs[-1]), 'zero')] if len(band_idxs) else []
            print("No significant peaks after smoothing; using single zero region.")

    # --- knot placement (always adaptive) ---
    def calculate_knots(self, degree: int = 3) -> np.ndarray:
        if self.smoothed_peaks is None or self.bin_regions is None:
            raise RuntimeError("Allocator not initialized")
        regions = list(self.bin_regions)
        if len(regions) == 0:
            self.knots_locations = np.array([0.0, 1.0])
            return self.knots_locations

        knots_hz: List[float] = []
        N = len(self.freqs)
        extra_thresh_bins = max(1, int(round(self.extra_thresh_multiplier * self.d)))

        for start_idx, end_idx, region_type in regions:
            s = max(0, min(int(start_idx), N - 1))
            e = max(0, min(int(end_idx), N - 1))
            if region_type == 'zero':
                center_freq = 0.5 * (self.freqs[s] + self.freqs[e])
                knots_hz.append(center_freq)
            else:
                left_freq = self.freqs[s]
                right_freq = self.freqs[e]
                local = self.smoothed_peaks[s:e + 1]
                if local.size == 0:
                    center_freq = 0.5 * (left_freq + right_freq)
                else:
                    center_idx = s + int(np.nanargmax(local))
                    center_idx = max(s, min(center_idx, e))
                    center_freq = self.freqs[center_idx]
                knots_hz.extend([left_freq, center_freq, right_freq])

                width_bins = e - s + 1
                if width_bins > extra_thresh_bins:
                    extras = int(np.floor(width_bins / extra_thresh_bins)) - 1
                    extras = max(0, min(extras, self.max_extra_per_peak))
                    if extras > 0 and np.sum(np.abs(local)) > 0:
                        dens = np.abs(local).astype(float)
                        dens_sum = np.sum(dens)
                        if dens_sum <= 0:
                            extra_freqs = np.linspace(self.freqs[s], self.freqs[e], extras + 2)[1:-1]
                        else:
                            dens = dens / dens_sum
                            cum = np.cumsum(dens)
                            x = np.concatenate([[0.0], cum])
                            y = np.linspace(self.freqs[s], self.freqs[e], len(local) + 1)
                            inv = interp1d(x, y, bounds_error=False,
                                           fill_value=(self.freqs[s], self.freqs[e]))
                            quantiles = (np.arange(1, extras + 1) / (extras + 1.0))
                            extra_freqs = inv(quantiles)
                        knots_hz.extend(np.asarray(extra_freqs).tolist())

        knots_hz.extend([self.fmin, self.fmax])
        knots_hz = np.array(knots_hz, dtype=float)
        knots_hz = np.unique(np.clip(knots_hz, self.fmin, self.fmax))
        norm = (knots_hz - self.fmin) / (self.fmax - self.fmin)
        self.knots_locations = np.sort(norm)
        return self.knots_locations


    # --- plotting ---
    def plot_analysis(self, figsize: Tuple[float, float] = (12, 8),
                      fname: str = 'psd_analysis.png', xscale: str = 'linear') -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        if xscale == 'log':
            ax1.loglog(self.freqs, self.psd, 'lightgray', alpha=0.7, label='PSD', linewidth=1)
            ax1.loglog(self.freqs, self.running_median, 'blue', label='Running median', linewidth=2)
        else:
            ax1.semilogy(self.freqs, self.psd, 'lightgray', alpha=0.7, label='PSD', linewidth=1)
            ax1.semilogy(self.freqs, self.running_median, 'blue', label='Running median', linewidth=2)

        valid_knots = np.array([])
        if self.knots_locations is not None:
            knots_hz = self.knots_locations * (self.fmax - self.fmin) + self.fmin
            valid_knots = knots_hz[(knots_hz >= self.fmin) & (knots_hz <= self.fmax)]
            if len(valid_knots) > 0:
                knot_psd_values = [self.psd[np.argmin(np.abs(self.freqs - kf))] for kf in valid_knots]
                ax1.scatter(valid_knots, knot_psd_values, c='orange', s=50,
                            marker='o', edgecolors='black', label=f'Knots ({len(valid_knots)})', zorder=6)

        if self.bin_regions:
            peak_count = sum(1 for _, _, t in self.bin_regions if t == 'peak')
            zero_count = len(self.bin_regions) - peak_count
            for start_idx, end_idx, region_type in self.bin_regions:
                start_freq = self.freqs[start_idx]
                end_freq = self.freqs[end_idx]
                color = 'green' if region_type == 'peak' else 'gray'
                alpha = 0.3 if region_type == 'peak' else 0.1
                ax1.axvline(start_freq, color=color, linestyle='--', alpha=alpha, linewidth=1)
                ax1.axvline(end_freq, color=color, linestyle='--', alpha=alpha, linewidth=1)
            if peak_count > 0:
                ax1.axvline(np.nan, color='green', linestyle='--', alpha=0.7,
                            label=f'Peak regions ({peak_count}) - edges+center')
            if zero_count > 0:
                ax1.axvline(np.nan, color='gray', linestyle='--', alpha=0.4,
                            label=f'Zero regions ({zero_count}) - center knots')

        ax1.set_ylabel('Power Spectral Density')
        ax1.set_title(f'PSD Analysis (smoothing d={self.d})')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(self.fmin, self.fmax)
        ax1.set_xscale(xscale)

        power_ratio = self.psd / (self.running_median + np.finfo(float).eps)
        log_power_ratio = np.log(power_ratio)
        log_threshold = np.log(self.threshold)
        ax2.plot(self.freqs, log_power_ratio, 'lightgray', alpha=0.7, linewidth=1, label='Log power ratio')
        ax2.plot(self.freqs, np.where(self.is_line_bin, log_power_ratio, np.nan),
                 color='red', linewidth=2, label='Detected lines')

        if self.smoothed_peaks is not None:
            sm = np.where(self.smoothed_peaks > 0, self.smoothed_peaks, np.nan)
            eps = 1e-12
            lp_finite = log_power_ratio[np.isfinite(log_power_ratio)]
            lp_max = np.nanmax(lp_finite) if lp_finite.size > 0 else 0.0
            sm_max = np.nanmax(sm[np.isfinite(sm)]) if np.any(np.isfinite(sm)) else 0.0
            if sm_max > eps:
                scale = (lp_max if lp_max > 0.0 else (log_threshold + 1.0)) / (sm_max + eps)
            else:
                scale = 1.0
            sm_scaled = sm * scale
            if np.any(np.isfinite(sm_scaled)):
                ax2.fill_between(self.freqs, 0, sm_scaled, alpha=0.3, color='purple',
                                 edgecolor='purple', linewidth=1.5, zorder=2, label='Knot density (scaled)')

        if self.bin_regions:
            for start_idx, end_idx, region_type in self.bin_regions:
                start_freq = self.freqs[start_idx]
                end_freq = self.freqs[end_idx]
                color = 'green' if region_type == 'peak' else 'gray'
                alpha = 0.3 if region_type == 'peak' else 0.1
                ax2.axvline(start_freq, color=color, linestyle='--', alpha=alpha, linewidth=1)
                ax2.axvline(end_freq, color=color, linestyle='--', alpha=alpha, linewidth=1)

        ax2.axhline(log_threshold, color='red', linestyle=':', alpha=0.7,
                    label=f'Log threshold={log_threshold:.2f}')

        if valid_knots.size > 0:
            lp_finite = log_power_ratio[np.isfinite(log_power_ratio)]
            lp_max = np.nanmax(lp_finite) if lp_finite.size > 0 else (log_threshold + 1.0)
            gap = max(0.3, 0.1 * abs(lp_max) + 0.3)
            marker_y = lp_max + gap
            ax2.scatter(valid_knots, np.ones_like(valid_knots) * marker_y,
                        c='orange', s=20, marker='v', zorder=6, alpha=0.8, edgecolors='black')

        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Log Power Ratio')
        ax2.legend(loc='upper left', fontsize='small')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(self.fmin, self.fmax)
        ax2.set_xscale(xscale)
        ax2.set_ylim(bottom=0.1)

        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f"KnotLoc saved as {fname} (xscale={xscale})")
        return fig, (ax1, ax2)
