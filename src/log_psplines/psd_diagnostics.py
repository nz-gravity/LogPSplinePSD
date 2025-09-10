from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.special
import scipy.stats


class PSDDiagnostics:
    """
    Perform diagnostics on an estimated PSD and (optionally) compare it to a reference PSD.
    Provides whitening diagnostics, Rayleigh statistic analysis, and PSD goodness-of-fit testing.

    The Rayleigh statistic is the coefficient of variation of the PSD, used to measure
    'Gaussianity' of the data:
    - Value of 1: Gaussian behavior (expected for proper noise)
    - Value < 1: Coherent variations (spectral lines, narrow features)
    - Value > 1: Incoherent variation (non-Gaussian, non-stationary noise)

    When a reference PSD is provided, comparisons are made:
    - Panel 1: Shows both estimated and reference PSDs
    - Panel 2: Shows Rayleigh statistics for both PSDs
    - Panel 3: Shows Anderson-Darling p-values for whitening with both PSDs

    Parameters
    ----------
    ts_data : np.ndarray
        Real-valued time series of length N.
    fs : float
        Sampling frequency in Hz.
    freqs : np.ndarray
        Frequency axis corresponding to `psd` (already masked if needed).
    psd : np.ndarray
        Estimated PSD (power per Hz) on `freqs` (already masked if needed).
    reference_psd : Optional[np.ndarray]
        Optional "ground truth" or reference PSD (already masked if needed).
        If provided, residuals and MSE are computed: (psd − reference_psd).
        Also enables comparison plots in panels 2 and 3.
    fftlength : float
        Length of each segment in seconds for Rayleigh spectrum calculation.
    overlap : Optional[float]
        Overlap between segments in seconds for Rayleigh. If None, uses 50% overlap.
    """

    def __init__(
        self,
        ts_data: np.ndarray,
        fs: float,
        freqs: np.ndarray,
        psd: np.ndarray,
        reference_psd: Optional[np.ndarray] = None,
        fftlength: float = 2.0,
        overlap: Optional[float] = None,
    ) -> None:
        self.ts_data = np.asarray(ts_data, dtype=float)
        self.fs = float(fs)
        self.n = len(ts_data)
        self.duration = self.n / self.fs

        # Frequency axis and PSD (assumed to be already masked)
        self.freqs = np.asarray(freqs, dtype=float)
        self.psd = np.asarray(psd, dtype=float)

        if self.psd.shape[0] != self.freqs.shape[0]:
            raise ValueError(
                f"psd and freqs must have same length, got {self.psd.shape[0]} and {self.freqs.shape[0]}"
            )

        # If a reference PSD is provided, compute residuals and MSE
        if reference_psd is not None:
            ref = np.asarray(reference_psd, dtype=float)
            if ref.shape != self.psd.shape:
                raise ValueError(
                    f"reference_psd must have shape {self.psd.shape}, got {ref.shape}"
                )
            self.reference_psd = ref
            self.residuals = self.psd - ref
            self.mse = float(np.mean(self.residuals**2))
        else:
            self.reference_psd = None
            self.residuals = None
            self.mse = None

        # Calculate Rayleigh spectrum immediately
        self.rayleigh_spectrum, self.rayleigh_freqs = (
            self._calculate_rayleigh_spectrum(fftlength, overlap)
        )

        # Calculate reference Rayleigh spectrum if reference PSD provided
        if self.reference_psd is not None:
            self.rayleigh_spectrum_ref = (
                self._calculate_rayleigh_spectrum_from_psd(
                    self.reference_psd, fftlength, overlap
                )
            )
        else:
            self.rayleigh_spectrum_ref = None

        # Placeholders for whitening diagnostics (calculated when plotting)
        self.h_f: Optional[np.ndarray] = None
        self.wh_f: Optional[np.ndarray] = None
        self.wh_f_ref: Optional[np.ndarray] = None

    def _calculate_rayleigh_spectrum(
        self,
        fftlength: float = 2.0,
        overlap: Optional[float] = None,
        window: str = "hann",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the Rayleigh statistic spectrum from the time series."""
        nperseg = int(fftlength * self.fs)
        if overlap is None:
            noverlap = nperseg // 2
        else:
            noverlap = int(overlap * self.fs)

        # Calculate segments and collect PSDs
        step = nperseg - noverlap
        n_segments = (self.n - noverlap) // step

        if n_segments < 2:
            # Not enough segments, return array of ones (Gaussian-like) on provided freq grid
            return np.ones(len(self.freqs)), self.freqs

        # Collect PSD for each segment
        psds = []
        for i in range(n_segments):
            start_idx = i * step
            end_idx = start_idx + nperseg
            if end_idx > self.n:
                break

            segment = self.ts_data[start_idx:end_idx]
            freqs_seg, psd_seg = scipy.signal.welch(
                segment,
                fs=self.fs,
                window=window,
                nperseg=len(segment),
                noverlap=0,
                return_onesided=True,
            )
            psds.append(psd_seg)

        psds = np.array(psds)

        # Calculate coefficient of variation for each frequency bin
        mean_psd = np.mean(psds, axis=0)
        std_psd = np.std(psds, axis=0, ddof=1)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rayleigh_spectrum = std_psd / mean_psd
            rayleigh_spectrum[mean_psd == 0] = np.nan

        # Interpolate Rayleigh spectrum to match provided frequency grid
        rayleigh_interp = np.interp(self.freqs, freqs_seg, rayleigh_spectrum)

        return rayleigh_interp, self.freqs

    def _calculate_rayleigh_spectrum_from_psd(
        self,
        reference_psd: np.ndarray,
        fftlength: float = 2.0,
        overlap: Optional[float] = None,
        window: str = "hann",
    ) -> np.ndarray:
        """Calculate Rayleigh spectrum using a reference PSD for comparison."""
        nperseg = int(fftlength * self.fs)
        if overlap is None:
            noverlap = nperseg // 2
        else:
            noverlap = int(overlap * self.fs)

        step = nperseg - noverlap
        n_segments = (self.n - noverlap) // step

        if n_segments < 2:
            # Not enough segments, return array of ones on provided freq grid
            return np.ones(len(self.freqs))

        # Collect PSD for each segment
        psds = []
        for i in range(n_segments):
            start_idx = i * step
            end_idx = start_idx + nperseg
            if end_idx > self.n:
                break

            segment = self.ts_data[start_idx:end_idx]
            freqs_seg, psd_seg = scipy.signal.welch(
                segment,
                fs=self.fs,
                window=window,
                nperseg=len(segment),
                noverlap=0,
                return_onesided=True,
            )
            psds.append(psd_seg)

        psds = np.array(psds)

        # Interpolate reference PSD to match segment frequency grid
        ref_psd_interp = np.interp(freqs_seg, self.freqs, reference_psd)

        # Calculate coefficient of variation: std of measured PSDs / reference PSD
        std_psd = np.std(psds, axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            rayleigh_spectrum_ref = std_psd / ref_psd_interp
            rayleigh_spectrum_ref[ref_psd_interp == 0] = np.nan

        # Interpolate result back to provided frequency grid
        rayleigh_ref_interp = np.interp(
            self.freqs, freqs_seg, rayleigh_spectrum_ref
        )

        return rayleigh_ref_interp

    def plot_diagnostics(
        self,
        fname: str = "psd_diagnostics.png",
        bin_width_Hz: float = 8,
        figsize: Tuple[float, float] = (10, 8),
    ) -> None:
        """
        Create simplified 3-panel PSD diagnostics plot.

        Parameters
        ----------
        fname : str
            Output filename for the plot.
        bin_width_Hz : float
            Frequency bin width for Anderson-Darling tests.
        figsize : Tuple[float, float]
            Figure size (width, height).
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Panel 1: Periodogram + Estimated PSD

        # plot data periodogram
        # pdgrm = np.abs(np.fft.rfft(self.ts_data)) **2
        # ax1.semilogy(self.freqs, pdgrm, color="k", linewidth=2, alpha=0.3, label="Peridogram")
        ax1.semilogy(
            self.freqs, self.psd, color="C0", linewidth=1, label="Welch PSD"
        )
        if self.reference_psd is not None:
            ax1.semilogy(
                self.freqs,
                self.reference_psd,
                color="C1",
                linestyle="--",
                linewidth=2,
                label="Reference PSD",
            )
        ax1.set_ylabel("PSD [power/Hz]")
        ax1.set_title("Power Spectral Density")
        ax1.grid(True, which="both", ls=":", alpha=0.5)
        ax1.legend()

        # Panel 2: Rayleigh Statistic (estimated and reference if available)
        welch_pass_percent = self._rayleigh_pass_percent(
            self.rayleigh_spectrum
        )
        ax2.plot(
            self.freqs,
            self.rayleigh_spectrum,
            color="C0",
            linewidth=1,
            label=f"Welch PSD [{welch_pass_percent}%]",
        )
        if (
            self.reference_psd is not None
            and self.rayleigh_spectrum_ref is not None
        ):
            ref_pass_percent = self._rayleigh_pass_percent(
                self.rayleigh_spectrum_ref
            )
            ax2.plot(
                self.freqs,
                self.rayleigh_spectrum_ref,
                color="C1",
                linestyle="--",
                linewidth=1,
                label=f"Reference PSD [{ref_pass_percent}%]",
                alpha=0.4,
            )

        ax2.axhline(
            1.5,
            color="blue",
            linestyle=":",
            alpha=1,
            label="Non-Gaussian (R=1.5)",
            zorder=10,
        )
        ax2.axhline(
            1.0,
            color="black",
            linestyle=":",
            alpha=1,
            label="Gaussian (R=1)",
            zorder=10,
        )
        ax2.axhline(
            0.5,
            color="red",
            linestyle=":",
            alpha=1,
            label="Coherent (R=0.5)",
            zorder=10,
        )

        # Highlight coherent features for estimated PSD
        coherent_mask = self.rayleigh_spectrum < 0.5
        if np.any(coherent_mask):
            ax2.fill_between(
                self.freqs,
                0,
                self.rayleigh_spectrum,
                where=coherent_mask,
                color="C0",
                alpha=0.2,
            )

        ax2.set_ylabel("Rayleigh statistic")
        ax2.set_title("Rayleigh Spectrum (Coefficient of Variation)")
        ax2.set_ylim(0, 2.5)
        ax2.grid(True, which="both", ls=":", alpha=0.5)
        ax2.legend(loc="upper right")

        # Panel 3: Anderson-Darling p-values (estimated and reference if available)
        # Prepare whitening data
        self._prepare_whitening_data()

        # Get frequency range for whitening analysis
        fmin = self.freqs[0] if len(self.freqs) > 0 else 0
        fmax = self.freqs[-1] if len(self.freqs) > 0 else np.inf

        # Always compute p-values for estimated PSD
        if self.wh_f is not None and len(self.wh_f) > 0:
            # Create frequency array for whitened data - match to actual whitened data length
            freqs_for_whitening = np.linspace(fmin, fmax, len(self.wh_f))

            fbins, pvals = self._fbins_anderson_p_value(
                freqs_for_whitening, self.wh_f, bin_width_Hz, fmin, fmax
            )

            if len(fbins) > 0:
                welch_pass = self._ad_pass_percent(pvals)
                ax3.plot(
                    fbins,
                    pvals,
                    alpha=0.7,
                    color="C0",
                    label=f"Welch PSD [{welch_pass}%]",
                )

        # Compute p-values for reference PSD if available
        if (
            self.reference_psd is not None
            and self.wh_f_ref is not None
            and len(self.wh_f_ref) > 0
        ):

            freqs_for_whitening_ref = np.linspace(
                fmin, fmax, len(self.wh_f_ref)
            )
            fbins_ref, pvals_ref = self._fbins_anderson_p_value(
                freqs_for_whitening_ref,
                self.wh_f_ref,
                bin_width_Hz,
                fmin,
                fmax,
            )
            if len(fbins_ref) > 0:
                ref_pass = self._ad_pass_percent(pvals_ref)
                ax3.plot(
                    fbins_ref,
                    pvals_ref,
                    alpha=0.7,
                    color="C1",
                    label=f"Reference PSD [{ref_pass}%]",
                )

        ax3.axhline(
            1e-2,
            color="red",
            linestyle=":",
            alpha=0.7,
            label="p=0.01 threshold",
            zorder=10,
        )
        ax3.set_xlabel("Frequency [Hz]")
        ax3.set_ylabel("p-value")
        ax3.set_yscale("log")
        ax3.set_title("Anderson-Darling p-values (Gaussianity Test)")
        ax3.grid(True, which="both", ls=":", alpha=0.5)
        ax3.legend(loc="upper right")

        # Set x-axis limits to match provided frequency range
        if len(self.freqs) > 0:
            ax3.set_xlim(self.freqs[0], self.freqs[-1])

        plt.tight_layout()
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        print(f"Diagnostics plot saved to {fname}")

    def _prepare_whitening_data(self) -> None:
        """Prepare data for whitening diagnostics."""
        n = self.n
        psd_alpha = 2 * 0.1 / self.duration  # Tukey alpha = 0.1
        window = scipy.signal.get_window(("tukey", psd_alpha), n)

        ts_win = self.ts_data * window
        H_full = np.fft.fft(ts_win)

        # Keep only positive-frequency bins, excluding DC and Nyquist
        # This should match the length of the provided frequency grid
        self.h_f = H_full[1 : n // 2]  # Skip DC, exclude Nyquist

        # We need to interpolate the provided PSD to match the FFT frequency grid
        fft_freqs = np.fft.fftfreq(n, 1 / self.fs)[
            1 : n // 2
        ]  # Positive freqs, skip DC

        # Interpolate provided PSD to FFT frequency grid
        asd_interp = np.sqrt(np.interp(fft_freqs, self.freqs, self.psd))

        # Whitened spectrum using interpolated PSD
        self.wh_f = self.h_f * np.sqrt(4.0 / self.duration) / asd_interp

        # Always compute reference whitened data if reference PSD is provided
        if self.reference_psd is not None:
            asd_ref_interp = np.sqrt(
                np.interp(fft_freqs, self.freqs, self.reference_psd)
            )
            # Avoid division by zero
            asd_ref_interp = np.where(
                asd_ref_interp == 0, np.finfo(float).eps, asd_ref_interp
            )
            self.wh_f_ref = (
                self.h_f * np.sqrt(4.0 / self.duration) / asd_ref_interp
            )
        else:
            self.wh_f_ref = None

    def _fbins_anderson_p_value(
        self,
        freqs: np.ndarray,
        data: np.ndarray,
        bin_width_Hz: float,
        fmin: float = 0,
        fmax: float = np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-bin Anderson-Darling p-values."""
        # Ensure we only use frequencies within the provided range
        valid_freq_mask = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
        freqs_clean = freqs[valid_freq_mask]
        data_clean = data[valid_freq_mask]

        if len(freqs_clean) == 0:
            return np.array([]), np.array([])

        duration = self.duration
        n = len(data_clean)
        bin_width = int(bin_width_Hz * duration)
        if bin_width < 1:
            bin_width = 1

        idxs = np.arange(0, n, bin_width)
        if len(idxs) > 1 and idxs[-1] >= n:
            idxs = idxs[:-1]

        pvals = []
        fbins = []
        for ii in idxs:
            end_idx = min(ii + bin_width, n)
            block = data_clean[ii:end_idx]
            freq_block = freqs_clean[ii:end_idx]

            if len(block) > 5 and len(freq_block) > 0:
                # Use center frequency of the block
                center_freq = freq_block[len(freq_block) // 2]
                if fmin <= center_freq <= fmax:
                    pvals.append(
                        anderson_p_value(block, freq_block, fmin, fmax)
                    )
                    fbins.append(center_freq)

        return np.array(fbins), np.array(pvals)

    def _rayleigh_pass_percent(self, rayleigh_stat):
        return int(
            np.sum((rayleigh_stat >= 0.5) & (rayleigh_stat <= 1.5))
            / len(rayleigh_stat)
            * 100
        )

    def _ad_pass_percent(self, p_values, threshold=0.01):
        return int(
            np.sum(p_values > threshold) / len(p_values) * 100
            if len(p_values) > 0
            else 0
        )

    def summary_stats(self) -> dict:
        """Get summary statistics for the diagnostics."""
        stats = {
            "duration_s": self.duration,
            "sample_rate_hz": self.fs,
            "n_samples": self.n,
            "freq_resolution_hz": self.fs / self.n,
            "analysis_freq_range_hz": (
                (self.freqs[0], self.freqs[-1])
                if len(self.freqs) > 0
                else (0, 0)
            ),
            "n_freq_bins": len(self.freqs),
        }

        if self.reference_psd is not None:
            stats["mse"] = self.mse
            stats["mae"] = np.mean(np.abs(self.residuals))

        # Rayleigh statistics for estimated PSD
        valid_rayleigh = self.rayleigh_spectrum[
            ~np.isnan(self.rayleigh_spectrum)
        ]
        if len(valid_rayleigh) > 0:
            stats["rayleigh_mean"] = np.mean(valid_rayleigh)
            stats["rayleigh_median"] = np.median(valid_rayleigh)
            stats["fraction_coherent"] = np.sum(valid_rayleigh < 0.5) / len(
                valid_rayleigh
            )
            stats["fraction_gaussian"] = np.sum(
                np.abs(valid_rayleigh - 1.0) < 0.1
            ) / len(valid_rayleigh)

        # Rayleigh statistics for reference PSD (if available)
        if self.rayleigh_spectrum_ref is not None:
            valid_rayleigh_ref = self.rayleigh_spectrum_ref[
                ~np.isnan(self.rayleigh_spectrum_ref)
            ]
            if len(valid_rayleigh_ref) > 0:
                stats["rayleigh_mean_ref"] = np.mean(valid_rayleigh_ref)
                stats["rayleigh_median_ref"] = np.median(valid_rayleigh_ref)
                stats["fraction_coherent_ref"] = np.sum(
                    valid_rayleigh_ref < 0.5
                ) / len(valid_rayleigh_ref)
                stats["fraction_gaussian_ref"] = np.sum(
                    np.abs(valid_rayleigh_ref - 1.0) < 0.1
                ) / len(valid_rayleigh_ref)

        return stats


# Anderson-Darling test functions


def anderson_darling_statistic(data: np.ndarray) -> float:
    """Calculate Anderson-Darling test statistic for normality."""
    n = len(data)
    if n < 2:
        return np.nan

    sorted_data = np.sort(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std == 0:
        return np.nan

    standardized = (sorted_data - mean) / std

    try:
        normal_cdf = 0.5 * (1 + scipy.special.erf(standardized / np.sqrt(2)))
        normal_cdf = np.clip(normal_cdf, 1e-15, 1 - 1e-15)

        s = np.sum(
            (2 * np.arange(1, n + 1) - 1)
            * (np.log(normal_cdf) + np.log(1 - normal_cdf[::-1]))
        )
        return -n - s / n
    except (ValueError, RuntimeWarning):
        return np.nan


def anderson_p_value(
    data: np.ndarray,
    freqs: Optional[np.ndarray] = None,
    fmin: float = 0,
    fmax: float = np.inf,
) -> float:
    """Calculate Anderson-Darling p-value for normality test on complex data."""
    if freqs is not None:
        idxs = (freqs >= fmin) & (freqs <= fmax)
        data = data[idxs]

    if len(data) == 0:
        return np.nan

    # Combine real and imaginary parts
    flat = np.concatenate([data.real, data.imag])
    flat = flat[np.isfinite(flat)]

    if len(flat) < 5:
        return np.nan

    A2 = anderson_darling_statistic(flat)

    if np.isnan(A2):
        return np.nan

    # Critical values and significance levels
    critical_values = np.array(
        [
            0.200,
            0.300,
            0.400,
            0.500,
            0.576,
            0.656,
            0.787,
            0.918,
            1.092,
            1.250,
            1.500,
            1.750,
            2.000,
            2.500,
            3.000,
            3.500,
            4.000,
            4.500,
            5.000,
            6.000,
            7.000,
            8.000,
            10.000,
        ]
    )
    significance_levels = np.array(
        [
            0.90,
            0.85,
            0.80,
            0.75,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.25,
            0.20,
            0.15,
            0.10,
            0.05,
            0.01,
            0.005,
            0.0025,
            0.001,
            0.0005,
            0.0002,
            0.0001,
            0.00005,
            0.00001,
        ]
    )

    if A2 < critical_values[0]:
        return significance_levels[0]
    elif A2 > critical_values[-1]:
        return significance_levels[-1]
    else:
        return float(np.interp(A2, critical_values, significance_levels))
