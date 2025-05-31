from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


class PSDDiagnostics:
    """
    Perform diagnostics on an estimated PSD and (optionally) compare it to a reference PSD.
    Also provides whitening diagnostics on a time series using the estimated PSD.

    Parameters
    ----------
    ts_data : np.ndarray
        Real-valued time series of length N.
    fs : float
        Sampling frequency in Hz.
    freqs : np.ndarray
        One-sided frequency axis (length N//2 + 1) corresponding to `psd`.
    psd : np.ndarray
        Estimated one-sided PSD (power per Hz) on `freqs`.
        Length must be N//2 + 1.
    reference_psd : Optional[np.ndarray]
        Optional “ground truth” or reference PSD (one-sided, same length as `psd`).
        If provided, residuals and MSE are computed: (psd − reference_psd).
    """

    def __init__(
        self,
        ts_data: np.ndarray,
        fs: float,
        freqs: np.ndarray,
        psd: np.ndarray,
        reference_psd: Optional[np.ndarray] = None,
    ) -> None:
        self.ts_data = ts_data
        self.fs = float(fs)
        self.n = len(ts_data)
        self.duration = self.n / self.fs

        # One-sided frequency axis (length N//2 + 1)
        self.freqs = freqs
        if self.freqs.shape[0] != self.n // 2 + 1:
            raise ValueError(f"freqs must have length {self.n//2 + 1}")

        # Estimated PSD (one-sided, length N//2 + 1)
        self.psd = psd
        if self.psd.shape[0] != self.n // 2 + 1:
            raise ValueError(f"psd must have length {self.n//2 + 1}")

        # If a reference PSD is provided, compute residuals and MSE
        if reference_psd is not None:
            ref = np.asarray(reference_psd, dtype=float)
            if ref.shape != self.psd.shape:
                raise ValueError(
                    f"reference_psd must have shape {self.psd.shape}"
                )
            self.reference_psd = ref
            self.residuals = self.psd - ref
            self.mse = float(np.mean(self.residuals**2))
        else:
            self.reference_psd = None
            self.residuals = None
            self.mse = None

        # Placeholders for whitening diagnostics
        self.h_f: Optional[np.ndarray] = None
        self.wh_f: Optional[np.ndarray] = None

    def plot_psd_comparison(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot the estimated PSD vs. reference PSD (if provided).
        If reference PSD is provided, also plot residuals on a lower subplot and annotate MSE.

        Parameters
        ----------
        ax : Optional[plt.Axes]
            If given, the top plot (PSD vs reference) is drawn on this Axes.
            Otherwise, a new figure with two subplots is created.

        Returns
        -------
        ax_top : plt.Axes
            The Axes object containing the top plot (PSD vs reference).
        """
        # If no reference PSD, just plot the estimated PSD
        if self.reference_psd is None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(
                self.freqs,
                self.psd,
                color="C0",
                linewidth=1,
                label="Estimated PSD",
            )
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD [power/Hz]")
            ax.set_title("Estimated PSD")
            ax.grid(True, which="both", ls=":", alpha=0.5)
            ax.legend()
            return ax

        # Reference PSD provided → create two stacked subplots if ax is None
        if ax is None:
            fig, (ax_top, ax_bot) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
        else:
            ax_top = ax
            fig = ax.figure
            ax_bot = fig.add_subplot(
                ax.get_gridspec()[1]
            )  # unused if ax passed

        # Top: estimated PSD vs reference PSD
        ax_top.semilogy(
            self.freqs,
            self.psd,
            color="C0",
            linewidth=1,
            label="Estimated PSD",
        )
        ax_top.semilogy(
            self.freqs,
            self.reference_psd,
            color="C1",
            linestyle="--",
            linewidth=2,
            label="Reference PSD",
        )
        ax_top.set_ylabel("PSD [power/Hz]")
        ax_top.set_title("PSD Comparison")
        ax_top.grid(True, which="both", ls=":", alpha=0.5)
        ax_top.legend()

        # Bottom: residuals = estimated − reference
        ax_bot.plot(self.freqs, self.residuals, color="k", linewidth=1)
        ax_bot.set_xlabel("Frequency [Hz]")
        ax_bot.set_ylabel("Residuals")
        ax_bot.set_title(f"Residuals (MSE = {self.mse:.3e})")
        ax_bot.grid(True, which="both", ls=":", alpha=0.5)

        return ax_top

    def plot_whiten_diagnostics(
        self,
        fname: str = "whiten_diagnostics.png",
        bin_width_Hz: float = 8,
        label: Optional[str] = None,
    ) -> None:
        """
        Run four-panel whitening diagnostics using the estimated PSD.

        Uses a Tukey window on the time series, FFTs to get H(f), then whitens by dividing
        by ASD = sqrt(psd) on the positive-frequency bins (length N//2).

        Panels:
          - (top-left) |H(f)| and ASD
          - (bottom-left) per-bin Anderson-Darling p-values (log scale)
          - (top-right) histogram of those p-values
          - (bottom-right) histogram of whitened real/imag vs standard normal

        After calling, self.h_f and self.wh_f hold the positive-frequency FFT and whitened FFT.

        Parameters
        ----------
        bin_width_Hz : float
            Frequency-bin width (in Hz) for Anderson-Darling tests.
        label : Optional[str]
            Label for plotting the ASD curve (default: "ASD").
        """
        # 1) Window + FFT
        n = self.n
        psd_alpha = 2 * 0.1 / self.duration  # Tukey alpha = 0.1
        window = scipy.signal.get_window(("tukey", psd_alpha), n)

        ts_win = self.ts_data * window
        H_full = np.fft.fft(ts_win)

        # Keep only positive-frequency bins (length N//2)
        self.h_f = H_full[: n // 2]

        # Build ASD from estimated PSD (drop Nyquist bin to match length n//2)
        asd = np.sqrt(self.psd[:-1])

        # Whitened spectrum
        self.wh_f = self.h_f * np.sqrt(4.0 / self.duration) / asd

        # 2) Create 2×2 figure for diagnostics
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]

        freqs_pos = self.freqs[:-1]  # drop Nyquist, so length matches h_f/asd

        # Top-left: |H(f)| and ASD
        ax1.semilogy(freqs_pos, np.abs(self.h_f), label=r"|H(f)|")
        ax1.semilogy(freqs_pos, asd, label=(label or "ASD"))
        ax1.set_ylabel("Amplitude [Hz⁻¹]")
        ax1.set_title("Spectrum & ASD")
        ax1.legend()
        ax1.grid(True, which="both", ls=":", alpha=0.5)

        # Bottom-left: per-bin Anderson-Darling p-values
        fbins, pvals = self._fbins_anderson_p_value(
            freqs_pos, self.wh_f, bin_width_Hz
        )
        ax3.scatter(fbins, pvals, s=2)
        ax3.axhline(1e-2, color="r", linestyle="--")
        ax3.set(xlabel="Frequency [Hz]", ylabel="p-value", yscale="log")
        ax3.set_title("Per-bin A-D p-values")
        ax3.grid(True, which="both", ls=":", alpha=0.5)

        # Top-right: histogram of those p-values
        bin_w = 0.025
        bins = np.arange(0, 1 + bin_w, bin_w)
        ax2.hist(pvals, bins=bins, density=True)
        ax2.set(xlabel="p-values", ylabel="Density", xlim=(0, 1))
        ax2.set_title("Histogram of p-values")
        ax2.grid(True, which="both", ls=":", alpha=0.5)

        # Bottom-right: whitened real & imag vs standard normal
        whf = self.wh_f
        kwargs = dict(bins="auto", alpha=0.5, density=True)
        ax4.hist(whf.real, label="real", **kwargs)
        ax4.hist(whf.imag, label="imag", **kwargs)
        xs = np.linspace(-5, 5, 1000)
        ax4.plot(
            xs,
            np.exp(-(xs**2) / 2) / np.sqrt(2 * np.pi),
            label="Std normal PDF",
        )
        ax4.set(xlabel="Whitened strain")
        ax4.set_title(f"P-val: {anderson_p_value(whf):.2f}")
        ax4.legend()
        ax4.grid(True, which="both", ls=":", alpha=0.5)

        fig.tight_layout()
        plt.savefig(fname, bbox_inches="tight", dpi=300)

    def _anderson_p_value(
        self,
        data: np.ndarray,
        freqs: Optional[np.ndarray] = None,
        fmin: float = 0,
        fmax: float = np.inf,
    ) -> float:
        """
        Compute the Anderson-Darling p-value for normality on (complex) data,
        optionally restricted to freqs in (fmin, fmax).
        """
        return anderson_p_value(data, freqs, fmin, fmax)

    def _fbins_anderson_p_value(
        self,
        freqs: np.ndarray,
        data: np.ndarray,
        bin_width_Hz: float,
        fmin: float = 0,
        fmax: float = np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (frequency-bin-center, p-value) for the whitened spectrum:
        slice into bins of width `bin_width_Hz`, run A-D test on each bin.
        """
        duration = self.duration
        n = len(data)
        bin_width = int(bin_width_Hz * duration)
        idxs = np.arange(0, n, bin_width)[:-1]
        pvals = []
        fbins = []
        for ii in idxs:
            block = data[ii : ii + bin_width]
            freq_block = freqs[ii : ii + bin_width]
            pvals.append(self._anderson_p_value(block, freq_block, fmin, fmax))
            fbins.append(freq_block[bin_width // 2])
        return np.array(fbins), np.array(pvals)


# Standalone Anderson-Darling functions (used internally)


def empirical_cdf(
    data: np.ndarray,
) -> Tuple[Callable[[float], float], np.ndarray]:
    sorted_data = np.sort(data)
    n = len(data)

    def ecdf(x: float) -> float:
        return np.searchsorted(sorted_data, x, side="right") / n

    return ecdf, sorted_data


def anderson_darling_statistic(data: np.ndarray) -> float:
    n = len(data)
    _, sorted_data = empirical_cdf(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    standardized = (sorted_data - mean) / std
    normal_cdf = 0.5 * (1 + scipy.special.erf(standardized / np.sqrt(2)))
    s = np.sum(
        (2 * np.arange(1, n + 1) - 1)
        * (np.log(normal_cdf) + np.log(1 - normal_cdf[::-1]))
    )
    return -n - s / n


def anderson_p_value(
    data: np.ndarray,
    freqs: Optional[np.ndarray] = None,
    fmin: float = 0,
    fmax: float = np.inf,
) -> float:
    if freqs is not None:
        idxs = (freqs > fmin) & (freqs < fmax)
        data = data[idxs]
    flat = np.concatenate([data.real, data.imag])
    if len(flat) == 0:
        return np.nan
    A2 = anderson_darling_statistic(flat)
    critical_values = [
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
    significance_levels = [
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
    if A2 < critical_values[0]:
        return significance_levels[0]
    elif A2 > critical_values[-1]:
        return significance_levels[-1]
    else:
        return float(np.interp(A2, critical_values, significance_levels))
