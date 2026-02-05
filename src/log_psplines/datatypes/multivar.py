from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.signal import csd, welch, windows

from ..logger import logger
from ..spectrum_utils import u_to_wishart_matrix, wishart_matrix_to_psd


def _interp_complex_matrix(
    freq_src: np.ndarray, freq_tgt: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a complex-valued matrix along the frequency axis."""
    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    flat = matrix.reshape(matrix.shape[0], -1)

    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_src, flat[:, idx].real)
            for idx in range(flat.shape[1])
        ]
    ).T

    if np.iscomplexobj(matrix):
        imag_interp = np.vstack(
            [
                np.interp(freq_tgt, freq_src, flat[:, idx].imag)
                for idx in range(flat.shape[1])
            ]
        ).T
        res = real_interp + 1j * imag_interp
    else:
        res = real_interp

    return res.reshape((freq_tgt.size,) + matrix.shape[1:])


@dataclass
class MultivarFFT:
    """
    Discrete FFTs for multivariate time series.
    Stores real/imaginary parts of the FFT together with Wishart replicates.

    Attributes:
        y_re: Real part of FFT (N, p)
        y_im: Imag part of FFT (N, p)
        u_re: Real part of eigenvector-weighted periodogram replicates
              (N, p, p)
        u_im: Imag part of eigenvector-weighted periodogram replicates
              (N, p, p)
        Nb: Degrees of freedom (number of averaged blocks)
        freq: Frequency grid (N,)
        N: Number of frequencies
        p: Number of channels
    """

    y_re: np.ndarray
    y_im: np.ndarray
    u_re: np.ndarray
    u_im: np.ndarray
    freq: np.ndarray
    N: int
    p: int
    Nb: int = 1
    scaling_factor: Optional[float] = 1.0  # Track the PSD scaling factor
    channel_stds: Optional[np.ndarray] = (
        None  # Per-channel standard deviations
    )
    fs: float = field(default=1.0, repr=False)
    # Duration (seconds) of each time-domain block used to form the FFT/Wishart
    # statistic. This matches the "T" factor in the Whittle likelihood.
    duration: float = field(default=1.0, repr=False)
    raw_psd: Optional[np.ndarray] = None
    raw_freq: Optional[np.ndarray] = None
    # For coarse-grained FFTs, stores the number of fine-grid frequencies that
    # contributed to each coarse bin. This lets likelihood code compute
    # per-bin averages independent of any subsequent weight normalization.
    freq_bin_counts: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.y_re = np.asarray(self.y_re, dtype=np.float64)
        self.y_im = np.asarray(self.y_im, dtype=np.float64)
        self.u_re = np.asarray(self.u_re, dtype=np.float64)
        self.u_im = np.asarray(self.u_im, dtype=np.float64)
        self.freq = np.asarray(self.freq, dtype=np.float64)
        self.duration = float(self.duration)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be a positive finite float")

        expected_fft_shape = (self.N, self.p)
        if self.y_re.shape != expected_fft_shape:
            raise ValueError(
                f"y_re must have shape {expected_fft_shape}, got {self.y_re.shape}"
            )
        if self.y_im.shape != expected_fft_shape:
            raise ValueError(
                f"y_im must have shape {expected_fft_shape}, got {self.y_im.shape}"
            )

        expected_u_shape = (self.N, self.p, self.p)
        if self.u_re.shape != expected_u_shape:
            raise ValueError(
                f"u_re must have shape {expected_u_shape}, got {self.u_re.shape}"
            )
        if self.u_im.shape != expected_u_shape:
            raise ValueError(
                f"u_im must have shape {expected_u_shape}, got {self.u_im.shape}"
            )

        if self.freq.shape != (self.N,):
            raise ValueError(
                f"freq must have length {self.N}, got {self.freq.shape}"
            )

        if self.raw_psd is not None:
            self.raw_psd = np.asarray(self.raw_psd, dtype=np.complex128)
            if self.raw_psd.shape != expected_u_shape:
                raise ValueError(
                    f"raw_psd must have shape {expected_u_shape}, got {self.raw_psd.shape}"
                )
        if self.raw_freq is not None:
            self.raw_freq = np.asarray(self.raw_freq, dtype=np.float64)
            if self.raw_freq.shape != (self.N,):
                raise ValueError(
                    f"raw_freq must have length {self.N}, got {self.raw_freq.shape}"
                )

        if self.freq_bin_counts is not None:
            self.freq_bin_counts = np.asarray(
                self.freq_bin_counts, dtype=np.float64
            )
            if self.freq_bin_counts.shape != (self.N,):
                raise ValueError(
                    f"freq_bin_counts must have length {self.N}, got {self.freq_bin_counts.shape}"
                )
            if np.any(self.freq_bin_counts <= 0):
                raise ValueError("freq_bin_counts must be positive")

        if self.channel_stds is not None:
            self.channel_stds = np.asarray(self.channel_stds, dtype=np.float64)
            if self.channel_stds.shape != (self.p,):
                raise ValueError(
                    "channel_stds must have length equal to number of channels"
                )

    @classmethod
    def compute_fft(
        cls,
        x: np.ndarray,
        fs: float = 1.0,
        fmin: float = None,
        fmax: float = None,
        scaling_factor: Optional[float] = 1.0,
        channel_stds: Optional[np.ndarray] = None,
        window: Optional[str | tuple] = "hann",
    ) -> "MultivarFFT":
        """Compute FFT and Wishart replicates with a single (full-length) block."""
        return cls.compute_wishart(
            x,
            fs=fs,
            Nb=1,
            fmin=fmin,
            fmax=fmax,
            scaling_factor=scaling_factor,
            channel_stds=channel_stds,
            window=window,
        )

    @classmethod
    def compute_wishart(
        cls,
        x: np.ndarray,
        fs: float,
        Nb: int,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        scaling_factor: Optional[float] = 1.0,
        channel_stds: Optional[np.ndarray] = None,
        window: Optional[str | tuple] = "hann",
    ) -> "MultivarFFT":
        """
        Compute block-averaged (Wishart) FFT statistics for multivariate series.

        Parameters
        ----------
        x : np.ndarray
            Input time series (n, p)
        fs : float
            Sampling frequency
        Nb : int
            Number of non-overlapping blocks to average. Must divide n.
        fmin, fmax : float, optional
            Optional frequency truncation applied after blocking.
        scaling_factor : float, optional
            PSD scaling factor carried through sampling.
        window : str or tuple, optional
            Taper applied to each block before the FFT. Defaults to Hann.
            Set to ``None`` to recover the previous rectangular-window behavior.
        """
        if Nb < 1:
            raise ValueError("Nb must be positive.")

        n, p = x.shape
        if n % Nb != 0:
            raise ValueError(f"n={n} must be divisible by Nb={Nb}.")

        Lb = n // Nb
        if Lb <= p:
            raise ValueError(
                "Block length must exceed number of channels for FFT stability."
            )

        blocks = x.reshape(Nb, Lb, p)
        # Detrend each block (Welch-style constant detrend) to reduce low-frequency
        # leakage into the first few positive bins when using tapered windows.
        blocks = blocks - np.mean(blocks, axis=1, keepdims=True)

        if window is None:
            taper = np.ones(Lb, dtype=np.float64)
        else:
            taper = windows.get_window(window, Lb, fftbins=True)
            taper = np.asarray(taper, dtype=np.float64)
        taper_energy = float(np.sum(taper**2))
        if taper_energy <= 0.0:
            raise ValueError("Window energy must be positive.")
        blocks = blocks * taper[None, :, None]

        block_ffts = np.fft.rfft(blocks, axis=1)
        freq = np.fft.rfftfreq(Lb, 1 / fs)
        # Drop the zero-frequency bin for numerical stability
        block_ffts = block_ffts[:, 1:, :]
        freq = freq[1:]
        if freq.size == 0:
            raise ValueError(
                "Block length too small to retain positive frequencies."
            )

        scale = np.full(
            freq.shape, 2.0 / (taper_energy * fs), dtype=np.float64
        )
        if Lb % 2 == 0 and scale.size > 0:
            scale[-1] = 1.0 / (taper_energy * fs)
        sqrt_scale = np.sqrt(scale, dtype=np.float64)[None, :, None]
        block_ffts = block_ffts * sqrt_scale
        # Convert from a "per-block-periodogram" normalisation to the Whittle
        # convention that keeps an explicit 1/T in the likelihood. This ensures
        # the likelihood uses the observation duration explicitly while PSD
        # conversions remain unchanged (see wishart_matrix_to_psd(duration=...)).
        duration = float(Lb) / float(fs)
        sqrt_duration = float(np.sqrt(np.asarray(duration, dtype=np.float64)))
        block_ffts = block_ffts * sqrt_duration

        if fmin is not None or fmax is not None:
            freq_min = float(freq[0])
            freq_max = float(freq[-1])
            fmin_eff = freq_min if fmin is None else float(fmin)
            fmax_eff = freq_max if fmax is None else float(fmax)

            fmin_eff = min(max(fmin_eff, freq_min), freq_max)
            fmax_eff = min(max(fmax_eff, freq_min), freq_max)
            if fmax_eff < fmin_eff:
                fmax_eff = fmin_eff

            freq_mask = (freq >= fmin_eff) & (freq <= fmax_eff)
            if not np.any(freq_mask):
                raise ValueError(
                    "Frequency truncation removed all bins; check fmin/fmax."
                )
            freq = freq[freq_mask]
            block_ffts = block_ffts[:, freq_mask, :]

        mean_fft = np.mean(block_ffts, axis=0)
        y_re = mean_fft.real
        y_im = mean_fft.imag

        Y = np.einsum("bnc,bnd->ncd", block_ffts, np.conj(block_ffts))
        eigvals, eigvecs = np.linalg.eigh(Y)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        sqrt_eigvals = np.sqrt(eigvals)[:, None, :]
        U = eigvecs * sqrt_eigvals
        u_re = U.real
        u_im = U.imag
        raw_psd = wishart_matrix_to_psd(
            u_to_wishart_matrix(U),
            Nb=Nb,
            duration=duration,
            scaling_factor=float(scaling_factor or 1.0),
        )

        return cls(
            y_re=y_re,
            y_im=y_im,
            freq=freq,
            N=len(freq),
            p=p,
            u_re=u_re,
            u_im=u_im,
            raw_psd=raw_psd,
            raw_freq=freq,
            Nb=Nb,
            scaling_factor=scaling_factor,
            fs=fs,
            duration=duration,
            channel_stds=(
                None
                if channel_stds is None
                else np.asarray(channel_stds, dtype=np.float64)
            ),
        )

    def cut(self, fmin: float, fmax: float) -> "MultivarFFT":
        """Return a new MultivarFFT within frequency range [fmin, fmax]."""
        if fmax < fmin:
            raise ValueError(
                f"Invalid frequency bounds supplied: fmin={fmin}, fmax={fmax}."
            )
        mask = (self.freq >= fmin) & (self.freq <= fmax)
        raw_psd = None
        raw_freq = None
        if self.raw_psd is not None:
            raw_psd = self.raw_psd[mask]
            raw_freq = self.freq[mask]
        freq_bin_counts = None
        if self.freq_bin_counts is not None:
            freq_bin_counts = np.asarray(self.freq_bin_counts)[mask]
        return MultivarFFT(
            y_re=self.y_re[mask],
            y_im=self.y_im[mask],
            freq=self.freq[mask],
            N=int(np.sum(mask)),
            p=self.p,
            u_re=self.u_re[mask],
            u_im=self.u_im[mask],
            raw_psd=raw_psd,
            raw_freq=raw_freq,
            freq_bin_counts=freq_bin_counts,
            Nb=self.Nb,
            scaling_factor=self.scaling_factor,
            fs=self.fs,
            duration=self.duration,
            channel_stds=self.channel_stds,
        )

    @property
    def amplitude_range(self) -> Tuple[float, float]:
        amps = (np.min(self.y_re), np.max(self.y_re))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))

    def __repr__(self):
        return f"MultivarFFT(N={self.N}, p={self.p}, amplitudes={self.amplitude_range})"

    @property
    def empirical_psd(self) -> "EmpiricalPSD":
        psd = self.get_empirical_psd(
            self.freq,
            self.y_re,
            self.y_im,
            self.duration,
            self.scaling_factor,
        )
        if self.channel_stds is not None:
            scale_matrix = np.outer(self.channel_stds, self.channel_stds)
            sf = float(self.scaling_factor or 1.0)
            psd = EmpiricalPSD(
                freq=psd.freq,
                psd=psd.psd * (scale_matrix / sf),
                coherence=psd.coherence,
                channels=psd.channels,
            )
        return psd

    @staticmethod
    def get_empirical_psd(
        freq: np.ndarray,
        y_re: np.ndarray,
        y_im: np.ndarray,
        duration: float,
        scaling: float = 1.0,
    ) -> "EmpiricalPSD":
        y_re = np.array(y_re, dtype=np.float64)
        y_im = np.array(y_im, dtype=np.float64)
        freq = np.asarray(freq, dtype=np.float64)
        N, p = y_re.shape
        if freq.shape != (N,):
            raise ValueError(f"freq must have shape ({N},), got {freq.shape}")
        y_complex = y_re + 1j * y_im
        Y = np.einsum("fi,fj->fij", y_complex, np.conj(y_complex))
        psd = wishart_matrix_to_psd(
            Y,
            Nb=1,
            duration=float(duration),
            scaling_factor=float(scaling),
        )
        coherence = _get_coherence(psd)
        return EmpiricalPSD(
            freq=freq,
            psd=psd,
            coherence=coherence,
        )


@dataclass
class MultivariateTimeseries:
    y: np.ndarray  # numpy array (n, p) for numerical stability
    t: np.ndarray = None  # numpy array
    std: np.ndarray = None  # numpy array, per-channel std
    scaling_factor: Optional[float] = (
        1.0  # numpy array for per-channel scaling
    )
    original_stds: Optional[np.ndarray] = (
        None  # numpy array for original per-channel stds
    )

    def __post_init__(self):
        if self.t is None:
            self.t = np.arange(self.y.shape[0])
        if self.std is None:
            self.std = np.std(self.y, axis=0)
        if self.y.shape[0] != self.t.shape[0]:
            raise ValueError("y and t must have the same length")
        if np.isnan(self.y).any() or np.isnan(self.t).any():
            raise ValueError("y or t contains NaN values.")

    @property
    def p(self):
        return self.y.shape[1] if self.y.ndim > 1 else 1

    @property
    def fs(self) -> float:
        return float(1 / (self.t[1] - self.t[0]))

    def standardise(self):
        self.std = np.std(self.y, axis=0)
        y = (self.y - np.mean(self.y, axis=0)) / self.std
        return MultivariateTimeseries(y, self.t, self.std)

    def standardise_for_psd(self):
        if self.original_stds is None:
            self.original_stds = np.std(self.y, axis=0)
        y_standardized = (
            self.y - np.mean(self.y, axis=0)
        ) / self.original_stds
        psd_scaling_factor = np.std(self.y) ** 2.0
        return MultivariateTimeseries(
            y=y_standardized,
            t=self.t,
            std=np.ones_like(self.original_stds),
            scaling_factor=psd_scaling_factor,
            original_stds=self.original_stds,
        )

    def to_cross_spectral_density(
        self, fmin: float = None, fmax: float = None
    ) -> "MultivarFFT":
        return MultivarFFT.compute_fft(
            self.y,
            fs=self.fs,
            fmin=fmin,
            fmax=fmax,
            scaling_factor=self.scaling_factor,
            channel_stds=self.original_stds,
        )

    def to_wishart_stats(
        self,
        Nb: int,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        window: Optional[str | tuple] = "hann",
    ) -> "MultivarFFT":
        n = self.y.shape[0]
        if Nb <= 0:
            raise ValueError("Nb must be positive.")
        if n % Nb != 0:
            raise ValueError(f"n={n} must be divisible by Nb={Nb}.")
        Lb = n // Nb

        wishart_fft = MultivarFFT.compute_wishart(
            self.y,
            fs=self.fs,
            Nb=Nb,
            fmin=fmin,
            fmax=fmax,
            scaling_factor=self.scaling_factor,
            channel_stds=self.original_stds,
            window=window,
        )
        log_msg = (
            f"Wishart averaging (blocks={Nb}): "
            f"n={n} -> Lb={Lb}, "
            f"N={wishart_fft.N}, p={wishart_fft.p}"
        )
        logger.info(log_msg)
        return wishart_fft

    @property
    def amplitude_range(self) -> Tuple[float, float]:
        amps = (np.min(self.y), np.max(self.y))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))

    def __repr__(self):
        return f"MultivariateTimeseries(n={self.y.shape[0]}, p={self.p}, fs={self.fs:.3f}, amplitudes={self.amplitude_range})"

    def get_empirical_psd(self, **kwargs) -> "EmpiricalPSD":
        return EmpiricalPSD.from_timeseries_data(
            self.y,
            fs=self.fs,
            **kwargs,
        )


@dataclass
class EmpiricalPSD:
    freq: np.ndarray  # (N,)
    psd: np.ndarray  # (N, p, p) complex CSD matrix
    coherence: np.ndarray  # (N, p, p) real-valued coherence matrix
    channels: Optional[np.ndarray] = None

    def __repr__(self):
        return f"EmpiricalPSD(N={self.freq.shape[0]}, p={self.psd.shape[1]})"

    @classmethod
    def from_timeseries_data(
        cls,
        data: np.ndarray,
        fs: float,
        nperseg: int | None = None,
        noverlap: int | None = None,
        window: str = "hann",
        detrend: str | bool = "constant",
    ) -> "EmpiricalPSD":
        p = data.shape[1]

        if nperseg is None:
            # Use half or full data length depending on total size
            n = data.shape[0]
            if n <= 512:
                nperseg = n  # full segment for short data
            else:
                nperseg = n // 2
        if noverlap is None:
            noverlap = nperseg // 2

        # --- auto spectra ---
        psds = []
        f_ref = None
        for i in range(p):
            f, Pxx = welch(
                data[:, i],
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                return_onesided=True,
                detrend=detrend,
                scaling="density",
            )
            psds.append(Pxx)
            if f_ref is None:
                f_ref = f
        psds = np.stack(psds, axis=1)  # (N, p)

        # --- full CSD matrix ---
        S = np.zeros((len(f_ref), p, p), dtype=complex)
        for i in range(p):
            S[:, i, i] = psds[:, i]
            for j in range(i + 1, p):
                _, Sij = csd(
                    data[:, i],
                    data[:, j],
                    fs=fs,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    return_onesided=True,
                    detrend=detrend,
                    scaling="density",
                )
                S[:, i, j] = Sij
                S[:, j, i] = np.conj(Sij)

        coh = _get_coherence(S)
        return cls(freq=f_ref, psd=S, coherence=coh)


def _get_coherence(psd: np.ndarray) -> np.ndarray:
    N, p, _ = psd.shape
    coh = np.zeros((N, p, p))
    for i in range(p):
        coh[:, i, i] = 1.0
        for j in range(i + 1, p):
            denom = np.abs(psd[:, i, i]) * np.abs(psd[:, j, j])
            with np.errstate(divide="ignore", invalid="ignore"):
                coh_ij = np.abs(psd[:, i, j]) ** 2 / denom
            coh[:, i, j] = np.where(denom > 0, coh_ij, 0.0)
            coh[:, j, i] = coh[:, i, j]
    return coh
