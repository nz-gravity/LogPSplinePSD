from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.signal import csd, welch

from ..logger import logger


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
        y_re: Real part of FFT (n_freq, n_dim)
        y_im: Imag part of FFT (n_freq, n_dim)
        u_re: Real part of eigenvector-weighted periodogram replicates
              (n_freq, n_dim, n_dim)
        u_im: Imag part of eigenvector-weighted periodogram replicates
              (n_freq, n_dim, n_dim)
        nu: Degrees of freedom (number of averaged blocks)
        freq: Frequency grid (n_freq,)
        n_freq: Number of frequencies
        n_dim: Number of channels
    """

    y_re: np.ndarray
    y_im: np.ndarray
    u_re: np.ndarray
    u_im: np.ndarray
    freq: np.ndarray
    n_freq: int
    n_dim: int
    nu: int = 1
    scaling_factor: Optional[float] = 1.0  # Track the PSD scaling factor
    fs: float = field(default=1.0, repr=False)
    raw_psd: Optional[np.ndarray] = None
    raw_freq: Optional[np.ndarray] = None

    @classmethod
    def compute_fft(
        cls,
        x: np.ndarray,
        fs: float = 1.0,
        fmin: float = None,
        fmax: float = None,
        scaling_factor: Optional[float] = 1.0,
    ) -> "MultivarFFT":
        """Compute FFT and Wishart replicates with a single (full-length) block."""
        return cls.compute_wishart(
            x,
            fs=fs,
            n_blocks=1,
            fmin=fmin,
            fmax=fmax,
            scaling_factor=scaling_factor,
        )

    @classmethod
    def compute_wishart(
        cls,
        x: np.ndarray,
        fs: float,
        n_blocks: int,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        scaling_factor: Optional[float] = 1.0,
    ) -> "MultivarFFT":
        """
        Compute block-averaged (Wishart) FFT statistics for multivariate series.

        Parameters
        ----------
        x : np.ndarray
            Input time series (n_time, n_channels)
        fs : float
            Sampling frequency
        n_blocks : int
            Number of non-overlapping blocks to average. Must divide n_time.
        fmin, fmax : float, optional
            Optional frequency truncation applied after blocking.
        scaling_factor : float, optional
            PSD scaling factor carried through sampling.
        """
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive.")

        n_time, n_dim = x.shape
        if n_time % n_blocks != 0:
            raise ValueError(
                f"n_time={n_time} must be divisible by n_blocks={n_blocks}."
            )

        block_len = n_time // n_blocks
        if block_len <= n_dim:
            raise ValueError(
                "Block length must exceed number of channels for FFT stability."
            )

        x_centered = x - np.mean(x, axis=0)
        blocks = x_centered.reshape(n_blocks, block_len, n_dim)

        block_ffts = np.fft.fft(blocks, axis=1) / np.sqrt(block_len)
        freqs = np.fft.fftfreq(block_len, 1 / fs)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        block_ffts = block_ffts[:, pos_mask, :]

        # Full-length FFT for empirical PSD/CSD before blocking
        full_fft = np.fft.fft(x_centered, axis=0) / np.sqrt(n_time)
        full_freq = np.fft.fftfreq(n_time, 1 / fs)
        full_mask = full_freq > 0
        full_freq = full_freq[full_mask]
        full_fft = full_fft[full_mask, :]

        if fmin is not None or fmax is not None:
            fmin_eff = freqs[0] if fmin is None else max(fmin, freqs[0])
            fmax_eff = freqs[-1] if fmax is None else min(fmax, freqs[-1])
            if fmax_eff < fmin_eff:
                raise ValueError(
                    f"Invalid frequency bounds: fmin={fmin_eff}, fmax={fmax_eff}."
                )
            freq_mask = (freqs >= fmin_eff) & (freqs <= fmax_eff)
            freqs = freqs[freq_mask]
            block_ffts = block_ffts[:, freq_mask, :]

            full_mask = (full_freq >= fmin_eff) & (full_freq <= fmax_eff)
            full_freq = full_freq[full_mask]
            full_fft = full_fft[full_mask, :]

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

        # Empirical PSD/CSD from full FFT
        norm_factor = 2 * np.pi
        full_psd = np.zeros(
            (full_freq.size, n_dim, n_dim), dtype=np.complex128
        )
        for i in range(n_dim):
            for j in range(n_dim):
                full_psd[:, i, j] = (
                    2
                    * (full_fft[:, i] * np.conj(full_fft[:, j]))
                    / norm_factor
                )

        if full_psd.shape[0] != freqs.size:
            full_psd = _interp_complex_matrix(full_freq, freqs, full_psd)
            full_freq = freqs

        scaling = float(scaling_factor or 1.0)
        full_psd *= scaling

        return cls(
            y_re=y_re,
            y_im=y_im,
            freq=freqs,
            n_freq=len(freqs),
            n_dim=n_dim,
            u_re=u_re,
            u_im=u_im,
            raw_psd=full_psd,
            raw_freq=full_freq,
            nu=n_blocks,
            scaling_factor=scaling_factor,
            fs=fs,
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
        return MultivarFFT(
            y_re=self.y_re[mask],
            y_im=self.y_im[mask],
            freq=self.freq[mask],
            n_freq=int(np.sum(mask)),
            n_dim=self.n_dim,
            u_re=self.u_re[mask],
            u_im=self.u_im[mask],
            raw_psd=raw_psd,
            raw_freq=raw_freq,
            nu=self.nu,
            scaling_factor=self.scaling_factor,
            fs=self.fs,
        )

    @property
    def amplitude_range(self) -> Tuple[float, float]:
        amps = (np.min(self.y_re), np.max(self.y_re))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))

    def __repr__(self):
        return f"MultivarFFT(n_freq={self.n_freq}, n_dim={self.n_dim}, amplitudes={self.amplitude_range})"

    @property
    def empirical_psd(self) -> "EmpiricalPSD":
        return self.get_empirical_psd(
            self.y_re, self.y_im, self.scaling_factor, self.fs
        )

    @staticmethod
    def get_empirical_psd(
        y_re, y_im, scaling=1.0, fs: float = 1
    ) -> "EmpiricalPSD":
        y_re = np.array(y_re, dtype=np.float64)
        y_im = np.array(y_im, dtype=np.float64)
        n_freq, n_dim = y_re.shape
        _y = y_re + 1j * y_im
        S = np.zeros((n_freq, n_dim, n_dim), dtype=np.complex128)
        norm_factor = 2 * np.pi
        for i in range(n_dim):
            for j in range(n_dim):
                S[:, i, j] = 2 * (_y[:, i] * np.conj(_y[:, j])) / norm_factor

        S *= scaling
        coh = _get_coherence(S)
        freq = np.fft.fftfreq(2 * n_freq, 1 / fs)[:n_freq]
        return EmpiricalPSD(
            freq=freq,
            psd=S,
            coherence=coh,
        )


@dataclass
class MultivariateTimeseries:
    y: np.ndarray  # numpy array (n_time, n_channels) for numerical stability
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
    def n_channels(self):
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
        )

    def to_wishart_stats(
        self,
        n_blocks: int,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> "MultivarFFT":
        n_time = self.y.shape[0]
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive.")
        if n_time % n_blocks != 0:
            raise ValueError(
                f"n_time={n_time} must be divisible by n_blocks={n_blocks}."
            )
        block_len = n_time // n_blocks

        wishart_fft = MultivarFFT.compute_wishart(
            self.y,
            fs=self.fs,
            n_blocks=n_blocks,
            fmin=fmin,
            fmax=fmax,
            scaling_factor=self.scaling_factor,
        )
        log_msg = (
            f"Wishart averaging (blocks={n_blocks}): "
            f"n_time={n_time} -> block_len={block_len}, "
            f"n_freq={wishart_fft.n_freq}, n_channels={wishart_fft.n_dim}"
        )
        logger.info(log_msg)
        return wishart_fft

    @property
    def amplitude_range(self) -> Tuple[float, float]:
        amps = (np.min(self.y), np.max(self.y))
        return (float(f"{amps[0]:.3g}"), float(f"{amps[1]:.3g}"))

    def __repr__(self):
        return f"MultivariateTimeseries(n_time={self.y.shape[0]}, n_channels={self.n_channels}, fs={self.fs:.3f}, amplitudes={self.amplitude_range})"

    def get_empirical_psd(self, **kwargs) -> "EmpiricalPSD":
        return EmpiricalPSD.from_timeseries_data(
            self.y,
            fs=self.fs,
            **kwargs,
        )


@dataclass
class EmpiricalPSD:
    freq: np.ndarray  # (n_freq,)
    psd: np.ndarray  # (n_freq, n_channels, n_channels) complex CSD matrix
    coherence: (
        np.ndarray
    )  # (n_freq, n_channels, n_channels) real-valued coherence matrix
    channels: Optional[np.ndarray] = None

    def __repr__(self):
        return f"EmpiricalPSD(n_freq={self.freq.shape[0]}, n_channels={self.psd.shape[1]})"

    @classmethod
    def from_timeseries_data(
        cls,
        data: np.ndarray,
        fs: float,
        nperseg: int | None = None,
        noverlap: int | None = None,
        window: str = "hann",
    ) -> "EmpiricalPSD":
        n_channels = data.shape[1]

        if nperseg is None:
            # Use half or full data length depending on total size
            n_time = data.shape[0]
            if n_time <= 512:
                nperseg = n_time  # full segment for short data
            else:
                nperseg = n_time // 2
        if noverlap is None:
            noverlap = nperseg // 2

        # --- auto spectra ---
        psds = []
        f_ref = None
        for i in range(n_channels):
            f, Pxx = welch(
                data[:, i],
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                return_onesided=True,
                detrend="constant",
                scaling="density",
            )
            psds.append(Pxx)
            if f_ref is None:
                f_ref = f
        psds = np.stack(psds, axis=1)  # (n_freq, n_channels)

        # --- full CSD matrix ---
        S = np.zeros((len(f_ref), n_channels, n_channels), dtype=complex)
        for i in range(n_channels):
            S[:, i, i] = psds[:, i]
            for j in range(i + 1, n_channels):
                _, Sij = csd(
                    data[:, i],
                    data[:, j],
                    fs=fs,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    return_onesided=True,
                    detrend="constant",
                    scaling="density",
                )
                S[:, i, j] = Sij
                S[:, j, i] = np.conj(Sij)

        coh = _get_coherence(S)
        return cls(freq=f_ref, psd=S, coherence=coh)


def _get_coherence(psd: np.ndarray) -> np.ndarray:
    n_freq, n_channels, _ = psd.shape
    coh = np.zeros((n_freq, n_channels, n_channels))
    for i in range(n_channels):
        coh[:, i, i] = 1.0
        for j in range(i + 1, n_channels):
            coh[:, i, j] = np.abs(psd[:, i, j]) ** 2 / (
                np.abs(psd[:, i, i]) * np.abs(psd[:, j, j])
            )
            coh[:, j, i] = coh[:, i, j]
    return coh
