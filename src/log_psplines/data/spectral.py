from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from log_psplines.data.spectral_utils import (
    U_to_Y,
    _get_coherence,
    u_re_im_to_U,
    wishart_u_to_psd,
)


@dataclass
class WishartData:
    """
    Discrete FFTs for multivariate time series.
    Stores Wishart replicates for multivariate spectral estimation.

    Attributes
    ----------
    u_re
        Real part of eigenvector-weighted periodogram replicates, shape
        ``(N, p, p)``.
    u_im
        Imaginary part of eigenvector-weighted periodogram replicates, shape
        ``(N, p, p)``.
    Nb
        Degrees of freedom, equal to the number of averaged blocks.
    freq
        Frequency grid, shape ``(N,)``.
    N
        Number of frequencies.
    p
        Number of channels.

    Notes
    -----
    The multivariate Wishart matrix is
    ``Y(f_k) = sum_b X_b(f_k) X_b(f_k)^H``. It is encoded through ``U`` such
    that ``Y(f_k) = U(f_k) U(f_k)^H``. The ``U`` factors are the sufficient
    statistics used by the Wishart likelihood. When coarse-graining, the code
    forms ``Y_bar`` by summing ``Y(f)`` across bins and then re-factorizes to
    get coarse ``U``.
    """

    u_re: np.ndarray
    u_im: np.ndarray
    freq: np.ndarray
    N: int
    p: int
    Nb: int = 1
    scaling_factor: float | None = 1.0  # Track the PSD scaling factor
    channel_stds: np.ndarray | None = None  # Per-channel standard deviations
    fs: float = field(default=1.0, repr=False)
    # Duration (seconds) of each time-domain block used to form the FFT/Wishart
    # statistic. This matches the "T" factor in the Whittle likelihood.
    duration: float = field(default=1.0, repr=False)
    raw_psd: np.ndarray | None = None
    raw_freq: np.ndarray | None = None
    # Coarse-grain multiplicity Nh. For equal-sized bins this is constant
    # across the retained frequency grid. Defaults to 1 (no coarse graining).
    Nh: int = 1
    # Normalized Equivalent Noise Bandwidth (NENBW) of the analysis window,
    # in units of frequency bins.  Computed automatically by compute_wishart()
    # as  NENBW = Lb * Σ w(t)² / (Σ w(t))²  (Heinzel et al. 2002, eq. 21).
    # Rectangular → 1.0;  Hann → 1.5;  Tukey(0.1) → 1.04.
    # The Whittle log-likelihood is divided by this factor so that the
    # posterior width accounts for the reduced effective DOF per frequency
    # bin when a non-rectangular taper is used.
    enbw: float = 1.0

    def __post_init__(self) -> None:
        self.u_re = np.asarray(self.u_re, dtype=np.float64)
        self.u_im = np.asarray(self.u_im, dtype=np.float64)
        self.freq = np.asarray(self.freq, dtype=np.float64)
        self.duration = float(self.duration)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be a positive finite float")

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

        if isinstance(self.Nb, bool) or not isinstance(
            self.Nb, (int, np.integer)
        ):
            raise TypeError("Nb must be a positive integer")
        self.Nb = int(self.Nb)
        if self.Nb <= 0:
            raise ValueError("Nb must be a positive integer")

        if isinstance(self.Nh, bool) or not isinstance(
            self.Nh, (int, np.integer)
        ):
            raise TypeError("Nh must be a positive integer")
        self.Nh = int(self.Nh)
        if self.Nh <= 0:
            raise ValueError("Nh must be a positive integer")

        self.enbw = float(self.enbw)
        if not np.isfinite(self.enbw) or self.enbw <= 0.0:
            raise ValueError("enbw must be a positive finite float")

        if self.channel_stds is not None:
            self.channel_stds = np.asarray(self.channel_stds, dtype=np.float64)
            if self.channel_stds.shape != (self.p,):
                raise ValueError(
                    "channel_stds must have length equal to number of channels"
                )

    def apply_mask(self, mask: np.ndarray) -> "WishartData":
        """Return a new WishartData retaining only bins where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.N,):
            raise ValueError(
                f"mask must have shape ({self.N},), got {mask.shape}."
            )
        n_kept = int(np.count_nonzero(mask))
        if n_kept <= 0:
            raise ValueError(
                "Frequency masking removed all bins; adjust exclusion bands."
            )
        raw_psd = None
        raw_freq = None
        if self.raw_psd is not None:
            raw_psd = self.raw_psd[mask]
        if self.raw_freq is not None:
            raw_freq = self.raw_freq[mask]
        return WishartData(
            freq=self.freq[mask],
            N=n_kept,
            p=self.p,
            u_re=self.u_re[mask],
            u_im=self.u_im[mask],
            raw_psd=raw_psd,
            raw_freq=raw_freq,
            Nh=self.Nh,
            Nb=self.Nb,
            scaling_factor=self.scaling_factor,
            fs=self.fs,
            duration=self.duration,
            channel_stds=self.channel_stds,
            enbw=self.enbw,
        )

    def filter_frequency_mask(self, mask: np.ndarray) -> "WishartData":
        """Alias for :meth:`apply_mask` for compatibility with newer callers."""
        return self.apply_mask(mask)

    def cut(self, fmin: float, fmax: float) -> "WishartData":
        """Return a new WishartData within frequency range [fmin, fmax]."""
        if fmax < fmin:
            raise ValueError(
                f"Invalid frequency bounds supplied: fmin={fmin}, fmax={fmax}."
            )
        return self.filter_frequency_mask(
            (self.freq >= fmin) & (self.freq <= fmax)
        )

    def exclude_frequency_bands(
        self,
        bands: Sequence[tuple[float, float]],
    ) -> "WishartData":
        """Return a new WishartData excluding bins in ``bands``."""
        if len(bands) == 0:
            return self
        mask = np.ones((self.N,), dtype=bool)
        for low, high in bands:
            low_f = float(low)
            high_f = float(high)
            if high_f < low_f:
                low_f, high_f = high_f, low_f
            mask &= ~((self.freq >= low_f) & (self.freq <= high_f))
        return self.filter_frequency_mask(mask)

    def __repr__(self):
        return f"WishartData(N={self.N}, p={self.p})"

    @property
    def empirical_psd(self) -> "EmpiricalPSD":
        sf = float(self.scaling_factor or 1.0)
        if self.raw_psd is not None:
            freq = (
                np.asarray(self.raw_freq, dtype=np.float64)
                if self.raw_freq is not None
                else np.asarray(self.freq, dtype=np.float64)
            )
            psd = np.asarray(self.raw_psd, dtype=np.complex128)
        else:
            freq = np.asarray(self.freq, dtype=np.float64)
            psd = wishart_u_to_psd(
                self.U,
                Nb=self.Nb,
                duration=self.duration,
                scaling_factor=sf,
                Nh=self.Nh,
            )

        coherence = _get_coherence(psd)
        out = EmpiricalPSD(freq=freq, psd=psd, coherence=coherence)
        if self.channel_stds is not None:
            scale_matrix = np.outer(self.channel_stds, self.channel_stds)
            out = EmpiricalPSD(
                freq=out.freq,
                psd=out.psd * (scale_matrix / sf),
                coherence=out.coherence,
                channels=out.channels,
            )
        return out

    @property
    def Y(self) -> np.ndarray:
        """Return the Wishart matrices Y[f] = U[f] U[f]^H."""
        return U_to_Y(self.U)

    @property
    def U(self) -> np.ndarray:
        """Return the complex Wishart factors U[f]."""
        return u_re_im_to_U(self.u_re, self.u_im)


@dataclass
class EmpiricalPSD:
    freq: np.ndarray  # (N,)
    psd: np.ndarray  # (N, p, p) complex CSD matrix
    coherence: np.ndarray  # (N, p, p) real-valued coherence matrix
    channels: np.ndarray | None = None

    def __repr__(self):
        return f"EmpiricalPSD(N={self.freq.shape[0]}, p={self.psd.shape[1]})"


@dataclass(frozen=True)
class PowerSpectrum:
    """Summed squared real components and exact counts on a spectral grid.

    Stationary shape (F,), time-frequency shape (T,F). Counts may be
    broadcastable on input. Missing cells must have power=count=0.
    Values are component variances, not automatically a PSD per Hz.
    """

    power: np.ndarray
    counts: np.ndarray
    frequency: np.ndarray
    time: np.ndarray | None = None
    units: str = "coefficient variance"

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequency, dtype=float)
        time = (
            None if self.time is None else np.asarray(self.time, dtype=float)
        )
        for grid in (frequency, time):
            if grid is not None and (
                grid.ndim != 1
                or grid.size == 0
                or not np.isfinite(grid).all()
                or np.any(np.diff(grid) <= 0)
            ):
                raise ValueError(
                    "spectral grids must be finite and increasing"
                )
        shape = (
            (len(frequency),) if time is None else (len(time), len(frequency))
        )
        power = np.asarray(self.power, dtype=float)
        counts = np.broadcast_to(
            np.asarray(self.counts, dtype=float), shape
        ).copy()
        if power.shape != shape:
            raise ValueError(f"power must have shape {shape}")
        if (
            not np.isfinite(power).all()
            or np.any(power < 0)
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
            or not np.any(counts > 0)
        ):
            raise ValueError(
                "powers/counts must be finite, non-negative and observed"
            )
        if np.any((counts == 0) & (power != 0)):
            raise ValueError("zero-count cells must have zero power")
        for name, value in (
            ("power", power),
            ("counts", counts),
            ("frequency", frequency),
            ("time", time),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ScatteredPowerSpectrum:
    """Summed squared real components observed at scattered (time, freq) points.

    Companion to :class:`PowerSpectrum` for transforms whose ordinates do not
    share a common time or frequency axis (e.g. the Tang zig-zag moving
    periodogram before pooling). ``time``, ``frequency``, ``power`` and
    ``counts`` are all one-dimensional and share the same length: ordinate
    ``i`` was observed at ``(time[i], frequency[i])``. Values are component
    variances, not automatically a PSD per Hz.
    """

    power: np.ndarray
    counts: np.ndarray
    time: np.ndarray
    frequency: np.ndarray
    units: str = "coefficient variance"

    def __post_init__(self) -> None:
        power = np.asarray(self.power, dtype=float)
        time = np.asarray(self.time, dtype=float)
        frequency = np.asarray(self.frequency, dtype=float)
        counts = np.broadcast_to(
            np.asarray(self.counts, dtype=float), power.shape
        ).copy()
        if power.ndim != 1 or power.size == 0:
            raise ValueError("scattered power must be non-empty and 1-D")
        if not (power.shape == counts.shape == time.shape == frequency.shape):
            raise ValueError(
                "power, counts, time and frequency must share shape"
            )
        for name, value in (
            ("power", power),
            ("counts", counts),
            ("time", time),
            ("frequency", frequency),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if np.any(power < 0) or np.any(counts <= 0):
            raise ValueError(
                "power must be non-negative and counts strictly positive"
            )
        for name, value in (
            ("power", power),
            ("counts", counts),
            ("time", time),
            ("frequency", frequency),
        ):
            object.__setattr__(self, name, value)
