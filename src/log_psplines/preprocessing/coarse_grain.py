from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import numpy as np
from jaxtyping import Complex, Float

from log_psplines.data.spectral import WishartData
from log_psplines.data.spectral_utils import (
    U_to_Y,
    Y_to_S,
    Y_to_U,
)

from ..logger import logger

__all__ = [
    "CoarseGrainConfig",
    "CoarseGrainSpec",
    "compute_binning_structure",
    "compute_adaptive_binning_structure",
    "apply_coarse_grain_multivar_fft",
]


def _as_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _closest_divisor(n: int, target: int) -> int:
    """Return the divisor of n closest to target (ties resolved downward)."""
    divisors: list[int] = []
    for k in range(1, isqrt(n) + 1):
        if n % k == 0:
            divisors.append(k)
            mate = n // k
            if mate != k:
                divisors.append(mate)
    return min(divisors, key=lambda d: (abs(d - target), d))


@dataclass(slots=True)
class CoarseGrainConfig:
    """Select fixed-width or pilot-guided adaptive frequency aggregation.

    For fixed bins, exactly one of Nc and Nh is required. For adaptive bins,
    Nc/Nh are ignored and the accepted width is bounded by max_bin_width.
    A smoothed empirical spectral matrix is only used to choose boundaries;
    unsmoothed Wishart statistics enter the likelihood.
    """

    enabled: bool = False
    Nc: int | None = 1000
    Nh: int | None = None
    method: str = "fixed"
    tolerance: float = 0.15
    max_bin_width: int = 32
    pilot_window: int = 31

    def __post_init__(self) -> None:
        if self.method not in ("fixed", "adaptive"):
            raise ValueError("method must be fixed or adaptive")
        if self.method == "fixed" and (self.Nc is None) == (self.Nh is None):
            raise ValueError("Exactly one of Nc or Nh must be set.")
        if self.method == "adaptive" and self.Nh is not None:
            raise ValueError("Nh is not used by adaptive binning")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        self.max_bin_width = _as_int("max_bin_width", self.max_bin_width)
        self.pilot_window = _as_int("pilot_window", self.pilot_window)
        if self.max_bin_width < 1 or self.pilot_window < 1:
            raise ValueError("max_bin_width and pilot_window must be positive")
        if self.pilot_window % 2 == 0:
            raise ValueError("pilot_window must be odd")
        if self.Nc is not None:
            self.Nc = _as_int("Nc", self.Nc)
            if self.Nc <= 0:
                raise ValueError("Nc must be positive when provided.")
        if self.Nh is not None:
            self.Nh = _as_int("Nh", self.Nh)
            if self.Nh <= 0:
                raise ValueError("Nh must be positive when provided.")

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "Nc": self.Nc,
            "Nh": self.Nh,
            "method": self.method,
            "tolerance": self.tolerance,
            "max_bin_width": self.max_bin_width,
            "pilot_window": self.pilot_window,
        }


@dataclass(slots=True)
class CoarseGrainSpec:
    """Disjoint contiguous partitions of the retained frequency index grid.

    J_stop is exclusive; freq_bin_counts is the number of fine frequencies
    per coarse bin. Nh remains a legacy scalar for fixed-width callers.
    """

    f_coarse: np.ndarray  # (Nc,)

    Nh: int  # constant bin size (odd)
    J_start: np.ndarray  # (Nc,)
    J_mid: np.ndarray  # (Nc,)
    Nc: int  # number of bins
    J_stop: np.ndarray | None = None  # exclusive boundaries
    freq_bin_counts: np.ndarray | None = None

    def __post_init__(self):
        self.J_start = np.asarray(self.J_start, dtype=np.int64)
        self.J_mid = np.asarray(self.J_mid, dtype=np.int64)
        if self.J_stop is None:
            self.J_stop = self.J_start + self.Nh
        self.J_stop = np.asarray(self.J_stop, dtype=np.int64)
        if self.freq_bin_counts is None:
            self.freq_bin_counts = self.J_stop - self.J_start
        self.freq_bin_counts = np.asarray(self.freq_bin_counts, dtype=np.int64)
        if any(
            a.shape != (self.Nc,)
            for a in (
                self.J_start,
                self.J_stop,
                self.J_mid,
                self.freq_bin_counts,
                self.f_coarse,
            )
        ):
            raise ValueError("partition arrays must have length Nc")
        if self.J_start[0] != 0 or np.any(
            self.J_start[1:] != self.J_stop[:-1]
        ):
            raise ValueError("partition must be contiguous and start at zero")
        if np.any(self.J_stop <= self.J_start) or np.any(
            self.freq_bin_counts != self.J_stop - self.J_start
        ):
            raise ValueError("partition has invalid bin counts")
        if np.any((self.J_mid < self.J_start) | (self.J_mid >= self.J_stop)):
            raise ValueError(
                "representative frequency must be within each bin"
            )

    def __repr__(self):
        return f"CoarseGrainSpec(Nc={self.Nc}, Nl={self.freq_bin_counts.sum()}, widths={self.freq_bin_counts.min()}..{self.freq_bin_counts.max()})"


def _resolve_equal_bin_params(
    *, Nl: int, Nc: int | None, Nh: int | None
) -> tuple[int, int]:
    """Resolve (Nc, Nh) for equal-sized bins.

    If Nh is provided explicitly, it must be positive.
    If Nc is provided, Nh = Nl // Nc may be even.
    """
    if (Nc is None) == (Nh is None):
        raise ValueError("Exactly one of Nc or Nh must be provided.")

    if Nh is not None:
        Nh = _as_int("Nh", Nh)
        if Nh <= 0:
            raise ValueError("Nh must be positive.")
        if Nl % Nh != 0:
            suggested_Nh = _closest_divisor(Nl, Nh)
            suggested_Nc = Nl // suggested_Nh
            raise ValueError(
                f"Nh={Nh} must divide Nl={Nl}. "
                f"Closest valid Nh={suggested_Nh} (Nc={suggested_Nc})."
            )
        Nc = Nl // Nh
        return int(Nc), int(Nh)

    # Nc provided
    assert Nc is not None
    Nc_int = _as_int("Nc", Nc)
    if Nc_int <= 0:
        raise ValueError(f"Nc={Nc_int} must be positive.")
    if Nl % Nc_int != 0:
        # Trim the last (Nl % Nc_int) bins so bins are exactly equal-sized.
        n_dropped = Nl % Nc_int
        Nl = Nl - n_dropped
        logger.info(
            f"Nl trimmed to {Nl} (dropped {n_dropped} trailing bins) "
            f"to make Nc={Nc_int} an exact divisor."
        )
    Nh_int = Nl // Nc_int
    return int(Nc_int), int(Nh_int)


def _coarse_grain_wishart_y_to_u(
    Y_sel: Complex[np.ndarray, "N p p"],
    *,
    Nc: int,
    Nh: int,
) -> np.ndarray:
    """Coarse-grain Wishart matrices by summing Y(f) within each bin.

    For each bin h, compute:
        Y_bar[h] = sum_{f in J_h} Y(f)
    and return U_bar such that:
        Y_bar[h] = U_bar[h] U_bar[h]^H

    Y_sel should be the fine-grid Wishart matrix Y(f_k), not the mean FFT.
    This mirrors the math directly and keeps the compact (p x p) factorization.
    """
    if Nc <= 0:
        raise ValueError("Nc must be positive.")
    if Nh <= 0:
        raise ValueError("Nh must be positive.")
    if Y_sel.ndim != 3:
        raise ValueError("Y_sel must have shape (Nl, p, p)")
    if Y_sel.shape[0] != Nc * Nh:
        raise ValueError(
            "Y_sel length must equal Nc * Nh on the retained frequency grid"
        )

    # Coarse-grain by summing within bins.
    Y_bar = Y_sel.reshape(Nc, Nh, *Y_sel.shape[1:]).sum(axis=1)

    return Y_to_U(Y_bar)


def compute_binning_structure(
    freqs: Float[np.ndarray, "N"],
    *,
    Nc: int | None = None,
    Nh: int | None = None,
) -> CoarseGrainSpec:
    """Compute equal-sized, consecutive coarse bins for a frequency grid.

    Provide exactly one of:
      - Nc : number of coarse bins
      - Nh : number of fine-grid frequencies per bin

    The retained frequency count Nl must be divisible by the chosen parameter so
    that bins are exactly equal-sized across the provided grid. Representative
    frequencies are midpoint Fourier frequencies (central indices), not averages.
    For even Nh, the lower-middle Fourier frequency is used.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1-D array")
    if freqs.size == 0:
        raise ValueError("freqs must be non-empty")
    if not np.all(np.diff(freqs) >= 0):
        raise ValueError("freqs must be monotonically increasing")

    Nl = int(freqs.size)

    Nc, Nh = _resolve_equal_bin_params(Nl=Nl, Nc=Nc, Nh=Nh)

    J_start = (np.arange(Nc, dtype=np.int64) * Nh).astype(np.int32)

    if (Nh % 2) == 0:
        logger.info(f"Nl={Nl} and Nc={Nc} imply even Nh={Nh}. ")
        J_mid = J_start + (Nh // 2) - 1
    else:
        J_mid = J_start + (Nh // 2)
    f_coarse = freqs[J_mid].astype(np.float64)

    return CoarseGrainSpec(
        f_coarse=f_coarse,
        Nh=Nh,
        J_start=J_start,
        J_mid=J_mid,
        Nc=Nc,
    )


def compute_adaptive_binning_structure(
    fft: WishartData,
    *,
    tolerance: float = 0.15,
    max_bin_width: int = 32,
    pilot_window: int = 31,
) -> CoarseGrainSpec:
    """Pilot-guided contiguous frequency bins with a bounded matrix change.

    The pilot *only* selects bin edges. Actual Wishart statistics are pooled
    from unsmoothed fine-frequency matrices. A relative operator-norm test
    protects both autospectra and complex cross-spectral phase. Gaps in a
    retained Fourier grid separate bins.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    max_bin_width = _as_int("max_bin_width", max_bin_width)
    pilot_window = _as_int("pilot_window", pilot_window)
    if max_bin_width < 1 or pilot_window < 1 or pilot_window % 2 != 1:
        raise ValueError(
            "max_bin_width must be positive, pilot_window positive and odd"
        )
    f = np.asarray(fft.freq, dtype=float)
    if f.size == 0 or not np.all(np.diff(f) > 0):
        raise ValueError(
            "frequency grid must be nonempty and strictly increasing"
        )
    matrix = (
        np.asarray(fft.raw_psd)
        if fft.raw_psd is not None
        else fft.Y / (fft.Nb * fft.duration)
    )
    if not np.isfinite(matrix).all():
        raise ValueError("pilot spectral matrix must be finite")
    # For a masked Fourier grid, smooth only within uninterrupted runs.
    steps = np.diff(f)
    gap = (
        np.r_[False, steps > 1.5 * np.min(steps)]
        if steps.size
        else np.array([False])
    )
    group_edges = np.r_[0, np.flatnonzero(gap), len(f)]
    pilot = np.empty_like(matrix)
    for left, right in zip(group_edges[:-1], group_edges[1:], strict=False):
        block = matrix[left:right]
        radius = pilot_window // 2
        indexes = np.arange(len(block))
        lo = np.maximum(0, indexes - radius)
        hi = np.minimum(len(block), indexes + radius + 1)
        prefix = np.concatenate(
            [np.zeros_like(block[:1]), np.cumsum(block, axis=0)], axis=0
        )
        pilot[left:right] = (prefix[hi] - prefix[lo]) / (hi - lo)[
            :, None, None
        ]
    pilot = 0.5 * (pilot + pilot.conj().swapaxes(-1, -2))

    def deviation(start: int, stop: int) -> float:
        mid = start + (stop - start - 1) // 2
        eig, vec = np.linalg.eigh(pilot[mid])
        # A local relative floor prevents near-singular TDI nulls from
        # making the whitened comparison numerically meaningless.
        floor = max(float(np.max(np.abs(eig))) * 1e-6, np.finfo(float).tiny)
        inverse_root = (vec / np.sqrt(np.maximum(eig, floor))) @ vec.conj().T
        normalized = inverse_root @ pilot[start:stop] @ inverse_root
        errors = np.linalg.eigvalsh(normalized - np.eye(fft.p, dtype=complex))
        return float(np.max(np.abs(errors)))

    starts, stops, mids = [], [], []
    start = 0
    while start < len(f):
        stop = start + 1
        limit = min(len(f), start + max_bin_width)
        while stop < limit and not gap[stop]:
            if deviation(start, stop + 1) > tolerance:
                break
            stop += 1
        starts.append(start)
        stops.append(stop)
        mids.append(start + (stop - start - 1) // 2)
        start = stop
    starts = np.asarray(starts, dtype=np.int64)
    stops = np.asarray(stops, dtype=np.int64)
    mids = np.asarray(mids, dtype=np.int64)
    return CoarseGrainSpec(
        f_coarse=f[mids],
        Nh=1,
        J_start=starts,
        J_stop=stops,
        J_mid=mids,
        Nc=len(starts),
        freq_bin_counts=stops - starts,
    )


def apply_coarse_grain_multivar_fft(
    fft: WishartData, spec: CoarseGrainSpec
) -> WishartData:
    """Coarse-grain WishartData with any contiguous frequency partition.

    Notes
    -----
    - Y(f_k) is the Wishart matrix built from block FFTs (second-order). We
      form Y_bar by summing Y(f_k) across bins and then re-factorize to obtain
      the coarse-grid U factors used by the likelihood.
    """

    # FFT is already on the retained analysis band.
    u_sel = fft.U

    Nc = int(spec.Nc)
    if Nc <= 0 or spec.J_stop[-1] > fft.N:
        raise ValueError("Invalid coarse-grain partition for frequency grid")
    # Each sum retains the Hermitian cross-spectral matrix, including phase.
    Y_sel = U_to_Y(u_sel[: spec.J_stop[-1]])
    Y_bar = np.add.reduceat(Y_sel, spec.J_start, axis=0)
    u_bins = Y_to_U(Y_bar)

    psd_coarse = Y_to_S(
        U_to_Y(u_bins),
        Nb=int(fft.Nb),
        duration=float(getattr(fft, "duration", 1.0) or 1.0),
        scaling_factor=float(fft.scaling_factor or 1.0),
        Nh=spec.freq_bin_counts,
    )

    f_coarse = np.asarray(spec.f_coarse, dtype=np.float64)

    fft_coarse = WishartData(
        u_re=u_bins.real.astype(np.float64),
        u_im=u_bins.imag.astype(np.float64),
        freq=f_coarse,
        N=int(f_coarse.shape[0]),
        p=int(fft.p),
        Nb=int(fft.Nb),
        scaling_factor=fft.scaling_factor,
        fs=fft.fs,
        duration=float(getattr(fft, "duration", 1.0) or 1.0),
        raw_psd=psd_coarse.astype(np.complex128),
        raw_freq=f_coarse,
        channel_stds=fft.channel_stds,
        Nh=spec.Nh,
        freq_bin_counts=spec.freq_bin_counts,
        enbw=float(getattr(fft, "enbw", 1.0) or 1.0),
        preprocessing=fft.preprocessing,
    )

    return fft_coarse
