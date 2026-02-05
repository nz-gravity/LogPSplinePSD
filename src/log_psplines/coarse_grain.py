from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .datatypes.multivar import MultivarFFT
from .logger import logger
from .spectrum_utils import (
    sum_wishart_outer_products,
    u_to_wishart_matrix,
    wishart_matrix_to_psd,
)

__all__ = [
    "CoarseGrainConfig",
    "CoarseGrainSpec",
    "compute_binning_structure",
    "apply_coarse_graining_univar",
    "apply_coarse_grain_multivar_fft",
]


@dataclass(slots=True)
class CoarseGrainConfig:
    """Configuration for frequency-domain coarse graining with equal-sized bins.

    Exactly one of (Nc, Nh) must be provided:
      - Nc: number of coarse bins
      - Nh: number of fine-grid frequencies per coarse bin (must be odd)

    The retained frequency count Nl must be divisible by the chosen parameter so
    that bins are exactly equal-sized across the analysis band. In Nc mode this
    implies Nh = Nl // Nc must be odd.
    """

    enabled: bool = False
    Nc: Optional[int] = 1000
    Nh: Optional[int] = None
    f_min: Optional[float] = None
    f_max: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.Nc is None) == (self.Nh is None):
            raise ValueError("Exactly one of Nc or Nh must be set.")
        if self.Nc is not None:
            self.Nc = int(self.Nc)
            if self.Nc <= 0:
                raise ValueError("Nc must be positive when provided.")
        if self.Nh is not None:
            self.Nh = int(self.Nh)
            if self.Nh <= 0:
                raise ValueError("Nh must be positive when provided.")
            if self.Nh % 2 == 0:
                raise ValueError(
                    "Nh must be odd to define a midpoint frequency."
                )

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "Nc": self.Nc,
            "Nh": self.Nh,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }


@dataclass(slots=True)
class CoarseGrainSpec:
    """Static description of equal-sized coarse bins on a retained frequency grid.

    The frequency grid is restricted to a contiguous analysis band [f_min, f_max],
    producing a retained grid of length Nl. The retained grid is partitioned into
    Nc disjoint, consecutive bins J_h, each of constant size Nh (odd).

    For each bin h:
      - J_h spans indices J_start[h] : J_start[h] + Nh on the retained grid
      - J_mid[h] = J_start[h] + (Nh // 2)
      - f_coarse[h] = freqs_retained[J_mid[h]]

    Notes
    -----
    This spec describes consecutive, full-band bins on the retained grid:
    no gaps, no overlaps, and no per-bin sorting.
    """

    f_coarse: np.ndarray  # (Nc,)
    selection_mask: np.ndarray  # (N_full,)

    Nh: int  # constant bin size (odd)
    J_start: np.ndarray  # (Nc,)
    J_mid: np.ndarray  # (Nc,)
    Nc: int  # number of bins

    def __repr__(self):
        return f"CoarseGrainSpec(Nc={self.Nc}, Nh={self.Nh},Nl={self.Nc * self.Nh}"


def _select_band(
    freqs: np.ndarray, f_min: Optional[float], f_max: Optional[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (selection_mask, freqs_sel) for the analysis band [f_min, f_max].

    The requested bounds are clamped to the available frequency grid.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1-D array")
    if freqs.size == 0:
        raise ValueError("freqs must be non-empty")
    if not np.all(np.diff(freqs) >= 0):
        raise ValueError("freqs must be monotonically increasing")

    f0 = float(freqs[0])
    f1 = float(freqs[-1])
    lo = f0 if f_min is None else float(f_min)
    hi = f1 if f_max is None else float(f_max)

    lo = min(max(lo, f0), f1)
    hi = min(max(hi, f0), f1)
    if hi < lo:
        hi = lo

    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        raise ValueError("No frequencies fall within the requested range")
    return mask, freqs[mask]


def _resolve_equal_bin_params(
    *, Nl: int, Nc: Optional[int], Nh: Optional[int]
) -> tuple[int, int]:
    """Resolve (Nc, Nh) for equal-sized bins.

    If Nh is provided explicitly, it must be odd.
    If Nc is provided, Nh = Nl // Nc may be even.
    """
    if (Nc is None) == (Nh is None):
        raise ValueError("Exactly one of Nc or Nh must be provided.")

    if Nh is not None:
        Nh = int(Nh)
        if Nh <= 0 or (Nh % 2) == 0:
            raise ValueError("Nh must be positive and odd.")
        if Nl % Nh != 0:
            suggested_Nh = max(1, (Nl // (Nl // Nh)))
            raise ValueError(
                f"Nh={Nh} must divide Nl={Nl}. Suggested Nh={suggested_Nh}."
            )
        Nc = Nl // Nh
        return int(Nc), int(Nh)

    # Nc provided
    Nc = int(Nc)
    if Nc <= 0 or Nl % Nc != 0:
        raise ValueError(f"Nc={Nc} must be a positive divisor of Nl={Nl}.")
    Nh = Nl // Nc
    return int(Nc), int(Nh)


def _sum_bins_equal(x: np.ndarray, *, Nh: int) -> np.ndarray:
    """Sum consecutive equal-sized bins of length Nh along axis 0.
    out[h, ...] = sum_{k = h*Nh}^{(h+1)*Nh - 1} x[k, ...].
    """
    x = np.asarray(x)
    Nl = x.shape[0]
    if Nl % Nh != 0:
        raise ValueError("Nl must be divisible by Nh")
    Nc = Nl // Nh
    return x.reshape(Nc, Nh, *x.shape[1:]).sum(axis=1)


def compute_binning_structure(
    freqs: np.ndarray,
    *,
    Nc: Optional[int] = None,
    Nh: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> CoarseGrainSpec:
    """Compute equal-sized, consecutive coarse bins for a frequency grid.

    Provide exactly one of:
      - Nc : number of coarse bins
      - Nh : number of fine-grid frequencies per bin (odd)

    The retained frequency count Nl must be divisible by the chosen parameter so
    that bins are exactly equal-sized across the analysis band. Representative
    frequencies are midpoint Fourier frequencies (central indices), not averages.
    """
    selection_mask, freqs_sel = _select_band(freqs, f_min, f_max)
    Nl = int(freqs_sel.size)

    Nc, Nh = _resolve_equal_bin_params(Nl=Nl, Nc=Nc, Nh=Nh)

    J_start = (np.arange(Nc, dtype=np.int64) * Nh).astype(np.int32)

    if (Nh % 2) == 0:
        logger.warning(
            f"Nl={Nl} and Nc={Nc} imply even Nh={Nh}. "
            "Midpoint is not unique; using lower-middle Fourier frequency.",
        )
        J_mid = J_start + (Nh // 2) - 1  # lower-middle
    else:
        J_mid = J_start + (Nh // 2)  # exact middle <-- IDEAL (when Nh is odd)
    f_coarse = freqs_sel[J_mid].astype(np.float64)

    return CoarseGrainSpec(
        f_coarse=f_coarse,
        selection_mask=selection_mask,
        Nh=Nh,
        J_start=J_start,
        J_mid=J_mid,
        Nc=Nc,
    )


def apply_coarse_graining_univar(
    power: np.ndarray,
    spec: CoarseGrainSpec,
    freqs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply coarse graining to a univariate array defined on the retained grid.

    Parameters
    ----------
    power : ndarray, shape (Nl,)
        Values defined on the retained frequency grid, ordered as freqs[selection_mask].
    spec : CoarseGrainSpec
        Binning specification from compute_binning_structure.
    freqs : ndarray, optional
        Optional retained-grid frequency array for a consistency check.

    Returns
    -------
    power_coarse : ndarray, shape (Nc,)
        Coarse-binned sums.
    weights : ndarray, shape (Nc,)
        Per-bin weights (all equal to Nh for equal-sized bins).
    """
    power = np.asarray(power)
    if power.ndim != 1:
        raise ValueError("power must be 1-D")

    Nl = int(np.sum(spec.selection_mask))
    if power.size != Nl:
        raise ValueError("power length must match retained grid size")

    if freqs is not None:
        freqs = np.asarray(freqs, dtype=np.float64)
        if freqs.ndim != 1 or freqs.size != Nl:
            raise ValueError("freqs must match retained grid size")

    out = _sum_bins_equal(power, Nh=int(spec.Nh)).astype(
        np.float64, copy=False
    )
    weights = np.full((int(spec.Nc),), float(spec.Nh), dtype=np.float64)
    return out, weights


def apply_coarse_grain_multivar_fft(
    fft: MultivarFFT, spec: CoarseGrainSpec
) -> Tuple[MultivarFFT, np.ndarray]:
    """Coarse-grain a MultivarFFT using equal-sized bins."""
    selection = np.asarray(spec.selection_mask, dtype=bool)
    if selection.ndim != 1:
        raise ValueError("CoarseGrainSpec.selection_mask must be 1-D")
    if selection.shape[0] != fft.freq.shape[0]:
        raise ValueError(
            "CoarseGrainSpec.selection_mask length does not match FFT frequencies"
        )

    # Restrict to retained analysis band.
    y_re_sel = np.asarray(fft.y_re)[selection]
    y_im_sel = np.asarray(fft.y_im)[selection]
    u_re_sel = np.asarray(fft.u_re)[selection]
    u_im_sel = np.asarray(fft.u_im)[selection]
    u_sel = (u_re_sel + 1j * u_im_sel).astype(np.complex128)

    Nc = int(spec.Nc)
    Nh = int(spec.Nh)
    if Nc <= 0:
        raise ValueError("Coarse-graining spec has no bins.")

    # y sums via shared helper
    y_re_bins = _sum_bins_equal(y_re_sel, Nh=Nh).astype(np.float64, copy=False)
    y_im_bins = _sum_bins_equal(y_im_sel, Nh=Nh).astype(np.float64, copy=False)

    # u still needs per-bin eigendecomposition
    p = int(fft.p)
    u_bins = np.zeros((Nc, p, p), dtype=np.complex128)

    for b in range(Nc):
        s = b * Nh
        sl = slice(s, s + Nh)

        Y_sum = sum_wishart_outer_products(u_sel[sl])
        try:
            eigvals, eigvecs = np.linalg.eigh(Y_sum)
        except np.linalg.LinAlgError:
            Ys = 0.5 * (Y_sum + Y_sum.conj().T)
            eigvals, eigvecs = np.linalg.eigh(Ys)

        eigvals = np.clip(eigvals.real, a_min=0.0, a_max=None)
        sqrt_eig = np.sqrt(eigvals).astype(np.float64)
        u_bins[b] = eigvecs * sqrt_eig[np.newaxis, :]

    weights = np.full((Nc,), float(Nh), dtype=np.float64)

    psd_coarse = wishart_matrix_to_psd(
        u_to_wishart_matrix(u_bins),
        Nb=int(fft.Nb),
        duration=float(getattr(fft, "duration", 1.0) or 1.0),
        scaling_factor=float(fft.scaling_factor or 1.0),
        weights=weights,
    )

    f_coarse = np.asarray(spec.f_coarse, dtype=np.float64)

    fft_coarse = MultivarFFT(
        y_re=y_re_bins.astype(np.float64),
        y_im=y_im_bins.astype(np.float64),
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
        freq_bin_counts=weights,
    )

    return fft_coarse, weights
