from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Optional

import numpy as np

from .._jaxtypes import Complex, Float, Int
from .._typecheck import runtime_typecheck
from ..datatypes.multivar import MultivarFFT
from ..datatypes.multivar_utils import (
    U_to_Y,
    Y_to_S,
)
from ..logger import logger

__all__ = [
    "CoarseGrainConfig",
    "CoarseGrainSpec",
    "compute_binning_structure",
    "apply_coarse_graining_univar",
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
    """Configuration for frequency-domain coarse graining with equal-sized bins.

    Exactly one of (Nc, Nh) must be provided:
      - Nc: number of coarse bins
      - Nh: number of fine-grid frequencies per coarse bin

    The retained frequency count Nl must be divisible by the chosen parameter so
    that bins are exactly equal-sized across the provided frequency grid.
    """

    enabled: bool = False
    Nc: Optional[int] = 1000
    Nh: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.Nc is None) == (self.Nh is None):
            raise ValueError("Exactly one of Nc or Nh must be set.")
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
        }


@dataclass(slots=True)
class CoarseGrainSpec:
    """Static description of equal-sized coarse bins on a retained frequency grid.

    The provided retained grid of length Nl is partitioned into Nc disjoint,
    consecutive bins J_h, each of constant size Nh (odd).

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

    Nh: int  # constant bin size (odd)
    J_start: np.ndarray  # (Nc,)
    J_mid: np.ndarray  # (Nc,)
    Nc: int  # number of bins

    def __repr__(self):
        return f"CoarseGrainSpec(Nc={self.Nc}, Nh={self.Nh},Nl={self.Nc * self.Nh}"


def _resolve_equal_bin_params(
    *, Nl: int, Nc: Optional[int], Nh: Optional[int]
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
    if Nc_int <= 0 or Nl % Nc_int != 0:
        raise ValueError(f"Nc={Nc_int} must be a positive divisor of Nl={Nl}.")
    Nh_int = Nl // Nc_int
    return int(Nc_int), int(Nh_int)


def _sum_bins_equal(x: np.ndarray, *, Nh: int) -> np.ndarray:
    """Sum consecutive equal-sized bins of length Nh along axis 0.
    out[h, ...] = sum_{k = h*Nh}^{(h+1)*Nh - 1} x[k, ...].

    Eg:
    x = [x0, x1, x2, x3, x4, x5], Nh=2 -> out = [x0+x1, x2+x3, x4+x5]
    """
    x = np.asarray(x)
    Nl = x.shape[0]
    if Nl % Nh != 0:
        raise ValueError("Nl must be divisible by Nh")
    Nc = Nl // Nh
    return x.reshape(Nc, Nh, *x.shape[1:]).sum(axis=1)


def _coarse_grain_wishart_y_to_u(
    Y_sel: Complex[np.ndarray, "nl p p"],
    *,
    Nc: int,
    Nh: int,
) -> Complex[np.ndarray, "nc p p"]:
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

    return MultivarFFT.Y_to_U(Y_bar)


@runtime_typecheck
def compute_binning_structure(
    freqs: Float[np.ndarray, "nl"],
    *,
    Nc: Optional[int] = None,
    Nh: Optional[int] = None,
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

    J_start: Int[np.ndarray, "nc"] = (
        np.arange(Nc, dtype=np.int64) * Nh
    ).astype(np.int32)

    if (Nh % 2) == 0:
        logger.warning(
            f"Nl={Nl} and Nc={Nc} imply even Nh={Nh}. "
            "Midpoint is not unique; using lower-middle Fourier frequency.",
        )
        J_mid: Int[np.ndarray, "nc"] = J_start + (Nh // 2) - 1
    else:
        J_mid = J_start + (Nh // 2)
    f_coarse: Float[np.ndarray, "nc"] = freqs[J_mid].astype(np.float64)

    return CoarseGrainSpec(
        f_coarse=f_coarse,
        Nh=Nh,
        J_start=J_start,
        J_mid=J_mid,
        Nc=Nc,
    )


@runtime_typecheck
def apply_coarse_graining_univar(
    power: Float[np.ndarray, "nl"] | Int[np.ndarray, "nl"],
    spec: CoarseGrainSpec,
    freqs: Optional[Float[np.ndarray, "nl"]] = None,
) -> Float[np.ndarray, "nc"]:
    """Apply coarse graining to a univariate array defined on the retained grid.

    Parameters
    ----------
    power : ndarray, shape (Nl,)
        Values defined on the retained frequency grid.
    spec : CoarseGrainSpec
        Binning specification from compute_binning_structure.
    freqs : ndarray, optional
        Optional retained-grid frequency array for a consistency check.

    Returns
    -------
    power_coarse : ndarray, shape (Nc,)
        Coarse-binned sums.
    """
    power = np.asarray(power)
    if power.ndim != 1:
        raise ValueError("power must be 1-D")

    Nl = int(power.size)
    if Nl != int(spec.Nc * spec.Nh):
        raise ValueError("power length must match retained grid size")

    if freqs is not None:
        freqs = np.asarray(freqs, dtype=np.float64)
        if freqs.ndim != 1 or freqs.size != Nl:
            raise ValueError("freqs must match retained grid size")

    out: Float[np.ndarray, "nc"] = _sum_bins_equal(
        power, Nh=int(spec.Nh)
    ).astype(np.float64, copy=False)
    return out


@runtime_typecheck
def apply_coarse_grain_multivar_fft(
    fft: MultivarFFT, spec: CoarseGrainSpec
) -> MultivarFFT:
    """Coarse-grain a MultivarFFT using equal-sized bins.

    Notes
    -----
    - Y(f_k) is the Wishart matrix built from block FFTs (second-order). We
      form Y_bar by summing Y(f_k) across bins and then re-factorize to obtain
      the coarse-grid U factors used by the likelihood.
    """

    # FFT is already on the retained analysis band.
    u_re_sel = np.asarray(fft.u_re)
    u_im_sel = np.asarray(fft.u_im)
    u_sel = (u_re_sel + 1j * u_im_sel).astype(np.complex128)

    Nc = int(spec.Nc)
    Nh = int(spec.Nh)
    if Nc <= 0:
        raise ValueError("Coarse-graining spec has no bins.")
    if fft.freq.shape[0] != Nc * Nh:
        raise ValueError("FFT frequency grid does not match coarse-grain spec.")

    # Build Y(f_k) explicitly (used only as an intermediate), then coarse-grain
    # and factorize: Y_bar[h] = sum_{f in J_h} Y(f), with Y_bar[h] = U_bar U_bar^H.
    Y_sel = U_to_Y(u_sel)
    u_bins = _coarse_grain_wishart_y_to_u(Y_sel, Nc=Nc, Nh=Nh)

    psd_coarse = Y_to_S(
        U_to_Y(u_bins),
        Nb=int(fft.Nb),
        duration=float(getattr(fft, "duration", 1.0) or 1.0),
        scaling_factor=float(fft.scaling_factor or 1.0),
        Nh=Nh,
    )

    f_coarse = np.asarray(spec.f_coarse, dtype=np.float64)

    fft_coarse = MultivarFFT(
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
        Nh=Nh,
    )

    return fft_coarse
