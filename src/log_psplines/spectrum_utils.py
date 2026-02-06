"""Shared PSD conversion helpers for multivariate spectral analysis."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def interp_matrix(
    freq_src: np.ndarray, mat: np.ndarray, freq_tgt: np.ndarray
) -> np.ndarray:
    """Interpolate a frequency-indexed matrix onto a new frequency grid.

    Parameters
    ----------
    freq_src : array, shape (F_src,)
        Source frequency grid.
    mat : array, shape (F_src, ..., ...)
        Matrix-valued data aligned with freq_src.
    freq_tgt : array, shape (F_tgt,)
        Target frequency grid.
    """

    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    mat = np.asarray(mat)

    # Guard against non-strictly-increasing grids (some generators copy
    # the first positive frequency into the zero bin).
    sort_idx = np.argsort(freq_src)
    freq_sorted = freq_src[sort_idx]
    mat_sorted = mat[sort_idx]
    freq_unique, uniq_idx = np.unique(freq_sorted, return_index=True)
    mat_unique = mat_sorted[uniq_idx]

    if freq_unique.shape == freq_tgt.shape and np.allclose(
        freq_unique, freq_tgt
    ):
        return np.asarray(mat_unique)

    flat = mat_unique.reshape(mat_unique.shape[0], -1)
    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, i].real)
            for i in range(flat.shape[1])
        ]
    ).T
    imag_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, i].imag)
            for i in range(flat.shape[1])
        ]
    ).T
    return (real_interp + 1j * imag_interp).reshape(
        (freq_tgt.size,) + mat_unique.shape[1:]
    )


def compute_effective_Nb(
    Nb: int, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return per-frequency degrees of freedom for Wishart statistics."""

    Nb_val = int(Nb)
    if Nb_val < 1:
        raise ValueError("Nb must be a positive integer")

    if weights is None:
        return np.asarray(Nb_val, dtype=np.float64)

    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr <= 0):
        raise ValueError("weights must be positive")

    return weights_arr * float(Nb_val)


def u_to_wishart_matrix(u: np.ndarray) -> np.ndarray:
    """Convert eigenvector-weighted periodogram components to Wishart matrices."""

    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 3:
        raise ValueError("u must have shape (N, p, p)")

    return np.einsum("fkc,flc->fkl", u, np.conj(u))


def sum_wishart_outer_products(u_stack: np.ndarray) -> np.ndarray:
    """Sum Wishart contributions across a stack of ``U`` matrices."""

    u_stack = np.asarray(u_stack, dtype=np.complex128)
    if u_stack.ndim != 3:
        raise ValueError("u_stack must have shape (n_rep, p, p)")

    return np.einsum("rik,rjk->ij", u_stack, np.conj(u_stack))


def wishart_matrix_to_psd(
    Y: np.ndarray,
    Nb: int,
    *,
    duration: float = 1.0,
    scaling_factor: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert Wishart matrices into one-sided PSD matrices."""

    Y = np.asarray(Y, dtype=np.complex128)
    if Y.ndim != 3:
        raise ValueError("Y must have shape (N, p, p)")

    duration_f = float(duration)
    if duration_f <= 0.0:
        raise ValueError("duration must be positive")

    eff_Nb = compute_effective_Nb(Nb, weights)
    eff_Nb = np.asarray(eff_Nb, dtype=np.float64)
    if eff_Nb.ndim == 0:
        eff_Nb = np.broadcast_to(eff_Nb, (Y.shape[0],))

    if eff_Nb.shape[0] != Y.shape[0]:
        raise ValueError(
            "Effective degrees of freedom must match the frequency dimension"
        )

    psd = Y / (eff_Nb[:, None, None] * duration_f)
    psd *= float(scaling_factor)
    return psd


def wishart_u_to_psd(
    u: np.ndarray,
    Nb: int,
    *,
    duration: float = 1.0,
    scaling_factor: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience wrapper combining :func:`u_to_wishart_matrix` and conversion."""

    Y = u_to_wishart_matrix(u)
    return wishart_matrix_to_psd(
        Y,
        Nb,
        duration=duration,
        scaling_factor=scaling_factor,
        weights=weights,
    )


__all__ = [
    "interp_matrix",
    "compute_effective_Nb",
    "sum_wishart_outer_products",
    "u_to_wishart_matrix",
    "wishart_matrix_to_psd",
    "wishart_u_to_psd",
]
