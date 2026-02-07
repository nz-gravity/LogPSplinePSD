"""Shared PSD conversion helpers for multivariate spectral analysis."""

from __future__ import annotations

import numpy as np

from .._jaxtypes import Complex, Float
from .._typecheck import runtime_typecheck


def _as_positive_int(name: str, value: int) -> int:
    """Validate positive integer inputs used in spectral scaling."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    value_int = int(value)
    if value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


@runtime_typecheck
def interp_matrix(
    freq_src: Float[np.ndarray, "f_src"],
    mat: Complex[np.ndarray, "f_src ..."] | Float[np.ndarray, "f_src ..."],
    freq_tgt: Float[np.ndarray, "f_tgt"],
) -> Complex[np.ndarray, "f_tgt ..."]:
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


@runtime_typecheck
def _interp_complex_matrix(
    freq_src: Float[np.ndarray, "f_src"],
    freq_tgt: Float[np.ndarray, "f_tgt"],
    matrix: Complex[np.ndarray, "f_src ..."] | Float[np.ndarray, "f_src ..."],
) -> Complex[np.ndarray, "f_tgt ..."]:
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


@runtime_typecheck
def u_to_wishart_matrix(
    u: Complex[np.ndarray, "..."] | Float[np.ndarray, "..."],
) -> Complex[np.ndarray, "..."]:
    """Convert eigenvector-weighted periodogram components to Wishart matrices."""

    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 3:
        raise ValueError("u must have shape (N, p, p)")

    return np.einsum("fkc,flc->fkl", u, np.conj(u))


@runtime_typecheck
def sum_wishart_outer_products(
    u_stack: Complex[np.ndarray, "..."] | Float[np.ndarray, "..."],
) -> Complex[np.ndarray, "..."]:
    """Sum Wishart contributions across a stack of ``U`` matrices."""

    u_stack = np.asarray(u_stack, dtype=np.complex128)
    if u_stack.ndim != 3:
        raise ValueError("u_stack must have shape (n_rep, p, p)")

    return np.einsum("rik,rjk->ij", u_stack, np.conj(u_stack))


@runtime_typecheck
def wishart_matrix_to_psd(
    Y: Complex[np.ndarray, "..."] | Float[np.ndarray, "..."],
    Nb: int,
    *,
    duration: float = 1.0,
    scaling_factor: float = 1.0,
    Nh: int = 1,
) -> Complex[np.ndarray, "..."]:
    """Convert Wishart matrices into one-sided PSD matrices."""

    Y = np.asarray(Y, dtype=np.complex128)
    if Y.ndim != 3:
        raise ValueError("Y must have shape (N, p, p)")

    duration_f = float(duration)
    if duration_f <= 0.0:
        raise ValueError("duration must be positive")

    Nb_val = _as_positive_int("Nb", Nb)
    Nh_val = _as_positive_int("Nh", Nh)

    psd = Y / (float(Nb_val * Nh_val) * duration_f)
    psd *= float(scaling_factor)
    return psd


@runtime_typecheck
def wishart_u_to_psd(
    u: Complex[np.ndarray, "..."] | Float[np.ndarray, "..."],
    Nb: int,
    *,
    duration: float = 1.0,
    scaling_factor: float = 1.0,
    Nh: int = 1,
) -> Complex[np.ndarray, "..."]:
    """Convenience wrapper combining :func:`u_to_wishart_matrix` and conversion."""

    Y = u_to_wishart_matrix(u)
    return wishart_matrix_to_psd(
        Y,
        Nb,
        duration=duration,
        scaling_factor=scaling_factor,
        Nh=Nh,
    )


@runtime_typecheck
def _get_coherence(
    psd: Complex[np.ndarray, "..."] | Float[np.ndarray, "..."],
) -> Float[np.ndarray, "..."]:
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


__all__ = [
    "interp_matrix",
    "_interp_complex_matrix",
    "sum_wishart_outer_products",
    "u_to_wishart_matrix",
    "wishart_matrix_to_psd",
    "wishart_u_to_psd",
    "_get_coherence",
]
