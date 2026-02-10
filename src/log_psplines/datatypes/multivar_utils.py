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


def _interp_frequency_indexed_array(
    freq_src: np.ndarray,
    freq_tgt: np.ndarray,
    values: np.ndarray,
    *,
    sort_and_dedup: bool = True,
    force_complex_output: bool = False,
) -> np.ndarray:
    """Interpolate values indexed by frequency along axis 0."""
    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    values = np.asarray(values)

    if freq_src.ndim != 1:
        raise ValueError("freq_src must be a 1-D array")
    if freq_tgt.ndim != 1:
        raise ValueError("freq_tgt must be a 1-D array")
    if values.ndim < 1:
        raise ValueError("values must be at least 1-D")
    if values.shape[0] != freq_src.size:
        raise ValueError("values and freq_src must have matching lengths")
    if freq_src.size == 0:
        raise ValueError("freq_src must be non-empty")

    if sort_and_dedup:
        sort_idx = np.argsort(freq_src)
        freq_src = freq_src[sort_idx]
        values = values[sort_idx]
        freq_src, uniq_idx = np.unique(freq_src, return_index=True)
        values = values[uniq_idx]

    if freq_src.shape == freq_tgt.shape and np.allclose(freq_src, freq_tgt):
        return np.asarray(values)

    flat = values.reshape(values.shape[0], -1)
    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_src, flat[:, i].real)
            for i in range(flat.shape[1])
        ]
    ).T

    if np.iscomplexobj(values) or force_complex_output:
        imag_interp = np.vstack(
            [
                np.interp(freq_tgt, freq_src, flat[:, i].imag)
                for i in range(flat.shape[1])
            ]
        ).T
        res = real_interp + 1j * imag_interp
    else:
        res = real_interp

    return res.reshape((freq_tgt.size,) + values.shape[1:])


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

    return _interp_frequency_indexed_array(
        freq_src,
        freq_tgt,
        mat,
        sort_and_dedup=True,
        force_complex_output=True,
    )


@runtime_typecheck
def _interp_complex_matrix(
    freq_src: Float[np.ndarray, "f_src"],
    freq_tgt: Float[np.ndarray, "f_tgt"],
    matrix: Complex[np.ndarray, "f_src ..."] | Float[np.ndarray, "f_src ..."],
) -> Complex[np.ndarray, "f_tgt ..."]:
    """Linearly interpolate a complex-valued matrix along the frequency axis."""
    return _interp_frequency_indexed_array(
        freq_src,
        freq_tgt,
        matrix,
        sort_and_dedup=True,
        force_complex_output=True,
    )


@runtime_typecheck
def u_to_wishart_matrix(
    u: Complex[np.ndarray, "..."] | Float[np.ndarray, "..."],
) -> Complex[np.ndarray, "..."]:
    """Convert eigenvector-weighted periodogram components to Wishart matrices.

    Eg:
    u = [[u11, u12], [u21, u22], [u31, u32]]
    -> Y = [[sum_k u1k*conj(u1k), sum_k u1k*conj(u2k)], [sum_k u2k*conj(u1k), sum_k u2k*conj(u2k)]]
    """

    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 3:
        raise ValueError("u must have shape (N, p, p)")

    return np.einsum("fkc,flc->fkl", u, np.conj(u))


@runtime_typecheck
def sum_wishart_outer_products(
    u_stack: Complex[np.ndarray, "..."] | Float[np.ndarray, "..."],
) -> Complex[np.ndarray, "..."]:
    """Sum Wishart contributions across a stack of ``U`` matrices.

    This is equivalent to summing the outer products of the rows of each U matrix:
    out[i, j] = sum_{r} sum_{k} u_stack[r, i, k] * conj(u_stack[r, j, k])

    Eg:
    [[U1], [U2], [U3]] -> U1^H @ U1 + U2^H @ U2 + U3^H @ U3

    """

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
    """Convert Wishart matrices into one-sided PSD matrices.

    Eg:
    Y = U^H @ U, Nb=10, Nh=2, duration=1.0 -> psd = Y / (10 * 2 * 1.0)

    """

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
    """Compute coherence matrices from PSD estimates.

    Cxy = |Pxy|^2 / (Pxx * Pyy) with diagonal elements set to 1.0 and clipped to [0, 1].

    """
    psd = np.asarray(psd)
    if psd.ndim != 3:
        raise ValueError("psd must have shape (N, p, p)")
    N, p, q = psd.shape
    if p != q:
        raise ValueError("psd must have square channel dimensions")

    coh = np.zeros((N, p, p), dtype=np.float64)
    for i in range(p):
        coh[:, i, i] = 1.0
        for j in range(i + 1, p):
            denom = np.abs(psd[:, i, i]) * np.abs(psd[:, j, j])
            with np.errstate(divide="ignore", invalid="ignore"):
                coh_ij = np.where(
                    denom > 0, (np.abs(psd[:, i, j]) ** 2) / denom, 0.0
                )
            coh[:, i, j] = np.clip(coh_ij.real, a_min=0.0, a_max=1.0)
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
