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


def strain_to_freq_psd_scale(
    freq: np.ndarray,
    *,
    laser_freq: float,
    arm_length: float,
    c_light: float = 299_792_458.0,
) -> np.ndarray:
    """Return scale factor to convert strain PSD to frequency PSD."""

    freq = np.asarray(freq, dtype=float)
    return (2.0 * np.pi * freq * laser_freq * arm_length / c_light) ** 2


def resolve_psd_plot_units(
    base_psd_units: str,
    plot_psd_units: str,
    *,
    laser_freq: float,
    arm_length: float,
    c_light: float = 299_792_458.0,
) -> Tuple[Callable[[np.ndarray], np.ndarray] | None, str]:
    """Return (scale_fn, unit_label) for plotting PSDs."""

    base_psd_units = str(base_psd_units).lower().strip()
    plot_psd_units = str(plot_psd_units).lower().strip()
    if plot_psd_units not in {"freq", "strain"}:
        raise ValueError(
            f"plot_psd_units must be 'freq' or 'strain', got {plot_psd_units!r}."
        )
    if base_psd_units not in {"freq", "strain"}:
        raise ValueError(
            f"base_psd_units must be 'freq' or 'strain', got {base_psd_units!r}."
        )

    unit_label = "Hz^2/Hz" if plot_psd_units == "freq" else "1/Hz"
    if plot_psd_units == base_psd_units:
        return None, unit_label
    if base_psd_units == "strain" and plot_psd_units == "freq":
        return (
            lambda f: strain_to_freq_psd_scale(
                f,
                laser_freq=laser_freq,
                arm_length=arm_length,
                c_light=c_light,
            ),
            unit_label,
        )
    raise NotImplementedError(
        "Only strain→freq conversion is supported for plotting."
    )


def compute_effective_nu(
    nu: float | np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return per-frequency degrees of freedom for Wishart statistics."""

    nu_arr = np.asarray(nu, dtype=np.float64)
    if nu_arr.ndim > 1:
        raise ValueError("nu must be scalar or 1-D array")

    if weights is None:
        return nu_arr

    weights_arr = np.asarray(weights, dtype=np.float64)
    if np.any(weights_arr <= 0):
        raise ValueError("weights must be positive")

    if nu_arr.ndim == 0:
        return weights_arr * float(nu_arr)

    if nu_arr.shape != weights_arr.shape:
        raise ValueError("weights must have the same shape as nu")

    return weights_arr * nu_arr


def u_to_wishart_matrix(u: np.ndarray) -> np.ndarray:
    """Convert eigenvector-weighted periodogram components to Wishart matrices."""

    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 3:
        raise ValueError("u must have shape (n_freq, n_dim, n_dim)")

    return np.einsum("fkc,flc->fkl", u, np.conj(u))


def sum_wishart_outer_products(u_stack: np.ndarray) -> np.ndarray:
    """Sum Wishart contributions across a stack of ``U`` matrices."""

    u_stack = np.asarray(u_stack, dtype=np.complex128)
    if u_stack.ndim != 3:
        raise ValueError("u_stack must have shape (n_rep, n_dim, n_dim)")

    return np.einsum("rik,rjk->ij", u_stack, np.conj(u_stack))


def wishart_matrix_to_psd(
    Y: np.ndarray,
    nu: float | np.ndarray,
    *,
    duration: float = 1.0,
    scaling_factor: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert Wishart matrices into one-sided PSD matrices."""

    Y = np.asarray(Y, dtype=np.complex128)
    if Y.ndim != 3:
        raise ValueError("Y must have shape (n_freq, n_dim, n_dim)")

    duration_f = float(duration)
    if duration_f <= 0.0:
        raise ValueError("duration must be positive")

    eff_nu = compute_effective_nu(nu, weights)
    eff_nu = np.asarray(eff_nu, dtype=np.float64)
    if eff_nu.ndim == 0:
        eff_nu = np.broadcast_to(eff_nu, (Y.shape[0],))

    if eff_nu.shape[0] != Y.shape[0]:
        raise ValueError(
            "Effective degrees of freedom must match the frequency dimension"
        )

    psd = Y / (eff_nu[:, None, None] * duration_f)
    psd *= float(scaling_factor)
    return psd


def wishart_u_to_psd(
    u: np.ndarray,
    nu: float | np.ndarray,
    *,
    duration: float = 1.0,
    scaling_factor: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience wrapper combining :func:`u_to_wishart_matrix` and conversion."""

    Y = u_to_wishart_matrix(u)
    return wishart_matrix_to_psd(
        Y,
        nu,
        duration=duration,
        scaling_factor=scaling_factor,
        weights=weights,
    )


__all__ = [
    "interp_matrix",
    "compute_effective_nu",
    "sum_wishart_outer_products",
    "strain_to_freq_psd_scale",
    "resolve_psd_plot_units",
    "u_to_wishart_matrix",
    "wishart_matrix_to_psd",
    "wishart_u_to_psd",
]
