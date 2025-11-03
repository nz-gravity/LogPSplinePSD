"""Shared PSD conversion helpers for multivariate spectral analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np

TWO_PI = 2.0 * np.pi


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
    scaling_factor: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert Wishart matrices into PSD matrices with consistent normalisation."""

    Y = np.asarray(Y, dtype=np.complex128)
    if Y.ndim != 3:
        raise ValueError("Y must have shape (n_freq, n_dim, n_dim)")

    eff_nu = compute_effective_nu(nu, weights)
    eff_nu = np.asarray(eff_nu, dtype=np.float64)
    if eff_nu.ndim == 0:
        eff_nu = np.broadcast_to(eff_nu, (Y.shape[0],))
    if eff_nu.shape[0] != Y.shape[0]:
        raise ValueError(
            "Effective degrees of freedom must match the frequency dimension"
        )

    psd = (2.0 / (eff_nu[:, None, None] * TWO_PI)) * Y
    psd *= float(scaling_factor)
    return psd


def wishart_u_to_psd(
    u: np.ndarray,
    nu: float | np.ndarray,
    *,
    scaling_factor: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience wrapper combining :func:`u_to_wishart_matrix` and conversion."""

    Y = u_to_wishart_matrix(u)
    return wishart_matrix_to_psd(
        Y, nu, scaling_factor=scaling_factor, weights=weights
    )


__all__ = [
    "TWO_PI",
    "compute_effective_nu",
    "sum_wishart_outer_products",
    "u_to_wishart_matrix",
    "wishart_matrix_to_psd",
    "wishart_u_to_psd",
]
