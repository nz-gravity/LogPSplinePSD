"""Whittle/Wishart log-likelihood utilities for the modified Cholesky PSD model.

This module is intentionally explicit about a common (and important) confusion:

    A sum over eigenvectors is not a factorisation over independent data;
    factorisation requires parameter separation, which occurs only after
    the Cholesky decomposition.

If ``Y`` is a Hermitian periodogram/Wishart matrix and ``S`` is the (unknown)
spectral matrix, the multivariate Whittle likelihood contains the term

    tr(S^{-1} Y).

Writing an eigendecomposition (or any rank decomposition) of ``Y`` as
``Y = Σ_v u_v u_v^H`` gives the algebraic identity

    tr(S^{-1} Y) = Σ_v u_v^H S^{-1} u_v,

but *each* quadratic form depends on *all* parameters in ``S^{-1}``. The sum is
therefore a computational identity, not a statistical factorisation.

In contrast, under the modified Cholesky parameterisation

    S^{-1} = T^H D^{-1} T,

with unit lower-triangular ``T`` (complex off-diagonals ``θ_{jl}``) and
``D = diag(δ_1^2, …, δ_p^2)``, we have

    tr(S^{-1} Y) = tr(D^{-1} T Y T^H)
                = Σ_{j=1..p} (T Y T^H)_{jj} / δ_j^2.

Crucially, ``(T Y T^H)_{jj}`` depends only on the parameters in the j-th row of
``T`` (i.e. ``θ_{j,<j}``) and on the data ``Y``. This is the parameter
separation that yields a likelihood factorisation across Cholesky rows.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

Array = jnp.ndarray


def unit_lower_from_theta(theta: Array) -> Array:
    """Build the unit lower-triangular Cholesky factor ``T`` from ``θ``.

    Parameters
    ----------
    theta
        Complex (or real) array with shape ``(..., p, p)`` where the strictly
        lower-triangular entries contain ``θ_{jl}`` and all other entries are
        ignored.

    Returns
    -------
    T
        Unit lower-triangular array with shape ``(..., p, p)`` such that
        ``T[j, l] = -θ_{jl}`` for ``j > l`` and diagonal entries are 1.
    """
    if theta.shape[-1] != theta.shape[-2]:
        raise ValueError(f"theta must be square, got shape {theta.shape}.")

    p = theta.shape[-1]
    eye = jnp.eye(p, dtype=theta.dtype)
    if theta.ndim > 2:
        eye = jnp.broadcast_to(eye, theta.shape)

    return eye - jnp.tril(theta, k=-1)


def whittle_loglik_cholesky_blocks(
    Y: Array,
    log_delta_sq: Array,
    theta: Array,
    *,
    nu: Array | float = 1.0,
    weights: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Compute row-wise Whittle/Wishart log-likelihood contributions.

    This returns the (constant-free) log-likelihood:

        log L(S | Y) = -nu * Σ_j log(δ_j^2) - tr(S^{-1} Y)

    where ``S^{-1} = T^H D^{-1} T`` and ``D = diag(δ_1^2, …, δ_p^2)``.

    Parameters
    ----------
    Y
        Hermitian matrix (or batch of matrices) with shape ``(..., p, p)``.
    log_delta_sq
        Log diagonal variances with shape ``(..., p)`` representing
        ``log(δ_j^2)``. This parameterisation enforces ``δ_j^2 > 0``.
    theta
        Strictly lower-triangular entries ``θ_{jl}`` with shape ``(..., p, p)``.
        The diagonal and upper triangle are ignored.
    nu
        Degrees of freedom scaling for the determinant term. Can be a scalar or
        broadcastable to the leading shape of ``Y``.
    weights
        Optional nonnegative weights broadcastable to the leading shape of ``Y``
        (e.g. frequency quadrature weights). If provided, each per-row term is
        multiplied by ``weights`` before summation.

    Returns
    -------
    loglik_total
        Scalar total log-likelihood (sum over rows and any batch dimensions).
    loglik_blocks
        Per-row contributions with shape ``(..., p)`` (before summing over the
        leading batch dimensions).
    """
    if Y.shape[-1] != Y.shape[-2]:
        raise ValueError(f"Y must be square, got shape {Y.shape}.")
    if log_delta_sq.shape[-1] != Y.shape[-1]:
        raise ValueError(
            "log_delta_sq must have trailing dimension p matching Y, "
            f"got {log_delta_sq.shape} vs Y {Y.shape}."
        )
    if theta.shape[-2:] != Y.shape[-2:]:
        raise ValueError(
            "theta must have trailing shape (p, p) matching Y, "
            f"got {theta.shape} vs Y {Y.shape}."
        )

    T = unit_lower_from_theta(theta)

    # Prefer computing T Y T^H directly and reading off diagonal entries.
    TY = jnp.matmul(T, Y)
    TYT_H = jnp.matmul(TY, jnp.swapaxes(jnp.conj(T), -1, -2))
    # Hermitian -> diagonal is real; take real part to remove numerical noise.
    diag_power = jnp.real(jnp.diagonal(TYT_H, axis1=-2, axis2=-1))

    inv_delta_sq = jnp.exp(-log_delta_sq)
    loglik_blocks = -jnp.asarray(nu) * log_delta_sq - diag_power * inv_delta_sq

    if weights is not None:
        w = jnp.asarray(weights, dtype=loglik_blocks.dtype)
        loglik_blocks = loglik_blocks * w[..., None]

    loglik_total = jnp.sum(loglik_blocks)
    return loglik_total, loglik_blocks


__all__ = ["unit_lower_from_theta", "whittle_loglik_cholesky_blocks"]
