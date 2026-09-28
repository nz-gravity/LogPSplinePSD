"""Identifiable frequency mean and centered time-frequency log correction."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

from log_psplines.basis import SplineBasis


def centered_time_basis(
    basis: Array | np.ndarray, penalty: Array | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center a partition-of-unity (T, K) basis on its full grid.

    C has shape (K, K-1); the returned penalty is C.T @ P @ C.
    """
    basis = np.asarray(basis, dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    if basis.ndim != 2 or basis.shape[1] < 2:
        raise ValueError("time basis must have shape (T, K) with K >= 2")
    if penalty.shape != (basis.shape[1], basis.shape[1]):
        raise ValueError("time penalty must have shape (K, K)")
    if not np.allclose(basis.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("time basis must form a partition of unity")
    means = basis.mean(axis=0)
    transform = np.eye(basis.shape[1])[:, :-1] - means[:-1][None, :]
    centered = basis @ transform
    if np.linalg.matrix_rank(centered) != centered.shape[1]:
        raise ValueError("centered time basis must have full column rank")
    return centered, transform, transform.T @ penalty @ transform


@dataclass(frozen=True)
class ANOVALogPSpline:
    """GridTV latent correction u(t,f) = g(f) + eta(t,f).

    The centered time basis is fixed on the full reconstruction grid.
    Reference PSD handling belongs to fitting, not this model.
    """

    frequency: SplineBasis
    time: SplineBasis
    sigma_eta_prior: float = 0.5
    time_basis: np.ndarray = field(init=False, repr=False)
    time_transform: np.ndarray = field(init=False, repr=False)
    time_penalty: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma_eta_prior) or self.sigma_eta_prior <= 0:
            raise ValueError("sigma_eta_prior must be finite and positive")
        centered, transform, penalty = centered_time_basis(
            self.time.basis, self.time.penalty
        )
        object.__setattr__(self, "time_basis", centered)
        object.__setattr__(self, "time_transform", transform)
        object.__setattr__(self, "time_penalty", penalty)

    def components(
        self, weights_g: Array | np.ndarray, weights_eta: Array | np.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return g (F,) and eta (T,F) on the reconstruction grid."""
        if np.shape(weights_g) != (self.frequency.basis.shape[1],):
            raise ValueError("weights_g must have shape (Kf,)")
        expected = (self.time_basis.shape[1], self.frequency.basis.shape[1])
        if np.shape(weights_eta) != expected:
            raise ValueError(f"weights_eta must have shape {expected}")
        bf = jnp.asarray(self.frequency.basis)
        g = bf @ jnp.asarray(weights_g)
        eta = jnp.einsum(
            "ti,ij,fj->tf",
            self.time_basis,
            weights_eta,
            bf,
            optimize="optimal",
        )
        return g, eta

    def __call__(
        self, weights_g: Array | np.ndarray, weights_eta: Array | np.ndarray
    ) -> jnp.ndarray:
        g, eta = self.components(weights_g, weights_eta)
        return g[None, :] + eta
