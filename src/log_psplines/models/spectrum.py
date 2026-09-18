"""Scalar spectral components shared by all channel counts."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from log_psplines.basis import SplineBasis


@jax.jit
def build_spline(basis: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.einsum("ij,j->i", basis, weights)


@dataclass
class LogPSpline:
    """Stationary log S(f) = Bf @ weights.

    time=None: weights (Kf,) -> (F,).
    Future time basis: weights (Kt, Kf) -> (T, F), Bt @ weights @ Bf.T.
    Time-dependent evaluation and priors are deliberately not implemented.
    """

    frequency: SplineBasis
    time: SplineBasis | None = None
    weights: jax.Array | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = jnp.zeros(self.n_basis, dtype=self.basis.dtype)
        else:
            self.weights = jnp.asarray(self.weights)
            if self.time is None and self.weights.ndim != 1:
                raise ValueError("weights must be 1-D")
            if self.time is None and self.weights.shape != (self.n_basis,):
                raise ValueError("weights length must match basis n_basis")

    def __call__(self, weights: jax.Array | None = None) -> jax.Array:
        if self.time is not None:
            raise NotImplementedError(
                "Time-dependent spline evaluation is reserved for a future implementation"
            )
        weights = self.weights if weights is None else weights
        if weights is None:
            raise ValueError("weights must be provided or initialized.")
        if weights.shape != (self.n_basis,):
            raise ValueError("weights must have shape (Kf,)")
        return build_spline(self.frequency.basis, weights)

    @property
    def basis(self) -> jax.Array:
        return self.frequency.basis

    @property
    def penalty_matrix(self) -> jax.Array:
        return self.frequency.penalty

    @property
    def knots(self) -> np.ndarray:
        return self.frequency.knots

    @property
    def grid_points(self) -> np.ndarray:
        return self.frequency.grid

    @property
    def degree(self) -> int:
        return self.frequency.degree

    @property
    def diffMatrixOrder(self) -> int:
        return self.frequency.penalty_order

    @property
    def n(self) -> int:
        return self.basis.shape[0]

    @property
    def n_basis(self) -> int:
        return self.basis.shape[1]

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    @property
    def order(self) -> int:
        return self.degree + 1
