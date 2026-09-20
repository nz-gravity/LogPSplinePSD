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
    """Scalar log spectrum on a frequency or time-frequency grid.

    time=None: weights (Kf,) -> (F,).
    With time: weights (Kt, Kf) -> (T, F), Bt @ weights @ Bf.T.
    Evaluation is independent of how the weights are fitted.
    """

    frequency: SplineBasis
    time: SplineBasis | None = None
    weights: jax.Array | None = None

    def __post_init__(self) -> None:
        shape = (
            (self.n_basis,)
            if self.time is None
            else (self.time.basis.shape[1], self.n_basis)
        )
        if self.weights is None:
            self.weights = jnp.zeros(shape, dtype=self.basis.dtype)
        else:
            self.weights = jnp.asarray(self.weights)
            if self.time is None and self.weights.ndim != 1:
                raise ValueError("weights must be 1-D")
            if self.weights.shape != shape:
                raise ValueError(
                    f"weights length and shape must match {shape}"
                )

    def __call__(self, weights: jax.Array | None = None) -> jax.Array:
        weights = self.weights if weights is None else weights
        if weights is None:
            raise ValueError("weights must be provided or initialized.")
        if self.time is not None:
            shape = (self.time.basis.shape[1], self.n_basis)
            if weights.shape != shape:
                raise ValueError(f"weights must have shape {shape}")
            # Reuse the WDM contraction without constructing kron(Bt, Bf).
            return jnp.einsum(
                "ti,ij,fj->tf",
                self.time.basis,
                weights,
                self.basis,
                optimize="optimal",
            )
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
