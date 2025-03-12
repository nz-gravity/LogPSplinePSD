import numpy as np
from jax import numpy as jnp
import jax

from .datasets import Periodogram
from .initialisation import init_basis_and_penalty, init_weights, init_knots

from dataclasses import dataclass


@dataclass
class LogPSplines:
    """Model for log power splines using a B-spline basis and a penalty matrix."""

    degree: int
    diffMatrixOrder: int
    n: int
    basis: jnp.ndarray
    penalty_matrix: jnp.ndarray
    knots: np.ndarray
    weights: jnp.ndarray
    L: jnp.ndarray

    def __post_init__(self):
        if self.degree < self.diffMatrixOrder:
            raise ValueError("Degree must be larger than diffMatrixOrder.")
        if self.degree not in [0, 1, 2, 3, 4, 5]:
            raise ValueError("Degree must be between 0 and 5.")
        if self.diffMatrixOrder not in [0, 1, 2]:
            raise ValueError("diffMatrixOrder must be 0, 1, or 2.")
        if self.n_knots < self.degree:
            raise ValueError(f"#knots: {self.n_knots}, degree: {self.degree}")

    def __repr__(self):
        return f"LogPSplines(knots={self.knots}, degree={self.degree}, n={self.n})"

    @classmethod
    def from_periodogram(
            cls,
            periodogram: Periodogram,
            n_knots: int,
            degree: int,
            diffMatrixOrder: int = 2,
            knot_kwargs: dict = {},
    ):
        knots = init_knots(periodogram, n_knots, **knot_kwargs)
        basis, penalty_matrix = init_basis_and_penalty(
            knots, degree, periodogram.n, diffMatrixOrder
        )
        model = cls(
            knots=knots,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            n=periodogram.n,
            basis=basis,
            penalty_matrix=penalty_matrix,
            weights=jnp.zeros(basis.shape[1]),
            L=cholesky(penalty_matrix, lower=True),
        )
        weights = init_weights(jnp.log(periodogram.power), model)
        model.weights = weights
        return model

    @property
    def order(self) -> int:
        return self.degree + 1

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    @property
    def n_basis(self) -> int:
        return self.n_knots + self.degree - 1

    def __call__(self, weights: jnp.ndarray = None) -> jnp.ndarray:
        """Compute the weighted sum of the B-spline basis functions minus a constant."""
        if weights is None:
            weights = self.weights
        return build_spline(self.basis, weights)



@jax.jit
def build_spline(ln_spline_basis:jnp.ndarray, weights:jnp.ndarray):
    return jnp.sum(weights[:, None] * ln_spline_basis.T, axis=0)