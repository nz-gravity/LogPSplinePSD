"""One-dimensional B-splines and normalized integrated-derivative penalties."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis


@dataclass(frozen=True)
class SplineBasis:
    """Mathematical basis on a normalized frequency or time grid.

    basis: (G, K), penalty: (K, K). No fitted state or inference machinery.
    The penalty is an integrated derivative, not a finite difference penalty.
    """

    grid: np.ndarray
    knots: np.ndarray
    basis: jnp.ndarray
    penalty: jnp.ndarray
    degree: int = 3
    penalty_order: int = 2

    def __post_init__(self) -> None:
        if self.basis.ndim != 2 or self.basis.shape[0] != len(self.grid):
            raise ValueError("basis must have shape (len(grid), K)")
        if self.penalty.shape != (self.basis.shape[1], self.basis.shape[1]):
            raise ValueError("penalty must have shape (K, K)")

    @classmethod
    def from_knots(
        cls,
        grid: np.ndarray,
        knots: np.ndarray,
        degree: int = 3,
        penalty_order: int = 2,
    ) -> "SplineBasis":
        return cls.create(
            grid=grid,
            knots=knots,
            degree=degree,
            penalty_order=penalty_order,
            n=len(grid),
        )

    @classmethod
    def create(
        cls,
        *,
        degree: int,
        penalty_order: int,
        n: int,
        knots: np.ndarray,
        basis: jnp.ndarray | None = None,
        penalty: jnp.ndarray | None = None,
        grid: np.ndarray | None = None,
    ) -> "SplineBasis":
        """Construct from knots, or validate supplied linear operators."""
        if degree < penalty_order:
            raise ValueError(
                f"Degree ({degree}) must be ≥ penalty_order ({penalty_order}) "
                "for mathematically well-defined penalty matrix."
            )
        if degree not in [0, 1, 2, 3, 4, 5]:
            raise ValueError(
                f"Degree must be between 0 and 5, got {degree}. "
                "Higher degrees may cause numerical instability."
            )
        if penalty_order not in [0, 1, 2, 3, 4]:
            raise ValueError(
                f"penalty_order must be between 0 and 4, got {penalty_order}."
            )
        if len(knots) < degree:
            raise ValueError(
                f"Number of knots ({len(knots)}) must be ≥ degree ({degree}) "
                "for well-defined B-spline basis."
            )

        knots = np.asarray(knots, dtype=np.float64)
        if knots.ndim != 1:
            raise ValueError(f"knots must be 1-D, got shape {knots.shape}")
        if knots.size == 0:
            raise ValueError("knots must be non-empty")
        if not np.all(np.isfinite(knots)):
            raise ValueError("knots must be finite")
        if np.any(np.diff(knots) < 0):
            raise ValueError("knots must be sorted ascending")

        if grid is None:
            grid = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
        else:
            grid = np.asarray(grid, dtype=np.float64)
            if grid.ndim != 1:
                raise ValueError(
                    f"grid must be 1-D with length n, got shape {grid.shape}"
                )
            if grid.shape[0] != int(n):
                raise ValueError(
                    f"grid length must match n={n}, got {grid.shape[0]}"
                )
            if not np.all(np.isfinite(grid)):
                raise ValueError("grid must be finite")
            if np.any(np.diff(grid) < 0):
                raise ValueError("grid must be sorted ascending")

        if basis is None or penalty is None:
            basis, penalty = init_basis_and_penalty(
                knots,
                degree,
                n,
                penalty_order,
                grid_points=grid,
            )

        basis = jnp.asarray(basis)
        penalty = jnp.asarray(penalty)
        if basis.ndim != 2:
            raise ValueError(
                f"basis must be 2-D (n, n_basis), got shape {basis.shape}"
            )
        if basis.shape[0] != int(n):
            raise ValueError(
                f"basis first dimension must match n={n}, got {basis.shape[0]}"
            )
        if penalty.ndim != 2:
            raise ValueError(f"penalty must be 2-D, got shape {penalty.shape}")
        if penalty.shape[0] != penalty.shape[1]:
            raise ValueError("penalty must be square")
        if penalty.shape[0] != basis.shape[1]:
            raise ValueError(
                "penalty dimension must match basis n_basis: "
                f"{penalty.shape[0]} vs {basis.shape[1]}"
            )

        return cls(grid, knots, basis, penalty, degree, penalty_order)


def init_basis_and_penalty(
    knots: np.ndarray,
    degree: int,
    n_grid_points: int,
    diff_matrix_order: int,
    epsilon: float = 1e-6,
    grid_points: np.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate B-spline basis matrix and penalty matrix.

    Parameters
    ----------
    knots : np.ndarray
        Array of knots (values between 0 and 1)
    degree : int
        Degree of the B-spline
    n_grid_points : int
        Number of grid points. Ignored if `grid_points` is provided.
    diff_matrix_order : int
        Order of the differential operator for regularization
    epsilon : float, default=1e-6
        Small constant for numerical stability
    grid_points : np.ndarray, optional
        Locations in [0, 1] at which to evaluate the basis. If None,
        uses a uniform grid of length `n_grid_points`.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        (basis_matrix, penalty_matrix) as JAX arrays
    """
    order = degree + 1
    basis = BSplineBasis(
        domain_range=[0, 1], order=order, knots=np.asarray(knots).tolist()
    )
    if grid_points is None:
        grid_points = np.linspace(0, 1, n_grid_points)
    else:
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.ndim != 1:
            raise ValueError("grid_points must be 1-D if provided")
        if grid_points.size != n_grid_points:
            raise ValueError("grid_points length must match n_grid_points")
        # Clip to [0,1] for numerical safety
        grid_points = np.clip(grid_points, 0.0, 1.0)

    # Compute basis matrix and keep it explicitly 2-D (n_grid, n_basis)
    basis_eval = basis.to_basis().to_grid(grid_points).data_matrix
    basis_eval = np.asarray(basis_eval, dtype=np.float64)
    basis_matrix_np = np.squeeze(basis_eval, axis=-1).T
    basis_matrix = jnp.asarray(basis_matrix_np)

    # Compute penalty matrix using L2 regularization
    regularization = L2Regularization(
        LinearDifferentialOperator(diff_matrix_order)
    )
    penalty_matrix = regularization.penalty_matrix(basis)
    penalty_matrix = penalty_matrix / np.max(penalty_matrix)
    penalty_matrix = penalty_matrix + epsilon * np.eye(penalty_matrix.shape[1])

    return basis_matrix, jnp.asarray(penalty_matrix)
