from typing import Tuple

import numpy as np
from jax import numpy as jnp
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis

from log_psplines.datasets import Periodogram


class LogPSplines:
    """Model for log power splines using a B-spline basis and a penalty matrix."""

    def __init__(
        self,
        knots: np.ndarray,
        degree: int,
        diffMatrixOrder: int = 2,
        n: int = None,
    ):
        assert (
            degree > diffMatrixOrder
        ), "Degree must be larger than diffMatrixOrder."
        assert degree in [0, 1, 2, 3, 4, 5], "Degree must be between 0 and 5."
        assert diffMatrixOrder in [
            0,
            1,
            2,
        ], "diffMatrixOrder must be 0, 1, or 2."
        assert len(knots) >= degree, f"#knots: {len(knots)}, degree: {degree}"

        self.knots = knots
        self.degree = degree
        self.order = self.degree + 1
        self.diffMatrixOrder = diffMatrixOrder
        self.n = max(501, 10 * self.n_basis) if n is None else n

        self.basis, self.penalty_matrix = generate_basis_and_penalty_matrix(
            self.knots, self.degree, self.n, self.diffMatrixOrder
        )

    def __repr__(self):
        return f"LogPSplines(knots={self.knots}, degree={self.degree}, n={self.n})"

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    @property
    def n_basis(self) -> int:
        return self.n_knots + self.degree - 1

    def __call__(self, weights: jnp.ndarray = None) -> jnp.ndarray:
        """Compute the weighted sum of the B-spline basis functions minus a constant."""
        weighted_splines = jnp.sum(weights[:, None] * self.basis.T, axis=0)
        return weighted_splines - jnp.log(2 * jnp.pi)


def generate_basis_and_penalty_matrix(
    knots: np.ndarray,
    degree: int,
    n_grid_points: int,
    diffMatrixOrder: int,
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a B-spline basis matrix and penalty matrix.

    Args:
        knots: Array of knots (values between 0 and 1).
        degree: Degree of the B-spline.
        n_grid_points: Number of grid points.
        diffMatrixOrder: Order of the differential operator for regularization.
        epsilon: Small constant for numerical stability.

    Returns:
        A tuple (basis_matrix, penalty_matrix) as JAX arrays.
    """
    order = degree + 1
    basis = BSplineBasis(domain_range=[0, 1], order=order, knots=knots)
    grid_points = np.linspace(0, 1, n_grid_points)
    basis_matrix = (
        basis.to_basis().to_grid(grid_points).data_matrix.squeeze().T
    )

    # Augment knots with boundary values for proper normalization.
    knots_with_boundary = np.concatenate(
        [np.repeat(0, degree), knots, np.repeat(1, degree)]
    )
    n_knots_total = len(knots_with_boundary)
    mid_to_end = knots_with_boundary[degree + 1 :]
    start_to_mid = knots_with_boundary[: (n_knots_total - degree - 1)]
    norm_factor = (mid_to_end - start_to_mid) / (degree + 1)
    norm_factor[norm_factor == 0] = np.inf  # Prevent division by zero.
    basis_matrix = basis_matrix / norm_factor

    # Compute the penalty matrix using L2 regularization.
    regularization = L2Regularization(
        LinearDifferentialOperator(diffMatrixOrder)
    )
    p = regularization.penalty_matrix(basis)
    p = p / np.max(p)
    p = p + epsilon * np.eye(p.shape[1])
    return jnp.array(basis_matrix), jnp.array(p)


def data_peak_knots(
    periodogram: Periodogram,
    n_knots: int,
    frac_uniform: float = 0.0,
    frac_log: float = 0.8,
) -> np.ndarray:
    """Select knots with a mix of uniform, log-spaced, and density-based placement.

             Instead of using a fixed grid (via log‐ or geomspace) to “force” a knot allocation,
         you can let the periodogram’s power distribution guide you. For example,
         you can interpret the (normalized) power as a probability density over frequency,
         compute its cumulative distribution function (CDF), and then choose knots at equally
         spaced quantiles of that CDF. In regions where the power (and hence “spikiness”) is higher,
          the CDF rises faster, so more knots will be allocated there.

        Ensures the first and last knots are at the min and max frequency.
        The remaining knots are allocated:
        - `frac_uniform`: Using uniform spacing.
        - `frac_log`: Using logarithmic spacing.
        - The rest: Using power-based density sampling.

    Args:
            periodogram: Periodogram object with freqs and power.
            n_knots: Total number of knots to select.
            frac_uniform: Fraction of knots to place uniformly (can be 0).
            frac_log: Fraction of knots to place logarithmically (can be 0).

        Returns:
            An array of knot locations (frequencies).
    """
    if n_knots < 2:
        raise ValueError(
            "At least two knots are required (min and max frequencies)."
        )

    min_freq, max_freq = periodogram.freqs[0], periodogram.freqs[-1]

    if n_knots == 2:
        return np.array([min_freq, max_freq])

    # Ensure fractions sum to at most 1
    frac_uniform = max(0.0, min(frac_uniform, 1.0))
    frac_log = max(0.0, min(frac_log, 1.0))
    frac_density = 1.0 - (frac_uniform + frac_log)

    # Compute number of knots in each category
    n_uniform = int(frac_uniform * (n_knots - 2)) if frac_uniform > 0 else 0
    n_log = int(frac_log * (n_knots - 2)) if frac_log > 0 else 0
    n_density = max(0, (n_knots - 2) - (n_uniform + n_log))  # Remaining knots

    # Uniformly spaced knots (excluding min/max)
    uniform_knots = (
        np.linspace(min_freq, max_freq, n_uniform + 2)[1:-1]
        if n_uniform > 0
        else np.array([])
    )

    # Log-spaced knots (excluding min/max)
    log_knots = (
        np.logspace(np.log10(min_freq), np.log10(max_freq), n_log + 2)[1:-1]
        if n_log > 0
        else np.array([])
    )

    # Power-based density sampling
    density_knots = np.array([])
    if n_density > 0:
        power = periodogram.power.copy()
        density = power / np.sum(power)
        cdf = np.cumsum(density)

        # Compute quantiles for density-based knots
        quantiles = np.linspace(0, 1, n_density + 2)[1:-1]
        density_knots = np.interp(quantiles, cdf, periodogram.freqs)

    # Combine and sort
    knots = np.concatenate(
        ([min_freq], uniform_knots, log_knots, density_knots, [max_freq])
    )
    knots = np.sort(knots)  # Ensure order

    # normalize to [0, 1]
    knots = (knots - min_freq) / (max_freq - min_freq)
    print(f"Selected knots: {knots}")
    return knots
