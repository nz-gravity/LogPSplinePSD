"""Exact banded SPD algebra for Gaussian coefficient marginalization.

Storage is ``a[column, offset] = A[column + offset, column]``; offset zero
contains the diagonal. Entries below the last matrix row must be zero.
Scans carry only a bandwidth-sized window: factorization and selected
inversion cost O(n * bandwidth**2), and a triangular solve O(n * bandwidth).
No dense covariance, tolerance-based truncation, or extra ridge is used.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cholesky_banded
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


class BandedPlan(NamedTuple):
    """Fixed ordering, storage positions, and prior operators for one row."""

    permutation: jax.Array
    inverse_permutation: jax.Array
    rows: jax.Array
    penalty: jax.Array
    penalty_factor: jax.Array


def prepare_plan(design_re, design_im, penalty, *, force=False):
    """Find a band containing every possible precision entry, without cuts.

    The structural graph includes every overlapping design column and every
    nonzero prior entry, even when a particular weighted sum cancels to zero.
    A permutation changes coordinates only; it cannot change the Gaussian.
    Auto selection is conservative about both size and bandwidth.
    """
    n = penalty.shape[0]
    if not force and (n < 256 or jax.default_backend() != "cpu"):
        return None
    support = csr_matrix(np.any((design_re != 0) | (design_im != 0), axis=1))
    pattern = support.T @ support + csr_matrix(penalty != 0)
    permutation = reverse_cuthill_mckee(pattern, symmetric_mode=True)
    inverse = np.argsort(permutation)
    row, col = pattern.nonzero()
    width = int(np.max(np.abs(inverse[row] - inverse[col]), initial=0))
    # CPU benchmarks favor dense BLAS for small or wide bands. Device
    # backends remain dense in auto mode until measured there separately.
    if not force and 8 * width >= n:
        return None
    rows = np.arange(n)[:, None] + np.arange(width + 1)
    valid = rows < n
    rows = np.minimum(rows, n - 1)
    prior = penalty[permutation[rows], permutation[:, None]] * valid
    factor = cholesky_banded(prior.T, lower=True, check_finite=False).T
    return BandedPlan(
        *(jnp.asarray(v) for v in (permutation, inverse, rows, prior, factor))
    )


def cholesky(a):
    """Lower Cholesky factor in the same band storage as the SPD input."""
    width = a.shape[1] - 1
    if width == 0:
        return jnp.sqrt(a)

    def step(updates, column):
        # Previous columns affect the next width rows. The new bottom row
        # has no predecessor within the band, so its accumulated update is 0.
        residual = column - jnp.pad(updates[:, 0], (0, 1))
        factor = residual / jnp.sqrt(residual[0])
        following = jnp.pad(updates[1:, 1:], ((0, 1), (0, 1)))
        following += jnp.outer(factor[1:], factor[1:])
        return following, factor

    return jax.lax.scan(step, jnp.zeros((width, width), a.dtype), a)[1]


def solve_triangular(factor, rhs, *, transpose=False):
    """Solve L x = rhs or L.T x = rhs for a vector and banded lower L."""
    width = factor.shape[1] - 1
    if width == 0:
        return rhs / factor[:, 0]

    def step(window, inputs):
        column, value = inputs
        if transpose:
            solved = (value - column[1:] @ window) / column[0]
            following = jnp.concatenate((solved[None], window[:-1]))
        else:
            solved = (value - window[0]) / column[0]
            following = jnp.pad(window[1:], (0, 1)) + column[1:] * solved
        return following, solved

    return jax.lax.scan(
        step,
        jnp.zeros(width, rhs.dtype),
        (factor, rhs),
        reverse=transpose,
    )[1]


def inverse_subset(factor):
    """Return the exact entries of (L L.T)^-1 inside L's band.

    Takahashi recursion: a reverse scan needs only the following square
    window of the inverse. The full covariance is generally dense, but its
    off-band entries are unnecessary for trace(H^-1 dH) when dH is banded.
    See Durrande et al. (2019), https://proceedings.mlr.press/v89/durrande19a.html.
    """
    width = factor.shape[1] - 1
    if width == 0:
        return 1 / factor**2

    def step(window, column):
        ratio = column[1:] / column[0]
        off_diagonal = -(window @ ratio)
        diagonal = 1 / column[0] ** 2 - ratio @ off_diagonal
        first = jnp.concatenate((diagonal[None], off_diagonal[:-1]))
        following = jnp.concatenate(
            (
                first[None, :],
                jnp.concatenate(
                    (off_diagonal[:-1, None], window[:-1, :-1]), axis=1
                ),
            ),
            axis=0,
        )
        return following, jnp.concatenate((diagonal[None], off_diagonal))

    return jax.lax.scan(
        step,
        jnp.zeros((width, width), factor.dtype),
        factor,
        reverse=True,
    )[1]


@jax.custom_jvp
def logdet_from_factor(precision, factor, scale):
    """Log determinant of H described by a factor of D^-1 H D^-1."""
    return 2 * jnp.sum(jnp.log(factor[:, 0]) + jnp.log(scale))


@logdet_from_factor.defjvp
def _logdet_jvp(primals, tangents):
    _, factor, scale = primals
    precision_tangent, _, _ = tangents
    width = factor.shape[1] - 1
    rows = jnp.arange(factor.shape[0])[:, None] + jnp.arange(width + 1)
    denominator = (
        scale[:, None] * jnp.pad(scale, (0, width), constant_values=1)[rows]
    )
    covariance = inverse_subset(factor) / denominator
    # Off-diagonal entries occur twice in the symmetric trace contraction.
    multiplicity = jnp.where(jnp.arange(width + 1) == 0, 1, 2)
    derivative = jnp.sum(covariance * precision_tangent * multiplicity)
    return logdet_from_factor(*primals), derivative
