from dataclasses import dataclass, field
from typing import (
    Literal,
    cast,
)

import jax.numpy as jnp
import numpy as np

from log_psplines.models.spectrum import LogPSpline


@dataclass(frozen=True)
class MultivarComponentKey:
    """Stable identifier for one multivariate spline component."""

    family: Literal["delta", "theta"]
    j: int
    l: int | None = None
    part: Literal["re", "im"] | None = None

    def __post_init__(self):
        if int(self.j) < 0:
            raise ValueError(f"j must be >= 0, got {self.j}")
        if self.family == "delta":
            if self.l is not None or self.part is not None:
                raise ValueError(
                    "delta components must have l=None and part=None"
                )
            return

        if self.family != "theta":
            raise ValueError(
                f"Unknown component family '{self.family}'. "
                "Use 'delta' or 'theta'."
            )
        if self.l is None:
            raise ValueError("theta components require l")
        if self.part not in ("re", "im"):
            raise ValueError("theta components require part in {'re','im'}")
        if not (0 <= int(self.l) < int(self.j)):
            raise ValueError(
                f"Invalid theta index (j={self.j}, l={self.l}); "
                "expected 0 <= l < j."
            )

    @property
    def name(self) -> str:
        if self.family == "delta":
            return f"delta_{self.j}"
        assert self.l is not None and self.part is not None
        return f"theta_{self.part}_{self.j}_{self.l}"


@dataclass
class SpectralComponents:
    """
    Prepared scalar spline components for stationary matrix inference.

    Uses Cholesky decomposition: S(f) = T^(-1) D T^(-H) where:
    - D is diagonal matrix with exp(log_delta_sq) elements (one P-spline per channel)
    - T is lower triangular with -theta terms (separate P-splines for real/imaginary parts)

    This enables flexible modeling of both auto-spectra and cross-spectra while
    ensuring positive definiteness of the estimated PSD matrix.

    Parameters
    ----------
    degree : int
        Polynomial degree of B-spline basis functions
    diffMatrixOrder : int
        Order of the integrated-derivative penalty
    N : int
        Number of frequency bins
    p : int
        Number of channels in multivariate data
    diagonal_models : List[LogPSpline]
        P-spline models for diagonal PSD components (one per channel)
    offdiag_re_models : Dict[Tuple[int, int], LogPSpline], optional
        P-spline models for real parts of off-diagonal terms keyed by
        ``(j, l)`` with ``j > l``.
    offdiag_im_models : Dict[Tuple[int, int], LogPSpline], optional
        P-spline models for imaginary parts of off-diagonal terms keyed by
        ``(j, l)`` with ``j > l``.
    """

    degree: int
    diffMatrixOrder: int
    N: int
    p: int

    # P-spline components for each Cholesky parameter
    diagonal_models: list[LogPSpline]  # One per channel
    offdiag_re_models: dict[tuple[int, int], LogPSpline] = field(
        default_factory=dict
    )
    offdiag_im_models: dict[tuple[int, int], LogPSpline] = field(
        default_factory=dict
    )
    component_scores: dict[MultivarComponentKey, np.ndarray] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        """Validate multivariate model parameters."""
        if len(self.diagonal_models) != self.p:
            raise ValueError(
                f"Number of diagonal models ({len(self.diagonal_models)}) "
                f"must match number of channels ({self.p})"
            )

        if self.n_theta == 0:
            self.offdiag_re_models = {}
            self.offdiag_im_models = {}
            return

        pairs = self.theta_pairs
        missing_re = [
            pair for pair in pairs if pair not in self.offdiag_re_models
        ]
        missing_im = [
            pair for pair in pairs if pair not in self.offdiag_im_models
        ]
        if missing_re or missing_im:
            raise ValueError(
                "Per-component off-diagonal models are incomplete. "
                f"Missing real models for pairs={missing_re}, "
                f"missing imag models for pairs={missing_im}."
            )

    @property
    def n_knots(self) -> int | list[list[int]]:
        """Knot counts per Cholesky entry, or one int when all are equal."""
        return self._component_count_matrix("n_knots")

    @property
    def n_basis(self) -> int | list[list[int]]:
        """Basis counts per Cholesky entry, or one int when all are equal."""
        return self._component_count_matrix("n_basis")

    @property
    def n_theta(self) -> int:
        """Number of off-diagonal parameters."""
        return int(self.p * (self.p - 1) / 2)

    @property
    def theta_pairs(self) -> list[tuple[int, int]]:
        """Lower-triangular (j, l) index pairs with j > l in model order."""
        return [(j, l) for j in range(1, self.p) for l in range(j)]

    def delta_key(self, j: int) -> MultivarComponentKey:
        return MultivarComponentKey("delta", int(j))

    def theta_key(self, part: str, j: int, l: int) -> MultivarComponentKey:
        part_val = str(part).strip().lower()
        if part_val not in ("re", "im"):
            raise ValueError(f"Unknown theta part '{part}'. Use 're' or 'im'.")
        return MultivarComponentKey(
            "theta",
            int(j),
            l=int(l),
            part=cast(Literal["re", "im"], part_val),
        )

    @property
    def expected_component_order(self) -> list[MultivarComponentKey]:
        order = [self.delta_key(j) for j in range(self.p)]
        order.extend(self.theta_key("re", j, l) for j, l in self.theta_pairs)
        order.extend(self.theta_key("im", j, l) for j, l in self.theta_pairs)
        return order

    def theta_pair_from_index(self, theta_idx: int) -> tuple[int, int]:
        pairs = self.theta_pairs
        if theta_idx < 0 or theta_idx >= len(pairs):
            raise IndexError(
                f"theta index {theta_idx} out of range [0, {len(pairs)})"
            )
        return pairs[theta_idx]

    def component(self, key: MultivarComponentKey) -> LogPSpline:
        if key.family == "delta":
            return self.diagonal_models[key.j]
        return self.get_theta_model(key.part, key.j, key.l)

    def theta_index(self, j: int, l: int) -> int:
        if not (0 <= l < j < self.p):
            raise ValueError(
                f"Invalid theta pair ({j}, {l}) for p={self.p}; expected 0 <= l < j < p."
            )
        return j * (j - 1) // 2 + l

    def get_theta_model(self, part: str, j: int, l: int) -> LogPSpline:
        """Return the model for theta_{j,l} real/imag part."""
        self.theta_key(part, j, l)
        models = (
            self.offdiag_re_models if part == "re" else self.offdiag_im_models
        )
        return models[(j, l)]

    @property
    def total_components(self) -> int:
        """Total number of P-spline components."""
        return self.p + (2 * self.n_theta if self.n_theta > 0 else 0)

    def get_all_bases_and_penalties(
        self,
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        """
        Get basis and penalty matrices for all components (for NumPyro model).

        Returns
        -------
        Tuple[List[jnp.ndarray], List[jnp.ndarray]]
            Lists of basis and penalty matrices for all components
        """
        all_bases = []
        all_penalties = []
        for key in self.expected_component_order:
            model = self.component(key)
            all_bases.append(model.basis)
            all_penalties.append(model.penalty_matrix)

        return all_bases, all_penalties

    def _component_count_matrix(
        self, quantity: Literal["n_knots", "n_basis"]
    ) -> int | list[list[int]]:
        """Return component counts as a matrix matching the Cholesky layout."""
        matrix = [[0 for _ in range(self.p)] for _ in range(self.p)]
        counts: list[int] = []

        for j in range(self.p):
            diag_model = self.diagonal_models[j]
            value = (
                int(len(diag_model.knots))
                if quantity == "n_knots"
                else int(diag_model.n_basis)
            )
            matrix[j][j] = value
            counts.append(value)

        for j, l in self.theta_pairs:
            re_model = self.get_theta_model("re", j, l)
            im_model = self.get_theta_model("im", j, l)
            re_value = (
                int(len(re_model.knots))
                if quantity == "n_knots"
                else int(re_model.n_basis)
            )
            im_value = (
                int(len(im_model.knots))
                if quantity == "n_knots"
                else int(im_model.n_basis)
            )
            matrix[j][l] = re_value
            matrix[l][j] = im_value
            counts.extend([re_value, im_value])

        return counts[0] if len(set(counts)) == 1 else matrix

    def __repr__(self):
        knot_counts = {
            len(self.component(key).knots)
            for key in self.expected_component_order
        }
        if len(knot_counts) == 1:
            knot_label = str(next(iter(knot_counts)))
        else:
            knot_label = f"mixed[{min(knot_counts)}-{max(knot_counts)}]"
        basis_shapes = [
            tuple(int(v) for v in self.component(key).basis.shape)
            for key in self.expected_component_order
        ]
        return (
            f"SpectralComponents(channels={self.p}, "
            f"knots={knot_label}, degree={self.degree}, "
            f"penaltyOrder={self.diffMatrixOrder}, N={self.N}, "
            f"basis_shapes={basis_shapes})"
        )
