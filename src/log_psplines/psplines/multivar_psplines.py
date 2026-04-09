from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np

from ..datatypes import MultivarFFT, Periodogram
from ..datatypes.multivar_utils import U_to_Y, psd_to_cholesky_components
from ..logger import logger
from .initialisation import init_weights
from .knots_locator import init_knots, multivar_psd_knot_scores
from .psplines import LogPSplines

_MULTIVAR_ALLOWED_KNOT_METHODS = ("uniform", "log", "density")
_MULTIVAR_KNOT_FAMILY_KEYS = ("delta", "theta_re", "theta_im")


@dataclass(frozen=True)
class MultivarComponentKey:
    """Stable identifier for one multivariate spline component."""

    family: Literal["delta", "theta"]
    j: int
    l: Optional[int] = None
    part: Optional[Literal["re", "im"]] = None

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
class MultivarComponentSpec:
    """Container for a component's model and optional knot-score metadata."""

    key: MultivarComponentKey
    model: LogPSplines
    score: Optional[np.ndarray] = None


def _resolve_family_knot_counts(
    n_knots: int | Mapping[object, object],
) -> dict[str, int]:
    """Return knot counts for delta, theta_re, and theta_im families."""
    if isinstance(n_knots, Mapping):
        return {key: int(n_knots[key]) for key in _MULTIVAR_KNOT_FAMILY_KEYS}

    count = int(n_knots)
    return {key: count for key in _MULTIVAR_KNOT_FAMILY_KEYS}


def _build_component_knots(
    *,
    freq: np.ndarray,
    n_knots: int,
    score: np.ndarray,
    guide_score: np.ndarray | None,
    n_freq: int,
    knot_kwargs: dict[str, object],
) -> np.ndarray:
    """Build knots for one spline component."""
    method_raw = knot_kwargs.get("method", "density")
    method = str(method_raw).strip().lower()
    if method not in _MULTIVAR_ALLOWED_KNOT_METHODS:
        allowed = ", ".join(_MULTIVAR_ALLOWED_KNOT_METHODS)
        raise ValueError(
            f"Unsupported multivariate knot method '{method}'. "
            f"Allowed methods: {allowed}. "
            "Use univariate-compatible names via knot_kwargs['method']."
        )
    score_array = np.asarray(score, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError(
            f"score must be 1-D with shape (N,), got {score_array.shape}"
        )
    if score_array.shape[0] != int(n_freq):
        raise ValueError(
            f"score length must match n_freq={n_freq}, got {score_array.shape[0]}"
        )
    # Scores may be signed (e.g. Cholesky theta oscillating around zero).
    # Clean NaN/inf but preserve sign so gradient-based placement sees
    # genuine shape transitions, not artificial kinks at zero crossings.
    score_array = np.nan_to_num(score_array, nan=0.0, posinf=0.0, neginf=0.0)
    knot_periodogram = Periodogram(
        freqs=np.asarray(freq, dtype=np.float64),
        power=score_array,
    )
    knots = init_knots(
        n_knots=n_knots,
        periodogram=knot_periodogram,
        parametric_model=None,
        guide_power=(
            None
            if guide_score is None
            else np.asarray(guide_score, dtype=np.float64)
        ),
        **{**knot_kwargs, "method": method},
    )
    return knots


def _build_pspline_from_log_target(
    *,
    log_target: np.ndarray,
    knots: np.ndarray,
    degree: int,
    diff_matrix_order: int,
    n_freq: int,
    grid_points: np.ndarray,
) -> LogPSplines:
    """Create a LogPSplines model initialized from log-target data."""
    return LogPSplines.from_knots(
        knots=np.asarray(knots, dtype=np.float64),
        degree=degree,
        diffMatrixOrder=diff_matrix_order,
        n=n_freq,
        grid_points=np.asarray(grid_points, dtype=np.float64),
        parametric_model=jnp.ones(n_freq),
        log_target=np.asarray(log_target, dtype=np.float64),
        init_num_steps=5000,
    )


@dataclass
class MultivariateLogPSplines:
    """
    Multivariate log P-splines using Cholesky parameterization for cross-spectral density matrices.

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
        Order of finite difference penalty matrix
    N : int
        Number of frequency bins
    p : int
        Number of channels in multivariate data
    diagonal_models : List[LogPSplines]
        P-spline models for diagonal PSD components (one per channel)
    offdiag_re_models : Dict[Tuple[int, int], LogPSplines], optional
        P-spline models for real parts of off-diagonal terms keyed by
        ``(j, l)`` with ``j > l``.
    offdiag_im_models : Dict[Tuple[int, int], LogPSplines], optional
        P-spline models for imaginary parts of off-diagonal terms keyed by
        ``(j, l)`` with ``j > l``.
    component_specs : Dict[MultivarComponentKey, MultivarComponentSpec], optional
        Unified typed registry for all components.
    component_order : List[MultivarComponentKey], optional
        Deterministic ordering used to assemble tuples passed to samplers.
    """

    degree: int
    diffMatrixOrder: int
    N: int
    p: int

    # P-spline components for each Cholesky parameter
    diagonal_models: List[LogPSplines]  # One per channel
    offdiag_re_models: Dict[Tuple[int, int], LogPSplines] = field(
        default_factory=dict
    )
    offdiag_im_models: Dict[Tuple[int, int], LogPSplines] = field(
        default_factory=dict
    )
    component_specs: Dict[MultivarComponentKey, MultivarComponentSpec] = field(
        default_factory=dict, repr=False
    )
    component_order: List[MultivarComponentKey] = field(
        default_factory=list, repr=False
    )
    component_scores: Dict[MultivarComponentKey, np.ndarray] = field(
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
            self._initialise_component_registry()
            return

        pairs = self.theta_pairs
        if self.component_specs:
            for j, l in pairs:
                key_re = MultivarComponentKey("theta", j, l=l, part="re")
                key_im = MultivarComponentKey("theta", j, l=l, part="im")
                if key_re in self.component_specs:
                    self.offdiag_re_models[(j, l)] = self.component_specs[
                        key_re
                    ].model
                if key_im in self.component_specs:
                    self.offdiag_im_models[(j, l)] = self.component_specs[
                        key_im
                    ].model

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

        self._initialise_component_registry()

    @classmethod
    def from_multivar_fft(
        cls,
        fft_data: MultivarFFT,
        n_knots: int | Mapping[object, object],
        degree: int = 3,
        diffMatrixOrder: int = 2,
        knot_kwargs: dict[str, object] | None = None,
        analytical_psd: np.ndarray | None = None,
    ) -> "MultivariateLogPSplines":
        """
        Factory method to construct multivariate P-spline model from FFT data.

        Parameters
        ----------
        fft_data : MultivarFFT
            Multivariate FFT data with real/imaginary components and design matrices
        n_knots : int or mapping
            Knot-count specification for the Cholesky components. Provide a
            single integer to reuse the same number of knots for every
            component, or a mapping with family counts for
            ``"delta"``, ``"theta_re"``, and ``"theta_im"``.
        degree : int, default=3
            Polynomial degree of B-spline basis
        diffMatrixOrder : int, default=2
            Order of difference penalty matrix
        knot_kwargs : dict, optional
            Additional arguments passed to the shared knot locator.
            Supported methods match univariate naming:
            ``method in {"uniform", "log", "density"}``.
            When omitted, defaults to ``method="density"``.
            For ``method="density"``, per-component knot scores are always
            computed from the Cholesky parameterization of the channel-space
            Wishart matrix.
        analytical_psd : np.ndarray or tuple, optional
            Known analytical PSD matrix (e.g. from a transfer function model).
            Either an ``(N, p, p)`` array already on the FFT frequency grid,
            or a ``(freq_ana, S_ana)`` tuple that will be interpolated to the
            FFT grid automatically. When provided, the Cholesky components of
            this matrix are used as guide signals for density-based knot
            placement, so known analytical features can attract knots without
            subtracting from the empirical scores.

        Returns
        -------
        MultivariateLogPSplines
            Fully initialized multivariate model
        """
        if knot_kwargs is None:
            knot_kwargs = {}

        N = fft_data.N
        p = fft_data.p
        family_knot_counts = _resolve_family_knot_counts(n_knots)

        # Create frequency grid for knot placement (normalized to [0,1])
        freq = np.asarray(fft_data.freq, dtype=np.float64)
        finite_mask = np.isfinite(freq)
        if not finite_mask.any():
            freq_norm = np.zeros_like(freq)
        else:
            freq_finite = freq[finite_mask]
            freq_min = float(freq_finite.min())
            freq_max = float(freq_finite.max())
            denom = freq_max - freq_min
            if denom <= 0:
                freq_norm = np.zeros_like(freq)
            else:
                freq_norm = (freq - freq_min) / denom
                freq_norm = np.where(finite_mask, freq_norm, 0.0)

        if fft_data.u_re is None or fft_data.u_im is None:
            raise ValueError(
                "Multivariate models require Wishart statistics (u_re/u_im)."
            )
        u_re_np = np.asarray(fft_data.u_re, dtype=np.float64)
        u_im_np = np.asarray(fft_data.u_im, dtype=np.float64)
        u_complex_np = u_re_np + 1j * u_im_np
        # Use channel-space Wishart matrices Y[f] = U[f] U[f]^H.
        # This is the matrix consumed by the likelihood and preserves
        # cross-spectral structure for knot scoring.
        Y_np = U_to_Y(u_complex_np)
        Nb = max(int(fft_data.Nb), 1)

        requested_scoring = knot_kwargs.get("scoring")
        if requested_scoring is not None:
            knot_kwargs = {
                key: value
                for key, value in knot_kwargs.items()
                if key != "scoring"
            }

        (
            diagonal_scores,
            offdiag_re_scores,
            offdiag_im_scores,
        ) = multivar_psd_knot_scores(
            Y_np,
            Nb,
            p,
        )

        analytical_diagonal_scores: list[np.ndarray] | None = None
        analytical_offdiag_re_scores: list[np.ndarray] | None = None
        analytical_offdiag_im_scores: list[np.ndarray] | None = None

        # Use analytical model Cholesky components as guide signals for
        # density-based knot placement without modifying the empirical score.
        if analytical_psd is not None:
            if isinstance(analytical_psd, tuple):
                from ..datatypes.multivar_utils import interp_matrix

                freq_ana, S_ana = analytical_psd
                analytical_psd = interp_matrix(
                    np.asarray(freq_ana, dtype=np.float64),
                    np.asarray(S_ana, dtype=np.complex128),
                    freq,
                )
            analytical_psd = np.asarray(analytical_psd, dtype=np.complex128)
            if analytical_psd.shape != (N, p, p):
                raise ValueError(
                    f"analytical_psd must have shape ({N}, {p}, {p}), "
                    f"got {analytical_psd.shape}"
                )
            ana_log_delta, ana_theta = psd_to_cholesky_components(
                analytical_psd
            )
            analytical_diagonal_scores = [
                np.asarray(ana_log_delta[:, i], dtype=np.float64)
                for i in range(p)
            ]
            analytical_offdiag_re_scores = []
            analytical_offdiag_im_scores = []
            for j in range(1, p):
                for l in range(j):
                    analytical_offdiag_re_scores.append(
                        np.asarray(
                            np.real(ana_theta[:, j, l]), dtype=np.float64
                        )
                    )
                    analytical_offdiag_im_scores.append(
                        np.asarray(
                            np.imag(ana_theta[:, j, l]), dtype=np.float64
                        )
                    )

        component_scores: Dict[MultivarComponentKey, np.ndarray] = {}

        # Create diagonal models (one per channel), each with its own
        # knot placement and basis construction.
        diagonal_models = []
        for i in range(p):
            delta_key = MultivarComponentKey("delta", i)
            score_diag = diagonal_scores[i]
            component_scores[delta_key] = np.asarray(
                score_diag, dtype=np.float64
            )
            knots_diag = _build_component_knots(
                freq=freq,
                n_knots=family_knot_counts["delta"],
                score=score_diag,
                guide_score=(
                    None
                    if analytical_diagonal_scores is None
                    else analytical_diagonal_scores[i]
                ),
                n_freq=N,
                knot_kwargs=knot_kwargs,
            )

            # Keep initialization empirical for now. The analytical PSD may
            # guide knot placement, but the starting spline target should still
            # reflect the observed Wishart matrix on the retained grid.
            empirical_diag_power = np.real(Y_np[:, i, i]) / Nb
            empirical_diag_power = np.maximum(
                empirical_diag_power, 1e-12
            )  # Avoid log(0)
            log_diag_target = np.log(empirical_diag_power)
            diagonal_model = _build_pspline_from_log_target(
                log_target=log_diag_target,
                knots=knots_diag,
                degree=degree,
                diff_matrix_order=diffMatrixOrder,
                n_freq=N,
                grid_points=freq_norm,
            )
            diagonal_models.append(diagonal_model)

        # Create off-diagonal models if needed
        offdiag_re_models: Dict[Tuple[int, int], LogPSplines] = {}
        offdiag_im_models: Dict[Tuple[int, int], LogPSplines] = {}
        theta_pairs = [(j, l) for j in range(1, p) for l in range(j)]

        if p > 1:
            if len(offdiag_re_scores) != len(theta_pairs):
                raise ValueError(
                    "Off-diagonal real knot score length mismatch: "
                    f"expected {len(theta_pairs)}, got {len(offdiag_re_scores)}."
                )
            if len(offdiag_im_scores) != len(theta_pairs):
                raise ValueError(
                    "Off-diagonal imag knot score length mismatch: "
                    f"expected {len(theta_pairs)}, got {len(offdiag_im_scores)}."
                )

            # Keep theta initialization empirical for now. The analytical PSD
            # may guide knot placement, but we do not inject it into the
            # initial spline weights unless we intentionally add that policy.
            # The sampler evaluates theta = B @ w directly (no exp), so weights
            # must be initialised to reproduce the empirical theta values, not
            # their log-magnitude.
            _, theta_emp = psd_to_cholesky_components(Y_np / max(Nb, 1))

            for theta_idx, (j_idx, l_idx) in enumerate(theta_pairs):
                theta_re_key = MultivarComponentKey(
                    "theta", j_idx, l=l_idx, part="re"
                )
                theta_im_key = MultivarComponentKey(
                    "theta", j_idx, l=l_idx, part="im"
                )
                score_theta_re = np.asarray(
                    offdiag_re_scores[theta_idx], dtype=np.float64
                )
                score_theta_im = np.asarray(
                    offdiag_im_scores[theta_idx], dtype=np.float64
                )
                component_scores[theta_re_key] = score_theta_re
                component_scores[theta_im_key] = score_theta_im
                # Use actual empirical theta components as initialisation targets.
                # Re and Im are initialised independently so each spline starts
                # from the correct signed value rather than a log-magnitude proxy.
                theta_re_init = np.real(theta_emp[:, j_idx, l_idx])
                theta_im_init = np.imag(theta_emp[:, j_idx, l_idx])
                knots_theta_re = _build_component_knots(
                    freq=freq,
                    n_knots=family_knot_counts["theta_re"],
                    score=score_theta_re,
                    guide_score=(
                        None
                        if analytical_offdiag_re_scores is None
                        else analytical_offdiag_re_scores[theta_idx]
                    ),
                    n_freq=N,
                    knot_kwargs=knot_kwargs,
                )
                knots_theta_im = _build_component_knots(
                    freq=freq,
                    n_knots=family_knot_counts["theta_im"],
                    score=score_theta_im,
                    guide_score=(
                        None
                        if analytical_offdiag_im_scores is None
                        else analytical_offdiag_im_scores[theta_idx]
                    ),
                    n_freq=N,
                    knot_kwargs=knot_kwargs,
                )
                offdiag_re_models[(j_idx, l_idx)] = (
                    _build_pspline_from_log_target(
                        log_target=theta_re_init,
                        knots=knots_theta_re,
                        degree=degree,
                        diff_matrix_order=diffMatrixOrder,
                        n_freq=N,
                        grid_points=freq_norm,
                    )
                )
                offdiag_im_models[(j_idx, l_idx)] = (
                    _build_pspline_from_log_target(
                        log_target=theta_im_init,
                        knots=knots_theta_im,
                        degree=degree,
                        diff_matrix_order=diffMatrixOrder,
                        n_freq=N,
                        grid_points=freq_norm,
                    )
                )

        return cls(
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            N=N,
            p=p,
            diagonal_models=diagonal_models,
            offdiag_re_models=offdiag_re_models,
            offdiag_im_models=offdiag_im_models,
            component_scores=component_scores,
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
    def theta_pairs(self) -> List[Tuple[int, int]]:
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
    def expected_component_order(self) -> List[MultivarComponentKey]:
        order = [self.delta_key(j) for j in range(self.p)]
        order.extend(self.theta_key("re", j, l) for j, l in self.theta_pairs)
        order.extend(self.theta_key("im", j, l) for j, l in self.theta_pairs)
        return order

    def _initialise_component_registry(self) -> None:
        """Create/validate a typed component registry from model fields."""
        expected_order = self.expected_component_order
        if not self.component_order:
            self.component_order = list(expected_order)

        if not self.component_specs:
            specs: Dict[MultivarComponentKey, MultivarComponentSpec] = {}
            for j, model in enumerate(self.diagonal_models):
                key = self.delta_key(j)
                specs[key] = MultivarComponentSpec(
                    key=key,
                    model=model,
                    score=self.component_scores.get(key),
                )
            for j, l in self.theta_pairs:
                key_re = self.theta_key("re", j, l)
                key_im = self.theta_key("im", j, l)
                specs[key_re] = MultivarComponentSpec(
                    key=key_re,
                    model=self.offdiag_re_models[(j, l)],
                    score=self.component_scores.get(key_re),
                )
                specs[key_im] = MultivarComponentSpec(
                    key=key_im,
                    model=self.offdiag_im_models[(j, l)],
                    score=self.component_scores.get(key_im),
                )
            self.component_specs = specs
        if not self.component_scores:
            self.component_scores = {
                key: spec.score
                for key, spec in self.component_specs.items()
                if spec.score is not None
            }

        missing = [k for k in expected_order if k not in self.component_specs]
        if missing:
            raise ValueError(
                "component_specs is missing required keys: "
                f"{[k.name for k in missing]}"
            )
        if len(self.component_order) != len(expected_order):
            raise ValueError(
                f"component_order must have {len(expected_order)} entries, "
                f"got {len(self.component_order)}."
            )
        if set(self.component_order) != set(expected_order):
            raise ValueError(
                "component_order must contain exactly the expected keys."
            )

    def theta_pair_from_index(self, theta_idx: int) -> Tuple[int, int]:
        pairs = self.theta_pairs
        if theta_idx < 0 or theta_idx >= len(pairs):
            raise IndexError(
                f"theta index {theta_idx} out of range [0, {len(pairs)})"
            )
        return pairs[theta_idx]

    def get_component_spec(
        self, key: MultivarComponentKey
    ) -> MultivarComponentSpec:
        return self.component_specs[key]

    def iter_component_specs(self) -> List[MultivarComponentSpec]:
        return [self.component_specs[key] for key in self.component_order]

    def theta_index(self, j: int, l: int) -> int:
        if not (0 <= l < j < self.p):
            raise ValueError(
                f"Invalid theta pair ({j}, {l}) for p={self.p}; expected 0 <= l < j < p."
            )
        return j * (j - 1) // 2 + l

    def get_theta_model(self, part: str, j: int, l: int) -> LogPSplines:
        """Return the model for theta_{j,l} real/imag part."""
        key = self.theta_key(part, j, l)
        return self.component_specs[key].model

    @property
    def total_components(self) -> int:
        """Total number of P-spline components."""
        return self.p + (2 * self.n_theta if self.n_theta > 0 else 0)

    def get_all_bases_and_penalties(
        self,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Get basis and penalty matrices for all components (for NumPyro model).

        Returns
        -------
        Tuple[List[jnp.ndarray], List[jnp.ndarray]]
            Lists of basis and penalty matrices for all components
        """
        all_bases = []
        all_penalties = []
        for key in self.component_order:
            model = self.component_specs[key].model
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
            diag_model = self.component_specs[self.delta_key(j)].model
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

    def compute_design_weights(
        self,
        design_psd: np.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Fit spline weights to a known design PSD matrix via Cholesky decomposition.

        Parameters
        ----------
        design_psd:
            Complex array of shape ``(N, p, p)`` giving the design PSD matrix at the
            model's frequency grid (after any coarse-graining).

        Returns
        -------
        dict
            Keys ``'delta_{j}'``, ``'theta_re_{j}_{l}'``, and
            ``'theta_im_{j}_{l}'`` mapping to fitted weight arrays.  The dict
            covers every model component so each lookup in the sampler is
            well-defined.
        """
        design_psd = np.asarray(design_psd)
        if design_psd.shape != (self.N, self.p, self.p):
            raise ValueError(
                f"design_psd must have shape ({self.N}, {self.p}, {self.p}), "
                f"got {design_psd.shape}"
            )

        # Lower Cholesky: S = L L^H, L lower-triangular with real positive diagonal
        L = np.linalg.cholesky(design_psd)  # (N, p, p)

        design_weights: dict[str, jnp.ndarray] = {}

        # Diagonal components: log δ_j(f)² = 2 log L_{jj}(f)
        for j in range(self.p):
            diag_model = self.component_specs[self.delta_key(j)].model
            log_delta_sq = 2.0 * np.log(np.abs(L[:, j, j]))  # (N,)
            design_weights[f"delta_{j}"] = init_weights(
                jnp.asarray(log_delta_sq), diag_model
            )

        # Off-diagonal components: θ_{j,l} = −T_{j,l} where T = D^{1/2} L^{-1}
        # Unit lower-triangular T^{-1} = L / diag(L), so T = (T^{-1})^{-1}.
        if self.n_theta > 0:
            # Build unit lower-triangular T^{-1}[f] = L[f] / L[f,l,l] per column l
            diag_L = np.abs(
                L[..., np.arange(self.p), np.arange(self.p)]
            )  # (N, p)
            T_inv = (
                L / diag_L[:, np.newaxis, :]
            )  # (N, p, p), unit lower triangular

            # Invert unit lower-triangular matrix at each frequency via scipy
            import scipy.linalg  # local import to keep top-level imports light

            T = np.stack(
                [
                    scipy.linalg.solve_triangular(
                        T_inv[f], np.eye(self.p), lower=True
                    )
                    for f in range(self.N)
                ]
            )  # (N, p, p)

            for j in range(1, self.p):
                for l in range(j):
                    # theta_{j,l} = -T_{j,l} (complex)
                    theta_jl = -T[:, j, l]  # (N,) complex
                    re_model = self.get_theta_model("re", j, l)
                    im_model = self.get_theta_model("im", j, l)
                    design_weights[f"theta_re_{j}_{l}"] = init_weights(
                        jnp.asarray(theta_jl.real), re_model
                    )
                    design_weights[f"theta_im_{j}_{l}"] = init_weights(
                        jnp.asarray(theta_jl.imag), im_model
                    )

        return design_weights

    def _psd_chunk_iterator(
        self,
        log_delta_sq_samples: np.ndarray,
        theta_re_samples: Optional[np.ndarray],
        theta_im_samples: Optional[np.ndarray],
        *,
        n_samps: int,
        chunk_size: int,
    ):
        """Yield reconstructed PSD chunks with shape (n_samps, chunk, n, n)."""

        N = log_delta_sq_samples.shape[1]
        p = log_delta_sq_samples.shape[2]
        n_theta = (
            theta_re_samples.shape[2] if theta_re_samples is not None else 0
        )
        tril_row, tril_col = np.tril_indices(p, k=-1)
        n_lower = len(tril_row)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            log_chunk = log_delta_sq_samples[:n_samps, start:end, :]
            theta_re_chunk = (
                theta_re_samples[:n_samps, start:end, :]
                if theta_re_samples is not None
                else None
            )
            theta_im_chunk = (
                theta_im_samples[:n_samps, start:end, :]
                if theta_im_samples is not None
                else None
            )

            chunk_len = end - start
            psd_chunk = np.empty(
                (n_samps, chunk_len, p, p),
                dtype=np.complex64,
            )

            for s in range(n_samps):
                for local_f in range(chunk_len):
                    diag_vals = np.exp(log_chunk[s, local_f]).astype(
                        np.float32
                    )
                    T = np.eye(p, dtype=np.complex64)

                    if n_theta > 0:
                        assert theta_re_chunk is not None
                        assert theta_im_chunk is not None
                        theta_complex = (
                            theta_re_chunk[s, local_f]
                            + 1j * theta_im_chunk[s, local_f]
                        )
                        n_use = min(theta_complex.shape[0], n_lower)
                        if n_use:
                            T[tril_row[:n_use], tril_col[:n_use]] = (
                                -theta_complex[:n_use]
                            )

                    Tinverse = np.linalg.inv(T)
                    D = np.diag(diag_vals)
                    psd_chunk[s, local_f] = (
                        Tinverse @ D @ Tinverse.conj().T
                    ).astype(np.complex64)

            yield start, end, psd_chunk

    def reconstruct_psd_matrix(
        self,
        log_delta_sq_samples: jnp.ndarray,
        theta_re_samples: jnp.ndarray,
        theta_im_samples: jnp.ndarray,
        n_samples_max: int = 50,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """
        Reconstruct PSD matrices from Cholesky components using NumPy.

        The computation streams over frequency chunks (default 2048 bins) so the
        peak memory stays modest even for very long spectra. Results are returned
        as a ``complex64`` NumPy array of shape
        ``(n_samps, N, p, p)``.
        """
        log_delta_sq_arr = np.asarray(log_delta_sq_samples)
        theta_re_arr = np.asarray(theta_re_samples)
        theta_im_arr = np.asarray(theta_im_samples)

        if log_delta_sq_arr.ndim == 4:
            log_delta_sq_arr = log_delta_sq_arr[0]
        if theta_re_arr.ndim == 4:
            theta_re_arr = theta_re_arr[0]
        if theta_im_arr.ndim == 4:
            theta_im_arr = theta_im_arr[0]

        n_samples, N, p = log_delta_sq_arr.shape
        n_theta = theta_re_arr.shape[2] if theta_re_arr.ndim > 2 else 0
        n_samps = min(int(n_samples_max), int(n_samples))

        if chunk_size is None or chunk_size <= 0:
            chunk_size = N

        log_delta_sq_arr = log_delta_sq_arr[:n_samps]
        theta_re_arr = theta_re_arr[:n_samps]
        theta_im_arr = theta_im_arr[:n_samps]

        psd = np.empty((n_samps, N, p, p), dtype=np.complex64)

        for start, end, psd_chunk in self._psd_chunk_iterator(
            log_delta_sq_arr,
            theta_re_arr if n_theta > 0 else None,
            theta_im_arr if n_theta > 0 else None,
            n_samps=n_samps,
            chunk_size=chunk_size,
        ):
            psd[:, start:end] = psd_chunk

        return psd

    def compute_psd_quantiles(
        self,
        log_delta_sq_samples: jnp.ndarray,
        theta_re_samples: jnp.ndarray,
        theta_im_samples: jnp.ndarray,
        *,
        percentiles: Optional[Sequence[float]] = None,
        n_samples_max: int = 50,
        chunk_size: int = 2048,
        compute_coherence: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute PSD (and optional coherence) percentiles without storing all draws.

        Returns
        -------
        psd_real_percentiles : np.ndarray
            Percentiles of the real part of the PSD matrix with shape
            ``(n_percentiles, N, p, p)``.
        psd_imag_percentiles : np.ndarray
            Percentiles of the imaginary part of the PSD matrix with matching shape.
        coherence_percentiles : Optional[np.ndarray]
            When ``compute_coherence`` is ``True`` and ``p > 1``, contains
            percentiles of the coherence matrix; otherwise ``None``.
        """

        if percentiles is None:
            percentiles = [5.0, 50.0, 95.0]

        log_delta_sq_arr = np.asarray(log_delta_sq_samples)
        theta_re_arr = np.asarray(theta_re_samples)
        theta_im_arr = np.asarray(theta_im_samples)

        if log_delta_sq_arr.ndim == 4:
            log_delta_sq_arr = log_delta_sq_arr[0]
        if theta_re_arr.ndim == 4:
            theta_re_arr = theta_re_arr[0]
        if theta_im_arr.ndim == 4:
            theta_im_arr = theta_im_arr[0]

        n_samples, N, p = log_delta_sq_arr.shape
        n_theta = theta_re_arr.shape[2] if theta_re_arr.ndim > 2 else 0
        n_samps = min(int(n_samples_max), int(n_samples))

        if chunk_size is None or chunk_size <= 0:
            chunk_size = N

        log_delta_sq_arr = log_delta_sq_arr[:n_samps]
        theta_re_arr = theta_re_arr[:n_samps]
        theta_im_arr = theta_im_arr[:n_samps]

        n_percentiles = len(percentiles)
        psd_percentiles = np.empty((n_percentiles, N, p, p), dtype=np.float64)
        psd_imag_percentiles = np.empty_like(psd_percentiles)

        coherence_percentiles = (
            np.empty(
                (n_percentiles, N, p, p),
                dtype=np.float64,
            )
            if compute_coherence and p > 1
            else None
        )

        for start, end, psd_chunk in self._psd_chunk_iterator(
            log_delta_sq_arr,
            theta_re_arr if n_theta > 0 else None,
            theta_im_arr if n_theta > 0 else None,
            n_samps=n_samps,
            chunk_size=chunk_size,
        ):
            psd_real = psd_chunk.real
            psd_imag = psd_chunk.imag

            real_q = np.percentile(psd_real, percentiles, axis=0)
            imag_q = np.percentile(psd_imag, percentiles, axis=0)

            psd_percentiles[:, start:end] = real_q
            psd_imag_percentiles[:, start:end] = imag_q

            if coherence_percentiles is not None:
                diag = np.abs(
                    np.diagonal(psd_chunk, axis1=2, axis2=3)
                )  # (samples, chunk, channels)
                denom = diag[..., :, None] * diag[..., None, :]
                denom = np.where(denom > 0.0, denom, np.nan)
                coh_samples = (np.abs(psd_chunk) ** 2) / denom
                coh_samples = np.nan_to_num(coh_samples, nan=0.0, posinf=0.0)
                coh_q = np.percentile(coh_samples, percentiles, axis=0)

                # enforce exact ones on diagonal to avoid numerical drift
                for idx in range(n_percentiles):
                    for c in range(p):
                        coh_q[idx, :, c, c] = 1.0

                coherence_percentiles[:, start:end] = coh_q

        return psd_percentiles, psd_imag_percentiles, coherence_percentiles

    def __repr__(self):
        knot_counts = {
            len(spec.model.knots) for spec in self.component_specs.values()
        }
        if len(knot_counts) == 1:
            knot_label = str(next(iter(knot_counts)))
        else:
            knot_label = f"mixed[{min(knot_counts)}-{max(knot_counts)}]"
        return (
            f"MultivariateLogPSplines(channels={self.p}, "
            f"knots={knot_label}, degree={self.degree}, "
            f"penaltyOrder={self.diffMatrixOrder}, N={self.N})"
        )

    def get_psd_matrix_percentiles(
        self, psd_matrix_samples: jnp.ndarray, percentiles=[2.5, 50, 97.5]
    ) -> np.ndarray:
        arr = np.asarray(psd_matrix_samples)
        if arr.ndim == 4 and arr.shape[0] == len(percentiles):
            return arr.astype(np.float64, copy=False)

        if arr.ndim == 3:
            arr = arr[None, ...]
        elif arr.ndim != 4:
            raise ValueError(
                f"Expected 4D array (samples, freqs, n, n), got {arr.shape}"
            )

        psd_matrix_real = _complex_to_real_batch(arr)
        posterior_percentiles = np.percentile(
            psd_matrix_real, percentiles, axis=0
        )
        return posterior_percentiles.astype(np.float64, copy=False)

    def get_psd_matrix_coverage(
        self, psd_matrix_samples: jnp.ndarray, empirical_psd: jnp.ndarray
    ) -> float:
        empirical_psd_real = _complex_to_real_batch(empirical_psd)

        psd_percentiles = self.get_psd_matrix_percentiles(psd_matrix_samples)
        coverage = np.mean(
            (empirical_psd_real >= psd_percentiles[0])
            & (empirical_psd_real <= psd_percentiles[-1])
        )
        return float(coverage)


def _complex_to_real_batch(mats):
    """
    Safe, vectorized transform:
      - Upper triangle (incl diag) -> real part
      - Strict lower triangle      -> imag part
    mats: (..., n, n) complex
    returns float32 with same leading dims
    """
    mats = np.asarray(mats)
    n = mats.shape[-1]
    # boolean masks that broadcast over leading dims
    upper = np.triu(np.ones((n, n), dtype=bool))
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)

    out = np.where(upper, mats.real, 0.0)
    out = np.where(lower, mats.imag, out)
    return out.astype(np.float64, copy=False)
