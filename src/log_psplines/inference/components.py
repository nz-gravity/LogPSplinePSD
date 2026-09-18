from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (
    Literal,
    cast,
)

import jax.numpy as jnp
import numpy as np

from log_psplines.data import WishartData
from log_psplines.data.spectral_utils import U_to_Y, psd_to_cholesky_components
from log_psplines.inference.initialisation import build_component, init_weights
from log_psplines.models.spectrum import LogPSpline
from log_psplines.preprocessing.knots_locator import (
    init_knots,
    multivar_psd_knot_scores,
)

_MULTIVAR_ALLOWED_KNOT_METHODS = ("uniform", "log", "density")
_MULTIVAR_KNOT_FAMILY_KEYS = ("delta", "theta_re", "theta_im")


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
    knots = init_knots(
        n_knots=n_knots,
        freqs=np.asarray(freq, dtype=np.float64),
        power=score_array,
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
) -> LogPSpline:
    """Create a LogPSpline model initialized from log-target data."""
    return build_component(
        knots=np.asarray(knots, dtype=np.float64),
        degree=degree,
        diffMatrixOrder=diff_matrix_order,
        n=n_freq,
        grid_points=np.asarray(grid_points, dtype=np.float64),
        log_target=np.asarray(log_target, dtype=np.float64),
        init_num_steps=5000,
    )


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

    @classmethod
    def from_multivar_fft(
        cls,
        fft_data: WishartData,
        n_knots: int | Mapping[object, object],
        degree: int = 3,
        diffMatrixOrder: int = 2,
        knot_kwargs: dict[str, object] | None = None,
        analytical_psd: np.ndarray | None = None,
    ) -> "SpectralComponents":
        """
        Factory method to construct multivariate P-spline model from FFT data.

        Parameters
        ----------
        fft_data : WishartData
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
        SpectralComponents
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
                from log_psplines.data.spectral_utils import interp_matrix

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

        component_scores: dict[MultivarComponentKey, np.ndarray] = {}

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
        offdiag_re_models: dict[tuple[int, int], LogPSpline] = {}
        offdiag_im_models: dict[tuple[int, int], LogPSpline] = {}
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
            diag_model = self.diagonal_models[j]
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
