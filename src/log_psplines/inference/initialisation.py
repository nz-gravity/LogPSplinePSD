"""Prepare spline components and fit initial coefficients from observations."""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from log_psplines.basis.splines import SplineBasis
from log_psplines.data import WishartData
from log_psplines.data.spectral_utils import U_to_Y, psd_to_cholesky_components
from log_psplines.inference.components import (
    MultivarComponentKey,
    SpectralComponents,
)
from log_psplines.models.spectrum import LogPSpline
from log_psplines.preprocessing.knot_locator import (
    init_knots,
    multivar_psd_knot_scores,
)

_MULTIVAR_ALLOWED_KNOT_METHODS = ("uniform", "log", "density")
_MULTIVAR_KNOT_FAMILY_KEYS = ("delta", "theta_re", "theta_im")


def init_weights(
    log_pdgrm: Float[Array | np.ndarray, "N"],
    log_psplines: "LogPSpline",
) -> Float[Array, "K_f"]:
    """Return a stabilized least-squares fit for the spline weights."""
    basis = jnp.asarray(log_psplines.basis)
    target = jnp.asarray(log_pdgrm)
    gram = basis.T @ basis
    rhs = basis.T @ target

    n_basis = int(gram.shape[0])
    trace_scale = jnp.trace(gram) / max(n_basis, 1)
    ridge = jnp.asarray(1e-6, dtype=gram.dtype) * jnp.maximum(
        trace_scale, jnp.asarray(1.0, dtype=gram.dtype)
    )
    system = gram + ridge * jnp.eye(n_basis, dtype=gram.dtype)
    return jnp.linalg.solve(system, rhs)


def build_component(
    *,
    degree: int,
    diffMatrixOrder: int,
    n: int,
    knots: Array | np.ndarray,
    basis: Array | np.ndarray | None = None,
    penalty_matrix: Array | np.ndarray | None = None,
    weights: Array | np.ndarray | None = None,
    grid_points: Array | np.ndarray | None = None,
    log_target: Array | np.ndarray | None = None,
) -> LogPSpline:
    """Prepare a scalar component, optionally fitting its initial weights."""
    frequency = SplineBasis.create(
        degree=degree,
        penalty_order=diffMatrixOrder,
        n=n,
        knots=knots,
        basis=basis,
        penalty=penalty_matrix,
        grid=grid_points,
    )
    model = LogPSpline(frequency, weights=weights)
    if log_target is not None:
        target = jnp.asarray(log_target)
        if target.ndim != 1 or target.shape[0] != n:
            raise ValueError("log_target must be 1-D with length n")
        model.weights = init_weights(target, model)
    return model


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
            f"score length must match n_freq={n_freq}, "
            f"got {score_array.shape[0]}"
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
    )


def prepare_components(
    fft_data: WishartData,
    n_knots: int | Mapping[object, object],
    degree: int = 3,
    diffMatrixOrder: int = 2,
    knot_kwargs: dict[str, object] | None = None,
    analytical_psd: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> "SpectralComponents":
    """
    Prepare stationary scalar components from Wishart observations.

    Parameters
    ----------
    fft_data : WishartData
        Wishart statistics on the retained frequency grid.
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
        Reference PSD matrix (for example, an analytical transfer function).
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
        ana_log_delta, ana_theta = psd_to_cholesky_components(analytical_psd)
        analytical_diagonal_scores = [
            np.asarray(ana_log_delta[:, i], dtype=np.float64) for i in range(p)
        ]
        analytical_offdiag_re_scores = []
        analytical_offdiag_im_scores = []
        for j in range(1, p):
            for column in range(j):
                analytical_offdiag_re_scores.append(
                    np.asarray(
                        np.real(ana_theta[:, j, column]), dtype=np.float64
                    )
                )
                analytical_offdiag_im_scores.append(
                    np.asarray(
                        np.imag(ana_theta[:, j, column]), dtype=np.float64
                    )
                )

    component_scores: dict[MultivarComponentKey, np.ndarray] = {}

    # Create diagonal models (one per channel), each with its own
    # knot placement and basis construction.
    diagonal_models = []
    for i in range(p):
        delta_key = MultivarComponentKey("delta", i)
        score_diag = diagonal_scores[i]
        component_scores[delta_key] = np.asarray(score_diag, dtype=np.float64)
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
    theta_pairs = [(j, column) for j in range(1, p) for column in range(j)]

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
            offdiag_re_models[(j_idx, l_idx)] = _build_pspline_from_log_target(
                log_target=theta_re_init,
                knots=knots_theta_re,
                degree=degree,
                diff_matrix_order=diffMatrixOrder,
                n_freq=N,
                grid_points=freq_norm,
            )
            offdiag_im_models[(j_idx, l_idx)] = _build_pspline_from_log_target(
                log_target=theta_im_init,
                knots=knots_theta_im,
                degree=degree,
                diff_matrix_order=diffMatrixOrder,
                n_freq=N,
                grid_points=freq_norm,
            )

    return SpectralComponents(
        degree=degree,
        diffMatrixOrder=diffMatrixOrder,
        N=N,
        p=p,
        diagonal_models=diagonal_models,
        offdiag_re_models=offdiag_re_models,
        offdiag_im_models=offdiag_im_models,
        component_scores=component_scores,
    )
