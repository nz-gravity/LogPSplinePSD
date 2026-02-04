from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coarse_grain.preprocess import CoarseGrainSpec
from ..datatypes.multivar import MultivarFFT
from ..spectrum_utils import sum_wishart_outer_products


@dataclass(frozen=True, slots=True)
class CoarseBinEquivalenceResult:
    """Result of a fine-vs-coarse likelihood equivalence check for one bin."""

    bin_index: int
    n_members: int
    ll_fine: float
    ll_coarse: float

    @property
    def delta(self) -> float:
        return float(self.ll_coarse - self.ll_fine)

    @property
    def rel_delta(self) -> float:
        denom = max(abs(self.ll_fine), abs(self.ll_coarse), 1.0)
        return float(self.delta / denom)


@dataclass(frozen=True, slots=True)
class BinDoublingStiffnessEpsResult:
    """Finite-difference curvature proxy results for one epsilon."""

    eps: float
    kappa_small: float
    kappa_large: float
    ratio: float
    expected_ratio: float


@dataclass(frozen=True, slots=True)
class BinDoublingStiffnessResult:
    """Result of a bin-doubling stiffness check (bin vs adjacent union)."""

    bin_index: int
    n_members_small: int
    n_members_large: int
    eps_results: tuple[BinDoublingStiffnessEpsResult, ...]


def _bin_member_indices_from_spec(
    *,
    spec: CoarseGrainSpec,
    bin_index: int,
) -> np.ndarray:
    """Return indices (into the selected-high array) belonging to a coarse bin."""

    if spec.n_bins_high <= 0:
        raise ValueError("Spec has no coarse bins (n_bins_high=0).")
    if bin_index < 0 or bin_index >= int(spec.n_bins_high):
        raise ValueError(
            f"bin_index out of range: {bin_index} not in [0, {int(spec.n_bins_high) - 1}]"
        )

    bin_counts = np.asarray(spec.bin_counts, dtype=np.int64)
    if bin_counts.shape != (int(spec.n_bins_high),):
        raise ValueError("Spec bin_counts has unexpected shape.")

    start = int(np.sum(bin_counts[:bin_index]))
    count = int(bin_counts[bin_index])
    if count <= 0:
        raise ValueError(f"Bin {bin_index} has non-positive count {count}.")
    return np.arange(start, start + count, dtype=np.int64)


def _extract_u_stack_for_bin(
    fft: MultivarFFT,
    spec: CoarseGrainSpec,
    *,
    bin_index: int,
) -> np.ndarray:
    """Return U matrices for all fine frequencies in the specified coarse bin.

    Returns:
        u_stack: complex array with shape (N_h, n_dim, n_dim)
    """
    selection = np.asarray(spec.selection_mask, dtype=bool)
    selected_indices = np.flatnonzero(selection)
    if selected_indices.size != int(selection.sum()):
        raise ValueError("Internal error constructing selected indices.")

    mask_high = np.asarray(spec.mask_high, dtype=bool)
    if mask_high.shape != (selected_indices.size,):
        raise ValueError("Spec mask_high has unexpected shape.")
    high_indices = selected_indices[mask_high]

    sort_idx = np.asarray(spec.sort_indices, dtype=np.int64)
    if sort_idx.shape != (high_indices.size,):
        raise ValueError("Spec sort_indices has unexpected shape.")
    high_indices_sorted = high_indices[sort_idx]

    member_offsets = _bin_member_indices_from_spec(
        spec=spec, bin_index=bin_index
    )
    member_indices = high_indices_sorted[member_offsets]

    u_re = np.asarray(fft.u_re, dtype=np.float64)
    u_im = np.asarray(fft.u_im, dtype=np.float64)
    u_complex = u_re + 1j * u_im
    return np.asarray(u_complex[member_indices], dtype=np.complex128)


def _aggregate_u_bin(u_stack: np.ndarray) -> np.ndarray:
    """Aggregate a stack of U matrices into a single U_bin with U_bin U_bin^H = sum U U^H."""
    y_sum = sum_wishart_outer_products(u_stack)
    eigvals, eigvecs = np.linalg.eigh(y_sum)
    eigvals = np.clip(eigvals.real, a_min=0.0, a_max=None)
    sqrt_eig = np.sqrt(eigvals).astype(np.float64)
    return (eigvecs * sqrt_eig[np.newaxis, :]).astype(np.complex128)


def _pack_bin_params(
    *,
    log_delta_sq: np.ndarray,  # (p,)
    theta_re: np.ndarray,  # (n_theta,)
    theta_im: np.ndarray,  # (n_theta,)
) -> np.ndarray:
    log_delta_sq = np.asarray(log_delta_sq, dtype=np.float64)
    theta_re = np.asarray(theta_re, dtype=np.float64)
    theta_im = np.asarray(theta_im, dtype=np.float64)
    return np.concatenate([log_delta_sq, theta_re, theta_im], axis=0)


def _unpack_bin_params(
    *,
    params: np.ndarray,
    n_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    n_theta = int(n_dim * (n_dim - 1) / 2)
    expected = int(n_dim + 2 * n_theta)
    if params.shape != (expected,):
        raise ValueError(
            f"params has unexpected shape {params.shape}; expected ({expected},)."
        )
    log_delta_sq = params[:n_dim]
    theta_re = params[n_dim : n_dim + n_theta]
    theta_im = params[n_dim + n_theta :]
    return log_delta_sq, theta_re, theta_im


def _finite_difference_kappa(
    ll_plus: float,
    ll_0: float,
    ll_minus: float,
    eps: float,
) -> float:
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    return float((ll_plus - 2.0 * ll_0 + ll_minus) / (eps * eps))


def bin_doubling_stiffness_check_from_u_stacks(
    *,
    u_stack_small: np.ndarray,  # (N, p, p)
    u_stack_large: np.ndarray,  # (M, p, p)
    nu: float,
    log_delta_sq_bin: np.ndarray,  # (p,)
    theta_re_bin: np.ndarray | None = None,  # (n_theta,)
    theta_im_bin: np.ndarray | None = None,  # (n_theta,)
    direction: np.ndarray | None = None,  # (p + 2*n_theta,)
    epsilons: tuple[float, ...] = (1e-3, 1e-2, 5e-2),
) -> BinDoublingStiffnessResult:
    """Finite-difference stiffness proxy comparing a bin vs a doubled member set.

    The log-likelihood is evaluated using aggregated U statistics and the model's
    scaling (freq_weights=freq_bin_counts=n_members). The same parameter vector is
    used for both (small, large) to isolate the expected N_h scaling.
    """
    u_stack_small = np.asarray(u_stack_small, dtype=np.complex128)
    u_stack_large = np.asarray(u_stack_large, dtype=np.complex128)
    if u_stack_small.ndim != 3 or u_stack_large.ndim != 3:
        raise ValueError("u_stacks must have shape (N, p, p).")

    n_small = int(u_stack_small.shape[0])
    n_large = int(u_stack_large.shape[0])
    if n_small <= 0 or n_large <= 0:
        raise ValueError("u_stacks must be non-empty.")
    if u_stack_small.shape[1:] != u_stack_large.shape[1:]:
        raise ValueError("u_stack_small/u_stack_large dimension mismatch.")

    n_dim = int(u_stack_small.shape[1])
    n_theta = int(n_dim * (n_dim - 1) / 2)

    log_delta_sq_bin = np.asarray(log_delta_sq_bin, dtype=np.float64)
    if log_delta_sq_bin.shape != (n_dim,):
        raise ValueError("log_delta_sq_bin must have shape (n_dim,).")

    if theta_re_bin is None:
        theta_re_bin = np.zeros((n_theta,), dtype=np.float64)
    if theta_im_bin is None:
        theta_im_bin = np.zeros((n_theta,), dtype=np.float64)
    theta_re_bin = np.asarray(theta_re_bin, dtype=np.float64)
    theta_im_bin = np.asarray(theta_im_bin, dtype=np.float64)
    if theta_re_bin.shape != (n_theta,) or theta_im_bin.shape != (n_theta,):
        raise ValueError("theta bins must have shape (n_theta,).")

    params0 = _pack_bin_params(
        log_delta_sq=log_delta_sq_bin,
        theta_re=theta_re_bin,
        theta_im=theta_im_bin,
    )

    if direction is None:
        direction = np.zeros_like(params0)
        direction[n_dim - 1] = 1.0
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != params0.shape:
        raise ValueError("direction has unexpected shape.")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0 or not np.isfinite(direction_norm):
        raise ValueError("direction must have non-zero finite norm.")
    direction = direction / direction_norm

    u_bin_small = _aggregate_u_bin(u_stack_small)
    u_bin_large = _aggregate_u_bin(u_stack_large)

    def _ll(u_bin: np.ndarray, n_members: int, params: np.ndarray) -> float:
        ld, tr, ti = _unpack_bin_params(params=params, n_dim=n_dim)
        return multivar_cholesky_loglik(
            u_re=u_bin.real[None, :, :],
            u_im=u_bin.imag[None, :, :],
            log_delta_sq=ld[None, :],
            theta_re=tr[None, :],
            theta_im=ti[None, :],
            nu=float(nu),
            freq_weights=np.array([float(n_members)], dtype=np.float64),
            freq_bin_counts=np.array([float(n_members)], dtype=np.float64),
        )

    ll0_small = _ll(u_bin_small, n_small, params0)
    ll0_large = _ll(u_bin_large, n_large, params0)

    expected_ratio = float(n_large / n_small)
    eps_results: list[BinDoublingStiffnessEpsResult] = []
    for eps in epsilons:
        eps = float(eps)
        params_plus = params0 + eps * direction
        params_minus = params0 - eps * direction

        llp_small = _ll(u_bin_small, n_small, params_plus)
        llm_small = _ll(u_bin_small, n_small, params_minus)
        llp_large = _ll(u_bin_large, n_large, params_plus)
        llm_large = _ll(u_bin_large, n_large, params_minus)

        kappa_small = _finite_difference_kappa(
            llp_small, ll0_small, llm_small, eps
        )
        kappa_large = _finite_difference_kappa(
            llp_large, ll0_large, llm_large, eps
        )
        ratio = (
            float(kappa_large / kappa_small) if kappa_small != 0.0 else np.nan
        )
        eps_results.append(
            BinDoublingStiffnessEpsResult(
                eps=float(eps),
                kappa_small=float(kappa_small),
                kappa_large=float(kappa_large),
                ratio=float(ratio),
                expected_ratio=float(expected_ratio),
            )
        )

    return BinDoublingStiffnessResult(
        bin_index=-1,
        n_members_small=n_small,
        n_members_large=n_large,
        eps_results=tuple(eps_results),
    )


def bin_doubling_stiffness_check(
    *,
    fft_fine: MultivarFFT,
    spec: CoarseGrainSpec,
    bin_index: int,
    nu: int,
    log_delta_sq_bin: np.ndarray,  # (p,)
    theta_re_bin: np.ndarray | None = None,  # (n_theta,)
    theta_im_bin: np.ndarray | None = None,  # (n_theta,)
    direction: np.ndarray | None = None,
    epsilons: tuple[float, ...] = (1e-3, 1e-2, 5e-2),
) -> BinDoublingStiffnessResult:
    """Run stiffness proxy on a coarse bin and its adjacent union (bin_index+1)."""
    if bin_index < 0 or bin_index + 1 >= int(spec.n_bins_high):
        raise ValueError("bin_index must allow an adjacent (bin_index+1) bin.")

    u_stack_small = _extract_u_stack_for_bin(
        fft_fine, spec, bin_index=bin_index
    )
    u_stack_next = _extract_u_stack_for_bin(
        fft_fine, spec, bin_index=bin_index + 1
    )
    u_stack_large = np.concatenate([u_stack_small, u_stack_next], axis=0)

    res = bin_doubling_stiffness_check_from_u_stacks(
        u_stack_small=u_stack_small,
        u_stack_large=u_stack_large,
        nu=float(nu),
        log_delta_sq_bin=log_delta_sq_bin,
        theta_re_bin=theta_re_bin,
        theta_im_bin=theta_im_bin,
        direction=direction,
        epsilons=epsilons,
    )
    return BinDoublingStiffnessResult(
        bin_index=int(bin_index),
        n_members_small=int(res.n_members_small),
        n_members_large=int(res.n_members_large),
        eps_results=res.eps_results,
    )


def multivar_cholesky_loglik(
    *,
    u_re: np.ndarray,  # (F, p, p)
    u_im: np.ndarray,  # (F, p, p)
    log_delta_sq: np.ndarray,  # (F, p)
    theta_re: np.ndarray,  # (F, n_theta)
    theta_im: np.ndarray,  # (F, n_theta)
    nu: float,
    freq_weights: np.ndarray,  # (F,)
    freq_bin_counts: np.ndarray,  # (F,)
) -> float:
    """Compute the multivariate log-likelihood used by the Cholesky Whittle model.

    This mirrors the scaling structure used in the NumPyro multivariate models:
      - a log-det term weighted by freq_weights
      - a quadratic term using per-bin mean sufficient statistics via freq_bin_counts
    """
    u_re = np.asarray(u_re, dtype=np.float64)
    u_im = np.asarray(u_im, dtype=np.float64)
    u_complex = (u_re + 1j * u_im).astype(np.complex128)

    n_freq, n_dim, _ = u_complex.shape
    log_delta_sq = np.asarray(log_delta_sq, dtype=np.float64)
    if log_delta_sq.shape != (n_freq, n_dim):
        raise ValueError("log_delta_sq has unexpected shape.")

    theta_re = np.asarray(theta_re, dtype=np.float64)
    theta_im = np.asarray(theta_im, dtype=np.float64)
    theta_complex = (theta_re + 1j * theta_im).astype(np.complex128)

    n_theta = int(n_dim * (n_dim - 1) / 2)
    if theta_complex.shape != (n_freq, n_theta):
        raise ValueError("theta arrays have unexpected shape.")

    fw = np.asarray(freq_weights, dtype=np.float64)
    if fw.shape != (n_freq,):
        raise ValueError("freq_weights has unexpected shape.")

    bc = np.asarray(freq_bin_counts, dtype=np.float64)
    if bc.shape != (n_freq,):
        raise ValueError("freq_bin_counts has unexpected shape.")
    bc = np.maximum(bc, 1.0)

    u_resid = u_complex.copy()
    idx = 0
    for row in range(1, n_dim):
        count = row
        coeff = theta_complex[:, idx : idx + count]  # (F, row)
        prev = u_resid[:, :row, :]  # (F, row, p)
        contrib = np.einsum("fl,flr->fr", coeff, prev)
        u_resid[:, row, :] = u_resid[:, row, :] - contrib
        idx += count

    residual_power_sum = np.sum(np.abs(u_resid) ** 2, axis=2)  # (F, p)
    residual_power_mean = residual_power_sum / bc[:, None]

    exp_neg_log_delta = np.exp(-log_delta_sq)
    sum_log_det = -float(nu) * float(np.sum(fw[:, None] * log_delta_sq))
    quad = -float(
        np.sum(fw[:, None] * residual_power_mean * exp_neg_log_delta)
    )
    return float(sum_log_det + quad)


def coarse_bin_likelihood_equivalence_check(
    *,
    fft_fine: MultivarFFT,
    spec: CoarseGrainSpec,
    bin_index: int,
    nu: int,
    log_delta_sq_bin: np.ndarray,  # (p,)
    theta_re_bin: np.ndarray | None = None,  # (n_theta,)
    theta_im_bin: np.ndarray | None = None,  # (n_theta,)
    include_theta: bool = False,
) -> CoarseBinEquivalenceResult:
    """Check that sum-of-fine log-likelihood matches coarse-bin log-likelihood.

    The check assumes parameters are (approximately) constant within the bin,
    and compares:
      ll_fine  = sum_{k in J_h} ll(S_h; Y_k)   using per-frequency u matrices
      ll_coarse = ll(S_h; Ybar_h)             using aggregated u_bin and weights

    Using the model's scaling, a correct implementation should satisfy
    ll_coarse ≈ ll_fine (up to bin-constant offsets), and in particular should
    not exhibit an extra factor of N_h.
    """
    u_stack = _extract_u_stack_for_bin(fft_fine, spec, bin_index=bin_index)
    n_h = int(u_stack.shape[0])
    if n_h <= 0:
        raise ValueError("Empty bin.")

    n_dim = int(fft_fine.n_dim)
    n_theta = int(n_dim * (n_dim - 1) / 2)

    log_delta_sq_bin = np.asarray(log_delta_sq_bin, dtype=np.float64)
    if log_delta_sq_bin.shape != (n_dim,):
        raise ValueError("log_delta_sq_bin must have shape (n_dim,).")

    if include_theta:
        if theta_re_bin is None or theta_im_bin is None:
            raise ValueError(
                "theta_re_bin/theta_im_bin are required when include_theta=True."
            )
        theta_re_bin = np.asarray(theta_re_bin, dtype=np.float64)
        theta_im_bin = np.asarray(theta_im_bin, dtype=np.float64)
        if theta_re_bin.shape != (n_theta,) or theta_im_bin.shape != (
            n_theta,
        ):
            raise ValueError("theta bins must have shape (n_theta,).")
    else:
        theta_re_bin = np.zeros((n_theta,), dtype=np.float64)
        theta_im_bin = np.zeros((n_theta,), dtype=np.float64)

    # Fine: treat each fine frequency as its own bin with weight=1, count=1.
    u_fine = u_stack
    ll_fine = multivar_cholesky_loglik(
        u_re=u_fine.real,
        u_im=u_fine.imag,
        log_delta_sq=np.repeat(log_delta_sq_bin[None, :], n_h, axis=0),
        theta_re=np.repeat(theta_re_bin[None, :], n_h, axis=0),
        theta_im=np.repeat(theta_im_bin[None, :], n_h, axis=0),
        nu=float(nu),
        freq_weights=np.ones((n_h,), dtype=np.float64),
        freq_bin_counts=np.ones((n_h,), dtype=np.float64),
    )

    # Coarse: aggregate U across the bin, and apply the N_h scaling via weights/counts.
    u_bin = _aggregate_u_bin(u_stack)
    ll_coarse = multivar_cholesky_loglik(
        u_re=u_bin.real[None, :, :],
        u_im=u_bin.imag[None, :, :],
        log_delta_sq=log_delta_sq_bin[None, :],
        theta_re=theta_re_bin[None, :],
        theta_im=theta_im_bin[None, :],
        nu=float(nu),
        freq_weights=np.array([float(n_h)], dtype=np.float64),
        freq_bin_counts=np.array([float(n_h)], dtype=np.float64),
    )

    return CoarseBinEquivalenceResult(
        bin_index=int(bin_index),
        n_members=int(n_h),
        ll_fine=float(ll_fine),
        ll_coarse=float(ll_coarse),
    )


__all__ = [
    "BinDoublingStiffnessEpsResult",
    "BinDoublingStiffnessResult",
    "CoarseBinEquivalenceResult",
    "bin_doubling_stiffness_check",
    "bin_doubling_stiffness_check_from_u_stacks",
    "coarse_bin_likelihood_equivalence_check",
    "multivar_cholesky_loglik",
]
