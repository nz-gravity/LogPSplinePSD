"""Shared helpers for diagnostics modules."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import numpy as np
from scipy.integrate import simpson


def interior_frequency_slice(n_freq: int) -> slice:
    """Return a slice that drops first/last frequency bins when possible."""
    return slice(1, -1) if n_freq > 3 else slice(None)


def as_scalar(value: Any) -> Optional[float]:
    """Best-effort conversion to a Python float."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass

    try:
        arr = np.asarray(value)
        if arr.size:
            return float(arr.reshape(-1)[0])
    except Exception:
        return None
    return None


def khat_status(khat_max: Optional[float]) -> Tuple[Optional[float], str]:
    """Map PSIS k-hat into a numeric status and human-readable label."""
    if khat_max is None or not np.isfinite(khat_max):
        return None, "unknown"
    if khat_max < 0.5:
        return 0.0, "ok"
    if khat_max <= 0.7:
        return 1.0, "warn"
    return 2.0, "fail"


def compute_riae(
    median_psd: np.ndarray, true_psd: np.ndarray, freqs: Iterable[float]
) -> float:
    """Relative integrated absolute error (univariate)."""
    freqs_arr = np.asarray(freqs, dtype=float)
    numerator = float(simpson(np.abs(median_psd - true_psd), x=freqs_arr))
    denominator = float(simpson(true_psd, x=freqs_arr))
    return float(numerator / denominator) if denominator != 0 else float("nan")


def compute_matrix_riae(
    median_psd_matrix: np.ndarray,
    true_psd_matrix: np.ndarray,
    freqs: Iterable[float],
) -> float:
    """RIAE for multivariate PSD matrices using the Frobenius norm."""
    freqs_arr = np.asarray(freqs, dtype=float)
    diff_frobenius = np.array(
        [
            np.linalg.norm(median_psd_matrix[k] - true_psd_matrix[k], "fro")
            for k in range(len(freqs_arr))
        ]
    )
    true_frobenius = np.array(
        [
            np.linalg.norm(true_psd_matrix[k], "fro")
            for k in range(len(freqs_arr))
        ]
    )
    numerator = float(simpson(diff_frobenius, x=freqs_arr))
    denominator = float(simpson(true_frobenius, x=freqs_arr))
    return float(numerator / denominator) if denominator != 0 else float("nan")


def compute_matrix_l2(
    median_psd_matrix: np.ndarray,
    true_psd_matrix: np.ndarray,
    freqs: Iterable[float],
) -> float:
    """Relative integrated L2 error for multivariate PSD matrices.

    Computed as::

        sqrt( integral ||S_hat(f) - S_true(f)||_F^2 df )
        / sqrt( integral ||S_true(f)||_F^2 df )
    """
    freqs_arr = np.asarray(freqs, dtype=float)
    diff_sq = np.array(
        [
            np.linalg.norm(median_psd_matrix[k] - true_psd_matrix[k], "fro")
            ** 2
            for k in range(len(freqs_arr))
        ]
    )
    true_sq = np.array(
        [
            np.linalg.norm(true_psd_matrix[k], "fro") ** 2
            for k in range(len(freqs_arr))
        ]
    )
    numerator = float(np.sqrt(max(simpson(diff_sq, x=freqs_arr), 0.0)))
    denominator = float(np.sqrt(max(simpson(true_sq, x=freqs_arr), 0.0)))
    return float(numerator / denominator) if denominator != 0 else float("nan")


def compute_riae_errorbars(
    psd_samples: np.ndarray, true_psd: np.ndarray, freqs: Iterable[float]
) -> dict:
    """Quantiles of RIAE across a collection of PSD samples."""
    riae_samples = [
        compute_riae(psd, true_psd, freqs) for psd in np.asarray(psd_samples)
    ]
    arr = np.asarray(riae_samples, dtype=float)
    return {
        "q05": float(np.percentile(arr, 5)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "q95": float(np.percentile(arr, 95)),
    }


def compute_ci_coverage_univar(
    psd_samples: np.ndarray, true_psd: np.ndarray
) -> float:
    """Compute 90% credible interval coverage for univariate PSD."""
    arr = np.asarray(psd_samples)
    if arr.ndim == 2 and arr.shape[0] == 3:
        posterior_lower = arr[0]
        posterior_upper = arr[-1]
    else:
        posterior_lower = np.percentile(arr, 5.0, axis=0)
        posterior_upper = np.percentile(arr, 95.0, axis=0)
    coverage = np.mean(
        (true_psd >= posterior_lower) & (true_psd <= posterior_upper)
    )
    return float(coverage)


def compute_ci_coverage_multivar(
    psd_matrix_samples: np.ndarray, true_psd_real: np.ndarray
) -> float:
    """Compute 90% credible interval coverage for multivariate PSD matrices."""
    true_psd_arr = np.asarray(true_psd_real)
    true_psd = np.zeros(true_psd_arr.shape, dtype=np.float64)
    for i in range(true_psd_arr.shape[0]):
        true_psd[i] = _complex_to_real(true_psd_arr[i])

    arr = np.asarray(psd_matrix_samples)
    if arr.ndim == 4 and arr.shape[0] == 3:
        posterior_lower_raw = arr[0]
        posterior_upper_raw = arr[-1]
        posterior_lower = np.zeros(posterior_lower_raw.shape, dtype=np.float64)
        posterior_upper = np.zeros(posterior_upper_raw.shape, dtype=np.float64)
        for i in range(posterior_lower_raw.shape[0]):
            posterior_lower[i] = _complex_to_real(posterior_lower_raw[i])
            posterior_upper[i] = _complex_to_real(posterior_upper_raw[i])
    else:
        if np.iscomplexobj(arr):
            psd_matrix_real = np.zeros_like(arr, dtype=np.float64)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    psd_matrix_real[i, j] = _complex_to_real(arr[i, j])
        else:
            psd_matrix_real = np.asarray(arr, dtype=np.float64)
        posterior_lower = np.percentile(psd_matrix_real, 5.0, axis=0)
        posterior_upper = np.percentile(psd_matrix_real, 95.0, axis=0)

    coverage = np.mean(
        (true_psd >= posterior_lower) & (true_psd <= posterior_upper)
    )
    return float(coverage)


def compute_ci_coverage_multivar_detailed(
    psd_matrix_samples: np.ndarray,
    true_psd_real: np.ndarray,
) -> dict[str, float]:
    """Compute 90% CI coverage broken down by element type.

    Returns a dict with keys:
    - ``overall``       : same value as :func:`compute_ci_coverage_multivar`
    - ``diag``          : coverage over diagonal (auto-spectral) real parts
    - ``offdiag_re``    : coverage over upper-triangle real parts (cross-spectral Re)
    - ``offdiag_im``    : coverage over lower-triangle imaginary parts (cross-spectral Im)
    - ``n_diag``        : number of (freq, element) pairs in diagonal
    - ``n_offdiag_re``  : number of pairs in real off-diagonal
    - ``n_offdiag_im``  : number of pairs in imaginary off-diagonal

    Parameters
    ----------
    psd_matrix_samples : ndarray, shape (3, F, p, p) or (S, F, p, p)
        Posterior samples or stacked [q05, q50, q95] percentile matrices.
        Complex-valued (q05_re + 1j*q05_im convention) when shape[0] == 3.
    true_psd_real : ndarray, shape (F, p, p)
        True PSD matrix (complex or real-encoded via :func:`_complex_to_real`).
    """
    true_psd_arr = np.asarray(true_psd_real)
    true_enc = np.zeros(true_psd_arr.shape, dtype=np.float64)
    for i in range(true_psd_arr.shape[0]):
        true_enc[i] = _complex_to_real(true_psd_arr[i])

    arr = np.asarray(psd_matrix_samples)
    if arr.ndim == 4 and arr.shape[0] == 3:
        lower_raw = arr[0]
        upper_raw = arr[-1]
        lower = np.zeros(lower_raw.shape, dtype=np.float64)
        upper = np.zeros(upper_raw.shape, dtype=np.float64)
        for i in range(lower_raw.shape[0]):
            lower[i] = _complex_to_real(lower_raw[i])
            upper[i] = _complex_to_real(upper_raw[i])
    else:
        if np.iscomplexobj(arr):
            real_arr = np.zeros_like(arr, dtype=np.float64)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    real_arr[i, j] = _complex_to_real(arr[i, j])
        else:
            real_arr = np.asarray(arr, dtype=np.float64)
        lower = np.percentile(real_arr, 5.0, axis=0)
        upper = np.percentile(real_arr, 95.0, axis=0)

    # covered[f, i, j] = True if true_enc[f, i, j] is inside [lower, upper]
    covered = (true_enc >= lower) & (true_enc <= upper)  # (F, p, p)

    p = covered.shape[-1]
    diag_mask = np.eye(p, dtype=bool)  # diagonal
    upper_tri_mask = np.triu(
        np.ones((p, p), dtype=bool), k=1
    )  # strict upper → Re
    lower_tri_mask = np.tril(
        np.ones((p, p), dtype=bool), k=-1
    )  # strict lower → Im

    cov_diag = float(np.mean(covered[:, diag_mask]))
    cov_re = float(np.mean(covered[:, upper_tri_mask]))
    cov_im = float(np.mean(covered[:, lower_tri_mask]))
    cov_all = float(np.mean(covered))

    F = covered.shape[0]
    return {
        "overall": cov_all,
        "diag": cov_diag,
        "offdiag_re": cov_re,
        "offdiag_im": cov_im,
        "n_diag": int(F * p),
        "n_offdiag_re": int(F * p * (p - 1) // 2),
        "n_offdiag_im": int(F * p * (p - 1) // 2),
    }


def find_posterior_inflation_factor(
    psd_matrix_samples: np.ndarray,
    true_psd_real: np.ndarray,
    *,
    target_coverage: float = 0.90,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> dict[str, float]:
    """Find the posterior inflation factor needed to achieve ``target_coverage``.

    Inflates (or deflates) each posterior sample around the posterior median by
    a scalar factor ``c``, then recomputes coverage.  Binary-searches for the ``c``
    such that coverage ≈ ``target_coverage``.

    Useful for quantifying how miscalibrated the Whittle posterior is: a value
    of ``c`` > 1 means the posterior is too narrow; the CI widths need to be
    multiplied by ``c`` to be well-calibrated.

    Parameters
    ----------
    psd_matrix_samples : ndarray, shape (3, F, p, p)
        Stacked [q05, q50, q95] percentile matrices (complex convention).
    true_psd_real : ndarray, shape (F, p, p)
        True PSD matrix (complex or real-encoded).
    target_coverage : float
        Desired coverage level (default 0.90).
    tol : float
        Convergence tolerance on coverage.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    dict with keys:
    - ``inflation_factor``  : the found c
    - ``achieved_coverage`` : coverage at that c
    - ``n_iter``            : bisection iterations used
    """
    arr = np.asarray(psd_matrix_samples)
    if not (arr.ndim == 4 and arr.shape[0] == 3):
        raise ValueError(
            "psd_matrix_samples must have shape (3, F, p, p) with [q05, q50, q95]."
        )

    q05_raw, q50_raw, q95_raw = arr[0], arr[1], arr[2]

    def _coverage_at_c(c: float) -> float:
        """Inflate CI around median and recompute coverage."""
        inflated_lower = q50_raw + c * (q05_raw - q50_raw)
        inflated_upper = q50_raw + c * (q95_raw - q50_raw)
        inflated_stack = np.stack(
            [inflated_lower, q50_raw, inflated_upper], axis=0
        )
        return compute_ci_coverage_multivar(inflated_stack, true_psd_real)

    # Check that c=1 gives current coverage
    c_low, c_high = 0.0, 20.0
    cov_low = _coverage_at_c(c_low)  # 0 at c=0
    cov_high = _coverage_at_c(c_high)  # should be ~1

    if cov_high < target_coverage:
        return {
            "inflation_factor": c_high,
            "achieved_coverage": cov_high,
            "n_iter": 0,
        }

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        c_mid = 0.5 * (c_low + c_high)
        cov_mid = _coverage_at_c(c_mid)
        if abs(cov_mid - target_coverage) < tol:
            break
        if cov_mid < target_coverage:
            c_low = c_mid
        else:
            c_high = c_mid

    return {
        "inflation_factor": float(c_mid),
        "achieved_coverage": float(cov_mid),
        "n_iter": n_iter,
    }


def extract_percentile(
    values: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    """Return the slice of ``values`` closest to the requested percentile."""
    idx = int(np.argmin(np.abs(np.asarray(percentiles, dtype=float) - target)))
    return values[idx]


def compute_coherence_coverage(
    coherence_quantiles: np.ndarray,
    true_psd: np.ndarray,
    percentiles: np.ndarray,
) -> float:
    """90% CI coverage for off-diagonal coherence.

    Parameters
    ----------
    coherence_quantiles:
        Shape (n_percentiles, F, p, p).  Real-valued coherence samples or
        stacked [q05, q50, q95] slices (shape[0] == 3).
    true_psd:
        True PSD matrix, shape (F, p, p).  May be real or complex; coherence
        is computed as |S_ij|^2 / (S_ii * S_jj).
    percentiles:
        Percentile values associated with the first axis of
        ``coherence_quantiles`` (e.g. [5, 50, 95]).
    """
    arr = np.asarray(coherence_quantiles, dtype=float)
    if arr.ndim != 4 or arr.shape[0] < 2:
        return float("nan")

    if arr.shape[0] == 3 and np.allclose(percentiles[:3], [5.0, 50.0, 95.0]):
        q05 = arr[0]
        q95 = arr[2]
    else:
        idx05 = int(np.argmin(np.abs(percentiles - 5.0)))
        idx95 = int(np.argmin(np.abs(percentiles - 95.0)))
        q05 = arr[idx05]
        q95 = arr[idx95]

    true_arr = np.asarray(true_psd)
    diag = np.real(np.diagonal(true_arr, axis1=1, axis2=2))  # (F, p)
    denom = np.sqrt(np.maximum(diag[..., None] * diag[:, None, :], 0.0))
    true_coh = np.zeros(true_arr.shape[:3], dtype=float)
    valid = denom > 0
    true_coh[valid] = np.abs(true_arr[valid]) ** 2 / denom[valid] ** 2

    p = true_arr.shape[-1]
    offdiag_mask = ~np.eye(p, dtype=bool)
    covered = (true_coh[:, offdiag_mask] >= q05[:, offdiag_mask]) & (
        true_coh[:, offdiag_mask] <= q95[:, offdiag_mask]
    )
    return float(np.mean(covered))


def _complex_to_real(mat: np.ndarray) -> np.ndarray:
    """Convert complex matrices to a real-valued representation for CI checks."""
    arr = np.asarray(mat)
    if not np.iscomplexobj(arr):
        return arr

    n = arr.shape[-1]
    upper = np.triu(np.ones((n, n), dtype=bool))
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)

    out = np.where(upper, arr.real, 0.0)
    out = np.where(lower, arr.imag, out)
    return out
