"""Sampler-independent preprocessing diagnostics.

These helpers are intended to be cheap checks that run before model fitting.
They avoid JAX dependencies and operate on NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from log_psplines.data.spectral_utils import psd_to_cholesky_components
from log_psplines.preprocessing.knot_locator import denoise_score


@dataclass(frozen=True)
class EigenvalueSeparationDiagnostics:
    """Eigenvalue separation diagnostics for multivariate spectral matrices.

    Shapes:
        freq: (N,)
        eigvals_desc: (N, p) ordered λ1 ≥ … ≥ λd
        ratios: adjacent ratios r_12 = λ2/λ1, ..., shape (N,) per key
        mask: (N,) bins retained for summaries (all True if unused)
    """

    freq: np.ndarray
    eigvals_desc: np.ndarray
    ratios: dict[str, np.ndarray]
    mask: np.ndarray
    lambda1_cutoff: float | None = None

    def ratio_summary(self, *, warn_threshold: float = 0.8) -> dict[str, str]:
        summaries: dict[str, str] = {}
        for key, ratio in self.ratios.items():
            summaries[key] = ratio_summary_string(
                key,
                ratio[self.mask],
                warn_threshold=warn_threshold,
            )
        return summaries

    def worst_frequencies(
        self, *, top_k: int = 10, warn_threshold: float | None = None
    ) -> dict[str, list[tuple[float, float]]]:
        worst: dict[str, list[tuple[float, float]]] = {}
        for key, ratio in self.ratios.items():
            worst[key] = worst_ratio_frequencies(
                self.freq, ratio, top_k=top_k, mask=self.mask
            )
        if warn_threshold is not None:
            worst = {
                key: items
                for key, items in worst.items()
                if np.any(self.ratios[key][self.mask] > float(warn_threshold))
            }
        return worst


def ordered_eigvals_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Return ordered eigenvalues (descending) for each frequency bin.

    Args:
        matrix: (N, p, p), approximately Hermitian.

    Returns:
        (N, p) with λ1 ≥ … ≥ λd, clipped to be non-negative.
    """

    matrix = np.asarray(matrix)
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"matrix must have shape (N, p, p); got {matrix.shape}."
        )
    herm = 0.5 * (matrix + np.swapaxes(np.conj(matrix), -1, -2))
    eig = np.linalg.eigvalsh(herm)  # ascending
    eig = np.maximum(eig.real, 0.0)
    return eig[:, ::-1]


def eig_ratios(
    eigvals_desc: np.ndarray, *, eps: float | None = None
) -> dict[str, np.ndarray]:
    """Compute adjacent eigenvalue ratios λ_{i+1}/λ_i.

    Args:
        eigvals_desc: (N, p) ordered descending.
        eps: Denominator cutoff; values with λ_i <= eps become NaN.

    Returns:
        Dict mapping "r_12", "r_23", ... to arrays of shape (N,) clipped
        to [0, 1].
    """

    eigvals_desc = np.asarray(eigvals_desc, dtype=np.float64)
    if eigvals_desc.ndim != 2:
        raise ValueError(
            f"eigvals_desc must have shape (N, p); got {eigvals_desc.shape}."
        )
    if eps is None:
        eps = float(np.finfo(np.float64).tiny)

    p = int(eigvals_desc.shape[1])
    ratios: dict[str, np.ndarray] = {}
    for idx in range(p - 1):
        den = eigvals_desc[:, idx]
        num = eigvals_desc[:, idx + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = num / den
        ratio = np.where(den <= eps, np.nan, ratio)
        ratios[f"r_{idx + 1}{idx + 2}"] = np.clip(ratio, 0.0, 1.0)
    return ratios


def ratio_summary_string(
    name: str, ratio: np.ndarray, *, warn_threshold: float = 0.8
) -> str:
    ratio = np.asarray(ratio, dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return f"{name}: (no finite values)"
    frac = float(np.mean(ratio > float(warn_threshold)))
    rmin = float(ratio.min())
    rmax = float(ratio.max())
    if np.isclose(rmin, rmax):
        return (
            f"{name}: constant={rmin:.3f}, "
            f"frac(>{warn_threshold:.2f})={frac * 100:.1f}%"
        )

    p05, p50, p95 = np.percentile(ratio, [5.0, 50.0, 95.0])
    return (
        f"{name}: q05/50/95={p05:.3f}/{p50:.3f}/{p95:.3f}, "
        f"range={rmin:.3f}-{rmax:.3f}, "
        f"frac(>{warn_threshold:.2f})={frac * 100:.1f}%"
    )


def worst_ratio_frequencies(
    freq: np.ndarray,
    ratio: np.ndarray,
    *,
    top_k: int = 10,
    mask: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Return the (frequency, ratio) pairs for the largest ratios."""

    top_k = int(top_k)
    if top_k <= 0:
        return []

    freq = np.asarray(freq, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    if freq.shape != ratio.shape:
        raise ValueError(
            f"freq and ratio must have the same shape; got {freq.shape} and {ratio.shape}."
        )

    good = np.isfinite(ratio)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != good.shape:
            raise ValueError(
                f"mask must have shape {good.shape}; got {mask.shape}."
            )
        good &= mask

    vals = np.where(good, ratio, -np.inf)
    idx = np.argsort(vals)[::-1][:top_k]
    idx = idx[np.isfinite(vals[idx])]
    return [(float(freq[i]), float(vals[i])) for i in idx]


def eigenvalue_separation_diagnostics(
    *,
    freq: np.ndarray,
    matrix: np.ndarray,
    min_lambda1_quantile: float = 0.0,
    eps: float | None = None,
) -> EigenvalueSeparationDiagnostics:
    """Compute eigenvalue separation diagnostics for a spectral matrix.

    Args:
        freq: (N,) frequency grid.
        matrix: (N, p, p) complex Hermitian spectral matrix.
        min_lambda1_quantile: If > 0, compute summaries only over bins where
            λ1 is above this quantile (useful for de-emphasizing deep notches).
        eps: Denominator cutoff for ratio computation.
    """

    freq = np.asarray(freq, dtype=float)
    eig_desc = ordered_eigvals_hermitian(matrix)
    if freq.shape != (eig_desc.shape[0],):
        raise ValueError(
            f"freq must have shape ({eig_desc.shape[0]},); got {freq.shape}."
        )

    mask = np.ones((freq.size,), dtype=bool)
    cutoff: float | None = None
    q = float(min_lambda1_quantile)
    if q > 0.0:
        if not (0.0 < q < 1.0):
            raise ValueError("min_lambda1_quantile must be in (0, 1).")
        cutoff = float(np.quantile(eig_desc[:, 0], q))
        mask = eig_desc[:, 0] > cutoff

    ratios = eig_ratios(eig_desc, eps=eps)
    return EigenvalueSeparationDiagnostics(
        freq=freq,
        eigvals_desc=eig_desc,
        ratios=ratios,
        mask=mask,
        lambda1_cutoff=cutoff,
    )


def model_component_curves(
    freq: np.ndarray,
    matrix: np.ndarray,
    *,
    cholesky_jitter: float = 1e-12,
    max_cholesky_jitter: float = 1e-4,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return raw and denoised model components for a pre-fit plot.

    ``freq`` has shape (N,) and ``matrix`` has shape (N, p, p).
    Values retain the same Cholesky transform and smoothing as knot selection.
    """
    freq = np.asarray(freq, dtype=float)
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 3 or matrix.shape[0] != freq.size:
        raise ValueError("matrix must have shape (N, p, p) with N=len(freq).")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must be square per frequency.")

    log_delta_sq, theta = psd_to_cholesky_components(
        matrix,
        cholesky_jitter=cholesky_jitter,
        max_cholesky_jitter=max_cholesky_jitter,
    )
    p = int(log_delta_sq.shape[1])
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in range(p):
        curves[f"LogDelta{row + 1}{row + 1}"] = (
            log_delta_sq[:, row],
            denoise_score(log_delta_sq[:, row], freq),
        )
        for col in range(row + 1, p):
            re = np.real(theta[:, col, row])
            im = np.imag(theta[:, col, row])
            curves[f"Re(Theta{row + 1}{col + 1})"] = (
                re,
                denoise_score(re, freq),
            )
            curves[f"Im(Theta{col + 1}{row + 1})"] = (
                im,
                denoise_score(im, freq),
            )
    return curves


def extract_component_knots(
    spline_model: object,
    freq: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract per-component knot positions from a SpectralComponents model.

    Returns a dict whose keys match the subplot labels used by
    :func:`log_psplines.plotting.preprocessing.plot_eigenvalue_separation` (e.g. ``"LogDelta11"``,
    ``"Re(Theta12)"``, ``"Im(Theta21)"``), and whose values are 1-D arrays
    of knot positions in frequency space.

    Args:
        spline_model: A ``SpectralComponents`` instance.
        freq: Frequency array (same grid the model was built on).

    Returns:
        Mapping from component label to knot frequency array.
    """
    freq = np.asarray(freq, dtype=float)
    f_min, f_max = float(freq[0]), float(freq[-1])
    result: dict[str, np.ndarray] = {}

    p = int(spline_model.p)

    # Diagonal: LogDelta{j+1}{j+1}
    for j in range(p):
        model = spline_model.diagonal_models[j]
        knots_norm = np.asarray(model.knots, dtype=float)
        result[f"LogDelta{j + 1}{j + 1}"] = f_min + knots_norm * (
            f_max - f_min
        )

    # Off-diagonal theta pairs
    theta_pairs = [
        (row_idx, col_idx)
        for row_idx in range(1, p)
        for col_idx in range(row_idx)
    ]
    for row_idx, col_idx in theta_pairs:
        # Re(Theta) — shown in upper triangle where row < col,
        # with row=col_idx, col=row_idx (since col_idx < row_idx).
        re_model = spline_model.get_theta_model("re", row_idx, col_idx)
        knots_norm = np.asarray(re_model.knots, dtype=float)
        re_freq = f_min + knots_norm * (f_max - f_min)
        result[f"Re(Theta{col_idx + 1}{row_idx + 1})"] = re_freq

        # Im(Theta) — shown in lower triangle where row > col,
        # with row=row_idx, col=col_idx (since row_idx > col_idx).
        im_model = spline_model.get_theta_model("im", row_idx, col_idx)
        knots_norm = np.asarray(im_model.knots, dtype=float)
        im_freq = f_min + knots_norm * (f_max - f_min)
        result[f"Im(Theta{row_idx + 1}{col_idx + 1})"] = im_freq

    return result
