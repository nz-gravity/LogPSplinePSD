"""Sampler-independent preprocessing diagnostics.

These helpers are intended to be cheap checks that run before model fitting.
They avoid JAX dependencies and operate on NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
            "matrix must have shape (N, p, p); " f"got {matrix.shape}."
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
            "eigvals_desc must have shape (N, p); "
            f"got {eigvals_desc.shape}."
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
        ratios[f"r_{idx+1}{idx+2}"] = np.clip(ratio, 0.0, 1.0)
    return ratios


def ratio_summary_string(
    name: str, ratio: np.ndarray, *, warn_threshold: float = 0.8
) -> str:
    ratio = np.asarray(ratio, dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return f"{name}: (no finite values)"
    p05, p50, p95 = np.percentile(ratio, [5.0, 50.0, 95.0])
    frac = float(np.mean(ratio > float(warn_threshold)))
    return (
        f"{name}: p05={p05:.3f}, p50={p50:.3f}, p95={p95:.3f}, "
        f"min={ratio.min():.3f}, max={ratio.max():.3f}, "
        f"frac(>{warn_threshold:.2f})={frac*100:.1f}%"
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


def save_eigenvalue_separation_plot(
    diag: EigenvalueSeparationDiagnostics,
    out: str,
    *,
    warn_threshold: float = 0.8,
    dpi: int = 200,
) -> None:
    """Save an eigenvalue separation plot (ratios + eigenvalues).

    Args:
        diag: Diagnostics output from :func:`eigenvalue_separation_diagnostics`.
        out: Output image path.
        warn_threshold: Horizontal threshold shown on ratio panel.
        dpi: Figure DPI for saved image.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to save eigenvalue separation plots."
        ) from e

    freq = np.asarray(diag.freq, dtype=float)
    eig = np.asarray(diag.eigvals_desc, dtype=float)
    ratios = {k: np.asarray(v, dtype=float) for k, v in diag.ratios.items()}

    use_log_x = bool(freq.size) and np.all(freq > 0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_ratio, ax_eig = axes

    # Ratios
    for key, ratio in ratios.items():
        if use_log_x:
            ax_ratio.semilogx(freq, ratio, label=key.replace("r_", "r"))
        else:
            ax_ratio.plot(freq, ratio, label=key.replace("r_", "r"))
    ax_ratio.axhline(
        float(warn_threshold),
        color="k",
        ls="--",
        lw=1,
        alpha=0.6,
        label="warn threshold",
    )
    ax_ratio.set_ylabel("Eigenvalue ratio")
    ax_ratio.set_ylim(0.0, 1.02)
    ax_ratio.grid(True, which="both", alpha=0.2)
    ax_ratio.legend(fontsize=8)
    ax_ratio.set_title("Eigenvalue separation ratios")

    # Eigenvalues
    p = int(eig.shape[1]) if eig.ndim == 2 else 0
    for idx in range(p):
        if use_log_x:
            ax_eig.loglog(freq, eig[:, idx], label=f"λ{idx+1}")
        else:
            ax_eig.semilogy(freq, eig[:, idx], label=f"λ{idx+1}")
    ax_eig.set_xlabel("Frequency")
    ax_eig.set_ylabel("Eigenvalue scale")
    ax_eig.grid(True, which="both", alpha=0.2)
    if p:
        ax_eig.legend(fontsize=8, ncol=min(3, p))

    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


__all__ = [
    "EigenvalueSeparationDiagnostics",
    "eig_ratios",
    "eigenvalue_separation_diagnostics",
    "ordered_eigvals_hermitian",
    "ratio_summary_string",
    "save_eigenvalue_separation_plot",
    "worst_ratio_frequencies",
]
