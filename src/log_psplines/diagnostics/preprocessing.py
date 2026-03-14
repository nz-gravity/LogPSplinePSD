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


def _raw_psd_to_model_components(
    matrix: np.ndarray,
    *,
    cholesky_jitter: float = 1e-12,
    max_cholesky_jitter: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw spectral matrices to model-native components.

    Args:
        matrix: (N, p, p) complex spectral matrices.
        cholesky_jitter: Initial diagonal jitter used if Cholesky fails.
        max_cholesky_jitter: Maximum diagonal jitter before failure.

    Returns:
        log_delta_sq: (N, p), where log_delta_sq[:, j] = log(delta_j^2).
        theta: (N, p, p) complex with lower-triangular model terms
            theta[:, j, l] for j > l and zeros elsewhere.
    """

    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must have shape (N, p, p).")

    if cholesky_jitter < 0.0 or max_cholesky_jitter < 0.0:
        raise ValueError(
            "cholesky_jitter and max_cholesky_jitter must be non-negative."
        )
    if max_cholesky_jitter < cholesky_jitter:
        raise ValueError("max_cholesky_jitter must be >= cholesky_jitter.")

    p = int(matrix.shape[-1])
    herm = 0.5 * (matrix + np.swapaxes(np.conj(matrix), -1, -2))
    eye = np.eye(p, dtype=np.complex128)[None, :, :]

    jitter = float(cholesky_jitter)
    last_error: Exception | None = None
    while True:
        try:
            L = np.linalg.cholesky(herm + jitter * eye)
            break
        except np.linalg.LinAlgError as err:
            last_error = err
            if jitter >= float(max_cholesky_jitter):
                raise np.linalg.LinAlgError(
                    "Cholesky decomposition failed for preprocessing plot "
                    f"up to jitter={max_cholesky_jitter:.3e}."
                ) from last_error
            jitter = max(10.0 * jitter, 1e-16)

    diag_L = np.maximum(
        np.real(L[..., np.arange(p), np.arange(p)]), np.finfo(float).tiny
    )
    log_delta_sq = 2.0 * np.log(diag_L)

    T_inv = L / diag_L[:, np.newaxis, :]
    T = np.linalg.inv(T_inv)
    theta = np.zeros_like(T)
    tril = np.tril_indices(p, k=-1)
    theta[:, tril[0], tril[1]] = -T[:, tril[0], tril[1]]
    return log_delta_sq, theta


def save_eigenvalue_separation_plot(
    diag: EigenvalueSeparationDiagnostics,
    out: str,
    *,
    warn_threshold: float = 0.8,
    cholesky_matrix: np.ndarray | None = None,
    cholesky_jitter: float = 1e-12,
    max_cholesky_jitter: float = 1e-4,
    dpi: int = 200,
) -> None:
    """Save an eigenvalue separation plot (ratios + eigenvalues).

    Args:
        diag: Diagnostics output from :func:`eigenvalue_separation_diagnostics`.
        out: Output image path.
        warn_threshold: Horizontal threshold shown on ratio panel.
        cholesky_matrix: Optional spectral matrix with shape (N, p, p) used to
            plot model-native components (log-delta and theta) across
            frequency.
        cholesky_jitter: Initial diagonal regularization used if Cholesky fails.
        max_cholesky_jitter: Maximum diagonal regularization before giving up.
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
    show_cholesky = cholesky_matrix is not None

    if show_cholesky:
        matrix = np.asarray(cholesky_matrix, dtype=np.complex128)
        if matrix.ndim != 3 or matrix.shape[0] != freq.size:
            raise ValueError(
                "cholesky_matrix must have shape (N, p, p) with N=len(freq)."
            )
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError("cholesky_matrix must be square per frequency.")

        log_delta_sq, theta = _raw_psd_to_model_components(
            matrix,
            cholesky_jitter=cholesky_jitter,
            max_cholesky_jitter=max_cholesky_jitter,
        )
        p_model = int(log_delta_sq.shape[1])

        fig = plt.figure(
            figsize=(max(10.0, 3.2 * p_model), 4.0 + 1.8 * p_model),
            constrained_layout=True,
        )
        grid = fig.add_gridspec(
            nrows=p_model + 2,
            ncols=p_model,
            height_ratios=[1.2, 1.2] + [1.0] * p_model,
        )
        ax_ratio = fig.add_subplot(grid[0, :])
        ax_eig = fig.add_subplot(grid[1, :], sharex=ax_ratio)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax_ratio, ax_eig = axes[0], axes[1]

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

    if show_cholesky:
        assert cholesky_matrix is not None
        p = p_model
        component_axes = []
        n_pairs = p * (p - 1) // 2

        for row in range(p):
            row_axes = []
            for col in range(p):
                ax = fig.add_subplot(grid[row + 2, col], sharex=ax_ratio)

                if row == col:
                    y = log_delta_sq[:, row]
                    label = f"LogDelta{row+1}{col+1}"
                    color = f"C{row % 10}"
                elif row < col:
                    y = np.real(theta[:, col, row])
                    label = f"Re(Theta{row+1}{col+1})"
                    pair_idx = row * (2 * p - row - 1) // 2 + (col - row - 1)
                    color = f"C{(pair_idx + p) % 10}"
                else:
                    y = np.imag(theta[:, row, col])
                    label = f"Im(Theta{row+1}{col+1})"
                    pair_idx = col * (2 * p - col - 1) // 2 + (row - col - 1)
                    color = f"C{(pair_idx + p) % 10}"

                if use_log_x:
                    ax.semilogx(freq, y, color=color, lw=1.2)
                else:
                    ax.plot(freq, y, color=color, lw=1.2)
                ax.set_title(label, fontsize=8)
                ax.grid(True, which="both", alpha=0.2)

                if row < p - 1:
                    ax.tick_params(labelbottom=False)
                if col > 0:
                    ax.tick_params(labelleft=False)

                row_axes.append(ax)
            component_axes.append(row_axes)

        if p > 0:
            component_axes[p - 1][0].set_xlabel("Frequency")
            if p > 1:
                component_axes[p - 1][p - 1].set_xlabel("Frequency")
        if p > 0:
            component_axes[p // 2][0].set_ylabel("Model component value")
        ax_eig.set_title(
            f"Eigenvalue scale (component grid: {p}x{p}, {n_pairs} theta pairs)"
        )
    else:
        fig.tight_layout()

    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


__all__ = [
    "EigenvalueSeparationDiagnostics",
    "_raw_psd_to_model_components",
    "eig_ratios",
    "eigenvalue_separation_diagnostics",
    "ordered_eigvals_hermitian",
    "ratio_summary_string",
    "save_eigenvalue_separation_plot",
    "worst_ratio_frequencies",
]
