"""Figures for data and model checks before fitting."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from log_psplines.preprocessing.diagnostics import (
    EigenvalueSeparationDiagnostics,
)


def plot_eigenvalue_separation(
    diag: EigenvalueSeparationDiagnostics,
    *,
    warn_threshold: float = 0.8,
    info_text: str | None = None,
    excluded_bands: tuple[tuple[float, float], ...] = (),
    component_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    component_knots: dict[str, np.ndarray] | None = None,
) -> Figure:
    """Draw eigenvalue separation and optional model components.

    Args:
        diag: Diagnostics output from :func:`eigenvalue_separation_diagnostics`.
        warn_threshold: Horizontal threshold shown on ratio panel.
        info_text: Optional metadata line shown at the top of the figure.
        excluded_bands: Frequency bands shaded on all panels to show bins
            removed during preprocessing.
        component_curves: Raw and denoised component values by plot label.
        component_knots: Optional mapping from component label to an array of
            knot positions in frequency space.  Keys should match the subplot
            labels used in the component grid, e.g.
            ``{"LogDelta11": array, "Re(Theta12)": array, "Im(Theta21)": array}``.
            When provided, knot locations are drawn as vertical tick marks on
            each matching subplot.
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
    excluded_bands = tuple(
        (float(low), float(high)) for low, high in excluded_bands
    )

    use_log_x = bool(freq.size) and np.all(freq > 0)
    show_cholesky = component_curves is not None

    def _shade_excluded_bands(ax, *, add_label: bool) -> None:
        for idx, (low, high) in enumerate(excluded_bands):
            left = min(low, high)
            right = max(low, high)
            ax.axvspan(
                left,
                right,
                color="#d8b365",
                alpha=0.28,
                zorder=0,
                label="excluded band" if add_label and idx == 0 else None,
            )

    if show_cholesky:
        assert component_curves is not None
        p_model = int(eig.shape[1])

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
    _shade_excluded_bands(ax_ratio, add_label=True)
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
    _shade_excluded_bands(ax_eig, add_label=False)
    p = int(eig.shape[1]) if eig.ndim == 2 else 0
    for idx in range(p):
        if use_log_x:
            ax_eig.loglog(freq, eig[:, idx], label=f"λ{idx + 1}")
        else:
            ax_eig.semilogy(freq, eig[:, idx], label=f"λ{idx + 1}")
    ax_eig.set_xlabel("Frequency")
    ax_eig.set_ylabel("Eigenvalue scale")
    ax_eig.grid(True, which="both", alpha=0.2)
    if p:
        ax_eig.legend(fontsize=8, ncol=min(3, p))

    if show_cholesky:
        assert component_curves is not None
        p = p_model
        component_axes = []
        n_pairs = p * (p - 1) // 2

        for row in range(p):
            row_axes = []
            for col in range(p):
                ax = fig.add_subplot(grid[row + 2, col], sharex=ax_ratio)

                if row == col:
                    label = f"LogDelta{row + 1}{col + 1}"
                    color = "tab:blue"
                elif row < col:
                    label = f"Re(Theta{row + 1}{col + 1})"
                    color = "tab:orange"
                else:
                    label = f"Im(Theta{row + 1}{col + 1})"
                    color = "tab:red"
                y, y_smooth = component_curves[label]

                _shade_excluded_bands(ax, add_label=False)
                if use_log_x:
                    ax.semilogx(freq, y, color=color, lw=0.7, alpha=0.45)
                else:
                    ax.plot(freq, y, color=color, lw=0.7, alpha=0.45)

                if use_log_x:
                    ax.semilogx(freq, y_smooth, color=color, lw=1.5)
                else:
                    ax.plot(freq, y_smooth, color=color, lw=1.5)

                # Draw knot locations as vertical tick marks if provided.
                if component_knots is not None and label in component_knots:
                    knot_freq = np.asarray(component_knots[label], dtype=float)
                    ylo, yhi = np.nanmin(y), np.nanmax(y)
                    knot_y = ylo - 0.08 * (yhi - ylo) if yhi > ylo else ylo
                    ax.plot(
                        knot_freq,
                        np.full_like(knot_freq, knot_y),
                        marker="|",
                        color="tab:green",
                        ms=6,
                        mew=1.0,
                        ls="none",
                        alpha=0.7,
                        zorder=5,
                    )
                    n_k = len(knot_freq)
                    ax.set_title(f"{label}  (K={n_k})", fontsize=8)
                else:
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
        if info_text is not None and str(info_text).strip():
            fig.suptitle(
                f"Preprocessing diagnostics ({info_text})", fontsize=10
            )
    else:
        if info_text is not None and str(info_text).strip():
            fig.suptitle(
                f"Preprocessing diagnostics ({info_text})", fontsize=10
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        else:
            fig.tight_layout()

    return fig
