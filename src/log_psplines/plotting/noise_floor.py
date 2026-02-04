from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logger import logger


def plot_noise_floor_overlay(
    *,
    idata,
    outdir: str | Path,
    block_idx: int = 2,
) -> None:
    """Plot eps2 floor vs delta^2 posterior for a given Cholesky block.

    Produces two plots in {outdir}/diagnostics:
    - floor_overlay_block{block_idx}_std.png
    - floor_overlay_block{block_idx}_phys.png

    Requires idata.noise_floor (eps2 arrays) and idata.sample_stats.log_delta_sq.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    outdir = Path(outdir)
    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(idata, "noise_floor"):
        logger.debug(
            "noise_floor group not present; skipping floor overlay plots"
        )
        return

    if (
        not hasattr(idata, "sample_stats")
        or "log_delta_sq" not in idata.sample_stats
    ):
        logger.debug(
            "sample_stats.log_delta_sq missing; skipping floor overlay plots"
        )
        return

    eps2_std = np.asarray(idata.noise_floor["eps2_floor_std"].values)
    eps2_phys = np.asarray(idata.noise_floor["eps2_floor_phys"].values)

    freq = np.asarray(idata.noise_floor["freq"].values, dtype=float)
    if eps2_std.ndim != 2:
        raise ValueError(
            "noise_floor.eps2_floor_std must have shape (freq, channels)"
        )

    if block_idx < 0 or block_idx >= eps2_std.shape[1]:
        raise ValueError(
            f"block_idx={block_idx} out of range for eps2_floor_std with channels={eps2_std.shape[1]}"
        )

    eps2_std_b = eps2_std[:, block_idx]
    eps2_phys_b = eps2_phys[:, block_idx]

    log_delta_sq = np.asarray(idata.sample_stats["log_delta_sq"].values)
    # (chain, draw, freq, channels)
    if log_delta_sq.ndim != 4:
        raise ValueError(
            "sample_stats.log_delta_sq must have 4 dims (chain, draw, freq, channels)"
        )
    if log_delta_sq.shape[2] != freq.shape[0]:
        raise ValueError(
            f"log_delta_sq freq dim {log_delta_sq.shape[2]} != noise_floor freq dim {freq.shape[0]}"
        )

    log_delta_b = log_delta_sq[:, :, :, block_idx]
    flat = log_delta_b.reshape(-1, log_delta_b.shape[-1])
    flat = np.clip(flat, a_min=-80.0, a_max=80.0)
    delta_sq = np.exp(flat)

    qlo, q50, qhi = np.percentile(delta_sq, [5.0, 50.0, 95.0], axis=0)
    delta_eff_50 = q50 + eps2_std_b

    factor_ch = np.asarray(idata.noise_floor["psd_rescale_factor"].values)
    factor = float(factor_ch[block_idx]) if factor_ch.size else 1.0
    qlo_phys = qlo * factor
    q50_phys = q50 * factor
    qhi_phys = qhi * factor
    delta_eff_50_phys = (q50 + eps2_std_b) * factor

    def _plot(
        path: Path,
        eps2: np.ndarray,
        lo: np.ndarray,
        med: np.ndarray,
        hi: np.ndarray,
        eff: np.ndarray,
        ylabel: str,
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
        ax.set_title(f"Noise Floor vs Innovation Variance (block {block_idx})")

        ax.fill_between(
            freq, lo, hi, color="C0", alpha=0.25, label="delta^2 90% CI"
        )
        ax.plot(freq, med, color="C0", lw=2.0, label="delta^2 median")
        ax.plot(freq, eps2, color="C3", lw=2.0, ls="--", label="eps2 floor")
        ax.plot(
            freq,
            eff,
            color="C2",
            lw=1.5,
            ls=":",
            label="delta^2 + eps2 (median)",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frequency")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", frameon=False)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    std_path = diag_dir / f"floor_overlay_block{block_idx}_std.png"
    phys_path = diag_dir / f"floor_overlay_block{block_idx}_phys.png"

    _plot(
        std_path,
        eps2_std_b,
        qlo,
        q50,
        qhi,
        delta_eff_50,
        ylabel="variance (standardized units)",
    )
    _plot(
        phys_path,
        eps2_phys_b,
        qlo_phys,
        q50_phys,
        qhi_phys,
        delta_eff_50_phys,
        ylabel="variance (physical units)",
    )

    logger.info(f"Saved floor overlay (std) to {std_path}")
    logger.info(f"Saved floor overlay (phys) to {phys_path}")
