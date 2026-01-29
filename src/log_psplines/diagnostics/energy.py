"""Energy-based NUTS diagnostics: E-BFMI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ..logger import logger


def compute_ebfmi(energy: np.ndarray) -> float:
    """
    Compute Energy Bayesian Fraction of Missing Information (E-BFMI).

    E-BFMI = E[(E_t - E_{t-1})^2] / Var(E_t)

    Measures how efficiently the sampler explores energy space.
    Higher is better (> ~0.3 usually fine, < ~0.1 suggests issues).

    Parameters
    ----------
    energy : np.ndarray
        Energy values, shape (n_draws,) or (n_chains, n_draws_per_chain).

    Returns
    -------
    float
        E-BFMI value. Returns NaN if energy variance is insufficient.
    """
    energy = np.asarray(energy).ravel()
    energy_diff = np.diff(energy)
    mean_sq_diff = np.mean(energy_diff**2)
    var_energy = np.var(energy)
    if var_energy < 1e-10:
        return np.nan
    return float(mean_sq_diff / var_energy)


def plot_ebfmi_diagnostics(
    idata, outdir: str | Path, filename: str = "ebfmi_diagnostics.png"
) -> Dict[str, float] | None:
    """
    Plot E-BFMI diagnostic including energy trace and transition magnitudes.

    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data object with sample_stats containing energy.
    outdir : str or Path
        Output directory for the plot.
    filename : str
        Output filename.

    Returns
    -------
    dict or None
        Summary dict with 'ebfmi' values per chain and 'overall' key.
        Returns None if energy data is unavailable.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "sample_stats" not in idata or "energy" not in idata.sample_stats:
        logger.warning(
            "No energy in sample_stats; skipping E-BFMI diagnostic."
        )
        return None

    energy = np.asarray(idata.sample_stats["energy"].values)
    # shape is typically (n_chains, n_draws)

    n_chains = energy.shape[0] if energy.ndim > 1 else 1
    ebfmi_by_chain = {}

    fig, axes = plt.subplots(n_chains + 1, 2, figsize=(14, 3 * (n_chains + 1)))
    if n_chains == 1:
        axes = axes.reshape(1, -1)
        axes = np.vstack([axes, np.zeros((1, 2))])  # Dummy row for overall

    # Per-chain diagnostics
    for ch in range(n_chains):
        e_ch = energy[ch] if energy.ndim > 1 else energy
        ebfmi_ch = compute_ebfmi(e_ch)
        ebfmi_by_chain[f"chain_{ch}"] = ebfmi_ch

        # Energy trace
        ax = axes[ch, 0]
        ax.plot(e_ch, alpha=0.7, linewidth=0.8)
        ax.axhline(
            np.mean(e_ch), color="r", linestyle="--", label="mean", alpha=0.7
        )
        ax.set_xlabel("Draw")
        ax.set_ylabel("Energy")
        ax.set_title(f"Chain {ch}: Energy Trace")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Energy transitions (absolute differences)
        ax = axes[ch, 1]
        e_diff = np.abs(np.diff(e_ch))
        ax.plot(e_diff, alpha=0.7, linewidth=0.8, color="orange")
        ax.axhline(
            np.mean(e_diff), color="r", linestyle="--", label="mean", alpha=0.7
        )
        ax.set_xlabel("Draw")
        ax.set_ylabel("|ΔE|")
        ax.set_title(f"Chain {ch}: Energy Transitions (E-BFMI={ebfmi_ch:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Overall summary
    ax = axes[n_chains, 0]
    chains = list(range(n_chains))
    ebfmi_vals = [ebfmi_by_chain[f"chain_{ch}"] for ch in chains]
    colors = [
        "green" if v > 0.3 else "orange" if v > 0.1 else "red"
        for v in ebfmi_vals
    ]
    bars = ax.bar(
        chains, ebfmi_vals, color=colors, alpha=0.7, edgecolor="black"
    )
    ax.axhline(
        0.3, color="green", linestyle="--", alpha=0.5, label="Good (>0.3)"
    )
    ax.axhline(
        0.1, color="orange", linestyle="--", alpha=0.5, label="Marginal (~0.1)"
    )
    ax.set_xlabel("Chain")
    ax.set_ylabel("E-BFMI")
    ax.set_title("E-BFMI per Chain")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, ebfmi_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Energy distribution
    ax = axes[n_chains, 1]
    all_energy = energy.ravel()
    ax.hist(all_energy, bins=50, alpha=0.7, edgecolor="black", color="skyblue")
    ax.axvline(
        np.mean(all_energy),
        color="r",
        linestyle="--",
        label="mean",
        linewidth=2,
    )
    ax.axvline(
        np.median(all_energy),
        color="orange",
        linestyle="--",
        label="median",
        linewidth=2,
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Frequency")
    ax.set_title("Overall Energy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = outdir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved E-BFMI diagnostics to {out_path}")
    plt.close()

    overall_ebfmi = compute_ebfmi(all_energy)
    ebfmi_by_chain["overall"] = overall_ebfmi

    return ebfmi_by_chain


def run(
    *,
    idata=None,
    config=None,
    truth=None,
    signals=None,
    psd_ref=None,
    idata_vi=None,
    outdir: str | Path | None = None,
) -> Dict[str, float]:
    """
    Compute E-BFMI metric and optionally generate diagnostic plots.

    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data with sample_stats.energy.
    outdir : str, Path, or None
        If provided, save diagnostic plots to this directory.
    **kwargs
        Additional arguments (ignored, for compatibility with diagnostic API).

    Returns
    -------
    dict
        Dictionary with E-BFMI values. Keys are 'ebfmi_overall' and
        'ebfmi_chain_{i}' for each chain.
    """
    if idata is None or "sample_stats" not in idata:
        return {}

    if "energy" not in idata.sample_stats:
        return {}

    metrics: Dict[str, float] = {}

    # Plot diagnostics if output directory is provided
    if outdir is not None:
        ebfmi_by_chain = plot_ebfmi_diagnostics(idata, outdir)
        if ebfmi_by_chain is not None:
            # Add metrics with standardized names
            for key, val in ebfmi_by_chain.items():
                if np.isfinite(val):
                    if key == "overall":
                        metrics["ebfmi_overall"] = val
                    else:
                        metrics[f"ebfmi_{key}"] = val
    else:
        # Just compute without plotting
        energy = np.asarray(idata.sample_stats["energy"].values)
        n_chains = energy.shape[0] if energy.ndim > 1 else 1

        for ch in range(n_chains):
            e_ch = energy[ch] if energy.ndim > 1 else energy
            val = compute_ebfmi(e_ch)
            if np.isfinite(val):
                metrics[f"ebfmi_chain_{ch}"] = val

        overall = compute_ebfmi(energy.ravel())
        if np.isfinite(overall):
            metrics["ebfmi_overall"] = overall

    return metrics
