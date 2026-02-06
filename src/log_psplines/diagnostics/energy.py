"""Energy-based NUTS diagnostics: E-BFMI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ..logger import logger


def _resolve_energy_map(idata) -> Dict[str, np.ndarray]:
    """Extract energy arrays from InferenceData sample_stats."""
    if idata is None or "sample_stats" not in idata:
        return {}

    if "energy" in idata.sample_stats:
        return {"energy": np.asarray(idata.sample_stats["energy"].values)}

    energy_keys = [
        str(key)
        for key in idata.sample_stats
        if str(key).startswith("energy_channel_")
    ]
    energy_map: Dict[str, np.ndarray] = {}
    for key in energy_keys:
        try:
            energy_map[key] = np.asarray(idata.sample_stats[key].values)
        except Exception:
            continue
    return energy_map


def _finite_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


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
    idata,
    outdir: str | Path,
    filename: str = "ebfmi_diagnostics.png",
    *,
    energy: np.ndarray | None = None,
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

    if energy is None:
        energy_map = _resolve_energy_map(idata)
        if not energy_map:
            logger.warning(
                "No energy in sample_stats; skipping E-BFMI diagnostic."
            )
            return None
        if len(energy_map) > 1:
            logger.warning(
                "Multiple energy arrays found; skipping combined E-BFMI plot."
            )
            return None
        energy = next(iter(energy_map.values()))
    else:
        energy = np.asarray(energy)
    # shape is typically (n_chains, n_draws)

    n_chains = energy.shape[0] if energy.ndim > 1 else 1
    ebfmi_by_chain = {}

    fig, axes = plt.subplots(n_chains + 1, 2, figsize=(14, 3 * (n_chains + 1)))

    # Per-chain diagnostics
    for ch in range(n_chains):
        e_ch = energy[ch] if energy.ndim > 1 else energy
        e_ch = np.asarray(e_ch).ravel()
        ebfmi_ch = compute_ebfmi(e_ch)
        ebfmi_by_chain[f"chain_{ch}"] = ebfmi_ch

        # Energy trace
        ax = axes[ch, 0]
        ax.plot(e_ch, alpha=0.7, linewidth=0.8)
        finite_e_ch = _finite_1d(e_ch)
        if finite_e_ch.size:
            ax.axhline(
                float(np.mean(finite_e_ch)),
                color="r",
                linestyle="--",
                label="mean",
                alpha=0.7,
            )
        ax.set_xlabel("Draw")
        ax.set_ylabel("Energy")
        ax.set_title(f"Chain {ch}: Energy Trace")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Energy transitions (absolute differences)
        ax = axes[ch, 1]
        e_diff = np.abs(np.diff(e_ch)) if e_ch.size > 1 else np.array([])
        ax.plot(e_diff, alpha=0.7, linewidth=0.8, color="orange")
        finite_e_diff = _finite_1d(e_diff)
        if finite_e_diff.size:
            ax.axhline(
                float(np.mean(finite_e_diff)),
                color="r",
                linestyle="--",
                label="mean",
                alpha=0.7,
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
    ebfmi_plot_vals = [
        0.0 if not np.isfinite(v) else float(v) for v in ebfmi_vals
    ]
    colors = [
        (
            "green"
            if np.isfinite(v) and v > 0.3
            else (
                "orange"
                if np.isfinite(v) and v > 0.1
                else "gray" if not np.isfinite(v) else "red"
            )
        )
        for v in ebfmi_vals
    ]
    bars = ax.bar(
        chains, ebfmi_plot_vals, color=colors, alpha=0.7, edgecolor="black"
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
        text = f"{val:.3f}" if np.isfinite(val) else "nan"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Energy distribution
    ax = axes[n_chains, 1]
    all_energy = energy.ravel()
    finite_energy = _finite_1d(all_energy)
    if finite_energy.size == 0:
        logger.warning(
            "Energy diagnostic: no finite energy values found; skipping histogram."
        )
        ax.text(
            0.5,
            0.5,
            "No finite energy values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title("Overall Energy Distribution")
    else:
        hist_kwargs = dict(alpha=0.7, edgecolor="black", color="skyblue")
        for bins in (50, 30, 20, 10, "auto"):
            try:
                ax.hist(finite_energy, bins=bins, **hist_kwargs)
                break
            except ValueError:
                continue
        else:
            ax.hist(finite_energy, bins=1, **hist_kwargs)
        ax.axvline(
            float(np.mean(finite_energy)),
            color="r",
            linestyle="--",
            label="mean",
            linewidth=2,
        )
        ax.axvline(
            float(np.median(finite_energy)),
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


def _run(
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

    energy_map = _resolve_energy_map(idata)
    if not energy_map:
        return {}

    metrics: Dict[str, float] = {}

    if outdir is None:
        config_outdir = getattr(config, "outdir", None)
        if config_outdir:
            outdir = Path(config_outdir)

    for energy_key, energy in energy_map.items():
        label = "ebfmi" if energy_key == "energy" else f"ebfmi_{energy_key}"
        if outdir is not None:
            filename = (
                "ebfmi_diagnostics.png"
                if energy_key == "energy"
                else f"ebfmi_diagnostics_{energy_key}.png"
            )
            ebfmi_by_chain = plot_ebfmi_diagnostics(
                idata,
                outdir,
                filename=filename,
                energy=energy,
            )
            if ebfmi_by_chain is not None:
                for key, val in ebfmi_by_chain.items():
                    if np.isfinite(val):
                        metrics[f"{label}_{key}"] = val
        else:
            n_chains = energy.shape[0] if energy.ndim > 1 else 1
            for ch in range(n_chains):
                e_ch = energy[ch] if energy.ndim > 1 else energy
                val = compute_ebfmi(e_ch)
                if np.isfinite(val):
                    metrics[f"{label}_chain_{ch}"] = val
            overall = compute_ebfmi(energy.ravel())
            if np.isfinite(overall):
                metrics[f"{label}_overall"] = overall

    return metrics
