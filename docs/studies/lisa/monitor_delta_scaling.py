#!/usr/bin/env python3
"""Utilities for inspecting how the third Cholesky block's noise scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_IDATA = HERE / "results" / "lisa" / "inference_data.nc"
DEFAULT_OUTDIR = DEFAULT_IDATA.parent


def _flatten_draws(array: np.ndarray) -> np.ndarray:
    """Concatenate the chain+draw axes so each row is one sample."""
    arr = np.asarray(array)
    if arr.ndim < 3:
        raise ValueError("Expected at least 3 dimensions (chain, draw, ...).")
    n_chains, n_draws = arr.shape[:2]
    return arr.reshape(n_chains * n_draws, *arr.shape[2:])


def _collapse_summary(
    freqs: np.ndarray, median: np.ndarray, threshold: float
) -> str:
    mask = median < threshold
    if not mask.any():
        return f"no bins drop below {threshold:.2e}"
    collapsed = freqs[mask]
    examples = collapsed[:5]
    suffix = (
        f"{len(collapsed)} bins (first freqs: "
        + ", ".join(f"{freq:.3e}" for freq in examples)
        + (", ..." if collapsed.size > 5 else "")
        + ")"
    )
    return f"{len(collapsed)} bins below {threshold:.2e} ({suffix})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the delta3 variance as a function of frequency and draws."
    )
    parser.add_argument(
        "--idata",
        type=Path,
        default=DEFAULT_IDATA,
        help="Path to the inference data netCDF produced by lisa_multivar.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=2,
        help="Zero-based channel index (2 corresponds to delta3).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where plots and summaries will be written.",
    )
    parser.add_argument(
        "--draws-to-plot",
        type=int,
        default=8,
        help="Number of representative delta2(f) draws to overlay on the variance plot.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-8,
        help="Flag frequency bins whose median delta^2 falls below this value.",
    )

    args = parser.parse_args()
    args.idata = args.idata.expanduser()
    args.outdir = args.outdir.expanduser()

    if not args.idata.exists():
        raise FileNotFoundError(args.idata)

    args.outdir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(str(args.idata))
    log_delta = np.asarray(idata.sample_stats["log_delta_sq"].values)
    n_channels = log_delta.shape[-1]
    if not (0 <= args.channel < n_channels):
        raise ValueError(
            f"channel {args.channel} out of bounds (n_channels={n_channels})."
        )

    log_delta_flat = _flatten_draws(log_delta)
    variance_samples = np.exp(log_delta_flat[..., args.channel])
    freqs = np.asarray(idata.posterior_psd["freq"].values)
    quantiles = np.percentile(variance_samples, [5.0, 50.0, 95.0], axis=0)

    delta_vals = np.asarray(
        idata.posterior[f"delta_{args.channel}"].values
    ).reshape(-1)
    phi_vals = np.asarray(
        idata.posterior[f"phi_delta_{args.channel}"].values
    ).reshape(-1)
    delta_ratio = delta_vals / phi_vals
    ratio_log10 = np.log10(delta_ratio)

    weights = np.asarray(
        idata.posterior[f"weights_delta_{args.channel}"].values
    ).reshape(-1, idata.posterior[f"weights_delta_{args.channel}"].shape[-1])
    weight_norms = np.linalg.norm(weights, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12))
    ax_var, ax_ratio, ax_weights = axes

    n_overlay = min(args.draws_to_plot, variance_samples.shape[0])
    overlay_idx = (
        np.linspace(0, variance_samples.shape[0] - 1, n_overlay, dtype=int)
        if n_overlay > 0
        else []
    )
    for idx in overlay_idx:
        ax_var.plot(
            freqs,
            variance_samples[idx],
            color="gray",
            alpha=0.25,
            linewidth=0.7,
            label="sample draws" if idx == overlay_idx[0] else None,
        )

    ax_var.plot(
        freqs,
        quantiles[1],
        color="tab:blue",
        label="median variance",
        linewidth=1.5,
    )
    ax_var.fill_between(
        freqs,
        quantiles[0],
        quantiles[2],
        color="tab:blue",
        alpha=0.3,
        label="5-95 percentile",
    )
    ax_var.set_xscale("log")
    ax_var.set_yscale("log")
    ax_var.grid(True, which="both", linestyle=":")
    ax_var.set_ylabel("delta^2 (variance)")
    ax_var.set_title("delta3(f) variance across frequencies")
    ax_var.legend(loc="best")

    draw_axis = np.arange(ratio_log10.size)
    ax_ratio.plot(draw_axis, ratio_log10, color="tab:orange", linewidth=0.5)
    median_ratio = np.median(ratio_log10)
    ax_ratio.axhline(
        median_ratio,
        color="k",
        linestyle="--",
        label=f"median log10(delta/phi) {median_ratio:.2f}",
    )
    ax_ratio.set_ylabel("log10(delta/phi)")
    ax_ratio.set_xlabel("flattened draw index")
    ax_ratio.grid(True, linestyle=":")
    ax_ratio.legend(loc="upper right")

    ax_weights.plot(draw_axis, weight_norms, color="tab:green", linewidth=0.6)
    median_norm = np.median(weight_norms)
    ax_weights.axhline(
        median_norm,
        color="k",
        linestyle="--",
        label=f"median weights norm {median_norm:.2e}",
    )
    ax_weights.set_ylabel("weights_delta norm (L2)")
    ax_weights.set_xlabel("flattened draw index")
    ax_weights.grid(True, linestyle=":")
    ax_weights.legend(loc="upper right")

    fig.tight_layout()
    figure_path = args.outdir / f"delta_scaling_channel_{args.channel}.png"
    fig.savefig(str(figure_path), dpi=150)
    plt.close(fig)

    median_variance = quantiles[1]
    collapse_msg = _collapse_summary(
        freqs, median_variance, args.variance_threshold
    )
    summary_lines = [
        f"Source: {args.idata}",
        f"Channel: {args.channel}",
        f"Draws flattened: {variance_samples.shape[0]}",
        f"Frequency bins: {freqs.size}",
        f"Variance median range: {median_variance.min():.2e} -> {median_variance.max():.2e}",
        f"Variance collapse summary: {collapse_msg}",
        f"delta/phi (log10) min/median/max: {ratio_log10.min():.2f} / {median_ratio:.2f} / {ratio_log10.max():.2f}",
        f"phi median: {np.median(phi_vals):.2e} (min {phi_vals.min():.2e}, max {phi_vals.max():.2e})",
        f"weights norm min/median/max: {weight_norms.min():.2e} / {median_norm:.2e} / {weight_norms.max():.2e}",
        f"Saved figure: {figure_path}",
    ]

    summary_path = args.outdir / (f"delta_scaling_channel_{args.channel}.txt")
    summary_path.write_text("\n".join(summary_lines))

    print("Wrote diagnostics:")
    print(f"  {figure_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
