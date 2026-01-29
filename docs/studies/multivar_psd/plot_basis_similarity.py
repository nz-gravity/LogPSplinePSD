"""Plot the alignment between off-diagonal spline bases in VAR3 / LISA inference data."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def _load_basis(idata: az.InferenceData) -> tuple[np.ndarray, np.ndarray]:
    """Extract the off-diagonal real/imag spline basis matrices."""
    spline = getattr(idata, "spline_model", None)
    if spline is None:
        raise RuntimeError("Inference data missing the `spline_model` group.")
    if "offdiag_re_basis" not in spline or "offdiag_im_basis" not in spline:
        raise RuntimeError(
            "Inference data does not expose off-diagonal bases."
        )
    re_basis = np.asarray(spline["offdiag_re_basis"].values, dtype=float)
    im_basis = np.asarray(spline["offdiag_im_basis"].values, dtype=float)
    if re_basis.shape != im_basis.shape:
        raise RuntimeError("Real/imag basis shapes differ.")
    return re_basis, im_basis


def _load_freq(idata: az.InferenceData) -> np.ndarray:
    """Retrieve the observed frequency grid from the inference data."""
    obs_group = getattr(idata, "observed_data", None)
    if obs_group is None:
        raise RuntimeError("Inference data missing `observed_data`.")
    candidate_keys = (
        "periodogram",
        "Y",
        "spectral_matrix",
        "observed_periodogram",
    )
    for key in candidate_keys:
        if key in obs_group:
            return np.asarray(
                obs_group[key].coords["freq"].values, dtype=float
            )
    raise RuntimeError("Could not find a freq coordinate under observed_data.")


def _basis_similarity(
    re_basis: np.ndarray, im_basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cosine similarity and Euclidean diff per frequency."""
    re_norm = np.linalg.norm(re_basis, axis=1)
    im_norm = np.linalg.norm(im_basis, axis=1)
    denom = np.maximum(re_norm * im_norm, np.finfo(float).eps)
    cos_sim = np.sum(re_basis * im_basis, axis=1) / denom
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    diff_norm = np.linalg.norm(re_basis - im_basis, axis=1)
    return cos_sim, diff_norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot alignment between off-diagonal spline bases."
    )
    parser.add_argument(
        "--idata",
        type=Path,
        required=True,
        help="Path to inference_data.nc (must contain spline_model).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "likelihood_slices"
        / "results",
        help="Directory to save the figure.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    idata = az.from_netcdf(str(args.idata))
    freq = _load_freq(idata)
    re_basis, im_basis = _load_basis(idata)
    if freq.shape[0] != re_basis.shape[0]:
        raise RuntimeError(
            "Frequency axis length differs from off-diagonal basis rows."
        )
    cos_sim, diff_norm = _basis_similarity(re_basis, im_basis)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(freq, cos_sim, color="tab:blue", linewidth=1.5)
    axes[0].set_title("Cosine similarity: Re / Im off-diagonal basis rows")
    axes[0].set_ylabel("cos(θ_re, θ_im)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(freq, diff_norm, color="tab:green", linewidth=1.5)
    axes[1].set_title("Euclidean difference between Re/Im rows")
    axes[1].set_ylabel("||B_re - B_im||")
    axes[1].set_xlabel("frequency")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].set_xscale("log")
    axes[0].set_xscale("log")
    fig.tight_layout()
    outpath = args.outdir / "offdiag_basis_alignment.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote basis alignment plot to {outpath}")


if __name__ == "__main__":
    main()
