"""Evaluate log-likelihood along coordinated ridge directions in spline-weight space."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def _load_likelihood_module() -> object:
    """Dynamically load the existing likelihood_slices helpers."""
    module_path = Path(__file__).resolve().parent / "likelihood_slices.py"
    spec = importlib.util.spec_from_file_location(
        "likelihood_slices", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load likelihood_slices module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_frequency(
    r23: np.ndarray, freq: np.ndarray, idx: int | None
) -> int:
    """Pick a frequency index either manually or via maximum ratio."""
    if idx is not None:
        if idx < 0 or idx >= freq.size:
            raise IndexError(f"freq index {idx} outside [0, {freq.size}).")
        return idx
    if np.all(np.isnan(r23)):
        raise RuntimeError("Ratio r23 is NaN everywhere.")
    return int(np.nanargmax(r23))


def _load_basis_and_penalty(
    idata: az.InferenceData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grab the off-diagonal spline bases + penalty."""
    spline_group = getattr(idata, "spline_model", None)
    if spline_group is None:
        raise RuntimeError("idata missing spline_model group.")
    required = (
        "offdiag_re_basis",
        "offdiag_im_basis",
        "offdiag_re_penalty_matrix",
    )
    for key in required:
        if key not in spline_group:
            raise RuntimeError(f"spline_model lacks {key}.")
    basis_re = np.asarray(spline_group["offdiag_re_basis"].values, dtype=float)
    basis_im = np.asarray(spline_group["offdiag_im_basis"].values, dtype=float)
    penalty = np.asarray(
        spline_group["offdiag_re_penalty_matrix"].values, dtype=float
    )
    return basis_re, basis_im, penalty


def _compute_direction(penalty: np.ndarray, index: int) -> np.ndarray:
    """Return the specified eigenvector of the penalty matrix."""
    evals, evecs = np.linalg.eigh(penalty)
    if index < 0 or index >= evecs.shape[1]:
        raise IndexError("Penalty eigenvector index out of range.")
    v = evecs[:, index]
    return v / np.linalg.norm(v)


def _load_freq_axis(idata: az.InferenceData) -> np.ndarray:
    obs_group = getattr(idata, "observed_data", None)
    if obs_group is None:
        raise RuntimeError("idata missing observed_data.")
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
    raise RuntimeError("Could not find freq axis in observed_data.")


def _compute_r23(matrix: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(
        0.5 * (matrix + np.swapaxes(np.conj(matrix), -1, -2))
    )
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = eigvals[..., ::-1]
    if eigvals.shape[1] < 3:
        raise RuntimeError("Need at least 3 eigenvalues.")
    denom = eigvals[:, 1]
    ratio = np.divide(
        eigvals[:, 2],
        denom,
        out=np.full_like(eigvals[:, 2], np.nan),
        where=np.abs(denom) > np.finfo(float).tiny,
    )
    return ratio


def _compute_logl_delta(
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    theta31: complex,
    theta32: complex,
    delta3: float,
    delta_theta: complex,
    c_scale: float,
    a_vals: np.ndarray,
) -> np.ndarray:
    """Evaluate ΔlogL for shifts along the ridge direction."""
    logls = np.empty_like(a_vals, dtype=float)
    for idx, a in enumerate(a_vals):
        theta31_a = theta31 + a * delta_theta
        theta32_a = theta32 - c_scale * a * delta_theta
        resid = u3 - theta31_a * u1 - theta32_a * u2
        sse = np.sum(np.abs(resid) ** 2)
        logls[idx] = -sse / delta3
    logls -= np.nanmax(logls)
    return logls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe log-likelihood along a coordinated spline-weight ridge direction."
    )
    parser.add_argument(
        "--idata", type=Path, required=True, help="Path to inference_data.nc."
    )
    parser.add_argument(
        "--freq-index",
        type=int,
        default=None,
        help="Frequency index to inspect (default=argmax r23).",
    )
    parser.add_argument(
        "--penalty-eig-index",
        type=int,
        default=0,
        help="Penalty eigenvector index (0 corresponds to smallest eigenvalue).",
    )
    parser.add_argument(
        "--a-range",
        type=float,
        default=1.0,
        help="Range of steps a to scan (±value around zero).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=201,
        help="Number of points across a-range.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "likelihood_slices"
        / "results",
        help="Directory to save the plot.",
    )
    args = parser.parse_args()

    module = _load_likelihood_module()
    prep_args = argparse.Namespace(idata=args.idata, npz=None)
    freq, matrices, u_modes, idata = module._prepare_data(prep_args)
    theta_hat_all, delta3_sq = module._compute_theta_hat_and_delta(u_modes)
    r23 = _compute_r23(matrices)
    k = _select_frequency(r23, freq, args.freq_index)

    basis_re, basis_im, penalty = _load_basis_and_penalty(idata)
    if k >= basis_re.shape[0]:
        raise RuntimeError("Selected frequency index exceeds basis length.")
    v = _compute_direction(penalty, args.penalty_eig_index)
    delta_theta = (basis_re[k] + 1j * basis_im[k]) @ v

    u1 = u_modes[k, :, 0]
    u2 = u_modes[k, :, 1]
    u3 = u_modes[k, :, 2]
    norm1 = np.linalg.norm(u1)
    norm2 = np.linalg.norm(u2)
    c_scale = norm1 / max(norm2, np.finfo(float).eps)

    a_vals = np.linspace(-args.a_range, args.a_range, args.n_steps)
    logl_vals = _compute_logl_delta(
        u1,
        u2,
        u3,
        theta_hat_all[k, 0],
        theta_hat_all[k, 1],
        delta3_sq[k],
        delta_theta,
        c_scale,
        a_vals,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a_vals, logl_vals, lw=1.5, color="tab:purple")
    ax.axhline(0.0, color="tab:gray", lw=0.7, linestyle="--")
    ax.set_title(
        f"ΔlogL along ridge direction (freq {freq[k]:.3e}, r23={r23[k]:.3f})"
    )
    ax.set_xlabel("step size a")
    ax.set_ylabel("ΔlogL (relative)")
    ax.grid(True, linestyle=":", alpha=0.6)
    description = (
        f"c=||u1||/||u2||={c_scale:.3f}, θ̂₃₁={theta_hat_all[k,0]:.3f}, "
        f"θ̂₃₂={theta_hat_all[k,1]:.3f}"
    )
    ax.text(
        0.02,
        0.95,
        description,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"
        ),
    )
    outpath = args.outdir / f"likelihood_ridge_freq{k:04d}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote ridge likelihood plot to {outpath}")


if __name__ == "__main__":
    main()
