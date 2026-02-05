"""Likelihood contour slices for (theta31, theta32) per frequency bin.

This sampler-agnostic utility inspects the likelihood surface defined by the
observed spectral statistics (empirical spectral matrices or pre-split ``u``-modes)
to uncover ridges / weak identifiability for the last blocked channel.

Usage is driven by ``--idata`` (e.g. inference_data.nc for LISA) or ``--npz``
(VAR3/preprocessed datasets). Contours are computed from
``SSE(theta31, theta32) = sum_nu |u3 - theta31*u1 - theta32*u2|^2`` and saved as
PNG outputs plus a combined panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _load_periodogram_from_idata(
    idata: az.InferenceData,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequencies and empirical matrices from ArviZ inference data."""
    obs_group = getattr(idata, "observed_data", None)
    if obs_group is None:
        raise RuntimeError("idata missing `observed_data` group.")

    candidate_keys = (
        "periodogram",
        "Y",
        "spectral_matrix",
        "observed_periodogram",
    )
    for key in candidate_keys:
        if key in obs_group:
            matrix = np.asarray(obs_group[key].values)
            freq = np.asarray(
                obs_group[key].coords["freq"].values, dtype=float
            )
            return freq, np.asarray(matrix, dtype=np.complex128)

    raise RuntimeError(
        "observed_data does not expose a known empirical matrix "
        "(checked keys: {}).".format(", ".join(candidate_keys))
    )


def _float_array_from(npz: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Extract any of the recognized frequency arrays from an NPZ archive."""
    freq_names = (
        "freq",
        "freq_true",
        "frequencies",
        "freqs",
        "freq_grid",
        "frequency",
    )
    for name in freq_names:
        if name in npz:
            return np.asarray(npz[name], dtype=float)
    return None


def _matrix_from_npz(npz: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Extract a 3×3 (or general) empirical spectral matrix if present."""
    matrix_names = (
        "Y",
        "periodogram",
        "spectral_matrix",
        "empirical_matrix",
        "empirical_psd",
    )
    for name in matrix_names:
        if name in npz:
            return np.asarray(npz[name], dtype=np.complex128)
    return None


def _u_components_from(npz: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Try to load preprocessed u-vectors from the NPZ archive."""
    # Try explicit u1/u2/u3 components first.
    candidate = []
    for ch in range(1, 5):
        key = f"u{ch}"
        if key in npz:
            candidate.append(np.asarray(npz[key]))
        else:
            break
    if candidate:
        base_shape = candidate[0].shape
        if not all(arr.shape == base_shape for arr in candidate):
            raise RuntimeError("u-components from NPZ have mismatched shapes.")
        stacked = np.stack(
            [np.asarray(arr, dtype=np.complex128) for arr in candidate],
            axis=-1,
        )
        return stacked  # shape: (freq, Nb, channels) after stacking

    # Fall back to generic ``u`` keys.
    generic_keys = ("u_modes", "u_vecs", "mode_vectors", "u")
    for key in generic_keys:
        if key in npz:
            arr = np.asarray(npz[key], dtype=np.complex128)
            if arr.ndim == 3:
                return arr
            if arr.ndim == 2:
                return arr[..., None]
    return None


def _u_from_matrices(matrices: np.ndarray) -> np.ndarray:
    """Construct eigen-modes `u` from spectral matrices (Hermitian)."""
    matrices = np.asarray(matrices, dtype=np.complex128)
    herm = 0.5 * (matrices + np.swapaxes(np.conj(matrices), -1, -2))
    eigvals, eigvecs = np.linalg.eigh(herm)
    eigvals_desc = eigvals[..., ::-1]
    eigvecs_desc = eigvecs[..., :, ::-1]
    sqrt_vals = np.sqrt(np.maximum(eigvals_desc, 0.0))
    scaled = eigvecs_desc * sqrt_vals[..., None, :]
    return np.transpose(scaled, axes=(0, 2, 1))


def _matrices_from_u(u_modes: np.ndarray) -> np.ndarray:
    """Reconstruct empirical matrices from eigen-modes ``u``."""
    u_modes = np.asarray(u_modes, dtype=np.complex128)
    return np.einsum("k m c, k m d -> k c d", u_modes, np.conj(u_modes))


def _load_npz_bundle(
    npz_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return (freq, matrices, u_modes) extracted from an NPZ file."""
    with np.load(npz_path, allow_pickle=False) as archive:
        freq = _float_array_from(archive)
        matrices = _matrix_from_npz(archive)
        u_modes = _u_components_from(archive)
    return freq, matrices, u_modes


def _ensure_lengths_align(
    freq: np.ndarray, matrices: np.ndarray, u_modes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate that the frequency axis matches the matrix/u data."""
    if freq.shape[0] != matrices.shape[0]:
        raise RuntimeError(
            "Frequency axis length differs from empirical matrices."
        )
    if matrices.shape[0] != u_modes.shape[0]:
        raise RuntimeError("u-modes length does not match empirical matrices.")
    if matrices.shape[1] != matrices.shape[2]:
        raise RuntimeError("Empirical matrices must be square.")
    if u_modes.shape[-1] != matrices.shape[1]:
        raise RuntimeError("u-mode channel count mismatches matrix dimension.")
    return freq, matrices, u_modes


def _prepare_data(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, az.InferenceData | None]:
    """Load data from the supplied IDATA/NPZ arguments."""
    freq: np.ndarray | None = None
    matrices: np.ndarray | None = None
    u_modes: np.ndarray | None = None
    idata: az.InferenceData | None = None

    if args.idata is not None:
        if not args.idata.exists():
            raise FileNotFoundError(f"{args.idata} not found.")
        idata = az.from_netcdf(str(args.idata))
        freq, matrices = _load_periodogram_from_idata(idata)

    if args.npz is not None:
        if not args.npz.exists():
            raise FileNotFoundError(f"{args.npz} not found.")
        freq_npz, matrices_npz, u_npz = _load_npz_bundle(args.npz)
        freq = freq if freq is not None else freq_npz
        matrices = matrices if matrices is not None else matrices_npz
        u_modes = u_modes if u_modes is not None else u_npz

    if freq is None:
        raise RuntimeError("Could not infer frequency axis from inputs.")
    if matrices is None and u_modes is None:
        raise RuntimeError(
            "Need either empirical matrices (Y/periodogram) or pre-split u-modes."
        )
    if matrices is None:
        matrices = _matrices_from_u(u_modes)
    if u_modes is None:
        u_modes = _u_from_matrices(matrices)

    validated = _ensure_lengths_align(freq, matrices, u_modes)
    return (*validated, idata)


def _ordered_eigvals(matrix: np.ndarray) -> np.ndarray:
    """Return descending eigenvalues for each frequency bin."""
    herm = 0.5 * (matrix + np.swapaxes(np.conj(matrix), -1, -2))
    eigvals = np.linalg.eigvalsh(herm)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvals[..., ::-1]


def _select_frequency_indices(
    r23: np.ndarray,
    freq: np.ndarray,
    n_slices: int,
    manual: Sequence[int] | None,
) -> list[int]:
    """Choose frequency bins either manually or via ratio quantiles."""
    if manual:
        valid = []
        for idx in manual:
            if idx < 0 or idx >= freq.size:
                raise IndexError(
                    f"k_index {idx} out of range [0, {freq.size})"
                )
            valid.append(idx)
        return sorted(dict.fromkeys(valid))

    quantiles = np.linspace(10.0, 90.0, max(1, n_slices))
    selected: list[int] = []
    quantile_values = np.nanpercentile(r23, quantiles)
    for target in quantile_values:
        if np.isnan(target):
            continue
        gap = np.abs(r23 - target)
        gap[np.isnan(gap)] = np.inf
        idx = int(np.nanargmin(gap))
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= n_slices:
            break

    if len(selected) < n_slices:
        fallback = np.argsort(np.where(np.isnan(r23), np.inf, r23))
        for idx in fallback:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= n_slices:
                break

    return selected[:n_slices]


def _build_grid(
    center: np.ndarray,
    theta_mode: str,
    size: int,
    max_fixed: float,
    pad_factor: float,
    min_span: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce theta31/theta32 grid around the selected center."""
    if theta_mode == "fixed":
        span = max_fixed
        mid31, mid32 = 0.0, 0.0
    else:
        mid31, mid32 = center
        span = max(pad_factor * max(abs(mid31), abs(mid32)), min_span)
    grid = np.linspace(-span + mid31, span + mid31, size)
    grid2 = np.linspace(-span + mid32, span + mid32, size)
    return np.meshgrid(grid, grid2, indexing="xy")


def _evaluate_logl(
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    theta31: np.ndarray,
    theta32: np.ndarray,
    delta3_sq: float,
) -> np.ndarray:
    """Evaluate SSE-based log likelihood on the theta grid (works for complex theta)."""
    theta31 = theta31[None, :, :]
    theta32 = theta32[None, :, :]
    resid = (
        u3[:, None, None]
        - u1[:, None, None] * theta31
        - u2[:, None, None] * theta32
    )
    sse = np.sum(np.abs(resid) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        logl = -sse / delta3_sq
    logl = np.where(np.isfinite(logl), logl, np.nan)
    return logl - np.nanmax(logl)


def _span_components(
    theta_hat: np.ndarray, pad_factor: float, min_span: float
) -> tuple[float, float]:
    """Return separate real/imag spans for the LSQ center."""
    real_max = max(np.max(np.abs(np.real(theta_hat))), min_span)
    imag_max = max(np.max(np.abs(np.imag(theta_hat))), min_span)
    return float(pad_factor * real_max), float(pad_factor * imag_max)


def _span_from_theta(
    theta_hat: np.ndarray, pad_factor: float, min_span: float
) -> float:
    """Return a single span capturing both the real and imaginary scales."""
    real_span, imag_span = _span_components(theta_hat, pad_factor, min_span)
    return max(real_span, imag_span)


def _linspace_center(center: float, span: float, size: int) -> np.ndarray:
    return np.linspace(center - span, center + span, size)


def _span_range(center: float, span: float) -> tuple[float, float]:
    return (center - span, center + span)


def _theta_lstsq(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute complex least-squares solution."""
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def _text_tag(value: float) -> str:
    """Encode floats into safe filename fragments."""
    tag = f"{value:.4e}"
    tag = tag.replace("-", "m").replace("+", "p").replace(".", "p")
    return tag


def _draw_r23_inset(
    ax: plt.Axes,
    freq_axis: np.ndarray,
    r23_curve: np.ndarray,
    freq_current: float,
) -> None:
    """Overlay a tiny r23 vs freq inset with a marker on the current bin."""
    inset = inset_axes(
        ax, width="35%", height="30%", loc="upper right", borderpad=0.25
    )
    inset.plot(freq_axis, r23_curve, color="tab:blue", linewidth=1.0)
    inset.axvline(freq_current, color="red", linestyle="--", linewidth=0.8)
    inset.set_xscale("log")
    inset.set_xlim(freq_axis[0], freq_axis[-1])
    max_r23 = (
        float(np.nanmax(r23_curve)) if np.any(np.isfinite(r23_curve)) else 1.0
    )
    inset.set_ylim(0.0, min(1.05, max_r23 + 0.05))
    inset.set_yticks([0.0, 0.5, 1.0])
    inset.tick_params(axis="both", labelsize=6)
    # inset.set_title("r23", fontsize=8)
    inset.set_xlabel("freq", fontsize=6)
    inset.set_ylabel(r"$\lambda_2/\lambda_3$", fontsize=6)


def _draw_contour_on_axis(
    ax: plt.Axes,
    theta31: np.ndarray,
    theta32: np.ndarray,
    logl: np.ndarray,
    theta_hat: np.ndarray,
    freq: float,
    ratio: float,
    lambdas: np.ndarray,
    freq_axis: np.ndarray,
    r23_curve: np.ndarray,
    cond: float,
    contour_levels: int,
    axis_x: np.ndarray | None = None,
    axis_y: np.ndarray | None = None,
    xlabel: str = r"$\theta_{31}$",
    ylabel: str = r"$\theta_{32}$",
    show_inset: bool = False,
    title: str | None = None,
    axis_x_range: tuple[float, float] | None = None,
    axis_y_range: tuple[float, float] | None = None,
    levels: np.ndarray | None = None,
    colorbar: bool = True,
) -> None:
    if levels is None:
        levels = np.linspace(np.nanmin(logl), 0.0, contour_levels)
    axis_x = axis_x if axis_x is not None else np.real(theta31)
    axis_y = axis_y if axis_y is not None else np.real(theta32)
    contour = ax.contourf(axis_x, axis_y, logl, levels=levels, cmap="viridis")
    ax.contour(
        axis_x,
        axis_y,
        logl,
        levels=levels[:: max(1, len(levels) // 4)],
        colors="k",
        linewidths=0.5,
    )
    ax.scatter(
        np.real(theta_hat[0]),
        np.real(theta_hat[1]),
        color="red",
        marker="x",
        label="LSQ (Re)",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if axis_x_range is not None:
        ax.set_xlim(*axis_x_range)
    if axis_y_range is not None:
        ax.set_ylim(*axis_y_range)
    if title is not None:
        ax.set_title(title, fontsize=10)
    else:
        ax.set_title(
            f"freq={freq:.3e} | r23={ratio:.3f} | Gram cond={cond:.2f}\n"
            f"λ={lambdas[0]:.3e}, {lambdas[1]:.3e}, {lambdas[2]:.3e}",
            fontsize=9,
        )
    ax.legend(loc="upper right")
    if show_inset:
        _draw_r23_inset(ax, freq_axis, r23_curve, freq)
    if colorbar:
        ax.figure.colorbar(contour, ax=ax, label=r"$\Delta \log L$")
    return contour


def _plot_slice(
    theta31: np.ndarray,
    theta32: np.ndarray,
    logl: np.ndarray,
    theta_hat: np.ndarray,
    freq: float,
    ratio: float,
    lambdas: np.ndarray,
    freq_axis: np.ndarray,
    r23_curve: np.ndarray,
    cond: float,
    outpath: Path,
    contour_levels: int,
    axis_x: np.ndarray | None = None,
    axis_y: np.ndarray | None = None,
    xlabel: str = r"$\theta_{31}$",
    ylabel: str = r"$\theta_{32}$",
) -> None:
    """Save a single-contour figure for the specified frequency bin."""
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_contour_on_axis(
        ax,
        theta31,
        theta32,
        logl,
        theta_hat,
        freq,
        ratio,
        lambdas,
        freq_axis,
        r23_curve,
        cond,
        contour_levels,
        axis_x=axis_x,
        axis_y=axis_y,
        xlabel=xlabel,
        ylabel=ylabel,
        show_inset=True,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _grid_range(arr: np.ndarray) -> tuple[float, float]:
    """Return the min/max of a grid (use real part if complex)."""
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _compute_theta_hat_and_delta(
    u_modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frequency θ estimates (θ31, θ32) and δ₃² scalings."""
    if u_modes.ndim != 3 or u_modes.shape[2] < 3:
        raise RuntimeError("Need at least three channels in u_modes.")
    N = u_modes.shape[0]
    theta_hat = np.empty((N, 2), dtype=np.complex128)
    delta3_sq = np.empty(N, dtype=float)
    tiny = np.finfo(float).tiny
    for idx in range(N):
        u1 = u_modes[idx, :, 0]
        u2 = u_modes[idx, :, 1]
        u3 = u_modes[idx, :, 2]
        X = np.stack([u1, u2], axis=-1)
        theta_hat[idx], *_ = np.linalg.lstsq(X, u3, rcond=None)
        delta3_sq[idx] = max(np.mean(np.abs(u3) ** 2), tiny)
    return theta_hat, delta3_sq


def _extract_offdiag_basis(
    idata: az.InferenceData,
) -> tuple[np.ndarray, np.ndarray]:
    """Grab the off-diagonal spline bases stored in the inference data."""
    spline_group = getattr(idata, "spline_model", None)
    if spline_group is None:
        raise RuntimeError(
            "idata missing `spline_model` group for basis extraction."
        )
    if (
        "offdiag_re_basis" not in spline_group
        or "offdiag_im_basis" not in spline_group
    ):
        raise RuntimeError("spline_model lacks off-diagonal basis matrices.")
    basis_re = np.asarray(spline_group["offdiag_re_basis"].values, dtype=float)
    basis_im = np.asarray(spline_group["offdiag_im_basis"].values, dtype=float)
    return basis_re, basis_im


def _weight_key(param: str, part: str) -> str:
    """Normalize parameter/part to the keys used in the weight info map."""
    suffix = "real" if part == "real" else "imag"
    return f"{param}_{suffix}"


def _build_weight_info(
    basis_re_row: np.ndarray,
    basis_im_row: np.ndarray,
    theta31: complex,
    theta32: complex,
) -> dict[str, dict[str, np.ndarray]]:
    """Construct minimal-norm weight direction entries per θ component."""
    tiny = float(np.finfo(float).tiny)
    norm_re = max(np.sum(basis_re_row**2), tiny)
    norm_im = max(np.sum(basis_im_row**2), tiny)
    dir_re = np.asarray(basis_re_row / norm_re, dtype=float)
    dir_im = np.asarray(basis_im_row / norm_im, dtype=float)

    def _entry(center: float, direction: np.ndarray) -> dict[str, np.ndarray]:
        center_val = float(center)
        return {
            "center": center_val,
            "w_center": direction * center_val,
            "w_dir": direction,
        }

    return {
        "theta31_real": _entry(np.real(theta31), dir_re),
        "theta31_imag": _entry(np.imag(theta31), dir_im),
        "theta32_real": _entry(np.real(theta32), dir_re),
        "theta32_imag": _entry(np.imag(theta32), dir_im),
    }


def _compute_coeff_logl_grid(
    axis_x_vals: np.ndarray,
    axis_y_vals: np.ndarray,
    axis_specs: tuple[tuple[str, str], tuple[str, str]],
    weight_info: dict[str, dict[str, np.ndarray]],
    basis_re: np.ndarray,
    basis_im: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    delta3_sq: np.ndarray,
) -> np.ndarray:
    """Evaluate the log likelihood surface over the coefficient axes."""
    ny = axis_y_vals.size
    nx = axis_x_vals.size
    logl_grid = np.empty((ny, nx), dtype=float)
    for iy, axis_y_val in enumerate(axis_y_vals):
        for ix, axis_x_val in enumerate(axis_x_vals):
            weights = {
                key: info["w_center"].copy()
                for key, info in weight_info.items()
            }
            for axis_val, spec in zip((axis_x_val, axis_y_val), axis_specs):
                key = _weight_key(spec[0], spec[1])
                info = weight_info[key]
                delta = axis_val - info["center"]
                weights[key] = info["w_center"] + delta * info["w_dir"]

            theta31_re = basis_re @ weights["theta31_real"]
            theta31_im = basis_im @ weights["theta31_imag"]
            theta32_re = basis_re @ weights["theta32_real"]
            theta32_im = basis_im @ weights["theta32_imag"]

            theta31 = theta31_re + 1j * theta31_im
            theta32 = theta32_re + 1j * theta32_im

            resid = u3 - u1 * theta31[:, None] - u2 * theta32[:, None]
            sse = np.sum(np.abs(resid) ** 2, axis=1)
            logl_val = -np.sum(sse / delta3_sq)
            logl_grid[iy, ix] = logl_val

    logl_grid -= np.nanmax(logl_grid)
    return logl_grid


def _build_coeff_slice_combos(
    *,
    freq_index: int,
    theta_hat: np.ndarray,
    basis_re: np.ndarray,
    basis_im: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    delta3_sq: np.ndarray,
    pad_factor: float,
    min_span: float,
    grid_size: int,
) -> list[dict]:
    """Produce contour combo dictionaries for coefficient-space slices."""
    theta31 = theta_hat[0]
    theta32 = theta_hat[1]
    span_31_re, span_31_im = _span_components(
        np.array([theta31], dtype=np.complex128), pad_factor, min_span
    )
    span_32_re, span_32_im = _span_components(
        np.array([theta32], dtype=np.complex128), pad_factor, min_span
    )
    weight_info = _build_weight_info(
        basis_re[freq_index], basis_im[freq_index], theta31, theta32
    )
    span_map = {
        "theta31": {"real": span_31_re, "imag": span_31_im},
        "theta32": {"real": span_32_re, "imag": span_32_im},
    }
    coeff_specs = [
        {
            "axis_x": ("theta31", "real"),
            "axis_y": ("theta32", "real"),
            "xlabel": r"$\Delta w_{31}$ (Re)",
            "ylabel": r"$\Delta w_{32}$ (Re)",
            "title": "Coefficient slice: Re(θ₃₁) / Re(θ₃₂)",
        },
        {
            "axis_x": ("theta31", "imag"),
            "axis_y": ("theta32", "imag"),
            "xlabel": r"$\Delta w_{31}$ (Im)",
            "ylabel": r"$\Delta w_{32}$ (Im)",
            "title": "Coefficient slice: Im(θ₃₁) / Im(θ₃₂)",
        },
        {
            "axis_x": ("theta31", "real"),
            "axis_y": ("theta31", "imag"),
            "xlabel": r"$\Delta w_{31}$ (Re)",
            "ylabel": r"$\Delta w_{31}$ (Im)",
            "title": "Coefficient slice: Re/Im (θ₃₁)",
        },
    ]

    combos: list[dict] = []
    for spec in coeff_specs:
        axis_x_key = _weight_key(*spec["axis_x"])
        axis_y_key = _weight_key(*spec["axis_y"])
        center_x = weight_info[axis_x_key]["center"]
        center_y = weight_info[axis_y_key]["center"]
        span_x = span_map[spec["axis_x"][0]][spec["axis_x"][1]]
        span_y = span_map[spec["axis_y"][0]][spec["axis_y"][1]]
        axis_x_vals = _linspace_center(center_x, span_x, grid_size)
        axis_y_vals = _linspace_center(center_y, span_y, grid_size)
        axis_x_grid, axis_y_grid = np.meshgrid(
            axis_x_vals, axis_y_vals, indexing="xy"
        )
        logl_grid = _compute_coeff_logl_grid(
            axis_x_vals,
            axis_y_vals,
            (spec["axis_x"], spec["axis_y"]),
            weight_info,
            basis_re,
            basis_im,
            u1,
            u2,
            u3,
            delta3_sq,
        )

        combos.append(
            {
                "theta31": axis_x_grid,
                "theta32": axis_y_grid,
                "logl": logl_grid,
                "axis_x": axis_x_grid,
                "axis_y": axis_y_grid,
                "xlabel": spec["xlabel"],
                "ylabel": spec["ylabel"],
                "title": spec["title"],
                "axis_x_range": (
                    float(np.nanmin(axis_x_grid)),
                    float(np.nanmax(axis_x_grid)),
                ),
                "axis_y_range": (
                    float(np.nanmin(axis_y_grid)),
                    float(np.nanmax(axis_y_grid)),
                ),
            }
        )

    return combos


def _plot_slice_columns(
    combos: list[dict],
    theta_hat: np.ndarray,
    freq: float,
    ratio: float,
    lambdas: np.ndarray,
    freq_axis: np.ndarray,
    r23_curve: np.ndarray,
    cond: float,
    outpath: Path,
    contour_levels: int,
) -> None:
    """Save a three-column figure showing Re/Re, Im/Im, and Re/Im slices."""
    fig, axes = plt.subplots(1, len(combos), figsize=(5.5 * len(combos), 5))
    if len(combos) == 1:
        axes = [axes]
    for ax, combo in zip(axes, combos):
        _draw_contour_on_axis(
            ax,
            combo["theta31"],
            combo["theta32"],
            combo["logl"],
            theta_hat,
            freq,
            ratio,
            lambdas,
            freq_axis,
            r23_curve,
            cond,
            contour_levels,
            axis_x=combo.get("axis_x"),
            axis_y=combo.get("axis_y"),
            xlabel=combo["xlabel"],
            ylabel=combo["ylabel"],
            show_inset=combo.get("show_inset", False),
            title=combo.get("title"),
            axis_x_range=combo.get("axis_x_range"),
            axis_y_range=combo.get("axis_y_range"),
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_combined(
    outputs: list[dict],
    freq_axis: np.ndarray,
    r23_curve: np.ndarray,
    outpath: Path,
    contour_levels: int,
) -> None:
    """Stack a grid of three-column slices for all selected frequencies."""
    if not outputs:
        return
    rows = len(outputs)
    cols = len(outputs[0]["combos"])
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    global_min = min(
        np.nanmin(combo["logl"])
        for entry in outputs
        for combo in entry["combos"]
        if combo["logl"] is not None
    )
    levels = np.linspace(global_min, 0.0, contour_levels)
    first_contour = None
    col_ranges_x = [(np.inf, -np.inf) for _ in range(cols)]
    col_ranges_y = [(np.inf, -np.inf) for _ in range(cols)]
    for entry in outputs:
        for j, combo in enumerate(entry["combos"]):
            rng_x = combo.get("axis_x_range")
            if rng_x is not None:
                col_ranges_x[j] = (
                    min(col_ranges_x[j][0], rng_x[0]),
                    max(col_ranges_x[j][1], rng_x[1]),
                )
            rng_y = combo.get("axis_y_range")
            if rng_y is not None:
                col_ranges_y[j] = (
                    min(col_ranges_y[j][0], rng_y[0]),
                    max(col_ranges_y[j][1], rng_y[1]),
                )
    for i, entry in enumerate(outputs):
        row_label = f"k={entry['k']} | freq={entry['freq']:.3e} | r23={entry['ratio']:.3f}"
        for j, combo in enumerate(entry["combos"]):
            ax = axes[i, j]
            combo_title = combo.get("title")
            axis_x_range = (
                col_ranges_x[j]
                if col_ranges_x[j][0] <= col_ranges_x[j][1]
                else None
            )
            axis_y_range = (
                col_ranges_y[j]
                if col_ranges_y[j][0] <= col_ranges_y[j][1]
                else None
            )
            contour_obj = _draw_contour_on_axis(
                ax,
                combo["theta31"],
                combo["theta32"],
                combo["logl"],
                entry["theta_hat"],
                entry["freq"],
                entry["ratio"],
                entry["lambdas"],
                freq_axis,
                r23_curve,
                entry["cond"],
                contour_levels,
                axis_x=combo.get("axis_x"),
                axis_y=combo.get("axis_y"),
                xlabel=combo.get("xlabel", r"$\theta_{31}$"),
                ylabel=combo.get("ylabel", r"$\theta_{32}$"),
                show_inset=(j == 0),
                title=combo_title if i == 0 else None,
                axis_x_range=axis_x_range,
                axis_y_range=axis_y_range,
                levels=levels,
                colorbar=False,
            )
            if first_contour is None:
                first_contour = contour_obj
            if j == 0:
                ax.text(
                    -0.1,
                    1.02,
                    row_label,
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                )
    if first_contour is not None:
        fig.colorbar(
            first_contour, ax=axes.flatten(), label=r"$\Delta \log L$"
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Likelihood contour slices for (theta31, theta32) | VAR3 / LISA data."
    )
    default_out = (
        Path(__file__).resolve().parent / "results" / "likelihood_slices"
    )
    parser.add_argument(
        "--idata", type=Path, default=None, help="Path to inference_data.nc."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to preprocessed NPZ bundle.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=default_out, help="Output directory."
    )
    parser.add_argument(
        "--n-freq-slices",
        type=int,
        default=3,
        help="How many representative bins.",
    )
    parser.add_argument(
        "--grid-size", type=int, default=151, help="Grid resolution per slice."
    )
    parser.add_argument(
        "--theta-range-mode",
        type=str,
        choices=("lsq", "fixed"),
        default="lsq",
        help="Center grid on LSQ estimate or keep zero-centered fixed range.",
    )
    parser.add_argument(
        "--theta-max-fixed",
        type=float,
        default=3.0,
        help="Half-width (per axis) when theta-range-mode=fixed.",
    )
    parser.add_argument(
        "--theta-pad-factor",
        type=float,
        default=3.0,
        help="Multiplier for |Re(theta_hat)| when theta-range-mode=lsq.",
    )
    parser.add_argument(
        "--theta-min-span",
        type=float,
        default=0.5,
        help="Minimum half-width for the LSQ-driven grid.",
    )
    parser.add_argument(
        "--k-indices",
        type=str,
        default="",
        help="Comma-separated frequency indices to override automatic selection.",
    )
    parser.add_argument(
        "--contour-levels",
        type=int,
        default=12,
        help="Number of filled contour levels.",
    )
    parser.add_argument(
        "--coeff-slices",
        action="store_true",
        help="Produce likelihood slices in spline coefficient space (requires --idata).",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    freq, matrices, u_modes, idata = _prepare_data(args)
    eigvals = _ordered_eigvals(matrices)
    if eigvals.shape[1] < 3:
        raise RuntimeError("Need at least 3 channels to inspect ratio r23.")
    r23 = np.divide(
        eigvals[:, 2],
        eigvals[:, 1],
        out=np.full_like(eigvals[:, 2], np.nan),
        where=np.abs(eigvals[:, 1]) > np.finfo(float).tiny,
    )

    theta_hat_all, delta3_sq = _compute_theta_hat_and_delta(u_modes)
    u1_full = u_modes[:, :, 0]
    u2_full = u_modes[:, :, 1]
    u3_full = u_modes[:, :, 2]
    basis_re = basis_im = None
    if args.coeff_slices:
        if idata is None:
            raise RuntimeError("Coefficient slices require --idata.")
        basis_re, basis_im = _extract_offdiag_basis(idata)
        if (
            basis_re.shape[0] != freq.shape[0]
            or basis_im.shape[0] != freq.shape[0]
        ):
            raise RuntimeError(
                "Basis frequency axis does not match observed data."
            )

    manual_indices = (
        [int(x) for x in args.k_indices.split(",") if x.strip()]
        if args.k_indices
        else None
    )
    selected_indices = _select_frequency_indices(
        r23, freq, args.n_freq_slices, manual_indices
    )

    outputs: list[dict] = []
    summary_lines: list[str] = []

    for k in selected_indices:
        u1 = u_modes[k, :, 0]
        u2 = u_modes[k, :, 1]
        u3 = u_modes[k, :, 2]
        y = u3.copy()
        X = np.stack([u1, u2], axis=-1)
        theta_hat = theta_hat_all[k]
        grams = X.conj().T @ X
        gram_eig = np.linalg.eigvalsh(grams)
        cond_gram = np.linalg.cond(grams)
        delta3_sq_k = delta3_sq[k]
        theta31_grid, theta32_grid = _build_grid(
            np.real(theta_hat),
            args.theta_range_mode,
            args.grid_size,
            args.theta_max_fixed,
            args.theta_pad_factor,
            args.theta_min_span,
        )
        logl = _evaluate_logl(
            u1, u2, u3, theta31_grid, theta32_grid, delta3_sq_k
        )
        span = _span_from_theta(
            theta_hat, args.theta_pad_factor, args.theta_min_span
        )
        size = args.grid_size
        imag_axis31 = _linspace_center(np.imag(theta_hat[0]), span, size)
        imag_axis32 = _linspace_center(np.imag(theta_hat[1]), span, size)
        theta31_im = np.real(theta_hat[0]) + 1j * imag_axis31[:, None]
        theta32_im = np.real(theta_hat[1]) + 1j * imag_axis32[None, :]
        theta31_im = np.broadcast_to(theta31_im, (size, size))
        theta32_im = np.broadcast_to(theta32_im, (size, size))
        logl_imim = _evaluate_logl(
            u1, u2, u3, theta31_im, theta32_im, delta3_sq_k
        )
        axis_x_im = np.broadcast_to(imag_axis31[:, None], (size, size))
        axis_y_im = np.broadcast_to(imag_axis32[None, :], (size, size))
        real_axis31 = _linspace_center(np.real(theta_hat[0]), span, size)
        imag_axis31_reim = _linspace_center(np.imag(theta_hat[0]), span, size)
        theta31_reim = real_axis31[:, None] + 1j * imag_axis31_reim[None, :]
        theta32_const = np.full_like(theta31_reim, theta_hat[1])
        logl_reim = _evaluate_logl(
            u1, u2, u3, theta31_reim, theta32_const, delta3_sq_k
        )
        axis_x_reim = np.broadcast_to(real_axis31[:, None], (size, size))
        axis_y_reim = np.broadcast_to(imag_axis31_reim[None, :], (size, size))
        real_range31 = _grid_range(theta31_grid)
        real_range32 = _grid_range(theta32_grid)
        imag_range31 = _grid_range(axis_x_im)
        imag_range32 = _grid_range(axis_y_im)
        re_reim_range31 = _grid_range(axis_x_reim)
        im_reim_range31 = _grid_range(axis_y_reim)
        combos = [
            {
                "theta31": theta31_grid,
                "theta32": theta32_grid,
                "logl": logl,
                "axis_x": theta31_grid,
                "axis_y": theta32_grid,
                "xlabel": r"$\Re(\theta_{31})$",
                "ylabel": r"$\Re(\theta_{32})$",
                "title": "Re(θ₃₁) / Re(θ₃₂)",
                "show_inset": True,
                "axis_x_range": real_range31,
                "axis_y_range": real_range32,
            },
            {
                "theta31": theta31_im,
                "theta32": theta32_im,
                "logl": logl_imim,
                "axis_x": axis_x_im,
                "axis_y": axis_y_im,
                "xlabel": r"$\Im(\theta_{31})$",
                "ylabel": r"$\Im(\theta_{32})$",
                "title": "Im(θ₃₁) / Im(θ₃₂)",
                "axis_x_range": imag_range31,
                "axis_y_range": imag_range32,
            },
            {
                "theta31": theta31_reim,
                "theta32": theta32_const,
                "logl": logl_reim,
                "axis_x": axis_x_reim,
                "axis_y": axis_y_reim,
                "xlabel": r"$\Re(\theta_{31})$",
                "ylabel": r"$\Im(\theta_{31})$",
                "title": "Re(θ₃₁) / Im(θ₃₁)",
                "axis_x_range": re_reim_range31,
                "axis_y_range": im_reim_range31,
            },
        ]
        combined_path = (
            args.outdir / f"likelihood_slice_k{k:04d}_threepanel.png"
        )
        _plot_slice_columns(
            combos,
            theta_hat,
            freq[k],
            float(r23[k]),
            eigvals[k, :3],
            freq,
            r23,
            cond_gram,
            combined_path,
            args.contour_levels,
        )
        if args.coeff_slices:
            coeff_combos = _build_coeff_slice_combos(
                freq_index=k,
                theta_hat=theta_hat,
                basis_re=basis_re,
                basis_im=basis_im,
                u1=u1_full,
                u2=u2_full,
                u3=u3_full,
                delta3_sq=delta3_sq,
                pad_factor=args.theta_pad_factor,
                min_span=args.theta_min_span,
                grid_size=args.grid_size,
            )
            coeff_path = args.outdir / f"likelihood_slice_k{k:04d}_coeff.png"
            _plot_slice_columns(
                coeff_combos,
                theta_hat,
                freq[k],
                float(r23[k]),
                eigvals[k, :3],
                freq,
                r23,
                cond_gram,
                coeff_path,
                args.contour_levels,
            )

        outputs.append(
            {
                "k": k,
                "freq": float(freq[k]),
                "ratio": float(r23[k]),
                "lambdas": eigvals[k, :3],
                "theta_hat": theta_hat,
                "cond": cond_gram,
                "combos": combos,
                "slice_path": combined_path,
            }
        )
        summary_lines.append(
            (
                f"k={k} freq={freq[k]:.4e} r23={float(r23[k]):.3f} "
                f"θ_hat=[{theta_hat[0]:.4f}, {theta_hat[1]:.4f}] "
                f"Gram cond={cond_gram:.1f} min(G)={gram_eig[0]:.3e} slice={combined_path.name}"
            )
        )

        summary_lines.append(
            (
                f"k={k} freq={freq[k]:.4e} r23={float(r23[k]):.3f} "
                f"θ_hat=[{theta_hat[0]:.4f}, {theta_hat[1]:.4f}] "
                f"Gram cond={cond_gram:.1f} min(G)={gram_eig[0]:.3e} "
                f"slice={combined_path.name}"
            )
        )

    combined_path = args.outdir / "likelihood_slices_combined.png"
    _plot_combined(outputs, freq, r23, combined_path, args.contour_levels)

    print("\n".join(summary_lines))
    print(f"Individual slices saved to {args.outdir}")
    print(f"Combined panel saved to {combined_path}")


if __name__ == "__main__":
    main()
