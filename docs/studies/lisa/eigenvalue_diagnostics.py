"""Eigenvalue separation diagnostics for LISA multivariate preprocessing.

This is a cheap, sampler-independent check for weak identifiability in the
multivariate blocked model. For each frequency bin f_k we compute the ordered
eigenvalues λ1≥λ2≥λ3 of the empirical 3×3 spectral matrix Σ(f_k) and inspect
ratios:

  r12 = λ2/λ1,  r23 = λ3/λ2.

When r12≈1 or r23≈1 over a wide band, the corresponding modes are weakly
separated and the last blocked channel (θ31, θ32, δ3) can exhibit ridges,
small step sizes, and max tree depth hits.

Default behavior:
- Load the empirical spectral matrix from `inference_data.nc` (observed_data).
- Overlay the analytic truth from `lisatools_synth_data.npz` if available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arviz as az  # noqa: E402


def _interp_complex_matrix(
    freq_src: np.ndarray, matrix: np.ndarray, freq_tgt: np.ndarray
) -> np.ndarray:
    """Interpolate a complex matrix along frequency with duplicate-safe handling."""
    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    matrix = np.asarray(matrix)

    sort_idx = np.argsort(freq_src)
    freq_sorted = freq_src[sort_idx]
    matrix_sorted = matrix[sort_idx]
    freq_unique, uniq_idx = np.unique(freq_sorted, return_index=True)
    matrix_unique = matrix_sorted[uniq_idx]

    if freq_unique.shape == freq_tgt.shape and np.allclose(
        freq_unique, freq_tgt
    ):
        return np.asarray(matrix_unique)

    flat = matrix_unique.reshape(matrix_unique.shape[0], -1)
    real_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, idx].real)
            for idx in range(flat.shape[1])
        ]
    ).T
    imag_interp = np.vstack(
        [
            np.interp(freq_tgt, freq_unique, flat[:, idx].imag)
            for idx in range(flat.shape[1])
        ]
    ).T
    return (real_interp + 1j * imag_interp).reshape(
        (freq_tgt.size,) + matrix_unique.shape[1:]
    )


def _ordered_eigvals_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Return eigenvalues ordered descending for each frequency bin.

    matrix: (n_freq, n_dim, n_dim), assumed (approximately) Hermitian.
    returns: (n_freq, n_dim) with λ1≥…≥λd.
    """
    matrix = np.asarray(matrix)
    # Numeric guard: enforce Hermitian symmetry in case of tiny imag asymmetry.
    herm = 0.5 * (matrix + np.swapaxes(np.conj(matrix), -1, -2))
    eig = np.linalg.eigvalsh(herm)  # ascending
    eig = np.maximum(eig.real, 0.0)
    return eig[:, ::-1]


def _ratio_summary(
    name: str, ratio: np.ndarray, *, warn_threshold: float = 0.8
) -> str:
    ratio = np.asarray(ratio, dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return f"{name}: (no finite values)"
    p05, p50, p95 = np.percentile(ratio, [5.0, 50.0, 95.0])
    frac = float(np.mean(ratio > warn_threshold))
    return (
        f"{name}: p05={p05:.3f}, p50={p50:.3f}, p95={p95:.3f}, "
        f"min={ratio.min():.3f}, max={ratio.max():.3f}, "
        f"frac(>{warn_threshold:.2f})={frac*100:.1f}%"
    )


def _eig_ratios(
    eigvals_desc: np.ndarray, *, eps: float | None = None
) -> dict[str, np.ndarray]:
    eigvals_desc = np.asarray(eigvals_desc, dtype=float)
    n_dim = eigvals_desc.shape[1]
    ratios: dict[str, np.ndarray] = {}
    if eps is None:
        eps = float(np.finfo(np.float64).tiny)
    for i in range(n_dim - 1):
        num = eigvals_desc[:, i + 1]
        den = eigvals_desc[:, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = num / den
        ratio = np.where(den <= eps, np.nan, ratio)
        ratios[f"r_{i+1}{i+2}"] = np.clip(ratio, 0.0, 1.0)
    return ratios


def main() -> None:
    here = Path(__file__).resolve().parent
    results_dir = here / "results" / "lisa"
    parser = argparse.ArgumentParser(
        description="Eigenvalue separation diagnostics for LISA empirical spectral matrices."
    )
    parser.add_argument(
        "--idata",
        type=Path,
        default=results_dir / "inference_data.nc",
        help="Path to ArviZ inference_data.nc (uses observed_data.periodogram).",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=results_dir / "lisatools_synth_data.npz",
        help="Path to lisatools_synth_data.npz (for true PSD overlay).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=results_dir / "eigenvalue_ratios.png",
        help="Output PNG path.",
    )
    parser.add_argument("--warn_threshold", type=float, default=0.8)
    parser.add_argument(
        "--min_lambda1_quantile",
        type=float,
        default=0.0,
        help=(
            "Optional filter for summaries: ignore bins where the largest eigenvalue "
            "λ1 is below this quantile (useful to de-emphasize deep notches). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many worst-separated frequencies to print per ratio.",
    )
    args = parser.parse_args()

    if not args.idata.exists():
        raise FileNotFoundError(
            f"{args.idata} not found. Run `docs/studies/lisa/lisa_multivar.py` first."
        )

    idata = az.from_netcdf(str(args.idata))
    if (
        "observed_data" not in idata
        or "periodogram" not in idata["observed_data"]
    ):
        raise RuntimeError("idata is missing observed_data['periodogram'].")
    periodogram = np.asarray(idata["observed_data"]["periodogram"].values)
    freq = np.asarray(
        idata["observed_data"]["periodogram"].coords["freq"].values,
        dtype=float,
    )

    eig_emp = _ordered_eigvals_hermitian(periodogram)
    ratios_emp = _eig_ratios(eig_emp)

    ratios_true = None
    if args.npz.exists():
        with np.load(args.npz, allow_pickle=False) as synth:
            if "freq_true" in synth.files and "true_matrix" in synth.files:
                freq_true = np.asarray(synth["freq_true"], dtype=float)
                true_matrix = np.asarray(synth["true_matrix"])
                true_on_grid = _interp_complex_matrix(
                    freq_true, true_matrix, freq
                )
                eig_true = _ordered_eigvals_hermitian(true_on_grid)
                ratios_true = _eig_ratios(eig_true)

    print("=== Eigenvalue Separation Diagnostics ===")
    mask = None
    if args.min_lambda1_quantile > 0.0:
        q = float(args.min_lambda1_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("--min_lambda1_quantile must be in (0,1).")
        cutoff = float(np.quantile(eig_emp[:, 0], q))
        mask = eig_emp[:, 0] > cutoff
        kept = int(np.count_nonzero(mask))
        print(
            f"Summary mask: keep λ1 > q{q:.2f} cutoff={cutoff:.3e} ({kept}/{mask.size} bins)"
        )

    def _maybe_mask(arr: np.ndarray) -> np.ndarray:
        if mask is None:
            return arr
        return np.asarray(arr)[mask]

    for key, ratio in ratios_emp.items():
        ratio_m = _maybe_mask(ratio)
        print(
            _ratio_summary(
                f"Empirical {key}", ratio_m, warn_threshold=args.warn_threshold
            )
        )
    if ratios_true is not None:
        for key, ratio in ratios_true.items():
            ratio_m = _maybe_mask(ratio)
            print(
                _ratio_summary(
                    f"True {key}", ratio_m, warn_threshold=args.warn_threshold
                )
            )

    top_k = max(0, int(args.top_k))
    if top_k:
        print("\nWorst-separated frequencies (empirical):")
        for key, ratio in ratios_emp.items():
            vals = np.asarray(ratio, dtype=float)
            good = np.isfinite(vals)
            if mask is not None:
                good &= mask
            vals = np.where(good, vals, -np.inf)
            idx = np.argsort(vals)[::-1][:top_k]
            idx = idx[np.isfinite(vals[idx])]
            if idx.size == 0:
                continue
            freqs = freq[idx]
            tops = vals[idx]
            joined = ", ".join(
                [f"{f:.4g}:{v:.3f}" for f, v in zip(freqs, tops)]
            )
            print(f"  {key}: {joined}")

    n_dim = eig_emp.shape[1]
    if ratios_true is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes = np.asarray(axes)

        # Ratios (empirical only)
        ax = axes[0]
        for key, ratio in ratios_emp.items():
            ax.semilogx(freq, ratio, label=f"Empirical {key}", alpha=0.9)
        ax.axhline(
            args.warn_threshold,
            color="k",
            ls="--",
            lw=1,
            alpha=0.6,
            label="warn threshold",
        )
        ax.set_ylabel("Eigenvalue ratio")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_title("Eigenvalue separation ratios (empirical)")

        # Raw eigenvalues (context)
        ax = axes[1]
        for idx in range(n_dim):
            ax.loglog(freq, eig_emp[:, idx], label=f"Empirical λ{idx+1}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Eigenvalue scale (arb.)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8, ncol=3)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes = np.asarray(axes)

        # Ratios: empirical
        ax = axes[0]
        for key, ratio in ratios_emp.items():
            ax.semilogx(freq, ratio, label=key.replace("r_", "r"), alpha=0.9)
        ax.axhline(
            args.warn_threshold,
            color="k",
            ls="--",
            lw=1,
            alpha=0.6,
            label="warn threshold",
        )
        ax.set_ylabel("Ratio")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_title("Eigenvalue ratios (empirical)")

        # Ratios: true
        ax = axes[1]
        for key, ratio in ratios_true.items():
            ax.semilogx(freq, ratio, label=key.replace("r_", "r"), lw=1.2)
        ax.axhline(
            args.warn_threshold,
            color="k",
            ls="--",
            lw=1,
            alpha=0.6,
            label="warn threshold",
        )
        ax.set_ylabel("Ratio")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_title("Eigenvalue ratios (true)")

        # Raw eigenvalues (context)
        ax = axes[2]
        for idx in range(n_dim):
            ax.loglog(freq, eig_emp[:, idx], label=f"Empirical λ{idx+1}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Eigenvalue scale (arb.)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8, ncol=3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
