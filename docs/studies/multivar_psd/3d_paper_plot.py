from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12
warnings.filterwarnings(
    "ignore", message="Attempt to set non-positive ylim on a log-scaled axis"
)


def _calculate_true_var_psd_hz(
    freqs_hz: np.ndarray,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    *,
    fs: float = 1.0,
) -> np.ndarray:
    """Compute one-sided theoretical PSD matrix for VAR(p)."""
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    ar_order, n_channels, _ = var_coeffs.shape
    omega = 2.0 * np.pi * freqs_hz / float(fs)
    psd = np.empty(
        (freqs_hz.shape[0], n_channels, n_channels), dtype=np.complex128
    )
    ident = np.eye(n_channels, dtype=np.complex128)

    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, ar_order + 1):
            a_f = a_f - var_coeffs[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        s_f = h_f @ sigma @ h_f.conj().T
        psd[idx] = (2.0 / float(fs)) * s_f

    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]

    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    psd = np.where(np.abs(psd) < EPS, EPS, psd)
    return psd


def _nearest_percentile(
    values: np.ndarray, percentiles: np.ndarray, q: float
) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - q)))
    return np.asarray(values[idx], dtype=np.float64)


def _resolve_default_idata(repo_root: Path) -> Path:
    preferred = (
        repo_root
        / "docs/studies/multivar_psd/out_var3/seed_0_large_N16384_K10/inference_data.nc"
    )
    if preferred.exists():
        return preferred

    candidates = sorted(
        (repo_root / "docs/studies/multivar_psd/out_var3").glob(
            "seed_*_*/inference_data.nc"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find any inference_data.nc under docs/studies/multivar_psd/out_var3."
        )
    return candidates[0]


def _resolve_default_ci_npz(repo_root: Path) -> Path | None:
    preferred = (
        repo_root
        / "docs/studies/multivar_psd/out_var3/seed_0_large_N16384_K10/compact_ci_curves.npz"
    )
    if preferred.exists():
        return preferred

    candidates = sorted(
        (repo_root / "docs/studies/multivar_psd/out_var3").glob(
            "seed_*_*/compact_ci_curves.npz"
        )
    )
    if candidates:
        return candidates[0]
    return None


def _reconstruct_quantiles_from_compact(data) -> tuple[np.ndarray, ...]:
    """Rebuild full (F, P, P) quantile arrays from compact diag/offdiag format."""
    freq = np.asarray(data["freq"], dtype=np.float64)
    diag_q05 = np.asarray(data["psd_diag_q05"], dtype=np.float64)
    diag_q50 = np.asarray(data["psd_diag_q50"], dtype=np.float64)
    diag_q95 = np.asarray(data["psd_diag_q95"], dtype=np.float64)
    off_re_q05 = np.asarray(data["psd_offre_q05"], dtype=np.float64)
    off_re_q50 = np.asarray(data["psd_offre_q50"], dtype=np.float64)
    off_re_q95 = np.asarray(data["psd_offre_q95"], dtype=np.float64)
    off_im_q05 = np.asarray(data["psd_offim_q05"], dtype=np.float64)
    off_im_q50 = np.asarray(data["psd_offim_q50"], dtype=np.float64)
    off_im_q95 = np.asarray(data["psd_offim_q95"], dtype=np.float64)
    pairs = np.asarray(data["offdiag_pairs"], dtype=int)

    p = int(diag_q50.shape[1])
    f = int(freq.size)

    q05_real = np.zeros((f, p, p), dtype=np.float64)
    q50_real = np.zeros((f, p, p), dtype=np.float64)
    q95_real = np.zeros((f, p, p), dtype=np.float64)
    q05_imag = np.zeros((f, p, p), dtype=np.float64)
    q50_imag = np.zeros((f, p, p), dtype=np.float64)
    q95_imag = np.zeros((f, p, p), dtype=np.float64)

    diag_idx = np.arange(p)
    q05_real[:, diag_idx, diag_idx] = diag_q05
    q50_real[:, diag_idx, diag_idx] = diag_q50
    q95_real[:, diag_idx, diag_idx] = diag_q95

    for k, (i, j) in enumerate(pairs):
        q05_real[:, i, j] = off_re_q05[:, k]
        q50_real[:, i, j] = off_re_q50[:, k]
        q95_real[:, i, j] = off_re_q95[:, k]

        # Hermitian structure: lower triangle real mirrors upper triangle.
        q05_real[:, j, i] = off_re_q05[:, k]
        q50_real[:, j, i] = off_re_q50[:, k]
        q95_real[:, j, i] = off_re_q95[:, k]

        # Lower triangle stores imag part in plotting convention.
        q05_imag[:, j, i] = off_im_q05[:, k]
        q50_imag[:, j, i] = off_im_q50[:, k]
        q95_imag[:, j, i] = off_im_q95[:, k]

        # Enforce antisymmetry in upper triangle for completeness.
        q05_imag[:, i, j] = -off_im_q05[:, k]
        q50_imag[:, i, j] = -off_im_q50[:, k]
        q95_imag[:, i, j] = -off_im_q95[:, k]

    return freq, q05_real, q50_real, q95_real, q05_imag, q50_imag, q95_imag


def _load_compact_ci_npz(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as data:
        freq = np.asarray(data["freq"], dtype=np.float64)

        if all(
            key in data
            for key in (
                "psd_real_q05",
                "psd_real_q50",
                "psd_real_q95",
                "psd_imag_q05",
                "psd_imag_q50",
                "psd_imag_q95",
            )
        ):
            q05_real = np.asarray(data["psd_real_q05"], dtype=np.float64)
            q50_real = np.asarray(data["psd_real_q50"], dtype=np.float64)
            q95_real = np.asarray(data["psd_real_q95"], dtype=np.float64)
            q05_imag = np.asarray(data["psd_imag_q05"], dtype=np.float64)
            q50_imag = np.asarray(data["psd_imag_q50"], dtype=np.float64)
            q95_imag = np.asarray(data["psd_imag_q95"], dtype=np.float64)
        else:
            (
                freq,
                q05_real,
                q50_real,
                q95_real,
                q05_imag,
                q50_imag,
                q95_imag,
            ) = _reconstruct_quantiles_from_compact(data)

        periodogram = None
        if "periodogram_real" in data and "periodogram_imag" in data:
            periodogram = np.asarray(
                data["periodogram_real"], dtype=np.float64
            ) + 1j * np.asarray(data["periodogram_imag"], dtype=np.float64)

        true_psd = None
        if "true_psd_real" in data and "true_psd_imag" in data:
            true_psd = np.asarray(
                data["true_psd_real"], dtype=np.float64
            ) + 1j * np.asarray(data["true_psd_imag"], dtype=np.float64)

    return (
        freq,
        q05_real,
        q50_real,
        q95_real,
        q05_imag,
        q50_imag,
        q95_imag,
        periodogram,
        true_psd,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper plot from 3D VAR(2) InferenceData."
    )
    parser.add_argument(
        "--idata",
        type=str,
        default=None,
        help="Path to inference_data.nc. Defaults to seed_0_large_N16384_K10 if present.",
    )
    parser.add_argument(
        "--ci-npz",
        type=str,
        default=None,
        help="Path to compact_ci_curves.npz. Preferred when available.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/manuscript/figures/var3_simulation_idata.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--with-true",
        action="store_true",
        help="Overlay theoretical true VAR(2) spectrum used in 3d_study.py.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=0.5,
        help="Upper x-limit in Hz for paper plot focus region.",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="Plot every Nth frequency point (default 1 = no decimation).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    loaded_source = ""
    ci_npz_path = None
    if args.ci_npz:
        ci_npz_path = Path(args.ci_npz)
        if not ci_npz_path.is_absolute():
            ci_npz_path = repo_root / ci_npz_path
    else:
        ci_npz_path = _resolve_default_ci_npz(repo_root)

    true_psd_file = None
    if ci_npz_path is not None and ci_npz_path.exists():
        (
            freq,
            q05_real,
            q50_real,
            q95_real,
            q05_imag,
            q50_imag,
            q95_imag,
            periodogram,
            true_psd_file,
        ) = _load_compact_ci_npz(ci_npz_path)
        loaded_source = f"compact CI NPZ: {ci_npz_path}"
    else:
        if args.idata:
            idata_path = Path(args.idata)
            if not idata_path.is_absolute():
                idata_path = repo_root / idata_path
        else:
            idata_path = _resolve_default_idata(repo_root)

        idata = az.from_netcdf(idata_path)
        if not hasattr(idata, "posterior_psd"):
            raise ValueError("InferenceData has no posterior_psd group.")
        if "psd_matrix_real" not in idata.posterior_psd:
            raise ValueError("posterior_psd has no psd_matrix_real variable.")

        psd_group = idata.posterior_psd
        freq = np.asarray(psd_group.coords["freq"].values, dtype=np.float64)
        percentiles = np.asarray(
            psd_group.coords["percentile"].values, dtype=np.float64
        )
        psd_real = np.asarray(
            psd_group["psd_matrix_real"].values, dtype=np.float64
        )
        psd_imag = np.asarray(
            psd_group["psd_matrix_imag"].values, dtype=np.float64
        )

        q05_real = _nearest_percentile(psd_real, percentiles, 5.0)
        q50_real = _nearest_percentile(psd_real, percentiles, 50.0)
        q95_real = _nearest_percentile(psd_real, percentiles, 95.0)
        q05_imag = _nearest_percentile(psd_imag, percentiles, 5.0)
        q50_imag = _nearest_percentile(psd_imag, percentiles, 50.0)
        q95_imag = _nearest_percentile(psd_imag, percentiles, 95.0)

        periodogram = None
        if (
            hasattr(idata, "observed_data")
            and "periodogram" in idata.observed_data
        ):
            periodogram = np.asarray(idata.observed_data["periodogram"].values)
        loaded_source = f"InferenceData: {idata_path}"

    true_psd = None
    if args.with_true:
        if true_psd_file is not None:
            true_psd = true_psd_file
        else:
            a1 = np.diag([0.4, 0.3, 0.2])
            a2 = np.array(
                [
                    [-0.2, 0.5, 0.0],
                    [0.4, -0.1, 0.0],
                    [0.0, 0.0, -0.1],
                ],
                dtype=np.float64,
            )
            var_coeffs = np.array([a1, a2], dtype=np.float64)
            sigma = np.array(
                [
                    [0.25, 0.0, 0.08],
                    [0.0, 0.25, 0.08],
                    [0.08, 0.08, 0.25],
                ],
                dtype=np.float64,
            )
            true_psd = _calculate_true_var_psd_hz(
                freq, var_coeffs, sigma, fs=1.0
            )

    n_channels = q50_real.shape[1]
    fig, axes = plt.subplots(
        n_channels,
        n_channels,
        figsize=(n_channels * 2.6, n_channels * 2.6),
        sharex=True,
        constrained_layout=False,
    )
    if n_channels == 1:
        axes = np.array([[axes]])

    step = max(1, int(args.decimate))
    idx = np.arange(0, freq.size, step, dtype=int)
    if idx[-1] != freq.size - 1:
        idx = np.append(idx, freq.size - 1)

    x_min = float(freq[0])
    x_max = float(args.xmax) if args.xmax is not None else float(freq[-1])
    x_max = min(x_max, float(freq[-1]))
    x_mask = (freq >= x_min) & (freq <= x_max)
    if not np.any(x_mask):
        x_mask = np.ones_like(freq, dtype=bool)

    panel_data: dict[tuple[int, int], dict[str, np.ndarray | str | None]] = {}
    re_candidates: list[np.ndarray] = []
    im_candidates: list[np.ndarray] = []
    re_obs_candidates: list[np.ndarray] = []
    im_obs_candidates: list[np.ndarray] = []

    for i in range(n_channels):
        for j in range(n_channels):
            if i <= j:
                median = q50_real[:, i, j]
                lower = q05_real[:, i, j]
                upper = q95_real[:, i, j]
                obs = (
                    np.real(periodogram[:, i, j])
                    if periodogram is not None
                    else None
                )
                truth = (
                    np.real(true_psd[:, i, j])
                    if true_psd is not None
                    else None
                )
                ylabel = r"$\Re\{S_{%d%d}(f)\}$" % (i + 1, j + 1)
                panel_type = "real"
            else:
                median = q50_imag[:, i, j]
                lower = q05_imag[:, i, j]
                upper = q95_imag[:, i, j]
                obs = (
                    np.imag(periodogram[:, i, j])
                    if periodogram is not None
                    else None
                )
                truth = (
                    np.imag(true_psd[:, i, j])
                    if true_psd is not None
                    else None
                )
                ylabel = r"$\Im\{S_{%d%d}(f)\}$" % (i + 1, j + 1)
                panel_type = "imag"

            panel_data[(i, j)] = {
                "panel_type": panel_type,
                "median": median,
                "lower": lower,
                "upper": upper,
                "obs": obs,
                "truth": truth,
                "ylabel": ylabel,
            }

            target = re_candidates if panel_type == "real" else im_candidates
            target.extend([lower[x_mask], upper[x_mask], median[x_mask]])
            if obs is not None:
                target.append(obs[x_mask])
                if panel_type == "real":
                    re_obs_candidates.append(obs[x_mask])
                else:
                    im_obs_candidates.append(obs[x_mask])
            if truth is not None:
                target.append(truth[x_mask])

    def _global_limits(
        candidates: list[np.ndarray], symmetric: bool
    ) -> tuple[float, float]:
        vals = np.concatenate([np.ravel(c) for c in candidates if c.size])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (-1.0, 1.0) if symmetric else (0.0, 1.0)
        if symmetric:
            # Robust symmetric bounds to avoid a few outlier spikes
            vmax = max(float(np.percentile(np.abs(vals), 99.0)), 1e-8)
            return -1.1 * vmax, 1.1 * vmax
        # Robust bounds for real-valued panels
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            span = max(abs(lo), 1.0)
            return lo - 0.1 * span, hi + 0.1 * span
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    # Prefer automatic limits from periodogram; fallback to posterior/truth.
    re_ylim = _global_limits(
        re_obs_candidates if re_obs_candidates else re_candidates,
        symmetric=False,
    )
    im_ylim = _global_limits(
        im_obs_candidates if im_obs_candidates else im_candidates,
        symmetric=True,
    )

    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j]
            panel = panel_data[(i, j)]
            panel_type = str(panel["panel_type"])
            median = np.asarray(panel["median"])
            lower = np.asarray(panel["lower"])
            upper = np.asarray(panel["upper"])
            obs = panel["obs"]
            truth = panel["truth"]
            ylabel = str(panel["ylabel"])

            freq_plot = freq[idx]
            med_plot = median[idx]
            low_plot = lower[idx]
            up_plot = upper[idx]

            ax.fill_between(
                freq_plot,
                low_plot,
                up_plot,
                color="tab:blue",
                alpha=0.60,
                linewidth=0.0,
                zorder=5,
                label="90% CI" if (i == 0 and j == 0) else None,
            )
            # ax.plot(
            #     freq_plot,
            #     med_plot,
            #     color="tab:blue",
            #     lw=1.2,
            #     zorder=6,
            #     alpha=0.9,
            #     label="Posterior median" if (i == 0 and j == 0) else None,
            # )

            if obs is not None:
                obs_arr = np.asarray(obs)
                ax.plot(
                    freq_plot,
                    obs_arr[idx],
                    color="0.82",
                    lw=0.7,
                    alpha=0.9,
                    zorder=-10,
                    label="Periodogram" if (i == 0 and j == 0) else None,
                )
            if truth is not None:
                truth_arr = np.asarray(truth)
                ax.plot(
                    freq_plot,
                    truth_arr[idx],
                    color="k",
                    lw=2.0,
                    ls="--",
                    zorder=0,
                    alpha=0.8,
                    label="True PSD" if (i == 0 and j == 0) else None,
                )

            ax.set_xlim(x_min, x_max)
            if panel_type == "real":
                ax.set_ylim(*re_ylim)
            else:
                ax.set_ylim(*im_ylim)
                ax.axhline(0.0, color="0.35", lw=0.7, alpha=0.7, zorder=2)
                ax.set_facecolor((0.96, 0.96, 0.96))

            # Panel-specific overrides requested for paper figure readability.
            panel_key = (i + 1, j + 1)
            if panel_key in {(1, 1), (2, 2)}:
                ax.set_ylim(0.0, 3.0)
            elif panel_key in {(1, 3), (2, 3)}:
                ax.set_ylim(-0.5, 1.0)
            elif panel_key == (3, 3):
                ax.set_ylim(0.0, 1.5)
            elif panel_key in {(3, 1), (3, 2)}:
                ax.set_ylim(-0.25, 0.5)

            ax.grid(alpha=0.25, linewidth=0.5)
            if i == n_channels - 1:
                ax.set_xlabel("Frequency (Hz)")
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"({i+1},{j+1})", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles),
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(
            output_path.with_suffix(".png"), dpi=220, bbox_inches="tight"
        )

    print(f"Loaded source: {loaded_source}")
    print(f"Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
