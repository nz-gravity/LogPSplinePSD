from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from paper_final_plots import (
    FMAX,
    FMIN,
    TRUE_COLOR,
    TRUE_LW,
    WELCH_ALPHA,
    WELCH_COLOR,
    WELCH_LW,
    _generate_xyz_for_welch,
    _welch_psd,
)

from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.logger import logger

HERE = Path(__file__).resolve().parent
DEFAULT_RUN_X = (
    HERE
    / "runs"
    / "run_x_d2_k48_uniform_no_excision"
    / "k48_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff"
    / "seed_0"
)
DEFAULT_OUTFILE = "debug_runx_data_only.png"
CHANNEL_LABELS = ["X", "Y", "Z"]


def _load_truth_from_idata(
    idata: az.InferenceData,
) -> tuple[np.ndarray, np.ndarray] | None:
    truth_group = getattr(idata, "truth_psd", None)
    if truth_group is None:
        return None
    if (
        "psd_matrix_real" not in truth_group
        or "psd_matrix_imag" not in truth_group
    ):
        return None
    freq = np.asarray(
        truth_group["psd_matrix_real"].coords["freq"].values, dtype=float
    )
    truth = np.asarray(
        truth_group["psd_matrix_real"].values
    ) + 1j * np.asarray(truth_group["psd_matrix_imag"].values)
    return freq, truth


def build_debug_plot(
    run_dir: Path,
    *,
    outdir: Path,
    filename: str,
    seed: int,
    duration_days: float,
    include_truth: bool = True,
) -> Path:
    idata_path = run_dir / "inference_data.nc"
    if not idata_path.exists():
        raise FileNotFoundError(f"InferenceData file not found: {idata_path}")

    idata = az.from_netcdf(str(idata_path))
    if (
        "observed_data" not in idata
        or "periodogram" not in idata["observed_data"]
    ):
        raise ValueError(f"{idata_path} is missing observed_data.periodogram.")

    periodogram = idata["observed_data"]["periodogram"]
    freq = np.asarray(periodogram.coords["freq"].values, dtype=float)
    psd_obs = np.asarray(periodogram.values, dtype=np.complex128)

    y_xyz, Nb, Lb, fs = _generate_xyz_for_welch(
        seed=seed, duration_days=duration_days
    )
    welch_freq, welch_psd = _welch_psd(y_xyz, Lb=Lb, fs=fs)
    welch_on_grid = interp_matrix(welch_freq, welch_psd, freq)

    truth_on_grid = None
    if include_truth:
        truth_loaded = _load_truth_from_idata(idata)
        if truth_loaded is None:
            logger.warning(
                "truth_psd not found in idata; omitting truth overlay."
            )
        else:
            truth_freq, truth_psd = truth_loaded
            truth_on_grid = interp_matrix(truth_freq, truth_psd, freq)

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    fig.suptitle("LISA Data Debug: observed_data vs block-Welch", fontsize=13)

    for idx, ax in enumerate(axes):
        obs_diag = np.asarray(psd_obs[:, idx, idx].real, dtype=float)
        welch_diag = np.asarray(welch_on_grid[:, idx, idx].real, dtype=float)

        ax.plot(
            freq,
            obs_diag,
            color="0.65",
            lw=0.7,
            alpha=0.9,
            label="observed_data.periodogram",
        )
        ax.plot(
            freq,
            welch_diag,
            color=WELCH_COLOR,
            lw=max(WELCH_LW, 1.0),
            alpha=max(WELCH_ALPHA, 0.85),
            label="block Welch",
        )
        if truth_on_grid is not None:
            ax.plot(
                freq,
                np.asarray(truth_on_grid[:, idx, idx].real, dtype=float),
                color=TRUE_COLOR,
                lw=TRUE_LW,
                ls="--",
                label="truth",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(FMIN, FMAX)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_title(f"{CHANNEL_LABELS[idx]} channel")
        if idx == 0:
            ax.set_ylabel("PSD [1/Hz]")
        ax.grid(alpha=0.15, which="both")

    axes[0].legend(fontsize=8, loc="best")
    out_path = outdir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved LISA data debug plot to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LISA debug plot showing only empirical data products "
            "(observed_data.periodogram and regenerated block-Welch), with "
            "optional truth."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_X)
    parser.add_argument("--outdir", type=Path, default=HERE / "paper_figs")
    parser.add_argument("--filename", type=str, default=DEFAULT_OUTFILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-days", type=float, default=365.0)
    parser.add_argument(
        "--no-truth",
        action="store_true",
        default=False,
        help="Omit the truth overlay and show only empirical data curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_debug_plot(
        run_dir=args.run_dir,
        outdir=args.outdir,
        filename=args.filename,
        seed=args.seed,
        duration_days=args.duration_days,
        include_truth=not args.no_truth,
    )


if __name__ == "__main__":
    main()
