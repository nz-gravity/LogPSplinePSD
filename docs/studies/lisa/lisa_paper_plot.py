from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
from scipy.ndimage import uniform_filter1d

from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.datatypes.multivar_utils import interp_matrix
from log_psplines.logger import logger
from log_psplines.plotting import psd_matrix as psd_matrix_mod
from log_psplines.plotting.psd_matrix import PSDMatrixPlotSpec, plot_psd_matrix

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "lisa"
DEFAULT_IDATA_PATH = RESULTS_DIR / "full" / "inference_data.nc"
DEFAULT_TRUTH_CANDIDATES = [
    RESULTS_DIR / "lisatools_synth_data.npz",
    RESULTS_DIR / "lisa_data.npz",
]
DEFAULT_OUTFILE = "psd_matrix_nc_empirical_paper.png"


def _resolve_truth_npz(truth_npz: Path | None) -> Path:
    if truth_npz is not None:
        return truth_npz

    for candidate in DEFAULT_TRUTH_CANDIDATES:
        if candidate.exists():
            logger.info(f"Using truth NPZ: {candidate}")
            return candidate

    candidate_text = ", ".join(str(path) for path in DEFAULT_TRUTH_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find a truth NPZ automatically. Checked: {candidate_text}"
    )


def _smooth_empirical(emp: EmpiricalPSD, window: int) -> EmpiricalPSD:
    """Apply a running average to empirical PSD (log-domain for diag, linear elsewhere)."""
    if window <= 1:
        return emp

    p = emp.psd.shape[1]
    psd_smooth = emp.psd.copy()

    for i in range(p):
        for j in range(p):
            raw = emp.psd[:, i, j]
            if i == j:
                # Smooth in log domain then back-transform to preserve positivity.
                log_raw = np.log(np.maximum(raw.real, 1e-300))
                log_sm = uniform_filter1d(log_raw, size=window, mode="nearest")
                psd_smooth[:, i, j] = np.exp(log_sm)
            else:
                psd_smooth[:, i, j] = uniform_filter1d(
                    raw.real, size=window, mode="nearest"
                ) + 1j * uniform_filter1d(
                    raw.imag, size=window, mode="nearest"
                )

    coh_smooth = emp.coherence.copy()
    for i in range(p):
        for j in range(p):
            if i != j:
                coh_smooth[:, i, j] = uniform_filter1d(
                    emp.coherence[:, i, j], size=window, mode="nearest"
                )

    return EmpiricalPSD(
        freq=emp.freq,
        psd=psd_smooth,
        coherence=coh_smooth,
        channels=emp.channels,
    )


def _load_empirical_from_idata(idata: az.InferenceData) -> EmpiricalPSD:
    if "observed_data" not in idata:
        raise ValueError("InferenceData missing observed_data group.")
    obs = idata["observed_data"]
    if "periodogram" not in obs:
        raise ValueError("observed_data missing periodogram variable.")

    periodogram = obs["periodogram"]
    freq = np.asarray(periodogram.coords["freq"].values, dtype=float)
    channels = np.asarray(periodogram.coords["channels"].values)
    psd = np.asarray(periodogram.values)

    power = np.abs(psd) ** 2
    diag = np.abs(np.diagonal(psd, axis1=1, axis2=2))
    denom = diag[:, :, None] * diag[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(denom > 0, power.real / denom, np.nan)
    for ch in range(psd.shape[1]):
        coherence[:, ch, ch] = 1.0

    return EmpiricalPSD(
        freq=freq, psd=psd, coherence=coherence, channels=channels
    )


def _load_truth_on_grid(
    truth_npz: Path, freq_target: np.ndarray
) -> np.ndarray:
    with np.load(truth_npz, allow_pickle=True) as synth:
        if "freq_true" not in synth.files:
            raise ValueError(f"truth NPZ missing freq_true: {truth_npz}")
        freq_true = np.asarray(synth["freq_true"], dtype=float)

        truth = None
        if "true_matrix_freq" in synth.files:
            true_matrix_freq = np.asarray(synth["true_matrix_freq"])
            if true_matrix_freq.dtype != object and true_matrix_freq.ndim >= 3:
                truth = true_matrix_freq
                logger.info(
                    "Using true_matrix_freq from NPZ for truth overlay."
                )

        if truth is None and "true_matrix" in synth.files:
            truth = np.asarray(synth["true_matrix"])
            logger.warning(
                "Using true_matrix from NPZ for truth overlay (true_matrix_freq unavailable)."
            )

        if truth is None:
            raise ValueError(
                f"truth NPZ missing usable true_matrix entries: {truth_npz}"
            )

    if truth.shape[0] != freq_true.shape[0]:
        raise ValueError(
            "Truth matrix frequency dimension does not match freq_true."
        )

    return interp_matrix(
        freq_true, truth, np.asarray(freq_target, dtype=float)
    )


def build_plot(
    idata_path: Path,
    truth_npz: Path | None,
    outdir: Path,
    filename: str,
    xscale: str,
    fmin: float | None,
    fmax: float | None,
    smooth_window: int = 30,
) -> Path:
    if not idata_path.exists():
        raise FileNotFoundError(f"InferenceData file not found: {idata_path}")

    truth_path = _resolve_truth_npz(truth_npz)
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth NPZ file not found: {truth_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(str(idata_path))
    empirical = _load_empirical_from_idata(idata)

    if "posterior_psd" in idata and "freq" in idata["posterior_psd"].coords:
        freq_target = np.asarray(
            idata["posterior_psd"].coords["freq"].values, dtype=float
        )
    else:
        freq_target = empirical.freq

    truth_interp = _load_truth_on_grid(truth_path, freq_target)

    # Smooth the Wishart single-sample empirical so it looks like the reference.
    if smooth_window > 1:
        empirical = _smooth_empirical(empirical, smooth_window)
        logger.info(
            f"Smoothed empirical with running window size {smooth_window}."
        )

    # Match paper-style visibility for empirical/truth overlays.
    psd_matrix_mod.EMPIRICAL_KWGS.update(
        {
            "color": "0.45",
            "lw": 1.3,
            "alpha": 0.8,
            "ls": "-",
            "label": "Empirical",
            "zorder": -5,
        }
    )
    psd_matrix_mod.TRUE_KWGS.update(
        {"color": "k", "lw": 1.3, "label": "Truth", "zorder": -2}
    )

    # Auto-detect freq_range from the actual data if not explicitly provided.
    if fmin is None:
        fmin = float(freq_target.min())
    if fmax is None:
        fmax = float(freq_target.max())
    freq_range = (fmin, fmax)

    plot_psd_matrix(
        PSDMatrixPlotSpec(
            idata=idata,
            empirical_psd=empirical,
            true_psd=truth_interp,
            outdir=str(outdir),
            filename=filename,
            diag_yscale="log",
            offdiag_yscale="linear",
            xscale=xscale,
            show_coherence=True,
            show_csd_magnitude=False,
            overlay_vi=True,
            show_knots=False,
            freq_range=freq_range,
        )
    )

    out_path = outdir / filename
    if not out_path.exists():
        raise RuntimeError(
            f"Plotting finished but output file not found: {out_path}"
        )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LISA multivariate paper plot from inference_data.nc with "
            "empirical overlay, truth, and no knot markers."
        )
    )
    parser.add_argument("--idata", type=Path, default=DEFAULT_IDATA_PATH)
    parser.add_argument(
        "--truth-npz",
        type=Path,
        default=None,
        help=(
            "Optional explicit truth NPZ path. If omitted, script tries: "
            + ", ".join(str(path) for path in DEFAULT_TRUTH_CANDIDATES)
        ),
    )
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_IDATA_PATH.parent
    )
    parser.add_argument("--filename", type=str, default=DEFAULT_OUTFILE)
    parser.add_argument(
        "--xscale", type=str, default="log", choices=["linear", "log"]
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=None,
        help="Minimum frequency for x-axis. Defaults to minimum in posterior data.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximum frequency for x-axis. Defaults to maximum in posterior data.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=30,
        help="Running-average window (in frequency bins) applied to the Wishart empirical data. 1 = no smoothing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = build_plot(
        idata_path=args.idata,
        truth_npz=args.truth_npz,
        outdir=args.outdir,
        filename=args.filename,
        xscale=args.xscale,
        fmin=args.fmin,
        fmax=args.fmax,
        smooth_window=args.smooth_window,
    )
    logger.info(f"Saved paper plot to {out_path}")


if __name__ == "__main__":
    main()
