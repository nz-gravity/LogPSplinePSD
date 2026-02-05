"""Grid runner for LISA multivariate sampling diagnostics.

This script runs the LISA XYZ dataset (typically the cached lisatools-synth NPZ)
across a configurable grid of settings to diagnose why multivariate NUTS
sampling is failing (max tree depth hits, tiny step sizes, poor ESS/R-hat, etc.).

The intent is to mirror the VAR3 mode-separation matrix workflow, but on the
actual LISA pipeline.

Key grid knobs (defaults are intentionally small-ish):
  - sampler: multivar_blocked_nuts vs multivar_nuts
  - coarse graining: on/off (linear full-band binning)
  - Nb: Wishart averaging blocks (blocked sampler only)
  - n_knots: spline complexity
  - alpha_delta (= beta_delta): prior strength on P-spline delta hyperparams
  - init mode: no-VI vs VI (diag / lowrank / flow:1)

Outputs are organised as:
  {out}/{seed_root}/sampler_{sampler}/cg_{cg_tag}/B{blocks}/K{knots}/ad{alpha_delta}/init_{init_tag}/...

Each run writes:
  - diagnostics_lisa_matrix.json  (single-run metrics + config)
  - inference_data.nc (optional; controlled by --save-netcdf/--full-outputs)

And the script maintains an aggregate CSV:
  - lisa_multivar_matrix.csv

Notes on runtime
----------------
The raw LISA XYZ time series can be extremely long, which implies a very fine
frequency grid (hundreds of thousands of bins in [fmin, fmax]) when using long
FFT blocks. If you disable coarse graining, make sure you also reduce the
frequency count via either:

- `--max-n-time` (truncate for matrix diagnostics), or
- larger `--time-blocks-grid` (shorter FFT blocks -> coarser freq spacing).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import arviz as az
import numpy as np

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.diagnostics._utils import (
    compute_ci_coverage_multivar,
    compute_matrix_riae,
    extract_percentile,
)
from log_psplines.example_datasets.lisa_data import LISAData
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc


def _float_tag(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if 1e-3 <= abs_value < 1e3:
        return f"{value:g}"
    sci = f"{value:.0e}"  # e.g. 1e-04
    sci = sci.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    return sci


def _sanitize_tag(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
    return "".join(out) if out else "unknown"


def _resolve_init(mode: str) -> tuple[bool, str | None, str]:
    mode = (mode or "").strip()
    key = mode.lower()
    if key in {"none", "novi", "no", "off", "false", "0"}:
        return False, None, "novi"
    if key == "diag":
        return True, "diag", "diag"
    if key == "lowrank":
        return True, "lowrank", "lowrank"
    if key.startswith("lowrank:"):
        return True, mode, _sanitize_tag(key.replace(":", ""))
    if key.startswith("flow"):
        return True, mode, _sanitize_tag(key.replace(":", ""))
    return True, mode, _sanitize_tag(key.replace(":", ""))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _iter_runs(
    *,
    samplers: Sequence[str],
    coarse_grid: Sequence[str],
    alpha_deltas: Sequence[float],
    init_modes: Sequence[str],
    time_blocks: Sequence[int],
    knots: Sequence[int],
    log_bins: Sequence[int],
) -> Iterable[dict[str, object]]:
    for sampler in samplers:
        blocks_grid = (
            [1] if str(sampler) == "multivar_nuts" else list(time_blocks)
        )
        for coarse in coarse_grid:
            coarse_key = str(coarse).lower().strip()
            coarse_on = coarse_key in {"on", "true", "1", "yes"}
            bins_grid = list(log_bins) if coarse_on else [1]
            for Nc in bins_grid:
                for Nb in blocks_grid:
                    for n_knots in knots:
                        for alpha_delta in alpha_deltas:
                            for init_mode in init_modes:
                                yield {
                                    "sampler": str(sampler),
                                    "coarse": str(coarse),
                                    "Nc": int(Nc),
                                    "Nb": int(Nb),
                                    "n_knots": int(n_knots),
                                    "alpha_delta": float(alpha_delta),
                                    "init_mode": str(init_mode),
                                }


def _interp_complex_matrix(
    freq_src: np.ndarray, freq_tgt: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    freq_src = np.asarray(freq_src, dtype=float)
    freq_tgt = np.asarray(freq_tgt, dtype=float)
    mat = np.asarray(matrix)
    flat = mat.reshape(mat.shape[0], -1)

    real_part = np.vstack(
        [
            np.interp(
                freq_tgt,
                freq_src,
                flat[:, idx].real,
                left=flat[0, idx].real,
                right=flat[-1, idx].real,
            )
            for idx in range(flat.shape[1])
        ]
    ).T

    if np.iscomplexobj(mat):
        imag_part = np.vstack(
            [
                np.interp(
                    freq_tgt,
                    freq_src,
                    flat[:, idx].imag,
                    left=flat[0, idx].imag,
                    right=flat[-1, idx].imag,
                )
                for idx in range(flat.shape[1])
            ]
        ).T
        resampled = real_part + 1j * imag_part
    else:
        resampled = real_part
    return resampled.reshape((freq_tgt.size,) + mat.shape[1:])


def _eigen_ratio_summaries(
    periodogram: np.ndarray, *, threshold: float = 0.8
) -> dict[str, float]:
    """Compute r23 = λ3/λ2 summaries for a (F,3,3) Hermitian periodogram."""
    pg = np.asarray(periodogram)
    if pg.ndim != 3 or pg.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected periodogram (F,3,3), got shape {pg.shape}."
        )
    eigvals = np.linalg.eigvalsh(pg)  # ascending: (F, 3)
    denom = np.maximum(eigvals[:, 1], np.asarray(1e-30, dtype=eigvals.dtype))
    r23 = eigvals[:, 0] / denom
    return {
        "r23_p50": float(np.median(r23)),
        "r23_frac_gt": float(np.mean(r23 > float(threshold))),
        "r23_min": float(np.min(r23)),
        "r23_max": float(np.max(r23)),
    }


def _summarize_sampling(
    idata: az.InferenceData, *, max_tree_depth: int
) -> dict[str, float]:
    """Extract sampler diagnostics in a sampler-agnostic way when possible."""
    stats = {}
    ss = idata.sample_stats
    max_steps = (2 ** int(max_tree_depth)) - 1

    # Blocked sampler exposes per-channel fields; coupled uses single fields.
    per_channel_steps = [
        name for name in ss.data_vars if name.startswith("num_steps_channel_")
    ]
    if per_channel_steps:
        for name in sorted(per_channel_steps):
            channel = int(name.split("_")[-1])
            steps = ss[name].values.reshape(-1)
            stats[f"ch{channel}_frac_max_tree_depth"] = float(
                np.mean(steps == max_steps)
            )
            stats[f"ch{channel}_num_steps_median"] = float(np.median(steps))

            step_name = f"step_size_channel_{channel}"
            if step_name in ss:
                step_sizes = ss[step_name].values.reshape(-1)
                stats[f"ch{channel}_step_size_median"] = float(
                    np.median(step_sizes)
                )
                stats[f"ch{channel}_step_size_min"] = float(np.min(step_sizes))

            acc_name = f"accept_prob_channel_{channel}"
            if acc_name in ss:
                acc = ss[acc_name].values.reshape(-1)
                stats[f"ch{channel}_accept_prob_median"] = float(
                    np.median(acc)
                )

            div_name = f"diverging_channel_{channel}"
            if div_name in ss:
                div = ss[div_name].values.reshape(-1)
                stats[f"ch{channel}_divergence_frac"] = float(np.mean(div > 0))

        frac_keys = [k for k in stats if k.endswith("_frac_max_tree_depth")]
        if frac_keys:
            stats["frac_max_tree_depth_max"] = float(
                max(stats[k] for k in frac_keys)
            )
            stats["frac_max_tree_depth_mean"] = float(
                np.mean([stats[k] for k in frac_keys])
            )

        step_keys = [k for k in stats if k.endswith("_step_size_median")]
        if step_keys:
            stats["step_size_median_min"] = float(
                min(stats[k] for k in step_keys)
            )
            stats["step_size_median_median"] = float(
                np.median([stats[k] for k in step_keys])
            )

        div_keys = [k for k in stats if k.endswith("_divergence_frac")]
        if div_keys:
            stats["divergence_frac_max"] = float(
                max(stats[k] for k in div_keys)
            )
            stats["divergence_frac_mean"] = float(
                np.mean([stats[k] for k in div_keys])
            )
        return stats

    # Coupled / single-field fallback
    if "num_steps" in ss:
        steps = ss["num_steps"].values.reshape(-1)
        stats["frac_max_tree_depth"] = float(np.mean(steps == max_steps))
        stats["num_steps_median"] = float(np.median(steps))
    if "step_size" in ss:
        step_sizes = ss["step_size"].values.reshape(-1)
        stats["step_size_median"] = float(np.median(step_sizes))
        stats["step_size_min"] = float(np.min(step_sizes))
    if "accept_prob" in ss:
        acc = ss["accept_prob"].values.reshape(-1)
        stats["accept_prob_median"] = float(np.median(acc))
    if "diverging" in ss:
        div = ss["diverging"].values.reshape(-1)
        stats["divergence_frac"] = float(np.mean(div > 0))
    return stats


def _summarize_rhat_ess(
    idata: az.InferenceData, *, include_weights: bool = True
) -> dict[str, float]:
    posterior = idata.posterior
    var_names: list[str] | None = None
    if not include_weights:
        var_names = [
            v for v in posterior.data_vars if not v.startswith("weights_")
        ]

    try:
        rhat = az.rhat(posterior, var_names=var_names)
        rhat_max = float(np.nanmax(rhat.to_array().values))
    except Exception:
        rhat_max = float("nan")

    try:
        ess = az.ess(posterior, var_names=var_names, method="bulk")
        ess_min = float(np.nanmin(ess.to_array().values))
    except Exception:
        ess_min = float("nan")

    return {"rhat_max": rhat_max, "ess_bulk_min": ess_min}


def _summarize_psd_accuracy(
    idata: az.InferenceData,
    *,
    true_freq: np.ndarray,
    true_matrix: np.ndarray,
) -> dict[str, float]:
    if not hasattr(idata, "posterior_psd") or idata.posterior_psd is None:
        return {}

    psd_ds = idata.posterior_psd
    if "psd_matrix_real" not in psd_ds:
        return {}

    freqs = np.asarray(psd_ds["freq"].values, dtype=float)
    true_interp = _interp_complex_matrix(true_freq, freqs, true_matrix)

    psd_real = np.asarray(psd_ds["psd_matrix_real"].values)
    percentiles = np.asarray(
        psd_ds["psd_matrix_real"].coords.get("percentile", []), dtype=float
    )
    if percentiles.size == 0:
        return {}

    psd_imag = (
        np.asarray(psd_ds["psd_matrix_imag"].values)
        if "psd_matrix_imag" in psd_ds
        else np.zeros_like(psd_real)
    )

    q50 = extract_percentile(
        psd_real, percentiles, 50.0
    ) + 1j * extract_percentile(psd_imag, percentiles, 50.0)
    q05 = extract_percentile(
        psd_real, percentiles, 5.0
    ) + 1j * extract_percentile(psd_imag, percentiles, 5.0)
    q95 = extract_percentile(
        psd_real, percentiles, 95.0
    ) + 1j * extract_percentile(psd_imag, percentiles, 95.0)

    riae = compute_matrix_riae(q50, true_interp, freqs)
    coverage = compute_ci_coverage_multivar(
        np.stack([q05, q50, q95], axis=0),
        true_interp,
    )
    return {"psd_riae_matrix": float(riae), "psd_coverage_90": float(coverage)}


def _load_lisa_inputs(
    *,
    npz_path: Path | None,
    max_n_time: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, y, true_freq, true_matrix)."""
    if npz_path is not None and npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as synth:
            t_full = synth["time"]
            y_full = synth["data"]
            true_matrix = synth["true_matrix"]
            true_freq = synth["freq_true"]
        logger.info(f"Loaded lisatools synth NPZ: {npz_path}")
    else:
        lisa_data = LISAData.load()
        t_full = lisa_data.time
        y_full = lisa_data.data
        true_freq = lisa_data.freq
        true_matrix = lisa_data.true_matrix
        logger.info("Loaded LISA dataset via LISAData.load().")

    if (
        max_n_time is not None
        and int(max_n_time) > 0
        and y_full.shape[0] > int(max_n_time)
    ):
        max_n = int(max_n_time)
        t_full = t_full[:max_n]
        y_full = y_full[:max_n]
        logger.info(f"Truncated to n={max_n} samples for the matrix run.")

    return (
        np.asarray(t_full, dtype=float),
        np.asarray(y_full, dtype=float),
        np.asarray(true_freq, dtype=float),
        np.asarray(true_matrix),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a diagnostic grid for LISA multivariate PSD inference."
    )
    parser.add_argument("--out", type=str, default="out_lisa_multivar_matrix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--chain-method",
        type=str,
        default=None,
        choices=["parallel", "vectorized", "sequential"],
        help="Override NumPyro chain execution method. Default uses LogPSpline auto-selection.",
    )
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument(
        "--dense-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a dense mass matrix in NUTS (recommended).",
    )
    parser.add_argument("--vi-steps", type=int, default=8000)
    parser.add_argument("--vi-lr", type=float, default=1e-2)
    parser.add_argument("--vi-posterior-draws", type=int, default=256)
    parser.add_argument(
        "--compute-rhat-ess", action="store_true", default=True
    )
    parser.add_argument(
        "--skip-rhat-ess",
        action="store_false",
        dest="compute_rhat_ess",
        help="Skip ArviZ R-hat/ESS computation (faster).",
    )
    parser.add_argument("--save-netcdf", action="store_true", default=False)
    parser.add_argument(
        "--full-outputs",
        action="store_true",
        help="Enable netcdf saving + heavier outputs (slower, larger).",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Optional path to lisatools_synth_data.npz. When unset, tries the default study location.",
    )
    parser.add_argument(
        "--max-n-time",
        type=int,
        default=8_388_608,
        help="Optional truncation of the time series to the first N samples. Use 0 to disable truncation.",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["multivar_blocked_nuts"],
    )
    parser.add_argument(
        "--coarse-grid",
        nargs="+",
        default=["on", "off"],
        help="Coarse graining: on|off.",
    )
    parser.add_argument(
        "--log-bins-grid",
        nargs="+",
        type=int,
        default=[512],
        help="Linear coarse-bin counts (full-band), used when coarse graining is enabled.",
    )
    parser.add_argument(
        "--alpha-delta-grid",
        nargs="+",
        type=float,
        default=[3.0],
        help="Grid for alpha_delta; beta_delta is set equal to alpha_delta.",
    )
    parser.add_argument(
        "--init-grid",
        nargs="+",
        default=["novi"],
        help="Init modes: novi | diag | lowrank | flow:1 (or any vi_guide string).",
    )
    parser.add_argument(
        "--time-blocks-grid",
        nargs="+",
        type=int,
        default=[384, 768, 1024],
        help="Nb values (Wishart blocks). Must be >= p. Prefer larger values when coarse=off.",
    )
    parser.add_argument("--knots-grid", nargs="+", type=int, default=[15, 30])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if diagnostics JSON exists (overwrites outputs).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect existing JSON diagnostics into the aggregate CSV.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true", default=False)
    args = parser.parse_args()

    set_level("INFO")

    here = Path(__file__).resolve().parent
    FMIN = 10**-4
    FMAX = 10**-1

    default_npz = here / "results" / "lisa" / "lisatools_synth_data.npz"
    npz_path = Path(args.npz).expanduser() if args.npz else default_npz
    if not npz_path.exists():
        npz_path = None

    samplers = [str(s) for s in args.samplers]
    coarse_grid = [str(x) for x in args.coarse_grid]
    alpha_deltas = [float(x) for x in args.alpha_delta_grid]
    init_modes = [str(x) for x in args.init_grid]
    time_blocks = [int(x) for x in args.time_blocks_grid]
    knots = [int(x) for x in args.knots_grid]
    log_bins = [int(x) for x in args.log_bins_grid]

    for Nb in time_blocks:
        if Nb < 3:
            raise ValueError("Nb must be >= 3 (p=3).")

    n_tag = (
        f"Nmax{int(args.max_n_time)}"
        if args.max_n_time is not None and int(args.max_n_time) > 0
        else "Nfull"
    )
    root_out = (
        here
        / str(args.out)
        / (
            f"seed_{int(args.seed)}_{n_tag}_S{int(args.samples)}_W{int(args.warmup)}_C{int(args.chains)}"
        )
    )
    root_out.mkdir(parents=True, exist_ok=True)

    all_runs = list(
        _iter_runs(
            samplers=samplers,
            coarse_grid=coarse_grid,
            alpha_deltas=alpha_deltas,
            init_modes=init_modes,
            time_blocks=time_blocks,
            knots=knots,
            log_bins=log_bins,
        )
    )
    if args.start < 0 or args.start > len(all_runs):
        raise ValueError(
            f"--start must be in [0, {len(all_runs)}], got {args.start}."
        )
    end = (
        len(all_runs)
        if args.limit is None
        else min(len(all_runs), args.start + int(args.limit))
    )
    runs = all_runs[int(args.start) : end]

    aggregate_path = root_out / "lisa_multivar_matrix.csv"

    if args.dry_run:
        print(f"Planned runs: {len(runs)} (of total {len(all_runs)})")
        for spec in runs[:30]:
            init_from_vi, vi_guide, init_tag = _resolve_init(
                str(spec["init_mode"])
            )
            ad_tag = _float_tag(float(spec["alpha_delta"]))
            cg = str(spec["coarse"]).lower().strip()
            cg_tag = "on" if cg in {"on", "true", "1", "yes"} else "off"
            bins_tag = f"bins{int(spec['Nc'])}" if cg_tag == "on" else "raw"
            sampler_tag = _sanitize_tag(str(spec["sampler"]))
            run_dir = (
                root_out
                / f"sampler_{sampler_tag}"
                / f"cg_{cg_tag}_{bins_tag}"
                / f"B{int(spec['Nb'])}"
                / f"K{int(spec['n_knots'])}"
                / f"ad{ad_tag}"
                / f"init_{init_tag}"
            )
            print(
                f"- {run_dir} (init_from_vi={init_from_vi}, vi_guide={vi_guide})"
            )
        return

    if args.collect_only:
        rows: list[dict[str, object]] = []
        for spec in runs:
            init_from_vi, _, init_tag = _resolve_init(str(spec["init_mode"]))
            _ = init_from_vi  # naming parity with dry-run
            ad_tag = _float_tag(float(spec["alpha_delta"]))
            cg = str(spec["coarse"]).lower().strip()
            cg_tag = "on" if cg in {"on", "true", "1", "yes"} else "off"
            Nc = int(spec["Nc"])
            bins_tag = f"bins{Nc}" if cg_tag == "on" else "raw"
            sampler_tag = _sanitize_tag(str(spec["sampler"]))
            Nb = int(spec["Nb"])
            n_knots = int(spec["n_knots"])
            run_dir = (
                root_out
                / f"sampler_{sampler_tag}"
                / f"cg_{cg_tag}_{bins_tag}"
                / f"B{Nb}"
                / f"K{n_knots}"
                / f"ad{ad_tag}"
                / f"init_{init_tag}"
            )
            diag_path = run_dir / "diagnostics_lisa_matrix.json"
            if diag_path.exists():
                rows.append(json.loads(diag_path.read_text()))
        _write_csv(aggregate_path, rows)
        logger.info(f"Wrote aggregate CSV to {aggregate_path}")
        return

    t_full, y_full, true_freq, true_matrix = _load_lisa_inputs(
        npz_path=npz_path, max_n_time=args.max_n_time
    )
    dt = float(t_full[1] - t_full[0])
    fs = 1.0 / dt

    # Keep a common length divisible by all requested time-block values so each
    # blocked-sampler run sees the same data length.
    uses_blocks = any(s == "multivar_blocked_nuts" for s in samplers)
    block_lcm = int(math.lcm(*time_blocks)) if uses_blocks else 1
    n = int(y_full.shape[0])
    n_used = n - (n % block_lcm)
    if n_used <= 0:
        raise ValueError("Time series too short after divisibility trimming.")
    if n_used != n:
        logger.info(
            f"Trimming {n - n_used} samples to make n divisible by lcm(time_blocks)={block_lcm}."
        )
        y_full = y_full[:n_used]
        t_full = t_full[:n_used]

    raw_series = MultivariateTimeseries(y=y_full, t=t_full)
    standardized_ts = raw_series.standardise_for_psd()

    # Cache expensive FFT preprocessing: one MultivarFFT per Wishart block count
    # and/or one full-periodogram FFT for multivar_nuts.
    wishart_cache = {}
    if uses_blocks:
        for Nb in sorted(set(time_blocks)):
            wishart_cache[int(Nb)] = standardized_ts.to_wishart_stats(
                Nb=int(Nb), fmin=FMIN, fmax=FMAX
            )

    csd_full = None
    if any(s == "multivar_nuts" for s in samplers):
        csd_full = standardized_ts.to_cross_spectral_density(
            fmin=FMIN, fmax=FMAX
        )

    coarse_keys = {str(x).lower().strip() for x in coarse_grid}
    coarse_off_requested = any(
        x in {"off", "false", "0", "no"} for x in coarse_keys
    )
    if coarse_off_requested:
        p = int(raw_series.p)
        if wishart_cache:
            for Nb, fft in sorted(wishart_cache.items()):
                n_total = int(fft.N) * p
                if n_total > 100_000:
                    logger.warning(
                        "coarse=off with Nb={} implies N={} -> N≈{} basis rows; expect very slow NUTS. "
                        "Consider larger Nb (e.g. 384/768/1024) or --max-n-time.",
                        int(Nb),
                        int(fft.N),
                        int(n_total),
                    )
        if csd_full is not None:
            n_total = int(csd_full.N) * p
            if n_total > 100_000:
                logger.warning(
                    "coarse=off with multivar_nuts implies N={} -> N≈{} basis rows; expect very slow NUTS. "
                    "Consider --max-n-time or enabling coarse graining.",
                    int(csd_full.N),
                    int(n_total),
                )

    (root_out / "matrix_config.json").write_text(
        json.dumps(
            dict(
                base=dict(
                    seed=int(args.seed),
                    n_samples=int(args.samples),
                    n_warmup=int(args.warmup),
                    num_chains=int(args.chains),
                    target_accept_prob=float(args.target_accept),
                    max_tree_depth=int(args.max_tree_depth),
                    dense_mass=bool(args.dense_mass),
                    vi_steps=int(args.vi_steps),
                    vi_lr=float(args.vi_lr),
                    vi_posterior_draws=int(args.vi_posterior_draws),
                    fmin=float(FMIN),
                    fmax=float(FMAX),
                    fs=float(fs),
                    dt=float(dt),
                    n=int(y_full.shape[0]),
                    data_source=(
                        str(npz_path)
                        if npz_path is not None
                        else "LISAData.load()"
                    ),
                    save_netcdf=bool(args.save_netcdf or args.full_outputs),
                    compute_rhat_ess=bool(args.compute_rhat_ess),
                ),
                grid=dict(
                    samplers=samplers,
                    coarse=coarse_grid,
                    log_bins=log_bins,
                    alpha_delta=alpha_deltas,
                    init_modes=init_modes,
                    Nb=time_blocks,
                    n_knots=knots,
                ),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    rows: list[dict[str, object]] = []
    for spec in runs:
        init_from_vi, vi_guide, init_tag = _resolve_init(
            str(spec["init_mode"])
        )
        alpha_delta = float(spec["alpha_delta"])
        beta_delta = float(spec["alpha_delta"])
        Nb = int(spec["Nb"])
        n_knots = int(spec["n_knots"])
        sampler = str(spec["sampler"])

        cg = str(spec["coarse"]).lower().strip()
        cg_on = cg in {"on", "true", "1", "yes"}
        cg_tag = "on" if cg_on else "off"
        Nc = int(spec["Nc"])
        bins_tag = f"bins{Nc}" if cg_on else "raw"

        sampler_tag = _sanitize_tag(sampler)
        ad_tag = _float_tag(alpha_delta)

        run_dir = (
            root_out
            / f"sampler_{sampler_tag}"
            / f"cg_{cg_tag}_{bins_tag}"
            / f"B{Nb}"
            / f"K{n_knots}"
            / f"ad{ad_tag}"
            / f"init_{init_tag}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        diag_path = run_dir / "diagnostics_lisa_matrix.json"
        idata_path = run_dir / "inference_data.nc"

        if diag_path.exists() and not args.overwrite:
            rows.append(json.loads(diag_path.read_text()))
            continue

        if sampler == "multivar_nuts":
            if csd_full is None:
                raise RuntimeError(
                    "Internal error: csd_full cache missing for multivar_nuts."
                )
            data_for_run = csd_full
        else:
            data_for_run = wishart_cache[Nb]

        coarse_cfg = CoarseGrainConfig(
            enabled=bool(cg_on),
            Nc=int(Nc),
            f_min=FMIN,
            f_max=FMAX,
        )

        run_config = dict(
            sampler=sampler,
            coarse_grain=bool(cg_on),
            Nc=int(Nc),
            Nb=int(Nb),
            n_knots=int(n_knots),
            alpha_delta=float(alpha_delta),
            beta_delta=float(beta_delta),
            init_from_vi=bool(init_from_vi),
            vi_guide=vi_guide,
            seed=int(args.seed),
            n_samples=int(args.samples),
            n_warmup=int(args.warmup),
            num_chains=int(args.chains),
            chain_method=args.chain_method,
            target_accept_prob=float(args.target_accept),
            max_tree_depth=int(args.max_tree_depth),
            dense_mass=bool(args.dense_mass),
            vi_steps=int(args.vi_steps),
            vi_lr=float(args.vi_lr),
            vi_posterior_draws=int(args.vi_posterior_draws),
            fmin=float(FMIN),
            fmax=float(FMAX),
        )

        try:
            start = time.perf_counter()
            idata = run_mcmc(
                data=data_for_run,
                sampler=sampler,
                n_samples=int(args.samples),
                n_warmup=int(args.warmup),
                num_chains=int(args.chains),
                chain_method=args.chain_method,
                n_knots=n_knots,
                degree=2,
                diffMatrixOrder=2,
                knot_kwargs=dict(strategy="log"),
                outdir=str(run_dir),
                verbose=True,
                compute_psis=False,
                skip_plot_diagnostics=True,
                diagnostics_summary_mode="off",
                diagnostics_summary_position="end",
                coarse_grain_config=coarse_cfg,
                Nb=Nb,
                fmin=FMIN,
                fmax=FMAX,
                alpha_delta=alpha_delta,
                beta_delta=beta_delta,
                init_from_vi=init_from_vi,
                vi_steps=int(args.vi_steps),
                vi_lr=float(args.vi_lr),
                vi_guide=vi_guide,
                vi_posterior_draws=int(args.vi_posterior_draws),
                vi_progress_bar=True,
                target_accept_prob=float(args.target_accept),
                max_tree_depth=int(args.max_tree_depth),
                dense_mass=bool(args.dense_mass),
                true_psd=(true_freq, true_matrix),
            )
            runtime = time.perf_counter() - start

            if args.save_netcdf or args.full_outputs:
                idata.to_netcdf(str(idata_path))

            sampling = _summarize_sampling(
                idata, max_tree_depth=int(args.max_tree_depth)
            )
            rhat_ess = (
                _summarize_rhat_ess(idata) if args.compute_rhat_ess else {}
            )
            eig = _eigen_ratio_summaries(
                np.asarray(idata.observed_data["periodogram"].values)
            )
            psd_acc = _summarize_psd_accuracy(
                idata, true_freq=true_freq, true_matrix=true_matrix
            )

            diag = dict(
                **run_config,
                runtime_seconds=float(runtime),
                output_dir=str(run_dir),
                success=True,
                **sampling,
                **rhat_ess,
                **eig,
                **psd_acc,
            )
        except Exception as exc:
            diag = dict(
                **run_config,
                runtime_seconds=float("nan"),
                output_dir=str(run_dir),
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            if args.fail_fast:
                raise

        diag_path.write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n")
        rows.append(diag)
        _write_csv(aggregate_path, rows)
    logger.info(f"Wrote aggregate CSV to {aggregate_path}")


if __name__ == "__main__":
    main()
