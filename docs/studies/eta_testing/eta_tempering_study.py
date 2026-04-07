"""η-tempering investigation for multivariate blocked NUTS.

This study runner covers three questions:

1. Warmup-only tempering:
   warm up at ``warmup_eta < 1`` and sample at ``eta = 1`` using the learned
   step size and mass matrix.
2. Generalised posterior calibration:
   run full inference at ``eta = c / (Nb * Nh)`` over a grid of ``c`` values.
3. Basis dependence:
   repeat the calibration sweep over multiple spline basis sizes.

Usage
-----
    python eta_tempering_study.py phase1 --seeds 0-9
    python eta_tempering_study.py phase2 --seeds 0-9
    python eta_tempering_study.py phase3 --seeds 0-9
    python eta_tempering_study.py phase4
    python eta_tempering_study.py collect
    python eta_tempering_study.py all --seeds 0-4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

set_level("INFO")

HERE = Path(__file__).resolve().parent
OUT = "out_eta"

DEGREE = 2
DIFF_MATRIX_ORDER = 2
DEFAULT_FS = 1.0
DEFAULT_BURN_IN = 512
EPS = 1e-12

# VAR(2) simulation used elsewhere in the repo.
A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [[-0.2, 0.5, 0.0], [0.4, -0.1, 0.0], [0.0, 0.0, -0.1]],
    dtype=np.float64,
)
VAR_COEFFS = np.array([A1, A2], dtype=np.float64)
SIGMA_VAL = 0.25
OFF_DIAG = 0.08
SIGMA = np.array(
    [
        [SIGMA_VAL, 0.0, OFF_DIAG],
        [0.0, SIGMA_VAL, OFF_DIAG],
        [OFF_DIAG, OFF_DIAG, SIGMA_VAL],
    ],
    dtype=np.float64,
)

DEFAULT_N_SAMPLES = 4000
DEFAULT_N_WARMUP = 4000
DEFAULT_NUM_CHAINS = 4
DEFAULT_TARGET_ACCEPT_PROB = 0.95
DEFAULT_MAX_TREE_DEPTH = 14
DEFAULT_VI_STEPS = 100_000
DEFAULT_VI_GUIDE = "lowrank:16"
DEFAULT_VI_LR = 5e-4
DEFAULT_ALPHA_DELTA = 1.0
DEFAULT_BETA_DELTA = 1.0

QUICK_N_SAMPLES = 1000
QUICK_N_WARMUP = 1000
QUICK_NUM_CHAINS = 2
QUICK_VI_STEPS = 20_000

PHASE1_N = 2048
PHASE1_NB = 4
PHASE1_NH = 1
PHASE1_K = 20
PHASE1_WARMUP_ETA_GRID = [0.01, 0.05, 0.1, 0.2]

PHASE2_N = 16384
PHASE2_NB = 4
PHASE2_NH_VALUES = [1, 2, 4]
PHASE2_K = 20
PHASE2_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

PHASE3_N = 16384
PHASE3_NB = 4
PHASE3_NH = 2
PHASE3_C_GRID = PHASE2_C_GRID
PHASE3_N_KNOTS = [12, 20, 32]


@dataclass(frozen=True)
class RunSpec:
    phase: str
    mode: Literal["baseline", "warmup_only", "full"]
    seed: int
    N: int
    Nb: int
    Nh: int
    n_knots: int
    eta: float | str
    warmup_eta: float | str | None = None
    c_value: float | None = None
    label: str = ""

    @property
    def coarse_Nh(self) -> int | None:
        return None if self.Nh <= 1 else int(self.Nh)

    @property
    def n_basis(self) -> int:
        return int(self.n_knots + DEGREE - 1)


def _simulate_var_process(
    n_samples: int,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    seed: int,
    *,
    fs: float = DEFAULT_FS,
    burn_in: int = DEFAULT_BURN_IN,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a VAR process and return ``t: (N,)`` and ``X: (N, C)``."""
    ar_order, n_channels, _ = var_coeffs.shape
    n_total = int(n_samples) + int(burn_in)
    rng = np.random.default_rng(int(seed))
    noise = rng.multivariate_normal(np.zeros(n_channels), sigma, size=n_total)
    x = np.zeros((n_total, n_channels), dtype=np.float64)
    for t_idx in range(ar_order, n_total):
        state = noise[t_idx].copy()
        for lag in range(1, ar_order + 1):
            state = state + var_coeffs[lag - 1] @ x[t_idx - lag]
        x[t_idx] = state
    x = x[burn_in:]
    t = np.arange(x.shape[0], dtype=np.float64) / float(fs)
    return t, x


def _calculate_true_var_psd_hz(
    freqs_hz: np.ndarray,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    *,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """Return the true PSD matrix ``S: (F, C, C)`` on the supplied grid."""
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
        psd[idx] = (2.0 / float(fs)) * (h_f @ sigma @ h_f.conj().T)
    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]
    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    return np.where(np.abs(psd) < EPS, EPS, psd)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _eta_to_float(value: float | str | None) -> float:
    if value is None or isinstance(value, str):
        return float("nan")
    return float(value)


def _channel_attr_values(
    attrs: dict[str, Any],
    prefix: str,
) -> list[float]:
    values = []
    for key, value in attrs.items():
        if str(key).startswith(prefix):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return sorted(values)


def _median_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _extract_width_quantiles(
    data_group,
    variable: str,
    *,
    diagonal_only: bool = False,
    offdiag_only: bool = False,
) -> float:
    if data_group is None or variable not in data_group:
        return float("nan")

    array = np.asarray(data_group[variable].values, dtype=np.float64)
    percentiles = np.asarray(
        data_group[variable].coords.get(
            "percentile", np.arange(array.shape[0], dtype=float)
        ),
        dtype=np.float64,
    )
    if array.shape[0] < 2:
        return float("nan")

    q05_idx = int(np.argmin(np.abs(percentiles - 5.0)))
    q95_idx = int(np.argmin(np.abs(percentiles - 95.0)))
    width = np.maximum(array[q95_idx] - array[q05_idx], 0.0)

    if width.ndim < 3:
        return float(np.median(width))

    p = width.shape[1]
    if diagonal_only:
        idx = np.arange(p)
        return float(np.median(width[:, idx, idx]))
    if offdiag_only:
        mask = ~np.eye(p, dtype=bool)
        return float(np.median(width[:, mask]))
    return float(np.median(width))


def _extract_nuts_diagnostics(idata) -> dict[str, float]:
    """Extract per-draw NUTS diagnostics from ``sample_stats``."""
    metrics: dict[str, float] = {}
    ss = getattr(idata, "sample_stats", None)
    if ss is None:
        return metrics

    divergences = []
    step_sizes = []
    num_steps = []
    accept_probs = []

    for key in ss:
        key_str = str(key)
        values = np.asarray(ss[key].values, dtype=np.float64)
        if "diverging" in key_str:
            divergences.append(float(np.sum(values)))
        if "step_size" in key_str:
            step_sizes.append(float(np.mean(values)))
        if "num_steps" in key_str:
            num_steps.append(float(np.mean(values)))
        if "accept_prob" in key_str:
            accept_probs.append(float(np.mean(values)))

    if divergences:
        metrics["divergences"] = float(np.sum(divergences))
    if step_sizes:
        metrics["mean_step_size"] = float(np.mean(step_sizes))
        metrics["min_step_size"] = float(np.min(step_sizes))
    if num_steps:
        metrics["mean_num_steps"] = float(np.mean(num_steps))
        approx_tree_depth = np.ceil(
            np.log2(np.maximum(np.asarray(num_steps), 1.0))
        )
        metrics["mean_tree_depth"] = float(np.mean(approx_tree_depth))
    if accept_probs:
        metrics["mean_accept_prob"] = float(np.mean(accept_probs))
    return metrics


def _extract_metrics(
    idata,
    *,
    spec: RunSpec,
    wallclock: float,
) -> dict[str, float | int | str]:
    attrs = getattr(idata, "attrs", {})
    ess_raw = np.asarray(attrs.get("ess", np.nan), dtype=np.float64)
    ess_median = float(np.nanmedian(ess_raw)) if ess_raw.size else float("nan")

    sampling_eta_values = _channel_attr_values(attrs, "sampling_eta_channel_")
    warmup_eta_values = _channel_attr_values(attrs, "warmup_eta_channel_")
    warmup_step_sizes = _channel_attr_values(
        attrs, "warmup_step_size_channel_"
    )
    warmup_mass_diag = _channel_attr_values(
        attrs, "warmup_inverse_mass_diag_median_channel_"
    )

    eta_effective = _mean_or_nan(sampling_eta_values)
    warmup_eta_effective = _mean_or_nan(warmup_eta_values)
    nbnh = int(spec.Nb * spec.Nh)

    metrics: dict[str, float | int | str] = {
        "phase": spec.phase,
        "mode": spec.mode,
        "label": spec.label,
        "seed": int(spec.seed),
        "N": int(spec.N),
        "Nb": int(spec.Nb),
        "Nh": int(spec.Nh),
        "NbNh": nbnh,
        "n_knots": int(spec.n_knots),
        "n_basis": int(spec.n_basis),
        "eta_input": (
            str(spec.eta) if isinstance(spec.eta, str) else float(spec.eta)
        ),
        "eta_effective": eta_effective,
        "c_input": (
            float(spec.c_value)
            if spec.c_value is not None
            else _eta_to_float(spec.eta) * nbnh
        ),
        "c_effective": eta_effective * nbnh,
        "warmup_eta_input": (
            ""
            if spec.warmup_eta is None
            else (
                str(spec.warmup_eta)
                if isinstance(spec.warmup_eta, str)
                else float(spec.warmup_eta)
            )
        ),
        "warmup_eta_effective": warmup_eta_effective,
        "tempered_warmup_enabled": int(
            attrs.get("tempered_warmup_enabled", 0)
        ),
        "coverage": float(attrs.get("coverage", np.nan)),
        "coverage_diag": float(attrs.get("coverage_diag", np.nan)),
        "coverage_offdiag_re": float(attrs.get("coverage_offdiag_re", np.nan)),
        "coverage_offdiag_im": float(attrs.get("coverage_offdiag_im", np.nan)),
        "coverage_coherence": float(attrs.get("coverage_coherence", np.nan)),
        "riae_matrix": float(
            attrs.get("riae_matrix", attrs.get("riae", np.nan))
        ),
        "riae_diag_mean": float(attrs.get("riae_diag_mean", np.nan)),
        "riae_offdiag_re": float(attrs.get("riae_offdiag_re", np.nan)),
        "riae_offdiag_im": float(attrs.get("riae_offdiag_im", np.nan)),
        "riae_coherence": float(attrs.get("coherence_riae", np.nan)),
        "ci_width": float(attrs.get("ci_width", np.nan)),
        "ci_width_diag_mean": float(attrs.get("ci_width_diag_mean", np.nan)),
        "ess_median": ess_median,
        "ess_per_second": (
            float(ess_median / wallclock)
            if np.isfinite(ess_median) and wallclock > 0.0
            else float("nan")
        ),
        "wallclock_s": round(wallclock, 3),
        "warmup_step_size_median": _median_or_nan(warmup_step_sizes),
        "warmup_inverse_mass_diag_median": _median_or_nan(warmup_mass_diag),
    }

    posterior_psd = getattr(idata, "posterior_psd", None)
    metrics["ciw_diag_median"] = _extract_width_quantiles(
        posterior_psd,
        "psd_matrix_real",
        diagonal_only=True,
    )
    metrics["ciw_offdiag_re_median"] = _extract_width_quantiles(
        posterior_psd,
        "psd_matrix_real",
        offdiag_only=True,
    )
    metrics["ciw_offdiag_im_median"] = _extract_width_quantiles(
        posterior_psd,
        "psd_matrix_imag",
        offdiag_only=True,
    )
    metrics["ciw_coherence_median"] = _extract_width_quantiles(
        posterior_psd,
        "coherence",
        offdiag_only=True,
    )

    metrics.update(_extract_nuts_diagnostics(idata))
    return metrics


def _run_dir(base_outdir: str, spec: RunSpec) -> Path:
    eta_label = spec.eta if isinstance(spec.eta, str) else f"{spec.eta:.4f}"
    warmup_label = (
        ""
        if spec.warmup_eta is None
        else (
            f"_warmup{spec.warmup_eta}"
            if isinstance(spec.warmup_eta, str)
            else f"_warmup{spec.warmup_eta:.4f}"
        )
    )
    label_suffix = f"_{spec.label}" if spec.label else ""
    run_name = (
        f"{spec.phase}_{spec.mode}{label_suffix}"
        f"_eta{eta_label}{warmup_label}"
        f"_seed{spec.seed}_N{spec.N}_Nb{spec.Nb}_Nh{spec.Nh}_K{spec.n_knots}"
    )
    return HERE / base_outdir / spec.phase / run_name


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=_json_ready)


def run_single(
    spec: RunSpec,
    *,
    outdir: str,
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run one experiment and save ``metrics.json``."""
    run_dir = _run_dir(outdir, spec)
    metrics_path = run_dir / "metrics.json"
    spec_path = run_dir / "run_spec.json"

    if metrics_path.exists() and not force:
        logger.info(f"Loading cached results from {metrics_path}")
        with metrics_path.open() as handle:
            return json.load(handle)

    n_samples = QUICK_N_SAMPLES if quick else DEFAULT_N_SAMPLES
    n_warmup = QUICK_N_WARMUP if quick else DEFAULT_N_WARMUP
    num_chains = QUICK_NUM_CHAINS if quick else DEFAULT_NUM_CHAINS
    vi_steps = QUICK_VI_STEPS if quick else DEFAULT_VI_STEPS

    t, data = _simulate_var_process(spec.N, VAR_COEFFS, SIGMA, spec.seed)
    ts = MultivariateTimeseries(t=t, y=data)
    freq_true_hz = np.fft.rfftfreq(spec.N, d=1.0 / DEFAULT_FS)[1:]
    true_psd = _calculate_true_var_psd_hz(freq_true_hz, VAR_COEFFS, SIGMA)

    coarse_grain_config = None
    if spec.coarse_Nh is not None:
        coarse_grain_config = dict(enabled=True, Nc=None, Nh=spec.coarse_Nh)

    run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(spec_path, asdict(spec))

    logger.info(
        f"Running {spec.phase} seed={spec.seed} eta={spec.eta} "
        f"warmup_eta={spec.warmup_eta} Nb={spec.Nb} Nh={spec.Nh} "
        f"K={spec.n_knots}"
    )

    t0 = time.time()
    idata = run_mcmc(
        data=ts,
        n_knots=spec.n_knots,
        degree=DEGREE,
        diffMatrixOrder=DIFF_MATRIX_ORDER,
        n_samples=n_samples,
        n_warmup=n_warmup,
        num_chains=num_chains,
        rng_key=int(spec.seed),
        outdir=str(run_dir),
        verbose=True,
        target_accept_prob=DEFAULT_TARGET_ACCEPT_PROB,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        init_from_vi=True,
        vi_steps=vi_steps,
        vi_guide=DEFAULT_VI_GUIDE,
        vi_lr=DEFAULT_VI_LR,
        Nb=spec.Nb,
        knot_kwargs=dict(method="density", scoring="spectral"),
        coarse_grain_config=coarse_grain_config,
        alpha_delta=DEFAULT_ALPHA_DELTA,
        beta_delta=DEFAULT_BETA_DELTA,
        compute_coherence_quantiles=True,
        true_psd=(freq_true_hz, true_psd),
        max_save_bytes=20_000_000,
        eta=spec.eta,
        warmup_eta=spec.warmup_eta,
    )
    wallclock = time.time() - t0

    metrics = _extract_metrics(idata, spec=spec, wallclock=wallclock)
    _save_json(metrics_path, metrics)
    logger.info(
        f"[{spec.phase} seed={spec.seed}] "
        f"coverage={float(metrics.get('coverage', np.nan)):.4f} "
        f"ciw_diag={float(metrics.get('ciw_diag_median', np.nan)):.4f} "
        f"div={float(metrics.get('divergences', 0.0)):.0f} "
        f"eta_eff={float(metrics.get('eta_effective', np.nan)):.4f} "
        f"wall={wallclock:.1f}s"
    )
    return metrics


def _eta_from_c(c_value: float, nb: int, nh: int) -> float:
    return min(1.0, float(c_value) / float(nb * nh))


def _phase1_specs(seeds: list[int]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in seeds:
        specs.append(
            RunSpec(
                phase="phase1",
                mode="baseline",
                label="eta1",
                seed=seed,
                N=PHASE1_N,
                Nb=PHASE1_NB,
                Nh=PHASE1_NH,
                n_knots=PHASE1_K,
                eta=1.0,
            )
        )
        for warmup_eta in PHASE1_WARMUP_ETA_GRID:
            specs.append(
                RunSpec(
                    phase="phase1",
                    mode="warmup_only",
                    label=f"warmup_eta_{warmup_eta:.2f}",
                    seed=seed,
                    N=PHASE1_N,
                    Nb=PHASE1_NB,
                    Nh=PHASE1_NH,
                    n_knots=PHASE1_K,
                    eta=1.0,
                    warmup_eta=float(warmup_eta),
                )
            )
    return specs


def _phase2_specs(seeds: list[int]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for nh in PHASE2_NH_VALUES:
        for c_value in PHASE2_C_GRID:
            eta = _eta_from_c(c_value, PHASE2_NB, nh)
            for seed in seeds:
                specs.append(
                    RunSpec(
                        phase="phase2",
                        mode="full",
                        label=f"c_{c_value:g}",
                        seed=seed,
                        N=PHASE2_N,
                        Nb=PHASE2_NB,
                        Nh=nh,
                        n_knots=PHASE2_K,
                        eta=eta,
                        c_value=float(c_value),
                    )
                )
    return specs


def _phase3_specs(seeds: list[int]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for n_knots in PHASE3_N_KNOTS:
        for c_value in PHASE3_C_GRID:
            eta = _eta_from_c(c_value, PHASE3_NB, PHASE3_NH)
            for seed in seeds:
                specs.append(
                    RunSpec(
                        phase="phase3",
                        mode="full",
                        label=f"c_{c_value:g}",
                        seed=seed,
                        N=PHASE3_N,
                        Nb=PHASE3_NB,
                        Nh=PHASE3_NH,
                        n_knots=n_knots,
                        eta=eta,
                        c_value=float(c_value),
                    )
                )
    return specs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        logger.warning(f"No rows to write for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {path}")


def _group_mean(
    rows: list[dict[str, Any]],
    value_key: str,
) -> tuple[float, float, int]:
    values = np.asarray(
        [row.get(value_key, np.nan) for row in rows], dtype=float
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), float(np.std(values)), int(values.size)


def _print_summary_table(rows: list[dict[str, Any]], phase_name: str) -> None:
    phases = sorted({str(row.get("phase", "")) for row in rows})
    include_phase = len(phases) > 1
    include_warmup_eta = any(
        str(row.get("mode", "")) == "warmup_only" for row in rows
    )

    groups: dict[tuple[str, str, str, int, int, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            str(row.get("phase", "?")),
            str(row.get("mode", "?")),
            (
                str(row.get("warmup_eta_input", ""))
                if str(row.get("mode", "")) == "warmup_only"
                else ""
            ),
            int(row.get("Nh", 0)),
            int(row.get("n_basis", 0)),
            str(row.get("c_input", "")),
        )
        groups[key].append(row)

    print(f"\n{'=' * 88}")
    print(phase_name)
    header_parts = []
    if include_phase:
        header_parts.append(f"{'phase':>10s}")
    header_parts.append(f"{'mode':>12s}")
    if include_warmup_eta:
        header_parts.append(f"{'warm_eta':>8s}")
    header_parts.extend(
        [
            f"{'Nh':>4s}",
            f"{'n_basis':>8s}",
            f"{'c_eff':>8s}",
            f"{'n':>4s}",
            f"{'coverage':>12s}",
            f"{'ciw_diag':>10s}",
            f"{'riae':>10s}",
            f"{'div':>8s}",
        ]
    )
    print(" ".join(header_parts))
    for key in sorted(groups):
        phase, mode, warmup_eta_input, nh, n_basis, c_input = key
        subset = groups[key]
        cov_mean, cov_std, n_cov = _group_mean(subset, "coverage")
        ciw_mean, _, _ = _group_mean(subset, "ciw_diag_median")
        riae_mean, _, _ = _group_mean(subset, "riae_matrix")
        div_mean, _, _ = _group_mean(subset, "divergences")
        row_parts = []
        if include_phase:
            row_parts.append(f"{phase:>10s}")
        row_parts.append(f"{mode:>12s}")
        if include_warmup_eta:
            warmup_eta_str = (
                f"{float(warmup_eta_input):>8.3f}"
                if warmup_eta_input not in ("", "nan")
                else f"{'-':>8s}"
            )
            row_parts.append(warmup_eta_str)
        c_input_value = (
            float(c_input) if c_input not in ("", "nan") else float("nan")
        )
        row_parts.extend(
            [
                f"{nh:>4d}",
                f"{n_basis:>8d}",
                f"{c_input_value:>8.3f}",
                f"{n_cov:>4d}",
                f"{cov_mean:>7.4f}±{cov_std:>4.3f}",
                f"{ciw_mean:>10.4f}",
                f"{riae_mean:>10.4f}",
                f"{div_mean:>8.2f}",
            ]
        )
        print(" ".join(row_parts))
    print("=" * 88)


def _run_phase(
    specs: list[RunSpec],
    *,
    outdir: str,
    quick: bool,
    force: bool,
    phase_name: str,
) -> list[dict[str, Any]]:
    rows = [
        run_single(spec, outdir=outdir, quick=quick, force=force)
        for spec in specs
    ]
    phase_dir = HERE / outdir / phase_name
    _write_csv(phase_dir / f"{phase_name}_per_seed.csv", rows)
    _print_summary_table(rows, phase_name)
    return rows


def phase1(
    seeds: list[int],
    *,
    outdir: str = OUT,
    quick: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    return _run_phase(
        _phase1_specs(seeds),
        outdir=outdir,
        quick=quick,
        force=force,
        phase_name="phase1",
    )


def phase2(
    seeds: list[int],
    *,
    outdir: str = OUT,
    quick: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    return _run_phase(
        _phase2_specs(seeds),
        outdir=outdir,
        quick=quick,
        force=force,
        phase_name="phase2",
    )


def phase3(
    seeds: list[int],
    *,
    outdir: str = OUT,
    quick: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    return _run_phase(
        _phase3_specs(seeds),
        outdir=outdir,
        quick=quick,
        force=force,
        phase_name="phase3",
    )


def collect_results(
    *,
    results_dir: str = OUT,
) -> list[dict[str, Any]]:
    base_dir = HERE / results_dir
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(base_dir.rglob("metrics.json")):
        with metrics_path.open() as handle:
            rows.append(json.load(handle))
    if not rows:
        logger.warning(f"No metrics.json files found under {base_dir}")
        return rows
    _write_csv(base_dir / "collected_per_seed.csv", rows)
    _print_summary_table(rows, "collected")
    return rows


def _group_rows(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        grouped[key].append(row)
    return grouped


def _summarise_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    value_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, key_fields)
    summary_rows: list[dict[str, Any]] = []
    for key, subset in grouped.items():
        summary: dict[str, Any] = {
            field: value for field, value in zip(key_fields, key, strict=True)
        }
        summary["n"] = len(subset)
        for value_key in value_keys:
            mean, std, _ = _group_mean(subset, value_key)
            summary[value_key] = mean
            summary[f"{value_key}_std"] = std
        summary_rows.append(summary)
    return sorted(
        summary_rows, key=lambda item: tuple(item.get(k) for k in key_fields)
    )


def _plot_results(
    rows: list[dict[str, Any]],
    *,
    results_dir: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return

    if not rows:
        logger.warning("No rows available for plotting.")
        return

    plot_dir = HERE / results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    phase2_rows = [row for row in rows if row.get("phase") == "phase2"]
    phase1_rows = [row for row in rows if row.get("phase") == "phase1"]
    phase3_rows = [row for row in rows if row.get("phase") == "phase3"]

    if phase2_rows:
        phase2_summary = _summarise_rows(
            phase2_rows,
            key_fields=("Nh", "c_input"),
            value_keys=(
                "eta_effective",
                "coverage",
                "ciw_diag_median",
                "riae_matrix",
                "divergences",
                "mean_num_steps",
                "ess_per_second",
                "wallclock_s",
                "coverage_diag",
                "coverage_offdiag_re",
                "coverage_offdiag_im",
                "coverage_coherence",
                "riae_diag_mean",
                "riae_offdiag_re",
                "riae_offdiag_im",
                "riae_coherence",
            ),
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        for nh in PHASE2_NH_VALUES:
            subset = [row for row in phase2_summary if int(row["Nh"]) == nh]
            ax.plot(
                [row["eta_effective"] for row in subset],
                [row["coverage"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
        ax.axhline(0.9, ls="--", color="grey")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("Coverage")
        ax.set_title("Coverage vs η")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "coverage_vs_eta.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for nh in PHASE2_NH_VALUES:
            subset = [row for row in phase2_summary if int(row["Nh"]) == nh]
            ax.plot(
                [row["eta_effective"] for row in subset],
                [row["ciw_diag_median"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("Median diagonal CI width")
        ax.set_title("CI Width vs η")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "ci_width_vs_eta.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for nh in PHASE2_NH_VALUES:
            subset = [row for row in phase2_summary if int(row["Nh"]) == nh]
            ax.plot(
                [row["eta_effective"] for row in subset],
                [row["riae_matrix"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("RIAE")
        ax.set_title("RIAE vs η")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "riae_vs_eta.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for nh in PHASE2_NH_VALUES:
            subset = [row for row in phase2_summary if int(row["Nh"]) == nh]
            ax.plot(
                [row["coverage"] for row in subset],
                [row["ciw_diag_median"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
        ax.axvline(0.9, ls="--", color="grey")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Median diagonal CI width")
        ax.set_title("Width vs Coverage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "width_vs_coverage.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for nh in PHASE2_NH_VALUES:
            subset = [row for row in phase2_summary if int(row["Nh"]) == nh]
            axes[0].plot(
                [row["eta_effective"] for row in subset],
                [row["mean_num_steps"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
            axes[1].plot(
                [row["eta_effective"] for row in subset],
                [row["ess_per_second"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
            axes[2].plot(
                [row["eta_effective"] for row in subset],
                [row["wallclock_s"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
        axes[0].set_ylabel("Mean num_steps")
        axes[1].set_ylabel("ESS / s")
        axes[2].set_ylabel("Wallclock (s)")
        for ax in axes:
            ax.set_xlabel(r"$\eta$")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "diagnostics_vs_eta.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for nh in PHASE2_NH_VALUES:
            subset = [row for row in phase2_summary if int(row["Nh"]) == nh]
            axes[0].plot(
                [float(row["c_input"]) for row in subset],
                [row["coverage"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
            axes[1].plot(
                [float(row["c_input"]) for row in subset],
                [row["ciw_diag_median"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
            axes[2].plot(
                [float(row["c_input"]) for row in subset],
                [row["riae_matrix"] for row in subset],
                "o-",
                lw=2,
                label=f"Nh={nh}",
            )
        axes[0].axhline(0.9, ls="--", color="grey")
        axes[0].set_ylabel("Coverage")
        axes[1].set_ylabel("Median diagonal CI width")
        axes[2].set_ylabel("RIAE")
        for ax in axes:
            ax.set_xlabel(r"$c = \eta Nb Nh$")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "scaling_collapse.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        component_keys = (
            ("coverage_diag", "Diag PSD"),
            ("coverage_offdiag_re", "Offdiag Re"),
            ("coverage_offdiag_im", "Offdiag Im"),
            ("coverage_coherence", "Coherence"),
        )
        for metric_key, label in component_keys:
            axes[0].plot(
                [row["eta_effective"] for row in phase2_summary],
                [row[metric_key] for row in phase2_summary],
                "o-",
                lw=1.5,
                label=label,
            )
        riae_keys = (
            ("riae_diag_mean", "Diag PSD"),
            ("riae_offdiag_re", "Offdiag Re"),
            ("riae_offdiag_im", "Offdiag Im"),
            ("riae_coherence", "Coherence"),
        )
        for metric_key, label in riae_keys:
            axes[1].plot(
                [row["eta_effective"] for row in phase2_summary],
                [row[metric_key] for row in phase2_summary],
                "o-",
                lw=1.5,
                label=label,
            )
        axes[0].axhline(0.9, ls="--", color="grey")
        axes[0].set_ylabel("Coverage")
        axes[1].set_ylabel("RIAE")
        for ax in axes:
            ax.set_xlabel(r"$\eta$")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "componentwise_metrics_vs_eta.png", dpi=150)
        plt.close(fig)

    if phase1_rows:
        phase1_summary = _summarise_rows(
            phase1_rows,
            key_fields=("mode", "warmup_eta_input"),
            value_keys=(
                "coverage",
                "ciw_diag_median",
                "divergences",
                "mean_num_steps",
                "mean_step_size",
                "warmup_step_size_median",
            ),
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        baseline = [row for row in phase1_summary if row["mode"] == "baseline"]
        tempered = [
            row for row in phase1_summary if row["mode"] == "warmup_only"
        ]
        baseline_cov = baseline[0]["coverage"] if baseline else float("nan")
        baseline_ciw = (
            baseline[0]["ciw_diag_median"] if baseline else float("nan")
        )
        baseline_steps = (
            baseline[0]["mean_num_steps"] if baseline else float("nan")
        )
        xs = [float(row["warmup_eta_input"]) for row in tempered]
        axes[0].plot(xs, [row["coverage"] for row in tempered], "o-", lw=2)
        axes[0].axhline(baseline_cov, ls="--", color="grey", label="baseline")
        axes[0].set_ylabel("Coverage")
        axes[1].plot(
            xs, [row["ciw_diag_median"] for row in tempered], "o-", lw=2
        )
        axes[1].axhline(baseline_ciw, ls="--", color="grey", label="baseline")
        axes[1].set_ylabel("Median diagonal CI width")
        axes[2].plot(
            xs, [row["mean_num_steps"] for row in tempered], "o-", lw=2
        )
        axes[2].axhline(
            baseline_steps, ls="--", color="grey", label="baseline"
        )
        axes[2].set_ylabel("Mean num_steps")
        for ax in axes:
            ax.set_xlabel(r"warmup $\eta$")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "warmup_only_comparison.png", dpi=150)
        plt.close(fig)

    if phase3_rows:
        phase3_summary = _summarise_rows(
            phase3_rows,
            key_fields=("n_basis", "c_input"),
            value_keys=("coverage", "ciw_diag_median", "riae_matrix"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        basis_values = sorted({int(row["n_basis"]) for row in phase3_summary})
        for n_basis in basis_values:
            subset = [
                row for row in phase3_summary if int(row["n_basis"]) == n_basis
            ]
            axes[0].plot(
                [float(row["c_input"]) for row in subset],
                [row["coverage"] for row in subset],
                "o-",
                lw=2,
                label=f"n_basis={n_basis}",
            )
            axes[1].plot(
                [float(row["c_input"]) for row in subset],
                [row["ciw_diag_median"] for row in subset],
                "o-",
                lw=2,
                label=f"n_basis={n_basis}",
            )
            axes[2].plot(
                [float(row["c_input"]) for row in subset],
                [row["riae_matrix"] for row in subset],
                "o-",
                lw=2,
                label=f"n_basis={n_basis}",
            )
        axes[0].axhline(0.9, ls="--", color="grey")
        axes[0].set_ylabel("Coverage")
        axes[1].set_ylabel("Median diagonal CI width")
        axes[2].set_ylabel("RIAE")
        for ax in axes:
            ax.set_xlabel(r"$c = \eta Nb Nh$")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "basis_dependence.png", dpi=150)
        plt.close(fig)

    logger.info(f"Saved plots to {plot_dir}")


def phase4(
    *,
    outdir: str = OUT,
) -> None:
    rows = collect_results(results_dir=outdir)
    _plot_results(rows, results_dir=outdir)


def _parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(item))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="η-tempering study runner for LogPSplinePSD"
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    for name in ("phase1", "phase2", "phase3", "all"):
        subparser = sub.add_parser(name)
        subparser.add_argument("--seeds", type=str, default="0-4")
        subparser.add_argument("--quick", action="store_true")
        subparser.add_argument("--force", action="store_true")
        subparser.add_argument("--outdir", type=str, default=OUT)

    p4 = sub.add_parser("phase4")
    p4.add_argument("--outdir", type=str, default=OUT)

    pc = sub.add_parser("collect")
    pc.add_argument("--results-dir", type=str, default=OUT)

    args = parser.parse_args()

    if args.phase == "phase1":
        phase1(
            _parse_seeds(args.seeds),
            outdir=args.outdir,
            quick=args.quick,
            force=args.force,
        )
        return
    if args.phase == "phase2":
        phase2(
            _parse_seeds(args.seeds),
            outdir=args.outdir,
            quick=args.quick,
            force=args.force,
        )
        return
    if args.phase == "phase3":
        phase3(
            _parse_seeds(args.seeds),
            outdir=args.outdir,
            quick=args.quick,
            force=args.force,
        )
        return
    if args.phase == "phase4":
        phase4(outdir=args.outdir)
        return
    if args.phase == "collect":
        collect_results(results_dir=args.results_dir)
        return

    phase1(
        _parse_seeds(args.seeds),
        outdir=args.outdir,
        quick=args.quick,
        force=args.force,
    )
    phase2(
        _parse_seeds(args.seeds),
        outdir=args.outdir,
        quick=args.quick,
        force=args.force,
    )
    phase3(
        _parse_seeds(args.seeds),
        outdir=args.outdir,
        quick=args.quick,
        force=args.force,
    )
    phase4(outdir=args.outdir)


if __name__ == "__main__":
    main()
