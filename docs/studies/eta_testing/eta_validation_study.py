"""η-tempering validation study: robustness across K, element types, and LISA data.

Tests whether η = c / (Nb × Nh) with c=2 generalises beyond the original
VAR(2) p=3 configuration.

Tests
-----
test3 : K-sweep at fixed Nb×Nh
    Varies K ∈ {10, 15, 20, 30, 40} with fixed Nb=4, Nh=2 and eta="auto"
    (c=2).  If n_basis truly doesn't matter, coverage should be ~0.90
    regardless of K.

test4 : Analytic eta estimate
    Computes η from the basis/penalty matrices:
        η_analytic = tr(P) / (Nb × Nh × tr(B'B))
    averaged across Cholesky components.  Compares the analytic η to the
    empirical c=2 formula and runs inference at both to check whether the
    analytic version produces comparable or better coverage.

test5 : Coverage by element type
    Runs with eta="auto" (c=2) and reports coverage broken down by:
    diagonal (auto-spectra), off-diagonal real (cross-spectra Re),
    off-diagonal imaginary (cross-spectra Im), and coherence.
    Uses multiple Nb×Nh configurations to check all element types
    benefit uniformly from tempering.

test6 : LISA sanity check
    Runs on LISA XYZ noise (7-day segment, short for tractability) with the
    analytic LISA noise PSD as truth.  Checks that eta="auto" produces
    reasonable coverage on real (non-VAR) coloured noise.

Usage
-----
    python eta_validation_study.py test3 --seeds 0-4
    python eta_validation_study.py test4 --seeds 0-4
    python eta_validation_study.py test5 --seeds 0-4
    python eta_validation_study.py test6 --seeds 0-2
    python eta_validation_study.py all --seeds 0-4
    python eta_validation_study.py plots
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
OUT = "out_eta_validation"

# ── Shared constants ─────────────────────────────────────────────────────────

DEGREE = 2
DIFF_MATRIX_ORDER = 2
DEFAULT_FS = 1.0
DEFAULT_BURN_IN = 512
EPS = 1e-12

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

# VAR(2) simulation (same as eta_tempering_study.py).
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

# ── Test 3: K-sweep ──────────────────────────────────────────────────────────

TEST3_N = 16384
TEST3_NB = 4
TEST3_NH = 2
TEST3_K_VALUES = [10, 15, 20, 30, 40]

# ── Test 4: analytic eta ─────────────────────────────────────────────────────

TEST4_N = 16384
TEST4_NB = 4
TEST4_NH = 2
TEST4_K = 20

# ── Test 5: element-type coverage ────────────────────────────────────────────

TEST5_CONFIGS = [
    # (N, Nb, Nh, K)
    (2048, 4, 1, 20),
    (16384, 4, 2, 20),
    (16384, 4, 4, 20),
]

# ── Test 6: LISA ─────────────────────────────────────────────────────────────

LISA_DURATION_DAYS = 7
LISA_BLOCK_DAYS = 1  # Nb = 7
LISA_K = 50  # need many knots to track LISA spectral shape
LISA_COARSE_NH = 4  # coarse-grain by averaging Nh=4 adjacent bins
LISA_NULL_EXCISION = True  # remove bins near TDI transfer function nulls
LISA_FMIN = 1e-4
LISA_FMAX = 1e-1

# ── Data generation ──────────────────────────────────────────────────────────


def _simulate_var_process(
    n_samples: int,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    seed: int,
    *,
    fs: float = DEFAULT_FS,
    burn_in: int = DEFAULT_BURN_IN,
) -> tuple[np.ndarray, np.ndarray]:
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


# ── Spec dataclass ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunSpec:
    test: str
    seed: int
    N: int
    Nb: int
    Nh: int
    n_knots: int
    eta: float | str
    eta_c: float = 2.0
    label: str = ""
    data_source: str = "var2"

    @property
    def coarse_Nh(self) -> int | None:
        return None if self.Nh <= 1 else int(self.Nh)

    @property
    def n_basis(self) -> int:
        return int(self.n_knots + DEGREE - 1)


# ── Metrics extraction ───────────────────────────────────────────────────────


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _channel_attr_values(attrs: dict[str, Any], prefix: str) -> list[float]:
    values = []
    for key, value in attrs.items():
        if str(key).startswith(prefix):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return sorted(values)


def _mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _extract_nuts_diagnostics(idata) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ss = getattr(idata, "sample_stats", None)
    if ss is None:
        return metrics
    divergences, step_sizes, num_steps, accept_probs = [], [], [], []
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
    if num_steps:
        metrics["mean_num_steps"] = float(np.mean(num_steps))
    if accept_probs:
        metrics["mean_accept_prob"] = float(np.mean(accept_probs))
    return metrics


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
    eta_effective = _mean_or_nan(sampling_eta_values)
    nbnh = int(spec.Nb * spec.Nh)

    metrics: dict[str, float | int | str] = {
        "test": spec.test,
        "label": spec.label,
        "seed": int(spec.seed),
        "data_source": spec.data_source,
        "N": int(spec.N),
        "Nb": int(spec.Nb),
        "Nh": int(spec.Nh),
        "NbNh": nbnh,
        "n_knots": int(spec.n_knots),
        "n_basis": int(spec.n_basis),
        "eta_input": (
            str(spec.eta) if isinstance(spec.eta, str) else float(spec.eta)
        ),
        "eta_c": float(spec.eta_c),
        "eta_effective": eta_effective,
        "c_effective": eta_effective * nbnh,
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
        "ess_median": ess_median,
        "wallclock_s": round(wallclock, 3),
    }

    posterior_psd = getattr(idata, "posterior_psd", None)
    metrics["ciw_diag_median"] = _extract_width_quantiles(
        posterior_psd, "psd_matrix_real", diagonal_only=True
    )
    metrics["ciw_offdiag_re_median"] = _extract_width_quantiles(
        posterior_psd, "psd_matrix_real", offdiag_only=True
    )
    metrics["ciw_offdiag_im_median"] = _extract_width_quantiles(
        posterior_psd, "psd_matrix_imag", offdiag_only=True
    )
    metrics.update(_extract_nuts_diagnostics(idata))
    return metrics


# ── Run helpers ──────────────────────────────────────────────────────────────


def _run_dir(base_outdir: str, spec: RunSpec) -> Path:
    eta_label = spec.eta if isinstance(spec.eta, str) else f"{spec.eta:.4f}"
    label_suffix = f"_{spec.label}" if spec.label else ""
    run_name = (
        f"{spec.test}{label_suffix}"
        f"_eta{eta_label}_c{spec.eta_c:g}"
        f"_seed{spec.seed}_N{spec.N}_Nb{spec.Nb}_Nh{spec.Nh}_K{spec.n_knots}"
    )
    return HERE / base_outdir / spec.test / run_name


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=_json_ready)


def _run_var_single(
    spec: RunSpec,
    *,
    outdir: str,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = _run_dir(outdir, spec)
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists() and not force:
        logger.info(f"Loading cached results from {metrics_path}")
        with metrics_path.open() as handle:
            return json.load(handle)

    t, data = _simulate_var_process(spec.N, VAR_COEFFS, SIGMA, spec.seed)
    ts = MultivariateTimeseries(t=t, y=data)
    freq_true_hz = np.fft.rfftfreq(spec.N, d=1.0 / DEFAULT_FS)[1:]
    true_psd = _calculate_true_var_psd_hz(freq_true_hz, VAR_COEFFS, SIGMA)

    coarse_grain_config = None
    if spec.coarse_Nh is not None:
        coarse_grain_config = dict(enabled=True, Nc=None, Nh=spec.coarse_Nh)

    run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "run_spec.json", asdict(spec))

    logger.info(
        f"Running {spec.test} seed={spec.seed} eta={spec.eta} "
        f"eta_c={spec.eta_c} Nb={spec.Nb} Nh={spec.Nh} K={spec.n_knots}"
    )

    t0 = time.time()
    idata = run_mcmc(
        data=ts,
        n_knots=spec.n_knots,
        degree=DEGREE,
        diffMatrixOrder=DIFF_MATRIX_ORDER,
        n_samples=DEFAULT_N_SAMPLES,
        n_warmup=DEFAULT_N_WARMUP,
        num_chains=DEFAULT_NUM_CHAINS,
        rng_key=int(spec.seed),
        outdir=str(run_dir),
        verbose=True,
        target_accept_prob=DEFAULT_TARGET_ACCEPT_PROB,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        init_from_vi=True,
        vi_steps=DEFAULT_VI_STEPS,
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
        eta_c=spec.eta_c,
    )
    wallclock = time.time() - t0
    metrics = _extract_metrics(idata, spec=spec, wallclock=wallclock)
    _save_json(metrics_path, metrics)
    logger.info(
        f"[{spec.test} seed={spec.seed}] "
        f"coverage={float(metrics.get('coverage', np.nan)):.4f} "
        f"cov_diag={float(metrics.get('coverage_diag', np.nan)):.4f} "
        f"cov_offdiag_re={float(metrics.get('coverage_offdiag_re', np.nan)):.4f} "
        f"cov_offdiag_im={float(metrics.get('coverage_offdiag_im', np.nan)):.4f} "
        f"eta_eff={float(metrics.get('eta_effective', np.nan)):.4f} "
        f"div={float(metrics.get('divergences', 0.0)):.0f} "
        f"wall={wallclock:.1f}s"
    )
    return metrics


# ── Analytic eta computation ─────────────────────────────────────────────────


def compute_analytic_eta_candidates(
    bases: tuple,
    penalties: tuple,
    Nb: int,
    Nh: int,
    N_freq: int,
) -> dict[str, float]:
    """Compute candidate analytic η values from basis/penalty matrices.

    Explores several formulas to find which (if any) predicts the right
    tempering factor from first principles.  Returns a dict of named
    candidates and supporting diagnostics.

    Candidates
    ----------
    naive_ratio : tr(P) / (Nb × Nh × tr(B'B))
        Direct ratio of penalty to data information.  Expected to be too
        small (O(n_basis / N_freq / NbNh)) since tr(B'B) ~ N × n_basis.

    edf_ratio : edf / (Nb × Nh × n_basis)
        Where edf = tr((B'B + φ P)^{-1} B'B) is the effective degrees of
        freedom at unit penalty (φ=1).  Captures how many basis functions
        are "free" given the smoothness constraint.

    simple_c2 : 2 / (Nb × Nh)
        The empirical formula with c=2 (for comparison).
    """
    details: dict[str, float] = {}

    for k, (basis, penalty) in enumerate(zip(bases, penalties)):
        B = np.asarray(basis, dtype=np.float64)
        P = np.asarray(penalty, dtype=np.float64)
        n_basis = B.shape[1]

        tr_P = float(np.trace(P))
        BtB = B.T @ B
        tr_BtB = float(np.trace(BtB))

        # Effective degrees of freedom at unit penalty (φ=1).
        # edf = tr((B'B + P)^{-1} B'B) ∈ [0, n_basis]
        try:
            hat_matrix = np.linalg.solve(BtB + P, BtB)
            edf = float(np.trace(hat_matrix))
        except np.linalg.LinAlgError:
            edf = float(n_basis)

        # At φ=10 (stronger penalty)
        try:
            hat_10 = np.linalg.solve(BtB + 10 * P, BtB)
            edf_10 = float(np.trace(hat_10))
        except np.linalg.LinAlgError:
            edf_10 = float(n_basis)

        details[f"tr_P_{k}"] = tr_P
        details[f"tr_BtB_{k}"] = tr_BtB
        details[f"n_basis_{k}"] = n_basis
        details[f"edf_phi1_{k}"] = edf
        details[f"edf_phi10_{k}"] = edf_10

    # Aggregate across components (use component 0 as representative).
    n_basis = int(details.get("n_basis_0", 1))
    tr_P = details.get("tr_P_0", 1.0)
    tr_BtB = details.get("tr_BtB_0", 1.0)
    edf = details.get("edf_phi1_0", float(n_basis))
    edf_10 = details.get("edf_phi10_0", float(n_basis))
    NbNh = float(Nb * Nh)

    # Candidate 1: naive trace ratio
    details["eta_naive"] = min(1.0, tr_P / (NbNh * tr_BtB))
    details["c_naive"] = details["eta_naive"] * NbNh

    # Candidate 2: edf-based (φ=1)
    details["eta_edf"] = min(1.0, edf / (NbNh * n_basis))
    details["c_edf"] = details["eta_edf"] * NbNh

    # Candidate 3: edf-based (φ=10)
    details["eta_edf10"] = min(1.0, edf_10 / (NbNh * n_basis))
    details["c_edf10"] = details["eta_edf10"] * NbNh

    # Candidate 4: n_basis / N_freq / NbNh (another common heuristic)
    details["eta_nbasis_nfreq"] = min(1.0, n_basis / (N_freq * NbNh))
    details["c_nbasis_nfreq"] = details["eta_nbasis_nfreq"] * NbNh

    # Reference: empirical c=2
    details["eta_empirical_c2"] = min(1.0, 2.0 / NbNh)
    details["c_empirical_c2"] = 2.0

    details["Nb"] = Nb
    details["Nh"] = Nh
    details["NbNh"] = NbNh
    details["N_freq"] = N_freq

    return details


def _compute_analytic_candidates_for_spec(spec: RunSpec) -> dict[str, float]:
    """Build basis/penalty matrices and compute analytic η candidates."""
    from log_psplines.psplines.initialisation import init_basis_and_penalty

    # Determine number of frequency bins after Bartlett + coarse graining.
    block_len = spec.N // spec.Nb
    n_freq = block_len // 2  # positive frequencies
    if spec.coarse_Nh is not None and spec.coarse_Nh > 1:
        n_freq = n_freq // spec.coarse_Nh

    # Place knots uniformly in [0, 1] (density placement depends on data,
    # but uniform is representative for the trace ratio).
    knots = np.linspace(0, 1, spec.n_knots)

    basis, penalty = init_basis_and_penalty(
        knots=knots,
        degree=DEGREE,
        n_grid_points=n_freq,
        diff_matrix_order=DIFF_MATRIX_ORDER,
    )

    return compute_analytic_eta_candidates(
        (basis,),
        (penalty,),
        spec.Nb,
        spec.Nh,
        n_freq,
    )


# ── Test spec builders ───────────────────────────────────────────────────────


def _test3_specs(seeds: list[int]) -> list[RunSpec]:
    specs = []
    for K in TEST3_K_VALUES:
        for seed in seeds:
            specs.append(
                RunSpec(
                    test="test3",
                    seed=seed,
                    N=TEST3_N,
                    Nb=TEST3_NB,
                    Nh=TEST3_NH,
                    n_knots=K,
                    eta="auto",
                    eta_c=2.0,
                    label=f"K{K}",
                )
            )
    return specs


def _test4_specs(
    seeds: list[int],
    analytic_details: dict[str, float],
) -> list[RunSpec]:
    """Generate specs for empirical c=2 + analytic candidates."""
    specs = []

    # Candidate etas to test (name → float eta value).
    candidates: dict[str, float] = {
        "empirical_c2": analytic_details["eta_empirical_c2"],
        "edf_phi1": analytic_details["eta_edf"],
        "edf_phi10": analytic_details["eta_edf10"],
        "no_tempering": 1.0,
    }

    for label, eta_val in candidates.items():
        c_val = eta_val * analytic_details["NbNh"]
        for seed in seeds:
            specs.append(
                RunSpec(
                    test="test4",
                    seed=seed,
                    N=TEST4_N,
                    Nb=TEST4_NB,
                    Nh=TEST4_NH,
                    n_knots=TEST4_K,
                    eta=float(eta_val),
                    eta_c=float(c_val),
                    label=label,
                )
            )
    return specs


def _test5_specs(seeds: list[int]) -> list[RunSpec]:
    specs = []
    for N, Nb, Nh, K in TEST5_CONFIGS:
        # With tempering (auto)
        for seed in seeds:
            specs.append(
                RunSpec(
                    test="test5",
                    seed=seed,
                    N=N,
                    Nb=Nb,
                    Nh=Nh,
                    n_knots=K,
                    eta="auto",
                    eta_c=2.0,
                    label=f"auto_Nb{Nb}_Nh{Nh}",
                )
            )
        # Without tempering (eta=1) as baseline
        for seed in seeds:
            specs.append(
                RunSpec(
                    test="test5",
                    seed=seed,
                    N=N,
                    Nb=Nb,
                    Nh=Nh,
                    n_knots=K,
                    eta=1.0,
                    label=f"noeta_Nb{Nb}_Nh{Nh}",
                )
            )
    return specs


# ── Run phases ───────────────────────────────────────────────────────────────


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


def _print_summary(rows: list[dict[str, Any]], title: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("label", ""))].append(row)

    print(f"\n{'=' * 120}")
    print(title)
    print(
        f"{'label':>25s} {'n':>4s} {'coverage':>12s} "
        f"{'cov_diag':>10s} {'cov_od_re':>10s} {'cov_od_im':>10s} "
        f"{'riae':>10s} {'ciw_diag':>10s} {'eta_eff':>8s} {'div':>6s}"
    )
    for label in sorted(groups):
        subset = groups[label]
        cov_m, cov_s, n = _group_mean(subset, "coverage")
        cd_m, _, _ = _group_mean(subset, "coverage_diag")
        cor_m, _, _ = _group_mean(subset, "coverage_offdiag_re")
        coi_m, _, _ = _group_mean(subset, "coverage_offdiag_im")
        riae_m, _, _ = _group_mean(subset, "riae_matrix")
        ciw_m, _, _ = _group_mean(subset, "ciw_diag_median")
        eta_m, _, _ = _group_mean(subset, "eta_effective")
        div_m, _, _ = _group_mean(subset, "divergences")
        print(
            f"{label:>25s} {n:>4d} {cov_m:>7.4f}±{cov_s:>4.3f} "
            f"{cd_m:>10.4f} {cor_m:>10.4f} {coi_m:>10.4f} "
            f"{riae_m:>10.4f} {ciw_m:>10.4f} {eta_m:>8.4f} {div_m:>6.1f}"
        )
    print("=" * 120)


def run_test3(
    seeds: list[int], *, outdir: str = OUT, force: bool = False
) -> list[dict]:
    specs = _test3_specs(seeds)
    rows = [
        _run_var_single(spec, outdir=outdir, force=force) for spec in specs
    ]
    test_dir = HERE / outdir / "test3"
    _write_csv(test_dir / "test3_per_seed.csv", rows)
    _print_summary(rows, "Test 3: K-sweep (eta=auto, c=2, fixed Nb×Nh)")
    return rows


def run_test4(
    seeds: list[int], *, outdir: str = OUT, force: bool = False
) -> list[dict]:
    # Compute analytic candidates from basis/penalty matrices.
    ref_spec = RunSpec(
        test="test4",
        seed=seeds[0],
        N=TEST4_N,
        Nb=TEST4_NB,
        Nh=TEST4_NH,
        n_knots=TEST4_K,
        eta="auto",
        label="ref",
    )
    analytic_details = _compute_analytic_candidates_for_spec(ref_spec)

    logger.info("Analytic η candidates:")
    for key in sorted(analytic_details):
        if (
            key.startswith("eta_")
            or key.startswith("c_")
            or key.startswith("edf")
        ):
            logger.info(f"  {key} = {analytic_details[key]:.6f}")

    # Save analytic details.
    details_path = HERE / outdir / "test4" / "analytic_details.json"
    _save_json(details_path, analytic_details)

    specs = _test4_specs(seeds, analytic_details)
    rows = [
        _run_var_single(spec, outdir=outdir, force=force) for spec in specs
    ]
    test_dir = HERE / outdir / "test4"
    _write_csv(test_dir / "test4_per_seed.csv", rows)
    _print_summary(rows, "Test 4: Analytic η candidates vs empirical c=2")
    return rows


def run_test5(
    seeds: list[int], *, outdir: str = OUT, force: bool = False
) -> list[dict]:
    specs = _test5_specs(seeds)
    rows = [
        _run_var_single(spec, outdir=outdir, force=force) for spec in specs
    ]
    test_dir = HERE / outdir / "test5"
    _write_csv(test_dir / "test5_per_seed.csv", rows)
    _print_summary(
        rows, "Test 5: Element-type coverage (auto vs no tempering)"
    )
    return rows


def run_test6(
    seeds: list[int], *, outdir: str = OUT, force: bool = False
) -> list[dict]:
    """LISA sanity check — requires lisatools."""
    import sys

    lisa_dir = HERE.parent / "lisa"
    if str(lisa_dir) not in sys.path:
        sys.path.insert(0, str(lisa_dir))

    try:
        from log_psplines.example_datasets.lisatools_backend import (
            ensure_lisatools_backends,
        )

        ensure_lisatools_backends()
        from utils.data import generate_lisa_data
        from utils.inference import attach_truth_psd_group, run_lisa_mcmc
        from utils.preprocessing import build_transfer_null_exclusion_bands
    except ImportError as e:
        logger.error(f"LISA dependencies not available: {e}")
        logger.error("Skipping test6. Install lisatools to run LISA tests.")
        return []

    from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

    rows = []
    for seed in seeds:
        spec = RunSpec(
            test="test6",
            seed=seed,
            N=0,  # determined by LISA data generation
            Nb=0,
            Nh=0,
            n_knots=LISA_K,
            eta="auto",
            eta_c=2.0,
            label=f"lisa_seed{seed}",
            data_source="lisa",
        )
        run_dir = _run_dir(outdir, spec)
        metrics_path = run_dir / "metrics.json"

        if metrics_path.exists() and not force:
            logger.info(f"Loading cached results from {metrics_path}")
            with metrics_path.open() as handle:
                rows.append(json.load(handle))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        ts, freq_true, S_true, Nb, Nh, fs = generate_lisa_data(
            seed=seed,
            duration_days=LISA_DURATION_DAYS,
            block_days=LISA_BLOCK_DAYS,
        )

        coarse_cfg = CoarseGrainConfig(
            enabled=True, Nc=None, Nh=LISA_COARSE_NH
        )

        # Build null-excision bands to remove TDI transfer function nulls.
        exclude_bands: tuple[tuple[float, float], ...] = ()
        if LISA_NULL_EXCISION:
            exclude_bands = build_transfer_null_exclusion_bands(
                freq_true,
                half_width=1e-3,
                fmin=LISA_FMIN,
                fmax=LISA_FMAX,
            )
            logger.info(f"Null excision: {len(exclude_bands)} bands removed")

        logger.info(
            f"LISA test6 seed={seed}: N_time={ts.y.shape[0]}, Nb={Nb}, "
            f"coarse_Nh={LISA_COARSE_NH}, K={LISA_K}, "
            f"null_excision={LISA_NULL_EXCISION}"
        )

        t0 = time.time()
        idata = run_lisa_mcmc(
            ts,
            Nb=Nb,
            coarse_cfg=coarse_cfg,
            freq_true=freq_true,
            S_true=S_true,
            K=LISA_K,
            n_samples=2000,
            n_warmup=2000,
            num_chains=4,
            target_accept=0.85,
            max_tree_depth=12,
            vi=True,
            vi_steps=50_000,
            vi_guide="lowrank:16",
            outdir=str(run_dir),
            fmin=LISA_FMIN,
            fmax=LISA_FMAX,
            exclude_freq_bands=exclude_bands,
            eta="auto",
            eta_c=2.0,
        )
        wallclock = time.time() - t0

        attrs = getattr(idata, "attrs", {})
        sampling_eta = _channel_attr_values(attrs, "sampling_eta_channel_")
        eta_eff = _mean_or_nan(sampling_eta)

        metrics: dict[str, Any] = {
            "test": "test6",
            "label": f"lisa_seed{seed}",
            "seed": seed,
            "data_source": "lisa",
            "N": int(ts.y.shape[0]),
            "Nb": Nb,
            "Nh": LISA_COARSE_NH,
            "NbNh": Nb * LISA_COARSE_NH,
            "n_knots": LISA_K,
            "n_basis": LISA_K + DEGREE - 1,
            "eta_input": "auto",
            "eta_c": 2.0,
            "eta_effective": eta_eff,
            "c_effective": eta_eff * Nb * LISA_COARSE_NH,
            "coverage": float(attrs.get("coverage", np.nan)),
            "coverage_diag": float(attrs.get("coverage_diag", np.nan)),
            "coverage_offdiag_re": float(
                attrs.get("coverage_offdiag_re", np.nan)
            ),
            "coverage_offdiag_im": float(
                attrs.get("coverage_offdiag_im", np.nan)
            ),
            "coverage_coherence": float(
                attrs.get("coverage_coherence", np.nan)
            ),
            "riae_matrix": float(
                attrs.get("riae_matrix", attrs.get("riae", np.nan))
            ),
            "riae_diag_mean": float(attrs.get("riae_diag_mean", np.nan)),
            "ess_median": float(
                np.nanmedian(
                    np.asarray(attrs.get("ess", np.nan), dtype=np.float64)
                )
            ),
            "wallclock_s": round(wallclock, 3),
        }
        metrics.update(_extract_nuts_diagnostics(idata))
        _save_json(metrics_path, metrics)
        rows.append(metrics)

        logger.info(
            f"[test6 seed={seed}] coverage={metrics['coverage']:.4f} "
            f"cov_diag={metrics['coverage_diag']:.4f} "
            f"eta_eff={eta_eff:.4f} wall={wallclock:.1f}s"
        )

    test_dir = HERE / outdir / "test6"
    _write_csv(test_dir / "test6_per_seed.csv", rows)
    _print_summary(rows, "Test 6: LISA sanity check (eta=auto, c=2)")
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────────


def _collect_all(outdir: str) -> list[dict[str, Any]]:
    base_dir = HERE / outdir
    rows = []
    for metrics_path in sorted(base_dir.rglob("metrics.json")):
        with metrics_path.open() as handle:
            rows.append(json.load(handle))
    return rows


def plot_results(*, outdir: str = OUT) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting.")
        return

    rows = _collect_all(outdir)
    if not rows:
        logger.warning("No results found.")
        return

    plot_dir = HERE / outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Test 3 plots: K-sweep ────────────────────────────────────────────
    t3 = [r for r in rows if r.get("test") == "test3"]
    if t3:
        K_values = sorted({int(r["n_knots"]) for r in t3})
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        cov_means, cov_stds = [], []
        riae_means = []
        ciw_means = []
        for K in K_values:
            subset = [r for r in t3 if int(r["n_knots"]) == K]
            cm, cs, _ = _group_mean(subset, "coverage")
            cov_means.append(cm)
            cov_stds.append(cs)
            rm, _, _ = _group_mean(subset, "riae_matrix")
            riae_means.append(rm)
            wm, _, _ = _group_mean(subset, "ciw_diag_median")
            ciw_means.append(wm)

        n_basis_values = [K + DEGREE - 1 for K in K_values]
        axes[0].errorbar(
            n_basis_values, cov_means, yerr=cov_stds, fmt="o-", lw=2
        )
        axes[0].axhline(0.9, ls="--", color="grey")
        axes[0].set_ylabel("Coverage")
        axes[0].set_xlabel("n_basis")
        axes[0].set_title("Coverage vs n_basis (c=2 fixed)")

        axes[1].plot(n_basis_values, riae_means, "s-", lw=2)
        axes[1].set_ylabel("RIAE")
        axes[1].set_xlabel("n_basis")
        axes[1].set_title("RIAE vs n_basis")

        axes[2].plot(n_basis_values, ciw_means, "^-", lw=2)
        axes[2].set_ylabel("CI width (diag)")
        axes[2].set_xlabel("n_basis")
        axes[2].set_title("CI width vs n_basis")

        fig.tight_layout()
        fig.savefig(plot_dir / "test3_k_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved test3_k_sweep.png")

    # ── Test 4 plots: analytic vs empirical ──────────────────────────────
    t4 = [r for r in rows if r.get("test") == "test4"]
    if t4:
        labels = sorted({str(r.get("label", "")) for r in t4})
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for label in labels:
            subset = [r for r in t4 if str(r.get("label", "")) == label]
            cm, cs, n = _group_mean(subset, "coverage")
            rm, _, _ = _group_mean(subset, "riae_matrix")
            em, _, _ = _group_mean(subset, "eta_effective")
            axes[0].bar(label, cm, yerr=cs, alpha=0.8)
            axes[1].bar(label, rm, alpha=0.8)
            axes[2].bar(label, em, alpha=0.8)

        axes[0].axhline(0.9, ls="--", color="grey")
        axes[0].set_ylabel("Coverage")
        axes[0].set_title("Analytic vs Empirical")
        axes[1].set_ylabel("RIAE")
        axes[2].set_ylabel("η effective")
        fig.tight_layout()
        fig.savefig(plot_dir / "test4_analytic_vs_empirical.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved test4_analytic_vs_empirical.png")

    # ── Test 5 plots: element-type coverage ──────────────────────────────
    t5 = [r for r in rows if r.get("test") == "test5"]
    if t5:
        labels = sorted({str(r.get("label", "")) for r in t5})
        element_keys = [
            ("coverage_diag", "Diagonal"),
            ("coverage_offdiag_re", "Offdiag Re"),
            ("coverage_offdiag_im", "Offdiag Im"),
            ("coverage_coherence", "Coherence"),
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        width = 0.18
        for i, (key, name) in enumerate(element_keys):
            means = []
            stds = []
            for label in labels:
                subset = [r for r in t5 if str(r.get("label", "")) == label]
                m, s, _ = _group_mean(subset, key)
                means.append(m)
                stds.append(s)
            ax.bar(
                x + i * width, means, width, yerr=stds, label=name, alpha=0.85
            )

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0.9, ls="--", color="grey", label="Nominal 90%")
        ax.set_ylabel("Coverage")
        ax.set_title("Coverage by element type")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "test5_element_coverage.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved test5_element_coverage.png")

    # ── Test 6 plots: LISA results ───────────────────────────────────────
    t6 = [r for r in rows if r.get("test") == "test6"]
    if t6:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        element_keys = [
            ("coverage_diag", "Diagonal"),
            ("coverage_offdiag_re", "Offdiag Re"),
            ("coverage_offdiag_im", "Offdiag Im"),
        ]
        for key, name in element_keys:
            values = [float(r.get(key, np.nan)) for r in t6]
            seeds_list = [int(r.get("seed", 0)) for r in t6]
            axes[0].scatter(seeds_list, values, label=name, s=60)

        axes[0].axhline(0.9, ls="--", color="grey")
        axes[0].set_xlabel("Seed")
        axes[0].set_ylabel("Coverage")
        axes[0].set_title("LISA: coverage by element type")
        axes[0].legend(fontsize=8)

        overall_cov = [float(r.get("coverage", np.nan)) for r in t6]
        eta_eff = [float(r.get("eta_effective", np.nan)) for r in t6]
        axes[1].scatter(eta_eff, overall_cov, s=60)
        axes[1].axhline(0.9, ls="--", color="grey")
        axes[1].set_xlabel("η effective")
        axes[1].set_ylabel("Overall coverage")
        axes[1].set_title("LISA: coverage vs η")

        fig.tight_layout()
        fig.savefig(plot_dir / "test6_lisa.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved test6_lisa.png")

    logger.info(f"All plots saved to {plot_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────


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
        description="η-tempering validation study",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("test3", "test4", "test5", "test6", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--seeds", type=str, default="0-4")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--outdir", type=str, default=OUT)

    sp_plots = sub.add_parser("plots")
    sp_plots.add_argument("--outdir", type=str, default=OUT)

    args = parser.parse_args()

    if args.cmd == "plots":
        plot_results(outdir=args.outdir)
        return

    seeds = _parse_seeds(args.seeds)
    outdir = args.outdir
    force = args.force

    if args.cmd == "test3":
        run_test3(seeds, outdir=outdir, force=force)
    elif args.cmd == "test4":
        run_test4(seeds, outdir=outdir, force=force)
    elif args.cmd == "test5":
        run_test5(seeds, outdir=outdir, force=force)
    elif args.cmd == "test6":
        run_test6(seeds, outdir=outdir, force=force)
    elif args.cmd == "all":
        run_test3(seeds, outdir=outdir, force=force)
        run_test4(seeds, outdir=outdir, force=force)
        run_test5(seeds, outdir=outdir, force=force)
        run_test6(seeds, outdir=outdir, force=force)
        plot_results(outdir=outdir)


if __name__ == "__main__":
    main()
