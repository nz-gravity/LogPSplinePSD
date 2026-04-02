"""η-tempering investigation for the multivariate Whittle likelihood.

Multiplies the Whittle log-likelihood by η ∈ (0, 1] to produce a generalized
(Safe Bayes) posterior.  The Whittle likelihood is a pseudo-likelihood whose
Fisher information overstates the true information by a factor related to
N_b × N_H.  Tempering with η < 1 widens the credible intervals and can
recover nominal 90 % coverage.

Four phases:
  Phase 1 — Static η grid: sweep η values, record coverage + CI width + NUTS
             diagnostics for 20 seeds per η.
  Phase 2 — Scaling test: repeat Phase 1 for several N_H values; check whether
             η* scales with 1/(N_b × N_H).
  Phase 3 — Two-stage annealing: warmup at low η, production at η=1; compare
             against baseline.
  Phase 4 — Diagnostic plots for one representative seed.

Usage
-----
# Phase 1 — static grid (20 seeds, short_nb4 mode)
    python eta_tempering_study.py phase1 --seeds 0-19

# Phase 1 — single seed (quick test)
    python eta_tempering_study.py phase1 --seeds 0 --quick

# Phase 2 — scaling test
    python eta_tempering_study.py phase2 --seeds 0-4

# Phase 3 — two-stage annealing
    python eta_tempering_study.py phase3 --seeds 0-4

# Phase 4 — diagnostic plots (single seed)
    python eta_tempering_study.py phase4 --seed 0

# Collect results from Phase 1 runs
    python eta_tempering_study.py collect --results-dir out_eta
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Literal

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

set_level("INFO")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("out_eta")

# ---------------------------------------------------------------------------
# VAR(2) setup (3 channels) — identical to 3d_study.py
# ---------------------------------------------------------------------------
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
DEFAULT_FS = 1.0
DEFAULT_BURN_IN = 512
EPS = 1e-12

# MCMC defaults (matching 3d_study.py production settings)
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

# Quick mode overrides
QUICK_N_SAMPLES = 1000
QUICK_N_WARMUP = 1000
QUICK_NUM_CHAINS = 2
QUICK_VI_STEPS = 20_000

# Phase 1 configuration
# Using short_nb4: N=2048, Nb=4, no coarse-graining (Nh=1 effective)
# This gives Nb*Nh = 4, a moderate effective sample size.
PHASE1_N = 2048
PHASE1_NB = 4
PHASE1_COARSE_NH = None  # no coarse-graining -> Nh=1
PHASE1_K = 20
PHASE1_ETA_GRID = ["auto", 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0]

# Phase 2: vary Nh to test scaling
PHASE2_N = 16384
PHASE2_NB = 4
PHASE2_NH_VALUES = [1, 2, 4]
PHASE2_ETA_GRID = ["auto", 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
PHASE2_K = 20


# ---------------------------------------------------------------------------
# Data generation helpers (from 3d_study.py)
# ---------------------------------------------------------------------------


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
        s_f = h_f @ sigma @ h_f.conj().T
        psd[idx] = (2.0 / float(fs)) * s_f
    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]
    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    psd = np.where(np.abs(psd) < EPS, EPS, psd)
    return psd


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def _extract_nuts_diagnostics(idata) -> dict[str, float]:
    """Extract NUTS diagnostics from sample_stats."""
    metrics: dict[str, float] = {}
    ss = getattr(idata, "sample_stats", None)
    if ss is None:
        return metrics

    p = 3  # number of channels
    total_divergences = 0
    step_sizes = []
    tree_depths = []

    for j in range(p):
        div_key = f"diverging_channel_{j}"
        steps_key = f"num_steps_channel_{j}"
        ss_key = f"step_size_channel_{j}"

        if div_key in ss:
            div_arr = np.asarray(ss[div_key].values)
            total_divergences += int(np.sum(div_arr))
        if steps_key in ss:
            tree_depths.append(float(np.mean(ss[steps_key].values)))
        if ss_key in ss:
            step_sizes.append(float(np.mean(ss[ss_key].values)))

    metrics["divergences"] = total_divergences
    if tree_depths:
        metrics["mean_tree_depth"] = float(np.mean(tree_depths))
    if step_sizes:
        metrics["mean_step_size"] = float(np.mean(step_sizes))

    return metrics


def _extract_metrics(
    idata,
    *,
    eta: float | str,
    seed: int,
    N: int,
    Nb: int,
    Nh: int,
    K: int,
    wallclock: float,
) -> dict[str, float | int | str]:
    attrs = idata.attrs
    ess_raw = attrs.get("ess", np.nan)
    ess_arr = np.asarray(ess_raw, dtype=float)
    ess_median = float(np.nanmedian(ess_arr)) if ess_arr.size else float("nan")

    metrics: dict[str, float | int | str] = {
        "eta": str(eta) if isinstance(eta, str) else float(eta),
        "seed": int(seed),
        "N": int(N),
        "Nb": int(Nb),
        "Nh": int(Nh),
        "K": int(K),
        "NbNh": int(Nb * Nh),
        "coverage": float(attrs.get("coverage", np.nan)),
        "riae_matrix": float(
            attrs.get("riae_matrix", attrs.get("riae", np.nan))
        ),
        "ess_median": ess_median,
        "wallclock_s": round(wallclock, 1),
    }

    # CI width metrics
    psd_group = getattr(idata, "posterior_psd", None)
    if psd_group is not None and "psd_matrix_real" in psd_group:
        psd_real = np.asarray(
            psd_group["psd_matrix_real"].values, dtype=np.float64
        )
        percentiles = np.asarray(
            psd_group["psd_matrix_real"].coords.get(
                "percentile", np.arange(psd_real.shape[0], dtype=float)
            ),
            dtype=np.float64,
        )
        if psd_real.shape[0] >= 2:
            q05_idx = int(np.argmin(np.abs(percentiles - 5.0)))
            q95_idx = int(np.argmin(np.abs(percentiles - 95.0)))
            width = np.maximum(psd_real[q95_idx] - psd_real[q05_idx], 0.0)
            p = width.shape[1]
            diag_idx = np.arange(p)
            metrics["ciw_diag_median"] = float(
                np.median(width[:, diag_idx, diag_idx])
            )
            offdiag_mask = ~np.eye(p, dtype=bool)
            metrics["ciw_offdiag_median"] = float(
                np.median(width[:, offdiag_mask])
            )
            metrics["ciw_overall_median"] = float(np.median(width))

    metrics.update(_extract_nuts_diagnostics(idata))
    return metrics


def _save_metrics(outdir: str, metrics: dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {path}")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_single(
    *,
    seed: int,
    eta: float | str,
    N: int,
    Nb: int,
    coarse_Nh: int | None,
    K: int,
    outdir: str,
    quick: bool = False,
) -> dict:
    """Run a single MCMC experiment with the given eta value."""
    Nh_effective = coarse_Nh if coarse_Nh is not None else 1
    eta_label = eta if isinstance(eta, str) else f"{eta:.4f}"
    run_label = f"eta{eta_label}_seed{seed}_N{N}_Nb{Nb}_Nh{Nh_effective}_K{K}"
    seed_outdir = os.path.join(HERE, outdir, run_label)
    os.makedirs(seed_outdir, exist_ok=True)

    # Check for cached results
    metrics_path = os.path.join(seed_outdir, "metrics.json")
    if os.path.exists(metrics_path):
        logger.info(f"Loading cached results from {metrics_path}")
        with open(metrics_path) as f:
            return json.load(f)

    n_samples = QUICK_N_SAMPLES if quick else DEFAULT_N_SAMPLES
    n_warmup = QUICK_N_WARMUP if quick else DEFAULT_N_WARMUP
    num_chains = QUICK_NUM_CHAINS if quick else DEFAULT_NUM_CHAINS
    vi_steps = QUICK_VI_STEPS if quick else DEFAULT_VI_STEPS

    # Generate data
    t, data = _simulate_var_process(N, VAR_COEFFS, SIGMA, seed)
    ts = MultivariateTimeseries(t=t, y=data)
    freq_true_hz = np.fft.rfftfreq(N, d=1.0 / DEFAULT_FS)[1:]
    true_psd = _calculate_true_var_psd_hz(freq_true_hz, VAR_COEFFS, SIGMA)

    coarse_grain_config = None
    if coarse_Nh is not None:
        coarse_grain_config = dict(enabled=True, Nc=None, Nh=int(coarse_Nh))

    t0 = time.time()
    idata = run_mcmc(
        data=ts,
        n_knots=K,
        degree=2,
        diffMatrixOrder=2,
        n_samples=n_samples,
        n_warmup=n_warmup,
        num_chains=num_chains,
        outdir=seed_outdir,
        verbose=True,
        target_accept_prob=DEFAULT_TARGET_ACCEPT_PROB,
        max_tree_depth=DEFAULT_MAX_TREE_DEPTH,
        init_from_vi=True,
        vi_steps=vi_steps,
        vi_guide=DEFAULT_VI_GUIDE,
        vi_lr=DEFAULT_VI_LR,
        Nb=Nb,
        knot_kwargs=dict(method="density", scoring="spectral"),
        coarse_grain_config=coarse_grain_config,
        alpha_delta=DEFAULT_ALPHA_DELTA,
        beta_delta=DEFAULT_BETA_DELTA,
        compute_coherence_quantiles=True,
        true_psd=(freq_true_hz, true_psd),
        max_save_bytes=20_000_000,
        eta=eta,
    )
    wallclock = time.time() - t0

    metrics = _extract_metrics(
        idata,
        eta=eta,
        seed=seed,
        N=N,
        Nb=Nb,
        Nh=Nh_effective,
        K=K,
        wallclock=wallclock,
    )
    _save_metrics(seed_outdir, metrics)

    logger.info(
        f"[eta={eta_label} seed={seed}] coverage={metrics['coverage']:.4f} "
        f"ciw_diag={metrics.get('ciw_diag_median', 'N/A')} "
        f"divergences={metrics.get('divergences', 'N/A')} "
        f"time={wallclock:.0f}s"
    )
    return metrics


# ---------------------------------------------------------------------------
# Phase 1: Static η grid
# ---------------------------------------------------------------------------


def phase1(seeds: list[int], quick: bool = False, outdir: str = OUT) -> None:
    """Sweep η values on the short_nb4 configuration."""
    all_metrics = []
    for eta in PHASE1_ETA_GRID:
        for seed in seeds:
            metrics = run_single(
                seed=seed,
                eta=eta,
                N=PHASE1_N,
                Nb=PHASE1_NB,
                coarse_Nh=PHASE1_COARSE_NH,
                K=PHASE1_K,
                outdir=outdir,
                quick=quick,
            )
            all_metrics.append(metrics)

    _save_phase_summary(all_metrics, os.path.join(HERE, outdir), "phase1")


# ---------------------------------------------------------------------------
# Phase 2: Scaling test
# ---------------------------------------------------------------------------


def phase2(seeds: list[int], quick: bool = False, outdir: str = OUT) -> None:
    """Repeat η sweep for multiple Nh values to test scaling."""
    all_metrics = []
    for Nh in PHASE2_NH_VALUES:
        coarse_Nh = None if Nh <= 1 else Nh
        # Adjust N to ensure divisibility: N must be divisible by Nb
        N = PHASE2_N
        for eta in PHASE2_ETA_GRID:
            for seed in seeds:
                metrics = run_single(
                    seed=seed,
                    eta=eta,
                    N=N,
                    Nb=PHASE2_NB,
                    coarse_Nh=coarse_Nh,
                    K=PHASE2_K,
                    outdir=outdir,
                    quick=quick,
                )
                all_metrics.append(metrics)

    _save_phase_summary(all_metrics, os.path.join(HERE, outdir), "phase2")


# ---------------------------------------------------------------------------
# Phase 3: Two-stage annealing
# ---------------------------------------------------------------------------


def phase3(seeds: list[int], quick: bool = False, outdir: str = OUT) -> None:
    """Compare two-stage warmup (low-η warmup → η=1 production) vs baseline.

    Strategy: run η=1 (baseline) and η=0.05 (warmup-only tempering) for each
    seed and compare warmup efficiency and final coverage.
    """
    all_metrics = []

    # Baseline: eta=1.0 throughout
    for seed in seeds:
        metrics = run_single(
            seed=seed,
            eta=1.0,
            N=PHASE1_N,
            Nb=PHASE1_NB,
            coarse_Nh=PHASE1_COARSE_NH,
            K=PHASE1_K,
            outdir=os.path.join(outdir, "phase3_baseline"),
            quick=quick,
        )
        metrics["phase3_group"] = "baseline"
        all_metrics.append(metrics)

    # Two-stage: eta=0.05 (serves as warmup-adapted target, then production
    # at full η=1 reuses mass matrix).  For now we approximate this by just
    # running at η=0.05 to see if the adapted mass matrix / step size are
    # more reasonable.  True two-stage requires MCMC API changes.
    for seed in seeds:
        metrics = run_single(
            seed=seed,
            eta=0.05,
            N=PHASE1_N,
            Nb=PHASE1_NB,
            coarse_Nh=PHASE1_COARSE_NH,
            K=PHASE1_K,
            outdir=os.path.join(outdir, "phase3_tempered"),
            quick=quick,
        )
        metrics["phase3_group"] = "tempered_warmup"
        all_metrics.append(metrics)

    _save_phase_summary(all_metrics, os.path.join(HERE, outdir), "phase3")


# ---------------------------------------------------------------------------
# Phase 4: Diagnostic plots
# ---------------------------------------------------------------------------


def phase4(seed: int = 0, quick: bool = False, outdir: str = OUT) -> None:
    """Generate diagnostic plots for one seed across η values."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for Phase 4 plots.")
        return

    plot_etas = [0.05, 0.1, 0.3, 0.5, 1.0]
    results = {}

    for eta in plot_etas:
        metrics = run_single(
            seed=seed,
            eta=eta,
            N=PHASE1_N,
            Nb=PHASE1_NB,
            coarse_Nh=PHASE1_COARSE_NH,
            K=PHASE1_K,
            outdir=outdir,
            quick=quick,
        )
        results[eta] = metrics

    plot_dir = os.path.join(HERE, outdir, "phase4_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: Coverage vs eta
    etas = sorted(results.keys())
    coverages = [results[e].get("coverage", np.nan) for e in etas]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(etas, coverages, "o-", color="C0", lw=2)
    ax.axhline(0.90, ls="--", color="grey", label="Nominal 90%")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Coverage (90% CI)")
    ax.set_title(f"Coverage vs η (seed={seed}, N={PHASE1_N}, Nb={PHASE1_NB})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "coverage_vs_eta.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved coverage_vs_eta.png")

    # Plot 2: CI width vs eta
    ciw_diag = [results[e].get("ciw_diag_median", np.nan) for e in etas]
    ciw_offdiag = [results[e].get("ciw_offdiag_median", np.nan) for e in etas]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(etas, ciw_diag, "s-", color="C1", label="Diagonal", lw=2)
    ax.plot(etas, ciw_offdiag, "D-", color="C2", label="Off-diagonal", lw=2)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Median CI width (90%)")
    ax.set_title(f"CI width vs η (seed={seed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "ciw_vs_eta.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved ciw_vs_eta.png")

    # Plot 3: NUTS diagnostics vs eta
    tree_depths = [results[e].get("mean_tree_depth", np.nan) for e in etas]
    step_sizes = [results[e].get("mean_step_size", np.nan) for e in etas]
    divergences = [results[e].get("divergences", 0) for e in etas]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(etas, tree_depths, "o-", color="C3", lw=2)
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel("Mean tree depth")
    axes[0].set_title("Tree depth vs η")

    axes[1].plot(etas, step_sizes, "o-", color="C4", lw=2)
    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel("Mean step size")
    axes[1].set_title("Step size vs η")

    axes[2].bar(
        range(len(etas)),
        divergences,
        color="C5",
        tick_label=[f"{e}" for e in etas],
    )
    axes[2].set_xlabel(r"$\eta$")
    axes[2].set_ylabel("Divergences")
    axes[2].set_title("Divergences vs η")

    fig.suptitle(f"NUTS diagnostics (seed={seed})")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "nuts_diagnostics_vs_eta.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved nuts_diagnostics_vs_eta.png")


# ---------------------------------------------------------------------------
# Results collection
# ---------------------------------------------------------------------------


def _save_phase_summary(
    all_metrics: list[dict],
    results_dir: str,
    phase_name: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{phase_name}_per_seed.csv")
    if not all_metrics:
        logger.warning("No metrics to save.")
        return

    fieldnames = list(all_metrics[0].keys())
    # Ensure consistent column ordering across all rows
    for m in all_metrics:
        for k in m:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info(f"Per-seed results saved to {csv_path}")

    _print_summary_table(all_metrics, phase_name)


def _print_summary_table(all_metrics: list[dict], phase_name: str) -> None:
    """Print aggregated summary grouped by (eta, NbNh)."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in all_metrics:
        key = (m.get("eta", "?"), m.get("NbNh", "?"))
        groups[key].append(m)

    header = f"\n{'='*70}\n{phase_name} Summary\n{'='*70}"
    header += f"\n{'eta':>8s} {'NbNh':>6s} {'n':>4s} {'coverage':>10s} {'ciw_diag':>10s} {'ciw_off':>10s} {'diverge':>8s} {'tree_d':>8s}"
    print(header)

    for key in sorted(groups.keys()):
        rows = groups[key]
        eta, nbnh = key
        n = len(rows)
        cov = np.nanmean([r.get("coverage", np.nan) for r in rows])
        cov_std = np.nanstd([r.get("coverage", np.nan) for r in rows])
        ciw_d = np.nanmean([r.get("ciw_diag_median", np.nan) for r in rows])
        ciw_o = np.nanmean([r.get("ciw_offdiag_median", np.nan) for r in rows])
        div = np.sum([r.get("divergences", 0) for r in rows])
        td = np.nanmean([r.get("mean_tree_depth", np.nan) for r in rows])
        print(
            f"{eta:>8.4f} {nbnh:>6} {n:>4d} "
            f"{cov:>7.4f}±{cov_std:>4.3f} "
            f"{ciw_d:>10.4f} {ciw_o:>10.4f} "
            f"{div:>8.0f} {td:>8.1f}"
        )
    print("=" * 70)


def collect_results(results_dir: str = OUT) -> None:
    """Walk results_dir, load all metrics.json files, print summary."""
    results_dir = os.path.join(HERE, results_dir)
    all_metrics = []
    for root, dirs, files in os.walk(results_dir):
        if "metrics.json" in files:
            with open(os.path.join(root, "metrics.json")) as f:
                m = json.load(f)
                all_metrics.append(m)
    if not all_metrics:
        logger.warning(f"No metrics.json found under {results_dir}")
        return
    logger.info(f"Collected {len(all_metrics)} results from {results_dir}")
    _print_summary_table(all_metrics, "collected")
    _save_phase_summary(all_metrics, results_dir, "collected")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(s: str) -> list[int]:
    """Parse seed spec like '0-19' or '0,1,5' or '3'."""
    seeds = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(
        description="η-tempering investigation for multivariate Whittle likelihood"
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    # Phase 1
    p1 = sub.add_parser("phase1", help="Static η grid sweep")
    p1.add_argument("--seeds", type=str, default="0-19")
    p1.add_argument("--quick", action="store_true")
    p1.add_argument("--outdir", type=str, default=OUT)

    # Phase 2
    p2 = sub.add_parser("phase2", help="Scaling test across Nh values")
    p2.add_argument("--seeds", type=str, default="0-4")
    p2.add_argument("--quick", action="store_true")
    p2.add_argument("--outdir", type=str, default=OUT)

    # Phase 3
    p3 = sub.add_parser("phase3", help="Two-stage annealing comparison")
    p3.add_argument("--seeds", type=str, default="0-4")
    p3.add_argument("--quick", action="store_true")
    p3.add_argument("--outdir", type=str, default=OUT)

    # Phase 4
    p4 = sub.add_parser("phase4", help="Diagnostic plots for one seed")
    p4.add_argument("--seed", type=int, default=0)
    p4.add_argument("--quick", action="store_true")
    p4.add_argument("--outdir", type=str, default=OUT)

    # Collect
    pc = sub.add_parser("collect", help="Collect and summarize results")
    pc.add_argument("--results-dir", type=str, default=OUT)

    args = parser.parse_args()

    if args.phase == "phase1":
        phase1(_parse_seeds(args.seeds), quick=args.quick, outdir=args.outdir)
    elif args.phase == "phase2":
        phase2(_parse_seeds(args.seeds), quick=args.quick, outdir=args.outdir)
    elif args.phase == "phase3":
        phase3(_parse_seeds(args.seeds), quick=args.quick, outdir=args.outdir)
    elif args.phase == "phase4":
        phase4(seed=args.seed, quick=args.quick, outdir=args.outdir)
    elif args.phase == "collect":
        collect_results(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
