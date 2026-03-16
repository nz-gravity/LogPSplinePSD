"""Coverage diagnostic tests for the multivariate P-spline PSD estimator.

Four tests that help isolate *why* empirical CI coverage is below the nominal
90% target.

TEST 1 – White-noise calibration
    Generate i.i.d. multivariate Gaussian data (flat, known PSD). Run the
    full MCMC pipeline with Nb=4 blocks and K=20 knots.  For a perfectly
    calibrated model, diagonal coverage should be ≈ 0.90.  Any systematic
    gap here points to a likelihood normalisation issue (not just smoothing).

TEST 2 – Element-wise coverage breakdown
    Load one or more existing compact_ci_curves.npz files (from previous
    simulation-study runs) and re-compute coverage split by:
        * diagonal auto-spectra (real parts only)
        * upper-triangle cross-spectra, real parts
        * lower-triangle cross-spectra, imaginary parts
    Together with the new compute_ci_coverage_multivar_detailed helper in
    _utils.py, this tells you *where* the undercoverage lives.

TEST 3 – Posterior inflation factor
    For each compact_ci_curves.npz, binary-search for the scalar ``c`` such
    that inflating every CI around its median by factor ``c`` achieves ≈ 90%
    coverage.  c > 1 quantifies how much the Whittle posterior is
    over-concentrated.

TEST 4 – Nb comparison  (Nb=2 vs Nb=3 vs Nb=4)
    Run the full simulation_study pipeline for seeds 0-4 at N=2048 with
    Nb=2, 3, and 4 blocks.  With p=3 channels, Nb < p means the per-frequency
    Wishart matrix is rank-deficient; Nb=4 is the first full-rank configuration.
    This test isolates the rank-deficiency hypothesis.

Usage (CLI examples)
--------------------
# Run all four tests:
    python coverage_tests.py all

# Run a single test:
    python coverage_tests.py test1
    python coverage_tests.py test2 [--glob "out_var3/seed_*_short_N2048_K20/compact_ci_curves.npz"]
    python coverage_tests.py test3 [--glob "..."]
    python coverage_tests.py test4

# Run test1 with a quick/debug 500-sample MCMC:
    python coverage_tests.py test1 --quick
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# JAX / project setup
# ---------------------------------------------------------------------------
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from log_psplines.diagnostics._utils import (  # noqa: E402
    compute_ci_coverage_multivar,
    compute_ci_coverage_multivar_detailed,
    find_posterior_inflation_factor,
)

# ---------------------------------------------------------------------------
# Shared VAR(2) constants (mirror of 3d_study.py)
# ---------------------------------------------------------------------------
DEFAULT_FS = 1.0
DEFAULT_BURN_IN = 512
EPS = 1e-12

A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [[-0.2, 0.5, 0.0], [0.4, -0.1, 0.0], [0.0, 0.0, -0.1]], dtype=np.float64
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_var(
    n: int, sigma: np.ndarray, var_coeffs: np.ndarray, seed: int
) -> np.ndarray:
    """Simulate a VAR(p) process, return (n, p) array."""
    ar_order, p, _ = var_coeffs.shape
    n_total = n + DEFAULT_BURN_IN
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(np.zeros(p), sigma, size=n_total)
    x = np.zeros((n_total, p), dtype=np.float64)
    for t in range(ar_order, n_total):
        state = noise[t].copy()
        for lag in range(1, ar_order + 1):
            state += var_coeffs[lag - 1] @ x[t - lag]
        x[t] = state
    return x[DEFAULT_BURN_IN:]


def _simulate_white_noise(n: int, sigma: np.ndarray, seed: int) -> np.ndarray:
    """Simulate i.i.d. Gaussian data with covariance ``sigma``."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=n)


def _white_noise_true_psd(
    sigma: np.ndarray, n_freqs: int, fs: float = 1.0
) -> np.ndarray:
    """One-sided true PSD for white noise: S(f) = 2*sigma/fs, shape (n_freqs, p, p)."""
    S = (2.0 / fs) * sigma
    return np.tile(S[None, :, :], (n_freqs, 1, 1)).astype(np.complex128)


def _var_true_psd(freqs_hz: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """One-sided theoretical PSD for the study's VAR(2) model."""
    p = VAR_COEFFS.shape[1]
    omega = 2.0 * np.pi * freqs_hz / fs
    psd = np.empty((freqs_hz.size, p, p), dtype=np.complex128)
    ident = np.eye(p, dtype=np.complex128)
    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, VAR_COEFFS.shape[0] + 1):
            a_f -= VAR_COEFFS[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        psd[idx] = (2.0 / fs) * (h_f @ SIGMA @ h_f.conj().T)
    if np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] *= 0.5
    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    return np.where(np.abs(psd) < EPS, EPS, psd)


def _build_percentiles_stack(
    npz: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (3, F, p, p) complex percentile stack and true PSD from .npz."""
    q05_re = npz["psd_real_q05"]
    q50_re = npz["psd_real_q50"]
    q95_re = npz["psd_real_q95"]
    q05_im = npz["psd_imag_q05"]
    q50_im = npz["psd_imag_q50"]
    q95_im = npz["psd_imag_q95"]

    stack = np.stack(
        [
            q05_re + 1j * q05_im,
            q50_re + 1j * q50_im,
            q95_re + 1j * q95_im,
        ],
        axis=0,
    )  # (3, F, p, p)

    # True PSD (stored inside .npz from _save_compact_ci_curves)
    true_psd = npz["true_psd_real"] + 1j * npz["true_psd_imag"]
    return stack, true_psd


def _print_header(title: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def _print_detailed_coverage(detail: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Overall   : {detail['overall']:.4f}")
    print(
        f"  {prefix}Diagonal  : {detail['diag']:.4f}  (n={detail['n_diag']})"
    )
    print(
        f"  {prefix}Re offdiag: {detail['offdiag_re']:.4f}  (n={detail['n_offdiag_re']})"
    )
    print(
        f"  {prefix}Im offdiag: {detail['offdiag_im']:.4f}  (n={detail['n_offdiag_im']})"
    )


# ---------------------------------------------------------------------------
# TEST 1 – White-noise calibration
# ---------------------------------------------------------------------------


def test1_white_noise_calibration(
    *,
    n: int = 2048,
    nb: int = 4,
    k: int = 20,
    seeds: list[int] | None = None,
    quick: bool = False,
    outdir: str = "out_tests/test1_whitenoise",
) -> None:
    """Run the full MCMC pipeline on white noise and check coverage.

    A well-calibrated model should give diagonal coverage ≈ 0.90.
    Off-diagonal elements of a white-noise PSD are identically zero (or very
    small), so off-diagonal coverage is less meaningful here.

    Parameters
    ----------
    n, nb, k : int
        Sample size, number of blocks, number of knots.
    seeds : list[int], optional
        Seeds to run. Defaults to [0, 1, 2].
    quick : bool
        If True, use 500 warmup + 500 samples (fast debugging mode).
    outdir : str
        Output directory for inference results.
    """
    _print_header("TEST 1 – White-noise calibration")

    from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

    seeds = seeds or [0, 1, 2]
    n_samples = 500 if quick else 2000
    n_warmup = 500 if quick else 2000

    p = SIGMA.shape[0]
    lb = n // nb  # block length
    n_freqs = lb // 2  # positive non-DC bins after rfft
    freqs = np.fft.rfftfreq(lb, 1.0 / DEFAULT_FS)[1:]
    true_psd = _white_noise_true_psd(SIGMA, n_freqs, fs=DEFAULT_FS)

    results = []
    os.makedirs(HERE + "/" + outdir, exist_ok=True)

    for seed in seeds:
        t0 = time.time()
        print(f"\n  → seed={seed}, N={n}, Nb={nb}, K={k}, quick={quick}")

        x = _simulate_white_noise(n, SIGMA, seed=seed)
        t_arr = np.arange(n, dtype=np.float64) / DEFAULT_FS
        ts = MultivariateTimeseries(t=t_arr, y=x)

        seed_outdir = os.path.join(
            HERE, outdir, f"seed_{seed}_N{n}_Nb{nb}_K{k}"
        )
        os.makedirs(seed_outdir, exist_ok=True)

        idata = run_mcmc(
            data=ts,
            n_knots=k,
            degree=2,
            diffMatrixOrder=2,
            n_samples=n_samples,
            n_warmup=n_warmup,
            num_chains=4,
            outdir=seed_outdir,
            verbose=False,
            target_accept_prob=0.95,
            max_tree_depth=14,
            init_from_vi=True,
            vi_steps=50_000,
            vi_guide="lowrank:16",
            vi_lr=5e-4,
            Nb=nb,
            knot_kwargs=dict(method="density"),
            alpha_delta=1.0,
            beta_delta=1.0,
            true_psd=(freqs, true_psd),
        )

        # Extract posterior PSD quantiles
        psd_group = getattr(idata, "posterior_psd", None)
        if psd_group is None or "psd_matrix_real" not in psd_group:
            print(f"  [seed={seed}] WARNING: no posterior_psd group found.")
            continue

        psd_real = np.asarray(
            psd_group["psd_matrix_real"].values, dtype=np.float64
        )
        psd_imag = np.asarray(
            psd_group["psd_matrix_imag"].values, dtype=np.float64
        )
        pcts = np.asarray(
            psd_group.coords["percentile"].values, dtype=np.float64
        )

        def _pct(arr, target):
            idx = int(np.argmin(np.abs(pcts - target)))
            return arr[idx]

        stack = np.stack(
            [
                _pct(psd_real, 5.0) + 1j * _pct(psd_imag, 5.0),
                _pct(psd_real, 50.0) + 1j * _pct(psd_imag, 50.0),
                _pct(psd_real, 95.0) + 1j * _pct(psd_imag, 95.0),
            ],
            axis=0,
        )

        detail = compute_ci_coverage_multivar_detailed(stack, true_psd)
        elapsed = time.time() - t0
        print(f"  Coverage (seed={seed}, {elapsed:.0f}s):")
        _print_detailed_coverage(detail)
        results.append({"seed": seed, **detail})

    if results:
        print("\n  Summary across seeds:")
        for key in ["overall", "diag", "offdiag_re", "offdiag_im"]:
            vals = [r[key] for r in results]
            print(f"    {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        out_json = os.path.join(HERE, outdir, "test1_results.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_json}")

    print("\n  INTERPRETATION:")
    print(
        "  If diagonal coverage << 0.90 → normalization bug in the likelihood."
    )
    print(
        "  If diagonal coverage ≈ 0.90  → the model is calibrated for flat spectra;"
    )
    print(
        "    undercoverage on VAR data comes from smoothing + leakage, not a bug."
    )


# ---------------------------------------------------------------------------
# TEST 2 – Element-wise coverage breakdown from saved .npz files
# ---------------------------------------------------------------------------


def test2_elementwise_coverage(
    *,
    npz_pattern: str = "out_var3/seed_*_short_N2048_K20/compact_ci_curves.npz",
) -> None:
    """Load saved compact_ci_curves.npz files and break down coverage by element type.

    Parameters
    ----------
    npz_pattern : str
        Glob pattern relative to this script's directory.
    """
    _print_header("TEST 2 – Element-wise coverage breakdown")

    pattern = os.path.join(HERE, npz_pattern)
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f"  No files found matching: {pattern}")
        print("  Run 3d_study.py first to generate inference data.")
        return

    print(f"  Found {len(files)} file(s).\n")

    all_results = []
    for path in files:
        label = os.path.basename(os.path.dirname(path))
        npz = np.load(path)
        stack, true_psd = _build_percentiles_stack(npz)
        detail = compute_ci_coverage_multivar_detailed(stack, true_psd)
        print(f"  {label}:")
        _print_detailed_coverage(detail, label="")
        all_results.append({"file": label, **detail})

    if len(all_results) > 1:
        print("\n  Aggregate across all files:")
        for key in ["overall", "diag", "offdiag_re", "offdiag_im"]:
            vals = [r[key] for r in all_results]
            print(
                f"    {key:12s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
            )

    print("\n  INTERPRETATION:")
    print(
        "  Large diag↔offdiag gap → Cholesky prior / leakage dominates for cross-spectra."
    )
    print(
        "  If diag ≈ 0.90 but offdiag << 0.90 → the smoothing model for θ is too tight."
    )
    print(
        "  If both diag and offdiag are low → smoothing or Whittle miscalibration."
    )

    out_json = os.path.join(HERE, "out_tests", "test2_elementwise.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_json}")


# ---------------------------------------------------------------------------
# TEST 3 – Posterior inflation factor
# ---------------------------------------------------------------------------


def test3_inflation_factor(
    *,
    npz_pattern: str = "out_var3/seed_*_short_N2048_K20/compact_ci_curves.npz",
    target_coverage: float = 0.90,
) -> None:
    """Find the CI inflation factor c that achieves ``target_coverage``.

    c > 1 means the Whittle posterior CIs are too narrow by factor c.
    c ≈ 1.1 is modest miscalibration; c > 1.5 is severe.

    Parameters
    ----------
    npz_pattern : str
        Glob pattern for compact_ci_curves.npz files.
    target_coverage : float
        Target coverage level (default 0.90).
    """
    _print_header(
        f"TEST 3 – Posterior inflation factor (target={target_coverage:.0%})"
    )

    pattern = os.path.join(HERE, npz_pattern)
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f"  No files found matching: {pattern}")
        return

    print(f"  Found {len(files)} file(s).\n")
    print(f"  {'File':<50s}  {'c':>6s}  {'Coverage':>9s}  {'Iters':>5s}")
    print(f"  {'-'*50}  {'-'*6}  {'-'*9}  {'-'*5}")

    all_c = []
    all_results = []
    for path in files:
        label = os.path.basename(os.path.dirname(path))
        npz = np.load(path)
        stack, true_psd = _build_percentiles_stack(npz)

        # Current (unadjusted) coverage
        cov_raw = compute_ci_coverage_multivar(stack, true_psd)

        result = find_posterior_inflation_factor(
            stack, true_psd, target_coverage=target_coverage
        )
        c = result["inflation_factor"]
        cov_achieved = result["achieved_coverage"]
        n_iter = result["n_iter"]
        all_c.append(c)
        all_results.append({"file": label, "raw_coverage": cov_raw, **result})
        print(f"  {label:<50s}  {c:>6.3f}  {cov_achieved:>9.4f}  {n_iter:>5d}")

    if all_c:
        print(f"\n  Inflation factor c across {len(all_c)} runs:")
        print(f"    mean ± std : {np.mean(all_c):.3f} ± {np.std(all_c):.3f}")
        print(f"    median     : {np.median(all_c):.3f}")
        print(f"    range      : [{np.min(all_c):.3f}, {np.max(all_c):.3f}]")

    print("\n  INTERPRETATION:")
    print("  c = 1.0  → perfectly calibrated posterior.")
    print("  c = 1.2  → CIs are ~17% too narrow; mild Whittle miscalibration.")
    print("  c = 1.5  → CIs are 50% too narrow; substantial miscalibration.")
    print("  c = 2.0  → posteriors need to be twice as wide; severe issue.")

    out_json = os.path.join(HERE, "out_tests", "test3_inflation.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_json}")


# ---------------------------------------------------------------------------
# TEST 4 – Nb comparison (rank-deficiency hypothesis)
# ---------------------------------------------------------------------------


def test4_nb_comparison(
    *,
    n: int = 2048,
    k: int = 20,
    nb_values: list[int] | None = None,
    seeds: list[int] | None = None,
    quick: bool = False,
) -> None:
    """Run full MCMC for multiple Nb values and compare coverage.

    For p=3 channels:
    - Nb=2 → rank-deficient Wishart (rank 2 < p=3) at each frequency
    - Nb=3 → exactly full rank (marginal case)
    - Nb=4 → safely over-determined

    Block lengths at N=2048:
    - Nb=2 → Lb=1024 (512 positive freq bins)
    - Nb=4 → Lb=512  (256 positive freq bins) — fewer bins, more DOF
    - Nb=8 → Lb=256  (128 positive freq bins)

    Parameters
    ----------
    n, k : int
        Sample size and knot count.
    nb_values : list[int]
        Nb values to test. Defaults to [2, 4, 8].
    seeds : list[int]
        Seeds to run. Defaults to [0, 1, 2, 3, 4].
    quick : bool
        If True, use reduced MCMC iterations.
    """
    _print_header("TEST 4 – Nb comparison (rank-deficiency hypothesis)")

    from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

    nb_values = nb_values or [2, 4, 8]
    seeds = seeds or [0, 1, 2, 3, 4]
    n_samples = 500 if quick else 2000
    n_warmup = 500 if quick else 2000

    print(f"  N={n}, K={k}, seeds={seeds}, Nb values={nb_values}\n")
    print(
        f"  {'Nb':>4s}  {'Lb':>6s}  {'Bins':>6s}  {'DOF/freq':>9s}  "
        f"{'Overall':>9s}  {'Diag':>7s}  {'ReOff':>7s}  {'ImOff':>7s}"
    )
    print("  " + "-" * 70)

    nb_summary: dict[int, list] = {nb: [] for nb in nb_values}

    freq_ref = np.fft.rfftfreq(n // nb_values[0], 1.0 / DEFAULT_FS)[1:]
    true_psd_ref = _var_true_psd(freq_ref)

    for nb in nb_values:
        lb = n // nb
        n_freq_bins = lb // 2
        freq = np.fft.rfftfreq(lb, 1.0 / DEFAULT_FS)[1:]
        true_psd = _var_true_psd(freq)

        for seed in seeds:
            t0 = time.time()
            outdir = os.path.join(
                HERE, f"out_tests/test4_nb/Nb{nb}/seed_{seed}_N{n}_K{k}"
            )
            os.makedirs(outdir, exist_ok=True)

            x = _simulate_var(n, SIGMA, VAR_COEFFS, seed=seed)
            t_arr = np.arange(n, dtype=np.float64) / DEFAULT_FS
            ts = MultivariateTimeseries(t=t_arr, y=x)

            idata = run_mcmc(
                data=ts,
                n_knots=k,
                degree=2,
                diffMatrixOrder=2,
                n_samples=n_samples,
                n_warmup=n_warmup,
                num_chains=4,
                outdir=outdir,
                verbose=False,
                target_accept_prob=0.95,
                max_tree_depth=14,
                init_from_vi=True,
                vi_steps=50_000,
                vi_guide="lowrank:16",
                vi_lr=5e-4,
                Nb=nb,
                knot_kwargs=dict(method="density"),
                alpha_delta=1.0,
                beta_delta=1.0,
                true_psd=(freq, true_psd),
            )

            psd_group = getattr(idata, "posterior_psd", None)
            if psd_group is None or "psd_matrix_real" not in psd_group:
                print(f"  [Nb={nb}, seed={seed}] WARNING: no posterior_psd.")
                continue

            psd_real = np.asarray(psd_group["psd_matrix_real"].values)
            psd_imag = np.asarray(psd_group["psd_matrix_imag"].values)
            pcts = np.asarray(psd_group.coords["percentile"].values)

            def _pct(arr, t):
                return arr[int(np.argmin(np.abs(pcts - t)))]

            stack = np.stack(
                [
                    _pct(psd_real, 5.0) + 1j * _pct(psd_imag, 5.0),
                    _pct(psd_real, 50.0) + 1j * _pct(psd_imag, 50.0),
                    _pct(psd_real, 95.0) + 1j * _pct(psd_imag, 95.0),
                ],
                axis=0,
            )
            detail = compute_ci_coverage_multivar_detailed(stack, true_psd)
            elapsed = time.time() - t0
            nb_summary[nb].append(detail)

            print(
                f"  {nb:>4d}  {lb:>6d}  {n_freq_bins:>6d}  {nb:>9d}  "
                f"{detail['overall']:>9.4f}  {detail['diag']:>7.4f}  "
                f"{detail['offdiag_re']:>7.4f}  {detail['offdiag_im']:>7.4f}"
                f"  [{elapsed:.0f}s, seed={seed}]"
            )

    print("\n  Mean across seeds:")
    print(
        f"  {'Nb':>4s}  {'Overall':>9s}  {'Diag':>7s}  {'ReOff':>7s}  {'ImOff':>7s}"
    )
    print("  " + "-" * 50)
    summary_rows = []
    for nb in nb_values:
        runs = nb_summary[nb]
        if not runs:
            continue
        row = {
            "Nb": nb,
            "overall": float(np.mean([r["overall"] for r in runs])),
            "diag": float(np.mean([r["diag"] for r in runs])),
            "offdiag_re": float(np.mean([r["offdiag_re"] for r in runs])),
            "offdiag_im": float(np.mean([r["offdiag_im"] for r in runs])),
        }
        summary_rows.append(row)
        print(
            f"  {nb:>4d}  {row['overall']:>9.4f}  {row['diag']:>7.4f}  "
            f"{row['offdiag_re']:>7.4f}  {row['offdiag_im']:>7.4f}"
        )

    print("\n  INTERPRETATION:")
    print(
        "  If coverage increases sharply from Nb=2→Nb=4 → rank-deficiency matters."
    )
    print(
        "  If Nb makes little difference → Whittle miscalibration / smoothing dominates."
    )

    out_json = os.path.join(HERE, "out_tests", "test4_nb_comparison.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "per_run": {str(nb): nb_summary[nb] for nb in nb_values},
                "summary": summary_rows,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {out_json}")


# ---------------------------------------------------------------------------
# TEST 5 – Window comparison (Hann vs rectangular)
# ---------------------------------------------------------------------------


def test5_window_comparison(
    *,
    n: int = 2048,
    nb: int = 2,
    k: int = 20,
    seeds: list[int] | None = None,
    quick: bool = False,
) -> None:
    """Compare coverage for Hann vs rectangular (no-taper) window.

    The Hann window has ENBW ≈ 1.5 frequency bins, meaning adjacent
    periodogram ordinates are correlated. The Whittle likelihood ignores this
    correlation and effectively overstates the effective DOF per frequency bin
    by a factor of ≈ 1.5.  A rectangular window (ENBW = 1.0) avoids this
    correlation but introduces more spectral leakage.

    If coverage is substantially higher with the rectangular window, it
    confirms that taper-induced correlation is a significant factor.

    Note: this test requires the ``wishart_window`` kwarg added to
    ``run_mcmc`` in the recent code changes.

    Parameters
    ----------
    n, nb, k : int
        Sample size, number of blocks, number of knots.
    seeds : list[int]
        Seeds. Defaults to [0, 1, 2, 3, 4].
    quick : bool
        If True, use reduced MCMC iterations.
    """
    _print_header("TEST 5 – Window comparison (Hann vs rectangular)")

    from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

    seeds = seeds or [0, 1, 2, 3, 4]
    n_samples = 500 if quick else 2000
    n_warmup = 500 if quick else 2000

    window_configs = [
        ("hann", "hann", 1.5),  # (label, window_arg, ENBW)
        ("rect", None, 1.0),
    ]

    print(f"  N={n}, Nb={nb}, K={k}, seeds={seeds}\n")
    print(
        f"  {'Window':<8s}  {'ENBW':>5s}  {'Overall':>9s}  {'Diag':>7s}  "
        f"{'ReOff':>7s}  {'ImOff':>7s}"
    )
    print("  " + "-" * 55)

    lb = n // nb
    freq = np.fft.rfftfreq(lb, 1.0 / DEFAULT_FS)[1:]
    true_psd = _var_true_psd(freq)

    window_summary: dict[str, list] = {lbl: [] for lbl, _, _ in window_configs}

    for lbl, window_arg, enbw in window_configs:
        for seed in seeds:
            t0 = time.time()
            outdir = os.path.join(
                HERE,
                f"out_tests/test5_window/{lbl}/seed_{seed}_N{n}_Nb{nb}_K{k}",
            )
            os.makedirs(outdir, exist_ok=True)

            x = _simulate_var(n, SIGMA, VAR_COEFFS, seed=seed)
            t_arr = np.arange(n, dtype=np.float64) / DEFAULT_FS
            ts = MultivariateTimeseries(t=t_arr, y=x)

            idata = run_mcmc(
                data=ts,
                n_knots=k,
                degree=2,
                diffMatrixOrder=2,
                n_samples=n_samples,
                n_warmup=n_warmup,
                num_chains=4,
                outdir=outdir,
                verbose=False,
                target_accept_prob=0.95,
                max_tree_depth=14,
                init_from_vi=True,
                vi_steps=50_000,
                vi_guide="lowrank:16",
                vi_lr=5e-4,
                Nb=nb,
                wishart_window=window_arg,
                knot_kwargs=dict(method="density"),
                alpha_delta=1.0,
                beta_delta=1.0,
                true_psd=(freq, true_psd),
            )

            psd_group = getattr(idata, "posterior_psd", None)
            if psd_group is None or "psd_matrix_real" not in psd_group:
                print(
                    f"  [window={lbl}, seed={seed}] WARNING: no posterior_psd."
                )
                continue

            psd_real = np.asarray(psd_group["psd_matrix_real"].values)
            psd_imag = np.asarray(psd_group["psd_matrix_imag"].values)
            pcts = np.asarray(psd_group.coords["percentile"].values)

            def _pct(arr, t):
                return arr[int(np.argmin(np.abs(pcts - t)))]

            stack = np.stack(
                [
                    _pct(psd_real, 5.0) + 1j * _pct(psd_imag, 5.0),
                    _pct(psd_real, 50.0) + 1j * _pct(psd_imag, 50.0),
                    _pct(psd_real, 95.0) + 1j * _pct(psd_imag, 95.0),
                ],
                axis=0,
            )
            detail = compute_ci_coverage_multivar_detailed(stack, true_psd)
            window_summary[lbl].append(detail)
            elapsed = time.time() - t0
            print(
                f"  {lbl:<8s}  {enbw:>5.1f}  {detail['overall']:>9.4f}  "
                f"{detail['diag']:>7.4f}  {detail['offdiag_re']:>7.4f}  "
                f"{detail['offdiag_im']:>7.4f}  [seed={seed}, {elapsed:.0f}s]"
            )

    print("\n  Mean across seeds:")
    print(
        f"  {'Window':<8s}  {'ENBW':>5s}  {'Overall':>9s}  {'Diag':>7s}  "
        f"{'ReOff':>7s}  {'ImOff':>7s}"
    )
    print("  " + "-" * 55)
    summary_rows = []
    for lbl, _, enbw in window_configs:
        runs = window_summary[lbl]
        if not runs:
            continue
        row = {
            "window": lbl,
            "enbw": enbw,
            "overall": float(np.mean([r["overall"] for r in runs])),
            "diag": float(np.mean([r["diag"] for r in runs])),
            "offdiag_re": float(np.mean([r["offdiag_re"] for r in runs])),
            "offdiag_im": float(np.mean([r["offdiag_im"] for r in runs])),
        }
        summary_rows.append(row)
        print(
            f"  {lbl:<8s}  {enbw:>5.1f}  {row['overall']:>9.4f}  "
            f"{row['diag']:>7.4f}  {row['offdiag_re']:>7.4f}  "
            f"{row['offdiag_im']:>7.4f}"
        )

    print("\n  INTERPRETATION:")
    print("  Hann→rect coverage gain ≈ effect of taper-induced correlation.")
    print(
        "  Rect should give coverage closer to 0.90 if ENBW is the dominant issue."
    )
    print("  Rect may show slightly higher RIAE due to spectral leakage.")

    out_json = os.path.join(HERE, "out_tests", "test5_window_comparison.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {"per_run": window_summary, "summary": summary_rows}, f, indent=2
        )
    print(f"\n  Results saved to {out_json}")


# ---------------------------------------------------------------------------
# TEST 6 – White-noise calibration across channel dimensions
# ---------------------------------------------------------------------------


def test_whitenoise_dim(
    *,
    n: int = 2048,
    nb: int = 4,
    k: int = 20,
    dims: list[int] | None = None,
    n_seeds: int = 100,
    wishart_window: str | tuple | None = None,  # None = rectangular
    outdir: str = "out_tests/test_whitenoise_dim",
    quick: bool = False,
) -> None:
    """White-noise calibration study across channel dimension p.

    Generates i.i.d. Gaussian data (flat, known PSD) for p=2 and p=3
    channels, runs the full MCMC pipeline with **equidistant knots**, and
    reports element-wise coverage.

    Because the true PSD is perfectly flat, the P-spline model is exactly
    specified regardless of K.  Any systematic coverage below 0.90 is
    therefore attributable solely to the Whittle likelihood approximation
    (pseudo-likelihood curvature) and/or the Nb rank condition, NOT to model
    mis-specification.

    Comparing p=2 vs p=3 coverage isolates whether the third Cholesky block
    (theta_{20}, theta_{21}) introduces additional miscalibration or a bug.

    Parameters
    ----------
    n, nb, k : int
        Sample size, blocks, knots.
    dims : list[int]
        Channel dimensions to test.  Defaults to [2, 3].
    n_seeds : int
        Number of Monte Carlo replications per dimension.
    wishart_window : str or None
        Taper applied before FFT.  None = rectangular (default); 'hann' for
        Hann window.
    outdir : str
        Output directory for per-seed inference data.
    quick : bool
        Use 500+500 MCMC samples for fast debugging.
    """
    _print_header("TEST – White-noise calibration across channel dimension p")

    from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

    dims = dims or [2, 3]
    n_samples = 500 if quick else 2000
    n_warmup = 500 if quick else 2000
    if wishart_window is None:
        window_label = "rect"
    elif isinstance(wishart_window, tuple):
        window_label = "_".join(
            str(v) for v in wishart_window
        )  # e.g. "tukey_0.1"
    else:
        window_label = wishart_window

    print(
        f"  N={n}, Nb={nb}, K={k}, window={window_label}, "
        f"n_seeds={n_seeds}, dims={dims}"
    )
    print(f"  Knot method: uniform (equidistant)\n")

    results: dict[int, list[dict]] = {p: [] for p in dims}

    for p in dims:
        # Build a p×p covariance: SIGMA_VAL on diagonal, OFF_DIAG everywhere else
        sigma_p = SIGMA_VAL * np.eye(p) + OFF_DIAG * (
            np.ones((p, p)) - np.eye(p)
        )

        # Block length and true frequency grid
        lb = n // nb
        freqs = np.fft.rfftfreq(lb, 1.0 / DEFAULT_FS)[1:]
        true_psd = _white_noise_true_psd(sigma_p, len(freqs), fs=DEFAULT_FS)

        print(f"  ── p={p}  (SIGMA = {SIGMA_VAL}·I + {OFF_DIAG}·(11ᵀ−I)) ──")

        for seed in range(n_seeds):
            t0 = time.time()

            x = _simulate_white_noise(n, sigma_p, seed=seed)
            t_arr = np.arange(n, dtype=np.float64) / DEFAULT_FS
            ts = MultivariateTimeseries(t=t_arr, y=x)

            seed_outdir = os.path.join(
                HERE,
                outdir,
                f"p{p}",
                f"seed_{seed}_N{n}_Nb{nb}_K{k}_{window_label}",
            )
            os.makedirs(seed_outdir, exist_ok=True)

            idata = run_mcmc(
                data=ts,
                n_knots=k,
                degree=2,
                diffMatrixOrder=2,
                n_samples=n_samples,
                n_warmup=n_warmup,
                num_chains=4,
                outdir=seed_outdir,
                verbose=False,
                target_accept_prob=0.95,
                max_tree_depth=14,
                init_from_vi=True,
                vi_steps=50_000,
                vi_guide="lowrank:16",
                vi_lr=5e-4,
                Nb=nb,
                wishart_window=wishart_window,
                knot_kwargs=dict(
                    method="uniform"
                ),  # equidistant – isolates likelihood
                alpha_delta=1.0,
                beta_delta=1.0,
                true_psd=(freqs, true_psd),
            )

            psd_group = getattr(idata, "posterior_psd", None)
            if psd_group is None or "psd_matrix_real" not in psd_group:
                print(
                    f"  [p={p}, seed={seed}] WARNING: no posterior_psd group."
                )
                continue

            psd_real = np.asarray(
                psd_group["psd_matrix_real"].values, dtype=np.float64
            )
            psd_imag = np.asarray(
                psd_group["psd_matrix_imag"].values, dtype=np.float64
            )
            pcts = np.asarray(
                psd_group.coords["percentile"].values, dtype=np.float64
            )

            def _pct(arr, target):
                return arr[int(np.argmin(np.abs(pcts - target)))]

            stack = np.stack(
                [
                    _pct(psd_real, 5.0) + 1j * _pct(psd_imag, 5.0),
                    _pct(psd_real, 50.0) + 1j * _pct(psd_imag, 50.0),
                    _pct(psd_real, 95.0) + 1j * _pct(psd_imag, 95.0),
                ],
                axis=0,
            )

            detail = compute_ci_coverage_multivar_detailed(stack, true_psd)
            elapsed = time.time() - t0
            results[p].append({"seed": seed, **detail})

            if seed % 10 == 0 or seed < 3:
                print(
                    f"  p={p}  seed={seed:3d}  overall={detail['overall']:.3f}  "
                    f"diag={detail['diag']:.3f}  re={detail['offdiag_re']:.3f}  "
                    f"im={detail['offdiag_im']:.3f}  [{elapsed:.0f}s]"
                )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(
        f"  White-noise summary  "
        f"(N={n}, Nb={nb}, K={k}, window={window_label}, knots=uniform)"
    )
    print("=" * 70)
    print(
        f"  {'p':>3}  {'mean':>8}  {'std':>8}  {'95% CI':>14}  "
        f"{'diag':>8}  {'re_off':>8}  {'im_off':>8}"
    )
    print("  " + "-" * 66)

    summary_rows = []
    for p in dims:
        runs = results[p]
        if not runs:
            print(f"  {p:>3}  (no results)")
            continue
        ov = [r["overall"] for r in runs]
        dg = [r["diag"] for r in runs]
        re = [r["offdiag_re"] for r in runs]
        im = [r["offdiag_im"] for r in runs]
        se = np.std(ov) / np.sqrt(len(ov))
        ci = f"[{np.mean(ov)-1.96*se:.3f}, {np.mean(ov)+1.96*se:.3f}]"
        print(
            f"  {p:>3}  {np.mean(ov):>8.4f}  {np.std(ov):>8.4f}  {ci:>14}  "
            f"{np.mean(dg):>8.4f}  {np.mean(re):>8.4f}  {np.mean(im):>8.4f}"
        )
        summary_rows.append(
            {
                "p": p,
                "n_seeds": len(runs),
                "mean": float(np.mean(ov)),
                "std": float(np.std(ov)),
                "ci_lo": float(np.mean(ov) - 1.96 * se),
                "ci_hi": float(np.mean(ov) + 1.96 * se),
                "diag": float(np.mean(dg)),
                "offdiag_re": float(np.mean(re)),
                "offdiag_im": float(np.mean(im)),
            }
        )

    print("\n  INTERPRETATION:")
    print("  p=2 vs p=3 gap absent  → no bug in third Cholesky block.")
    print(
        "  p=2 vs p=3 gap present → bug or DOF issue in block j=2 likelihood."
    )
    print(
        "  Both below 0.90        → residual Whittle pseudo-likelihood miscalibration."
    )
    print(
        "  Both near 0.90         → model well-calibrated; VAR undercoverage = smoothing."
    )

    os.makedirs(os.path.join(HERE, outdir), exist_ok=True)
    out_json = os.path.join(
        HERE,
        outdir,
        f"whitenoise_dim_N{n}_Nb{nb}_K{k}_{window_label}_{n_seeds}seeds.json",
    )
    with open(out_json, "w") as f:
        json.dump(
            {
                "summary": summary_rows,
                "per_seed": {str(p): results[p] for p in dims},
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {out_json}")


# ---------------------------------------------------------------------------
# BONUS: Quick diagnostic on existing .npz files (no MCMC needed)
# ---------------------------------------------------------------------------


def quick_diagnostics(
    *,
    npz_pattern: str = "out_var3/seed_*_short_N2048_K20/compact_ci_curves.npz",
) -> None:
    """Run tests 2 and 3 together on saved results (fast, no MCMC)."""
    _print_header(
        "QUICK DIAGNOSTICS (Tests 2 + 3 combined, from saved .npz files)"
    )
    test2_elementwise_coverage(npz_pattern=npz_pattern)
    test3_inflation_factor(npz_pattern=npz_pattern)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_window(raw: str) -> str | tuple | None:
    """Convert a CLI window string to the form expected by compute_wishart.

    Examples
    --------
    "rect"       -> None          (rectangular, no taper)
    "hann"       -> "hann"
    "tukey_0.1"  -> ('tukey', 0.1)
    "tukey_0.5"  -> ('tukey', 0.5)
    """
    raw = raw.strip().lower()
    if raw in ("rect", "none", ""):
        return None
    if raw.startswith("tukey_"):
        try:
            alpha = float(raw.split("_", 1)[1])
        except ValueError:
            raise ValueError(
                f"Cannot parse Tukey alpha from '{raw}'. Use e.g. 'tukey_0.1'."
            )
        return ("tukey", alpha)
    return raw  # e.g. "hann"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "test",
        choices=[
            "all",
            "test1",
            "test2",
            "test3",
            "test4",
            "test5",
            "quick",
            "whitenoise_dim",
        ],
        help=(
            "Which test to run: "
            "test1=white-noise calibration (3-chan), "
            "test2=element-wise coverage, "
            "test3=inflation factor, "
            "test4=Nb comparison, "
            "test5=window comparison, "
            "whitenoise_dim=white-noise p=2 vs p=3 (100 seeds, uniform knots), "
            "quick=tests2+3 only (no MCMC), "
            "all=all five tests."
        ),
    )
    p.add_argument(
        "--glob",
        default="out_var3/seed_*_short_N2048_K20/compact_ci_curves.npz",
        help="Glob pattern for compact_ci_curves.npz (tests 2 & 3).",
    )
    p.add_argument(
        "--n", type=int, default=2048, help="Sample size (tests 1, 4, 5)."
    )
    p.add_argument("--nb", type=int, default=4, help="Nb for test1 and test5.")
    p.add_argument(
        "--nb-values",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Nb values to compare in test4.",
    )
    p.add_argument(
        "--k", type=int, default=20, help="Number of P-spline knots."
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run. Default: [0,1,2] for test1, [0-4] for tests 4 & 5.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced MCMC iterations (500 warmup + 500 samples).",
    )
    p.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Channel dimensions for whitenoise_dim test (default: 2 3).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=100,
        help="Number of MC seeds for whitenoise_dim test (default: 100).",
    )
    p.add_argument(
        "--window",
        type=str,
        default="rect",
        help="Taper for whitenoise_dim: 'hann' or 'rect' (default: rect).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.test in ("test1", "all"):
        test1_white_noise_calibration(
            n=args.n,
            nb=args.nb,
            k=args.k,
            seeds=args.seeds,
            quick=args.quick,
        )
    if args.test in ("test2", "all", "quick"):
        test2_elementwise_coverage(npz_pattern=args.glob)
    if args.test in ("test3", "all", "quick"):
        test3_inflation_factor(npz_pattern=args.glob)
    if args.test in ("test4", "all"):
        test4_nb_comparison(
            n=args.n,
            k=args.k,
            nb_values=args.nb_values,
            seeds=args.seeds,
            quick=args.quick,
        )
    if args.test in ("test5", "all"):
        test5_window_comparison(
            n=args.n,
            nb=args.nb,
            k=args.k,
            seeds=args.seeds,
            quick=args.quick,
        )
    if args.test == "whitenoise_dim":
        ww = _parse_window(args.window)
        test_whitenoise_dim(
            n=args.n,
            nb=args.nb,
            k=args.k,
            dims=args.dims,
            n_seeds=args.n_seeds,
            wishart_window=ww,
            quick=args.quick,
        )


if __name__ == "__main__":
    main()
