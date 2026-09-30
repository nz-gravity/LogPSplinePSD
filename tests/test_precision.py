"""Precision regressions run in a process with X64 selected before import.

The parent pytest process can keep its usual JAX configuration. Supported
mathematical checks never become skips when the parent has X64 disabled.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/precision_comparison.py"
pytestmark = pytest.mark.precision


def run_study(mode: str, output: Path) -> dict:
    env = dict(
        os.environ, JAX_ENABLE_X64="1", JAX_DEFAULT_MATMUL_PRECISION="highest"
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", mode, "--output", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((output / "results.json").read_text())


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    return run_study("deterministic", tmp_path_factory.mktemp("precision"))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_constant_posterior_density_gradient_and_scalar_identities(
    audit, dtype
):
    # The study checks IG Jacobians/gradients against analytic expressions,
    # real/complex conventions, constant-covariance pooling and finite masks.
    assert audit["deterministic"]["analytic"][dtype] == "passed"


@pytest.mark.parametrize("case", ["ordinary", "larger_grid"])
def test_complete_unconstrained_target_and_reconstruction(audit, case):
    checked = audit["deterministic"][case]
    assert checked["classification"] == "within_budget"
    assert checked["local_total"] < 0.01
    assert checked["gradient_abs"] < 5e-4
    assert checked["gradient_scaled"] < 5e-6
    assert checked["reconstruction_relative"] < 1e-6
    assert set(checked["gradient_dtypes"]["fp32"].values()) == {"float32"}


def test_float32_limitations_are_retained_as_stress_cases(audit):
    # These tests characterize unsupported numerical regimes; they do not
    # certify high-count or extreme-smoothing float32 inference.
    stress = audit["deterministic"]
    assert stress["high_count"]["exact_count"] == 2**24 + 1
    assert stress["high_count"]["float32_count"] == 2**24
    for case in ("high_count", "strong_smoothing"):
        assert stress[case]["classification"] == "outside_budget"
        assert stress[case]["local_total"] > 0.01
    assert stress["physical_normalization"] == "passed"
    assert stress["raw_tiny_power"] == "underflow"


@pytest.mark.slow
def test_float32_only_sampler_state_executes(tmp_path):
    result = run_study("smoke", tmp_path)
    for run in result["runs"]:
        dtype = "float32" if run["policy"] == "fp32" else "float64"
        assert run["target_dtype"] == dtype
        assert run["compiled_float_types"] == [
            "f32" if dtype == "float32" else "f64"
        ]
        assert set(run["sample_dtypes"].values()) == {dtype}
        floating = {
            x
            for x in run["adapted_state_dtypes"].values()
            if x.startswith("float")
        }
        assert floating == {dtype}
    # A 30/40 run is execution coverage, and must not certify equivalence.
    assert all(
        not c["diagnostics_satisfactory"] for c in result["comparisons"]
    )


@pytest.mark.slow
def test_sampled_analytic_posterior_matches_independent_moments(tmp_path):
    result = run_study("analytic", tmp_path)
    for run in result["runs"]:
        assert run["divergences"] == 0
        for summary in run["summaries"].values():
            assert summary["rhat"] < 1.01
            assert min(summary["ess_bulk"], summary["ess_tail"]) > 400
            # Six MCSE protects the fixed-seed regression across platforms.
            # This is an analytic calibration check, not the tighter pilot
            # equivalence test, whose intervals/classifications are saved.
            assert abs(summary["mean"] - summary["truth"]) < (
                6 * summary["mcse_mean"] + 0.005 * summary["analytic_sd"]
            )
            assert abs(summary["sd"] - summary["analytic_sd"]) < (
                6 * summary["mcse_sd"] + 0.005 * summary["analytic_sd"]
            )
