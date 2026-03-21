from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    MultivariateTimeseries,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig


def _simulate_var2_3d(n: int, *, seed: int) -> np.ndarray:
    a1 = np.diag([0.4, 0.3, 0.2])
    a2 = np.array(
        [[-0.2, 0.5, 0.0], [0.4, -0.1, 0.0], [0.0, 0.0, -0.1]],
        dtype=np.float64,
    )
    sigma = np.array(
        [[0.25, 0.0, 0.08], [0.0, 0.25, 0.08], [0.08, 0.08, 0.25]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    burn = 128
    noise = rng.multivariate_normal(np.zeros(3), sigma, size=n + burn)
    x = np.zeros((n + burn, 3), dtype=np.float64)
    for t in range(2, n + burn):
        x[t] = noise[t] + a1 @ x[t - 1] + a2 @ x[t - 2]
    return x[burn:]


def _base_vi_config(refine_steps: int) -> VIConfig:
    return VIConfig(
        vi_steps=40,
        vi_lr=1e-2,
        vi_posterior_draws=16,
        coarse_grain_config_vi=CoarseGrainConfig(enabled=True, Nh=4, Nc=None),
        coarse_vi_fine_refine_steps=refine_steps,
        coarse_vi_fine_refine_guide="diag",
    )


def _run_worker(scenario: str, refine_steps: int) -> dict[str, Any]:
    if scenario == "univar":
        data = ARData(order=2, duration=1.0, fs=128, sigma=0.5, seed=2).ts
        config = RunMCMCConfig(
            n_samples=8,
            n_warmup=8,
            num_chains=1,
            model=ModelConfig(n_knots=5),
            diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
            vi=_base_vi_config(refine_steps),
        )
    elif scenario == "multivar":
        n = 128
        data = MultivariateTimeseries(
            t=np.arange(n, dtype=np.float64),
            y=_simulate_var2_3d(n, seed=11),
        )
        config = RunMCMCConfig(
            n_samples=4,
            n_warmup=4,
            num_chains=1,
            Nb=2,
            model=ModelConfig(n_knots=5),
            diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
            vi=_base_vi_config(refine_steps),
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    t0 = time.perf_counter()
    idata = run_mcmc(data, config=config)
    elapsed = time.perf_counter() - t0

    return {
        "scenario": scenario,
        "refine_steps": refine_steps,
        "seconds": elapsed,
        "coarse_vi_success": int(idata.attrs.get("coarse_vi_success", 0)),
        "coarse_vi_attempted": int(idata.attrs.get("coarse_vi_attempted", 0)),
    }


def _run_case_subprocess(
    *,
    module_path: Path,
    scenario: str,
    refine_steps: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        module_path.as_posix().replace("/", ".").removesuffix(".py"),
        "--worker",
        "--scenario",
        scenario,
        "--refine-steps",
        str(refine_steps),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        return {
            "scenario": scenario,
            "refine_steps": refine_steps,
            "failed": True,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "scenario": scenario,
            "refine_steps": refine_steps,
            "failed": True,
            "returncode": 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "error": "Worker produced no JSON output",
        }
    return json.loads(lines[-1])


def _summarise_case(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed_runs = [entry for entry in results if bool(entry.get("failed"))]
    successful_runs = [entry for entry in results if not bool(entry.get("failed"))]
    if not successful_runs:
        return {
            "n_runs": int(len(results)),
            "n_success": 0,
            "n_failed": int(len(failed_runs)),
            "failed": True,
            "failures": failed_runs,
        }

    seconds = np.asarray(
        [entry["seconds"] for entry in successful_runs], dtype=float
    )
    return {
        "n_runs": int(len(results)),
        "n_success": int(len(successful_runs)),
        "n_failed": int(len(failed_runs)),
        "seconds": seconds.tolist(),
        "mean_seconds": float(np.mean(seconds)),
        "median_seconds": float(np.median(seconds)),
        "min_seconds": float(np.min(seconds)),
        "max_seconds": float(np.max(seconds)),
        "coarse_vi_success_all": bool(
            all(
                int(entry.get("coarse_vi_success", 0)) == 1
                for entry in successful_runs
            )
        ),
        "failures": failed_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cold-start coarse-VI benchmark in fresh subprocesses."
    )
    parser.add_argument(
        "--scenario",
        choices=["univar", "multivar", "both"],
        default="both",
    )
    parser.add_argument(
        "--refine-steps",
        default="0,75",
        help="Comma-separated refine-step settings to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Cold-start repeats per setting.",
    )
    parser.add_argument(
        "--out",
        default="test_output/benchmarks/cold_start_coarse_vi.json",
        help="Path to JSON output report.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker:
        result = _run_worker(args.scenario, int(args.refine_steps))
        print(json.dumps(result))
        return

    refine_steps_values = [
        int(item.strip())
        for item in str(args.refine_steps).split(",")
        if item.strip()
    ]
    scenarios = (
        ["univar", "multivar"] if args.scenario == "both" else [args.scenario]
    )

    report: dict[str, Any] = {
        "repeats": int(args.repeats),
        "refine_steps": refine_steps_values,
        "results": {},
    }
    module_path = Path(__file__).resolve().relative_to(
        Path(__file__).resolve().parents[2]
    )

    for scenario in scenarios:
        scenario_results: dict[str, Any] = {}
        for refine_steps in refine_steps_values:
            runs = []
            for _ in range(int(args.repeats)):
                runs.append(
                    _run_case_subprocess(
                        module_path=module_path,
                        scenario=scenario,
                        refine_steps=refine_steps,
                    )
                )
            scenario_results[str(refine_steps)] = _summarise_case(runs)
        report["results"][scenario] = scenario_results

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
