"""Fixed-target optimization experiment using the existing fit_vi call path.

Load saved observations and reuse the accepted reference. Each job runs in a
separate process; complete jobs are retained on resume, and failures are saved.
No new observations, reference inference, parameterization or guide families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from inspect import getsource
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import optax
import xarray as xr
from run import (
    Target,
    features,
    mmd_comparison,
    packed_reference,
    posterior_geometry,
    write_json,
)

from log_psplines.basis import SplineBasis
from log_psplines.config import PowerConfig
from log_psplines.data.spectral import PowerData
from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    VIDiagnosticState,
    compare_features,
    evaluate_objective,
    fingerprint,
    packed_log_densities,
    rebuild_guide,
    require_same_target,
    runtime_provenance,
)
from log_psplines.inference.power import prepare_power_model
from log_psplines.inference.vi import fit_vi
from log_psplines.models.spectrum import LogPSpline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "runs/vi-nuts-validation-v2/exact_time_varying"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_target(directory):
    """Rebuild from archived arrays, without calling the data generator."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["target"] != "exact_time_varying":
        raise ValueError(
            "only the frozen exact time-varying target is allowed"
        )
    arrays = np.load(directory / "target.npz")
    data = PowerData(
        arrays["power"],
        arrays["counts"],
        arrays["frequency"],
        arrays["time"],
        manifest["descriptor"]["units"],
    )
    spline = LogPSpline(
        SplineBasis.from_grid(
            data.frequency, interior_knots=arrays["knots_frequency"][4:-4]
        ),
        time=SplineBasis.from_grid(
            data.time, interior_knots=arrays["knots_time"][4:-4]
        ),
    )
    for axis in ("time", "frequency"):
        np.testing.assert_array_equal(
            getattr(spline, axis).basis, arrays[f"basis_{axis}"]
        )
        np.testing.assert_array_equal(
            getattr(spline, axis).penalty, arrays[f"penalty_{axis}"]
        )
    config = PowerConfig(progress_bar=False)
    model, init, pair = prepare_power_model(data, spline, config)
    identity = fingerprint(
        data.power,
        data.counts,
        data.time,
        data.frequency,
        spline.time.basis,
        spline.frequency.basis,
        pair,
        {
            name: getattr(config, name)
            for name in (
                "roughness_scale",
                "null_precision",
                "ridge_eps",
                "centered",
            )
        },
        data.units,
        getsource(prepare_power_model),
    )
    require_same_target(identity, manifest["target_fingerprint"])
    return Target(
        model,
        spline,
        data,
        config,
        pair,
        init,
        identity,
        manifest["descriptor"],
    )


def freeze_reference(reference, out, cfg, config_path):
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "frozen"
    frozen.mkdir(exist_ok=True)
    filenames = (
        "manifest.json",
        "target.npz",
        "truth.npz",
        "reference.json",
        "reference_posterior.nc",
        "reference_sample_stats.nc",
        "reference_features.npz",
    )
    hashes = {name: sha256(reference / name) for name in filenames}
    if (out / "frozen_contract.json").exists():
        contract = json.loads((out / "frozen_contract.json").read_text())
        if contract["config"] != cfg or contract["file_sha256"] != hashes:
            raise ValueError(
                "refusing to change frozen inputs/config on resume"
            )
        for name in filenames:
            if sha256(frozen / name) != hashes[name]:
                raise ValueError(f"frozen artifact changed: {name}")
        return contract
    health = json.loads((reference / "reference.json").read_text())
    if health["status"] != "accepted_screen":
        raise ValueError("accepted reference required")
    target = load_target(reference)
    require_same_target(target.identity, health["target_fingerprint"])
    # Verify the current prepared density at previously saved unconstrained
    # points. This includes likelihood, prior factors and support Jacobians.
    old = VIDiagnosticState.load(reference / "diag_seed7101")
    guide = rebuild_guide(
        old, target.model, target_fingerprint=target.identity
    )
    points = old.arrays["seed_8101_packed"][:128]
    p, _ = jax.jit(
        jax.vmap(
            lambda point: packed_log_densities(
                target.model, guide, old.params, point
            )
        )
    )(points)
    saved = old.arrays["seed_8101_log_joint"][:128]
    np.testing.assert_allclose(p, saved, rtol=1e-12, atol=1e-8)
    for name in filenames:
        shutil.copyfile(reference / name, frozen / name)
    shutil.copyfile(config_path, out / "optimization.toml")
    contract = {
        "target": "exact_time_varying",
        "target_fingerprint": target.identity,
        "source_reference": str(reference.resolve()),
        "file_sha256": hashes,
        "config": cfg,
        "density_replay_points": len(points),
        "density_max_absolute_difference": float(np.max(np.abs(p - saved))),
        "reference_status": health["status"],
        "guide_family": cfg["guide"],
        "parameterization": "noncentered",
        "provenance": runtime_provenance(),
        "reference_inference_jobs": 0,
        "new_observation_generations": 0,
    }
    write_json(out / "frozen_contract.json", contract)
    return contract


def settings(cfg):
    return [
        (schedule, particles, seed)
        for schedule in cfg["schedules"]
        for particles in cfg["optimization_particles"]
        for seed in cfg["optimization_seeds"]
    ]


def job_name(schedule, particles, seed):
    return f"{schedule['name']}_p{particles}_seed{seed}"


def learning_rate(schedule, steps):
    if schedule["type"] == "constant":
        return schedule["peak_lr"]
    if schedule["type"] == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=schedule["initial_lr"],
            peak_value=schedule["peak_lr"],
            warmup_steps=schedule["warmup_steps"],
            decay_steps=schedule.get("decay_steps", steps),
            end_value=schedule["end_lr"],
        )
    raise ValueError(f"unknown schedule: {schedule}")


def events_for(names):
    events = []
    for index, name in enumerate(names):
        if name.startswith("D_"):
            events.extend(
                (name + f"_below{x}", index, x) for x in (0.005, 0.02, 0.08)
            )
        elif name == "time_contrast_logV_f.25":
            events.append(("positive_time_contrast", index, 0.0))
    return events


def draw_dataset(guide, params, key, count):
    packed = guide.get_posterior(params).sample(key, (count,))
    samples = guide._unpack_and_constrain(packed, params)
    dataset = xr.Dataset(
        {
            name: xr.DataArray(
                np.asarray(value)[None],
                dims=(
                    "chain",
                    "draw",
                    *[f"{name}_dim{i}" for i in range(np.ndim(value) - 1)],
                ),
            )
            for name, value in samples.items()
            if not name.startswith("log_likelihood")
        }
    )
    return np.asarray(packed), dataset


def run_job(out, cfg, schedule, particles, seed):
    directory = out / job_name(schedule, particles, seed)
    if (directory / "complete.json").exists():
        return
    if directory.exists():
        raise ValueError(f"refusing to overwrite incomplete job: {directory}")
    directory.mkdir()
    started = perf_counter()
    manifest = {
        "schedule": schedule,
        "optimization_particles": particles,
        "optimization_seed": seed,
        "status": "running",
        "provenance": runtime_provenance(),
    }
    write_json(directory / "manifest.json", manifest)
    try:
        target = load_target(out / "frozen")
        reference = xr.load_dataset(out / "frozen/reference_posterior.nc")
        saved = np.load(out / "frozen/reference_features.npz")
        p, names = saved["values"], saved["names"].tolist()
        health = json.loads((out / "frozen/reference.json").read_text())
        if health["status"] != "accepted_screen":
            raise ValueError("reference is not accepted")
        require_same_target(target.identity, health["target_fingerprint"])
        diagnostics = VIDiagnosticConfig(
            seeds=tuple(cfg["diagnostic_seeds"]),
            num_particles=cfg["diagnostic_particles"],
            chunk_size=cfg["diagnostic_chunk"],
            evaluation_seeds=tuple(cfg["evaluation_seeds"]),
            evaluation_particles=cfg["evaluation_particles"],
            checkpoint_steps=tuple(cfg["checkpoints"]),
            target_fingerprint=target.identity,
            parameterization="noncentered",
        )
        rate = learning_rate(schedule, max(cfg["checkpoints"]))
        before = perf_counter()
        vi = fit_vi(
            target.model,
            rng_key=jax.random.PRNGKey(seed),
            vi_steps=max(cfg["checkpoints"]),
            optimizer_lr=rate,
            optimizer_lr_metadata=schedule,
            optimization_particles=particles,
            guide=cfg["guide"],
            posterior_draws=cfg["posterior_draws"],
            init_values=target.init,
            early_stopping=False,
            diagnostics=diagnostics,
        )
        fit_seconds = perf_counter() - before
        vi.diagnostics.save(directory)
        vi.posterior.to_netcdf(directory / "posterior.nc", engine="h5netcdf")
        guide = rebuild_guide(
            vi.diagnostics, target.model, target_fingerprint=target.identity
        )
        pr = packed_reference(guide, reference)
        rows = []
        for step in cfg["checkpoints"]:
            before = perf_counter()
            params = vi.diagnostics.checkpoints[str(step)]
            packed, posterior = draw_dataset(
                guide,
                params,
                jax.random.PRNGKey(cfg["checkpoint_draw_seed"]),
                cfg["posterior_draws"],
            )
            q, feature_names = features(target, posterior)
            if feature_names != names:
                raise ValueError("feature definitions changed")
            comparison = compare_features(
                q,
                p,
                names=names,
                vi_fingerprint=target.identity,
                reference_fingerprint=health["target_fingerprint"],
                reference_status=health["status"],
                events=events_for(names),
            )
            distribution = guide.get_posterior(params)
            np.savez_compressed(
                directory / f"checkpoint_{step}.npz",
                packed=packed,
                features=q,
                feature_names=np.array(names),
                packed_mean=np.asarray(distribution.mean),
                packed_sd=np.sqrt(np.asarray(distribution.variance)),
            )
            rows.append(
                {
                    "step": step,
                    "comparison": comparison,
                    "fixed_objective": vi.diagnostics.metadata["optimization"][
                        "checkpoint_objectives"
                    ][str(step)],
                    "independent_objective": evaluate_objective(
                        target.model,
                        guide,
                        params,
                        diagnostics,
                        seeds=cfg["diagnostic_seeds"],
                    ),
                    "functional_mmd": mmd_comparison(q, p, cfg),
                    "latent_mmd": mmd_comparison(packed[None], pr, cfg),
                    "posterior_geometry": posterior_geometry(
                        guide, params, pr
                    ),
                    "checkpoint_analysis_seconds": perf_counter() - before,
                    "learning_rate_at_last_update": float(rate(step - 1))
                    if callable(rate)
                    else float(rate),
                }
            )
        record = {
            **manifest,
            "status": "complete",
            "target_fingerprint": target.identity,
            "fit_workflow_seconds": fit_seconds,
            "wall_seconds": perf_counter() - started,
            "timings": vi.timings,
            "optimization": vi.diagnostics.metadata["optimization"],
            "native_psis": vi.diagnostics.metadata["native_psis"],
            "weights": vi.diagnostics.metadata["weights"],
            "checkpoints": rows,
            "feature_names": names,
        }
        write_json(directory / "complete.json", record)
        print(
            f"completed {directory.name} in {record['wall_seconds']:.1f}s",
            flush=True,
        )
    except Exception as error:
        manifest.update(
            status="failed",
            error=repr(error),
            wall_seconds=perf_counter() - started,
        )
        write_json(directory / "failure.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("optimization.toml"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--job", help="one exact configured job name (internal)"
    )
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError("JAX_ENABLE_X64=true required")
    cfg = tomllib.loads(args.config.read_text())
    if cfg["target"] != "exact_time_varying" or cfg["guide"] != "diag":
        raise ValueError(
            "this experiment freezes the exact TV target and diagonal guide"
        )
    if args.job:
        for schedule, particles, seed in settings(cfg):
            if job_name(schedule, particles, seed) == args.job:
                run_job(args.out, cfg, schedule, particles, seed)
                return
        raise ValueError(f"unknown job {args.job}")
    freeze_reference(args.reference, args.out, cfg, args.config)
    failures = []
    for schedule, particles, seed in settings(cfg):
        name = job_name(schedule, particles, seed)
        if (args.out / name / "complete.json").exists():
            continue
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(args.config),
                "--out",
                str(args.out),
                "--job",
                name,
            ],
            check=False,
        )
        if result.returncode:
            failures.append({"job": name, "returncode": result.returncode})
    write_json(
        args.out / "dispatch.json",
        {
            "planned_jobs": len(settings(cfg)),
            "failed_jobs": failures,
            "completed_jobs": len(list(args.out.glob("*/complete.json"))),
            "reference_inference_jobs": 0,
        },
    )
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; artifacts retained")


if __name__ == "__main__":
    main()
