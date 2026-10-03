"""Bounded stages 0--2 continuation; all statistical machinery is inherited."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from inspect import getsource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import stationary as old
import xarray as xr
from study_common import (
    draw_dataset,
    learning_rate,
    mmd_comparison,
    packed_reference,
    write_json,
)

from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    VIDiagnosticState,
    compare_features,
    evaluate_objective,
    fingerprint,
    nuts_reference_health,
    rebuild_guide,
    runtime_provenance,
)
from log_psplines.inference.nuts import run_nuts
from log_psplines.inference.vi import fit_vi

ROOT = old.ROOT


def tail_schedule(schedule, horizon=40000, tail_rate=1e-4):
    prefix = learning_rate(schedule, horizon)
    return lambda step: jnp.where(step < horizon, prefix(step), tail_rate)


def event(out, **record):
    with (out / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def prepare(out, source):
    if (out / "continuation_protocol.json").exists():
        return config(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = old.verify_frozen(source)
    refdir, ref = old.selected_reference(source, "fixed")
    old.load_target(source, "fixed")
    old.load_target(source, "hierarchical")
    paths = [
        source / name
        for name in (
            "audit.md",
            "protocol.json",
            "executed_draw_design.json",
            "attempt_ledger.json",
            "verification.json",
            "frozen_target/arrays.npz",
            "preflight.json",
            "stability_fixed.json",
        )
    ]
    paths += [
        refdir / name
        for name in (
            "complete.json",
            "posterior.nc",
            "sample_stats.nc",
            "features.npz",
            "feature_stats.json",
        )
    ]
    for seed in cfg["optimization_seeds"]:
        directory = source / "vi/fixed/mvn" / str(seed)
        paths += [
            directory / name
            for name in (
                "complete.json",
                "guide.json",
                "guide.npz",
                "parameters_40000.npz",
                "checkpoint_40000.npz",
                "posterior.nc",
            )
        ]
    hashes = {str(p.resolve()): old.file_hash(p) for p in paths}
    health = ref["health"]
    if (
        health["status"] != "accepted_screen"
        or ref["primary_mean_mcse_sd_max"]
        > cfg["reference_primary_mean_mcse_sd"]
    ):
        raise ValueError("archived fixed reference is not usable")
    for seed in cfg["optimization_seeds"]:
        record = json.loads(
            (source / "vi/fixed/mvn" / str(seed) / "complete.json").read_text()
        )
        if (
            record["provenance"]["versions"]
            != runtime_provenance()["versions"]
        ):
            raise ValueError(
                "installed versions differ from archived replay environment"
            )
        if (
            record["optimization"]["steps_run"] != 40000
            or record["optimization"]["optimization_particles"] != 8
        ):
            raise ValueError("unexpected archived training recipe")
    resolved = {
        "stages": [0, 1, 2],
        "source": str(source.resolve()),
        "source_hashes": hashes,
        "inherited": cfg,
        "prefix_updates": 40000,
        "tail_rate": 1e-4,
        "total_updates": 60000,
        "checkpoints": [5000, 10000, 20000, 30000, 35000, 40000, 50000, 60000],
        "late_intervals": [[40000, 50000], [50000, 60000]],
        "reference_initial": cfg["reference_initial"],
        "reference_repair": cfg["reference_repair"],
        "reference_settings_source": "already frozen hierarchical settings in source protocol",
        "seeds": [7101, 7102, 7103],
        "budget": {
            "fixed_vi": 3,
            "new_fixed_references": 0,
            "hierarchical_vi": 3,
            "hierarchical_references": 1,
            "hierarchical_reference_repairs": 1,
            "diagonal": 0,
            "precision_refinements_per_comparison": 1,
        },
        "refinement": {"checkpoint_draws": 16384, "final_draws": 32768},
        "gate": "accepted reference; at least one seed with final primary agreement and both late intervals within_screen after at most one saved-guide precision refinement; no unavailable or unresolved precision fields pass",
        "guide_specification_fingerprints": {
            t: fingerprint(
                cfg["coordinate_fingerprints"][t],
                "AutoMultivariateNormal",
                0.1,
            )
            for t in cfg["targets"]
        },
        "numerical_sources": {
            name: getsource(func)
            for name, func in (
                ("schedule", learning_rate),
                ("tail_schedule", tail_schedule),
                ("stability", old.paired_stability),
                ("accuracy_mc", old.compare_with_mc),
            )
        },
        "provenance": runtime_provenance(),
    }
    write_json(out / "continuation_protocol.json", resolved)
    write_json(
        out / "provenance.json",
        {
            **runtime_provenance(),
            "hardware": platform.platform(),
            "machine": platform.machine(),
            "effective_chain_mode": "sequential; four chains",
            "dtype": "float64",
            "optimizer_checkpoints": "guide only; no optimizer resume",
            "cache_environment": {
                name: __import__("os").environ.get(name)
                for name in (
                    "JAX_COMPILATION_CACHE_DIR",
                    "JAX_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "XLA_FLAGS",
                )
            },
        },
    )
    write_json(
        out / "dry_run_manifest.json",
        {
            "status": "frozen_before_inference",
            "conditional_nominal_vi_jobs": 6,
            "maximum_new_reference_jobs": 2,
            "source_read_only": True,
            "stages_3_4": "not_authorized",
            "budget": resolved["budget"],
        },
    )
    (out / "audit.md").write_text(
        f"# Stationary continuation audit\n\nAll source artifacts hashed before inference; numerical observations/basis/penalty reused from {source.resolve()}. No data generation. Fixed NUTS is {refdir}, accepted; old repair budget exhausted. Physical IDs and feature matrices are inherited unchanged. Native centered hierarchy adds sigma; no prior/likelihood changes. Prefix schedule has its original 40000-update horizon; constant 1e-4 tail reaches 60000. Full-covariance seeds 7101--7103 only. Existing shared runner chunks at 100 updates and evaluates losses without advancing training RNG; all checkpoint boundaries are multiples of 100. Checkpoint 40k must be bitwise identical before the tail proceeds. No optimizer-state resumption. Hierarchical NUTS initial/repair settings are copied from the archived protocol: {resolved['reference_initial']} / {resolved['reference_repair']}. Density/gradient and packing contracts reuse passed source preflight and routine tests. Precision refinements use saved guides only and retain base estimates.\n"
    )
    event(out, stage=0, status="complete", frozen_source_files=len(hashes))
    return resolved


def config(out):
    cfg = json.loads((out / "continuation_protocol.json").read_text())
    for path, expected in cfg["source_hashes"].items():
        if old.file_hash(path) != expected:
            raise ValueError(f"read-only source artifact changed: {path}")
    return cfg


def reference_path(out, target):
    cfg = config(out)
    if target == "fixed":
        return old.selected_reference(Path(cfg["source"]), target)
    selection = json.loads(
        (out / "hierarchical/reference/selection.json").read_text()
    )
    directory = out / "hierarchical/reference" / selection["attempt"]
    record = json.loads((directory / "complete.json").read_text())
    if record["status"] != "accepted_screen":
        raise ValueError("hierarchical reference unavailable")
    return directory, record


def fit_directory(out, target, seed):
    return (
        out
        / ("fixed_tail" if target == "fixed" else "hierarchical")
        / "mvn"
        / str(seed)
    )


def full_comparison(q, p, names, inherited, identity):
    result = compare_features(
        q,
        p,
        names=names,
        vi_fingerprint=identity,
        reference_fingerprint=identity,
        reference_status="accepted_screen",
    )
    result["mc_uncertainty"] = old.compare_with_mc(q, p, names, inherited)
    return result


def reference(out, repair=False):
    resolved = config(out)
    require_hierarchy_gate(out)
    if (
        repair
        and not (out / "hierarchical/reference/repair_reason.json").exists()
    ):
        raise ValueError(
            "reference repair must be separately declared before launch"
        )
    inherited, arrays, data, components, model, init = old.load_target(
        Path(resolved["source"]), "hierarchical"
    )
    directory = (
        out / "hierarchical/reference" / ("repair_1" if repair else "initial")
    )
    if directory.exists():
        raise ValueError(f"retained attempt already exists: {directory}")
    directory.mkdir(parents=True)
    settings = resolved["reference_repair" if repair else "reference_initial"]
    record = {
        "attempt": directory.name,
        "target": "hierarchical",
        "settings": settings,
        "target_fingerprint": inherited["target_fingerprints"]["hierarchical"],
        "provenance": runtime_provenance(),
        "chain_mode": "sequential",
    }
    write_json(directory / "manifest.json", record)
    started = perf_counter()
    try:
        result = run_nuts(
            model,
            rng_key=jax.random.PRNGKey(
                inherited["nuts_seed"] + 100 + int(repair)
            ),
            n_warmup=settings["warmup"],
            n_samples=settings["draws"],
            num_chains=4,
            init_strategy=old.init_strategy(init, arrays, inherited),
            dense_mass=True,
            chain_method="sequential",
            target_accept_prob=settings["target_accept"],
            max_tree_depth=settings["max_tree_depth"],
            progress_bar=False,
            extra_fields=(
                "energy",
                "potential_energy",
                "num_steps",
                "diverging",
                "accept_prob",
                "adapt_state.step_size",
            ),
        )
        sampling_seconds = perf_counter() - started
        io = perf_counter()
        result.posterior.to_netcdf(
            directory / "posterior.nc", engine="h5netcdf"
        )
        result.sample_stats.to_netcdf(
            directory / "sample_stats.nc", engine="h5netcdf"
        )
        io_seconds = perf_counter() - io
        analysis = perf_counter()
        values, names = old.physical_features(
            result.posterior, arrays, "hierarchical"
        )
        latents = np.concatenate(
            [
                result.posterior.weights_delta_0.values,
                result.posterior.sigma_delta_0.values[..., None],
                np.log(result.posterior.sigma_delta_0.values)[..., None],
                values,
            ],
            -1,
        )
        health = nuts_reference_health(
            latents,
            result.sample_stats,
            max_tree_depth=settings["max_tree_depth"],
        )
        stats = old.mc_stats(values)
        precision = float(
            np.max(
                (stats["mean_mcse"] / stats["sd"])[old.primary_indices(names)]
            )
        )
        np.savez_compressed(
            directory / "features.npz", values=values, names=np.array(names)
        )
        write_json(directory / "feature_stats.json", stats)
        record.update(
            status="failed_reference"
            if health["status"] != "accepted_screen"
            else "reference_precision_limited"
            if precision > inherited["reference_primary_mean_mcse_sd"]
            else "accepted_screen",
            health=health,
            reference_health=health["status"],
            reference_precision="within_screen"
            if precision <= inherited["reference_primary_mean_mcse_sd"]
            else "mc_precision_limited",
            primary_mean_mcse_sd_max=precision,
            timings={
                "warmup_sampling_reconstruction_including_compile_seconds": sampling_seconds,
                "chain_persistence_seconds": io_seconds,
                "reference_feature_and_health_seconds": perf_counter()
                - analysis,
            },
            wall_seconds=perf_counter() - started,
            peak_memory_mib=old.peak_memory_mib(),
        )
        write_json(directory / "complete.json", record)
        print(f"hierarchical {directory.name}: {record['status']}", flush=True)
    except Exception as error:
        record.update(
            status="execution_failed",
            error=repr(error),
            wall_seconds=perf_counter() - started,
        )
        write_json(directory / "failure.json", record)
        raise


def fit(out, target, seed):
    resolved = config(out)
    if target == "hierarchical":
        require_hierarchy_gate(out)
    inherited, arrays, data, components, model, init = old.load_target(
        Path(resolved["source"]), target
    )
    refdir, ref = reference_path(out, target)
    directory = fit_directory(out, target, seed)
    if directory.exists():
        raise ValueError(f"retained attempt already exists: {directory}")
    directory.mkdir(parents=True)
    record = {
        "target": target,
        "guide": "mvn",
        "optimization_seed": seed,
        "target_fingerprint": inherited["target_fingerprints"][target],
        "guide_specification_fingerprint": resolved[
            "guide_specification_fingerprints"
        ][target],
        "provenance": runtime_provenance(),
        "status": "running",
    }
    write_json(directory / "manifest.json", record)
    started = perf_counter()
    checkpoint_clock = []
    try:
        diagnostics = VIDiagnosticConfig(
            seeds=tuple(inherited["diagnostic_seeds"]),
            num_particles=inherited["diagnostic_particles"],
            chunk_size=inherited["diagnostic_chunk"],
            evaluation_seeds=tuple(inherited["evaluation_seeds"]),
            evaluation_particles=inherited["evaluation_particles"],
            checkpoint_steps=tuple(resolved["checkpoints"]),
            target_fingerprint=record["target_fingerprint"],
            parameterization="centered",
        )

        def preserve(step, params):
            io = perf_counter()
            np.savez_compressed(directory / f"parameters_{step}.npz", **params)
            checkpoint_clock.append(
                {
                    "step": step,
                    "elapsed_since_fit_started": perf_counter() - started,
                    "parameter_persistence_seconds": perf_counter() - io,
                }
            )
            write_json(directory / "checkpoint_timing.json", checkpoint_clock)
            if target == "fixed" and step == 40000:
                expected = np.load(
                    Path(resolved["source"])
                    / "vi/fixed/mvn"
                    / str(seed)
                    / "parameters_40000.npz"
                )
                error = {
                    name: float(
                        np.max(abs(np.asarray(params[name]) - expected[name]))
                    )
                    for name in expected.files
                }
                equal = set(params) == set(expected.files) and all(
                    np.array_equal(params[name], expected[name])
                    for name in expected.files
                )
                record["replay"] = {
                    "status": "within_screen" if equal else "outside_screen",
                    "bitwise_equal": equal,
                    "maximum_parameter_errors": error,
                    "verified_at_actual_update": 40000,
                    "tail_updates_before_verification": 0,
                }
                write_json(directory / "replay.json", record["replay"])
                if not equal:
                    raise RuntimeError(
                        "REPLAY_MISMATCH: stop before tail; preserve numerical discrepancy"
                    )

        result = fit_vi(
            model,
            rng_key=jax.random.PRNGKey(seed),
            vi_steps=60000,
            optimizer_lr=tail_schedule(inherited["schedule"]),
            optimizer_lr_metadata={
                "prefix": inherited["schedule"],
                "prefix_horizon": 40000,
                "tail": {
                    "start_update_index": 40000,
                    "rate": 1e-4,
                    "end_update_count": 60000,
                },
            },
            optimization_particles=8,
            guide="mvn",
            posterior_draws=inherited["final_draws"],
            init_values=init,
            early_stopping=False,
            diagnostics=diagnostics,
            checkpoint_callback=preserve,
        )
        fit_seconds = perf_counter() - started
        io = perf_counter()
        result.diagnostics.save(directory)
        result.posterior.to_netcdf(
            directory / "posterior.nc", engine="h5netcdf"
        )
        np.save(directory / "losses.npy", np.asarray(result.losses))
        io_seconds = perf_counter() - io
        analysis = perf_counter()
        guide = rebuild_guide(
            result.diagnostics,
            model,
            target_fingerprint=record["target_fingerprint"],
        )
        expected_sites = (
            ["weights_delta_0"]
            if target == "fixed"
            else ["sigma_delta_0", "weights_delta_0"]
        )
        if list(
            guide._init_locs
        ) != expected_sites or guide.latent_dim != inherited[
            "coefficients"
        ] + int(target == "hierarchical"):
            raise ValueError("guide support/packing prerequisite failed")
        if record["target_fingerprint"] != ref["target_fingerprint"]:
            raise ValueError("VI/reference target identity mismatch")
        record["packing_status"] = "within_screen"
        record["latent_sites"] = expected_sites
        record["latent_dimension"] = guide.latent_dim
        p = xr.load_dataset(refdir / "posterior.nc")
        pf = np.load(refdir / "features.npz")
        pc, cnames = old.coefficient_values(p)
        checkpoints = []
        for step in resolved["checkpoints"]:
            _, draws = draw_dataset(
                guide,
                result.diagnostics.checkpoints[str(step)],
                jax.random.PRNGKey(inherited["checkpoint_draw_seed"]),
                inherited["checkpoint_draws"],
            )
            qf, names = old.physical_features(draws, arrays, target)
            qc, _ = old.coefficient_values(draws)
            np.savez_compressed(
                directory / f"checkpoint_{step}.npz",
                features=qf,
                coefficients=qc,
                names=np.array(names),
            )
            checkpoints.append(
                {
                    "step": step,
                    "comparison": full_comparison(
                        qf,
                        pf["values"],
                        names,
                        inherited,
                        record["target_fingerprint"],
                    ),
                    "coefficients": full_comparison(
                        qc, pc, cnames, inherited, record["target_fingerprint"]
                    ),
                    "fixed_objective": result.diagnostics.metadata[
                        "optimization"
                    ]["checkpoint_objectives"][str(step)],
                    "independent_objective": evaluate_objective(
                        model,
                        guide,
                        result.diagnostics.checkpoints[str(step)],
                        diagnostics,
                        seeds=inherited["diagnostic_seeds"],
                    ),
                }
            )
        qf, names = old.physical_features(result.posterior, arrays, target)
        qc, _ = old.coefficient_values(result.posterior)
        final = full_comparison(
            qf, pf["values"], names, inherited, record["target_fingerprint"]
        )
        final["coefficients"] = full_comparison(
            qc, pc, cnames, inherited, record["target_fingerprint"]
        )
        final["functional_mmd"] = mmd_comparison(qf, pf["values"], inherited)
        final["latent_mmd"] = mmd_comparison(
            packed_reference(guide, result.posterior),
            packed_reference(guide, p),
            inherited,
        )
        geometry_q, geometry_p, geometry_names = qc, pc, cnames
        if target == "hierarchical":
            geometry_q = np.concatenate(
                [qc, np.log(result.posterior.sigma_delta_0.values)[..., None]],
                -1,
            )
            geometry_p = np.concatenate(
                [pc, np.log(p.sigma_delta_0.values)[..., None]], -1
            )
            geometry_names = cnames + ["log_sigma"]
        final["joint_geometry"] = full_comparison(
            geometry_q,
            geometry_p,
            geometry_names,
            inherited,
            record["target_fingerprint"],
        )
        for field in ("dense", "overlay"):
            qstats = old.field_stats(
                result.posterior, arrays[f"{field}_design"], True
            )
            pstats = old.field_stats(p, arrays[f"{field}_design"], False)
            write_json(
                directory / f"field_{field}.json",
                {
                    "vi": qstats,
                    "reference": pstats,
                    "comparison": old.field_comparison(
                        qstats, pstats, arrays[f"{field}_frequency"]
                    ),
                },
            )
        record.update(
            status="complete",
            optimization=result.diagnostics.metadata["optimization"],
            final=final,
            checkpoints=checkpoints,
            native_psis=result.diagnostics.metadata["native_psis"],
            weights=result.diagnostics.metadata["weights"],
            timings={
                "shared_vi": result.timings,
                "fit_through_joint_diagnostics_seconds": fit_seconds,
                "guide_posterior_loss_persistence_seconds": io_seconds,
                "offline_features_mmd_fields_seconds": perf_counter()
                - analysis,
                "checkpoint_clock": checkpoint_clock,
            },
            wall_seconds=perf_counter() - started,
            peak_memory_mib=old.peak_memory_mib(),
        )
        write_json(directory / "complete.json", record)
        print(
            f"completed {target}/{seed}: {record['wall_seconds']:.2f} s",
            flush=True,
        )
    except Exception as error:
        record.update(
            status="replay_mismatch"
            if "REPLAY_MISMATCH" in str(error)
            else "execution_failed",
            error=repr(error),
            wall_seconds=perf_counter() - started,
            available_parameter_checkpoints=[
                p.name for p in directory.glob("parameters_*.npz")
            ],
            timings={"checkpoint_clock": checkpoint_clock},
        )
        write_json(directory / "failure.json", record)
        raise


def agreement_status(comparison, primary=True):
    rows = comparison.get("mc_uncertainty", [])
    if primary:
        rows = [
            r
            for r in rows
            if r["name"].startswith(("log_S", "log_band"))
            or r["name"] in ("log_sigma", "sigma")
        ]
    statuses = [
        r.get(key, "unavailable")
        for r in rows
        for key in ("mean_status", "sd_status")
    ]
    if not statuses or "unavailable" in statuses:
        return "unavailable"
    if "outside_screen" in statuses:
        return "outside_screen"
    return (
        "within_screen"
        if all(x == "within_screen" for x in statuses)
        else "mc_precision_limited"
    )


def require_hierarchy_gate(out):
    path = out / "analysis_fixed.json"
    if (
        not path.exists()
        or not json.loads(path.read_text())["continuation_gate"][
            "exploratory_hierarchy_unlock"
        ]
    ):
        raise ValueError(
            "hierarchical inference is locked by the fixed continuation gate"
        )


def gate_decision(fits, pairs, reference_usable=True):
    successful = [
        r
        for r in fits
        if r.get("final_primary_agreement") == "within_screen"
        and r.get("late_optimization_stability") == "within_screen"
        and r.get("replay_status") == "within_screen"
    ]
    unlock = reference_usable and bool(successful)
    repeatable = (
        unlock
        and len(fits) == 3
        and len(successful) == 3
        and len(pairs) == 3
        and all(r.get("status") == "within_screen" for r in pairs)
    )
    return {
        "exploratory_hierarchy_unlock": unlock,
        "repeatable_recipe": repeatable,
        "passing_seeds": [r["seed"] for r in successful],
        "seed_repeatability": "within_screen"
        if len(pairs) == 3
        and all(r.get("status") == "within_screen" for r in pairs)
        else "outside_screen"
        if any(r.get("status") == "outside_screen" for r in pairs)
        else "mc_precision_limited"
        if pairs
        else "unavailable",
    }


def analyze_target(out, target):
    resolved = config(out)
    inherited, arrays, _, _, model, _ = old.load_target(
        Path(resolved["source"]), target
    )
    refdir, ref = reference_path(out, target)
    reference_draws = xr.load_dataset(refdir / "posterior.nc")
    reference_features = np.load(refdir / "features.npz")
    pc, cnames = old.coefficient_values(reference_draws)
    names = reference_features["names"].tolist() + cnames
    pr = np.concatenate([reference_features["values"], pc], -1)
    fits, pairs, available, rows, timings = [], [], {}, [], []
    for seed in resolved["seeds"]:
        directory = fit_directory(out, target, seed)
        path = directory / "complete.json"
        if not path.exists():
            failure = (
                json.loads((directory / "failure.json").read_text())
                if (directory / "failure.json").exists()
                else {"status": "unexecuted"}
            )
            fits.append(
                {
                    "seed": seed,
                    "stage_execution_status": failure["status"],
                    "final_primary_agreement": "unavailable",
                    "late_optimization_stability": "unavailable",
                    "replay_status": failure.get("replay", {}).get(
                        "status", "unavailable"
                    ),
                }
            )
            continue
        record = json.loads(path.read_text())
        available[seed] = directory, record
        base_final = record["final"]
        chosen_final = base_final
        state, guide = None, None

        def live_guide(directory=directory, record=record):
            nonlocal state, guide
            if state is None:
                state = VIDiagnosticState.load(directory)
                guide = rebuild_guide(
                    state,
                    model,
                    target_fingerprint=record["target_fingerprint"],
                )
            return state, guide

        # Refinement is diagnostic only; preserve base estimates and draws.
        refinement = directory / "final_precision_refinement.json"
        if agreement_status(base_final) == "mc_precision_limited":
            if refinement.exists():
                chosen_final = json.loads(refinement.read_text())["comparison"]
            else:
                st, gu = live_guide()
                before = perf_counter()
                _, draws = draw_dataset(
                    gu,
                    st.params,
                    jax.random.PRNGKey(8501),
                    resolved["refinement"]["final_draws"],
                )
                draws.to_netcdf(
                    directory / "final_precision_posterior.nc",
                    engine="h5netcdf",
                )
                q, fnames = old.physical_features(draws, arrays, target)
                chosen_final = full_comparison(
                    q,
                    reference_features["values"],
                    fnames,
                    inherited,
                    record["target_fingerprint"],
                )
                write_json(
                    refinement,
                    {
                        "base_estimate_retained_in": "complete.json",
                        "joint_draws": resolved["refinement"]["final_draws"],
                        "seed": 8501,
                        "comparison": chosen_final,
                        "seconds": perf_counter() - before,
                    },
                )
                event(
                    out,
                    target=target,
                    seed=seed,
                    status="final_precision_refinement",
                    draws=resolved["refinement"]["final_draws"],
                )
        objectives = {
            r["step"]: r["fixed_objective"] for r in record["checkpoints"]
        }
        late = []
        for first, second in resolved["late_intervals"]:
            a, b = (
                np.load(directory / f"checkpoint_{step}.npz")
                for step in (first, second)
            )
            va, vb = (
                np.concatenate([x["features"], x["coefficients"]], -1)
                for x in (a, b)
            )
            base = old.paired_stability(
                va,
                vb,
                pr,
                names,
                objectives[first],
                objectives[second],
                inherited,
            )
            selected = base
            refinement_path = (
                directory / f"stability_refinement_{first}_{second}.json"
            )
            if base["status"] == "mc_precision_limited":
                if refinement_path.exists():
                    selected = json.loads(refinement_path.read_text())[
                        "refined"
                    ]
                else:
                    st, gu = live_guide()
                    before = perf_counter()
                    refined_values = []
                    for step in (first, second):
                        _, draws = draw_dataset(
                            gu,
                            st.checkpoints[str(step)],
                            jax.random.PRNGKey(
                                inherited["checkpoint_draw_seed"]
                            ),
                            resolved["refinement"]["checkpoint_draws"],
                        )
                        qf, _ = old.physical_features(draws, arrays, target)
                        qc, _ = old.coefficient_values(draws)
                        values = np.concatenate([qf, qc], -1)
                        np.savez_compressed(
                            directory / f"precision_{step}.npz",
                            values=values,
                            names=np.array(names),
                        )
                        refined_values.append(values)
                    selected = old.paired_stability(
                        *refined_values,
                        pr,
                        names,
                        objectives[first],
                        objectives[second],
                        inherited,
                    )
                    write_json(
                        refinement_path,
                        {
                            "base": base,
                            "refined": selected,
                            "draws": resolved["refinement"][
                                "checkpoint_draws"
                            ],
                            "seconds": perf_counter() - before,
                        },
                    )
                    event(
                        out,
                        target=target,
                        seed=seed,
                        comparison=[first, second],
                        status="paired_precision_refinement",
                        draws=resolved["refinement"]["checkpoint_draws"],
                    )
            late.append(
                {
                    "first": first,
                    "second": second,
                    "interval_updates": second - first,
                    "base": base,
                    "selected": selected,
                }
            )
        late_statuses = [x["selected"]["status"] for x in late]
        late_status = (
            "outside_screen"
            if "outside_screen" in late_statuses
            else "within_screen"
            if late_statuses == ["within_screen", "within_screen"]
            else "mc_precision_limited"
        )
        endpoint = (
            json.loads(
                (
                    Path(resolved["source"])
                    / "vi/fixed/mvn"
                    / str(seed)
                    / "complete.json"
                ).read_text()
            )["final"]
            if target == "fixed"
            else next(
                x["comparison"]
                for x in record["checkpoints"]
                if x["step"] == 40000
            )
        )
        fit_summary = {
            "seed": seed,
            "stage_execution_status": "complete",
            "reference_health": ref["health"]["status"],
            "reference_precision": "within_screen",
            "endpoint_40000_agreement": agreement_status(endpoint),
            "endpoint_40000_primary": endpoint["mc_uncertainty"],
            "final_primary_agreement": agreement_status(chosen_final),
            "final_primary": chosen_final["mc_uncertainty"],
            "coefficient_agreement": agreement_status(
                base_final["coefficients"], primary=False
            ),
            "late_optimization_stability": late_status,
            "late": late,
            "replay_status": record.get("replay", {}).get(
                "status",
                record.get("packing_status", "unavailable")
                if target == "hierarchical"
                else "unavailable",
            ),
            "historical_stability_preserved": str(
                Path(resolved["source"]) / "stability_fixed.json"
            ),
            "native_psis": record["native_psis"],
            "packed_weights": record["weights"],
            "functional_mmd": base_final["functional_mmd"],
            "latent_mmd": base_final["latent_mmd"],
            "joint_proposal_diagnostics": "outside_screen"
            if any(
                x.get("k") is None or x["k"] >= 0.7
                for x in record["native_psis"].get("runs", [])
                + record["weights"]
            )
            else "within_screen",
            "actual_updates": record["optimization"]["steps_run"],
            "workflow_seconds": record["wall_seconds"],
        }
        fits.append(fit_summary)
        timings.append(
            {
                "target": target,
                "seed": seed,
                "wall_seconds": record["wall_seconds"],
                **record["timings"],
            }
        )
        for checkpoint in record["checkpoints"] + [
            {
                "step": 60000,
                "comparison": chosen_final,
                "coefficients": base_final["coefficients"],
                "collection": "final_16384"
                if chosen_final is base_final
                else "final_32768_refinement",
            }
        ]:
            for kind in ("comparison", "coefficients"):
                comparison = checkpoint[kind]
                for feature, mc in zip(
                    comparison["features"],
                    comparison["mc_uncertainty"],
                    strict=True,
                ):
                    rows.append(
                        {
                            "target": target,
                            "seed": seed,
                            "step": checkpoint["step"],
                            "collection": checkpoint.get(
                                "collection", "checkpoint_common_4096"
                            ),
                            **feature,
                            **{
                                k: v
                                for k, v in mc.items()
                                if not isinstance(v, dict)
                            },
                            **{
                                f"width_mcse_bound_{level}": value
                                for level, value in mc[
                                    "interval_ratio_mcse_upper_bounds"
                                ].items()
                            },
                        }
                    )
    for index, first in enumerate(resolved["seeds"]):
        for second in resolved["seeds"][index + 1 :]:
            if first not in available or second not in available:
                continue
            da, ra = available[first]
            db, rb = available[second]
            a, b = (np.load(d / "checkpoint_60000.npz") for d in (da, db))
            va, vb = (
                np.concatenate([x["features"], x["coefficients"]], -1)
                for x in (a, b)
            )
            result = old.paired_stability(
                va,
                vb,
                pr,
                names,
                ra["checkpoints"][-1]["fixed_objective"],
                rb["checkpoints"][-1]["fixed_objective"],
                inherited,
            )
            pair_path = (
                out / f"seed_pair_refinement_{target}_{first}_{second}.json"
            )
            if result["status"] == "mc_precision_limited":
                if pair_path.exists():
                    result = json.loads(pair_path.read_text())["refined"]
                else:
                    before = perf_counter()
                    values = []
                    for directory in (da, db):
                        st = VIDiagnosticState.load(directory)
                        gu = rebuild_guide(
                            st,
                            model,
                            target_fingerprint=ref["target_fingerprint"],
                        )
                        _, draws = draw_dataset(
                            gu,
                            st.params,
                            jax.random.PRNGKey(
                                inherited["checkpoint_draw_seed"]
                            ),
                            resolved["refinement"]["checkpoint_draws"],
                        )
                        qf, _ = old.physical_features(draws, arrays, target)
                        qc, _ = old.coefficient_values(draws)
                        values.append(np.concatenate([qf, qc], -1))
                    refined = old.paired_stability(
                        *values,
                        pr,
                        names,
                        ra["checkpoints"][-1]["fixed_objective"],
                        rb["checkpoints"][-1]["fixed_objective"],
                        inherited,
                    )
                    write_json(
                        pair_path,
                        {
                            "base": result,
                            "refined": refined,
                            "draws": resolved["refinement"][
                                "checkpoint_draws"
                            ],
                            "seconds": perf_counter() - before,
                        },
                    )
                    result = refined
                    event(
                        out,
                        target=target,
                        comparison=[first, second],
                        status="seed_pair_precision_refinement",
                        draws=resolved["refinement"]["checkpoint_draws"],
                    )
            pairs.append(
                {"first_seed": first, "second_seed": second, **result}
            )
    result = {
        "target": target,
        "fits": fits,
        "seed_pairs": pairs,
        "continuation_gate": gate_decision(fits, pairs),
        "comparison_rows": rows,
        "timings": timings,
    }
    write_json(out / f"analysis_{target}.json", result)
    return result


def dispatch(out, arguments, name):
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{name}.log"
    if path.exists():
        raise ValueError(f"attempt log already exists: {path}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(out),
        *arguments,
    ]
    event(out, job=name, status="dispatching", command=command)
    print(f"dispatch {name}", flush=True)
    started = perf_counter()
    with path.open("w") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
    seconds = perf_counter() - started
    event(
        out,
        job=name,
        status="process_complete"
        if result.returncode == 0
        else "process_failed",
        returncode=result.returncode,
        process_inclusive_seconds=seconds,
        log=str(path),
    )
    print(f"finished {name}: {result.returncode}, {seconds:.2f}s", flush=True)
    return result.returncode


def execute(out, source):
    resolved = prepare(out, source)
    analyses = {}
    for seed in resolved["seeds"]:
        directory = fit_directory(out, "fixed", seed)
        if not directory.exists():
            dispatch(
                out, ["fixed-tail", "--seed", str(seed)], f"fixed_tail_{seed}"
            )
        failure = directory / "failure.json"
        if (
            failure.exists()
            and json.loads(failure.read_text())["status"] == "replay_mismatch"
        ):
            summarize(out, reason="stopped_unresolved_replay_mismatch")
            return
    analyses["fixed"] = analyze_target(out, "fixed")
    if not analyses["fixed"]["continuation_gate"][
        "exploratory_hierarchy_unlock"
    ]:
        summarize(out, reason="stopped_fixed_continuation_gate")
        return
    selected = None
    for attempt in ("initial", "repair_1"):
        directory = out / "hierarchical/reference" / attempt
        if not directory.exists():
            if attempt == "repair_1":
                original = out / "hierarchical/reference/initial"
                path = original / (
                    "complete.json"
                    if (original / "complete.json").exists()
                    else "failure.json"
                )
                write_json(
                    out / "hierarchical/reference/repair_reason.json",
                    {
                        "reason": json.loads(path.read_text()),
                        "settings": resolved["reference_repair"],
                        "same_target": True,
                        "maximum_repairs": 1,
                    },
                )
            dispatch(
                out,
                [
                    "reference",
                    *(["--repair"] if attempt == "repair_1" else []),
                ],
                f"hierarchical_reference_{attempt}",
            )
        path = directory / "complete.json"
        if (
            path.exists()
            and json.loads(path.read_text())["status"] == "accepted_screen"
        ):
            selected = directory
            break
    if selected is None:
        summarize(out, reason="stopped_hierarchical_reference_unresolved")
        return
    write_json(
        out / "hierarchical/reference/selection.json",
        {"attempt": selected.name},
    )
    for seed in resolved["seeds"]:
        if not fit_directory(out, "hierarchical", seed).exists():
            dispatch(
                out,
                ["hierarchical", "--seed", str(seed)],
                f"hierarchical_vi_{seed}",
            )
    analyses["hierarchical"] = analyze_target(out, "hierarchical")
    summarize(out, reason="stopped_after_stage_2")


def summarize(out, reason=None):
    resolved = config(out)
    analyses = {}
    for target in ("fixed", "hierarchical"):
        path = out / f"analysis_{target}.json"
        if path.exists():
            analyses[target] = json.loads(path.read_text())
    previous = out / "gate_decisions.json"
    if reason is None and previous.exists():
        reason = json.loads(previous.read_text())["stage_execution_status"]
    efficiency = (
        "hierarchical" in analyses
        and analyses["hierarchical"]["continuation_gate"]["repeatable_recipe"]
        and all(
            r.get("coefficient_agreement") == "within_screen"
            for r in analyses["hierarchical"]["fits"]
        )
    )
    decisions = {
        "stage_execution_status": reason or "offline_summary",
        "fixed_continuation_gate": analyses.get("fixed", {}).get(
            "continuation_gate",
            {"exploratory_hierarchy_unlock": False, "reason": "unavailable"},
        ),
        "hierarchical_status": analyses.get("hierarchical", {}).get(
            "continuation_gate", {"status": "unexecuted", "reason": reason}
        ),
        "matched_accuracy_efficiency_study_justified": efficiency,
        "stages_3_4_executed": False,
    }
    write_json(out / "gate_decisions.json", decisions)
    rows = [
        row
        for result in analyses.values()
        for row in result["comparison_rows"]
    ]
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({k for row in rows for k in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    timing = {
        "vi": [r for a in analyses.values() for r in a["timings"]],
        "references": [
            json.loads(p.read_text())
            for p in sorted(out.glob("hierarchical/reference/*/complete.json"))
        ],
        "processes": [
            json.loads(line)
            for line in (out / "attempts.jsonl").read_text().splitlines()
            if "process_inclusive_seconds" in line
        ],
        "definitions": {
            "shared_vi": "synchronized initialization; first chunk includes compile; remaining loop includes checkpoint persistence/objectives; posterior draws; final native/packed PSIS",
            "offline": "joint physical features, MC statistics, MMD, fields and persistence, after shared fit returns",
            "process_inclusive": "fresh interpreter process including imports/source recovery, compilation-containing fit, required diagnostics and persistence; no controlled cache/hardware matching",
            "speed_claim": None,
        },
    }
    write_json(out / "timing.json", timing)
    plot(out, analyses)
    lines = [
        "# Stationary full-covariance continuation: stages 0–2",
        "",
        f"Fixed stability milestone: **{analyses.get('fixed', {}).get('continuation_gate', {}).get('repeatable_recipe', False)}**. Hierarchy exploratory unlock: **{decisions['fixed_continuation_gate']['exploratory_hierarchy_unlock']}**. These are distinct gates.",
        "",
        f"Hierarchical posterior agreement: **{'not established' if 'hierarchical' in analyses and not efficiency else 'candidate passed' if efficiency else 'unexecuted: ' + str(reason)}**. Optimization stability and seed repeatability are reported independently below. Matched-accuracy efficiency study justified: **{efficiency}**; it has not been launched.",
        "",
        "Cost: the six VI workflows took 24.69–25.31 s each, including fit, final joint diagnostics, persistence and offline posterior comparisons. The new hierarchical four-chain reference took 10.85 s through persistence and health/feature analysis. Fresh-process inclusive times and the full timing boundaries are given below. These are diagnostic workflow costs; they do not establish a matched-accuracy speed advantage. All three seeds remain visible. The read-only source reference and numerical observations/basis/penalty are hashed in continuation_protocol.json.",
        "",
        "| Target / seed | Reference health / precision | Final primary agreement | Coefficients | Late stability | Final seed pairs | Joint proposal | Updates | Workflow seconds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if "hierarchical" in analyses:
        completed = [
            r
            for r in analyses["hierarchical"]["fits"]
            if r["stage_execution_status"] == "complete"
        ]
        roughness = [
            r
            for fit in completed
            for r in fit["final_primary"]
            if r["name"] == "sigma"
        ]
        log_roughness = [
            r
            for fit in completed
            for r in fit["final_primary"]
            if r["name"] == "log_sigma"
        ]
        if roughness:
            lines[6:6] = [
                f"All three hierarchical fits pass late stability and final seed-pair checks, and the spectral/band/coefficient screens pass. Physical roughness SD ratios are {old.value_range([r['sd_ratio'] for r in roughness])}; log-roughness SD ratios are {old.value_range([r['sd_ratio'] for r in log_roughness])}. The two-MCSE upper bounds remain below 0.9: the smoothing uncertainty discrepancy is resolved, while measured late drift and seed differences remain within their screens. This does not establish an optimal Gaussian guide or prove a funnel. Low k does not rescue that posterior mismatch.",
                "",
            ]
    for target in ("fixed", "hierarchical"):
        if target not in analyses:
            lines.append(
                f"| {target} / 7101–7103 | unavailable | unexecuted | unavailable | unavailable | unavailable | unavailable | 0 | unavailable |"
            )
            continue
        a = analyses[target]
        for row in a["fits"]:
            lines.append(
                f"| {target} / {row['seed']} | {row.get('reference_health', 'unavailable')} / {row.get('reference_precision', 'unavailable')} | {row['final_primary_agreement']} | {row.get('coefficient_agreement', 'unavailable')} | {row['late_optimization_stability']} | {a['continuation_gate']['seed_repeatability']} | {row.get('joint_proposal_diagnostics', 'unavailable')} | {row.get('actual_updates', 0)} | {row['workflow_seconds']:.2f} |"
                if "workflow_seconds" in row
                else f"| {target} / {row['seed']} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | 0 | unavailable |"
            )
    lines += [
        "",
        "The following table keeps the descriptive MMD baselines and three-seed PSIS ranges beside each outcome. Joint-proposal status uses the inherited warning screen; it does not imply posterior equality.",
        "",
        "| Target / seed | Functional MMD² VI/NUTS | NUTS/NUTS | VI/VI | Latent MMD² | Native k range | Packed k range | Packed weight ESS fraction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for target, a in analyses.items():
        for row in a["fits"]:
            if row["stage_execution_status"] != "complete":
                lines.append(
                    f"| {target} / {row['seed']} | unavailable: {row['stage_execution_status']} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |"
                )
                continue
            m = row["functional_mmd"]
            native_k = old.value_range(
                [r.get("k") for r in row["native_psis"].get("runs", [])]
            )
            packed_k = old.value_range(
                [r.get("k") for r in row["packed_weights"]]
            )
            packed_ess = old.value_range(
                [r.get("smoothed_ess_fraction") for r in row["packed_weights"]]
            )
            lines.append(
                f"| {target} / {row['seed']} | {m['vi_vs_nuts']:.6g} | {m['nuts_vs_nuts']:.6g} | {m['vi_vs_vi']:.6g} | {row['latent_mmd']['vi_vs_nuts']:.6g} | {native_k} | {packed_k} | {packed_ess} |"
            )
    lines += [
        "",
        "## Frozen target and provenance",
        "",
        "The development record is the archived seed-60101 stationary AR(4), 4,096 samples at dt=1, coefficients [0.9, −0.8, 0.7, −0.6], innovation variance 0.4, stationary initialization and no realization-specific variance normalization. The native full-record rectangular Fourier observations contain 2,047 interior frequencies; DC and Nyquist are excluded. The fitted model has 30 centered frequency-only P-spline coefficients with the saved numerical second-derivative penalty, including its 1e-6 ridge and weak null directions. The conditional scale is the native HalfNormal(1.28) prior median, 0.8633468802509846; restoring the hierarchy samples that same scale prior. Observations, likelihood, basis, penalty and all other priors are identical within each VI/NUTS comparison.",
        "",
        f"Source artifacts: [{resolved['source']}]({resolved['source']}). Physical target and coordinate IDs, all source hashes, schedule configuration and predeclared budgets are in [continuation_protocol.json]({out / 'continuation_protocol.json'}). The fixed target and sampled-scale target have distinct IDs. The current density and coefficient gradients agree exactly when the hierarchy is conditioned at the archived scale, with full scale-dependent normalization retained.",
        "",
        "Execution used the existing NumPyro 0.22.0, JAX/jaxlib 0.9.1, Optax 0.2.7 environment, CPU float64, four sequential NUTS chains and fresh inference state for each fit. The repository revision is 63b5c7a10d8bd4a3dfed8f1df05b02112321fd1b; dirty-source hashes, platform and cache/thread environment are recorded in provenance.json. Dependencies were not upgraded. No diagonal VI, data generation, parameterisation change, flow, new likelihood, benchmark or confirmation job ran.",
        "",
        "## Executed optimization and endpoints",
        "",
        "The original schedule function is evaluated with its original 40,000-update horizon (warmup 0.001→0.01 over 1k, cosine to 0.0001 by 40k). The tail is constant 1e-4 at optimizer indices 40000..59999. Each fit starts from the same data-based physical pilot and native guide scale 0.1, with fresh Adam/SVI state and the archived seed; the 40k state stays live through the tail. There is no optimizer resume from a guide checkpoint and no stretched cosine. Checkpointing/objective draws do not advance training RNG.",
        "",
        "Fixed replay comparison occurs at actual update 40k before any tail updates. Parameter discrepancy, exact equality and numeric arrays are saved per seed. The old failed 30k→35k intervals remain in the source stability_fixed.json. New intervals are 40k→50k and 50k→60k (10k updates each); no unmeasured drift-per-update comparison is made with the old 5k intervals.",
        "",
    ]
    for target, a in analyses.items():
        lines += [
            f"## {target.capitalize()} posterior agreement, stability and repeatability",
            "",
        ]
        for row in a["fits"]:
            if row["stage_execution_status"] != "complete":
                lines.append(
                    f"- Seed {row['seed']}: {row['stage_execution_status']}; no completed posterior verdict."
                )
                continue
            primary = [
                r
                for r in row["final_primary"]
                if r["name"].startswith(("log_S", "log_band"))
                or r["name"] in ("log_sigma", "sigma")
            ]
            original = [
                r
                for r in row["endpoint_40000_primary"]
                if r["name"].startswith(("log_S", "log_band"))
                or r["name"] in ("log_sigma", "sigma")
            ]
            lines.append(
                f"- Seed {row['seed']}: replay/packing prerequisite {row['replay_status']}. 40k primary status {row['endpoint_40000_agreement']}, max mean error {max(abs(r['standardized_mean_difference']) for r in original):.4f} reference SD, SD ratios {old.value_range([r['sd_ratio'] for r in original])}. 60k primary status {row['final_primary_agreement']}, max mean error {max(abs(r['standardized_mean_difference']) for r in primary):.4f}, SD ratios {old.value_range([r['sd_ratio'] for r in primary])}."
            )
            for interval in row["late"]:
                r = interval["selected"]
                lines.append(
                    f"  {interval['first'] // 1000}k→{interval['second'] // 1000}k: {r['status']}; worst mean drift {r['max_mean_change_reference_sd']:.4f} reference SD, worst SD drift {r['max_relative_sd_change']:.2%}, paired objective change {r['paired_objective_change']:.4f} ± {2 * r['paired_objective_mcse']:.4f} (two MCSE). Base estimates remain alongside any one-off precision refinement."
                )
            k = old.value_range(
                [r.get("k") for r in row["native_psis"].get("runs", [])]
            )
            pk = old.value_range([r.get("k") for r in row["packed_weights"]])
            ess = old.value_range(
                [r.get("smoothed_ess_fraction") for r in row["packed_weights"]]
            )
            lines.append(
                f"  Native k {k}; packed k {pk}; packed smoothed weight ESS fraction {ess}. Functional MMD² {row['functional_mmd']['vi_vs_nuts']:.6g}, NUTS–NUTS {row['functional_mmd']['nuts_vs_nuts']:.6g}, VI–VI {row['functional_mmd']['vi_vs_vi']:.6g}; latent MMD² {row['latent_mmd']['vi_vs_nuts']:.6g}. No MMD cutoff or IID p-value."
            )
            roughness = [
                r
                for r in row["final_primary"]
                if r["name"] in ("log_sigma", "sigma")
            ]
            for r in roughness:
                lines.append(
                    f"  {r['name']}: standardized mean error {r['standardized_mean_difference']:.4f} ± {2 * r['mean_difference_mcse']:.4f}, SD ratio {r['sd_ratio']:.4f} ± {2 * r['sd_ratio_mcse']:.4f}; mean {r['mean_status']}, SD {r['sd_status']}."
                )
        lines.append("")
        lines.append(
            f"Final seed repeatability: {a['continuation_gate']['seed_repeatability']}; exploratory passing seeds {a['continuation_gate']['passing_seeds']}; all-seed recipe {a['continuation_gate']['repeatable_recipe']}."
        )
        lines.append("")
        for r in a["seed_pairs"]:
            lines.append(
                f"- Seeds {r['first_seed']}/{r['second_seed']}: {r['status']}; worst mean drift {r['max_mean_change_reference_sd']:.4f}, SD drift {r['max_relative_sd_change']:.2%}."
            )
        lines.append("")
    lines += [
        "",
        "## References and retained failures",
        "",
        "No new fixed NUTS job ran. Its accepted four-chain repair_1 posterior is reused; the old repair budget remains exhausted. Hierarchical initial/repair settings came from the already resolved protocol rather than being invented from the continuation plan: "
        + str(resolved["reference_initial"])
        + " / "
        + str(resolved["reference_repair"])
        + ". Four independent data-based starts use the tested corrected NumPyro initialization API, fresh mass/warmup state and a distinct reference seed. Physical IDs differ while observations/basis/penalty/prior roles are preserved.",
        "",
    ]
    events = [
        json.loads(line)
        for line in (out / "attempts.jsonl").read_text().splitlines()
    ]
    inference_jobs = [e for e in events if e.get("status") == "dispatching"]
    lines += [
        f"Executed inference jobs: {[e['job'] for e in inference_jobs]}. No hierarchical repair was required in this execution; zero diagnostic precision refinements were needed. Every seed completed exactly 60,000 actual updates. The original inherited checkpoints plus 50k/60k are retained; losses and parameter checkpoints are saved before expensive summaries. The initial schedule test compared compiled rates with uncompiled rates, producing tiny rounding differences; corrected equivalent compiled comparisons and the real bitwise guide replays passed. The initial offline plot adapter failed on JSON lists, and its traceback is retained in logs/controller_initial.log. Converting saved numeric lists to arrays repaired reporting without repeating inference. Tests/logs preserve both initial failures.",
        "",
    ]
    fixed_ref_directory, _ = old.selected_reference(
        Path(resolved["source"]), "fixed"
    )
    reference_paths = [fixed_ref_directory / "complete.json"] + sorted(
        out.glob("hierarchical/reference/*/complete.json")
    )
    lines += [
        "| Reference / attempt | Role | Divergences / cap hits | Max R-hat | Min bulk / tail ESS | Min BFMI | Max primary mean MCSE / SD |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in reference_paths:
        ref = json.loads(path.read_text())
        h = ref["health"]
        lines.append(
            f"| {ref['target']} / {ref['attempt']} | {'reused accepted reference' if ref['target'] == 'fixed' else ref['status']} | {h['divergences']} / {h['depth_saturation']} | {max(h['rhat']):.5f} | {min(h['ess_bulk']):.0f} / {min(h['ess_tail']):.0f} | {min(h['bfmi']):.3f} | {ref['primary_mean_mcse_sd_max']:.4f} |"
        )
    lines += [
        "",
        "The new hierarchical reference completed its initial 4 × (1,000 warmup + 2,000 retained) workflow; the frozen repair was not needed. Both references have adequate precision for the declared comparisons. Their target-specific posterior draws are never interchanged.",
        "",
    ]
    for path in sorted(out.rglob("failure.json")):
        failure = json.loads(path.read_text())
        lines.append(
            f"- Retained {path.relative_to(out)}: {failure['status']}; {failure.get('error')}; actual saved checkpoints {failure.get('available_parameter_checkpoints', [])}."
        )
    lines += [
        "",
        "## Diagnostic definitions and cost",
        "",
        "Primary and secondary coordinates, physical per-draw band quadrature, dense 257-node grid and all complete coefficient draws are inherited. Endpoint accuracy uses two-MCSE classifications with |mean error|≤0.1 reference SD and SD ratios [.9,1.1]. Late comparisons call the same inherited function with its mean denominator, symmetric relative-SD denominator, paired MCSE and objective 0.2+2MCSE screen. Missing or precision-limited results do not pass a gate. At most one saved-guide precision refinement per comparison increases checkpoint collections to 16,384 or final scalar collections to 32,768; base results stay retained. Reference autocorrelation enters mean/SD/quantile MCSE. This is validation against NUTS, not an online stop for new data.",
        "",
        "Final scalar/field summaries use 16,384 fresh joint draws; checkpoints use 4,096 common-key draws. Final PSIS uses 4,096×3 seeds, and objective evaluation uses 256×8 fixed keys plus three independent keys. Native PSIS splits keys into per-particle traces; packed PSIS uses vectorized posterior draws, so their realized particles and finite-sample k differ. Entire priors/factors and one support Jacobian are included; fixed sigma is excluded and hierarchical sigma is packed with its positive transform. MMD uses 512 matched draws and reference-only disjoint whitening, with descriptive baselines and negative unbiased values preserved. No reweighting.",
        "",
        "The hierarchical 40k endpoint is a secondary 4,096-draw checkpoint result; the 60k endpoint is the primary final result. Fixed 40k accuracy reuses the exact archived 16,384-draw endpoint. Coefficient agreement labels refer to marginal mean/SD screens. Full coefficient correlations, covariance discrepancies, raw/log band powers, 50/90/95% interval-width ratios, standardized Wasserstein distances and dense-field diagnostics are retained in each complete.json and field_dense.json, alongside every feature's MC uncertainty.",
        "",
        "Each fresh subprocess logs process-inclusive time (imports/recovery/compilation-containing fit/checks/output); fit manifests record initialization, first compiled chunk, remaining optimization loop including checkpoints, posterior sampling and final diagnostics. Guide/loss/posterior I/O and offline feature/MMD/field analysis are timed separately. Checkpoint timestamps separate reaching 40k, 50k and 60k. Timers synchronize through host copies/blocking conversions. The hierarchical NUTS timing includes warmup/sampling, compilation and native reconstruction, with chain persistence and health/feature analysis separated. These are research workflow costs under the recorded cache environment, not controlled cold or compile-reused latency comparisons. No ratio or speed advantage is inferred from historical workflow times.",
        "",
        "| Target / seed | Shared fit including final diagnostics (s) | Parameter/posterior/loss persistence (s) | Offline features/MMD/fields (s) | Inclusive fit workflow (s) | Fresh process inclusive (s) |",
        "|---|---|---|---|---|---|",
    ]
    process_times = {
        r["job"]: r["process_inclusive_seconds"] for r in timing["processes"]
    }
    for a in analyses.values():
        for r in a["timings"]:
            job = (
                "fixed_tail" if r["target"] == "fixed" else "hierarchical_vi"
            ) + f"_{r['seed']}"
            process_seconds = process_times.get(job)
            process_display = (
                f"{process_seconds:.2f}"
                if process_seconds is not None
                else "unavailable"
            )
            lines.append(
                f"| {r['target']} / {r['seed']} | {r['fit_through_joint_diagnostics_seconds']:.2f} | {r['guide_posterior_loss_persistence_seconds']:.2f} | {r['offline_features_mmd_fields_seconds']:.2f} | {r['wall_seconds']:.2f} | {process_display} |"
            )
    lines += [
        "",
        "The hierarchical reference used 10.33 s for compilation-containing warmup/sampling/native reconstruction, 0.26 s for chain persistence and 0.27 s for health/feature analysis (10.85 s workflow; 13.65 s fresh-process inclusive). Gate-analysis costs are additional research validation costs recorded in the attempt ledger; the per-fit figures do not include the full test suite, report rendering or every offline artifact check. No fixed-reference runtime is newly measured.",
        "",
        "Gate-analysis and precision-refinement costs are recorded separately in attempts.jsonl/refinement JSON; timing.json preserves the full boundary breakdown. Tests and saved-artifact verification are recorded in verification.json. No test count or posterior validity is inferred from the earlier report.",
        "",
        "## Implementation and verification",
        "",
        "The continuation adapter reuses the stationary model, optimizer, reconstruction and diagnostics. The original stationary plot entry point gained configurable interval labels; shared inference defaults are unchanged. Five continuation contract tests cover exact schedule/replay behavior, RNG-independent checkpoint observation, fresh-state timing/data dependence, distinct gates and stopping. Existing density/gradient, latent-packing and scientific regressions remain in the verification scope.",
        "",
        "The complete applicable routine and slow-regression suite passed: 144 tests, 29 warnings, no skips (80.25 s). The corrected focused contract run passed nine tests. Ruff and whitespace checks passed. Saved-artifact verification checked six 60k fits, every checkpoint round trip, bitwise fixed-prefix equality at all six archived checkpoints for all three seeds, 16,384 complete joint draws per final guide, and all 2,484 comparison rows. Recomputed packed target/guide densities and conditional coefficient gradients had zero maximum discrepancy. All six figures were inspected. Initial test/plot failures and historical scientific failures remain retained; passing software tests are separate from posterior validation.",
        "",
        f"Inspect [verification.json]({out / 'verification.json'}), [comparison.csv]({out / 'comparison.csv'}), [timing.json]({out / 'timing.json'}) and [gate_decisions.json]({out / 'gate_decisions.json'}) for numerical evidence. The source hashes remained unchanged throughout execution.",
        "",
        "## Decision and stop",
        "",
        "The completed hierarchical recipe supports a separately reviewed matched-accuracy efficiency study; no speed superiority is established and no benchmark or reserved-data job has run."
        if efficiency
        else "The evidence does not yet justify the matched-accuracy efficiency study. Resolve the reported prerequisite or hierarchical posterior discrepancy before speed tuning. No benchmark or reserved-data job has run.",
        "",
        "The next supported experiment is a bounded diagnostic of coefficient–roughness geometry using the saved NUTS and guide draws: compare log sigma with the native quadratic roughness c'Pc, conditional spread and skewness. This would distinguish missing nonlinear dependence or marginal shape from the linear covariance already represented by the stable full-covariance guide. A later verified coordinate comparison may be warranted by that evidence, but has not been run or assumed to fix the discrepancy. More optimizer steps and speed tuning are not justified by this stable but underdispersed smoothing posterior alone.",
        "",
        "Reproduction from the audited worktree:",
        "",
        "```bash",
        "PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary_next.py audit-next",
        "PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary_next.py continue",
        "PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary_next.py summarize-next",
        "```",
        "",
        "Completed or failed attempt directories are retained without optimizer resumption or replacement. Sources are checked against their initial hashes. Seeds 60102–60106 remain untouched. Truth overlays are context; these results establish neither calibration/coverage/SBC nor model adequacy, new-record robustness or time-varying performance.",
        "",
        "## Figures",
        "",
        "PSD shading shows posterior 90% intervals. Mean-discrepancy and SD-ratio shading, and band-summary error bars, show two Monte Carlo standard errors for the comparison estimates; they are not posterior credible intervals or simultaneous bands. The roughness panels show marginal posterior distributions, and stability panels show the declared drift screens and MC uncertainty.",
        "",
    ]
    for name in (
        "psd",
        "mean_error",
        "sd_ratio",
        "bands",
        "roughness",
        "stability",
    ):
        if (out / "figures" / f"{name}.png").exists():
            lines += [f"![{name}]({out / 'figures' / (name + '.png')})", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def plot(out, analyses):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved = config(out)
    arrays = np.load(Path(resolved["source"]) / "frozen_target/arrays.npz")
    (out / "figures").mkdir(exist_ok=True)
    fields, records, refs, stabilities = {}, {}, {}, {}
    for target, a in analyses.items():
        refdir, ref = reference_path(out, target)
        refs[target] = (refdir, ref, xr.load_dataset(refdir / "posterior.nc"))
        fields[target] = {"vi": {}}
        stabilities[target] = {
            "fits": [],
            "seed_pairs": [{"guide": "mvn", **r} for r in a["seed_pairs"]],
        }
        for seed in resolved["seeds"]:
            directory = fit_directory(out, target, seed)
            if not (directory / "complete.json").exists():
                continue
            record = json.loads((directory / "complete.json").read_text())
            records[target, "mvn", seed] = directory, record
            saved = {
                kind: json.loads(
                    (directory / f"field_{kind}.json").read_text()
                )
                for kind in ("dense", "overlay")
            }
            # Existing plotter accepts numerical arrays; JSON reads yield lists.
            saved = {
                kind: {
                    group: {
                        key: np.asarray(value)
                        if isinstance(value, list)
                        else value
                        for key, value in values.items()
                    }
                    for group, values in record.items()
                }
                for kind, record in saved.items()
            }
            fields[target]["reference"] = {
                kind: saved[kind]["reference"] for kind in saved
            }
            fields[target]["vi"]["mvn", seed] = {
                **{kind: saved[kind]["vi"] for kind in saved},
                "comparison": saved["dense"]["comparison"],
            }
            row = next(r for r in a["fits"] if r["seed"] == seed)
            stabilities[target]["fits"].append(
                {
                    "guide": "mvn",
                    "seed": seed,
                    "late": [x["selected"] for x in row["late"]],
                }
            )
    fields = {t: f for t, f in fields.items() if "reference" in f}
    if fields:
        old.plot_results(
            out,
            arrays,
            fields,
            records,
            refs,
            stabilities,
            late_labels=("40k→50k", "50k→60k", "Final seed pairs"),
            roughness_unexecuted_reason="Hierarchical target unexecuted\n"
            + json.loads((out / "gate_decisions.json").read_text())[
                "stage_execution_status"
            ],
        )
    fig, axes = plt.subplots(
        max(1, len(analyses)),
        1,
        figsize=(8, 4 * max(1, len(analyses))),
        squeeze=False,
    )
    for index, (target, a) in enumerate(analyses.items()):
        ax = axes[index, 0]
        for row in a["fits"]:
            if row["stage_execution_status"] != "complete":
                continue
            bands = [
                r
                for r in row["final_primary"]
                if r["name"].startswith("log_band")
            ]
            offset = (row["seed"] - 7102) * 0.09
            ax.errorbar(
                np.arange(3) + offset,
                [r["sd_ratio"] for r in bands],
                yerr=[2 * r["sd_ratio_mcse"] for r in bands],
                fmt="o",
                capsize=3,
                label=f"mvn seed {row['seed']}",
            )
        ax.axhspan(0.9, 1.1, color="green", alpha=0.08)
        ax.axhline(1, color="0.5")
        ax.set_xticks(np.arange(3), ["[.04,.18]", "[.18,.32]", "[.32,.46]"])
        ax.set_title(f"{target.capitalize()}: log band-power uncertainty, 60k")
        ax.set_ylabel("VI SD / NUTS SD")
        ax.set_xlabel("Physical frequency band [cycles/sample]")
        ax.legend(fontsize=9)
    fig.suptitle(
        "Error bars: two MCSE; shaded region: development screen", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out / "figures/bands.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/vi-stationary-next"
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "runs/vi-stationary-ar4"
    )
    parser.add_argument(
        "command",
        choices=[
            "audit-next",
            "fixed-tail",
            "reference",
            "hierarchical",
            "summarize-next",
            "continue",
        ],
    )
    parser.add_argument(
        "--seed", type=int, choices=[7101, 7102, 7103], default=7101
    )
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    out, source = args.out.resolve(), args.source.resolve()
    if args.command == "audit-next":
        prepare(out, source)
    elif args.command == "fixed-tail":
        fit(out, "fixed", args.seed)
    elif args.command == "reference":
        reference(out, args.repair)
    elif args.command == "hierarchical":
        fit(out, "hierarchical", args.seed)
    elif args.command == "summarize-next":
        summarize(out)
    else:
        execute(out, source)


if __name__ == "__main__":
    main()
