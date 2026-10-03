"""Bounded multivariate and scalar time-varying posterior validation.

Each worker starts in a fresh process. Historical outcomes and failed attempts
are immutable. This is a study controller, not a production VI preset.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from comparison import (
    compare_with_mc,
    mc_stats,
    paired_stability,
    primary_indices,
)
from extension_targets import (
    ROOT,
    TV_SOURCE,
    density_preflight,
    feature_values,
    freeze_matrix,
    load_matrix,
    load_time_varying,
    select_target,
)
from numpyro.infer.initialization import init_to_uniform
from study_common import (
    draw_dataset,
    learning_rate,
    mmd_comparison,
    packed_reference,
    posterior_geometry,
    write_json,
)

from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    VIDiagnosticState,
    compare_features,
    evaluate_objective,
    nuts_reference_health,
    packed_log_densities,
    rebuild_guide,
    runtime_provenance,
)
from log_psplines.inference.nuts import run_nuts
from log_psplines.inference.vi import fit_vi

DEFAULT_OUT = ROOT / "runs/vi-matrix-tv"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tail_schedule(schedule):
    prefix = learning_rate(schedule, 40000)
    return lambda step: jnp.where(step < 40000, prefix(step), 1e-4)


def event(out, **record):
    with (out / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def prepare(out):
    if (out / "protocol.json").exists():
        return config(out)
    if out.exists():
        raise ValueError(
            "incomplete preparation retained; use a new output directory"
        )
    out.mkdir(parents=True)
    source = ROOT / "runs/vi-stationary-next/continuation_protocol.json"
    inherited = json.loads(source.read_text())["inherited"]
    load_time_varying()  # Existing arrays/fingerprint must pass before inference.
    protocol = {
        "cases": ["matrix", "time_varying"],
        "smoothing": ["fixed", "hierarchical"],
        "seeds": [7101, 7102, 7103],
        "guide": "mvn",
        "inherited": inherited,
        "schedule_source": str(source),
        "updates": 60000,
        "prefix_horizon": 40000,
        "tail_rate": 1e-4,
        "checkpoints": [10000, 20000, 30000, 40000, 50000, 60000],
        "late_intervals": [[40000, 50000], [50000, 60000]],
        "reference_initial": inherited["reference_initial"],
        "reference_repair": inherited["reference_repair"],
        "budget": {
            "reference_initial_per_target": 1,
            "reference_repairs_per_target": 1,
            "vi_per_target": 3,
            "vi_retries": 0,
            "saved_guide_precision_refinements_per_seed": 1,
        },
        "refinement": {"checkpoint_draws": 16384, "final_draws": 32768},
        "gate": "accepted reference plus at least one seed with final primary agreement AND both late intervals within_screen; all three seeds and all three seed pairs separately required for repeatable_recipe",
        "matrix": {
            "generator": "stationary VAR(1)",
            "data_seed": 62001,
            "samples": 4096,
            "dt": 1.0,
            "Nb": 8,
            "taper": None,
            "detrend": False,
            "eigenvalue_floor": None,
            "coarse_graining": None,
            "coefficients_per_component": 8,
            "parameterization": "native centered",
            "sigma_fixed": 1.28 * 0.6744897501960817,
            "transition": [[0.55, 0.12], [-0.08, 0.35]],
            "innovation_covariance": [[0.4, 0.12], [0.12, 0.3]],
            "init": "native empirical component PLS; diagonal coefficients shifted by -log(duration) to match physical units; independent chain perturbations",
        },
        "time_varying": {
            "source": str(TV_SOURCE),
            "model": "native scalar tensor power",
            "parameterization": "native noncentered",
            "sigma_fixed": 10 * 0.6744897501960817,
            "observations": "archived exact Gamma power/count control; no raw moving periodogram",
            "inherited_hierarchical_reference": "reuse if it meets new functional precision; otherwise one declared repair",
            "init": "independent uniform unconstrained radius 0.05; no VI initialization",
        },
        "feature_contract": {
            "points": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            "bands": [[0.04, 0.18], [0.18, 0.32], [0.32, 0.46]],
            "band_quadrature_points": 129,
            "times": [0.25, 0.5, 0.75],
            "primary": "log autospectra, real/imag cross spectra, squared coherence, log band power, physical/log scales, temporal contrasts",
            "secondary": "all physical coefficients, penalty energies; correlation, 50/90/95 interval widths and Wasserstein in comparisons",
            "units_tv": "coefficient variance; frequency-integrated variance is a model functional, not calibrated raw-process variance",
        },
        "forbidden": [
            "diagonal reruns",
            "parameterization sweeps",
            "flows",
            "new likelihoods",
            "AR4 seeds 60102-60106",
            "benchmarks",
            "untouched-seed confirmation",
        ],
        "provenance": runtime_provenance(),
    }
    write_json(
        out / "protocol.json", protocol
    )  # Settings frozen BEFORE observations/inference.
    freeze_matrix(out / "matrix/frozen_target.npz")
    sources = [
        source,
        TV_SOURCE / "target.npz",
        TV_SOURCE / "manifest.json",
        TV_SOURCE / "reference_posterior.nc",
        TV_SOURCE / "reference_sample_stats.nc",
        TV_SOURCE / "reference.json",
        out / "matrix/frozen_target.npz",
    ]
    sources += [
        Path(__file__),
        Path(__file__).with_name("extension_targets.py"),
        Path(__file__).with_name("comparison.py"),
        Path(__file__).with_name("study_common.py"),
    ]
    sources += [
        ROOT / "src/log_psplines" / name
        for name in (
            "inference/model.py",
            "inference/power.py",
            "inference/vi.py",
            "inference/nuts.py",
            "diagnostics/variational.py",
            "likelihoods/whittle.py",
            "likelihoods/wishart.py",
        )
    ]
    write_json(
        out / "source_hashes.json",
        {str(p.resolve()): file_hash(p) for p in sources},
    )
    checks = {}
    for case in protocol["cases"]:
        base = load_base(out, case)
        checks[case] = {
            "target": base.descriptor,
            "fingerprints": {
                s: select_target(base, s).identity
                for s in protocol["smoothing"]
            },
            "density": density_preflight(base, select_target(base, "fixed")),
        }
    write_json(out / "preflight.json", checks)
    event(out, stage="preflight", status="passed", checks=checks)
    return protocol


def config(out):
    cfg = json.loads((out / "protocol.json").read_text())
    for path, expected in json.loads(
        (out / "source_hashes.json").read_text()
    ).items():
        if file_hash(path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    if not (out / "preflight.json").exists():
        raise ValueError("unresolved preflight")
    return cfg


def load_base(out, case):
    return (
        load_matrix(out / "matrix/frozen_target.npz")
        if case == "matrix"
        else load_time_varying()
    )


def get_target(out, case, smoothing):
    config(out)
    if smoothing == "hierarchical":
        gate_path = out / case / "fixed/analysis.json"
        if (
            not gate_path.exists()
            or not json.loads(gate_path.read_text())["gate"][
                "exploratory_hierarchy_unlock"
            ]
        ):
            raise ValueError("fixed-stage gate has not passed")
    return select_target(load_base(out, case), smoothing)


def init_strategy(target):
    if target.descriptor.get("generator") != "stationary_VAR1":
        return init_to_uniform(radius=0.05)

    def independent(site):
        if site["type"] != "sample" or site.get("is_observed", False):
            return None
        name, key = site["name"], site["kwargs"]["rng_key"]
        value = target.init[name]
        if name.startswith("sigma_"):
            return value * jnp.exp(0.1 * jax.random.normal(key))
        return value + 0.05 * jax.random.normal(key, value.shape)

    return partial(independent)


def evaluate_reference(target, posterior, stats, settings, cfg):
    features, names = feature_values(target, posterior)
    health = nuts_reference_health(
        features, stats, max_tree_depth=settings["max_tree_depth"]
    )
    summary = mc_stats(features)
    precision = float(
        np.max((summary["mean_mcse"] / summary["sd"])[primary_indices(names)])
    )
    return (
        features,
        names,
        summary,
        {
            "health": health,
            "primary_mean_mcse_sd_max": precision,
            "status": "failed_reference"
            if health["status"] != "accepted_screen"
            else "reference_precision_limited"
            if precision > cfg["inherited"]["reference_primary_mean_mcse_sd"]
            else "accepted_screen",
        },
    )


def reference(out, case, smoothing, attempt):
    cfg = config(out)
    target = get_target(out, case, smoothing)
    directory = out / case / smoothing / "reference" / attempt
    if directory.exists():
        raise ValueError(f"retained attempt exists: {directory}")
    directory.mkdir(parents=True)
    settings = cfg[
        "reference_repair" if attempt == "repair_1" else "reference_initial"
    ]
    record = {
        "case": case,
        "smoothing": smoothing,
        "attempt": attempt,
        "settings": settings,
        "target_fingerprint": target.identity,
        "provenance": runtime_provenance(),
    }
    write_json(directory / "manifest.json", record)
    started = perf_counter()
    try:
        inherited = (
            case == "time_varying"
            and smoothing == "hierarchical"
            and attempt == "initial"
        )
        if inherited:
            posterior = xr.load_dataset(TV_SOURCE / "reference_posterior.nc")
            stats = xr.load_dataset(TV_SOURCE / "reference_sample_stats.nc")
            record["inherited_from"] = str(TV_SOURCE)
            record["historical_reference"] = json.loads(
                (TV_SOURCE / "reference.json").read_text()
            )
        else:
            result = run_nuts(
                target.model,
                rng_key=jax.random.PRNGKey(
                    6101
                    + 100 * (case == "time_varying")
                    + 10 * (smoothing == "hierarchical")
                    + int(attempt == "repair_1")
                ),
                n_warmup=settings["warmup"],
                n_samples=settings["draws"],
                num_chains=4,
                init_strategy=init_strategy(target),
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
            posterior, stats = result.posterior, result.sample_stats
        sampled = perf_counter()
        posterior.to_netcdf(directory / "posterior.nc", engine="h5netcdf")
        stats.to_netcdf(directory / "sample_stats.nc", engine="h5netcdf")
        persisted = perf_counter()
        features, names, summary, assessed = evaluate_reference(
            target, posterior, stats, settings, cfg
        )
        np.savez_compressed(
            directory / "features.npz", values=features, names=np.array(names)
        )
        write_json(directory / "feature_stats.json", summary)
        record.update(
            assessed,
            timings={
                "inherited_load_seconds"
                if inherited
                else "warmup_sampling_including_compile_seconds": sampled
                - started,
                "posterior_stats_persistence_seconds": persisted - sampled,
                "reconstruction_health_precision_seconds": perf_counter()
                - persisted,
            },
            wall_seconds=perf_counter() - started,
        )
        write_json(directory / "complete.json", record)
        event(
            out,
            stage="reference",
            **{
                k: record[k]
                for k in (
                    "case",
                    "smoothing",
                    "attempt",
                    "status",
                    "wall_seconds",
                )
            },
        )
        print(
            f"{case}/{smoothing}/{attempt}: {record['status']}; precision={assessed['primary_mean_mcse_sd_max']:.4f}",
            flush=True,
        )
    except Exception as error:
        record.update(
            status="execution_failed",
            error=repr(error),
            wall_seconds=perf_counter() - started,
        )
        write_json(directory / "failure.json", record)
        raise


def selected_reference(out, case, smoothing):
    path = out / case / smoothing / "reference"
    selection = json.loads((path / "selection.json").read_text())
    directory = path / selection["attempt"]
    record = json.loads((directory / "complete.json").read_text())
    if record["status"] != "accepted_screen":
        raise ValueError("reference is unusable")
    return directory, record


def fit(out, case, smoothing, seed):
    cfg = config(out)
    target = get_target(out, case, smoothing)
    _, ref = selected_reference(out, case, smoothing)
    if target.identity != ref["target_fingerprint"]:
        raise ValueError("VI/NUTS physical target mismatch")
    directory = out / case / smoothing / "vi" / str(seed)
    if directory.exists():
        raise ValueError(f"retained VI attempt exists: {directory}")
    directory.mkdir(parents=True)
    inherited = cfg["inherited"]
    record = {
        "case": case,
        "smoothing": smoothing,
        "seed": seed,
        "target_fingerprint": target.identity,
        "provenance": runtime_provenance(),
    }
    write_json(directory / "manifest.json", record)
    started = perf_counter()
    clock = []
    try:
        diagnostics = diagnostic_config(cfg, target)

        def preserve(step, params):
            io = perf_counter()
            np.savez_compressed(directory / f"parameters_{step}.npz", **params)
            clock.append(
                {
                    "step": step,
                    "elapsed_seconds": perf_counter() - started,
                    "persistence_seconds": perf_counter() - io,
                }
            )
            write_json(directory / "checkpoint_timing.json", clock)

        result = fit_vi(
            target.model,
            rng_key=jax.random.PRNGKey(seed),
            vi_steps=60000,
            optimizer_lr=tail_schedule(inherited["schedule"]),
            optimizer_lr_metadata={
                "prefix": inherited["schedule"],
                "prefix_horizon": 40000,
                "constant_tail_rate": 1e-4,
            },
            optimization_particles=8,
            guide="mvn",
            posterior_draws=inherited["final_draws"],
            init_values=target.init,
            early_stopping=False,
            diagnostics=diagnostics,
            checkpoint_callback=preserve,
        )
        fitted = perf_counter()
        result.diagnostics.save(directory)
        result.posterior.to_netcdf(
            directory / "posterior.nc", engine="h5netcdf"
        )
        np.save(directory / "losses.npy", np.asarray(result.losses))
        record.update(
            status="complete",
            optimization=result.diagnostics.metadata["optimization"],
            native_psis=result.diagnostics.metadata["native_psis"],
            weights=result.diagnostics.metadata["weights"],
            timings={
                "shared_vi": result.timings,
                "fit_through_joint_diagnostics_seconds": fitted - started,
                "persistence_seconds": perf_counter() - fitted,
            },
            wall_seconds=perf_counter() - started,
        )
        write_json(directory / "complete.json", record)
        event(
            out,
            stage="vi",
            case=case,
            smoothing=smoothing,
            seed=seed,
            status="complete",
            wall_seconds=record["wall_seconds"],
        )
        print(
            f"completed {case}/{smoothing}/{seed}: {record['wall_seconds']:.2f}s",
            flush=True,
        )
    except Exception as error:
        record.update(
            status="execution_failed",
            error=repr(error),
            wall_seconds=perf_counter() - started,
            checkpoint_clock=clock,
        )
        write_json(directory / "failure.json", record)
        raise


def diagnostic_config(cfg, target):
    c = cfg["inherited"]
    return VIDiagnosticConfig(
        seeds=tuple(c["diagnostic_seeds"]),
        num_particles=c["diagnostic_particles"],
        chunk_size=c["diagnostic_chunk"],
        evaluation_seeds=tuple(c["evaluation_seeds"]),
        evaluation_particles=c["evaluation_particles"],
        checkpoint_steps=tuple(cfg["checkpoints"]),
        target_fingerprint=target.identity,
        parameterization="noncentered"
        if target.descriptor["model"] == "scalar_tensor_power"
        else "centered",
    )


def statuses(rows, indices):
    values = [
        rows[i][key] for i in indices for key in ("mean_status", "sd_status")
    ]
    return (
        "unavailable"
        if not values or "unavailable" in values
        else "outside_screen"
        if "outside_screen" in values
        else "within_screen"
        if all(v == "within_screen" for v in values)
        else "mc_precision_limited"
    )


def analyze(out, case, smoothing):
    cfg = config(out)
    target = get_target(out, case, smoothing)
    refdir, reference_record = selected_reference(out, case, smoothing)
    posterior = xr.load_dataset(refdir / "posterior.nc")
    saved = np.load(refdir / "features.npz")
    p, names = saved["values"], saved["names"].tolist()
    c = cfg["inherited"]
    records, finals, objectives = [], {}, {}
    started = perf_counter()
    for seed in cfg["seeds"]:
        directory = out / case / smoothing / "vi" / str(seed)
        if not (directory / "complete.json").exists():
            records.append(
                {
                    "seed": seed,
                    "status": "execution_failed"
                    if (directory / "failure.json").exists()
                    else "not_run",
                    "final_primary_agreement": "unavailable",
                    "late_optimization_stability": "unavailable",
                }
            )
            continue
        state = VIDiagnosticState.load(directory)
        guide = rebuild_guide(
            state, target.model, target_fingerprint=target.identity
        )
        # Round-trip densities are checked against archived packed draws.
        density_errors = []
        for diagnostic_seed in c["diagnostic_seeds"]:
            prefix = f"seed_{diagnostic_seed}"
            points = state.arrays[f"{prefix}_packed"][:3]
            density = partial(
                packed_log_densities, target.model, guide, state.params
            )
            logp, logq = jax.vmap(density)(jnp.asarray(points))
            for actual, key in ((logp, "log_joint"), (logq, "log_guide")):
                density_errors.append(
                    float(
                        np.max(
                            abs(
                                np.asarray(actual)
                                - state.arrays[f"{prefix}_{key}"][:3]
                            )
                        )
                    )
                )
        if max(density_errors) > 1e-8:
            raise ValueError("saved-guide density round trip failed")
        final = xr.load_dataset(directory / "posterior.nc")
        q, qnames = feature_values(target, final)
        if qnames != names:
            raise ValueError("VI/NUTS feature schema mismatch")

        def assess(
            q, checkpoint_count, guide=guide, state=state, directory=directory
        ):
            rows = compare_with_mc(q, p, names, c)
            checkpoints, evaluations, stability = {}, {}, []
            for step in (40000, 50000, 60000):
                _, ds = draw_dataset(
                    guide,
                    state.checkpoints[str(step)],
                    jax.random.PRNGKey(c["checkpoint_draw_seed"]),
                    checkpoint_count,
                )
                checkpoints[step], _ = feature_values(target, ds)
                evaluations[step] = state.metadata["optimization"][
                    "checkpoint_objectives"
                ][str(step)]
                np.savez_compressed(
                    directory / f"features_{step}_{checkpoint_count}.npz",
                    values=checkpoints[step],
                    names=np.array(names),
                )
            for a, b in cfg["late_intervals"]:
                item = paired_stability(
                    checkpoints[a],
                    checkpoints[b],
                    p,
                    names,
                    evaluations[a],
                    evaluations[b],
                    c,
                )
                stability.append({"interval": [a, b], **item})
            return rows, stability, evaluations

        rows, stability, evaluations = assess(q, c["checkpoint_draws"])
        base = {
            "final_primary_agreement": statuses(rows, primary_indices(names)),
            "stability": stability,
            "rows": rows,
        }
        refinement = None
        if base["final_primary_agreement"] == "mc_precision_limited" or any(
            s["status"] == "mc_precision_limited" for s in stability
        ):
            _, ds = draw_dataset(
                guide,
                state.params,
                jax.random.PRNGKey(8401),
                cfg["refinement"]["final_draws"],
            )
            q, _ = feature_values(target, ds)
            rows, stability, evaluations = assess(
                q, cfg["refinement"]["checkpoint_draws"]
            )
            refinement = {
                "kind": "one saved-guide precision refinement; no optimization",
                **cfg["refinement"],
                "before": base,
            }
        comparison = compare_features(
            q,
            p,
            names=names,
            vi_fingerprint=target.identity,
            reference_fingerprint=target.identity,
            reference_status="accepted_screen",
        )
        comparison["mc_uncertainty"] = rows
        comparison["functional_mmd"] = mmd_comparison(q, p, c)
        packed_p = packed_reference(guide, posterior)
        packed_q = packed_reference(guide, final)
        comparison["latent_mmd"] = mmd_comparison(packed_q, packed_p, c)
        comparison["guide_geometry"] = posterior_geometry(
            guide, state.params, packed_p
        )
        objective = evaluate_objective(
            target.model,
            guide,
            state.params,
            diagnostic_config(cfg, target),
            seeds=c["diagnostic_seeds"],
        )
        objectives[seed] = objective
        finals[seed] = q
        stability_status = (
            "outside_screen"
            if any(s["status"] == "outside_screen" for s in stability)
            else "within_screen"
            if all(s["status"] == "within_screen" for s in stability)
            else "mc_precision_limited"
        )
        record = {
            "seed": seed,
            "status": "complete",
            "final_primary_agreement": statuses(rows, primary_indices(names)),
            "coefficient_agreement": statuses(
                rows,
                [
                    i
                    for i, n in enumerate(names)
                    if n.startswith("coefficient_")
                ],
            ),
            "late_optimization_stability": stability_status,
            "stability": stability,
            "final": comparison,
            "refinement": refinement,
            "independent_objective": objective,
            "guide_density_roundtrip_max_error": max(density_errors)
            if density_errors
            else None,
        }
        write_json(directory / "analysis.json", record)
        records.append(record)
    pairs = []
    for a, b in itertools.combinations(finals, 2):
        differences = np.asarray(objectives[b]["values"]) - np.asarray(
            objectives[a]["values"]
        )
        pairs.append(
            {
                "seeds": [a, b],
                "paired_objective_change": float(differences.mean()),
                "paired_objective_mcse": float(
                    differences.std(ddof=1) / np.sqrt(len(differences))
                ),
            }
        )
    # Cross-seed comparisons have independent final draws: use unpaired MCSE,
    # while preserving paired checkpoint screens strictly within each seed.
    for pair in pairs:
        a, b = pair["seeds"]
        qa, qb = finals[a], finals[b]
        sa, sb, sp = (
            mc_stats(qa, iid=True),
            mc_stats(qb, iid=True),
            mc_stats(p),
        )
        from comparison import screen_interval

        rows = []
        for i, name in enumerate(names):
            delta = (sb["mean"][i] - sa["mean"][i]) / sp["sd"][i]
            error = (
                np.sqrt(
                    sa["mean_mcse"][i] ** 2
                    + sb["mean_mcse"][i] ** 2
                    + (delta * sp["sd_mcse"][i]) ** 2
                )
                / sp["sd"][i]
            )
            drift = (
                2 * (sb["sd"][i] - sa["sd"][i]) / (sa["sd"][i] + sb["sd"][i])
            )
            de = (
                np.sqrt(
                    (4 * sb["sd"][i] * sa["sd_mcse"][i]) ** 2
                    + (4 * sa["sd"][i] * sb["sd_mcse"][i]) ** 2
                )
                / (sa["sd"][i] + sb["sd"][i]) ** 2
            )
            rows.append(
                {
                    "name": name,
                    "mean_change_reference_sd": delta,
                    "mean_mcse": error,
                    "relative_sd_change": drift,
                    "sd_change_mcse": de,
                    "mean_status": screen_interval(delta, error, -0.1, 0.1),
                    "sd_status": screen_interval(drift, de, -0.05, 0.05),
                }
            )
        relevant = primary_indices(names) + [
            i for i, n in enumerate(names) if n.startswith("coefficient_")
        ]
        pair.update(
            features=rows,
            status=statuses(rows, relevant),
            coupling="independent final joint draws; unpaired MCSE; reference SD uncertainty retained",
        )
        if (
            abs(pair["paired_objective_change"])
            > c["objective_change_tolerance"]
            + 2 * pair["paired_objective_mcse"]
        ):
            pair["status"] = "outside_screen"
    passing = [
        r["seed"]
        for r in records
        if r["final_primary_agreement"] == "within_screen"
        and r["late_optimization_stability"] == "within_screen"
    ]
    repeatability = (
        "within_screen"
        if len(pairs) == 3
        and all(p["status"] == "within_screen" for p in pairs)
        else "outside_screen"
        if any(p["status"] == "outside_screen" for p in pairs)
        else "mc_precision_limited"
        if pairs
        else "unavailable"
    )
    result = {
        "case": case,
        "smoothing": smoothing,
        "target_fingerprint": target.identity,
        "reference": reference_record,
        "seeds": records,
        "seed_pairs": pairs,
        "gate": {
            "exploratory_hierarchy_unlock": bool(passing),
            "passing_seeds": passing,
            "seed_repeatability": repeatability,
            "repeatable_recipe": len(passing) == 3
            and repeatability == "within_screen",
        },
        "analysis_seconds": perf_counter() - started,
    }
    write_json(out / case / smoothing / "analysis.json", result)
    print(
        f"{case}/{smoothing}: gate={bool(passing)}; seeds={[(r['seed'], r['final_primary_agreement'], r['late_optimization_stability']) for r in records]}; repeatability={repeatability}",
        flush=True,
    )
    return result


def dispatch(out, arguments, name):
    log = out / "logs" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    before = perf_counter()
    with log.open("x") as stream:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--out",
                str(out),
                *arguments,
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    event(
        out,
        worker=name,
        exit_code=result.returncode,
        wall_seconds=perf_counter() - before,
        log=str(log),
    )
    print(f"worker {name}: exit={result.returncode}", flush=True)
    return result.returncode == 0


def execute(out):
    cfg = prepare(out)
    for case in cfg["cases"]:
        for smoothing in cfg["smoothing"]:
            if smoothing == "hierarchical" and not json.loads(
                (out / case / "fixed/analysis.json").read_text()
            ).get("gate", {}).get("exploratory_hierarchy_unlock", False):
                event(
                    out,
                    case=case,
                    smoothing=smoothing,
                    status="not_run_fixed_gate",
                )
                break
            selected = None
            for attempt in ("initial", "repair_1"):
                directory = out / case / smoothing / "reference" / attempt
                if not directory.exists():
                    dispatch(
                        out,
                        [
                            "reference",
                            "--case",
                            case,
                            "--smoothing",
                            smoothing,
                            "--attempt",
                            attempt,
                        ],
                        f"{case}-{smoothing}-reference-{attempt}",
                    )
                if (directory / "complete.json").exists() and json.loads(
                    (directory / "complete.json").read_text()
                )["status"] == "accepted_screen":
                    selected = attempt
                    break
            if selected is None:
                write_json(
                    out / case / smoothing / "analysis.json",
                    {
                        "case": case,
                        "smoothing": smoothing,
                        "status": "stopped_unresolved_reference",
                        "gate": {"exploratory_hierarchy_unlock": False},
                        "seeds": [
                            {
                                "seed": s,
                                "status": "not_run_reference_prerequisite",
                            }
                            for s in cfg["seeds"]
                        ],
                    },
                )
                event(
                    out,
                    case=case,
                    smoothing=smoothing,
                    status="stopped_unresolved_reference",
                )
                break
            write_json(
                out / case / smoothing / "reference/selection.json",
                {"attempt": selected},
            )
            for seed in cfg["seeds"]:
                directory = out / case / smoothing / "vi" / str(seed)
                if not directory.exists():
                    dispatch(
                        out,
                        [
                            "fit",
                            "--case",
                            case,
                            "--smoothing",
                            smoothing,
                            "--seed",
                            str(seed),
                        ],
                        f"{case}-{smoothing}-vi-{seed}",
                    )
            if not (
                out / case / smoothing / "analysis.json"
            ).exists() and not dispatch(
                out,
                ["analyze", "--case", case, "--smoothing", smoothing],
                f"{case}-{smoothing}-analysis",
            ):
                event(
                    out,
                    case=case,
                    smoothing=smoothing,
                    status="stopped_unresolved_analysis",
                )
                break
    event(out, status="bounded_execution_finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "execute", "reference", "fit", "analyze"],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case", choices=["matrix", "time_varying"])
    parser.add_argument("--smoothing", choices=["fixed", "hierarchical"])
    parser.add_argument(
        "--attempt", choices=["initial", "repair_1"], default="initial"
    )
    parser.add_argument("--seed", type=int, choices=[7101, 7102, 7103])
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.command == "prepare":
        prepare(args.out)
    elif args.command == "execute":
        execute(args.out)
    elif args.command == "reference":
        reference(args.out, args.case, args.smoothing, args.attempt)
    elif args.command == "fit":
        fit(args.out, args.case, args.smoothing, args.seed)
    else:
        analyze(args.out, args.case, args.smoothing)


if __name__ == "__main__":
    main()
