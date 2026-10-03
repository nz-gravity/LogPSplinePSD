"""Analyze saved optimization fits, with stability separate from accuracy."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from optimize import events_for, job_name, settings
from run import write_json

from log_psplines.diagnostics.variational import VIDiagnosticState


def paired_check(
    first, second, first_row, second_row, reference, latent_sd, cfg
):
    a, b = (
        first["features"].reshape(-1, reference.shape[-1]),
        second["features"].reshape(-1, reference.shape[-1]),
    )
    names = first["feature_names"].tolist()
    reference_sd = reference.reshape(-1, reference.shape[-1]).std(0, ddof=1)
    changes = b - a
    means = np.abs(changes.mean(0)) / reference_sd
    mean_mcse = changes.std(0, ddof=1) / np.sqrt(len(changes)) / reference_sd
    sd_a, sd_b = a.std(0, ddof=1), b.std(0, ddof=1)
    sd_drift = np.abs(sd_b - sd_a) / ((sd_a + sd_b) / 2)
    width_a, width_b = (
        np.diff(np.quantile(x, [0.025, 0.975], axis=0), axis=0)[0]
        for x in (a, b)
    )
    widths = np.abs(width_b - width_a) / ((width_a + width_b) / 2)
    events = []
    for name, index, threshold in events_for(names):
        indicator_a, indicator_b = (x[:, index] <= threshold for x in (a, b))
        difference = indicator_b.astype(float) - indicator_a
        events.append(
            {
                "name": name,
                "absolute_change": abs(float(difference.mean())),
                "paired_mcse": float(
                    difference.std(ddof=1) / np.sqrt(len(difference))
                )
                if np.any(difference)
                else None,
                "status": "constant_indicator_no_tail_validation"
                if len(np.unique(np.r_[indicator_a, indicator_b])) == 1
                else "estimated",
            }
        )
    objective = np.asarray(
        second_row["fixed_objective"]["values"]
    ) - np.asarray(first_row["fixed_objective"]["values"])
    objective_mean = float(objective.mean())
    objective_mcse = float(objective.std(ddof=1) / np.sqrt(len(objective)))
    latent_mean = (
        np.abs(second["packed_mean"] - first["packed_mean"]) / latent_sd
    )
    latent_drift = np.abs(second["packed_sd"] - first["packed_sd"]) / (
        (second["packed_sd"] + first["packed_sd"]) / 2
    )
    event_max = max(x["absolute_change"] for x in events)
    functional_flags = {
        "means": bool(
            np.max(means) <= cfg["mean_drift_tolerance_reference_sd"]
        ),
        "sd": bool(np.max(sd_drift) <= cfg["relative_sd_drift_tolerance"]),
        "events": bool(event_max <= cfg["event_drift_tolerance"]),
    }
    latent_flags = {
        "means": bool(
            np.max(latent_mean) <= cfg["mean_drift_tolerance_reference_sd"]
        ),
        "sd": bool(np.max(latent_drift) <= cfg["relative_sd_drift_tolerance"]),
    }
    objective_flag = abs(objective_mean) <= (
        cfg["objective_change_tolerance"]
        + cfg["objective_mcse_multiplier"] * objective_mcse
    )
    return {
        "feature_names": names,
        "feature_mean_changes_reference_sd": means,
        "paired_mean_change_mcse_reference_sd": mean_mcse,
        "feature_relative_sd_changes": sd_drift,
        "feature_relative_95_width_changes": widths,
        "max_mean_change_reference_sd": float(np.max(means)),
        "worst_mean_feature": names[int(np.argmax(means))],
        "max_relative_sd_change": float(np.max(sd_drift)),
        "max_relative_95_width_change": float(np.max(widths)),
        "max_event_change": event_max,
        "events": events,
        "max_latent_mean_change_reference_sd": float(np.max(latent_mean)),
        "latent_mean_changes_reference_sd": latent_mean,
        "worst_latent_mean_index": int(np.argmax(latent_mean)),
        "max_latent_relative_sd_change": float(np.max(latent_drift)),
        "latent_relative_sd_changes": latent_drift,
        "paired_objective_change": objective_mean,
        "paired_objective_change_mcse": objective_mcse,
        "functional_flags": functional_flags,
        "latent_flags": latent_flags,
        "objective_flag": bool(objective_flag),
        "within_optimization_tolerances": all(functional_flags.values())
        and all(latent_flags.values())
        and objective_flag,
        "comparison_design": "common diagnostic draws/evaluation keys; never a posterior-accuracy certificate",
    }


def analyze(out):
    contract = json.loads((out / "frozen_contract.json").read_text())
    cfg = contract["config"]
    p = np.load(out / "frozen/reference_features.npz")["values"]
    # Repack the complete reference with the identical target/guide schema.
    import xarray as xr
    from optimize import load_target
    from run import packed_reference

    from log_psplines.diagnostics.variational import rebuild_guide

    target = load_target(out / "frozen")
    first_dir = next(out.glob("*/complete.json")).parent
    state = VIDiagnosticState.load(first_dir)
    guide = rebuild_guide(
        state, target.model, target_fingerprint=target.identity
    )
    pr = packed_reference(
        guide, xr.load_dataset(out / "frozen/reference_posterior.nc")
    )
    latent_sd = pr.reshape(-1, pr.shape[-1]).std(0, ddof=1)
    records, missing = {}, []
    for schedule, particles, seed in settings(cfg):
        name = job_name(schedule, particles, seed)
        if (out / name / "complete.json").exists():
            records[name] = json.loads(
                (out / name / "complete.json").read_text()
            )
        else:
            missing.append(name)
    summary = {
        "frozen_contract": contract,
        "completed_jobs": len(records),
        "missing_jobs": missing,
        "settings": [],
        "seed_pairs": [],
        "latent_schema": state.metadata["latent_schema"],
    }
    steps = cfg["stability_steps"]
    for schedule in cfg["schedules"]:
        for particles in cfg["optimization_particles"]:
            setting = {
                "name": schedule["name"],
                "optimization_particles": particles,
                "late_pairs": [],
                "seed_pairs": [],
                "jobs": [],
            }
            for seed in cfg["optimization_seeds"]:
                name = job_name(schedule, particles, seed)
                if name not in records:
                    continue
                record = records[name]
                by_step = {row["step"]: row for row in record["checkpoints"]}
                for first, second in zip(steps[:-1], steps[1:], strict=True):
                    result = paired_check(
                        np.load(out / name / f"checkpoint_{first}.npz"),
                        np.load(out / name / f"checkpoint_{second}.npz"),
                        by_step[first],
                        by_step[second],
                        p,
                        latent_sd,
                        cfg,
                    )
                    setting["late_pairs"].append(
                        {"seed": seed, "steps": [first, second], **result}
                    )
                final = by_step[max(cfg["checkpoints"])]
                comparison = final["comparison"]
                spectral = [
                    x
                    for x in comparison["features"]
                    if x["name"].startswith(
                        ("log_V", "band_power", "time_contrast")
                    )
                ]
                native = [
                    x["k"]
                    for x in record["native_psis"]["runs"]
                    if x.get("k") is not None
                ]
                setting["jobs"].append(
                    {
                        "seed": seed,
                        "job": name,
                        "negative_elbo": final["fixed_objective"][
                            "negative_elbo"
                        ],
                        "objective_mcse": final["fixed_objective"]["mcse"],
                        "independent_negative_elbo": final[
                            "independent_objective"
                        ]["negative_elbo"],
                        "max_spectral_mean_error_reference_sd": max(
                            abs(x["standardized_mean_difference"])
                            for x in spectral
                        ),
                        "median_spectral_sd_ratio": float(
                            np.median([x["sd_ratio"] for x in spectral])
                        ),
                        "max_all_feature_mean_error_reference_sd": max(
                            abs(x["standardized_mean_difference"])
                            for x in comparison["features"]
                        ),
                        "max_event_error": max(
                            abs(x["difference"]) for x in comparison["events"]
                        ),
                        "functional_mmd": final["functional_mmd"],
                        "latent_mmd": final["latent_mmd"],
                        "native_k": native,
                        "raw_weight_ess_fraction_min": min(
                            x["raw_ess_fraction"] for x in record["weights"]
                        ),
                        "smoothed_weight_ess_fraction_min": min(
                            x.get("smoothed_ess_fraction", 0)
                            for x in record["weights"]
                        ),
                        "wall_seconds": record["wall_seconds"],
                        "fit_workflow_seconds": record["fit_workflow_seconds"],
                    }
                )
            for seed_a, seed_b in itertools.combinations(
                cfg["optimization_seeds"], 2
            ):
                a, b = (
                    job_name(schedule, particles, seed)
                    for seed in (seed_a, seed_b)
                )
                if a not in records or b not in records:
                    continue
                step = max(cfg["checkpoints"])
                result = paired_check(
                    np.load(out / a / f"checkpoint_{step}.npz"),
                    np.load(out / b / f"checkpoint_{step}.npz"),
                    records[a]["checkpoints"][-1],
                    records[b]["checkpoints"][-1],
                    p,
                    latent_sd,
                    cfg,
                )
                setting["seed_pairs"].append(
                    {"seeds": [seed_a, seed_b], **result}
                )
            for key in ("late_pairs", "seed_pairs"):
                setting[key + "_within_tolerances"] = all(
                    x["within_optimization_tolerances"] for x in setting[key]
                ) and len(setting[key]) == (6 if key == "late_pairs" else 3)
            setting["stable_and_repeatable_screen"] = (
                setting["late_pairs_within_tolerances"]
                and setting["seed_pairs_within_tolerances"]
            )
            summary["settings"].append(setting)
    baseline = []
    source = Path(contract["source_reference"])
    for seed in cfg["optimization_seeds"]:
        if not (out / f"constant_0.01_p1_seed{seed}/complete.json").exists():
            continue
        old = VIDiagnosticState.load(source / f"diag_seed{seed}")
        new = VIDiagnosticState.load(out / f"constant_0.01_p1_seed{seed}")
        a, b = old.checkpoints["15000"], new.checkpoints["15000"]
        differences = {
            name: float(
                np.max(np.abs(np.asarray(a[name]) - np.asarray(b[name])))
            )
            for name in a
        }
        baseline.append(
            {
                "seed": seed,
                "max_parameter_difference": max(differences.values()),
                "per_parameter_difference": differences,
            }
        )
    summary["historical_baseline_at_15000"] = baseline
    prefix_parity = []
    if "prefix_reference" in cfg:
        prefix_source = Path(cfg["prefix_reference"])
        for seed in cfg["optimization_seeds"]:
            for particles in cfg["optimization_particles"]:
                old = VIDiagnosticState.load(
                    prefix_source
                    / f"{cfg['prefix_schedule']}_p{particles}_seed{seed}"
                )
                new = VIDiagnosticState.load(
                    out / job_name(cfg["schedules"][0], particles, seed)
                )
                step = str(cfg["prefix_step"])
                differences = [
                    float(
                        np.max(
                            np.abs(
                                np.asarray(value)
                                - np.asarray(new.checkpoints[step][name])
                            )
                        )
                    )
                    for name, value in old.checkpoints[step].items()
                ]
                prefix_parity.append(
                    {
                        "seed": seed,
                        "step": int(step),
                        "max_parameter_difference": max(differences),
                    }
                )
    summary["extension_prefix_parity"] = prefix_parity
    write_json(out / "optimization_summary.json", summary)
    return summary, records


def fmt(value):
    return f"{value:.3g}"


def write_report(summary, records, out, report):
    cfg = summary["frozen_contract"]["config"]
    stable = [
        f"{x['name']} / {x['optimization_particles']} particles"
        for x in summary["settings"]
        if x["stable_and_repeatable_screen"]
    ]
    schedule_descriptions = []
    for schedule in cfg["schedules"]:
        if schedule["type"] == "constant":
            schedule_descriptions.append(
                f"constant rate {schedule['peak_lr']}"
            )
        else:
            decay = schedule.get("decay_steps", max(cfg["checkpoints"]))
            text = f"{schedule['warmup_steps']}-step warmup from {schedule['initial_lr']} to {schedule['peak_lr']}, then cosine decay to {schedule['end_lr']} at step {decay}"
            if decay < max(cfg["checkpoints"]):
                text += f", followed by a constant {schedule['end_lr']} tail through {max(cfg['checkpoints'])}"
            schedule_descriptions.append(text)
    protocol = "; ".join(schedule_descriptions)
    intervals = " and ".join(
        f"{a // 1000}k→{b // 1000}k"
        for a, b in zip(
            cfg["stability_steps"][:-1],
            cfg["stability_steps"][1:],
            strict=True,
        )
    )
    parity = ""
    if summary["historical_baseline_at_15000"]:
        parity += f" Historical constant-rate/one-particle fits reproduced their 15,000-step learned parameters with maximum differences {[x['max_parameter_difference'] for x in summary['historical_baseline_at_15000']]}."
    if summary["extension_prefix_parity"]:
        parity += f" The extension reproduces the selected initial experiment's {cfg['prefix_step']}-step learned parameters with maximum differences {[x['max_parameter_difference'] for x in summary['extension_prefix_parity']]}. This extension was selected after inspecting the initial 18-fit experiment and remains exploratory."
    lines = [
        "# Fixed exact time-varying target: optimization follow-up",
        "",
        "Run date: 2026-10-02 (Pacific/Auckland). This experiment asks whether the unchanged diagonal guide can be fitted repeatably and stably before attributing remaining error to its shape.",
        "",
        f"Completed {summary['completed_jobs']}/{len(settings(cfg))} planned fits. Settings within all predeclared late-drift and seed-repeatability screens: **{'; '.join(stable) if stable else 'none'}**. These are development screens, not convergence proofs or posterior validation.",
        "",
        "## Frozen target and reference",
        "",
        "Observations, counts, physical coordinates, 8×6 cubic basis, eigenbasis prior, roughness-scale priors, likelihood, noncentered parameterization and float64 precision are unchanged. Inputs were loaded from the previous exact control; no data generation or new NUTS inference occurred. The previously accepted four-chain reference is copied with SHA256 identities, and its accepted screen remains conditional evidence rather than a convergence proof.",
        "",
        f"Target fingerprint: `{summary['frozen_contract']['target_fingerprint']}`. Rebuilt log joint agreed with 128 archived unconstrained density evaluations to maximum absolute difference {summary['frozen_contract']['density_max_absolute_difference']:.3g}. Numerical basis/penalty arrays match exactly.{parity} Input/reference file hashes are in frozen_contract.json.",
        "",
        "## Optimization protocol",
        "",
        f"One fixed diagonal guide; Adam and global gradient clipping at 1 are unchanged. Schedules: {protocol}. Training particles: {cfg['optimization_particles']}; seeds: {cfg['optimization_seeds']}. Early stopping is disabled. Checkpoints: {cfg['checkpoints']} actual updates. Equal update budgets have different particle work; timings are reported separately.",
        "",
        "Evaluation particles are independent of training particles: 256 per evaluation ×8 fixed keys, with independent evaluations on three other keys. This reduces objective noise compared with the previous 32×4 audit. Checkpoint comparisons use 4096 full joint guide draws and the same draw key, feature definitions, event thresholds, MMD transforms and bandwidths as the earlier study. Common random numbers reduce comparison noise without changing the fitted guide. All quantities and key counts are saved.",
        "",
        f"Optimization screens require both {intervals} intervals in all seeds, plus all three final seed pairs: all 10 feature means and all 50 unconstrained latent means change by ≤0.1 reference SD; their SD changes are ≤5%; declared event probability changes are ≤0.01; paired negative-ELBO changes are ≤0.2 plus twice their paired MCSE. Interval-width changes are measured separately. Reference uncertainty affects the SD scale; it is not counted as another independent replicate.",
        "",
        "## Stability and repeatability",
        "",
        "| Schedule | Training particles | Max late mean drift / ref SD | Max late relative SD drift | Max latent mean drift / ref SD | Max seed mean spread / ref SD | Max seed relative SD spread | Late pairs within screens | Seed pairs within screens |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for x in summary["settings"]:
        late, seed = x["late_pairs"], x["seed_pairs"]
        if not late or not seed:
            continue
        lines.append(
            f"| {x['name']} | {x['optimization_particles']} | {fmt(max(z['max_mean_change_reference_sd'] for z in late))} | {fmt(max(z['max_relative_sd_change'] for z in late))} | {fmt(max(z['max_latent_mean_change_reference_sd'] for z in late))} | {fmt(max(z['max_mean_change_reference_sd'] for z in seed))} | {fmt(max(z['max_relative_sd_change'] for z in seed))} | {sum(z['within_optimization_tolerances'] for z in late)}/6 | {sum(z['within_optimization_tolerances'] for z in seed)}/3 |"
        )
    lines += [
        "",
        "All maxima include roughness scales, contrasts and D distributions, not only plotted spectrum points. Detailed paired objective changes/MCSE, feature identities, latent SD drift, interval-width drift, probabilities and failed flags are in optimization_summary.json. Constant indicators carry an unresolved-tail status and cannot validate tail probabilities.",
        "",
        "## Remaining NUTS discrepancies and importance proposals",
        "",
        f"At {max(cfg['checkpoints']):,} steps, ranges cover the three optimization seeds. Agreement with NUTS and usefulness for importance sampling remain separate from optimizer stability.",
        "",
        "| Schedule | Particles | Negative ELBO range | Max spectral mean error / ref SD | Median spectral SD ratio range | Max all-feature mean error / ref SD | Functional MMD² range / NUTS baseline | Native k range | Minimum raw/smoothed ESS fraction | Median fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary["settings"]:
        jobs = x["jobs"]
        if not jobs:
            continue

        def interval(field, jobs=jobs):
            values = [z[field] for z in jobs]
            if field == "negative_elbo":
                return f"{min(values):.3f}–{max(values):.3f}"
            return fmt(min(values)) + "–" + fmt(max(values))

        mmd = [z["functional_mmd"]["vi_vs_nuts"] for z in jobs]
        k = [value for z in jobs for value in z["native_k"]]
        lines.append(
            f"| {x['name']} | {x['optimization_particles']} | {interval('negative_elbo')} | {fmt(max(z['max_spectral_mean_error_reference_sd'] for z in jobs))} | {interval('median_spectral_sd_ratio')} | {fmt(max(z['max_all_feature_mean_error_reference_sd'] for z in jobs))} | {fmt(min(mmd))}–{fmt(max(mmd))}/{fmt(jobs[0]['functional_mmd']['nuts_vs_nuts'])} | {fmt(min(k))}–{fmt(max(k))} | {fmt(min(z['raw_weight_ess_fraction_min'] for z in jobs))}/{fmt(min(z['smoothed_weight_ess_fraction_min'] for z in jobs))} | {fmt(float(np.median([z['fit_workflow_seconds'] for z in jobs])))} |"
        )
    lines += [
        "",
        "The unchanged MMD calculation uses a disjoint reference subset to train one whitening transform for both methods, three predeclared RBF bandwidths and 512 matched draws. NUTS-versus-NUTS/VI-versus-VI baselines are descriptive; there are no IID permutation p-values. Reference autocorrelation and MCSE remain available in the frozen reference record. Repeated final PSIS uses 4096 independent guide particles ×3 seeds, separately from plotting and objective evaluations; weight ESS is not MCMC ESS. No reweighted posterior was produced.",
        "",
        "## Interpretation and limitations",
        "",
        "A decaying learning rate can make consecutive checkpoints move little while leaving the optimizer in different seed-dependent locations. A flat objective, a stable last checkpoint or an improved mean therefore does not answer repeatability alone. The independent objective evaluations and all seed pairs are retained to expose this distinction.",
        "",
        "This experiment varies two optimization controls jointly in a small factorial design. It does not isolate irreducible diagonal-family error unless a setting becomes stable and repeatable. No new parameterization, full-covariance fit, flow, transform change, predictive adequacy test or calibration study was run. Development seeds were reused deliberately for matched diagnosis; success here would still need untouched-seed confirmation before a validated protocol is claimed.",
        "",
        f"Artifacts: `{out.resolve()}`. Total sum of per-job workflow times: {sum(z['wall_seconds'] for z in records.values()):.1f} seconds; medians in the table cover fit initialization/compilation, optimization, checkpoint evaluations, guide draws and final diagnostics. Additional reconstruction/comparison work is saved per checkpoint. Process startup and freezing/report costs are separate. Full joint packed checkpoint draws, final constrained posterior draws and numerical learned guide parameters are retained.",
        "",
        "Reproduce with `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/optimize.py --reference <saved-exact-target-directory> --out <new-directory>`. Analyze saved fits with `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/optimization_report.py --out <artifact-directory>`.",
        "",
    ]
    verification = out / "verification.json"
    if verification.exists():
        lines.extend(
            [
                "## Verification",
                "",
                json.loads(verification.read_text())["summary"],
                "",
            ]
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def plot(summary, records, out, *, late=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cfg = summary["frozen_contract"]["config"]
    counts = cfg["optimization_particles"]
    fig, axes = plt.subplots(
        len(counts),
        3,
        figsize=(13, 3.5 * len(counts)),
        sharex=True,
        squeeze=False,
    )
    labels = cfg["schedules"]
    colors = ["#3f79a5", "#dc9142", "#4c966e"][: len(labels)]
    for row, particles in enumerate(counts):
        for color, schedule in zip(colors, labels, strict=True):
            for seed in cfg["optimization_seeds"]:
                record = records.get(job_name(schedule, particles, seed))
                if not record:
                    continue
                checkpoints = [
                    z
                    for z in record["checkpoints"]
                    if not late or z["step"] >= min(cfg["stability_steps"])
                ]
                steps = np.array([z["step"] for z in checkpoints]) / 1000
                loss = [
                    z["fixed_objective"]["negative_elbo"] for z in checkpoints
                ]
                mean = [
                    max(
                        abs(x["standardized_mean_difference"])
                        for x in z["comparison"]["features"]
                    )
                    for z in checkpoints
                ]
                sd = [
                    np.median(
                        [
                            x["sd_ratio"]
                            for x in z["comparison"]["features"]
                            if x["name"].startswith(
                                ("log_V", "band_power", "time_contrast")
                            )
                        ]
                    )
                    for z in checkpoints
                ]
                for ax, values in zip(
                    axes[row], (loss, mean, sd), strict=True
                ):
                    ax.plot(
                        steps, values, color=color, alpha=0.7, linewidth=1.3
                    )
        axes[row, 0].set_ylabel(
            f"{particles} training particle{'s' if particles > 1 else ''}\nNegative ELBO"
        )
        axes[row, 1].set_ylabel(
            "Max mean error / reference SD\n(all declared features)"
        )
        axes[row, 2].set_ylabel("Median spectral SD ratio\nVI / reference")
        axes[row, 2].axhline(1, color="#555555", linestyle="--", linewidth=0.8)
        for ax in axes[row]:
            ax.axvspan(
                min(cfg["stability_steps"]) / 1000,
                max(cfg["stability_steps"]) / 1000,
                color="#eeeeee",
                alpha=0.2,
            )
            ax.set_xlabel("Actual optimizer steps (thousands)")
            ax.grid(alpha=0.15)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle(
        "Same exact TV target and diagonal guide: "
        + ("late checkpoints" if late else "optimization settings"),
        fontsize=14,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, label=label["name"])
            for color, label in zip(colors, labels, strict=True)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "Three matched optimization seeds per setting. Shaded region: late stability checkpoints. Stability does not establish posterior accuracy.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    fig.savefig(
        out / ("optimization_late.png" if late else "optimization.png"),
        dpi=170,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs/development/vi_nuts_validation/optimization_report.md",
    )
    args = parser.parse_args()
    summary, records = analyze(args.out)
    write_report(summary, records, args.out, args.report)
    plot(summary, records, args.out)
    plot(summary, records, args.out, late=True)


if __name__ == "__main__":
    main()
