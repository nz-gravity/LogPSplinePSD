"""Offline report and figures for the bounded native extension studies."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from comparison import primary_indices
from extension_targets import ROOT, feature_values
from extensions import config, load_base, select_target
from study_common import write_json


def bounds(x):
    return [float(np.min(x)), float(np.max(x))] if len(x) else None


def render_case(out, case, smoothing, analysis):
    target = select_target(load_base(out, case), smoothing)
    directory = out / case / smoothing
    refdir = (
        directory
        / "reference"
        / json.loads((directory / "reference/selection.json").read_text())[
            "attempt"
        ]
    )
    reference = np.load(refdir / "features.npz")
    names = reference["names"].tolist()
    series = [("NUTS", reference["values"], "black")]
    colors = ["#167aaa", "#d6681d", "#349567"]
    for seed, color in zip((7101, 7102, 7103), colors, strict=True):
        p = directory / "vi" / str(seed) / "posterior.nc"
        if p.exists():
            values, labels = feature_values(target, xr.load_dataset(p))
            assert labels == names
            series.append((str(seed), values, color))
    points = np.arange(1, 10) / 20
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    panels = (
        [(f"log_S_c{c}_f", f"Channel {c} PSD", True) for c in range(2)]
        + [
            ("csd_re_f", "Real cross spectrum S[1,0]", False),
            ("csd_im_f", "Imaginary cross spectrum S[1,0]", False),
            ("coherence_f", "Squared coherence", False),
        ]
        if case == "matrix"
        else [
            (f"log_S_t{time:g}_f", f"Variance surface at t={time:g}", True)
            for time in (0.25, 0.5, 0.75)
        ]
        + [("temporal_contrast_f", "Log variance: t=.75 minus t=.25", False)]
    )
    for axis, (prefix, title, exponentiate) in zip(
        axes.flat, panels, strict=False
    ):
        ix = [names.index(prefix + f"{f:g}") for f in points]
        for label, values, color in series:
            draws = values.reshape(-1, len(names))[:, ix]
            if exponentiate:
                draws = np.exp(draws)
            lo, med, hi = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
            axis.plot(points, med, color=color, label=label, lw=1.4)
            axis.fill_between(points, lo, hi, color=color, alpha=0.13)
        axis.set(
            title=title,
            xlabel="Frequency",
            ylabel="Variance"
            if case == "time_varying" and exponentiate
            else "Native PSD"
            if exponentiate
            else "Functional value",
        )
        if exponentiate:
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    axes.flat[0].legend(title="90% intervals", fontsize=9)
    fig.suptitle(
        f"{case.replace('_', ' ')} / {smoothing}: all joint draws, selected frequency points"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"posterior_functionals.{suffix}", dpi=150)
    plt.close(fig)
    primary = primary_indices(names)
    fig, axis = plt.subplots(figsize=(14, 4), constrained_layout=True)
    axis.axhspan(
        0.9, 1.1, color="#bce0c1", alpha=0.5, label="Declared SD screen"
    )
    for seed, color in zip(analysis["seeds"], colors, strict=False):
        if seed.get("status") != "complete":
            continue
        rows = seed["final"]["mc_uncertainty"]
        axis.errorbar(
            np.arange(len(primary)),
            [rows[i]["sd_ratio"] for i in primary],
            yerr=[2 * rows[i]["sd_ratio_mcse"] for i in primary],
            fmt=".",
            capsize=2,
            color=color,
            label=str(seed["seed"]),
            alpha=0.75,
        )
    axis.axhline(1, color="black", lw=0.8)
    axis.set(
        xticks=np.arange(len(primary)),
        xticklabels=[names[i] for i in primary],
        ylabel="VI / NUTS posterior SD",
        title=f"{case} / {smoothing}: primary uncertainty and two-MCSE intervals",
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.legend(ncol=4)
    fig.savefig(directory / "uncertainty_ratios.png", dpi=150)
    fig.savefig(directory / "uncertainty_ratios.pdf")
    plt.close(fig)


def summarize(out, report):
    started = perf_counter()
    cfg = config(out)
    references = []
    seeds = []
    analyses = {}
    csvrows = []
    timings = []
    attempts = [
        json.loads(line)
        for line in (out / "attempts.jsonl").read_text().splitlines()
    ]
    for path in sorted(out.glob("*/*/reference/*/complete.json")):
        x = json.loads(path.read_text())
        h = x["health"]
        references.append(
            {
                "case": x["case"],
                "smoothing": x["smoothing"],
                "attempt": x["attempt"],
                "status": x["status"],
                "rhat_max": max(h["rhat"]),
                "bulk_ess_min": min(h["ess_bulk"]),
                "tail_ess_min": min(h["ess_tail"]),
                "bfmi_min": min(h["bfmi"]),
                "divergences": h["divergences"],
                "depth_saturation": h["depth_saturation"],
                "primary_mean_mcse_sd_max": x["primary_mean_mcse_sd_max"],
                "inherited": bool(x.get("inherited_from")),
                "path": str(path.resolve()),
                "wall_seconds": x["wall_seconds"],
            }
        )
        timings.append(
            {
                "case": x["case"],
                "smoothing": x["smoothing"],
                "kind": "reference",
                "identifier": x["attempt"],
                "wall_seconds": x["wall_seconds"],
                **x["timings"],
            }
        )
    for case in cfg["cases"]:
        for smoothing in cfg["smoothing"]:
            path = out / case / smoothing / "analysis.json"
            if not path.exists():
                continue
            analysis = json.loads(path.read_text())
            analyses[f"{case}/{smoothing}"] = {
                k: analysis.get(k)
                for k in (
                    "status",
                    "gate",
                    "target_fingerprint",
                    "analysis_seconds",
                )
            }
            for r in analysis.get("seeds", []):
                if r.get("status") != "complete":
                    seeds.append({"case": case, "smoothing": smoothing, **r})
                    continue
                rows = r["final"]["mc_uncertainty"]
                names = [a["name"] for a in rows]
                primary = [rows[i] for i in primary_indices(names)]
                complete = json.loads(
                    (
                        out
                        / case
                        / smoothing
                        / "vi"
                        / str(r["seed"])
                        / "complete.json"
                    ).read_text()
                )
                metadata = json.loads(
                    (
                        out
                        / case
                        / smoothing
                        / "vi"
                        / str(r["seed"])
                        / "guide.json"
                    ).read_text()
                )["metadata"]
                weights = complete["weights"]
                bad = [
                    {
                        k: a[k]
                        for k in (
                            "name",
                            "mean_status",
                            "sd_status",
                            "standardized_mean_difference",
                            "sd_ratio",
                            "sd_ratio_mcse",
                        )
                    }
                    for a in primary
                    if a["mean_status"] != "within_screen"
                    or a["sd_status"] != "within_screen"
                ]
                summary = {
                    "case": case,
                    "smoothing": smoothing,
                    "seed": r["seed"],
                    "status": r["status"],
                    "agreement": r["final_primary_agreement"],
                    "stability": r["late_optimization_stability"],
                    "coefficient_agreement": r["coefficient_agreement"],
                    "max_abs_mean_reference_sd": max(
                        abs(a["standardized_mean_difference"]) for a in primary
                    ),
                    "primary_sd_ratio_range": bounds(
                        [a["sd_ratio"] for a in primary]
                    ),
                    "interval_width_ratios": {
                        str(level): bounds(
                            [
                                a[f"interval_width_ratio_{level}"]
                                for a in r["final"]["features"]
                                if a["name"] in {b["name"] for b in primary}
                            ]
                        )
                        for level in (50, 90, 95)
                    },
                    "unresolved_or_outside": bad,
                    "packed_psis_k": bounds(
                        [a["k"] for a in weights if a.get("k") is not None]
                    ),
                    "smoothed_weight_ess_fraction": bounds(
                        [
                            a["smoothed_ess_fraction"]
                            for a in weights
                            if a.get("smoothed_ess_fraction") is not None
                        ]
                    ),
                    "native_psis_k": bounds(
                        [
                            a["k"]
                            for a in complete["native_psis"].get("runs", [])
                            if a.get("k") is not None
                        ]
                    ),
                    "functional_mmd": r["final"]["functional_mmd"],
                    "latent_mmd": r["final"]["latent_mmd"],
                    "reference_covariance_eigenvalues_in_guide_coordinates": bounds(
                        r["final"]["guide_geometry"][
                            "nuts_covariance_eigenvalues_in_q_coordinates"
                        ]
                    ),
                    "max_functional_correlation_difference": float(
                        np.max(
                            np.abs(
                                np.asarray(r["final"]["vi_correlation"])
                                - np.asarray(r["final"]["nuts_correlation"])
                            )
                        )
                    ),
                    "density_roundtrip_max_error": r[
                        "guide_density_roundtrip_max_error"
                    ],
                    "refined": bool(r["refinement"]),
                    "latent_schema": metadata["latent_schema"],
                    "late_intervals": [
                        {
                            k: s[k]
                            for k in (
                                "interval",
                                "status",
                                "max_mean_change_reference_sd",
                                "max_relative_sd_change",
                                "paired_objective_change",
                                "paired_objective_mcse",
                            )
                        }
                        for s in r["stability"]
                    ],
                    "wall_seconds": complete["wall_seconds"],
                }
                seeds.append(summary)
                timings.append(
                    {
                        "case": case,
                        "smoothing": smoothing,
                        "kind": "vi",
                        "identifier": r["seed"],
                        "wall_seconds": complete["wall_seconds"],
                        **complete["timings"]["shared_vi"],
                        "persistence_seconds": complete["timings"][
                            "persistence_seconds"
                        ],
                    }
                )
                for row in rows:
                    csvrows.append(
                        {
                            "case": case,
                            "smoothing": smoothing,
                            "seed": r["seed"],
                            **{
                                k: v
                                for k, v in row.items()
                                if not isinstance(v, dict)
                            },
                        }
                    )
            if any(
                r.get("status") == "complete"
                for r in analysis.get("seeds", [])
            ):
                render_case(out, case, smoothing, analysis)
    # Every not-launched seed is explicit, including locked hierarchy.
    for case in cfg["cases"]:
        for smoothing in cfg["smoothing"]:
            for seed in cfg["seeds"]:
                if not any(
                    s["case"] == case
                    and s["smoothing"] == smoothing
                    and s["seed"] == seed
                    for s in seeds
                ):
                    seeds.append(
                        {
                            "case": case,
                            "smoothing": smoothing,
                            "seed": seed,
                            "status": "not_run_fixed_gate"
                            if smoothing == "hierarchical"
                            and not analyses.get(f"{case}/fixed", {})
                            .get("gate", {})
                            .get("exploratory_hierarchy_unlock")
                            else "not_run_unresolved_prerequisite",
                        }
                    )
    figure_directory = report.parent / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    for case in cfg["cases"]:
        directory = out / case / "fixed"
        for kind in ("posterior_functionals", "uncertainty_ratios"):
            source = directory / f"{kind}.png"
            if source.exists():
                shutil.copyfile(
                    source,
                    figure_directory / f"extensions_{case}_fixed_{kind}.png",
                )
    result = {
        "references": references,
        "seeds": seeds,
        "analyses": analyses,
        "attempts": attempts,
        "timings": timings,
        "offline_report_seconds": perf_counter() - started,
    }
    write_json(out / "summary.json", result)
    for name, records in (("comparison_rows", csvrows), ("timings", timings)):
        if records:
            keys = list(dict.fromkeys(k for row in records for k in row))
            with (out / f"{name}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
    lines = [
        "# Native multivariate and scalar time-varying VI/NUTS study",
        "",
        "This is a bounded extension on `hacking-vi-nuts-validation`. The optimizer and MC-aware screens are inherited from the completed stationary continuation; observations, numerical bases, penalties, priors and likelihoods are identical within each fixed/hierarchical comparison. Fixed and inferred smoothing are distinct statistical targets.",
        "",
        "Multivariate fixed smoothing passed final primary and coefficient agreement, both late stability screens and all three seed-pair comparisons. Its inferred-smoothing reference remained unusable after one repair, so hierarchical VI was not run. The time-varying fixed control passed final agreement only for seed 7101; every seed failed late stability and the seed pairs failed repeatability. Its hierarchy stayed locked. These results support the multivariate fixed-target candidate, not a general stationary or time-varying VI preset.",
        "",
        "Exactly four new four-chain NUTS attempts and six full-covariance 60k VI fits ran. No VI attempt failed to execute, no saved-guide precision refinement was triggered, and all six fits/checkpoints/density reconstructions were preserved. Two diagnosed fixed references are usable. The two failed hierarchical multivariate reference attempts are retained; the inherited time-varying hierarchical reference was not opened for a scientific comparison because the fixed gate failed.",
        "",
        "## What ran",
        "",
        "| Case | Smoothing | Seed | Final posterior agreement | Late optimization stability | Physical coefficient agreement |",
        "|---|---|---|---|---|---|",
    ]
    for s in seeds:
        lines.append(
            f"| {s['case']} | {s['smoothing']} | {s['seed']} | {s.get('agreement', s['status'])} | {s.get('stability', 'not assessed')} | {s.get('coefficient_agreement', 'not assessed')} |"
        )
    lines += [
        "",
        "Seed repeatability is assessed independently of accuracy and within-seed stability:",
        "",
    ]
    for key, x in analyses.items():
        gate = x.get("gate") or {}
        lines.append(
            f"- {key}: repeatability `{gate.get('seed_repeatability', 'unavailable')}`, repeatable recipe `{gate.get('repeatable_recipe', False)}`, hierarchy unlock `{gate.get('exploratory_hierarchy_unlock', False)}`."
        )
    lines += [
        "",
        "## Reference diagnoses",
        "",
        "| Case/target | Attempt | Status | Max Rhat | Min bulk/tail ESS | Min BFMI | Divergences/cap hits | Max primary mean MCSE/SD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in references:
        lines.append(
            f"| {r['case']}/{r['smoothing']} | {r['attempt']} | {r['status']} | {r['rhat_max']:.5f} | {r['bulk_ess_min']:.0f}/{r['tail_ess_min']:.0f} | {r['bfmi_min']:.3f} | {r['divergences']}/{r['depth_saturation']} | {r['primary_mean_mcse_sd_max']:.4f} |"
        )
    lines += [
        "",
        "Four chains run sequentially in float64 with dense mass. Initial settings: 1000 warmup/2000 draws per chain, acceptance .95, depth 10. One retained repair per target: 2000/4000, acceptance .99, depth 10. Health requires Rhat <1.01, bulk/tail ESS >=400, BFMI >=.3, no divergences or cap hits, mean MCSE/SD <=.05; primary-function precision additionally requires <=.02. A failed reference prevents scientific VI-versus-NUTS claims for that target.",
        "",
        "## Posterior uncertainty and joint diagnostics",
        "",
        "| Case/target/seed | Max absolute mean error / reference SD | Primary SD ratio range | Packed PSIS k range | Smoothed weight ESS fraction |",
        "|---|---|---|---|---|",
    ]
    for s in seeds:
        if s["status"] != "complete":
            continue

        def fmt(pair):
            return (
                "unavailable"
                if pair is None
                else f"{pair[0]:.3f}–{pair[1]:.3f}"
            )

        lines.append(
            f"| {s['case']}/{s['smoothing']}/{s['seed']} | {s['max_abs_mean_reference_sd']:.4f} | {fmt(s['primary_sd_ratio_range'])} | {fmt(s['packed_psis_k'])} | {fmt(s['smoothed_weight_ess_fraction'])} |"
        )
    lines += [
        "",
        "The primary mean screen is ±.1 reference SD and the SD-ratio screen [.9,1.1]. Both retain two-MCSE uncertainty. Late intervals are 40k→50k and 50k→60k, with paired joint draw keys; mean drift is limited to .1 reference SD and symmetric SD drift to .05. Objective changes use .2 plus two paired MCSE. Seed comparisons use unpaired posterior MC uncertainty. Unavailable or precision-limited outcomes never pass. One optional saved-guide precision refinement uses 16,384 checkpoint and 32,768 final draws; base estimates remain saved.",
        "",
        "Every final comparison retains 50/90/95% interval-width ratios, standardized Wasserstein distances, coefficient marginals/correlations, roughness penalty energies and MMD on functionals and packed native latents. MMD uses disjoint reference training/chain subsets and matched counts with NUTS–NUTS and VI–VI baselines; it is descriptive for autocorrelated chains and has no IID permutation p-value. Joint PSIS evaluates full priors, hyperpriors, likelihood and Jacobians together on three independent 4096-particle diagnostic streams. PSIS is not a posterior-accuracy pass criterion. Raw VI draws are never reweighted.",
        "",
        "### Joint and interval summaries",
        "",
    ]
    lines += [
        "| Case/target/seed | 90% width ratio | 95% width ratio | Native PSIS k | Functional MMD²: VI–NUTS / NUTS–NUTS / VI–VI | Latent MMD²: same order |",
        "|---|---|---|---|---|---|",
    ]
    for item in seeds:
        if item["status"] != "complete":
            continue

        def triplet(key, item=item):
            m = item[key]
            return " / ".join(
                f"{m[k]:.3g}"
                for k in ("vi_vs_nuts", "nuts_vs_nuts", "vi_vs_vi")
            )

        lines.append(
            f"| {item['case']}/{item['smoothing']}/{item['seed']} | {fmt(item['interval_width_ratios']['90'])} | {fmt(item['interval_width_ratios']['95'])} | {fmt(item['native_psis_k'])} | {triplet('functional_mmd')} | {triplet('latent_mmd')} |"
        )
    lines += [
        "",
        "MMD values average the frozen kernel bandwidths. Negative values can occur for this unbiased estimator. The multivariate VI–NUTS values are of the same order as the disjoint NUTS-chain baseline. The time-varying values exceed its near-zero reference baseline, supporting a retained joint discrepancy even though its marginal SDs are close. This is descriptive evidence, not a formal significance test. No joint-diagnostic threshold was silently added to the continuation gate.",
        "",
        "All primary time-varying SD and interval-width point ratios are close to one; its resolved discrepancies are in means and late mean drift. Maximum 40k→50k / 50k→60k mean drift in reference SD units was .138/.197 (7101), .148/.234 (7102), and .184/.106 (7103). Changes in paired objective stayed inside the inherited tolerance, so a quiet objective would have missed the instability. The multivariate hierarchical repair removed divergences but had three trajectory-cap hits and primary mean MCSE/SD .0340; physical/log sigma_theta_re_1_0 precision was .0340/.0317, and log sigma_delta_1 was .0219. Both sampler depth and named roughness precision remain unresolved.",
        "",
        "### Primary feature details",
        "",
    ]
    for s in seeds:
        if s["status"] != "complete":
            continue
        bad = s["unresolved_or_outside"]
        lines.append(
            f"- {s['case']}/{s['smoothing']}/{s['seed']}: "
            + (
                "all primary mean and SD screens resolved within bounds."
                if not bad
                else f"{len(bad)} primary features have unresolved/outside fields; see `summary.json` and `comparison_rows.csv`."
            )
        )
        for r in bad[:12]:
            lines.append(
                f"  - `{r['name']}`: mean {r['mean_status']}, SD {r['sd_status']}; SD ratio {r['sd_ratio']:.3f} ± {2 * r['sd_ratio_mcse']:.3f} (two MCSE)."
            )
    lines += [
        "",
        "## Frozen inputs, scope and budgets",
        "",
        "The multivariate observation is one two-channel stationary VAR(1) development record (seed 62001, 4096 samples, dt=1), initialized from its Lyapunov stationary covariance, without realization normalization. Eight rectangular FFT blocks produce 255 positive non-Nyquist bins, duration 512, no taper, detrending, eigenvalue floor or coarse-graining. The native centered modified-Cholesky target uses four cubic eight-coefficient splines, integrated second-derivative penalties and HalfNormal(1.28) scales. Fixed scales are each 0.86334688025. The joint model exactly sums existing native channel models; it uses one joint full Gaussian rather than the public fitter's product of per-channel guides. This validates the prepared native target and optimizer, not an untested public preset.",
        "",
        "The scalar time-varying target reuses `runs/vi-nuts-validation-v2/exact_time_varying/target.npz`: 33 time cells ×33 frequency cells (1089 cells, each count=2), native 8×6 tensor basis, saved penalties, exact independent Gamma powers and counts. Its physical target fingerprint is `1e7a3be31733307e19296b3a4230829b16065b39f9ac7c405db465e46ae23bdb`. Native noncentered coordinates, HalfNormal(10), null precision 1e-4 and ridge 1e-6 are unchanged. Fixed scales are each 6.74489750196. This is a variance/power control; integrated band variance is a model functional, not raw-process calibrated band variance. No moving-periodogram fit or locality approximation is assessed here.",
        "",
        "Primary quantities cover nine frequencies, three bands integrated on 129 quadrature nodes, both multivariate autospectra, signed real/imaginary cross spectrum S[1,0], squared coherence, three time slices and temporal contrasts. Inferred targets additionally include every physical/log roughness scale. All reconstruction uses complete joint draws, never PSD previews. Positive-definite/Hermitian matrices and coherence bounds are checked on every draw at the primary grid.",
        "",
        "Each fitted guide uses seeds 7101–7103, eight optimization particles, Adam with existing clipping, the original .001→.01 warmup (1000 updates), cosine decay with exactly the original 40k horizon and end rate 1e-4, followed by a constant 1e-4 tail through 60k. No cosine stretching, guide-checkpoint optimizer resume, diagonal rerun, parameterization sweep, flow, benchmark, untouched confirmation or AR4 reserved seed was run. At most three VI fits and one reference repair per target were permitted. A hierarchy unlock requires at least one accepted-accuracy seed stable in both late intervals; a repeatable recipe separately requires all three such seeds and all three seed pairs.",
        "",
        "The cleanup extracted shared numerical helpers and archived superseded drivers without deleting evidence. Pre-cleanup versus extracted MC statistics/comparisons/stability were bitwise equal. Thirty-one stationary source-artifact hashes still match. Density/gradient, conditioning, joint-draw integration and stop-gate tests passed. A pre-inference implementation amendment is saved separately; it changed no statistical settings or observations. The full suite passed 176 tests (28 warnings) in 64.80 s; warnings are retained in the test log.",
        "",
        "## Costs and retained artifacts",
        "",
        "| Case/target | Kind | Identifier | Wall seconds |",
        "|---|---|---|---|",
    ]
    for t in timings:
        lines.append(
            f"| {t['case']}/{t['smoothing']} | {t['kind']} | {t['identifier']} | {t['wall_seconds']:.2f} |"
        )
    lines += [
        "",
        "`timings.csv` separates initialization, the first compile-containing chunk, remaining optimization, joint posterior draws, density diagnostics and persistence. Reference timing separates warmup/sampling from persistence and feature/health analysis. Checkpoint clocks, losses, complete guide parameters, all physical posterior draws, three joint density streams, source hashes, health arrays, comparisons and failed attempts are retained. Worker and offline reporting costs are also separate. These descriptive costs are not a matched-accuracy speed comparison; historical times cannot establish a speed advantage.",
        "",
        f"Full ignored numerical archive: `{out.resolve()}`. Saved [summary](../../../runs/vi-matrix-tv/summary.json), [feature comparisons](../../../runs/vi-matrix-tv/comparison_rows.csv), [timing breakdown](../../../runs/vi-matrix-tv/timings.csv) and [attempt ledger](../../../runs/vi-matrix-tv/attempts.jsonl). These local artifacts require separate preservation; Git contains code and this report.",
        "",
        "## Figures",
        "",
        "The posterior overlays below use every saved final joint draw and 90% intervals at the declared frequency points. Their visual similarity does not replace the MC-aware mean and late-stability screens.",
        "",
        "![Multivariate fixed posterior](figures/extensions_matrix_fixed_posterior_functionals.png)",
        "",
        "![Time-varying fixed posterior](figures/extensions_time_varying_fixed_posterior_functionals.png)",
        "",
        "[Multivariate uncertainty ratios](figures/extensions_matrix_fixed_uncertainty_ratios.png) and [time-varying uncertainty ratios](figures/extensions_time_varying_fixed_uncertainty_ratios.png) show two-MCSE intervals for all primary SD ratios. Source PNG/PDF figures remain in each target directory.",
        "",
        "## Supported next experiment",
        "",
        "For the time-varying control, keep exactly these fixed-smoothing observations, basis and likelihood and compare a small declared set of lower tail learning rates and/or more optimization particles using all three seeds and the same checkpoint quantities. The observed mean drift, rather than deficient marginal SDs or the objective, makes optimization stability the first question. Lower rates or more particles are candidate explanations to test, not established fixes. Do not add hierarchy or guide flexibility before that gate passes.\n\nFor the multivariate hierarchy, first inspect saved trajectory lengths, per-chain physical/log roughness and coefficient–penalty dependence. A separately budgeted deeper/longer four-chain reference may be warranted, but the present repair budget is exhausted; no claim about hierarchical VI accuracy follows from these failed references. No parameterization change was made. The fixed multivariate result alone could motivate a future target-specific matched-accuracy study, but the combined results do not justify general efficiency benchmarking or a speed claim.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/vi-matrix-tv")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "docs/development/vi_nuts_validation/extensions_report.md",
    )
    args = parser.parse_args()
    summarize(args.out, args.report)


if __name__ == "__main__":
    main()
