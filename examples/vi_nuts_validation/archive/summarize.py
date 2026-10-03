"""Saved-draw analysis and report for A--C; does not run optimization/NUTS."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import xarray as xr
from arviz_stats.base import array_stats
from run import DEFAULT_PREVIOUS, build_target, quadrature, write_json

from log_psplines.diagnostics.variational import (
    VIDiagnosticState,
    rebuild_guide,
    runtime_provenance,
)
from log_psplines.inference.power import _collect_power_samples


def selected_reference(directory):
    selection = directory / "reference_selection.json"
    if selection.exists():
        return json.loads(selection.read_text())["prefix"]
    return (
        "reference_repair"
        if (directory / "reference_repair.json").exists()
        else "reference"
    )


def field_moments(target, posterior):
    if target.pair is None:
        band = (target.data.freq >= 0.08) & (target.data.freq <= 0.42)
        freq = target.data.freq[band]
        weights = np.asarray(posterior.weights_delta_0)
        logs = np.einsum(
            "fk,cdk->cdf", np.asarray(target.spline.basis)[band], weights
        )
        values = logs.reshape(-1, len(freq))
        return (
            values.mean(0),
            values.std(0, ddof=1),
            np.diff(np.quantile(values, [0.025, 0.975], axis=0), axis=0)[0],
            freq,
            None,
        )

    class View:
        pass

    view = View()
    view.posterior = posterior
    weights = _collect_power_samples(
        view, target.pair, target.config
    ).weights.values
    freq, time = np.linspace(0.08, 0.42, 33), np.linspace(0.1, 0.9, 17)
    bt = target.spline.time.design_at(time)
    bf = target.spline.frequency.design_at(freq)
    mean, sd, width = [np.empty((len(time), len(freq))) for _ in range(3)]
    for start in range(0, len(freq), 8):
        logs = np.einsum(
            "ti,cdij,fj->cdtf",
            bt,
            weights,
            bf[start : start + 8],
            optimize=True,
        )
        values = logs.reshape(-1, len(time), logs.shape[-1])
        mean[:, start : start + 8] = values.mean(0)
        sd[:, start : start + 8] = values.std(0, ddof=1)
        width[:, start : start + 8] = np.diff(
            np.quantile(values, [0.025, 0.975], axis=0), axis=0
        )[0]
    return mean, sd, width, freq, time


def coefficient_draws(target, posterior):
    if target.pair is None:
        return np.asarray(posterior.weights_delta_0).reshape(
            -1, posterior.weights_delta_0.shape[-1]
        )

    class View:
        pass

    view = View()
    view.posterior = posterior
    values = _collect_power_samples(
        view, target.pair, target.config
    ).weights.values
    return values.reshape(-1, int(np.prod(values.shape[-2:])))


def projection_variance(target, q, p):
    cq = np.cov(coefficient_draws(target, q), rowvar=False)
    cp = np.cov(coefficient_draws(target, p), rowvar=False)
    if target.pair is None:
        projections = np.asarray(target.spline.basis)[[23, 64, 104]]
    else:
        projections = np.stack(
            [
                np.outer(
                    target.spline.time.design_at(np.array([t]))[0],
                    target.spline.frequency.design_at(np.array([f]))[0],
                ).reshape(-1)
                for t, f in ((0.25, 0.15), (0.5, 0.25), (0.75, 0.35))
            ]
        )
    values = []
    for b in projections:
        vq, vp = float(b @ cq @ b), float(b @ cp @ b)
        dq, dp = (
            float(np.sum(b * b * np.diag(cq))),
            float(np.sum(b * b * np.diag(cp))),
        )
        values.append(
            {
                "functional_variance_ratio": vq / vp,
                "vi_diagonal_contribution": dq,
                "nuts_diagonal_contribution": dp,
                "vi_cross_contribution": vq - dq,
                "nuts_cross_contribution": vp - dp,
            }
        )
    return {
        "coordinate_system": "original spline coefficients",
        "coefficient_marginal_variance_ratio_median": float(
            np.median(np.diag(cq) / np.diag(cp))
        ),
        "projections": values,
        "interpretation": "measured covariance decomposition; not a causal attribution to guide family",
    }


def field_comparison(q, p):
    qm, qs, qw, freq, time = q
    pm, ps, pw, pfreq, ptime = p
    if not np.array_equal(freq, pfreq) or (
        time is not None and not np.array_equal(time, ptime)
    ):
        raise ValueError("physical field coordinates differ")
    valid = (
        ps
        > 100
        * np.finfo(float).eps
        * np.maximum(np.abs(pm), np.finfo(float).tiny)
    ) & (pw > 0)
    quadrature_weights = quadrature(freq)
    if time is not None:
        quadrature_weights = (
            quadrature(time)[:, None] * quadrature_weights[None, :]
        )
    quadrature_weights = quadrature_weights / quadrature_weights.sum()
    if not np.all(valid):
        return {
            "status": "zero_reference_sd",
            "invalid_cells": int(np.sum(~valid)),
        }
    error = (qm - pm) / ps
    ratios = qs / ps
    width_ratios = qw / pw
    worst = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    output = {
        "status": "descriptive_field_comparison",
        "coordinates": {"frequency": freq, "time": time},
        "normalization": "physical trapezoid weights normalized to unit area; cells are not simulation replicates",
        "standardized_mean_error_rms": float(
            np.sqrt(np.sum(quadrature_weights * error**2))
        ),
        "standardized_mean_error_max_abs": float(np.abs(error[worst])),
        "worst_region_frequency": float(freq[worst[-1]]),
        "worst_region_time": float(time[worst[0]])
        if time is not None
        else None,
        "sd_ratio_mean_area_weighted": float(
            np.sum(quadrature_weights * ratios)
        ),
        "sd_ratio_min": float(ratios.min()),
        "sd_ratio_max": float(ratios.max()),
        "interval_width_ratio_95_mean_area_weighted": float(
            np.sum(quadrature_weights * width_ratios)
        ),
    }
    return output


def add_fields(out, previous, cfg):
    analysis_started = perf_counter()
    completed = []
    for name in cfg["targets"]:
        directory = out / name
        if not (directory / "complete.json").exists():
            continue
        source_manifest = json.loads((directory / "complete.json").read_text())
        analysis_dir = directory / "field_analysis"
        analysis_dir.mkdir(exist_ok=True)
        target = build_target(name, cfg, previous, analysis_dir)
        if target.identity != source_manifest["target_fingerprint"]:
            raise ValueError(f"{name}: changed target fingerprint")
        reference_prefix = selected_reference(directory)
        posterior = xr.load_dataset(
            directory / f"{reference_prefix}_posterior.nc", engine="h5netcdf"
        )
        p = field_moments(target, posterior)
        ref_features = np.load(
            directory / f"{reference_prefix}_features.npz", allow_pickle=False
        )["values"]
        ntrain = ref_features.shape[1] // 4
        count = cfg["mmd_count"]
        mmd_ess = []
        for chain in (0, 2):
            mmd_ess.append(
                array_stats.ess(
                    ref_features[chain : chain + 1, ntrain : ntrain + count],
                    chain_axis=0,
                    draw_axis=1,
                    method="bulk",
                ).tolist()
            )
        summary = []
        for path in sorted(directory.glob("*_seed*/comparison.json")):
            before = perf_counter()
            job = path.parent
            state = VIDiagnosticState.load(job)
            guide = rebuild_guide(
                state, target.model, target_fingerprint=target.identity
            )
            record = json.loads(path.read_text())
            fields = []
            for checkpoint in record["checkpoints"]:
                step = checkpoint["actual_steps"]
                distribution = guide.get_posterior(
                    state.checkpoints[str(step)]
                )
                u = distribution.sample(
                    jax.random.PRNGKey(8301), (cfg["posterior_draws"],)
                )
                samples = guide._unpack_and_constrain(
                    u, state.checkpoints[str(step)]
                )
                q = xr.Dataset(
                    {
                        key: xr.DataArray(
                            np.asarray(values)[None],
                            dims=(
                                "chain",
                                "draw",
                                *[
                                    f"{key}_dim{i}"
                                    for i in range(np.ndim(values) - 1)
                                ],
                            ),
                        )
                        for key, values in samples.items()
                        if not key.startswith("log_likelihood")
                    }
                )
                fields.append(
                    {
                        "projection_variance_decomposition": projection_variance(
                            target, q, posterior
                        ),
                        "actual_steps": step,
                        **field_comparison(field_moments(target, q), p),
                    }
                )
            write_json(
                job / "fields.json",
                {
                    "target_fingerprint": target.identity,
                    "fields": fields,
                    "mmd_reference_subset_bulk_ess_by_feature": mmd_ess,
                    "postprocessing_seconds": perf_counter() - before,
                    "provenance": runtime_provenance(),
                },
            )
            summary.append({"job": job.name, "fields": fields})
        completed.append({"target": name, "jobs": summary})
    write_json(
        out / "field_analysis.json",
        {
            "targets": completed,
            "wall_seconds": perf_counter() - analysis_started,
            "provenance": runtime_provenance(),
        },
    )


def fmt(x):
    return "unavailable" if x is None else f"{x:.3g}"


def summarize(out, cfg, report_path):
    rows, reference_rows, records, fields = [], [], [], {}
    for name in cfg["targets"]:
        directory = out / name
        if not (directory / "complete.json").exists():
            continue
        reference_prefix = selected_reference(directory)
        ref = json.loads((directory / f"{reference_prefix}.json").read_text())
        h = ref["health"]
        reference_rows.append(
            {
                "target": name,
                "status": ref["status"],
                "divergences": h["divergences"],
                "rhat_max": max(h["rhat"]),
                "ess_bulk_min": min(h["ess_bulk"]),
                "ess_tail_min": min(h["ess_tail"]),
                "bfmi_min": min(h["bfmi"]),
                "depth_hits": h["depth_saturation"],
                "mcse_mean_sd_max": max(h["mcse_mean_sd_units"]),
                "nuts_workflow_seconds": ref["wall_seconds"],
            }
        )
        for path in sorted(directory.glob("*_seed*/comparison.json")):
            record = json.loads(path.read_text())
            record["target"] = name
            records.append(record)
            if (path.parent / "fields.json").exists():
                fields[str(path)] = json.loads(
                    (path.parent / "fields.json").read_text()
                )
        for family in cfg["guides"]:
            family_records = [
                r
                for r in records
                if r["target"] == name and r["guide"] == family
            ]
            final = [r["checkpoints"][-1] for r in family_records]
            feature_rows = [
                f
                for r in final
                for f in r["comparison"]["features"]
                if f["status"] == "ok"
            ]
            spectral_rows = [
                f
                for f in feature_rows
                if f["name"].startswith(
                    ("log_V", "band_power", "time_contrast")
                )
            ]
            ratios = [
                r
                for record in family_records
                for r in record["weights"]
                if r["status"] == "ok"
            ]
            psis = [
                run["k"]
                for record in family_records
                for run in record["native_psis"].get("runs", [])
                if run["k"] is not None
            ]
            events = [
                event for r in final for event in r["comparison"]["events"]
            ]
            row = {
                "target": name,
                "guide": family,
                "reference_status": ref["status"],
                "moment_mean_error_max_abs_sd": max(
                    abs(f["standardized_mean_difference"])
                    for f in spectral_rows
                ),
                "spectral_sd_ratio_median": float(
                    np.median([f["sd_ratio"] for f in spectral_rows])
                ),
                "interval_95_ratio_median": float(
                    np.median(
                        [f["interval_width_ratio_95"] for f in spectral_rows]
                    )
                ),
                "event_probability_difference_max_abs": max(
                    [abs(event["difference"]) for event in events], default=0.0
                )
                if events
                else None,
                "native_k_min": min(psis),
                "native_k_max": max(psis),
                "adapter_k_min": min(r["k"] for r in ratios),
                "adapter_k_max": max(r["k"] for r in ratios),
                "raw_ess_fraction_min": min(
                    r["raw_ess_fraction"] for r in ratios
                ),
                "smoothed_ess_fraction_min": min(
                    r["smoothed_ess_fraction"] for r in ratios
                ),
                "mmd_functional_median": float(
                    np.median(
                        [r["functional_mmd"]["vi_vs_nuts"] for r in final]
                    )
                ),
                "mmd_functional_reference_baseline": float(
                    np.median(
                        [r["functional_mmd"]["nuts_vs_nuts"] for r in final]
                    )
                ),
                "mmd_latent_median": float(
                    np.median([r["latent_mmd"]["vi_vs_nuts"] for r in final])
                ),
                "mmd_latent_reference_baseline": float(
                    np.median([r["latent_mmd"]["nuts_vs_nuts"] for r in final])
                ),
                "fit_workflow_seconds_median": float(
                    np.median(
                        [r["fit_workflow_seconds"] for r in family_records]
                    )
                ),
            }
            rows.append(row)
    aggregate = {
        "references": reference_rows,
        "families": rows,
        "optimization": [],
    }
    for record in records:
        checks = record["checkpoints"]
        middle = checks[1]
        final = checks[-1]

        def feature(c):
            return np.array(
                [
                    f["standardized_mean_difference"]
                    for f in c["comparison"]["features"]
                    if f["status"] == "ok"
                ]
            )

        def probability(c):
            return np.array(
                [e["vi_probability"] for e in c["comparison"]["events"]]
            )

        change = probability(final) - probability(middle)
        aggregate["optimization"].append(
            {
                "target": record["target"],
                "guide": record["guide"],
                "seed": record["optimization_seed"],
                "actual_steps": [c["actual_steps"] for c in checks],
                "mean_change_5000_to_15000_max_sd": float(
                    np.max(np.abs(feature(final) - feature(middle)))
                ),
                "event_change_5000_to_15000_max_abs": float(
                    np.max(np.abs(change))
                )
                if change.size
                else None,
                "fixed_seed_objectives": record["optimization"][
                    "checkpoint_objectives"
                ],
            }
        )
    write_json(out / "summary.json", aggregate)
    accepted = [
        r["target"] for r in reference_rows if r["status"] == "accepted_screen"
    ]
    unresolved = [
        r["target"] for r in reference_rows if r["status"] != "accepted_screen"
    ]
    outcomes = [
        "## What the experiment resolves",
        "",
        "**Reference error:** accepted screens for "
        + (", ".join(accepted) or "none")
        + "; unresolved references for "
        + (", ".join(unresolved) or "none")
        + ". Discrepancies against unresolved references remain descriptive.",
        "",
    ]
    additional = out / "raw_ar1_m8/repair_2/reference_repair.json"
    if additional.exists():
        repair = json.loads(additional.read_text())
        outcomes.extend(
            [
                f"The additional AR(1) tuning repair at acceptance .999/depth 12 has status {repair['status']} and {repair['health']['divergences']} divergences. It is promoted only if all reference screens pass.",
                "",
            ]
        )
    selected = next(
        (
            r
            for r in records
            if r["target"] == "exact_time_varying"
            and r["guide"] == "diag"
            and r["optimization_seed"] == 7101
        ),
        None,
    )
    if selected is not None:
        objectives = selected["optimization"]["checkpoint_objectives"]
        mid, last = (
            selected["checkpoints"][1]["actual_steps"],
            selected["checkpoints"][-1]["actual_steps"],
        )
        a, b = objectives[str(mid)], objectives[str(last)]
        outcomes.extend(
            [
                f"**Optimization error:** on the exact time-varying diagonal guide, seed 7101, fixed-seed negative ELBO changed from {a['negative_elbo']:.2f} at {mid} steps to {b['negative_elbo']:.2f} at {last} (final evaluation MCSE {b['mcse']:.2f}). Increasing steps or adding covariance parameters does not establish optimizer convergence. The checkpoint table includes roughness scales and D losses as well as spectral features.",
                "",
            ]
        )
    slow = [r for r in rows if r["target"] == "raw_slow_m32"]
    if slow:
        values = ", ".join(
            f"{r['guide']}={r['spectral_sd_ratio_median']:.2f}" for r in slow
        )
        outcomes.extend(
            [
                "**Family/geometry restrictions:** richer covariance guides improve some spectral/MMD comparisons, but remaining optimization instability prevents isolating irreducible family error. Smooth raw-series spectral SD ratios are "
                + values
                + ". The measured spectrum bands are not uniformly too narrow.",
                "",
            ]
        )
    projection_path = out / "raw_slow_m32/diag_seed7101/fields.json"
    if projection_path.exists():
        projected = json.loads(projection_path.read_text())["fields"][-1]
        decomposition = projected["projection_variance_decomposition"]
        ratios = ", ".join(
            f"{x['functional_variance_ratio']:.2f}"
            for x in decomposition["projections"]
        )
        outcomes.extend(
            [
                f"For the smooth raw-series diagonal guide at seed 7101/{projected['actual_steps']} steps, the median original-coefficient marginal variance ratio is {decomposition['coefficient_marginal_variance_ratio_median']:.3f}, while three selected log-spectrum variance ratios are {ratios}. The saved diagonal/cross-covariance decomposition measures covariance cancellation; it does not attribute all differences solely to the guide family. The exact Gaussian anticorrelation example is separately tested analytically.",
                "",
            ]
        )
    warned = sum(r["native_k_max"] >= 0.7 for r in rows)
    outcomes.extend(
        [
            f"**Importance proposals:** {warned}/{len(rows)} target/family combinations have at least one native k ≥.7 across optimization/diagnostic seeds. Joint weight ESS and repeated-seed stability are reported separately. No reweighted posterior is claimed.",
            "",
            "**Model adequacy:** exact independent Gamma controls separate transform approximation from algorithm discrepancies. Raw-series transform dependence and predictive adequacy remain untested here. No whitening or calibration result is inferred from posterior comparisons.",
            "",
            "A bounded follow-on should stabilize optimization on these same targets before estimating a guide-family accuracy limit. Unresolved reference geometry requires its own validated repair. New parameterization experiments were not executed.",
            "",
        ]
    )
    lines = [
        "# VI versus NUTS validation: stages A–C",
        "",
        "Run date: 2026-10-02 (Pacific/Auckland). These are small-basis development experiments, not repeated-simulation calibration.",
        "",
        "## Reference health and scope",
        "",
        "| Target | Reference status | Divergences | Max R-hat | Min bulk/tail ESS | Min BFMI | Depth hits | Max mean MCSE/SD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reference_rows:
        lines.append(
            f"| {r['target']} | {r['status']} | {r['divergences']} | {r['rhat_max']:.5f} | {r['ess_bulk_min']:.0f}/{r['ess_tail_min']:.0f} | {r['bfmi_min']:.3f} | {r['depth_hits']} | {r['mcse_mean_sd_max']:.3f} |"
        )
    lines += [
        "",
        "Reference screening uses four independent initializations, rank-normalized split R-hat <1.01, bulk/tail ESS ≥400, no divergences/depth saturation, BFMI ≥0.3 and mean MCSE ≤0.05 posterior SD. Latent sites and declared functionals are checked together. These are screens, not convergence proofs. A failed screen remains a limitation in every associated comparison. Initial references and repairs are separate artifacts.",
        "",
        "## Task-level posterior comparisons",
        "",
        "At 15,000 actual optimization steps. Values aggregate three optimization seeds. Spectral features are declared log variances, integrated band powers and time contrasts; roughness scales and D distributions remain available per feature in comparison.json.",
        "",
        "| Target | Guide | Max spectral mean error / NUTS SD | Median SD ratio | Median 95% width ratio | Max event probability difference | Functional MMD² / NUTS baseline | Latent MMD² / NUTS baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['guide']} | {fmt(r['moment_mean_error_max_abs_sd'])} | {fmt(r['spectral_sd_ratio_median'])} | {fmt(r['interval_95_ratio_median'])} | {fmt(r['event_probability_difference_max_abs'])} | {fmt(r['mmd_functional_median'])}/{fmt(r['mmd_functional_reference_baseline'])} | {fmt(r['mmd_latent_median'])}/{fmt(r['mmd_latent_reference_baseline'])} |"
        )
    verification_path = out / "verification.json"
    verification = (
        json.loads(verification_path.read_text())
        if verification_path.exists()
        else None
    )
    lines.extend(
        [
            "",
            "### Physical field errors",
            "",
            "| Target | Guide | Max area-weighted RMS mean error / reference SD | Worst-region mean error / reference SD |",
            "|---|---|---:|---:|",
        ]
    )
    for row in rows:
        values = []
        for path, record in fields.items():
            source = json.loads(Path(path).read_text())
            if (
                source["target"] == row["target"]
                and source["guide"] == row["guide"]
            ):
                values.append(record["fields"][-1])
        rms = max(
            (v["standardized_mean_error_rms"] for v in values), default=None
        )
        worst = max(
            (v["standardized_mean_error_max_abs"] for v in values),
            default=None,
        )
        lines.append(
            f"| {row['target']} | {row['guide']} | {fmt(rms)} | {fmt(worst)} |"
        )
    lines.extend(
        [
            "",
            "Field values use normalized physical trapezoid quadrature, all posterior draws and bounded frequency chunks. For tensor targets the declared region is time [.1,.9] × frequency [.08,.42]; field artifacts retain exact coordinates and worst-region locations. Matched MMD subset counts are 512; the actual bulk ESS of untransformed reference subset features is separately saved in fields.json and may be much smaller. The 9-dimensional stationary lowrank:10 guide uses a factor rank exceeding its dimension, so it does not impose a low-rank covariance restriction there.",
        ]
    )
    lines += [
        "",
        "Univariate Wasserstein distances, 50/90/95% interval ratios, event MCSE and correlations are saved for every feature/seed/checkpoint. Event differences must be judged against their function-specific MCSE; constant indicators are explicitly unresolved tails, not zero-MCSE certificates. Field analyses reconstruct all joint draws in frequency chunks on a declared physical grid, with normalized quadrature RMS and worst-region errors (fields.json). No spatial cells are treated as independent datasets.",
        "",
        "MMD uses the same regularized reference whitening for both methods, trained on a disjoint first-quarter reference subset. Bandwidths are sqrt(feature dimension) times [0.5,1,2]; estimator is unbiased and excludes within-sample diagonals. Negative baseline estimates are preserved. All comparisons use 512 matched draws and report descriptive NUTS-versus-NUTS and VI-versus-VI baselines. Reference draws remain autocorrelated; there are no IID permutation p-values or universal MMD cutoffs.",
        "",
        "## Joint importance proposal diagnostics",
        "",
        "| Target | Guide | Native k range | Adapter k range | Min raw ESS fraction | Min smoothed ESS fraction | Median fit workflow seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['guide']} | {fmt(r['native_k_min'])}–{fmt(r['native_k_max'])} | {fmt(r['adapter_k_min'])}–{fmt(r['adapter_k_max'])} | {fmt(r['raw_ess_fraction_min'])} | {fmt(r['smoothed_ess_fraction_min'])} | {fmt(r['fit_workflow_seconds_median'])} |"
        )
    lines += [
        "",
        "Each diagnostic uses 4096 joint particles ×3 independent diagnostic seeds, in chunks of 128. NumPyro native PSIS and the packed-density ArviZ weight adapter use different draws, so finite-sample k values need not match. Both densities use the same unconstrained measure; NumPyro potential_energy includes all prior/likelihood factors and one support Jacobian. No densities over PSD pixels and no KDE substitute for the learned guide. Weight ESS is not MCMC ESS. No posterior reweighting was performed.",
        "",
        "ArviZ Stats 1.0.0 negates its array input internally. A tested lower-tail identity selects and records the input sign; the adapter explicitly normalizes log weights. Exact matching densities are labeled constant_weights, with full ESS and undefined tail k, even when the native GPD routine returns a nonfinite value.",
        "",
        "## Optimization audit",
        "",
        "| Target | Guide | Max mean change 5000→15000 / NUTS SD | Max probability change 5000→15000 |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        selected = [
            x
            for x in aggregate["optimization"]
            if x["target"] == r["target"] and x["guide"] == r["guide"]
        ]
        probabilities = [
            x["event_change_5000_to_15000_max_abs"]
            for x in selected
            if x["event_change_5000_to_15000_max_abs"] is not None
        ]
        lines.append(
            f"| {r['target']} | {r['guide']} | {fmt(max(x['mean_change_5000_to_15000_max_sd'] for x in selected))} | {fmt(max(probabilities) if probabilities else None)} |"
        )
    lines += [
        "",
        "All guide families use optimization seeds 7101/7102/7103 and checkpoints at 1000, 5000 and 15000 actual steps. Early stopping was disabled. Objectives use fixed evaluation seeds 8201–8204 and 32 particles; independent evaluations use diagnostic seeds 8101–8103. The legacy relative-loss stopping test is demonstrably sensitive to additive data constants. An opt-in paired objective/noise and location-stability rule is provided, while diagnostic collection alone preserves the existing stopping behavior. Flat objective values do not certify posterior accuracy.",
        "",
        "## Costs, artifacts and unexecuted work",
        "",
        f"Executed {len(reference_rows)} target comparisons and {len(records)} VI fits, with three saved checkpoint comparisons per fit. Retained NUTS jobs: {len(list(out.glob('**/*posterior.nc'))) - len(records)} (four initial references, two first repairs and one additional AR(1) repair in this run). Each target directory records the exact completed jobs and source SHA, dirty-content hash, versions, CPU device/backend, float64 parameter dtypes, model fingerprint and timings. Original failed references and execution failures are retained. NumPyro 0.22.0 was installed in an isolated worktree environment; the main checkout and its environment were preserved.",
        "",
        "Fit workflow timings include initialization/compilation, optimization, checkpoint evaluations, guide draws and density diagnostics. Additional checkpoint reconstruction/comparison and offline field/report costs are saved separately. The historical remaining_optimization_seconds phase also includes checkpoint objective evaluations; it is not pure optimizer kernel time. Guide draw throughput is not used as accurate posterior throughput.",
        "",
        "Exact-model controls generate powers from the intended spline surface using Gamma(nu/2, scale=2V). Raw-process comparisons reuse archived paired observations and the exact 8×6 cubic basis at fixed m=8 or m=32, with all priors, likelihoods and normalizations held fixed. No inference results are reused as four-chain references. Raw transform dependence/model adequacy remain unresolved.",
        "",
        "Stage D predictive whitening/PIT/held-out adequacy and stage E SBC/repeated coverage were not run. The coefficient-whitening primitive is covered by analytic contract tests, but power observations cannot recover original coefficient signs/phases. Optimizer-state resumption, flow checkpoint reconstruction and large-basis recovery are not implemented or validated. The saved checkpoint is guide-only. No adaptive window selection, parameterization experiment, new flow or remote/OzSTAR execution occurred.",
        "",
        f"Bulk artifact directory: `{out.resolve()}`. Reproduction: `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/archive/run.py --previous <prior-artifact-directory> --out <new-directory>`. Reports/field analysis: `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/archive/summarize.py --out <artifact-directory> --fields`.",
        "",
        "Official APIs checked against installed versions: [NumPyro PSIS](https://num.pyro.ai/en/stable/utilities.html#psis-diagnostic), [NumPyro 0.22.0 release](https://github.com/pyro-ppl/numpyro/releases/tag/0.22.0), [ArviZ array PSIS](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.base.array_stats.psislw.html).",
    ]
    lines[4:4] = outcomes
    lines.extend(
        [
            "",
            "## Verification",
            "",
            verification["summary"]
            if verification is not None
            else "No test-verification record was supplied for this artifact directory.",
            "",
            "Two execution failures are retained separately: an initial tail-ESS API call before chain persistence (no chains saved from that trial), and NumPy scalar fingerprint serialization before the AR(1) fit. Both were repaired and rerun. References are now saved before diagnostic evaluation. Tests do not certify posterior accuracy, raw-process coverage or model adequacy.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    return aggregate


def plot_summary(aggregate, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    rows = aggregate["families"]
    failed_targets = {
        r["target"]
        for r in aggregate["references"]
        if r["status"] != "accepted_screen"
    }
    labels = [
        r["target"].replace("exact_", "").replace("raw_", "")
        + " / "
        + r["guide"]
        + (" *" if r["target"] in failed_targets else "")
        for r in rows
    ]
    y = np.arange(len(rows))
    axes[0].barh(
        y, [r["moment_mean_error_max_abs_sd"] for r in rows], color="#4379a8"
    )
    axes[0].set_title("Posterior mean discrepancy")
    axes[0].set_xlabel("Maximum spectral mean error / reference SD")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].scatter(
        [r["spectral_sd_ratio_median"] for r in rows], y, color="#438564"
    )
    axes[1].axvline(1, color="black", linestyle="--")
    axes[1].set_title("Spectral uncertainty")
    axes[1].set_xlabel("Median SD ratio, VI / reference")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    for index, row in enumerate(rows):
        axes[2].plot(
            [row["native_k_min"], row["native_k_max"]],
            [index, index],
            color="#9562a4",
            marker="o",
            markersize=3,
        )
    axes[2].axvline(0.7, color="#b65050", linestyle="--")
    axes[2].set_title("Joint proposal tail stability")
    axes[2].set_xlabel(
        "Native Pareto k range over optimization/diagnostic seeds"
    )
    axes[2].set_yticks(y, [])
    axes[2].invert_yaxis()
    fig.suptitle(
        "Small-basis diagnostics answer separate questions", fontsize=14
    )
    fig.text(
        0.5,
        0.01,
        "* Failed reference screen: comparisons are descriptive. Red line: importance-proposal warning at k = 0.7.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--fields", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs/development/vi_nuts_validation/report.md",
    )
    args = parser.parse_args()
    cfg = tomllib.loads(Path(__file__).with_name("config.toml").read_text())
    before = perf_counter()
    if args.fields:
        add_fields(args.out, args.previous, cfg)
    aggregate = summarize(args.out, cfg, args.report)
    plot_summary(aggregate, args.out)
    write_json(
        args.out / "report_execution.json",
        {
            "wall_seconds": perf_counter() - before,
            "provenance": runtime_provenance(),
        },
    )


if __name__ == "__main__":
    main()
