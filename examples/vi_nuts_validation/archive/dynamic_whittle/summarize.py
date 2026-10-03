"""Render the stage 0--3 review from saved runs; never run inference."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_investigation import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    ROOT,
    metrics,
    truth,
    write_json,
)

from log_psplines.preprocessing.moving_periodogram import (
    scattered_moving_periodogram,
)
from log_psplines.results import PSDResult


def comparison(reference: dict, approximate: dict, out: Path) -> dict:
    ref = np.load(out / reference["job"]["id"] / "summary.npz")
    approx = np.load(out / approximate["job"]["id"] / "summary.npz")
    times, freq = ref["time"], ref["frequency"]
    selected = (times >= 0.1) & (times <= 0.9)
    band = (freq >= 0.08) & (freq <= 0.42)
    sections = np.ix_(selected, band)
    ref_width = np.log(ref["quantiles"][2] / ref["quantiles"][0])[sections]
    app_width = np.log(approx["quantiles"][2] / approx["quantiles"][0])[
        sections
    ]
    refd = np.load(out / reference["job"]["id"] / "loss_distributions.npz")
    appd = np.load(out / approximate["job"]["id"] / "loss_distributions.npz")
    distributions = {}
    probability_difference = []
    for name in refd.files:
        probabilities = {
            str(eps): [
                float(np.mean(refd[name] <= eps)),
                float(np.mean(appd[name] <= eps)),
            ]
            for eps in (0.005, 0.02, 0.08)
        }
        probability_difference.extend(
            abs(a - b) for a, b in probabilities.values()
        )
        distributions[name] = {
            "nuts_quantiles": np.quantile(refd[name], [0.05, 0.5, 0.95]),
            "vi_quantiles": np.quantile(appd[name], [0.05, 0.5, 0.95]),
            "probabilities_nuts_vi": probabilities,
        }
    return {
        "family": approximate["job"]["family"],
        "m": approximate["job"]["m"],
        "guide": approximate["job"].get("guide"),
        "inference_seed": approximate["job"]["inference_seed"],
        "vi_steps": approximate["job"].get("vi_steps", 5000),
        "id": approximate["job"]["id"],
        "reference": reference["job"]["id"],
        "log_mean_difference_rmse": float(
            np.sqrt(
                np.mean(
                    np.log(approx["mean"][sections] / ref["mean"][sections])
                    ** 2
                )
            )
        ),
        "median_log_interval_width_ratio": float(
            np.median(app_width / ref_width)
        ),
        "max_D_probability_difference": float(max(probability_difference)),
        "D_distributions": distributions,
    }


def main() -> None:
    cfg = tomllib.loads(DEFAULT_CONFIG.read_text())
    out = DEFAULT_OUTPUT
    records = [
        json.loads(p.read_text()) for p in sorted(out.glob("*/metrics.json"))
    ]
    # Recompute summaries only, with exact common retained-center bounds and
    # fixed-grid region weights. Preserve original fit metrics for provenance.
    for record in records:
        if record["status"] != "ok":
            continue
        job = record["job"]
        n = cfg["calibration_n"] if job.get("calibration") else cfg["n"]
        directory = out / job["id"]
        observations = np.load(directory / "observations.npz")
        data = scattered_moving_periodogram(
            observations["x"], dt=cfg["dt"], m=job["m"], thin=cfg["thin"]
        )
        result = PSDResult.from_netcdf(directory / "result.nc")
        target = truth(
            job["family"], result.time, result.frequency, n, cfg["dt"]
        )
        updated = metrics(result, target, cfg, data, job["m"], n)
        if "original_metrics" not in record:
            record["original_metrics"] = record["metrics"]
        record["metrics"] = updated
        record["metric_revision"] = (
            "exact whole-cycle common bounds; restricted fixed-grid time quadrature; feature diagnostics"
        )
        write_json(directory / "metrics.json", record)
    fixed = [
        r
        for r in records
        if r["status"] == "ok"
        and not r["job"].get("calibration")
        and not r["job"].get("larger")
    ]
    sensitivity = [
        r for r in records if r["status"] == "ok" and r["job"].get("larger")
    ]
    calibration = [
        r
        for r in records
        if r["status"] == "ok" and r["job"].get("calibration")
    ]
    comparisons = []
    for family in cfg["calibration_families"]:
        for m in cfg["calibration_windows"]:
            reference = [
                r
                for r in calibration
                if r["job"]["family"] == family
                and r["job"]["m"] == m
                and r["job"]["method"] == "nuts"
            ]
            reference = next(
                (r for r in reference if "repair" in r["job"]["id"]),
                reference[0],
            )
            for approx in calibration:
                if (
                    approx["job"]["family"] == family
                    and approx["job"]["m"] == m
                    and approx["job"]["method"] == "vi"
                ):
                    comparisons.append(comparison(reference, approx, out))
    write_json(out / "vi_nuts_comparison.json", comparisons)
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), layout="constrained", sharex=True, sharey=True
    )
    table = []
    for family, ax in zip(cfg["families"], axes.flat, strict=True):
        rows = []
        for seed in cfg["seeds"]:
            subset = sorted(
                [
                    r
                    for r in fixed
                    if r["job"]["family"] == family
                    and r["job"]["data_seed"] == seed
                ],
                key=lambda r: r["job"]["m"],
            )
            vals = [r["metrics"]["matched"]["risk"] for r in subset]
            ax.plot(
                cfg["windows"], vals, "o--", alpha=0.7, label=f"seed {seed}"
            )
            rows.append(vals)
        average = np.mean(rows, axis=0)
        table.append(
            {
                "family": family,
                "risk_by_window": average,
                "paired_seed_risks": rows,
                "best_descriptive_m": cfg["windows"][int(np.argmin(average))],
            }
        )
        ax.plot(cfg["windows"], average, "k-", lw=2, label="two-seed mean")
        ax.set_title(family)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(cfg["windows"], cfg["windows"])
        ax.set_xlabel("Half-window m")
        ax.set_ylabel("Forward Whittle risk")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Fixed-window landscape: common retained-centre mask, 2 development seeds"
    )
    fig.savefig(out / "figures/fixed_window_landscape.png", dpi=160)
    plt.close(fig)
    write_json(out / "fixed_window_landscape.json", table)
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.6), layout="constrained", sharey=True
    )
    for region, ax in zip(("quiet", "slow", "fast"), axes, strict=True):
        for seed in cfg["seeds"]:
            subset = sorted(
                [
                    r
                    for r in fixed
                    if r["job"]["family"] == "mixed"
                    and r["job"]["data_seed"] == seed
                ],
                key=lambda r: r["job"]["m"],
            )
            ax.plot(
                cfg["windows"],
                [r["metrics"][region]["risk"] for r in subset],
                "o-",
                label=f"seed {seed}",
            )
        ax.set_title(region)
        ax.set_xscale("log", base=2)
        ax.set_xticks(cfg["windows"], cfg["windows"])
        ax.set_yscale("log")
        ax.set_xlabel("Half-window m")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Region forward Whittle risk")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Mixed evolution: fast-region basis error limits the window comparison"
    )
    fig.savefig(out / "figures/mixed_region_landscape.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
    for row, family in enumerate(("mixed", "ls2", "resonance")):
        data = np.load(out / f"{family}_seed3101_m8_diag/summary.npz")
        target = np.log(data["truth"])
        low, high = target.min(), target.max()
        for col, m in enumerate((None, 8, 64)):
            values = (
                target
                if m is None
                else np.log(
                    np.load(out / f"{family}_seed3101_m{m}_diag/summary.npz")[
                        "mean"
                    ]
                )
            )
            image = axes[row, col].pcolormesh(
                data["time"],
                data["frequency"],
                values.T,
                shading="auto",
                vmin=low,
                vmax=high,
            )
            axes[row, col].set_title(
                f"{family}: {'truth' if m is None else f'VI m={m}'}"
            )
            axes[row, col].set_xlabel("Full-record u")
            axes[row, col].set_ylabel("Frequency [Hz]")
            axes[row, col].axhline(0.08, color="white", lw=0.6, ls="--")
            axes[row, col].axhline(0.42, color="white", lw=0.6, ls="--")
        fig.colorbar(
            image,
            ax=axes[row, :],
            label="log coefficient variance",
            shrink=0.85,
        )
    fig.savefig(out / "figures/truth_and_fits.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 7), layout="constrained", sharey=True
    )
    labels = []
    for c in comparisons:
        if c["inference_seed"] != 7101:
            continue
        labels.append(
            f"{c['family']} m={c['m']} {c['guide']} {c['vi_steps'] // 1000}k"
        )
        axes[0].plot(
            c["median_log_interval_width_ratio"],
            len(labels) - 1,
            "o",
            color="C0" if c["guide"] == "diag" else "C1",
        )
        axes[1].plot(
            c["max_D_probability_difference"],
            len(labels) - 1,
            "o",
            color="C0" if c["guide"] == "diag" else "C1",
        )
    axes[0].axvline(1, color="black", ls="--")
    axes[0].set_xlabel("Median 90% log-width: VI / NUTS")
    axes[1].set_xlabel("Maximum difference in Pr(D ≤ epsilon)")
    for ax in axes:
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Same model and observations; fixed-window diagnostics, no horizon selection"
    )
    fig.savefig(out / "figures/vi_nuts_comparison.png", dpi=160)
    plt.close(fig)
    oracle = json.loads((out / "oracle_checks.json").read_text())
    transform = json.loads((out / "transform_checks.json").read_text())
    seconds = sum(r.get("total_seconds", 0) for r in records)
    vi = [
        r
        for r in records
        if r["status"] == "ok" and r["job"]["method"] == "vi"
    ]
    nuts = [r for r in calibration if r["job"]["method"] == "nuts"]

    def interval(values):
        return f"{np.median(values):.2f} (range {np.min(values):.2f}–{np.max(values):.2f})"

    lines = [
        "# Moving-periodogram investigation: stages 0–3 review",
        "",
        "**Decision: do not implement adaptive window selection yet.** The truth horizon varies locally, but this first fixed-window landscape does not establish a robust local reconstruction gain from adaptation. Moderate m=32 is competitive across controls and LS2. Fast mixed evolution and the narrow stochastic resonance remain strongly limited by the frozen spline basis; VI uncertainty also differs materially from NUTS. No posterior horizon selection, adaptive refit, or LISA run was executed.",
        "",
        "## Scope, reuse, and changes",
        "",
        "Audited checkout `7b317130787c9d8d3b29f5933de4acbde3556a54`; preserved the pre-existing `.github/workflows/pypi.yml` change. Reused the exact scattered moving periodogram, PowerData, `prepare_power_model`, existing eigenbasis tensor prior, generic `fit_vi`, coefficient reconstruction and PSDResult. Added scalar `method='vi'` dispatch, PLS initialization for built-in guides, bounded-row scattered PLS normal equations, measured VI timings, persisted VI diagnostics, and pure stationarity diagnostics. NUTS remains the default. Parametric VI is explicitly rejected rather than silently running NUTS. No parallel likelihood or spline prior. During execution an external commit advanced HEAD to `63b5c7a` and changed only the release workflow; per-run manifests retain the actual SHA and dirty state. No commits were created by this investigation, and concurrent staging/docs-ignore edits were left intact.",
        "",
        "Normalization and equation-to-code mapping: [audit.md](audit.md). Native powers are coefficient variance: P=2I, counts=2, E[I_white]=variance/(2*pi); one-sided interior PSD/Hz=4*pi*dt*S. Times remain (zero-based centre+1)/n; LS2 digital truth is divided by 2*pi and evaluated at u−1/n. Window sample-count duration and centre span are distinct and saved.",
        "",
        "## Actually executed",
        "",
        f"- {len(fixed)} fixed-window diagonal-VI fits: 6 raw-series families × 2 paired seeds × 4 windows, n=4096, 5000 optimizer steps maximum, 256 posterior draws, fixed cubic 16×10 coefficient basis.",
        f"- {len(sensitivity)} separate larger-basis fits, seed3101 at m=16/64 for mixed/LS2/resonance; cubic 24×14 coefficients.",
        f"- {len(calibration)} calibration fits on n=1024: 6 original two-chain NUTS targets, 12 diagonal VI fits across two inference seeds, 6 lowrank:10 VI fits, plus one recorded NUTS repair and four separately recorded 15000-step VI budget checks. Calibration uses the same model but a smaller 8×6 basis; this does not establish large-basis NUTS parity.",
        "- 12 exact Gaussian transform diagnostics at n=512 with 4000 independent raw Gaussian replicates each: white, AR(1), and mixed modulation; two fixed windows, a predetermined guarded piecewise transform, and naive overlapping-spectrogram negative control.",
        "- Deterministic truth loss/horizon curves for all 6 families at 3 tolerances; raw/prefix policies and censoring saved. These use known truth only. Reserved final seeds 9101–9110 were not opened.",
        f"- Failure count among inference jobs: {sum(r['status'] != 'ok' for r in records)}. Failed jobs are retained in manifests, not excluded silently.",
        "",
        "## Tests and execution failures",
        "",
        "Final routine regression: **132 passed, 5 slow tests deselected**, 25.41 s. Includes small univariate and multivariate public fits, positivity, Hermitian/PD/coherence checks, ArviZ and rendering. Targeted contracts: **43 passed, 1 slow test deselected**, 12.04 s. Initial targeted run had 9 failures (24 passed): scalar ndarray return annotations and a config negative-test exception expectation under the runtime type hook; repaired and rerun. Ruff checks passed for all edited Python files; both staged and working-tree diff whitespace checks passed. The graph refresh command and result are recorded in verification.json. No full slow WDM recovery rerun was performed.",
        "",
        "## Fixed-operator approximation audit",
        "",
        "White means passed the predeclared five-MC-standard-error tolerance; the physical white PSD integral at dt=.25 equals 1. All marginal mean checks passed against exact finite-window expectations. Power variance uses |C|²+|P|², not C_ii² alone. Block-MC standard errors for variance and the highest-correlation pair are saved. The largest variance discrepancy was 5.61 block standard errors for a naive-spectrogram negative control; variance/covariance errors are descriptive and were not predeclared CI pass/fail gates. Marginal moments matching their exact Gaussian values does not make the independence/local-spectrum likelihood exact.",
        "",
        "| Family | Operator | Maximum power correlation | RMS relative mean leakage/bias | Maximum noncircularity | Cross-epoch maximum |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in transform["results"]:
        lines.append(
            f"| {r['family']} | {r['operator']} | {r['max_offdiag_power_correlation']:.4f} | {r['mean_relative_bias_rms']:.4f} | {r['max_non_circularity']:.4f} | {r['cross_epoch_max']:.4f} |"
        )
    lines += [
        "",
        "Guarded epoch supports were disjoint. Cross-epoch covariance was zero to shown precision, including coloured noise whose underlying AR correlations persist but are very small across guards. Guarding does not remove within-epoch dependence or mean bias. The fast mixed n=512 diagnostic deliberately has more rapid sample-scale evolution than the n=4096 sweep; its numerical biases are not estimates for that larger sweep. Stationary AR leakage exists while D=0, so a temporal horizon cannot certify leakage control. Fixed-operator identities do not prove a data-selected transform distribution.",
        "",
        "## Fixed-window risk landscape",
        "",
        "Posterior mean spectrum; normalized forward Whittle risk on the same interior band/grid and exact common retained-centre mask. Bounds are computed from complete cycles, including trailing truncation. Full native-support, quiet/slow/fast/change/boundary risks, pointwise 90% coverage/width, edge risk, feature location/width and retention are saved per run. Coverage here is area-weighted pointwise coverage on TWO realizations, not a repeated-sampling calibration result. Small differences cannot be judged significant from two seeds.",
        "",
        "| Family | m=8 | m=16 (fixed n^(1/3) rule) | m=32 | m=64 | Descriptive best mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['family']} | "
            + " | ".join(f"{v:.5f}" for v in r["risk_by_window"])
            + f" | {r['best_descriptive_m']} |"
        )
    lines += [
        "",
        "Best fixed here uses development truth and is an optimistic descriptive reference; it is not a deployable tuned baseline tested on final seeds. No local risk envelope is treated as an adaptive posterior. Native/support differences remain inspectable in metrics.json.",
        "",
        "| Family | Frozen-basis log-spectrum projection RMSE in band |",
        "|---|---:|",
    ]
    for r in oracle:
        lines.append(f"| {r['family']} | {r['basis_log_rmse_band']:.5f} |")
    lines += [
        "",
        "The larger-basis representation-only check gives mixed .2860, LS2 .0090, resonance .3236 log RMSE (primary .2946/.0452/.4803). The planned larger basis substantially improves LS2 representability but still cannot resolve the mixed fast oscillations or narrow resonance. Calibration mixed projection RMSE is .3389. These are basis limitations, so the stress cases cannot isolate window smearing. Abrupt covariance change is deliberate smooth-model misspecification.",
        "",
        "| Larger-basis sensitivity | Primary matched risk | Larger matched risk |",
        "|---|---:|---:|",
    ]
    for r in sensitivity:
        job = r["job"]
        primary = next(
            x
            for x in fixed
            if x["job"]["family"] == job["family"]
            and x["job"]["m"] == job["m"]
            and x["job"]["data_seed"] == job["data_seed"]
        )
        lines.append(
            f"| {job['family']} m={job['m']} seed={job['data_seed']} | {primary['metrics']['matched']['risk']:.5f} | {r['metrics']['matched']['risk']:.5f} |"
        )
    lines += [
        "",
        "## VI versus NUTS",
        "",
        "| Family/window/guide/init/budget | RMS log posterior-mean difference | Median log-interval width VI/NUTS | Max absolute D pass-probability difference |",
        "|---|---:|---:|---:|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['family']} m={c['m']} {c['guide']} init={c['inference_seed']} steps={c['vi_steps']} | {c['log_mean_difference_rmse']:.4f} | {c['median_log_interval_width_ratio']:.3f} | {c['max_D_probability_difference']:.3f} |"
        )
    lines += [
        "",
        "D probabilities above are diagnostics on declared fixed complete windows at centres .12/.38/.68/.9, half-widths 8/32 and epsilon .005/.02/.08; no posterior horizon or schedule was selected. Full joint surface draws enter D. Monte Carlo granularity is 1/256 for VI; NUTS uses 2000 draws with ESS diagnostics below. Differences near thresholds need more draws/decision calibration before automation. No posterior variance multiplier was introduced.",
        "",
        "| NUTS target | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Depth hits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in nuts:
        d = r["diagnostics"]["nuts"][0]
        lines.append(
            f"| {r['job']['id']} | {d['divergences']} | {d['rhat_max']} | {d['ess_bulk_min']} | {d['ess_tail_min']} | {d['max_treedepth_hits']} |"
        )
    lines += [
        "",
        "Longer-budget checks reduced the slow/mixed diagonal-guide probability disagreements at m=32 to .020/.017, but median interval widths remained 1.338/1.335 times NUTS. Lowrank remained 1.219/1.165 times NUTS, with slow-case probability disagreement .143 after 15000 steps. Optimizer budget and guide shape both remain relevant; more steps did not make them interchangeable. The original mixed m=32 NUTS target had one divergence. It is preserved and a separate repair increased target acceptance to .99 and warmup to 1000. Comparisons use the repair when available. Step size was not requested from NumPyro and is unavailable, not zero. Healthy NUTS is a reference for this approximate likelihood, not a cure for transform dependence or basis misspecification. VI loss traces and independent initializations are retained; 5000 steps is a measured development budget, not proof of optimizer convergence. Raw ELBO is never used to rank different windows.",
        "",
        "## Computational cost",
        "",
        f"Sum of measured job wall times: **{seconds:.1f} s** ({seconds / 60:.1f} min), excluding process startup/import overhead and transform/reporting commands. Separate sweep and calibration dispatcher times are in execution.json; they overlapped, so their sum is not elapsed session time.",
        f"VI fit wall seconds, median (range): **{interval([r['fit_seconds'] for r in vi])}**. NUTS: **{interval([r['fit_seconds'] for r in nuts])}**.",
        f"Isolated-process peak RSS MiB, median (range): **{interval([r['peak_rss_mib'] for r in records if r['status'] == 'ok'])}**; includes Python imports/JAX/runtime, not just posterior arrays.",
        f"VI initialization seconds: {interval([r['vi_timings']['initialization_seconds'] for r in vi])}; first chunk including compilation: {interval([r['vi_timings']['first_chunk_including_compile_seconds'] for r in vi])}; remaining optimization: {interval([r['vi_timings']['remaining_optimization_seconds'] for r in vi])}; posterior draws: {interval([r['vi_timings']['posterior_draw_seconds'] for r in vi])}.",
        "Launch/import-inclusive original dispatcher times were 286.75 s for the 54-job sweep and 178.90 s for 24 calibration jobs; these ran concurrently. Additional repair and four 15000-step jobs are separately recorded. Initialization includes tracing/compilation. Pure compiler time is not independently measured. Preprocessing, model preparation, reconstruction and total times are saved per run; first-chunk time includes 100 optimizer steps. Posterior previews store 32 draws but all 256/2000 posterior draws enter summaries. Result round trips preserve native units, exact paired observations, posterior coefficients, model bases and VI losses/timings. Existing complex-valued h5netcdf storage emits a nonstandard-NetCDF warning; round trips through PSDResult/h5netcdf were verified, but generic NetCDF interoperability was not.",
        "",
        "## Review gate",
        "",
        "Truth D distinguishes short admissible windows during fast evolution from right-censored long windows in quiet regions. That supports a horizon diagnostic as a research question. It does not establish an estimation gain: the first landscape and basis sensitivities leave the hardest regimes confounded, and VI/NUTS widths/probabilities are insufficiently interchangeable for automatic selection. Stage 4 is **not justified as an automatic adaptive selector** by these results. Before approving it, resolve representation capacity and guide convergence/uncertainty on fixed-window targets, then expand paired development seeds and check approximation at the actual n. Keep the moderate fixed-window comparator. No adaptive refitting or LISA is warranted yet.",
        "",
        "## Reproduction and artifacts",
        "",
        "```bash",
        "JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode transforms",
        "JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode sweep",
        "JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode calibration",
        'JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode job --job \'{"id":"cal_mixed_m32_nuts_repair","family":"mixed","data_seed":3101,"m":32,"method":"nuts","inference_seed":7101,"calibration":true,"nuts_warmup":1000,"nuts_target_accept":0.99}\'',
        *[
            "JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode job --job '"
            + json.dumps(
                {
                    "id": f"cal_{family}_m32_vi_{guide.replace(':', '')}_long15000",
                    "family": family,
                    "data_seed": 3101,
                    "m": 32,
                    "method": "vi",
                    "inference_seed": 7101,
                    "calibration": True,
                    "guide": guide,
                    "vi_steps": 15000,
                }
            )
            + "'"
            for family in ("slow", "mixed")
            for guide in ("diag", "lowrank:10")
        ],
        "JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/summarize.py",
        "JAX_ENABLE_X64=true .venv/bin/python -m pytest -m 'not slow' -q",
        "```",
        "",
        "All jobs/configs/data hashes, repo SHA/dirty state, versions/backend/dtype, inference settings, supports, timings, diagnostics, seed-specific metrics, loss distributions and structured failure states are in `runs/dynamic-whittle-stages-0-3/`. Bulk observations/posteriors remain ignored by Git. Source/config/report are retained. Commands preserve completed inference jobs; use a new --output directory for a fresh run. Transform summaries overwrite only their deterministic diagnostics when rerun.",
        "",
        "![Fixed-window landscape](../../../../runs/dynamic-whittle-stages-0-3/figures/fixed_window_landscape.png)",
        "",
        "![VI/NUTS uncertainty comparison](../../../../runs/dynamic-whittle-stages-0-3/figures/vi_nuts_comparison.png)",
        "",
        "Prior methods: Tang's dynamic-Whittle construction supplies the observation representation; [van Delft and Eichler](https://arxiv.org/abs/1512.00825) already adapt local smoothing neighbourhoods iteratively. The proposed research distinction is choosing the observation-window representation from a posterior horizon with the existing log-P-spline prior; no novelty theorem or comprehensive novelty claim is established. Published Tang DOI access failed; the audit uses the specified v1 preprint.",
    ]
    report = "\n".join(lines) + "\n"
    (ROOT / "docs/development/adaptive_dynamic_whittle/review.md").write_text(
        report
    )
    (out / "report.md").write_text(report)
    print(
        f"Review written: {len(records)} records, {len(comparisons)} VI/NUTS comparisons"
    )


if __name__ == "__main__":
    main()
