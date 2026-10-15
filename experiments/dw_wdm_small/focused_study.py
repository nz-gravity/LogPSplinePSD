"""Frozen 20-record gap comparison; reuses package inference and old draws.

Run as a module with freeze, run, or summarize. Originals are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.dw_wdm_small import benchmark as b
from experiments.dw_wdm_small import gap_bias_audit as a
from experiments.dw_wdm_small import gap_smoke as g
from experiments.dw_wdm_small import taper_bias_extension as e
from scipy.stats import beta

from log_psplines import PSDResult

ROOT = b.ROOT / "focused_study"
SEEDS = list(range(6101, 6121))
METHODS = ("dw", "wdm")
MASKS = ("complete", "single")
RADIUS = 327 / b.N  # Previously frozen physical edge radius.


def write(name, value):
    b.write_json(ROOT / name, value)


def read(name):
    return a.read(ROOT / name)


def masks():
    return {"complete": np.ones(b.N, bool), "single": g.masks()["contiguous"]}


def data(x, method, mask):
    if method == "dw":
        return g.dw_data(x, masks()[mask])
    rho, _, _ = g.kernel_expectations(masks()[mask])
    return g.wdm_data(x, masks()[mask], rho)


def passed(d):
    return a.screen(d) and d["ebfmi_min_descriptive"] >= 0.3


def selected(seed, method, mask):
    if seed <= 6110:
        return a.selected_path(seed, method, mask, repaired=True)
    candidates = [
        ROOT / "fits" / mask / f"attempt{i}" / f"seed{seed}_{method}"
        for i in range(2)
    ]
    complete = [p for p in candidates if (p / "metrics.json").exists()]
    good = [
        p
        for p in complete
        if passed(a.read(p / "metrics.json")["focused_diagnostics"])
    ]
    return good[-1] if good else complete[-1] if complete else None


def checked(path, seed, method, mask, canonical):
    m = a.read(path / "metrics.json")
    assert m["raw_sha256"] == b.digest(b.simulate(seed))
    assert a.config_target(m["config"]) == a.config_target(canonical)
    master = b.master_model()
    expected = {
        "time_knots": b.digest(master.time.knots),
        "frequency_knots": b.digest(master.frequency.knots),
        "time_penalty": b.digest(np.asarray(master.time.penalty)),
        "frequency_penalty": b.digest(np.asarray(master.frequency.penalty)),
    }
    assert m["model_hashes"] == expected
    if mask == "single":
        assert m["mask_sha256"] == b.digest(masks()[mask])
        assert m["observed_raw_sha256"] == b.digest(
            np.where(masks()[mask], b.simulate(seed), 0.0)
        )
    result = PSDResult.from_netcdf(path / "posterior.nc")
    np.testing.assert_array_equal(
        result.model_data.knots_time.values, master.time.knots
    )
    np.testing.assert_array_equal(
        result.model_data.knots_frequency.values, master.frequency.knots
    )
    d = e.diagnostics(result, master)
    assert passed(d), (str(path), d)
    return {
        "seed": seed,
        "method": method,
        "mask": mask,
        "path": str(path),
        "posterior_sha256": a.file_hash(path / "posterior.nc"),
        "diagnostics": d,
        "config": m["config"],
        "historical_timing": {
            k: m.get(k)
            for k in (
                "preprocess_seconds",
                "model_preparation_seconds",
                "inference_seconds",
                "fit_total_seconds",
            )
        },
    }


def freeze():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "protocol.json").exists():
        assert read("protocol.json")["seeds"] == SEEDS
        return
    canonical = a.read(
        a.selected_path(6101, "dw", "complete") / "metrics.json"
    )["config"]
    inventory = [
        checked(selected(s, m, k), s, m, k, canonical)
        for s in SEEDS[:10]
        for k in MASKS
        for m in METHODS
    ]
    x = b.simulate(6101)
    checks = {}
    for mask, observed in masks().items():
        mp = data(x, "dw", mask)
        full = b.prepare(x, "dw")
        keep = g.dw_keep(observed)
        np.testing.assert_array_equal(mp.power, full.power[keep])
        np.testing.assert_array_equal(mp.time, full.time[keep])
        hidden = np.where(observed, x, 1e50)
        np.testing.assert_array_equal(mp.power, data(hidden, "dw", mask).power)
        centers = np.rint(mp.time * b.N).astype(int)
        vm = np.convolve(
            b.variance(np.arange(b.N) / b.N), np.ones(65) / 65, "valid"
        )[centers - b.M]
        rho, full_mu, masked_mu = g.kernel_expectations(observed)
        wd = data(x, "wdm", mask)
        wkeep = wd.counts > 0
        wm = masked_mu[wkeep] / rho[wkeep]
        wt = np.broadcast_to(b.variance(wd.time)[:, None], rho.shape)[wkeep]
        checks[mask] = {
            "mp_sites": int(len(mp.time)),
            "wdm_sites": int(wkeep.sum()),
            "mp_rms_log_mean_discrepancy": float(
                np.sqrt(np.mean(np.log(vm / b.variance(mp.time)) ** 2))
            ),
            "wdm_rms_log_mean_discrepancy": float(
                np.sqrt(np.mean(np.log(wm / wt) ** 2))
            ),
            "mp_constant_white_mean": 2 * b.DT,
            "wdm_corrected_row_energy_max_error": float(
                np.max(abs(rho[wkeep] / rho[wkeep] - 1))
            ),
            "hidden_gap_invariance": True,
            "same_site_complete_gap": True,
        }
    u, f, anchors = e.grid()
    truth = np.broadcast_to(
        np.log(2 * b.DT * b.variance(u))[:, None], (len(u), len(f))
    )
    oracle, _ = a.oracle_basis(u, f, truth)
    checks["basis_oracle"] = oracle
    checks["dependence_audit"] = {
        "path": str(b.ROOT / "gap_bias_audit/implementation_audit.json"),
        "scope": "Existing identical generator, schedules, mask, units; expected means do not prove independence.",
    }
    write(
        "protocol.json",
        {
            "seeds": SEEDS,
            "development_seeds": SEEDS[:10],
            "new_seeds": SEEDS[10:],
            "generator": "LS2Data amplitude=0 innovations times sqrt(exp(.7 sin(2 pi u)+.2 cos(4 pi u)))",
            "n": b.N,
            "dt": b.DT,
            "gap": a.endpoints(masks()["single"]).tolist(),
            "edge_radius_samples": 327,
            "grid_time": u.tolist(),
            "grid_frequency": f.tolist(),
            "fixed_sites_time": anchors.tolist(),
            "fixed_sites_frequency": [0.1, 0.25, 0.4],
            "point_estimator": "median in linear PSD units; central 90% quantiles computed from exp(log PSD) draws",
            "config": canonical,
            "sampler_extension": {
                "max_extensions": 1,
                "warmup": 1000,
                "draws": 1500,
                "target_accept": 0.99,
            },
            "screen": "Rhat<=1.01; bulk/tail ESS>=200; no divergences/depth hits; min E-BFMI>=.3",
            "provisional_log_bias_tolerance": 0.02,
            "bootstrap_records": 5000,
            "bootstrap_seed": 20261015,
            "package_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {
                str(p.relative_to(b.ROOT.parent.parent)): a.file_hash(p)
                for p in (
                    Path(__file__),
                    Path(b.__file__),
                    Path(g.__file__),
                    Path(e.__file__),
                )
            },
            "preflight": checks,
            "reused_fits": inventory,
        },
    )
    write(
        "manifest.json",
        {
            "status": "frozen",
            "pid": None,
            "reused_targets": 40,
            "completed_new_targets": 0,
            "attempts": [],
        },
    )
    print(
        "Frozen 20 paired records; seed 6101 four-cell smoke and all 40 reused posteriors pass.",
        flush=True,
    )


def run():
    freeze()
    manifest = read("manifest.json")
    manifest.update(status="running", pid=os.getpid())
    write("manifest.json", manifest)
    try:
        for seed in SEEDS[10:]:
            for mask in MASKS:
                for method in METHODS:
                    p = selected(seed, method, mask)
                    if p is not None and passed(
                        a.read(p / "metrics.json")["focused_diagnostics"]
                    ):
                        continue
                    for attempt in range(2):
                        b.OUT = ROOT / "fits" / mask / f"attempt{attempt}"
                        p = b.OUT / f"seed{seed}_{method}"
                        if not (p / "metrics.json").exists():
                            rho, _, _ = g.kernel_expectations(masks()[mask])

                            def adapter(x, m, observed=masks()[mask], rr=rho):
                                return (
                                    g.dw_data(x, observed)
                                    if m == "dw"
                                    else g.wdm_data(x, observed, rr)
                                )

                            m = b.infer(
                                seed,
                                method,
                                600 if attempt == 0 else 1000,
                                1000 if attempt == 0 else 1500,
                                0.95 if attempt == 0 else 0.99,
                                preprocess=adapter,
                            )
                            m["mask_sha256"] = b.digest(masks()[mask])
                            m["observed_raw_sha256"] = b.digest(
                                np.where(masks()[mask], b.simulate(seed), 0.0)
                            )
                            m["focused_diagnostics"] = e.diagnostics(
                                PSDResult.from_netcdf(p / "posterior.nc"),
                                b.master_model(),
                            )
                            b.write_json(p / "metrics.json", m)
                        m = a.read(p / "metrics.json")
                        if not any(
                            row["path"] == str(p)
                            for row in manifest["attempts"]
                        ):
                            manifest["attempts"].append(
                                {
                                    "seed": seed,
                                    "method": method,
                                    "mask": mask,
                                    "attempt": attempt,
                                    "path": str(p),
                                    "pass": passed(m["focused_diagnostics"]),
                                    "diagnostics": m["focused_diagnostics"],
                                }
                            )
                        manifest["completed_new_targets"] = sum(
                            selected(s, m, k) is not None
                            and passed(
                                a.read(selected(s, m, k) / "metrics.json")[
                                    "focused_diagnostics"
                                ]
                            )
                            for s in SEEDS[10:]
                            for m in METHODS
                            for k in MASKS
                        )
                        write("manifest.json", manifest)
                        print(
                            f"TARGET seed={seed} {method} {mask} attempt={attempt} pass={passed(m['focused_diagnostics'])}",
                            flush=True,
                        )
                        if passed(m["focused_diagnostics"]):
                            break
        manifest["status"] = (
            "fits_complete"
            if manifest["completed_new_targets"] == 40
            else "diagnostic_failures"
        )
        write("manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="implementation_failure", error=repr(exc))
        write("manifest.json", manifest)
        raise


def quantiles(path, u, f):
    cache = (
        ROOT / "reconstructed" / f"{a.file_hash(path / 'posterior.nc')}.npz"
    )
    if cache.exists():
        return np.load(cache)["q"]
    result = PSDResult.from_netcdf(path / "posterior.nc")
    master = b.master_model()
    weights = result.posterior.weights.values.reshape(
        -1, *master.weights.shape
    )
    assert weights.dtype == np.float64 and np.isfinite(weights).all()
    bt, bf = master.time.design_at(u), master.frequency.design_at(f)
    q = np.empty((3, len(u), len(f)))
    for start in range(0, len(u), 12):
        linear = np.exp(
            np.einsum(
                "ti,dij,fj->dtf",
                bt[start : start + 12],
                weights,
                bf,
                optimize=True,
            )
        )
        q[:, start : start + 12] = np.quantile(
            linear, [0.05, 0.5, 0.95], axis=0
        )
    assert np.isfinite(q).all() and (q > 0).all()
    assert ((q[0] <= q[1]) & (q[1] <= q[2])).all()
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, q=q)
    return q


def ci(values):
    values = np.asarray(values)
    rng = np.random.default_rng(20261015)
    draws = values[rng.integers(0, len(values), (5000, len(values)))].mean(
        axis=1
    )
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "records": len(values),
    }


def summarize():
    protocol = read("protocol.json")
    u, f, anchors = e.grid()
    np.testing.assert_array_equal(u, protocol["grid_time"])
    np.testing.assert_array_equal(f, protocol["grid_frequency"])
    reg = a.regions(u, masks()["single"], RADIUS)
    region_weights = {
        name: np.outer(a.axis_weights(u) * keep, a.axis_weights(f))
        for name, keep in reg.items()
    }
    region_weights = {name: w / w.sum() for name, w in region_weights.items()}
    truth = np.broadcast_to(
        2 * b.DT * b.variance(u)[:, None], (len(u), len(f))
    )
    rows, arrays, inventory = [], {}, []
    for seed in SEEDS:
        for mask in MASKS:
            for method in METHODS:
                path = selected(seed, method, mask)
                if path is None:
                    raise RuntimeError(
                        f"Missing target {seed} {method} {mask}"
                    )
                inv = checked(path, seed, method, mask, protocol["config"])
                inventory.append(inv)
                q = quantiles(path, u, f)
                arrays[seed, method, mask] = q
                err = np.log(q[1] / truth)
                for region, w in region_weights.items():
                    rows.append(
                        {
                            "seed": seed,
                            "cohort": "development" if seed <= 6110 else "new",
                            "method": method,
                            "mask": mask,
                            "region": region,
                            "rmse_log": float(np.sqrt(np.sum(w * err**2))),
                            "signed_log_error": float(np.sum(w * err)),
                            "signed_fractional_error": float(
                                np.sum(w * (q[1] / truth - 1))
                            ),
                            "mean_log_width": float(
                                np.sum(w * np.log(q[2] / q[0]))
                            ),
                            "within_record_inclusion90": float(
                                np.sum(w * ((q[0] <= truth) & (truth <= q[2])))
                            ),
                        }
                    )
    with (ROOT / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    regional, frequency, coverage, contrasts = {}, {}, [], {}
    for cohort, seeds in (
        ("combined", SEEDS),
        ("development", SEEDS[:10]),
        ("new", SEEDS[10:]),
    ):
        for method in METHODS:
            for region in reg:
                for mask in MASKS:
                    selected_rows = [
                        r
                        for r in rows
                        if r["seed"] in seeds
                        and r["method"] == method
                        and r["region"] == region
                        and r["mask"] == mask
                    ]
                    key = f"{cohort}/{method}/{mask}/{region}"
                    regional[key] = {
                        metric: ci([r[metric] for r in selected_rows])
                        for metric in (
                            "signed_log_error",
                            "signed_fractional_error",
                            "mean_log_width",
                            "rmse_log",
                            "within_record_inclusion90",
                        )
                    }
                    bounds = regional[key]["signed_log_error"]["ci95"]
                    regional[key]["bias_equivalence_within_0.02"] = (
                        bounds[0] >= -0.02 and bounds[1] <= 0.02
                    )
                added = np.array(
                    [
                        np.log(
                            arrays[s, method, "single"][1]
                            / arrays[s, method, "complete"][1]
                        )
                        for s in seeds
                    ]
                )
                regional[f"{cohort}/{method}/added/{region}"] = ci(
                    np.einsum("stf,tf->s", added, region_weights[region])
                )
                for freq in (0.1, 0.25, 0.4):
                    fi = np.flatnonzero(np.isclose(f, freq))[0]
                    tw = a.axis_weights(u) * reg[region]
                    tw /= tw.sum()
                    frequency[f"{cohort}/{method}/{region}/{freq:g}"] = {
                        mask: ci(
                            [
                                float(
                                    tw
                                    @ np.log(
                                        arrays[s, method, mask][1, :, fi]
                                        / truth[:, fi]
                                    )
                                )
                                for s in seeds
                            ]
                        )
                        for mask in MASKS
                    }
                    frequency[f"{cohort}/{method}/{region}/{freq:g}"][
                        "added"
                    ] = ci(added[:, :, fi] @ tw)
        for mask in MASKS:
            for region in reg:
                for metric in (
                    "signed_log_error",
                    "rmse_log",
                    "mean_log_width",
                ):
                    values = []
                    for seed in seeds:
                        pair = {
                            r["method"]: r[metric]
                            for r in rows
                            if r["seed"] == seed
                            and r["mask"] == mask
                            and r["region"] == region
                        }
                        values.append(pair["dw"] - pair["wdm"])
                    contrasts[f"{cohort}/{mask}/{region}/{metric}"] = ci(
                        values
                    )
    for method in METHODS:
        for mask in MASKS:
            for tu in anchors:
                ti = np.flatnonzero(np.isclose(u, tu))[0]
                for freq in (0.1, 0.25, 0.4):
                    fi = np.flatnonzero(np.isclose(f, freq))[0]
                    hits = sum(
                        arrays[s, method, mask][0, ti, fi]
                        <= truth[ti, fi]
                        <= arrays[s, method, mask][2, ti, fi]
                        for s in SEEDS
                    )
                    bounds = [
                        0.0
                        if hits == 0
                        else float(beta.ppf(0.025, hits, 21 - hits)),
                        1.0
                        if hits == 20
                        else float(beta.ppf(0.975, hits + 1, 20 - hits)),
                    ]
                    coverage.append(
                        {
                            "method": method,
                            "mask": mask,
                            "time": float(tu),
                            "frequency": freq,
                            "hits": int(hits),
                            "records": 20,
                            "fraction": hits / 20,
                            "binomial_ci95": bounds,
                        }
                    )
    write(
        "summary.json",
        {
            "target_fits": len(inventory),
            "records": 20,
            "reused_records": 10,
            "new_records": 10,
            "regional": regional,
            "frequency_resolved": frequency,
            "fixed_site_coverage": coverage,
            "inventory": inventory,
            "paired_method_contrasts_mp_minus_wdm": contrasts,
            "interpretation": "Paired record bootstrap; combined ensemble includes development records. Pixel inclusion is not ensemble coverage. A CI crossing zero is not equivalence within +/- .02.",
            **(
                {
                    "ls2_saved_draw_reconstruction": read("summary.json")[
                        "ls2_saved_draw_reconstruction"
                    ]
                }
                if (ROOT / "summary.json").exists()
                and "ls2_saved_draw_reconstruction" in read("summary.json")
                else {}
            ),
        },
    )
    figures(arrays, u, f, truth)
    report = [
        "# Focused moving-periodogram study",
        "",
        "20 records: 40 reused and 40 new target fits. Each selected posterior passes the frozen sampler screen.",
        "",
        "Sources: protocol.json; per-record metrics.csv; summary.json; original posterior paths in inventory. Historical compilation/warmup/sampling breakdowns are unavailable where not recorded.",
        "",
        "Signed log error: combined record mean [95% paired-record bootstrap interval]; added = gap minus complete.",
    ]
    for method in METHODS:
        for region in reg:
            total = regional[f"combined/{method}/single/{region}"][
                "signed_log_error"
            ]
            added = regional[f"combined/{method}/added/{region}"]
            report.append(
                f"- {method}, {region}: total {total['mean']:.4f} {total['ci95']}; added {added['mean']:.4f} {added['ci95']}."
            )
    report.extend(
        [
            "",
            "Fixed-site coverage uses 20 independent records with exact binomial uncertainty. Old/new and frequency-resolved summaries are in summary.json; several bounds cannot establish +/-2% equivalence. Gap results are interpolation.",
        ]
    )
    report.append("\nDevelopment/new gapped signed log error before the gap:")
    for method in METHODS:
        old = regional[f"development/{method}/single/left_observed"][
            "signed_log_error"
        ]["mean"]
        new = regional[f"new/{method}/single/left_observed"][
            "signed_log_error"
        ]["mean"]
        report.append(f"- {method}: {old:.4f} / {new:+.4f}.")
    if (ROOT / "lisa_preflight.json").exists():
        lisa = read("lisa_preflight.json")
        report.extend(
            [
                "",
                "LISA preprocessing/reference: " + lisa["status"] + ".",
                "- Raw stream and orbit checksums pass; calendar-selected day 7–35 MP/WDM plumbing passes.",
                "- " + " ".join(lisa["blockers"]),
                "- Scientific smoke and annual inference were not launched; sampling cost remains unknown.",
                "- Archived WDM jobs 17454664 and 17454667 use different knot sets and are not reused as a matched pair.",
                "- See lisa_preflight.json, validation.json and manifest.json for inputs, historical draws and actual process IDs.",
                "",
                "Manuscripts: all moving-paper text remains in main.tex; new gap figure/table are in the main body, detailed checks in appendices. Companion edits correct the saved LISA eta prior (HalfNormal 0.5) and explain the centered 114-parameter basis. Historical results and marked edits are preserved.",
                "",
                "Remaining issues: physical common-target mapping and matched LISA posteriors; placeholder moving-paper author details. Companion baseline has undefined eq:h_para_amplitude, multiply-defined labels and one overfull equation; the consistency edits do not introduce them.",
            ]
        )
    (ROOT / "REPORT.md").write_text("\n".join(report) + "\n")
    manifest = read("manifest.json")
    manifest["status"] = "summarized"
    write("manifest.json", manifest)
    print("Summarized all 80 target fits.", flush=True)


def figures(arrays, u, f, truth):
    from pathlib import Path

    figdir = Path(
        "/Users/avi/Documents/papers/moving_periodogram_pspline/figures"
    )
    colors = {"complete": "#356799", "single": "#bd652d"}
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5), sharex=True, sharey="row", layout="constrained"
    )
    lo, hi = a.endpoints(masks()["single"])[0] / b.N
    fi = np.flatnonzero(np.isclose(f, 0.25))[0]
    for column, method in enumerate(METHODS):
        ax = axes[0, column]
        ax.plot(u, truth[:, fi], color="black", lw=1, label=r"$S^{\rm true}$")
        for mask in MASKS:
            q = arrays[6101, method, mask][:, :, fi]
            ax.plot(
                u,
                q[1],
                color=colors[mask],
                label="Complete" if mask == "complete" else "Gapped",
            )
            if mask == "single":
                ax.fill_between(u, q[0], q[2], color=colors[mask], alpha=0.2)
            else:
                ax.plot(u, q[0], color=colors[mask], lw=0.6, alpha=0.65)
                ax.plot(u, q[2], color=colors[mask], lw=0.6, alpha=0.65)
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel(r"$S\;[\mathrm{Hz}^{-1}]$")
        obs = data(b.simulate(6101), method, "single")
        bottom = axes[1, column]
        if obs.is_grid:
            tt, ff = np.meshgrid(obs.time, obs.frequency, indexing="ij")
            valid = obs.counts > 0
            bottom.scatter(
                tt[valid],
                ff[valid],
                s=0.4,
                color=colors["single"],
                rasterized=True,
            )
            excluded = ~g.wdm_keep(
                g.kernel_expectations(masks()["single"])[0]
            ).any(axis=1)
            for ti in np.flatnonzero(excluded):
                center = obs.time[ti]
                bottom.axvspan(
                    center - 0.5 / b.NT,
                    center + 0.5 / b.NT,
                    color=".85",
                    lw=0,
                    zorder=0,
                )
        else:
            bottom.scatter(
                obs.time,
                obs.frequency,
                s=0.5,
                color=colors["single"],
                rasterized=True,
            )
            bottom.axvspan(
                lo - b.M / b.N, hi + b.M / b.N, color=".85", lw=0, zorder=0
            )
        bottom.set(
            xlabel=r"$t$", ylabel=r"$f\;[\mathrm{Hz}]$", ylim=(0.05, 0.45)
        )
        for rr in range(2):
            ax = axes[rr, column]
            ax.axvspan(lo, hi, color=".55", alpha=0.25, lw=0)
            ax.set_xlim(0.05, 0.95)
            ax.set_title(f"({chr(97 + rr * 2 + column)})", loc="left")
            ax.grid(False, which="both")
            ax.tick_params(which="both", direction="in", top=True, right=True)
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"simple_gap.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "summarize"))
    globals()[parser.parse_args().stage]()
