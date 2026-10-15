"""Restore availability on the existing days-7--35 LISA target and summarize draws."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import arviz_stats as azs
import numpy as np
import xarray as xr
from experiments.dw_wdm_small import hann_paper as study
from experiments.dw_wdm_small import lisa_hann_one_case as mp

sys.path.insert(0, str(study.PAPER / "scripts"))
import lisa_gap_comparison as wdm
from plot_lisa_gap_comparison import field_samples


def arguments(results: Path) -> SimpleNamespace:
    return SimpleNamespace(
        results=results,
        mp_results=study.LISA_MP,
        lisa_source=mp.LISA,
        wdm_source=mp.LEGACY,
        complete=True,
        warmup=1200,
        samples=1600,
    )


def parity(method: str, complete: Path) -> dict:
    old = np.load(
        (study.LISA_MP if method == "MP" else study.LISA_WDM) / "prepared.npz"
    )
    new = np.load(complete / "prepared.npz")
    # Match every original paired site, independently of array position or row counts.
    if method == "MP":
        lookup = {int(row[0]): i for i, row in enumerate(new["sample_index"])}
        keep = np.array([lookup[int(row[0])] for row in old["sample_index"]])
        np.testing.assert_array_equal(
            new["sample_index"][keep], old["sample_index"]
        )
        np.testing.assert_array_equal(new["power"][keep], old["power"])
        np.testing.assert_array_equal(new["reference"][keep], old["reference"])
        max_relative_power_change = 0.0
    else:
        lookup = {int(row): i for i, row in enumerate(new["rows"])}
        keep = np.array([lookup[int(row)] for row in old["rows"]])
        # WDM atoms have infinite support: unmasking changes retained gapped
        # coefficients slightly. This is measured, not asserted invariant.
        np.testing.assert_allclose(
            new["reference"][keep], old["reference"], rtol=1e-13
        )
        max_relative_power_change = float(
            np.max(
                abs(new["power"][keep] - old["power"])
                / np.maximum(old["power"], 1e-40)
            )
        )
    return dict(
        gapped_rows=old["power"].shape[0],
        complete_rows=new["power"].shape[0],
        frequency_sites=old["power"].shape[1],
        reference_same_sites_verified=True,
        shared_power_max_relative_change=max_relative_power_change,
        note="MP shared ordinates exactly invariant; WDM restored tails may change coefficients near guards",
    )


def run() -> None:
    assert study.read(study.ROOT / "preflight.json")["status"] == "pass"
    for method in ("MP", "WDM"):
        for attempt in range(2):
            path = (
                study.ROOT
                / "fits"
                / f"LISA_complete_{method}"
                / f"attempt{attempt}"
            )
            if (path / "fit.json").exists():
                if study.read(path / "fit.json")["diagnostics"]["pass"]:
                    break
                continue
            path.mkdir(parents=True, exist_ok=True)
            started = perf_counter()
            study.write(
                "execution_lisa.json",
                dict(
                    status="running",
                    pid=os.getpid(),
                    method=method,
                    attempt=attempt,
                ),
            )
            print(
                json.dumps(
                    dict(starting=method, attempt=attempt, path=str(path))
                ),
                flush=True,
            )
            args = arguments(path)
            args.warmup, args.samples = (
                (1200, 1600) if attempt == 0 else (1800, 2400)
            )
            try:
                if attempt:
                    primary = path.parent / "attempt0"
                    for name in (
                        "prepared.npz",
                        "preparation.json",
                        "preflight.json",
                    ):
                        if (primary / name).exists():
                            shutil.copy2(primary / name, path / name)
                elif method == "MP":
                    mp.OUT = path
                    shutil.copy2(
                        mp.HERE / "lisa_hann_one_case/preflight.json",
                        path / "preflight.json",
                    )
                    mp.prepare(complete=True)
                else:
                    wdm.prepare(args)
                receipt = parity(method, path)
                study.b.write_json(path / "complete_gap_parity.json", receipt)
                if method == "MP":
                    mp.OUT = path
                    mp.fit_case(
                        args.warmup, args.samples, noncentered_eta=True
                    )
                    basis = np.load(path / "model_basis.npz")
                    old = np.load(study.LISA_MP / "model_basis.npz")
                    for name in old.files:
                        np.testing.assert_allclose(
                            basis[name], old[name], atol=1e-13
                        )
                else:
                    wdm.fit(args)
                # Independently reload both chains, including physical eta.
                post = xr.load_dataset(path / "posterior.nc")
                assert (
                    post.sizes["chain"] == 2
                    and post.sizes["draw"] == args.samples
                )
                assert np.isfinite(post.eta).all()
                study.b.write_json(
                    path / "execution_receipt.json",
                    dict(
                        pid=os.getpid(),
                        total_seconds=perf_counter() - started,
                        source_sha256={
                            str(p): study.audit.file_hash(p)
                            for p in (
                                Path(__file__),
                                Path(mp.__file__),
                                Path(wdm.__file__),
                            )
                        },
                        complete_gap_parity=receipt,
                    ),
                )
                d = study.read(path / "fit.json")["diagnostics"]
                print(
                    json.dumps(
                        dict(completed=method, attempt=attempt, diagnostics=d)
                    ),
                    flush=True,
                )
                if d["pass"]:
                    break
            except Exception as exc:
                study.b.write_json(
                    path / "failure.json",
                    dict(error=repr(exc), seconds=perf_counter() - started),
                )
                print(
                    json.dumps(dict(failed=method, error=repr(exc))),
                    flush=True,
                )
                break
    study.write(
        "execution_lisa.json",
        dict(status="bounded LISA batch finished", pid=os.getpid()),
    )
    study.inventory()


def regions(days: np.ndarray) -> dict[str, np.ndarray]:
    """Use the original gapped condition to define regions for all four cells."""
    schedule = study.read(mp.LISA / "config/paper_gap_schedule.json")[
        "gaps_seconds"
    ]
    inside = np.zeros(len(days), bool)
    for lo, hi in schedule:
        inside |= (days * 86400 >= lo) & (days * 86400 <= hi)
    mp_saved = np.load(study.LISA_MP / "prepared.npz")
    wdm_saved = np.load(study.LISA_WDM / "prepared.npz")
    # Cell support comes from the archived retained row centers and the fixed
    # original cycle spacing. It marks observed neighbourhoods removed by guards.
    mp_centers = mp_saved["cycle_time"] * mp.N * mp.DT / 86400
    wd_centers = wdm_saved["cycle_time"] * mp.N * mp.DT / 86400
    spacing_mp = 2 * mp.M * mp.DT / 86400
    spacing_wdm = wdm.NF * wdm.DT / 86400
    mp_near = np.min(abs(days[:, None] - mp_centers), axis=1) <= spacing_mp / 2
    wd_near = (
        np.min(abs(days[:, None] - wd_centers), axis=1) <= spacing_wdm / 2
    )
    return dict(
        all=np.ones(len(days), bool),
        observed=~inside,
        gap_interpolation=inside,
        observed_excluded_by_either_guard=(~inside) & (~mp_near | ~wd_near),
        common_retained_neighbourhood=(~inside) & mp_near & wd_near,
    )


def summarize() -> None:

    inv = study.inventory()
    selected = {
        f"{c['mask']}_{c['method']}": Path(c["path"])
        for c in inv
        if c["dataset"] == "LISA" and c.get("path")
    }
    args = arguments(study.ROOT / "lisa_summary")
    statuses = {
        f"{c['mask']}_{c['method']}": c["status"]
        for c in inv
        if c["dataset"] == "LISA"
    }
    for c in inv:
        if c["dataset"] != "LISA" or c.get("path"):
            continue
        attempts = [
            Path(a["path"])
            for a in c["attempts"]
            if (Path(a["path"]) / "posterior.nc").exists()
        ]
        if attempts:
            selected[f"{c['mask']}_{c['method']}"] = attempts[-1]
    args.results.mkdir(exist_ok=True)
    (study.ROOT / "figures").mkdir(exist_ok=True)
    spline = wdm.model_basis(args)
    basis = np.load(study.LISA_MP / "model_basis.npz")
    archived = np.load(study.LISA_WDM / "plot_data.npz")
    days, f, truth, nominal = (
        archived[k] for k in ("days", "frequency_hz", "truth", "nominal")
    )
    assert truth.shape == (61, 181)
    time_weights = study.audit.axis_weights(days)
    time_weights /= time_weights.sum()
    frequency_weights = study.audit.axis_weights(f)
    below5_weights = study.audit.band_weights(f, (float(f[0]), 0.005))
    region = regions(days)
    rows = []
    audit = {}
    curves = {}
    temporal = {}
    temporal_recovery = {}
    saved = {"days": days, "frequency_hz": f, "truth": truth}
    for key, path in selected.items():
        post = xr.load_dataset(path / "posterior.nc")
        field = field_samples(spline, basis, post, days, f)
        linear = np.exp(field) * nominal[None, :, :]
        q = np.quantile(linear, [0.05, 0.5, 0.95], axis=0)
        saved[key + "_quantiles"] = q
        # Integrate each physical PSD draw, then compute quantiles of averages.
        time_average = np.einsum("dtf,t->df", linear, time_weights)
        curves[key] = np.quantile(time_average, [0.05, 0.5, 0.95], axis=0)
        saved[key + "_time_average_quantiles"] = curves[key]
        fw = below5_weights.copy()
        fw /= fw.sum()
        temporal_draws = np.einsum("dtf,f->dt", linear, fw)
        temporal[key] = np.quantile(temporal_draws, [0.05, 0.5, 0.95], axis=0)
        injected_band = truth @ fw
        shared = region["common_retained_neighbourhood"]
        temporal_recovery[key] = dict(
            median_curve_peak_to_peak_over_truth=float(
                np.ptp(temporal[key][1]) / np.ptp(injected_band)
            ),
            draw_peak_to_peak_over_truth_quantiles=np.quantile(
                np.ptp(temporal_draws, axis=1) / np.ptp(injected_band),
                [0.05, 0.5, 0.95],
            ).tolist(),
            shared_observed_median_curve_peak_to_peak_over_truth=float(
                np.ptp(temporal[key][1, shared])
                / np.ptp(injected_band[shared])
            ),
            same_shared_observed_days=days[shared].tolist(),
        )
        saved[key + "_low_band_temporal_quantiles"] = temporal[key]
        ds = xr.Dataset(
            {
                "log_psd": (
                    ("chain", "draw", "t", "f"),
                    np.log(linear).reshape(2, -1, len(days), len(f))[
                        :, :, ::10, ::30
                    ],
                )
            }
        )
        diag = azs.summary(ds, round_to="none")
        functional = dict(
            rhat_max=float(diag.r_hat.max()),
            ess_bulk_min=float(diag.ess_bulk.min()),
            ess_tail_min=float(diag.ess_tail.min()),
        )
        functional["pass"] = bool(
            functional["rhat_max"] <= 1.01
            and min(functional["ess_bulk_min"], functional["ess_tail_min"])
            >= 400
        )
        # Verify eta is present and quantify it independently of the plotted PSD.
        eta = post.eta.values.reshape(-1, 6, 16)
        bt, bf = spline.design(days * 86400 / (mp.N * mp.DT), f)
        eta_field = np.einsum(
            "ti,dij,fj->dtf",
            bt @ basis["U_time"],
            eta,
            bf @ basis["U_freq"],
            optimize=True,
        )
        full_from_parts = (
            eta_field
            + (post.g.values.reshape(-1, 16) @ (bf @ basis["U_freq"]).T)[
                :, None, :
            ]
        )
        np.testing.assert_allclose(full_from_parts, field, atol=1e-12)
        audit[key] = dict(
            functional_diagnostics=functional,
            sigma_eta_quantiles=np.quantile(
                post.sigma_eta.values, [0.05, 0.5, 0.95]
            ).tolist(),
            eta_low_frequency_peak_to_peak_median=float(
                np.median(
                    np.ptp(eta_field[:, :, np.argmin(abs(f - 0.002))], axis=1)
                )
            ),
            eta_included_and_physical_time_mapping_verified=True,
            posterior_sha256=study.audit.file_hash(path / "posterior.nc"),
            time_basis_scope="full days7-35 grid of2046 sites, fixed before masks",
            gapped_time_average_includes_interpolation=key.startswith(
                "single"
            ),
        )
        for rn, keep in region.items():
            for band, band_weights in (
                ("full", frequency_weights),
                ("below5mHz", below5_weights),
            ):
                weights = np.outer(
                    study.audit.axis_weights(days) * keep,
                    band_weights,
                )
                if weights.sum() == 0:
                    continue
                rows.append(
                    dict(
                        cell=key,
                        inference_status=statuses[key],
                        region=rn,
                        band=band,
                        **study.scores(q, truth, weights),
                    )
                )
        del linear, field, eta_field, full_from_parts
    truth_average = time_weights @ truth
    truth_temporal = truth @ below5_weights / np.sum(below5_weights)
    saved["truth_time_average"] = truth_average
    saved["truth_low_band_temporal"] = truth_temporal
    np.savez_compressed(args.results / "summaries.npz", **saved)
    study.csv_write("lisa_metrics.csv", rows)
    study.write(
        "lisa_summary.json",
        dict(
            cells=list(selected),
            cell_status=statuses,
            metrics=rows,
            component_audit=audit,
            low_frequency_temporal_recovery=temporal_recovery,
            regions={k: v.tolist() for k, v in region.items()},
            common_grid=dict(days=days.tolist(), frequency_hz=f.tolist()),
            quadrature_weights=dict(
                time=time_weights.tolist(),
                frequency=frequency_weights.tolist(),
                below5_frequency=below5_weights.tolist(),
            ),
            weighting="physical trapezoidal dt df on common61x181 grid; draw-wise arithmetic PSD time averages; low bands use df weights",
            caveat="One realization; within-grid truth inclusion is descriptive. Gapped averages include model interpolation. Guard neighbourhoods use nearest archived row-center cells.",
        ),
    )
    from experiments.dw_wdm_small.hann_paper_figures import plot_lisa

    plot_lisa(saved)
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"Condition & Method & Band & RMSE$_{\ln S}$ & $W$ & Inc90\\",
        r"\hline",
    ]
    for r in rows:
        if r["region"] == "all":
            condition, method = r["cell"].split("_")
            condition_label = "Gapped" if condition == "single" else "Complete"
            method_label = "Hann MP" if method == "MP" else method
            band_label = (
                "Full" if r["band"] == "full" else r"$f<5\,\mathrm{mHz}$"
            )
            if r["inference_status"] not in ("reused", "completed"):
                lines.append(
                    f"{condition_label} & {method_label} & {band_label} & unresolved & -- & --"
                    + r" \\"
                )
                continue
            lines.append(
                f"{condition_label} & {method_label} & {band_label} & {r['RMSE_logS']:.4f} & {r['W']:.4f} & {r['Inc90']:.3f} \\\\"
            )
    lines.append(r"\end{tabular}")
    (study.ROOT / "lisa_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summarize"))
    args = parser.parse_args()
    run() if args.action == "run" else summarize()


if __name__ == "__main__":
    main()
