"""Authorised four-length LS2 extension, retaining the original endpoint study."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import xarray as xr
from experiments.dw_wdm_small import hann_paper as s
from scipy import sparse
from scipy.signal import periodogram

from log_psplines import PowerData, PSDResult, TimeSeries, fit
from log_psplines.preprocessing.wdm import wdm_periodogram

ROOT = s.ROOT / "ls2_intermediate_20261015"
LENGTHS = (2048, 4096, 8192, 16384)
METRICS = ("RMSE_logS", "W", "Inc90")


def write(name: str, value: object) -> None:
    (ROOT / name).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )


def cells() -> list[dict]:
    return [
        dict(
            id=f"LS2_n{n}_r{r}_{method}",
            dataset="LS2",
            n=n,
            dt=0.1,
            record=r,
            seed=20261001 + r,
            mask="complete",
            method=method,
        )
        for n in LENGTHS
        for r in range(5)
        for method in s.METHODS
    ]


def observations(cell: dict) -> PowerData:
    x = np.load(ROOT / "inputs" / f"n{cell['n']}_r{cell['record']}.npz")["x"]
    d = s.hann_observations(x, np.ones(len(x), bool), 0.1)
    return PowerData(d.power, d.counts, d.frequency / 5, d.time, d.units)


def audit_observations(
    x: np.ndarray, n: int, *, rungs: tuple[int, ...] | None = None
) -> dict:
    """Check direct powers and exact coloured Gaussian window dependence."""
    h = np.hanning(65)
    d = s.hann_observations(x, np.ones(n, bool), 0.1, rungs=rungs)
    centers = np.rint(d.time * n).astype(int)
    rungs = np.rint(d.frequency * 6.5).astype(int)
    windows = np.lib.stride_tricks.sliding_window_view(x, 65)[centers - 32]
    _, reference = periodogram(
        windows, fs=10, window=h, detrend=False, scaling="density", axis=-1
    )
    reference = reference[np.arange(len(centers)), rungs]
    np.testing.assert_allclose(d.power / 2, reference, atol=2e-13, rtol=1e-12)
    assert np.all((centers - 32) % 64 == rungs - 1)
    coef = s.LS2Data(n_samples=n, fs=10, seed=20261001).coefficient(
        np.arange(n) / n
    )
    wb = np.lib.stride_tricks.sliding_window_view(coef, 65)[centers - 32]
    expected = (
        0.2
        * (
            (1 + wb**2) @ h**2
            + 2
            * np.cos(2 * np.pi * d.frequency * 0.1)
            * (wb[:, 1:] @ (h[1:] * h[:-1]))
        )
        / 24
    )
    truth = 0.2 * (
        1
        + coef[centers] ** 2
        + 2 * coef[centers] * np.cos(2 * np.pi * d.frequency * 0.1)
    )
    value = (
        h[None, :]
        * np.exp(-2j * np.pi * rungs[:, None] * np.arange(65) / 65)
        / np.sqrt(24)
    )
    rows = sparse.csr_matrix(
        (
            value.ravel(),
            (
                np.repeat(np.arange(len(centers)), 65),
                (centers[:, None] + np.arange(-32, 33)).ravel(),
            ),
        ),
        shape=(len(centers), n),
    )
    return dict(
        sites=len(centers),
        direct_max_error=float(np.max(abs(d.power / 2 - reference))),
        rms_log_expected_discrepancy=float(
            np.sqrt(np.mean(np.log(expected / truth) ** 2))
        ),
        maximum_fractional_expected_discrepancy=float(
            np.max(abs(expected / truth - 1))
        ),
        dependence=s.audit.gaussian_dependence(rows, 1 + coef**2, coef),
    )


def freeze() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "protocol.json").exists():
        assert s.read(ROOT / "protocol.json")["lengths"] == list(LENGTHS)
        return
    assert s.read(s.ROOT / "preflight.json")["status"] == "pass"
    (ROOT / "inputs").mkdir(exist_ok=True)
    recovered, audits = {}, {}
    for n in LENGTHS:
        model = s.endpoint_model(n)
        model_hashes = s.save_basis(ROOT / "inputs" / f"n{n}_basis.npz", model)
        for r in range(5):
            stem = s.LS2 / f"moving_n{n}_r{r}"
            meta = s.read(stem.with_suffix(".json"))
            assert meta["data_seed"] == 20261001 + r
            dataset = s.LS2Data(n_samples=n, fs=10, seed=meta["data_seed"])
            x = dataset.data[:, 0]
            old_mp = np.load(stem.with_suffix(".npz"))
            raw_mp = s.scattered_moving_periodogram(x, dt=0.1, m=32, thin=2)
            wstem = s.LS2 / f"wdm_n{n}_r{r}"
            old_wdm = np.load(wstem.with_suffix(".npz"))
            raw_wdm = wdm_periodogram(
                TimeSeries(data=x, t=np.arange(n) * 0.1), nt=n // 32
            )
            for old, raw in ((old_mp, raw_mp), (old_wdm, raw_wdm)):
                for key in ("power", "counts", "time", "frequency"):
                    np.testing.assert_array_equal(
                        old[f"observed_{key}"], getattr(raw, key)
                    )
                assert old["weights"].shape[-2:] == model.weights.shape
            np.savez_compressed(ROOT / "inputs" / f"n{n}_r{r}.npz", x=x)
            recovered[f"n{n}_r{r}"] = dict(
                raw_sha256=s.b.digest(x),
                model_hashes=model_hashes,
                archive_mp_sha256=s.audit.file_hash(stem.with_suffix(".npz")),
                archive_wdm_sha256=s.audit.file_hash(
                    wstem.with_suffix(".npz")
                ),
            )
            if r == 0:
                audits[str(n)] = audit_observations(x, n)
    protocol = dict(
        date="2026-10-15",
        authorization="User requested Hann fits at all four intermediate lengths and mean plus/minus sample SD across five records.",
        lengths=LENGTHS,
        targets=cells(),
        recovered_inputs=recovered,
        preflight=audits,
        time_interior_knots={str(n): s.LS2_TIME_KNOTS[n] for n in LENGTHS},
        frequency_interior_knots=6,
        observation="symmetric Hann L=65, energy24, m32, thin2, original phase, rungs3,6,...30; exact physical center/N",
        priors="unchanged shared tensor model and physical priors from original endpoint study",
        scoring="same uniform96x48 physical grid on t[0.05,0.95], f[0.25,4.75]Hz; pointwise linear-PSD median and central90 intervals",
        attempts={
            str(n): [
                asdict(s.config(dict(dataset="LS2", n=n, record=0), a))
                for a in range(2)
            ]
            for n in LENGTHS
        },
        record_seed_rule="20261001+r; sampler17000+r",
        extensions="one predeclared sampler extension only; no target changes or records added",
        original_protocol_sha256=s.audit.file_hash(s.ROOT / "protocol.json"),
        runner_sha256=s.audit.file_hash(__file__),
        caveat="Finite-window mean mismatch and residual dependence are measured, not calibration passes.",
    )
    write("protocol.json", protocol)


def run(
    *,
    root: Path | None = None,
    target_cells: list[dict] | None = None,
    data_function=None,
    protocol: dict | None = None,
) -> list[dict]:
    if root is None:
        freeze()
        root = ROOT
    protocol = s.read(root / "protocol.json") if protocol is None else protocol
    target_cells = cells() if target_cells is None else target_cells
    data_function = observations if data_function is None else data_function

    def save(name: str, value: object) -> None:
        (root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    inventory = []
    for cell in target_cells:
        model = s.endpoint_model(cell["n"])
        provenance = protocol["recovered_inputs"][
            f"n{cell['n']}_r{cell['record']}"
        ]
        row = dict(**cell, **provenance, attempts=[], status="unresolved")
        if cell["method"] == "WDM":
            path = s.LS2 / f"wdm_n{cell['n']}_r{cell['record']}"
            m = s.read(path.with_suffix(".json"))
            for key in (
                "roughness_scale",
                "null_precision",
                "ridge_eps",
                "centered",
                "degree_time",
                "degree_freq",
                "penalty_order_time",
                "penalty_order_freq",
            ):
                assert m["config"][key] == asdict(s.config(cell, 0))[key], key
            post = xr.load_dataset(path.with_suffix(".posterior.nc"))
            stats = xr.load_dataset(path.with_suffix(".sample_stats.nc"))
            d = s.diagnostics(post, stats, model, "LS2")
            row.update(
                path=str(path),
                diagnostics=d,
                posterior_sha256=s.audit.file_hash(
                    path.with_suffix(".posterior.nc")
                ),
            )
            if d["pass"]:
                row["status"] = "reused"
        else:
            data = data_function(cell)
            for attempt in range(2):
                path = root / "fits" / cell["id"] / f"attempt{attempt}"
                path.mkdir(parents=True, exist_ok=True)
                metadata = path / "metrics.json"
                if not metadata.exists():
                    cfg = s.config(cell, attempt)
                    save(
                        "execution.json",
                        dict(
                            status="running",
                            target=cell["id"],
                            attempt=attempt,
                            pid=os.getpid(),
                        ),
                    )
                    print(
                        json.dumps(dict(starting=cell["id"], attempt=attempt)),
                        flush=True,
                    )
                    started = perf_counter()
                    try:
                        result = fit(data, cfg, model=model)
                        result.to_netcdf(path / "posterior.nc")
                        loaded = PSDResult.from_netcdf(path / "posterior.nc")
                        np.testing.assert_array_equal(
                            loaded.posterior.weights, result.posterior.weights
                        )
                        d = s.diagnostics(
                            loaded.posterior, loaded.sample_stats, model, "LS2"
                        )
                        m = dict(
                            config=asdict(cfg),
                            diagnostics=d,
                            **provenance,
                            posterior_sha256=s.audit.file_hash(
                                path / "posterior.nc"
                            ),
                            observations_sha256={
                                k: s.b.digest(getattr(data, k))
                                for k in (
                                    "power",
                                    "counts",
                                    "time",
                                    "frequency",
                                )
                            },
                            likelihood_observations=int(data.power.size),
                            inference_seconds=float(
                                result.metadata["inference_seconds"]
                            ),
                            total_seconds=perf_counter() - started,
                            runner_sha256=s.audit.file_hash(__file__),
                        )
                        s.b.write_json(metadata, m)
                        del result, loaded
                        jax.clear_caches()
                        gc.collect()
                    except Exception as exc:
                        s.b.write_json(
                            path / "failure.json",
                            dict(
                                error=repr(exc),
                                seconds=perf_counter() - started,
                            ),
                        )
                        raise
                m = s.read(metadata)
                assert (
                    m["raw_sha256"] == provenance["raw_sha256"]
                    and m["model_hashes"] == provenance["model_hashes"]
                )
                row["attempts"].append(dict(path=str(path), **m))
                if m["diagnostics"]["pass"]:
                    row.update(
                        status="completed",
                        path=str(path),
                        diagnostics=m["diagnostics"],
                        posterior_sha256=m["posterior_sha256"],
                    )
                    print(
                        json.dumps(
                            dict(
                                completed=cell["id"],
                                attempt=attempt,
                                diagnostics=m["diagnostics"],
                            )
                        ),
                        flush=True,
                    )
                    break
        inventory.append(row)
        save("inventory.json", inventory)
    save(
        "execution.json",
        dict(
            status="finished",
            resolved=sum(
                c["status"] in ("reused", "completed") for c in inventory
            ),
            targets=len(inventory),
        ),
    )
    return inventory


def summarize(
    *,
    root: Path | None = None,
    target_count: int = 40,
    include_original: bool = True,
    center_band_hz: tuple[float, float] = (3 / 6.5, 30 / 6.5),
) -> None:
    root = ROOT if root is None else root
    inv = s.read(root / "inventory.json")
    assert len(inv) == target_count and all(
        c["status"] in ("reused", "completed") for c in inv
    ), "Unresolved fit; do not promote diagnostic scores"
    u, f = np.linspace(0.05, 0.95, 96), np.linspace(0.25, 4.75, 48)
    rows = []
    for cell in inv:
        path = Path(cell["path"])
        posterior = (
            xr.load_dataset(path.with_suffix(".posterior.nc"))
            if cell["method"] == "WDM"
            else PSDResult.from_netcdf(path / "posterior.nc").posterior
        )
        q = s.quantiles(posterior, s.endpoint_model(cell["n"]), u, f / 5)
        truth = 0.2 * s.LS2Data(
            n_samples=cell["n"], fs=10, seed=cell["seed"]
        ).get_true_psd(time_grid=u, freq_grid=f)
        for region, mask in (
            ("all", np.ones(len(f), bool)),
            (
                "edge_prediction",
                (f < center_band_hz[0]) | (f > center_band_hz[1]),
            ),
            (
                "within_center_frequency_range",
                (f >= center_band_hz[0]) & (f <= center_band_hz[1]),
            ),
        ):
            if not mask.any():
                continue
            rows.append(
                dict(
                    target=cell["id"],
                    dataset="LS2",
                    n=cell["n"],
                    record=cell["record"],
                    method=cell["method"],
                    mask="complete",
                    region=region,
                    posterior_sha256=cell["posterior_sha256"],
                    **s.scores(q, truth, np.broadcast_to(mask, truth.shape)),
                )
            )
    with (root / "metrics.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = (
        list(csv.DictReader((s.ROOT / "metrics.csv").open()))
        if include_original
        else []
    )
    all_rows = [
        r for r in original if r["dataset"] == "LS2" and r["region"] == "all"
    ] + [r for r in rows if r["region"] == "all"]
    aggregate = {}
    for n in (2**k for k in range(10, 16)):
        for method in s.METHODS:
            records = [
                r
                for r in all_rows
                if int(r["n"]) == n and r["method"] == method
            ]
            assert (
                len(records) == 5
                and len({int(r["record"]) for r in records}) == 5
            )
            aggregate[f"{n}_{method}"] = dict(
                records=5,
                **{
                    metric: dict(
                        mean=float(
                            np.mean([float(r[metric]) for r in records])
                        ),
                        sample_sd=float(
                            np.std([float(r[metric]) for r in records], ddof=1)
                        ),
                    )
                    for metric in METRICS
                },
            )
    summary = dict(
        status="complete",
        aggregate=aggregate,
        reporting="mean plus/minus sample standard deviation across five independent records (ddof=1), not standard error or posterior uncertainty",
        status_counts={
            k: sum(c["status"] == k for c in inv)
            for k in ("reused", "completed")
        },
        failed_attempts=[
            a
            for c in inv
            for a in c["attempts"]
            if not a["diagnostics"]["pass"]
        ],
        inputs=dict(
            original_metrics_sha256=s.audit.file_hash(s.ROOT / "metrics.csv")
            if include_original
            else None,
            extension_metrics_sha256=s.audit.file_hash(root / "metrics.csv"),
        ),
    )
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    table = [
        r"\begin{tabular}{llccc}",
        r"$N$ & Method & RMSE$_{\log S}$ & $\overline W$ & Inc90\\",
        r"\hline",
    ]
    for key, result in aggregate.items():
        n, method = key.split("_")
        values = [
            f"${result[m]['mean']:.3f}\\pm{result[m]['sample_sd']:.3f}$"
            for m in METRICS
        ]
        table.append(
            f"{int(n):,} & {'Hann MP' if method == 'MP' else 'WDM'} & "
            + " & ".join(values)
            + r" \\"
        )
    table.append(r"\end{tabular}")
    (root / "ls2_all_lengths_table.tex").write_text("\n".join(table) + "\n")
    print(
        json.dumps(
            dict(
                status="complete",
                records=len(all_rows),
                new_fits=sum(c["status"] == "completed" for c in inv),
                reused_WDM=sum(c["status"] == "reused" for c in inv),
            )
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "summarize"))
    args = parser.parse_args()
    if args.command == "run":
        run()
        summarize()
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
