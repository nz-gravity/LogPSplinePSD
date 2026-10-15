"""Fixed-mask gap pilot; reuse the original model, prior and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from experiments.dw_wdm_small import benchmark as bench

from log_psplines import PowerConfig, PowerData, fit

ROOT = Path(__file__).resolve().parent / "gaps" / "column_gated"
EPSILON = 1e-4


def masks() -> dict[str, np.ndarray]:
    contiguous = np.ones(bench.N, dtype=bool)
    length = round(0.1 * bench.N)
    start = (bench.N - length) // 2
    contiguous[start : start + length] = False
    fragmented = np.ones(bench.N, dtype=bool)
    length = round(0.1 * bench.N / 16)
    for centre in np.linspace(0, bench.N, 18, dtype=int)[1:-1]:
        fragmented[centre - length // 2 : centre - length // 2 + length] = (
            False
        )
    return {"contiguous": contiguous, "fragmented": fragmented}


def dw_keep(observed: np.ndarray, *, whole_cycles: bool = False) -> np.ndarray:
    """Subselect the original thinned schedule; never compress physical time."""
    raw = bench.tang_moving_periodogram(
        np.zeros(bench.N), m=bench.M, thin=bench.THIN
    )
    centres = np.rint(raw["u"] * bench.N).astype(int) - 1
    complete = np.convolve(
        observed.astype(int), np.ones(2 * bench.M + 1, dtype=int), "valid"
    )
    keep = complete[centres - bench.M] == 2 * bench.M + 1
    if whole_cycles:
        keep = np.repeat(keep.reshape(-1, bench.M).all(axis=1), bench.M)
    return keep


def dw_data(x: np.ndarray, observed: np.ndarray) -> PowerData:
    # Filling is only an internal transform convenience. All windows touching
    # these placeholders are excluded, so retained coefficients cannot use them.
    data = bench.prepare(np.where(observed, x, 0.0), "dw")
    keep = dw_keep(observed)
    return PowerData(
        data.power[keep],
        data.counts[keep],
        data.frequency[keep],
        data.time[keep],
        data.units,
    )


def kernel_expectations(
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact diagonal means for this independent heteroscedastic Gaussian case."""
    rho = np.zeros((bench.NT, bench.NF - 1))
    full = np.zeros_like(rho)
    masked = np.zeros_like(rho)
    v = bench.variance(np.arange(bench.N) / bench.N)
    spectra = [
        np.fft.fft(a) for a in (observed.astype(float), v, v * observed)
    ]
    for f in range(1, bench.NF):
        for parity in (0, 1):
            t0 = bench.NT // 2 + parity
            row = bench.wdm_row(t0, f)
            weights = row**2
            np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
            wf = np.conj(np.fft.fft(weights))
            times = np.arange(parity, bench.NT, 2)
            shifts = ((times - t0) * bench.NF) % bench.N
            for output, spectrum in zip(
                (rho, full, masked), spectra, strict=True
            ):
                output[times, f - 1] = np.fft.ifft(spectrum * wf).real[shifts]
    return rho[1:-1], full[1:-1], masked[1:-1]


def wdm_keep(rho: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Retain a whole time column only if every frequency passes the guard."""
    columns = np.all(rho >= 1 - epsilon, axis=1)
    return np.broadcast_to(columns[:, None], rho.shape)


def wdm_data(
    x: np.ndarray, observed: np.ndarray, rho: np.ndarray
) -> PowerData:
    data = bench.prepare(np.where(observed, x, 0.0), "wdm")
    keep = wdm_keep(rho)
    power = np.divide(
        data.power, rho, out=np.zeros_like(data.power), where=keep
    )
    return PowerData(
        power,
        np.where(keep, data.counts, 0.0),
        data.frequency,
        data.time,
        data.units,
    )


def probes(observed: np.ndarray, rho: np.ndarray) -> dict:
    edges = np.flatnonzero(observed[1:] != observed[:-1]) + 1
    dw = dw_data(np.zeros(bench.N), observed)
    distances = np.min(abs(dw.time[:, None] * bench.N - edges), axis=1)
    indices = np.argsort(distances, kind="stable")[:32]
    dw_rows = np.zeros((len(indices), bench.N), dtype=complex)
    for i, q in enumerate(indices):
        centre = round(dw.time[q] * bench.N)
        dw_rows[i, centre - bench.M : centre + bench.M + 1] = np.exp(
            -2j
            * np.pi
            * bench.DT
            * dw.frequency[q]
            * np.arange(2 * bench.M + 1)
        ) / np.sqrt(2 * bench.M + 1)
    locations = np.argwhere(wdm_keep(rho))
    if not len(locations):
        return {"wdm": {"no_retained_columns": True}}
    distance = np.min(
        abs((locations[:, 0, None] + 1) * bench.NF - edges), axis=1
    )
    # Check adjacent frequencies/time rows near the guard, not just mid-record.
    locations = locations[np.argsort(distance, kind="stable")[:32]]
    wdm_rows = np.array(
        [
            bench.wdm_row(int(t + 1), int(f + 1))
            * observed
            / np.sqrt(rho[t, f])
            for t, f in locations
        ]
    )
    v = bench.variance(np.arange(bench.N) / bench.N)
    return {
        method: {
            name: bench.power_correlation(rows, values)
            for name, values in (("white", np.ones(bench.N)), ("modulated", v))
        }
        for method, rows in (("dw", dw_rows), ("wdm", wdm_rows))
    }


def audit() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    cases = {}
    x = bench.simulate(6101)
    for name, observed in masks().items():
        keep = dw_keep(observed)
        data = dw_data(x, observed)
        full_dw = bench.prepare(x, "dw")
        np.testing.assert_array_equal(data.power, full_dw.power[keep])
        changed = x.copy()
        changed[~observed] = 1e50
        np.testing.assert_array_equal(
            data.power, dw_data(changed, observed).power
        )
        nan = x.copy()
        nan[~observed] = np.nan
        np.testing.assert_array_equal(data.power, dw_data(nan, observed).power)
        kernel_started = perf_counter()
        rho, full, masked = kernel_expectations(observed)
        kernel_seconds = perf_counter() - kernel_started
        np.savez_compressed(
            ROOT / f"{name}_mask.npz",
            observed=observed,
            rho=rho,
            full_mean=full,
            masked_mean=masked,
        )
        kept = wdm_keep(rho)
        v = bench.variance(np.arange(1, bench.NT - 1) / bench.NT)[:, None]
        wdm_bias = (masked[kept] / rho[kept]) / np.broadcast_to(v, rho.shape)[
            kept
        ] - 1
        cases[name] = {
            "raw_gap_fraction": float(np.mean(~observed)),
            "dw_observations": int(keep.sum()),
            "dw_retained_fraction": float(keep.mean()),
            "dw_complete_cycle_observations": int(
                dw_keep(observed, whole_cycles=True).sum()
            ),
            "wdm_observations": int(kept.sum()),
            "wdm_retained_fraction": float(kept.mean()),
            "wdm_retained_time_columns": int(kept.all(axis=1).sum()),
            "wdm_kernel_preparation_seconds": kernel_seconds,
            "wdm_partial_time_columns": int(
                (kept.any(axis=1) != kept.all(axis=1)).sum()
            ),
            "wdm_threshold_sensitivity": {
                str(e): int(wdm_keep(rho, e).sum())
                for e in (1e-2, 1e-3, 1e-4, 1e-6)
            },
            "wdm_max_pointwise_mean_bias": (
                float(np.max(abs(wdm_bias))) if kept.any() else None
            ),
            "wdm_max_naive_center_only_power_bias": float(
                np.max(
                    abs(masked / full - 1)[
                        np.broadcast_to(
                            observed[(np.arange(1, bench.NT - 1) * bench.NF)],
                            rho.shape[::-1],
                        ).T
                    ]
                )
            ),
            "dependence_probes": probes(observed, rho),
            "dw_retained_powers_identical_to_ungapped": True,
            "dw_missing_values_invariance": True,
        }
    bench.write_json(
        ROOT / "audit.json",
        {
            "seed": 6101,
            "wdm_max_lost_energy": EPSILON,
            "wdm_gating": "whole_time_columns_all_frequencies",
            "cases": cases,
        },
    )
    print(json.dumps(cases, indent=2))


def fit_case(name: str) -> None:
    saved = np.load(ROOT / f"{name}_mask.npz")
    observed, rho = saved["observed"], saved["rho"]
    audit = json.loads((ROOT / "audit.json").read_text())["cases"][name]
    assert audit["wdm_observations"] > 0, "No WDM columns survive; do not fit."
    assert audit["wdm_max_pointwise_mean_bias"] < 0.01
    assert (
        audit["dependence_probes"]["wdm"]["modulated"]["max_power_correlation"]
        < 0.01
    )

    for method in ("dw", "wdm"):
        path = ROOT / name / method
        path.mkdir(parents=True, exist_ok=True)
        x = bench.simulate(6101)
        started = perf_counter()
        data = (
            dw_data(x, observed)
            if method == "dw"
            else wdm_data(x, observed, rho)
        )
        processed = perf_counter()
        baseline_path = bench.OUT / f"seed6101_{method}"
        baseline = json.loads((baseline_path / "metrics.json").read_text())
        config = PowerConfig(**baseline["config"])
        master = bench.master_model()
        model = (
            bench.on_grid(master, data.time, data.frequency)
            if data.is_grid
            else master
        )
        result = fit(data, config, model=model)
        saved = np.load(baseline_path / "surface.npz")
        u, f = saved["time"], saved["freq"]
        logs = np.einsum(
            "ti,cdij,fj->cdtf",
            master.time.design_at(u),
            result.posterior.weights.values,
            master.frequency.design_at(f),
            optimize=True,
        )
        small = xr.Dataset(
            {"log_psd": (("chain", "draw", "t", "f"), logs[:, :, ::10, ::14])}
        )
        summaries = [
            azs.summary(
                result.posterior[["s", "sigma_time", "sigma_freq"]],
                round_to="none",
            ),
            azs.summary(small, round_to="none"),
        ]
        diag = {
            "divergences": int(result.sample_stats.diverging.sum()),
            "depth_hits": int(
                (
                    result.sample_stats.n_steps >= 2**config.max_tree_depth - 1
                ).sum()
            ),
            "rhat_max": float(max(s.r_hat.max() for s in summaries)),
            "ess_bulk_min": float(min(s.ess_bulk.min() for s in summaries)),
            "ess_tail_min": float(min(s.ess_tail.min() for s in summaries)),
        }
        diag["pass"] = (
            diag["divergences"] == 0
            and diag["depth_hits"] == 0
            and diag["rhat_max"] <= 1.01
            and min(diag["ess_bulk_min"], diag["ess_tail_min"]) >= 200
        )
        low, median, high = np.quantile(
            logs.reshape(-1, len(u), len(f)), [0.05, 0.5, 0.95], axis=0
        )
        np.savez_compressed(
            path / "surface.npz",
            time=u,
            freq=f,
            truth=saved["truth"],
            low=low,
            median=median,
            high=high,
        )
        result.to_netcdf(path / "posterior.nc")
        metrics = {
            "seed": 6101,
            "method": method,
            "case": name,
            "mask_sha256": bench.digest(observed),
            "raw_sha256": bench.digest(x),
            "observed_raw_sha256": bench.digest(np.where(observed, x, 0.0)),
            "model_hashes": baseline["model_hashes"],
            "config": baseline["config"],
            "preprocess_seconds": processed - started,
            "model_preparation_seconds": result.metadata[
                "preparation_seconds"
            ],
            "inference_seconds": result.metadata["inference_seconds"],
            "likelihood_observations": int(np.count_nonzero(data.counts)),
            "real_component_count": float(data.counts.sum()),
            "log_psd_mse": bench.area_mean(
                (median - saved["truth"]) ** 2, u, f
            ),
            "coverage90": bench.area_mean(
                (low <= saved["truth"]) & (saved["truth"] <= high), u, f
            ),
            "median_interval_width": float(
                np.median(np.exp(high) - np.exp(low))
            ),
            "diagnostics": diag,
            "gap_kernel_audit_excluded_from_preprocessing": True,
            "wdm_kernel_preparation_seconds": (
                audit["wdm_kernel_preparation_seconds"]
                if method == "wdm"
                else 0.0
            ),
            "wdm_gating": "whole_time_columns_all_frequencies",
        }
        bench.write_json(path / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)


def summarize(name: str) -> None:
    observed = masks()[name]
    edges = np.flatnonzero(observed[1:] != observed[:-1]) + 1
    x = bench.simulate(6101)
    kernel = np.load(ROOT / f"{name}_mask.npz")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), layout="constrained")
    summary = {}
    for column, (method, color, label) in enumerate(
        (("dw", "#2866A2", "Dynamic Whittle"), ("wdm", "#B46A1D", "WDM"))
    ):
        d = np.load(ROOT / name / method / "surface.npz")
        base = np.load(bench.OUT / f"seed6101_{method}" / "surface.npz")
        metrics = json.loads(
            (ROOT / name / method / "metrics.json").read_text()
        )
        u, f = d["time"], d["freq"]
        in_gap = ~observed[np.minimum((u * bench.N).astype(int), bench.N - 1)]
        distance = np.min(abs(u[:, None] - edges[None, :] / bench.N), axis=1)
        regions = {
            "gap": in_gap,
            "observed_near_edges": (~in_gap) & (distance <= 0.05),
            "observed_away": (~in_gap) & (distance > 0.05),
        }
        out = {}
        for region, mask in regions.items():
            mask = np.broadcast_to(mask[:, None], d["median"].shape)
            denominator = bench.area_mean(mask, u, f)
            out[region] = {
                "log_psd_mse": bench.area_mean(
                    (d["median"] - d["truth"]) ** 2 * mask, u, f
                )
                / denominator,
                "truth_in_90pct_interval": bench.area_mean(
                    (
                        (d["low"] <= d["truth"]) & (d["truth"] <= d["high"])
                    ).astype(float)
                    * mask,
                    u,
                    f,
                )
                / denominator,
                "median_width": float(
                    np.median((np.exp(d["high"]) - np.exp(d["low"]))[mask])
                ),
                "width_ratio_to_ungapped": float(
                    np.median((np.exp(d["high"]) - np.exp(d["low"]))[mask])
                    / np.median(
                        (np.exp(base["high"]) - np.exp(base["low"]))[mask]
                    )
                ),
            }
        summary[method] = {
            "regions": out,
            "diagnostics": metrics["diagnostics"],
            "inference_seconds": metrics["inference_seconds"],
        }
        j = int(np.argmin(abs(f - 0.25)))
        ax = axes[0, column]
        ax.plot(
            u,
            np.exp(d["truth"][:, j]),
            color="#222222",
            ls=":",
            lw=2,
            label="Truth",
        )
        ax.plot(
            u,
            np.exp(base["median"][:, j]),
            color=color,
            ls="--",
            alpha=0.65,
            label="No-gap median",
        )
        ax.plot(u, np.exp(d["median"][:, j]), color=color, label="Gap median")
        ax.fill_between(
            u,
            np.exp(d["low"][:, j]),
            np.exp(d["high"][:, j]),
            color=color,
            alpha=0.18,
            label="Gap: pointwise 90% interval",
        )
        ax.set(
            title=f"{label}: PSD at f={f[j]:.2f} Hz",
            xlabel="Rescaled time u",
            ylabel="One-sided PSD",
            xlim=(0.30, 0.70),
            ylim=(0, 4.5),
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
        data = (
            dw_data(x, observed)
            if method == "dw"
            else wdm_data(x, observed, kernel["rho"])
        )
        ax = axes[1, column]
        if data.is_grid:
            ts, fs = np.meshgrid(data.time, data.frequency, indexing="ij")
            active = data.counts > 0
            ts, fs = ts[active], fs[active]
        else:
            ts, fs = data.time, data.frequency
        ax.scatter(ts, fs, s=1.5, color=color, rasterized=True)
        ax.set(
            title="Spectral coefficients used in the fit",
            xlabel="Rescaled time u",
            ylabel="Frequency (Hz)",
            xlim=(0.30, 0.70),
            ylim=(0, 0.5),
        )
        for a in axes[:, column]:
            for lo, hi in zip(edges[::2], edges[1::2], strict=True):
                a.axvspan(
                    lo / bench.N,
                    hi / bench.N,
                    color="#999999",
                    alpha=0.15,
                    zorder=0,
                )
    fig.suptitle(
        "One fixed 10% gap; seed 6101; shared PSD model and prior", fontsize=13
    )
    fig.savefig(ROOT / "gap_comparison.png", dpi=180)
    fig.savefig(ROOT / "gap_comparison.pdf")
    plt.close(fig)
    bench.write_json(
        ROOT / "regional_metrics.json",
        {"case": name, "seed": 6101, "results": summary},
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "fit", "summarize"))
    parser.add_argument(
        "--case", choices=("contiguous", "fragmented"), default="contiguous"
    )
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, default=bench.OUT)
    args = parser.parse_args()
    ROOT = args.output
    bench.OUT = args.baseline
    if args.action == "audit":
        audit()
    elif args.action == "fit":
        fit_case(args.case)
    else:
        summarize(args.case)


if __name__ == "__main__":
    main()
