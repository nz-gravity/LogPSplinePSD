"""Repeated central-gap taper comparison using the existing Gaussian benchmark.

Run audit, then fit, then summarize. The latter scores archived posterior
quantiles independently of the inference helper. Production code is unchanged.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import gap_smoke as gap
from experiments.dw_wdm_small import taper_smoke as taper
from experiments.dw_wdm_small.rerun_summary import axis_weights
from scipy.signal import periodogram
from scipy.stats import t as student_t

from log_psplines import PowerData, PSDResult

ROOT = bench.ROOT / "taper_study"
SEEDS = list(range(6101, 6111))
VARIANTS = ("rectangular", "boxcar_sparse", "hann_sparse", "wdm")
LABELS = {
    "rectangular": "Rectangular (original)",
    "boxcar_sparse": "Rectangular (matched sparse)",
    "hann_sparse": "Hann (sparse)",
    "wdm": "WDM (guarded)",
}
COLORS = ("#2866A2", "#777777", "#B46A1D", "#697D35")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def data_adapter(
    x: np.ndarray,
    method: str,
    *,
    variant: str,
    observed: np.ndarray,
    rho: np.ndarray,
) -> PowerData:
    if variant == "rectangular":
        return gap.dw_data(x, observed)
    if variant == "wdm":
        return gap.wdm_data(x, observed, rho)
    return taper.prepare(
        x, "hann" if variant == "hann_sparse" else "boxcar", observed
    )


def all_local_dependence(h: np.ndarray, observed: np.ndarray) -> dict:
    """Every retained overlapping pair, for the full 8192-sample schedule.

    Disjoint rows are exactly independent for this amplitude-modulated white
    process. Thus these are full selection maxima, rather than probe maxima.
    """
    raw = taper.schedule()
    selected = np.flatnonzero(taper.retained(observed))
    centres = np.rint(raw["u"][selected] * bench.N).astype(int) - 1
    phase = np.exp(-1j * raw["omega"][selected, None] * np.arange(len(h)))
    rows = phase * h / np.linalg.norm(h)
    white_power_variance = 1 + abs(np.sum(rows**2, axis=1)) ** 2
    v = bench.variance(np.arange(bench.N) / bench.N)
    windows = np.lib.stride_tricks.sliding_window_view(v, len(h))[
        centres - bench.M
    ]
    marginal = np.sum(abs(rows) ** 2 * windows, axis=1)
    pseudo = np.sum(rows**2 * windows, axis=1)
    power_variance = marginal**2 + abs(pseudo) ** 2
    maxima = {"white": 0.0, "modulated": 0.0}
    sums = {k: np.zeros(len(rows)) for k in maxima}
    pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            shift = centres[j] - centres[i]
            if shift >= len(h):
                break
            a, b = rows[i, shift:], rows[j, : len(h) - shift]
            cv, pv = a @ b.conj(), a @ b
            corr_white = (abs(cv) ** 2 + abs(pv) ** 2) / np.sqrt(
                white_power_variance[i] * white_power_variance[j]
            )
            vv = windows[i, shift:]
            cv, pv = (a * vv) @ b.conj(), (a * vv) @ b
            corr = (abs(cv) ** 2 + abs(pv) ** 2) / np.sqrt(
                power_variance[i] * power_variance[j]
            )
            for key, value in (("white", corr_white), ("modulated", corr)):
                maxima[key] = max(maxima[key], float(value))
                sums[key][i] += value
                sums[key][j] += value
            pairs += 1
    return {
        "retained": len(rows),
        "overlapping_pairs_checked": pairs,
        "max_power_correlation": maxima,
        "max_modulated_nonproperness": float(np.max(abs(pseudo) / marginal)),
        "max_sum_off_diagonal_correlations": {
            key: float(values.max()) for key, values in sums.items()
        },
    }


def audit() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    observed = gap.masks()["contiguous"]
    start = perf_counter()
    rho, _, _ = gap.kernel_expectations(observed)
    kernel_seconds = perf_counter() - start
    np.savez_compressed(ROOT / "mask.npz", observed=observed, rho=rho)
    taper.audit()
    result = {"normalization": load_json(taper.ROOT / "audit.json")}
    x = bench.simulate(6101)
    raw = taper.schedule()
    keep = taper.retained(observed)
    centres = np.rint(raw["u"] * bench.N).astype(int) - 1
    scipy_errors = {}
    for name in ("boxcar", "hann"):
        h = taper.weights(name)
        data = taper.prepare(x, name, observed)
        full = taper.prepare(x, name, np.ones(bench.N, dtype=bool))
        full_keep = taper.retained(np.ones(bench.N, dtype=bool))
        np.testing.assert_array_equal(data.power, full.power[keep[full_keep]])
        for value in (np.nan, 1e50, -1e50):
            changed = x.copy()
            changed[~observed] = value
            np.testing.assert_array_equal(
                data.power, taper.prepare(changed, name, observed).power
            )
        for c in centres[keep]:
            assert observed[c - bench.M : c + bench.M + 1].all()
        # Independent library normalization at every retained window/rung.
        windows = np.lib.stride_tricks.sliding_window_view(x, len(h))[
            centres[keep] - bench.M
        ]
        freq, spectrum = periodogram(
            windows,
            fs=1 / bench.DT,
            window=h,
            detrend=False,
            scaling="density",
            axis=-1,
        )
        rungs = np.rint(data.frequency * len(h) * bench.DT).astype(int)
        reference = spectrum[np.arange(len(rungs)), rungs]
        np.testing.assert_allclose(data.power / 2, reference, atol=2e-13)
        scipy_errors[name] = float(np.max(abs(data.power / 2 - reference)))
    a, b = (taper.prepare(x, name, observed) for name in ("boxcar", "hann"))
    for field in ("time", "frequency", "counts"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    result.update(
        {
            "scipy_density_max_error": scipy_errors,
            "missing_value_invariance": True,
            "matched_sparse_sites_and_counts": True,
            "retained_power_equals_ungapped_power": True,
            "whole_selection_dependence": {
                name: all_local_dependence(taper.weights(name), observed)
                for name in ("boxcar", "hann")
            },
            "wdm_kernel_seconds_once_per_mask": kernel_seconds,
            "counts": {
                name: int(
                    np.count_nonzero(
                        data_adapter(
                            x,
                            "wdm" if name == "wdm" else "dw",
                            variant=name,
                            observed=observed,
                            rho=rho,
                        ).counts
                    )
                )
                for name in VARIANTS
            },
        }
    )
    for name in ("boxcar", "hann"):
        dep = result["whole_selection_dependence"][name]
        assert dep["max_power_correlation"]["modulated"] < 0.01
        assert dep["max_modulated_nonproperness"] < 0.02
    bench.write_json(ROOT / "validation.json", result)
    files = [
        "src/log_psplines/inference/power.py",
        "src/log_psplines/preprocessing/moving_periodogram.py",
        "src/log_psplines/preprocessing/wdm.py",
        "experiments/dw_wdm_small/benchmark.py",
        "experiments/dw_wdm_small/gap_smoke.py",
        "experiments/dw_wdm_small/taper_smoke.py",
        "experiments/dw_wdm_small/taper_study.py",
    ]
    bench.write_json(
        ROOT / "protocol.json",
        {
            "seeds": SEEDS,
            "variants": list(VARIANTS),
            "N": bench.N,
            "m": bench.M,
            "thin": bench.THIN,
            "wdm_nt": bench.NT,
            "scoring_domain": {
                "time": [0.05, 0.95],
                "frequency_hz": [0.05, 0.45],
            },
            "primary_mask": "one central 819-sample gap",
            "mask_sha256": bench.digest(observed),
            "sampler": "2 chains, 600 warmup, 1000 draws, target_accept=0.95",
            "python": platform.python_version(),
            "versions": {
                k: version(k)
                for k in ("jax", "numpy", "numpyro", "wdm_transform", "scipy")
            },
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "preexisting_tracked_diff": subprocess.check_output(
                ["git", "diff", "--stat"], text=True
            ),
            "source_sha256": {
                p: bench.digest(
                    np.frombuffer(Path(p).read_bytes(), dtype=np.uint8)
                )
                for p in files
            },
            "seed6101_baselines_reused": "gaps/column_gated/contiguous/{dw,wdm}",
            "limitations": [
                "amplitude-modulated white Gaussian noise",
                "fixed central mask",
                "guarded diagonal WDM likelihood",
                "10 development seeds; exploratory paired intervals",
            ],
        },
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "normalization"}, indent=2
        ),
        flush=True,
    )


def fit_study(
    seeds: list[int],
    variants: list[str],
    *,
    warmup: int = 600,
    samples: int = 1000,
    target_accept: float = 0.95,
    repair: bool = False,
) -> None:
    saved = np.load(ROOT / "mask.npz")
    observed, rho = saved["observed"], saved["rho"]
    audit_result = load_json(ROOT / "validation.json")
    from functools import partial

    for seed in seeds:
        for variant in variants:
            method = "wdm" if variant == "wdm" else "dw"
            bench.OUT = ROOT / ("repairs" if repair else "fits") / variant
            path = bench.OUT / f"seed{seed}_{method}"
            if (path / "metrics.json").exists():
                print(f"Resume: retained {variant}, seed {seed}", flush=True)
                continue
            old = gap.ROOT / "contiguous" / method
            if (
                not repair
                and seed == 6101
                and variant in ("rectangular", "wdm")
            ):
                shutil.copytree(old, path, dirs_exist_ok=True)
                row = load_json(path / "metrics.json")
                assert row["raw_sha256"] == bench.digest(bench.simulate(seed))
                assert row["mask_sha256"] == bench.digest(observed)
                row["reused_from"] = str(old.relative_to(bench.ROOT))
            else:
                adapter = partial(
                    data_adapter, variant=variant, observed=observed, rho=rho
                )
                row = bench.infer(
                    seed,
                    method,
                    warmup,
                    samples,
                    target_accept,
                    preprocess=adapter,
                )
            # A gapped WDM check must use exactly the retained masked powers.
            # Recompute this metric rather than use the helper's full-record check.
            data = data_adapter(
                bench.simulate(seed),
                method,
                variant=variant,
                observed=observed,
                rho=rho,
            )
            result = PSDResult.from_netcdf(path / "posterior.nc")
            master = bench.master_model()
            weights = np.median(result.posterior.weights.values, axis=(0, 1))
            model = (
                bench.on_grid(master, data.time, data.frequency)
                if data.is_grid
                else master
            )
            if data.is_grid:
                fitted = model(weights)
            else:
                fitted = np.einsum(
                    "pi,ij,pj->p",
                    master.time.design_at(data.time),
                    weights,
                    master.frequency.design_at(data.frequency),
                )
            valid = data.counts > 0
            row.update(
                {
                    "variant": variant,
                    "case": "contiguous",
                    "mask_sha256": bench.digest(observed),
                    "observed_raw_sha256": bench.digest(
                        np.where(observed, bench.simulate(seed), 0.0)
                    ),
                    "whitening": {
                        "mean_squared_magnitude_retained_only": float(
                            np.mean(
                                data.power[valid]
                                / data.counts[valid]
                                / np.exp(fitted[valid])
                            )
                        ),
                        "expected": 1.0,
                        "diagnostic_only_fitted_on_same_data": True,
                    },
                    "wdm_kernel_seconds_once_per_mask": audit_result[
                        "wdm_kernel_seconds_once_per_mask"
                    ]
                    if variant == "wdm"
                    else 0.0,
                }
            )
            bench.write_json(path / "metrics.json", row)
            print(
                json.dumps(
                    {
                        "completed": variant,
                        "seed": seed,
                        "diagnostics": row["diagnostics"],
                        "inference_seconds": row["inference_seconds"],
                    }
                ),
                flush=True,
            )


def score(path: Path) -> dict:
    d = np.load(path / "surface.npz")
    observed = np.load(ROOT / "mask.npz")["observed"]
    u, f = d["time"], d["freq"]
    band = (f >= 0.05) & (f <= 0.45)
    f = f[band]
    area = np.outer(axis_weights(u), axis_weights(f))
    delta = d["median"][:, band] - d["truth"][:, band]
    included = (d["low"][:, band] <= d["truth"][:, band]) & (
        d["truth"][:, band] <= d["high"][:, band]
    )
    width = np.exp(d["high"][:, band]) - np.exp(d["low"][:, band])
    edges = (np.flatnonzero(observed[1:] != observed[:-1]) + 1) / bench.N
    inside = ~observed[np.rint(u * bench.N).astype(int)]
    near = np.min(abs(u[:, None] - edges), axis=1) <= 0.03
    regions = {
        "all": np.ones(len(u), dtype=bool),
        "gap": inside,
        "edge_observed": near & ~inside,
        "observed_far": ~near & ~inside,
    }
    out = {}
    for name, selected in regions.items():
        weights = area * selected[:, None]
        weights /= weights.sum()
        out[name] = {
            "log_psd_mse": float(np.sum(weights * delta**2)),
            "signed_log_error": float(np.sum(weights * delta)),
            "mean_relative_psd_error": float(
                np.sum(weights * np.expm1(delta))
            ),
            "coverage90": float(np.sum(weights * included)),
            "median_interval_width": float(np.median(width[selected])),
            "median_log_interval_width": float(
                np.median((d["high"][:, band] - d["low"][:, band])[selected])
            ),
        }
    return out


def paired_interval(values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    radius = float(
        student_t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return {
        "mean": mean,
        "exploratory_t_interval95": [mean - radius, mean + radius],
    }


def summarize() -> None:
    rows = []
    aggregate = {}
    reference_hashes = {}
    observed_hashes = {}
    configs = {}
    model_hashes = None
    for variant in VARIANTS:
        selected = []
        method = "wdm" if variant == "wdm" else "dw"
        for seed in SEEDS:
            path = ROOT / "fits" / variant / f"seed{seed}_{method}"
            row = load_json(path / "metrics.json")
            row["regions"] = score(path)
            reference_hashes.setdefault(seed, row["raw_sha256"])
            observed_hashes.setdefault(seed, row["observed_raw_sha256"])
            configs.setdefault(seed, row["config"])
            assert row["raw_sha256"] == reference_hashes[seed]
            assert row["observed_raw_sha256"] == observed_hashes[seed]
            assert row["mask_sha256"] == bench.digest(
                np.load(ROOT / "mask.npz")["observed"]
            )
            assert row["config"] == configs[seed]
            if model_hashes is None:
                model_hashes = row["model_hashes"]
            assert row["model_hashes"] == model_hashes
            row["variant"] = variant
            rows.append(row)
            selected.append(row)
        aggregate[variant] = {
            "sampler_passes": sum(r["diagnostics"]["pass"] for r in selected),
            "flagged_seeds": [
                r["seed"] for r in selected if not r["diagnostics"]["pass"]
            ],
            "observations": selected[0]["likelihood_observations"],
            "real_components": selected[0]["real_component_count"],
            "mean_inference_seconds": float(
                np.mean([r["inference_seconds"] for r in selected])
            ),
            "mean_preprocess_seconds": float(
                np.mean([r["preprocess_seconds"] for r in selected])
            ),
            "mean_fresh_paired_inference_seconds": float(
                np.mean(
                    [
                        r["inference_seconds"]
                        for r in selected
                        if r["seed"] != 6101
                    ]
                )
            ),
            "runtime_common_fresh_seeds": SEEDS[1:],
            "regions": {
                region: {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    r["regions"][region][metric]
                                    for r in selected
                                ]
                            )
                        ),
                        "sd": float(
                            np.std(
                                [
                                    r["regions"][region][metric]
                                    for r in selected
                                ],
                                ddof=1,
                            )
                        ),
                    }
                    for metric in selected[0]["regions"][region]
                }
                for region in selected[0]["regions"]
            },
        }
    paired = {}
    for a, b in (
        ("hann_sparse", "boxcar_sparse"),
        ("hann_sparse", "rectangular"),
        ("hann_sparse", "wdm"),
        ("rectangular", "wdm"),
    ):
        paired[f"{a}_minus_{b}"] = {}
        for region in ("all", "gap", "edge_observed", "observed_far"):
            values = np.array(
                [
                    next(
                        r["regions"][region]["log_psd_mse"]
                        for r in rows
                        if r["seed"] == seed and r["variant"] == a
                    )
                    - next(
                        r["regions"][region]["log_psd_mse"]
                        for r in rows
                        if r["seed"] == seed and r["variant"] == b
                    )
                    for seed in SEEDS
                ]
            )
            paired[f"{a}_minus_{b}"][region] = {
                **paired_interval(values),
                "first_method_lower_error_seeds": int((values < 0).sum()),
            }
    bench.write_json(ROOT / "rows.json", rows)
    counterfactual = {}
    for variant, method in (("rectangular", "dw"), ("wdm", "wdm")):
        shifts, differences = [], []
        for seed in SEEDS:
            baseline_path = (
                bench.ROOT / "results/reuse_rerun" / f"seed{seed}_{method}"
            )
            baseline_metrics = load_json(baseline_path / "metrics.json")
            row = next(
                r
                for r in rows
                if r["variant"] == variant and r["seed"] == seed
            )
            assert baseline_metrics["raw_sha256"] == row["raw_sha256"]
            assert baseline_metrics["model_hashes"] == row["model_hashes"]
            assert baseline_metrics["config"] == row["config"]
            baseline = score(baseline_path)["gap"]
            shifts.append(
                row["regions"]["gap"]["signed_log_error"]
                - baseline["signed_log_error"]
            )
            differences.append(
                row["regions"]["gap"]["log_psd_mse"] - baseline["log_psd_mse"]
            )
        counterfactual[variant] = {
            "gap_region_mean_log_shift_from_ungapped": paired_interval(
                np.asarray(shifts)
            ),
            "gap_region_mse_change_from_ungapped": paired_interval(
                np.asarray(differences)
            ),
            "source": "results/reuse_rerun, same ten raw records",
        }
    bench.write_json(
        ROOT / "aggregate.json",
        {
            "methods": aggregate,
            "paired_mse_differences": paired,
            "ungapped_counterfactual": counterfactual,
            "n_seeds": len(SEEDS),
            "frequency_band_hz": [0.05, 0.45],
        },
    )
    import csv

    with (ROOT / "metrics.csv").open("w") as fp:
        fields = [
            "seed",
            "variant",
            "region",
            "log_psd_mse",
            "signed_log_error",
            "mean_relative_psd_error",
            "coverage90",
            "median_interval_width",
            "sampler_pass",
            "inference_seconds",
        ]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for region, metrics in row["regions"].items():
                writer.writerow(
                    {
                        **{k: metrics[k] for k in fields if k in metrics},
                        "seed": row["seed"],
                        "variant": row["variant"],
                        "region": region,
                        "sampler_pass": row["diagnostics"]["pass"],
                        "inference_seconds": row["inference_seconds"],
                    }
                )
    figures(rows)
    print(json.dumps({"methods": aggregate, "paired": paired}, indent=2))


def figures(rows: list[dict]) -> None:
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    specs = [
        ("all", "log_psd_mse", "Whole-domain log-PSD MSE"),
        ("gap", "log_psd_mse", "Gap-interior log-PSD MSE"),
        ("gap", "coverage90", "Gap-interior truth inclusion"),
        ("gap", "signed_log_error", "Gap-interior signed log error"),
    ]
    for ax, (region, metric, title) in zip(axes.flat, specs, strict=True):
        for i, (variant, color) in enumerate(
            zip(VARIANTS, COLORS, strict=True)
        ):
            values = np.array(
                [
                    r["regions"][region][metric]
                    for r in rows
                    if r["variant"] == variant
                ]
            )
            # The same seed has the same horizontal offset for every method.
            ax.scatter(
                i + np.linspace(-0.12, 0.12, len(values)),
                values,
                s=30,
                color=color,
                alpha=0.7,
                marker="o",
                edgecolors="white",
                linewidths=0.4,
            )
            ax.plot(
                [i - 0.2, i + 0.2], [values.mean()] * 2, color="#222222", lw=2
            )
        ax.set(
            title=title,
            xticks=range(4),
            xticklabels=[
                "Original\nrectangular",
                "Matched\nrectangular",
                "Hann",
                "WDM",
            ],
        )
        ax.grid(axis="y", alpha=0.15)
        if metric == "coverage90":
            ax.axhline(0.9, color="#222222", ls="--", lw=1)
            ax.set(ylim=(-0.03, 1.03), ylabel="Fraction (nominal 0.90)")
        elif metric == "signed_log_error":
            ax.axhline(0, color="#222222", ls="--", lw=1)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle(
        "Central 10% gap: ten paired Gaussian records\nDots: independent records; black bars: record mean"
    )
    fig.savefig(ROOT / "repeated_comparison.png", dpi=180)
    fig.savefig(ROOT / "repeated_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True
    )
    for ax, variant, color in zip(axes.flat, VARIANTS, COLORS, strict=True):
        method = "wdm" if variant == "wdm" else "dw"
        d = np.load(
            ROOT / "fits" / variant / f"seed6101_{method}" / "surface.npz"
        )
        j = int(np.argmin(abs(d["freq"] - 0.25)))
        ax.plot(
            d["time"],
            np.exp(d["truth"][:, j]),
            color="#222222",
            ls=":",
            lw=2,
            label="Truth",
        )
        ax.plot(
            d["time"],
            np.exp(d["median"][:, j]),
            color=color,
            lw=1.8,
            label="Posterior median",
        )
        ax.fill_between(
            d["time"],
            np.exp(d["low"][:, j]),
            np.exp(d["high"][:, j]),
            color=color,
            alpha=0.17,
            label="90% interval",
        )
        mask = gap.masks()["contiguous"]
        edges = (np.flatnonzero(mask[1:] != mask[:-1]) + 1) / bench.N
        ax.axvspan(*edges, color="#999999", alpha=0.15)
        ax.set(
            title=LABELS[variant],
            xlabel="Rescaled time u",
            ylabel="One-sided PSD",
            xlim=(0.3, 0.7),
        )
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Seed 6101 at 0.25 Hz; gray band: raw missing samples")
    fig.savefig(ROOT / "seed6101_comparison.png", dpi=180)
    fig.savefig(ROOT / "seed6101_comparison.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "fit", "summarize"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    args = parser.parse_args()
    if args.action == "audit":
        audit()
    elif args.action == "fit":
        fit_study(
            args.seeds,
            args.variants,
            warmup=args.warmup,
            samples=args.samples,
            target_accept=args.target_accept,
            repair=args.repair,
        )
    else:
        summarize()


if __name__ == "__main__":
    main()
