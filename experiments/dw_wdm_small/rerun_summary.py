"""Independently score the frequency-reuse rerun and compare its provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from log_psplines import PSDResult

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/reuse_rerun"


def axis_weights(axis: np.ndarray) -> np.ndarray:
    spacing = np.diff(axis)
    return np.r_[
        spacing[0] / 2, (spacing[:-1] + spacing[1:]) / 2, spacing[-1] / 2
    ]


def main() -> None:
    rows = []
    checks = []
    fields = (
        "integrated_log_psd_squared_error",
        "log_psd_mse",
        "coverage90",
        "median_interval_width",
        "median_log_interval_width",
        "preprocess_seconds",
        "model_preparation_seconds",
        "inference_seconds",
    )
    for seed in range(6101, 6111):
        for method in ("dw", "wdm"):
            path = OUT / f"seed{seed}_{method}"
            row = json.loads((path / "metrics.json").read_text())
            original = json.loads(
                (
                    ROOT / f"results/seed{seed}_{method}/metrics.json"
                ).read_text()
            )
            for key in (
                "raw_sha256",
                "config",
                "model_hashes",
                "likelihood_observations",
                "real_component_count",
            ):
                assert row[key] == original[key], (seed, method, key)
            surface = np.load(path / "surface.npz")
            old_surface = np.load(
                ROOT / f"results/seed{seed}_{method}/surface.npz"
            )
            for key in ("time", "freq", "truth"):
                np.testing.assert_array_equal(surface[key], old_surface[key])
            area = np.outer(
                axis_weights(surface["time"]), axis_weights(surface["freq"])
            )
            squared_error = (surface["median"] - surface["truth"]) ** 2
            inclusion = (
                (surface["low"] <= surface["truth"])
                & (surface["truth"] <= surface["high"])
            ).astype(float)
            computed = {
                "integrated_log_psd_squared_error": float(
                    np.sum(area * squared_error)
                ),
                "log_psd_mse": float(
                    np.sum(area * squared_error) / area.sum()
                ),
                "coverage90": float(np.sum(area * inclusion) / area.sum()),
                "median_interval_width": float(
                    np.median(np.exp(surface["high"]) - np.exp(surface["low"]))
                ),
            }
            for key, value in computed.items():
                np.testing.assert_allclose(
                    row[key], value, rtol=1e-12, atol=1e-14
                )
            assert np.isfinite(surface["median"]).all()
            assert (np.exp(surface["low"]) > 0).all()
            result = PSDResult.from_netcdf(path / "posterior.nc")
            row["mean_retained_leapfrog_steps"] = float(
                result.sample_stats.n_steps.mean()
            )
            row["mean_acceptance"] = float(
                result.sample_stats.acceptance_rate.mean()
            )
            row["median_change_log_psd_rms"] = float(
                np.sqrt(
                    np.mean((surface["median"] - old_surface["median"]) ** 2)
                )
            )
            rows.append(row)
            checks.append(
                {
                    "seed": seed,
                    "method": method,
                    "provenance_and_metrics_pass": True,
                }
            )

    aggregate = {}
    for method in ("dw", "wdm"):
        selected = [r for r in rows if r["method"] == method]
        aggregate[method] = {
            key: {
                "mean": float(np.mean([r[key] for r in selected])),
                "sd": float(np.std([r[key] for r in selected], ddof=1)),
            }
            for key in fields
            + ("mean_retained_leapfrog_steps", "mean_acceptance")
        }
        aggregate[method]["sampler_passes"] = sum(
            r["diagnostics"]["pass"] for r in selected
        )
        aggregate[method]["flagged_seeds"] = [
            r["seed"] for r in selected if not r["diagnostics"]["pass"]
        ]
        aggregate[method]["observations"] = selected[0][
            "likelihood_observations"
        ]
        aggregate[method]["real_components"] = selected[0][
            "real_component_count"
        ]
        aggregate[method]["inference_total_seconds"] = sum(
            r["inference_seconds"] for r in selected
        )
        aggregate[method]["original_mean_inference_seconds"] = float(
            np.mean(
                [
                    json.loads(
                        (
                            ROOT
                            / f"results/seed{r['seed']}_{method}/metrics.json"
                        ).read_text()
                    )["inference_seconds"]
                    for r in selected
                ]
            )
        )
        aggregate[method]["median_change_rms_mean"] = float(
            np.mean([r["median_change_log_psd_rms"] for r in selected])
        )

    differences = np.array(
        [
            next(
                r["log_psd_mse"]
                for r in rows
                if r["seed"] == seed and r["method"] == "dw"
            )
            - next(
                r["log_psd_mse"]
                for r in rows
                if r["seed"] == seed and r["method"] == "wdm"
            )
            for seed in range(6101, 6111)
        ]
    )
    mean = differences.mean()
    radius = t.ppf(0.975, 9) * differences.std(ddof=1) / np.sqrt(10)
    aggregate["paired_error_difference_dw_minus_wdm"] = {
        "mean": float(mean),
        "exploratory_t_interval95": [
            float(mean - radius),
            float(mean + radius),
        ],
        "wdm_smaller_error_seeds": int((differences > 0).sum()),
    }
    aggregate["inference_ratio_dw_over_wdm"] = (
        aggregate["dw"]["inference_seconds"]["mean"]
        / aggregate["wdm"]["inference_seconds"]["mean"]
    )
    for name, content in (
        ("aggregate.json", aggregate),
        ("validation.json", checks),
    ):
        (OUT / name).write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
