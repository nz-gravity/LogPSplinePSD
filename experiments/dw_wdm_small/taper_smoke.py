"""Matched Hann/boxcar pilot with whole-cycle gap gating and existing NUTS."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache, partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small.gap_smoke import dw_keep, masks

from log_psplines import PowerData

ROOT = Path(__file__).resolve().parent / "taper"
RUNGS = np.arange(2, bench.M - 2, 3)  # Zero-based: Fourier rungs 3,6,...,30.


@lru_cache(maxsize=1)
def schedule() -> dict:
    return bench.tang_moving_periodogram(
        np.zeros(bench.N), m=bench.M, thin=bench.THIN
    )


def weights(taper: str) -> np.ndarray:
    if taper not in ("boxcar", "hann"):
        raise ValueError(f"Unknown taper: {taper}")
    return (
        np.ones(2 * bench.M + 1)
        if taper == "boxcar"
        else np.hanning(2 * bench.M + 1)
    )


def retained(observed: np.ndarray, *, sparse: bool = True) -> np.ndarray:
    keep = dw_keep(observed, whole_cycles=True)
    if sparse:
        keep &= np.isin(np.arange(len(keep)) % bench.M, RUNGS)
    return keep


def prepare(x: np.ndarray, taper: str, observed: np.ndarray) -> PowerData:
    """Normalize by full taper energy; never use a window crossing an outage."""
    keep = retained(observed)
    clean = np.where(observed, x, 0.0)
    if taper == "boxcar":
        original = bench.prepare(clean, "dw")
        return PowerData(
            original.power[keep],
            original.counts[keep],
            original.frequency[keep],
            original.time[keep],
            original.units,
        )
    raw = schedule()
    centres = np.rint(raw["u"] * bench.N).astype(int) - 1
    windows = np.lib.stride_tricks.sliding_window_view(clean, 2 * bench.M + 1)
    windows = windows[centres[keep] - bench.M]
    h = weights(taper)
    phase = np.exp(-1j * raw["omega"][keep, None] * np.arange(len(h)))
    coeff = np.sum(windows * h * phase, axis=1) / np.sqrt(2 * np.pi * (h @ h))
    return PowerData(
        8 * np.pi * bench.DT * np.abs(coeff) ** 2,
        np.full(int(keep.sum()), 2.0),
        raw["omega"][keep] / (2 * np.pi * bench.DT),
        centres[keep] / bench.N,
        "one-sided PSD (data-unit squared/Hz)",
    )


def correlation_audit(taper: str, *, sparse: bool) -> dict:
    raw = schedule()
    centres = np.rint(raw["u"] * bench.N).astype(int) - 1
    indices = np.concatenate(
        [k * bench.M + np.arange(bench.M) for k in (1, 2, 63, 64)]
    )
    if sparse:
        indices = indices[np.isin(indices % bench.M, RUNGS)]
    h = weights(taper)
    rows = np.zeros((len(indices), bench.N), dtype=complex)
    for i, q in enumerate(indices):
        rows[i, centres[q] - bench.M : centres[q] + bench.M + 1] = (
            h
            * np.exp(-1j * raw["omega"][q] * np.arange(len(h)))
            / np.sqrt(h @ h)
        )
    return {
        name: bench.power_correlation(rows, v)
        for name, v in (
            ("white", np.ones(bench.N)),
            ("modulated", bench.variance(np.arange(bench.N) / bench.N)),
        )
    }


def audit() -> None:
    ROOT.mkdir(exist_ok=True)
    result = {}
    v = bench.variance(np.arange(bench.N) / bench.N)
    raw = schedule()
    centres = np.rint(raw["u"] * bench.N).astype(int) - 1
    rng = np.random.default_rng(203)
    record_means = {t: [] for t in ("boxcar", "hann")}
    full = np.ones(bench.N, dtype=bool)
    for _ in range(128):
        x = rng.normal(size=bench.N)
        for taper in record_means:
            d = prepare(x, taper, full)
            record_means[taper].append(float(np.mean(d.power / d.counts)))
    for taper in record_means:
        h = weights(taper)
        expected = np.convolve(v, h**2 / (h @ h), "valid")[centres - bench.M]
        bias = expected / bench.variance(centres / bench.N) - 1
        values = record_means[taper]
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        z = (np.mean(values) - 2 * bench.DT) / se
        assert abs(z) < 5
        result[taper] = {
            "taper_energy": float(h @ h),
            "effective_time_samples": float((h @ h) ** 2 / np.sum(h**4)),
            "enbw_fourier_bins": float(len(h) * (h @ h) / h.sum() ** 2),
            "max_pointwise_mean_bias": float(np.max(abs(bias))),
            "all_rungs_correlations": correlation_audit(taper, sparse=False),
            "selected_correlations": correlation_audit(taper, sparse=True),
            "white_normalization": {
                "mean": float(np.mean(values)),
                "expected": 2 * bench.DT,
                "mc_se": float(se),
                "z": float(z),
                "records": 128,
            },
        }
        checked = result[taper]["selected_correlations"]["modulated"]
        assert checked["max_power_correlation"] < 0.01
        assert checked["max_nonproperness"] < 0.02
    result["selection"] = {
        "rungs": (RUNGS + 1).tolist(),
        "frequency_hz": (
            (RUNGS + 1) / ((2 * bench.M + 1) * bench.DT)
        ).tolist(),
        "gap_gating": "discard the full original Tang cycle if any window touches a gap",
        "observations": {
            name: int(retained(observed).sum())
            for name, observed in {"no_gap": full, **masks()}.items()
        },
        "correlations_are_probe_maxima": True,
    }
    bench.write_json(ROOT / "audit.json", result)
    print(json.dumps(result, indent=2))


def fit_pilot() -> None:
    evidence = json.loads((ROOT / "audit.json").read_text())
    assert (
        evidence["hann"]["selected_correlations"]["modulated"][
            "max_power_correlation"
        ]
        < 0.01
    )
    for case in ("no_gap", "contiguous"):
        observed = (
            np.ones(bench.N, dtype=bool) if case == "no_gap" else masks()[case]
        )
        for taper in ("boxcar", "hann"):
            bench.OUT = ROOT / f"{case}_{taper}"
            metrics = bench.infer(
                6101,
                "dw",
                600,
                1000,
                0.95,
                preprocess=lambda x, method, adapter=partial(prepare, taper=taper, observed=observed): (
                    adapter(x)
                ),
            )
            metrics.update(
                {
                    "taper": taper,
                    "case": case,
                    "frequency_rungs": (RUNGS + 1).tolist(),
                    "gap_gating": evidence["selection"]["gap_gating"],
                }
            )
            bench.write_json(bench.OUT / "seed6101_dw/metrics.json", metrics)


def summarize() -> None:
    observed = masks()["contiguous"]
    edges = np.flatnonzero(observed[1:] != observed[:-1]) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    summary = {}
    for ax, taper, color in zip(
        axes, ("boxcar", "hann"), ("#2866A2", "#B46A1D"), strict=True
    ):
        surfaces = {
            case: np.load(ROOT / f"{case}_{taper}/seed6101_dw/surface.npz")
            for case in ("no_gap", "contiguous")
        }
        d = surfaces["contiguous"]
        u, f = d["time"], d["freq"]
        j = int(np.argmin(abs(f - 0.25)))
        common_band = (f >= 0.05) & (f <= 0.45)
        regions = {
            "whole_scoring_domain": np.ones_like(u, dtype=bool),
            "gap": ~observed[(u * bench.N).astype(int)],
        }
        out = {}
        for case, s in surfaces.items():
            metrics = json.loads(
                (ROOT / f"{case}_{taper}/seed6101_dw/metrics.json").read_text()
            )
            out[case] = {
                "inference_seconds": metrics["inference_seconds"],
                "preprocess_seconds": metrics["preprocess_seconds"],
                "observations": metrics["likelihood_observations"],
                "diagnostics": metrics["diagnostics"],
                "regions": {},
            }
            for name, mask in regions.items():
                area_mask = np.broadcast_to(
                    mask[:, None], s["median"][:, common_band].shape
                )
                denominator = bench.area_mean(area_mask, u, f[common_band])
                delta = (
                    s["median"][:, common_band] - s["truth"][:, common_band]
                )
                included = (
                    s["low"][:, common_band] <= s["truth"][:, common_band]
                ) & (s["truth"][:, common_band] <= s["high"][:, common_band])
                width = np.exp(s["high"][:, common_band]) - np.exp(
                    s["low"][:, common_band]
                )
                out[case]["regions"][name] = {
                    "log_psd_mse": bench.area_mean(
                        delta**2 * area_mask, u, f[common_band]
                    )
                    / denominator,
                    "mean_relative_psd_error": bench.area_mean(
                        np.expm1(delta) * area_mask, u, f[common_band]
                    )
                    / denominator,
                    "truth_in_90pct_interval": bench.area_mean(
                        included.astype(float) * area_mask, u, f[common_band]
                    )
                    / denominator,
                    "median_interval_width": float(
                        np.median(width[area_mask])
                    ),
                }
        summary[taper] = out
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
            np.exp(surfaces["no_gap"]["median"][:, j]),
            color=color,
            ls="--",
            label="No-gap median",
        )
        ax.plot(u, np.exp(d["median"][:, j]), color=color, label="Gap median")
        ax.fill_between(
            u,
            np.exp(d["low"][:, j]),
            np.exp(d["high"][:, j]),
            color=color,
            alpha=0.18,
            label="Gap: 90% interval",
        )
        ax.axvspan(
            edges[0] / bench.N, edges[1] / bench.N, color="#999999", alpha=0.15
        )
        ax.set(
            title=f"{taper.capitalize()}: same selected frequency rungs",
            xlabel="Rescaled time u",
            ylabel="One-sided PSD",
            xlim=(0.3, 0.7),
            ylim=(0, 4.5),
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Taper control; seed 6101; f=0.25 Hz; identical model, prior and sampler"
    )
    fig.savefig(ROOT / "comparison.png", dpi=180)
    fig.savefig(ROOT / "comparison.pdf")
    plt.close(fig)
    bench.write_json(
        ROOT / "summary.json",
        {"common_scoring_frequency_band": [0.05, 0.45], "results": summary},
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "fit", "summarize"))
    args = parser.parse_args()
    {"audit": audit, "fit": fit_pilot, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()
