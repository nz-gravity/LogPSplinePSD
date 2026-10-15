"""Exact finite-window mean audit on the existing coloured LS2 MA(1) process.

This diagnoses leakage; it does not fit or validate a coloured-noise posterior.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import gap_smoke as gap
from experiments.dw_wdm_small import taper_smoke as taper
from experiments.dw_wdm_small import taper_study as study
from matplotlib.colors import LinearSegmentedColormap

from log_psplines.example_datasets.ls2_data import LS2Data


def expected_dw(
    h: np.ndarray, u: np.ndarray, f: np.ndarray, b: np.ndarray
) -> np.ndarray:
    centres = np.rint(u * bench.N).astype(int)
    window_b = np.lib.stride_tricks.sliding_window_view(b, len(h))[
        centres - bench.M
    ]
    # Cov(X_j, X_{j-1})=b(j/N), Var(X_j)=1+b(j/N)^2.
    diagonal = (1 + window_b**2) @ (h**2) / (h @ h)
    adjacent = window_b[:, 1:] @ (h[1:] * h[:-1]) / (h @ h)
    return (
        2
        * bench.DT
        * (diagonal + 2 * np.cos(2 * np.pi * f * bench.DT) * adjacent)
    )


def expected_wdm(
    b: np.ndarray, observed: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    diagonal = np.fft.fft(observed * (1 + b**2))
    off_diagonal = observed * np.roll(observed, 1) * b
    off_diagonal[0] = 0  # Innovations do not wrap around the finite record.
    adjacent = np.fft.fft(off_diagonal)
    expected = np.zeros((bench.NT, bench.NF - 1))
    for frequency in range(1, bench.NF):
        for parity in (0, 1):
            t0 = bench.NT // 2 + parity
            row = bench.wdm_row(t0, frequency)
            c0 = np.fft.ifft(diagonal * np.conj(np.fft.fft(row**2))).real
            c1 = np.fft.ifft(
                adjacent * np.conj(np.fft.fft(row * np.roll(row, 1)))
            ).real
            times = np.arange(parity, bench.NT, 2)
            shifts = ((times - t0) * bench.NF) % bench.N
            expected[times, frequency - 1] = c0[shifts] + 2 * c1[shifts]
    kept = gap.wdm_keep(rho)
    return np.divide(
        2 * bench.DT * expected[1:-1], rho, out=np.zeros_like(rho), where=kept
    )


def main() -> None:
    saved = np.load(study.ROOT / "mask.npz")
    observed, rho = saved["observed"], saved["rho"]
    simulator = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=8200)
    b = simulator.coefficient(simulator.rescaled_time)
    records = {}
    exact = {}
    summary = {}
    for variant in study.VARIANTS:
        method = "wdm" if variant == "wdm" else "dw"
        data = study.data_adapter(
            simulator.data[:, 0],
            method,
            variant=variant,
            observed=observed,
            rho=rho,
        )
        if data.is_grid:
            predicted = expected_wdm(b, observed, rho)
            uu, ff = np.meshgrid(data.time, data.frequency, indexing="ij")
        else:
            h = taper.weights("hann" if variant == "hann_sparse" else "boxcar")
            predicted = expected_dw(h, data.time, data.frequency, b)
            uu, ff = data.time, data.frequency
        local_b = simulator.coefficient(uu)
        truth = (
            2
            * bench.DT
            * (
                1
                + local_b**2
                + 2 * local_b * np.cos(2 * np.pi * ff * bench.DT)
            )
        )
        valid = (
            (data.counts > 0)
            & (ff >= 0.05)
            & (ff <= 0.45)
            & (uu >= 0.05)
            & (uu <= 0.95)
        )
        delta = np.log(predicted[valid]) - np.log(truth[valid])
        summary[variant] = {
            "sites": int(valid.sum()),
            "rms_log_expected_power_discrepancy": float(
                np.sqrt(np.mean(delta**2))
            ),
            "mean_absolute_relative_expected_power_discrepancy": float(
                np.mean(abs(np.expm1(delta)))
            ),
            "max_absolute_relative_expected_power_discrepancy": float(
                np.max(abs(np.expm1(delta)))
            ),
        }
        records[variant] = []
        exact[variant] = predicted
        # Check exact Gaussian row projection at independent explicit rows.
        if variant == "wdm":
            for t, f in [(20, 3), (80, 16), (200, 29)]:
                if not gap.wdm_keep(rho)[t - 1, f - 1]:
                    continue
                row = bench.wdm_row(t, f) * observed
                reference = (
                    2
                    * bench.DT
                    * (
                        row**2 @ (1 + b**2)
                        + 2 * np.sum(row[1:] * row[:-1] * b[1:])
                    )
                    / rho[t - 1, f - 1]
                )
                np.testing.assert_allclose(
                    predicted[t - 1, f - 1], reference, atol=2e-12
                )
        np.savez_compressed(
            study.ROOT / f"coloured_{variant}.npz",
            time=uu,
            freq=ff,
            truth=truth,
            expected=predicted,
            valid=valid,
        )
    # Check the expected variance using independent raw records. Statistics are
    # record means, so overlapping coefficients do not inflate sample size.
    for seed in range(8200, 8328):
        x = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=seed).data[:, 0]
        for variant in study.VARIANTS:
            method = "wdm" if variant == "wdm" else "dw"
            data = study.data_adapter(
                x, method, variant=variant, observed=observed, rho=rho
            )
            selected = data.counts > 0
            records[variant].append(
                float(
                    np.mean(
                        data.power[selected]
                        / data.counts[selected]
                        / exact[variant][selected]
                    )
                )
            )
    for variant, values in records.items():
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        z = (np.mean(values) - 1) / se
        assert abs(z) < 5
        summary[variant]["variance_mc_check"] = {
            "record_mean": float(np.mean(values)),
            "expected": 1.0,
            "mc_se": float(se),
            "z": float(z),
            "records": len(values),
        }
    bench.write_json(
        study.ROOT / "coloured_audit.json",
        {
            "process": "existing LS2 MA(1), amplitude=1.1, offset=1.5, modulation=4",
            "scope": "expected powers only; no coloured posterior inference",
            "frequency_band_hz": [0.05, 0.45],
            "time_domain": [0.05, 0.95],
            "results": summary,
        },
    )
    cmap = LinearSegmentedColormap.from_list(
        "signed_bias", ["#2866A2", "#FFFFFF", "#B46A1D"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, variant, title in zip(
        axes,
        ("boxcar_sparse", "hann_sparse"),
        ("Rectangular (matched sparse)", "Hann (same sites)"),
        strict=True,
    ):
        d = np.load(study.ROOT / f"coloured_{variant}.npz")
        v = d["valid"]
        im = ax.scatter(
            d["time"][v],
            d["freq"][v],
            c=100 * (d["expected"][v] / d["truth"][v] - 1),
            cmap=cmap,
            vmin=-5,
            vmax=5,
            s=12,
            marker="s",
            linewidths=0,
        )
        ax.set(title=title, xlabel="Rescaled time u", ylabel="Frequency (Hz)")
        edges = (np.flatnonzero(observed[1:] != observed[:-1]) + 1) / bench.N
        ax.axvspan(*edges, color="#999999", alpha=0.15, zorder=-1)
    fig.colorbar(
        im,
        ax=axes,
        label="Expected power / pointwise PSD − 1 (%)",
        shrink=0.9,
        extend="both",
    )
    fig.suptitle(
        "Coloured LS2: exact finite-window mean discrepancy at retained sites"
    )
    fig.savefig(study.ROOT / "coloured_expected_power.png", dpi=180)
    fig.savefig(study.ROOT / "coloured_expected_power.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
