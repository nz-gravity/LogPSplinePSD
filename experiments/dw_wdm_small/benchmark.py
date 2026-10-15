"""Small paired Gaussian PSD benchmark using existing preprocessing and NUTS.

Run from the repository root with .venv/bin/python. All model, prior and
sampler code comes from log_psplines; only unit adapters and study metrics
live here. PSDs are one-sided, in data-unit squared per Hz, with dt=1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import arviz_stats as azs
import jax
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wdm_transform import WDM
from wdm_transform import TimeSeries as WDMTimeSeries

from log_psplines import (
    LogPSpline,
    PowerConfig,
    PowerData,
    SplineBasis,
    TimeSeries,
    fit,
)
from log_psplines.example_datasets.ls2_data import LS2Data
from log_psplines.preprocessing.moving_periodogram import (
    scattered_moving_periodogram,
    tang_moving_periodogram,
)
from log_psplines.preprocessing.wdm import wdm_periodogram

N, NT, M, THIN, DT = 8192, 256, 32, 2, 1.0
jax.config.update("jax_enable_x64", True)
NF = N // NT
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def variance(u: np.ndarray) -> np.ndarray:
    """Known variance of amplitude-modulated white Gaussian noise."""
    return np.exp(0.7 * np.sin(2 * np.pi * u) + 0.2 * np.cos(4 * np.pi * u))


def simulate(seed: int) -> np.ndarray:
    # Reuse the existing Gaussian simulator at its white-noise special case.
    base = LS2Data(n_samples=N, fs=1 / DT, seed=seed, amplitude=0.0)
    return base.data[:, 0] * np.sqrt(variance(base.rescaled_time))


def master_model() -> LogPSpline:
    return LogPSpline(
        SplineBasis.from_grid(np.linspace(0, 0.5 / DT, 65), 2),
        time=SplineBasis.from_grid(np.linspace(0, 1, 65), 8),
    )


def prepare(x: np.ndarray, method: str) -> PowerData:
    if method == "dw":
        native = scattered_moving_periodogram(x, dt=DT, m=M, thin=THIN)
        return PowerData(
            native.power * (4 * np.pi * DT),
            native.counts,
            native.frequency,
            native.time - 1 / N,  # Convert one-based Tang centre to sample u.
            "one-sided PSD (data-unit squared/Hz)",
        )
    native = wdm_periodogram(TimeSeries(x), nt=NT)
    return PowerData(
        native.power * (2 * DT / N),
        native.counts,
        native.frequency,
        native.time,
        "one-sided PSD (data-unit squared/Hz)",
    )


def on_grid(
    model: LogPSpline, time: np.ndarray, freq: np.ndarray
) -> LogPSpline:
    """Change evaluation rows, preserving exactly the knots and penalties."""
    return LogPSpline(
        replace(
            model.frequency, grid=freq, basis=model.frequency.design_at(freq)
        ),
        time=replace(model.time, grid=time, basis=model.time.design_at(time)),
    )


def wdm_row(t: int, f: int) -> np.ndarray:
    """Normalized real analysis row, checked against the forward transform."""
    c = np.zeros((NT, NF + 1))
    c[t, f] = 1
    return np.sqrt(N) * np.asarray(WDM(c, dt=DT).to_time_series().data).ravel()


def power_correlation(rows: np.ndarray, sample_variance: np.ndarray) -> dict:
    """Exact Gaussian power correlation, including complex pseudo-covariance."""
    c = (rows * sample_variance) @ rows.conj().T
    p = (rows * sample_variance) @ rows.T
    power_cov = np.abs(c) ** 2 + np.abs(p) ** 2
    corr = power_cov / np.sqrt(
        np.outer(np.diag(power_cov), np.diag(power_cov))
    )
    np.fill_diagonal(corr, 0)
    result = {
        "probe_count": len(rows),
        "max_power_correlation": float(corr.max()),
    }
    if np.iscomplexobj(rows):
        result["max_nonproperness"] = float(
            np.max(np.abs(np.diag(p)) / np.diag(c).real)
        )
    return result


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    v = variance(np.arange(N) / N)
    raw = tang_moving_periodogram(np.zeros(N), m=M, thin=THIN)
    centres = np.rint(raw["u"] * N).astype(int) - 1
    exact_dw = np.convolve(v, np.ones(2 * M + 1) / (2 * M + 1), "valid")
    exact_dw = exact_dw[centres - M]
    dw_bias = exact_dw / variance(centres / N) - 1

    # Translation of even/odd prototype rows gives every WDM marginal mean.
    # This uses the actual installed Meyer transform, not a pointwise PSD rule.
    expected_wdm = np.zeros((NT, NF - 1))
    effective = []
    for f in range(1, NF):
        for parity in (0, 1):
            t0 = NT // 2 + parity
            row = wdm_row(t0, f)
            np.testing.assert_allclose(np.dot(row, row), 1, atol=1e-10)
            weights = row**2
            effective.append(1 / np.sum(weights**2))
            correlations = np.fft.ifft(
                np.fft.fft(v) * np.conj(np.fft.fft(weights))
            ).real
            times = np.arange(parity, NT, 2)
            expected_wdm[times, f - 1] = correlations[((times - t0) * NF) % N]
    rng = np.random.default_rng(200)
    x = rng.normal(size=N)
    wx = np.asarray(WDMTimeSeries(x, dt=DT).to_wdm(nt=NT).coeffs)[0] / np.sqrt(
        N
    )
    duality = []
    for t, f in [(1, 1), (20, 8), (127, 16), (128, 17), (254, 31)]:
        row = wdm_row(t, f)
        duality.append(abs(row @ x - wx[t, f]))
        np.testing.assert_allclose(row @ x, wx[t, f], atol=1e-10)
        np.testing.assert_allclose(
            row**2 @ v, expected_wdm[t, f - 1], atol=1e-10
        )
    wdm_bias = (
        expected_wdm[1:-1] / variance(np.arange(1, NT - 1)[:, None] / NT) - 1
    )

    # Local-neighbour probes: measured maxima are not full-transform maxima.
    dw_idx = np.concatenate([b * M + np.arange(8, 16) for b in (1, 2, 63, 64)])
    dw_rows = np.zeros((len(dw_idx), N), dtype=complex)
    for i, q in enumerate(dw_idx):
        dw_rows[i, centres[q] - M : centres[q] + M + 1] = np.exp(
            -1j * raw["omega"][q] * np.arange(2 * M + 1)
        ) / np.sqrt(2 * M + 1)
    wdm_probes = [
        (t, f) for t in (20, 21, 22, 127, 128, 129) for f in (8, 9, 10)
    ]
    wdm_rows = np.array([wdm_row(t, f) for t, f in wdm_probes])
    correlations = {
        method: {
            name: power_correlation(rows, vv)
            for name, vv in (("white", np.ones(N)), ("modulated", v))
        }
        for method, rows in (("dw", dw_rows), ("wdm", wdm_rows))
    }

    # Per-record means supply Monte Carlo standard errors without pretending
    # that selected overlapping ordinates are independent.
    record_means = {"dw": [], "wdm": []}
    for _ in range(128):
        white = rng.normal(size=N)
        for method in record_means:
            data = prepare(white, method)
            record_means[method].append(
                float(np.mean(data.power / data.counts))
            )
    white_checks = {}
    for method, values in record_means.items():
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        white_checks[method] = {
            "mean": float(np.mean(values)),
            "expected": 2 * DT,
            "mc_se": float(se),
            "z": float((np.mean(values) - 2 * DT) / se),
            "replicates": len(values),
        }
        assert abs(white_checks[method]["z"]) < 5
    np.savez_compressed(
        OUT / "expected_power.npz", dw=exact_dw, wdm=expected_wdm
    )
    write_json(
        OUT / "audit.json",
        {
            "white_noise_checks": white_checks,
            "wdm_inverse_forward_max_error": float(max(duality)),
            "expected_power_relative_bias": {
                "dw": {
                    "rms": float(np.sqrt(np.mean(dw_bias**2))),
                    "max": float(np.max(abs(dw_bias))),
                },
                "wdm": {
                    "rms": float(np.sqrt(np.mean(wdm_bias**2))),
                    "max": float(np.max(abs(wdm_bias))),
                },
            },
            "dependence_probes": correlations,
            "wdm_effective_window_samples": {
                "min": float(min(effective)),
                "max": float(max(effective)),
            },
            "frequency_spacing": {
                "dw": 1 / ((2 * M + 1) * DT),
                "wdm": 1 / (2 * NF * DT),
            },
            "seconds": perf_counter() - started,
        },
    )
    print(
        json.dumps(json.loads((OUT / "audit.json").read_text()), indent=2),
        flush=True,
    )


def area_mean(values: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
    # Boolean NumPy addition is logical OR; trapezoidal integration must
    # cast inclusion indicators before adding adjacent values.
    values = np.asarray(values, dtype=float)
    return float(
        np.trapezoid(np.trapezoid(values, f, axis=-1), u)
        / ((u[-1] - u[0]) * (f[-1] - f[0]))
    )


def infer(
    seed: int,
    method: str,
    warmup: int,
    samples: int,
    target_accept: float,
    *,
    preprocess: Callable[[np.ndarray, str], PowerData] | None = None,
) -> dict:
    path = OUT / f"seed{seed}_{method}"
    path.mkdir(parents=True, exist_ok=True)
    x = simulate(seed)
    started = perf_counter()
    data = (prepare if preprocess is None else preprocess)(x, method)
    transformed = perf_counter()
    master = master_model()
    model = (
        on_grid(master, data.time, data.frequency) if data.is_grid else master
    )
    config = PowerConfig(
        n_interior_knots_time=8,
        n_interior_knots_freq=2,
        n_warmup=warmup,
        n_samples=samples,
        num_chains=2,
        centered=False,
        dense_mass=True,
        target_accept_prob=target_accept,
        max_tree_depth=10,
        seed=seed + 10000,
        progress_bar=False,
        spectrum_draws=16,
    )
    result = fit(data, config, model=model)
    completed = perf_counter()
    weights = result.posterior.weights.values
    u, f = np.linspace(0.05, 0.95, 61), np.linspace(0.025, 0.475, 57) / DT
    logs = np.einsum(
        "ti,cdij,fj->cdtf",
        master.time.design_at(u),
        weights,
        master.frequency.design_at(f),
        optimize=True,
    ).reshape(-1, len(u), len(f))
    low, med, high = np.quantile(logs, [0.05, 0.5, 0.95], axis=0)
    truth = np.broadcast_to(np.log(2 * DT * variance(u))[:, None], med.shape)
    np.testing.assert_array_equal(model.time.knots, master.time.knots)
    np.testing.assert_array_equal(
        model.frequency.penalty, master.frequency.penalty
    )
    assert np.isfinite(logs).all()
    assert np.isfinite(np.exp(high)).all() and np.all(np.exp(low) > 0)

    # Compute chain diagnostics in coefficient and functional space, without
    # including redundant deterministic likelihood/weight sites.
    param_summary = azs.summary(
        result.posterior[["s", "sigma_time", "sigma_freq"]], round_to="none"
    )
    small_logs = logs.reshape(2, samples, len(u), len(f))[:, :, ::10, ::14]
    functional = xr.Dataset(
        {"log_psd": (("chain", "draw", "t", "f"), small_logs)}
    )
    functional_summary = azs.summary(functional, round_to="none")
    summary = [param_summary, functional_summary]
    diagnostics = {
        "divergences": int(result.sample_stats.diverging.sum()),
        "depth_hits": int(
            (result.sample_stats.n_steps >= 2**config.max_tree_depth - 1).sum()
        ),
        "rhat_max": float(max(s.r_hat.max() for s in summary)),
        "ess_bulk_min": float(min(s.ess_bulk.min() for s in summary)),
        "ess_tail_min": float(min(s.ess_tail.min() for s in summary)),
    }
    diagnostics["pass"] = (
        diagnostics["divergences"] == 0
        and diagnostics["depth_hits"] == 0
        and diagnostics["rhat_max"] <= 1.01
        and diagnostics["ess_bulk_min"] >= 200
        and diagnostics["ess_tail_min"] >= 200
    )
    mse = area_mean((med - truth) ** 2, u, f)
    # Existing repository whitening diagnostics are placeholders. Use the
    # transform's signed coefficients for a small descriptive check only.
    if method == "dw":
        bt, bf = (
            master.time.design_at(data.time),
            master.frequency.design_at(data.frequency),
        )
        fitted_logs = np.einsum(
            "pi,ij,pj->p", bt, np.median(weights, axis=(0, 1)), bf
        )
        # Equal to squared signed-coefficient whitening, including when an
        # experimental taper/subselection supplies the already normalized data.
        squared = data.power / data.counts / np.exp(fitted_logs)
        whiten = {
            "mean_squared_magnitude": float(np.mean(squared)),
            "expected": 1.0,
        }
    else:
        w = np.asarray(WDMTimeSeries(x, dt=DT).to_wdm(nt=NT).coeffs)[
            0, 1:-1, 1:-1
        ]
        fitted_logs = model(np.median(weights, axis=(0, 1)))
        z = w * np.sqrt(2 * DT / N) / np.sqrt(np.exp(fitted_logs))
        whiten = {
            "mean_square": float(np.mean(z**2)),
            "expected": 1.0,
            "mean": float(np.mean(z)),
            "excess_kurtosis": float(
                np.mean((z - z.mean()) ** 4) / np.var(z) ** 2 - 3
            ),
        }
    metrics = {
        "seed": seed,
        "method": method,
        "raw_sha256": digest(x),
        "log_psd_mse": mse,
        "log_psd_rmse": np.sqrt(mse),
        "integrated_log_psd_squared_error": mse
        * (u[-1] - u[0])
        * (f[-1] - f[0]),
        "coverage90": area_mean((low <= truth) & (truth <= high), u, f),
        "median_interval_width": float(np.median(np.exp(high) - np.exp(low))),
        "median_log_interval_width": float(np.median(high - low)),
        "preprocess_seconds": transformed - started,
        "model_preparation_seconds": result.metadata["preparation_seconds"],
        "inference_seconds": result.metadata["inference_seconds"],
        "fit_total_seconds": completed - transformed,
        "likelihood_observations": int(np.count_nonzero(data.counts)),
        "real_component_count": float(data.counts.sum()),
        "model_hashes": {
            "time_knots": digest(master.time.knots),
            "frequency_knots": digest(master.frequency.knots),
            "time_penalty": digest(np.asarray(master.time.penalty)),
            "frequency_penalty": digest(np.asarray(master.frequency.penalty)),
        },
        "diagnostics": diagnostics,
        "whitening": whiten,
        "config": asdict(config),
    }
    result.to_netcdf(path / "posterior.nc")
    np.savez_compressed(
        path / "surface.npz",
        time=u,
        freq=f,
        truth=truth,
        low=low,
        median=med,
        high=high,
    )
    write_json(path / "metrics.json", metrics)
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k not in ("config", "raw_sha256")
            }
        ),
        flush=True,
    )
    return metrics


def summarize() -> None:
    # Recompute coverage from archived quantiles; inference need not be rerun
    # after a metric-only correction. Includes the retained failed pilot.
    for path in OUT.rglob("metrics.json"):
        surface = path.parent / "surface.npz"
        if not surface.exists():
            continue
        d = np.load(surface)
        metrics = json.loads(path.read_text())
        metrics["phase"] = (
            "initial_debug"
            if "pilot_initial" in path.parts
            else "paired_check"
            if "repairs" in path.parts
            else "debug"
            if metrics["seed"] == 6100
            else "primary"
        )
        metrics["integrated_log_psd_squared_error"] = (
            metrics["log_psd_mse"]
            * (d["time"][-1] - d["time"][0])
            * (d["freq"][-1] - d["freq"][0])
        )
        metrics["coverage90"] = area_mean(
            (d["low"] <= d["truth"]) & (d["truth"] <= d["high"]),
            d["time"],
            d["freq"],
        )
        write_json(path, metrics)
    rows = [
        json.loads(p.read_text())
        for p in sorted(OUT.glob("seed*/metrics.json"))
    ]
    fields = [
        "phase",
        "seed",
        "method",
        "log_psd_mse",
        "log_psd_rmse",
        "integrated_log_psd_squared_error",
        "coverage90",
        "median_interval_width",
        "median_log_interval_width",
        "preprocess_seconds",
        "model_preparation_seconds",
        "inference_seconds",
        "fit_total_seconds",
        "likelihood_observations",
        "real_component_count",
    ]
    with (OUT / "metrics.csv").open("w") as fp:
        writer = csv.DictWriter(
            fp,
            fields
            + [
                "sampler_pass",
                "rhat_max",
                "ess_bulk_min",
                "ess_tail_min",
                "divergences",
                "depth_hits",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{k: row[k] for k in fields},
                    **{
                        k: row["diagnostics"][k]
                        for k in (
                            "rhat_max",
                            "ess_bulk_min",
                            "ess_tail_min",
                            "divergences",
                            "depth_hits",
                        )
                    },
                    "sampler_pass": row["diagnostics"]["pass"],
                }
            )
    write_json(OUT / "summary.json", rows)


def figure(seed: int) -> None:
    data = {
        m: np.load(OUT / f"seed{seed}_{m}" / "surface.npz")
        for m in ("dw", "wdm")
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    first = data["dw"]
    for ax, title, values in zip(
        axes[0],
        (
            "True PSD",
            "Dynamic Whittle: posterior median",
            "WDM: posterior median",
        ),
        (first["truth"], first["median"], data["wdm"]["median"]),
        strict=True,
    ):
        im = ax.pcolormesh(
            first["time"],
            first["freq"],
            np.exp(values).T,
            cmap="cividis",
            vmin=0.5,
            vmax=4,
            shading="auto",
        )
        ax.set(title=title, xlabel="Rescaled time u", ylabel="Frequency (Hz)")
    fig.colorbar(
        im,
        ax=axes[0].tolist(),
        label="One-sided PSD (data-unit squared/Hz)",
        shrink=0.85,
    )
    for i, u0 in enumerate((0.2, 0.5, 0.8)):
        ax = axes[1, i]
        j = int(np.argmin(abs(first["time"] - u0)))
        ax.plot(
            first["freq"],
            np.exp(first["truth"][j]),
            color="#222222",
            ls=":",
            lw=2,
            label="Truth",
        )
        for method, color, label, ls in (
            ("dw", "#2866A2", "Dynamic Whittle", "-"),
            ("wdm", "#B46A1D", "WDM", "--"),
        ):
            d = data[method]
            ax.fill_between(
                d["freq"],
                np.exp(d["low"][j]),
                np.exp(d["high"][j]),
                color=color,
                alpha=0.18,
            )
            ax.plot(
                d["freq"],
                np.exp(d["median"][j]),
                color=color,
                ls=ls,
                label=label,
            )
        ax.set(
            title=f"u = {first['time'][j]:.2f}; pointwise 90% intervals",
            xlabel="Frequency (Hz)",
            ylabel="One-sided PSD",
            ylim=(0, 5),
        )
        ax.grid(alpha=0.15)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        f"Smooth Gaussian amplitude modulation; N={N}; seed={seed}",
        fontsize=13,
    )
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("audit", "fit", "summarize", "figure")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[6100])
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    if args.action == "audit":
        audit()
        write_json(
            OUT / "environment.json",
            {
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "platform": platform.platform(),
                "jax_devices": str(jax.devices()),
                "versions": {
                    p: version(p)
                    for p in (
                        "numpy",
                        "jax",
                        "numpyro",
                        "wdm-transform",
                        "LogPSplinePSD",
                    )
                },
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "study": {
                    "N": N,
                    "nt": NT,
                    "nf": NF,
                    "m": M,
                    "thin": THIN,
                    "dt": DT,
                },
            },
        )
    elif args.action == "fit":
        for seed in args.seeds:
            for method in ("dw", "wdm") if seed % 2 == 0 else ("wdm", "dw"):
                infer(
                    seed, method, args.warmup, args.samples, args.target_accept
                )
        summarize()
    elif args.action == "figure":
        figure(args.seeds[0])
    else:
        summarize()


if __name__ == "__main__":
    main()
