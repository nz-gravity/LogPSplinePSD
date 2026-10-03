"""Reproducible stages 0--3; no selected schedule or adaptive refit.

Run each inference target in an isolated process to bound compilation memory
and measure process peak RSS. Bulk results live in ignored runs/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path
from time import perf_counter

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

from log_psplines import PowerConfig, fit
from log_psplines.basis import SplineBasis
from log_psplines.diagnostics.sampling import sampling_diagnostics
from log_psplines.diagnostics.stationarity import (
    horizon_summary,
    stationarity_loss,
    whittle_loss,
)
from log_psplines.example_datasets.ls2_data import LS2Data
from log_psplines.models.spectrum import LogPSpline
from log_psplines.preprocessing.moving_periodogram import (
    scattered_moving_periodogram,
    tang_moving_periodogram,
)
from log_psplines.results import PSDResult

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "configs/smoke.toml"
DEFAULT_OUTPUT = ROOT / "runs/dynamic-whittle-stages-0-3"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
        + "\n"
    )


def envelope(u: np.ndarray, family: str) -> np.ndarray:
    if family == "slow":
        logvar = 0.8 * np.sin(2 * np.pi * u)
    elif family == "mixed":
        slow = (
            0.7
            * np.exp(-(((u - 0.38) / 0.13) ** 4))
            * np.sin(2 * np.pi * (u - 0.23) / 0.4)
        )
        fast = (
            1.2
            * np.exp(-(((u - 0.68) / 0.08) ** 4))
            * np.sin(2 * np.pi * (u - 0.60) / 0.075)
        )
        logvar = slow + fast
    elif family == "abrupt":
        logvar = np.where(u < 0.5, 0.0, np.log(4.0))
    else:
        logvar = np.zeros_like(u)
    return np.exp(logvar / 2)


def truth(
    family: str, u: np.ndarray, freq: np.ndarray, n: int, dt: float = 1.0
) -> np.ndarray:
    # u is the transform's one-based time coordinate; generator uses k/n.
    sample_u = np.asarray(u) - 1 / n
    omega = 2 * np.pi * np.asarray(freq) * dt
    if family == "ls2":
        ls2 = LS2Data(n_samples=2, fs=1 / dt, seed=0)
        return ls2.get_true_psd(time_grid=sample_u, freq_grid=freq) / (
            2 * np.pi
        )
    if family == "resonance":
        f0 = 0.2 + 0.05 * np.sin(4 * np.pi * sample_u)
        denominator = (
            abs(
                1
                - 2
                * 0.96
                * np.cos(2 * np.pi * f0[:, None])
                * np.exp(-1j * omega)
                + 0.96**2 * np.exp(-2j * omega)
            )
            ** 2
        )
        return 1 / (2 * np.pi * denominator)
    stationary = (
        np.ones_like(omega)
        if family == "white"
        else 1 / abs(1 - 0.7 * np.exp(-1j * omega)) ** 2
    )
    return (
        envelope(sample_u, family)[:, None] ** 2
        * stationary[None, :]
        / (2 * np.pi)
    )


def simulate(family: str, n: int, seed: int, dt: float) -> np.ndarray:
    if family == "ls2":
        return LS2Data(n_samples=n, fs=1 / dt, seed=seed).data[:, 0]
    rng = np.random.default_rng(seed)
    if family == "white":
        return rng.normal(size=n)
    if family == "resonance":
        values = np.zeros(n + 514)
        innovations = rng.normal(size=n + 514)
        for k in range(2, n + 514):
            u = max(0, k - 514) / n
            f0 = 0.2 + 0.05 * np.sin(4 * np.pi * u)
            values[k] = (
                2 * 0.96 * np.cos(2 * np.pi * f0) * values[k - 1]
                - 0.96**2 * values[k - 2]
                + innovations[k]
            )
        return values[514:]
    values = np.empty(n)
    previous = rng.normal(scale=np.sqrt(1 / (1 - 0.7**2)))
    for k, innovation in enumerate(rng.normal(size=n)):
        previous = 0.7 * previous + innovation
        values[k] = previous
    return values * envelope(np.arange(n) / n, family)


def build_model(
    cfg: dict, calibration: bool = False, larger: bool = False
) -> LogPSpline:
    prefix = (
        "calibration_" if calibration else "sensitivity_" if larger else ""
    )
    return LogPSpline(
        SplineBasis.from_grid(
            np.linspace(0.0, 0.5 / cfg["dt"], cfg["frequency_nodes"]),
            cfg[prefix + "frequency_knots"],
        ),
        time=SplineBasis.from_grid(
            np.linspace(0.0, 1.0, cfg["time_nodes"]),
            cfg[prefix + "time_knots"],
        ),
    )


def trap_weights(grid: np.ndarray) -> np.ndarray:
    weights = np.r_[
        np.diff(grid)[0] / 2, (grid[2:] - grid[:-2]) / 2, np.diff(grid)[-1] / 2
    ]
    return weights / weights.sum()


def metrics(
    result: PSDResult, target: np.ndarray, cfg: dict, data, m: int, n: int
) -> dict:
    mean = result.spectrum_summary["mean"].values[..., 0, 0].real
    quant = result.quantiles(kind="real").values[..., 0, 0]
    freq = result.frequency
    times = result.time
    band = (freq >= cfg["frequency_band"][0]) & (
        freq <= cfg["frequency_band"][1]
    )
    loss = whittle_loss(np.log(target), np.log(mean))
    output = {}
    bounds = [
        (
            (half + 1) / n,
            (
                cfg["thin"]
                * half
                * ((n - 2 * half) // (cfg["thin"] * half) - 1)
                + 2 * half
            )
            / n,
        )
        for half in cfg["windows"]
    ]
    masks = {
        "native": (times >= data.time.min()) & (times <= data.time.max()),
        "matched": (times >= max(low for low, _ in bounds))
        & (times <= min(high for _, high in bounds)),
        "quiet": ((times < 0.2) | (times > 0.85)),
        "slow": (times >= 0.23) & (times < 0.5),
        "fast": (times >= 0.59) & (times <= 0.77),
        "boundary": (times < 0.05) | (times > 0.95),
        "change": abs(times - 0.5) < 0.05,
    }
    for label, mask in masks.items():
        mask &= (
            masks["matched"]
            if label not in ("native", "matched", "boundary")
            else True
        )
        if mask.sum() < 2:
            continue
        wt = trap_weights(times)[mask]
        wt /= wt.sum()
        wf = trap_weights(freq[band])
        weights = wt[:, None] * wf
        output[label] = {
            "risk": float(np.sum(loss[mask][:, band] * weights)),
            "log_rmse": float(
                np.sqrt(
                    np.sum(
                        (np.log(mean[mask][:, band] / target[mask][:, band]))
                        ** 2
                        * weights
                    )
                )
            ),
            "coverage_90": float(
                np.sum(
                    (
                        (target[mask][:, band] >= quant[0][mask][:, band])
                        & (target[mask][:, band] <= quant[2][mask][:, band])
                    )
                    * weights
                )
            ),
            "log_width_90": float(
                np.sum(
                    np.log(quant[2][mask][:, band] / quant[0][mask][:, band])
                    * weights
                )
            ),
        }
    edge = ~band
    output["edge_risk"] = float(loss[masks["matched"]][:, edge].mean())
    output["center_retention_fraction"] = float(
        data.time.max() - data.time.min()
    )
    output["ordinate_fraction"] = float(data.power.size / n)
    output["window_duration"] = (2 * m + 1) * cfg["dt"]
    output["window_span"] = 2 * m * cfg["dt"]
    output["matched_time_bounds"] = [
        max(low for low, _ in bounds),
        min(high for _, high in bounds),
    ]
    output["peak_frequency_rmse"] = float(
        np.sqrt(
            np.mean(
                (
                    freq[np.argmax(mean[masks["matched"]], axis=1)]
                    - freq[np.argmax(target[masks["matched"]], axis=1)]
                )
                ** 2
            )
        )
    )

    def peak_width(surface):
        widths = []
        for row in surface:
            peak = np.argmax(row)
            left = right = peak
            while left > 0 and row[left - 1] >= row[peak] / 2:
                left -= 1
            while right < len(row) - 1 and row[right + 1] >= row[peak] / 2:
                right += 1
            widths.append(freq[right] - freq[left])
        return np.array(widths)

    output["peak_width_rmse"] = float(
        np.sqrt(
            np.mean(
                (
                    peak_width(mean[masks["matched"]])
                    - peak_width(target[masks["matched"]])
                )
                ** 2
            )
        )
    )
    # Descriptive abrupt-transition diagnostic; meaningful only for that family.
    amplitude = np.log(mean[:, band].mean(axis=1))
    pre, post = (
        amplitude[(times > 0.2) & (times < 0.4)].mean(),
        amplitude[(times > 0.6) & (times < 0.8)].mean(),
    )
    crossing = []
    for fraction in (0.1, 0.9):
        selected = np.flatnonzero(
            (times > 0.35)
            & (times < 0.65)
            & (amplitude >= pre + fraction * (post - pre))
        )
        crossing.append(
            float(times[selected[0]]) if selected.size else float("nan")
        )
    output["abrupt_transition_10_90_width_u"] = crossing[1] - crossing[0]
    return output


def posterior_losses(
    result: PSDResult, model: LogPSpline, n: int, cfg: dict
) -> dict[str, np.ndarray]:
    # Stage 3 comparison only: D distributions, no posterior horizon decisions.
    weights = result.posterior.weights.values.reshape(
        -1, *result.posterior.weights.shape[-2:]
    )
    freq = np.linspace(*cfg["frequency_band"], 33)
    bf = model.frequency.design_at(freq)
    arrays = {}
    for center in (0.12, 0.38, 0.68, 0.9):
        for half in cfg["calibration_windows"]:
            u = center + np.arange(-half, half + 1) / n
            bt = model.time.design_at(u)
            logs = np.einsum("ti,dij,fj->dtf", bt, weights, bf, optimize=True)
            arrays[f"center{center}_m{half}"] = stationarity_loss(logs)
    return arrays


def provenance(data, n: int, m: int, dt: float) -> dict[str, np.ndarray]:
    centers = np.rint(data.time * n).astype(int) - 1
    return {
        "centers": centers,
        "support_start": centers - m,
        "support_stop": centers + m + 1,
        "cycle_index": np.arange(centers.size) // m,
        "m": np.full(centers.size, m),
        "sample_count_duration": np.full(centers.size, (2 * m + 1) * dt),
        "center_span": np.full(centers.size, 2 * m * dt),
    }


def manifest(cfg: dict) -> dict:
    return {
        "repository_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "config": cfg,
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "jax",
                "jaxlib",
                "numpyro",
                "numpy",
                "scipy",
                "optax",
                "xarray",
            )
        },
        "backend": jax.default_backend(),
        "x64": jax.config.x64_enabled,
        "units": "moving-periodogram coefficient variance",
        "time": "(zero_based_center+1)/n",
        "frequency": "Hz",
        "physical_one_sided_conversion": "4*pi*dt, interior frequencies only",
    }


def run_job(cfg: dict, out: Path, job: dict) -> None:
    directory = out / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    record = {"job": job, "status": "running", **manifest(cfg)}
    write_json(directory / "manifest.json", record)
    try:
        n = cfg["calibration_n"] if job.get("calibration") else cfg["n"]
        x = simulate(job["family"], n, job["data_seed"], cfg["dt"])
        record["data_hash"] = hashlib.sha256(x.tobytes()).hexdigest()
        record["simulation_seconds"] = perf_counter() - started
        before = perf_counter()
        data = scattered_moving_periodogram(
            x, dt=cfg["dt"], m=job["m"], thin=cfg["thin"]
        )
        record["preprocessing_seconds"] = perf_counter() - before
        np.savez_compressed(
            directory / "observations.npz",
            x=x,
            time=data.time,
            frequency=data.frequency,
            power=data.power,
            counts=data.counts,
            **provenance(data, n, job["m"], cfg["dt"]),
        )
        model = build_model(
            cfg,
            calibration=job.get("calibration", False),
            larger=job.get("larger", False),
        )
        config = PowerConfig(
            method=job["method"],
            seed=job["inference_seed"],
            progress_bar=False,
            vi_guide=job.get("guide", "diag"),
            vi_steps=job.get("vi_steps", cfg["vi_steps"]),
            vi_lr=cfg["vi_lr"],
            vi_posterior_draws=cfg["posterior_draws"],
            spectrum_draws=32,
            num_chains=cfg["nuts_chains"],
            n_warmup=job.get("nuts_warmup", cfg["nuts_warmup"]),
            n_samples=cfg["nuts_draws"],
            target_accept_prob=job.get(
                "nuts_target_accept", cfg["nuts_target_accept"]
            ),
            dense_mass=job["method"] == "nuts",
        )
        record["inference_config"] = config.__dict__
        before = perf_counter()
        result = fit(data, config, model=model)
        record["fit_seconds"] = perf_counter() - before
        target = truth(
            job["family"], result.time, result.frequency, n, cfg["dt"]
        )
        record["metrics"] = metrics(result, target, cfg, data, job["m"], n)
        record["diagnostics"] = sampling_diagnostics(result)
        record["timings"] = {
            k: v for k, v in result.metadata.items() if k.endswith("seconds")
        }
        if result.vi is not None:
            record["vi_timings"] = result.vi.timings
            record["loss_trace"] = np.asarray(result.vi.losses).tolist()
            if not np.isfinite(result.vi.losses).all():
                raise ValueError("Nonfinite VI losses")
        result.to_netcdf(directory / "result.nc")
        loaded = PSDResult.from_netcdf(directory / "result.nc")
        np.testing.assert_array_equal(loaded.observed_data.time, data.time)
        np.testing.assert_array_equal(
            loaded.observed_data.frequency, data.frequency
        )
        assert loaded.metadata["units"] == data.units
        np.testing.assert_allclose(
            loaded.spectrum_summary["mean"], result.spectrum_summary["mean"]
        )
        if job.get("calibration"):
            np.savez_compressed(
                directory / "loss_distributions.npz",
                **posterior_losses(result, model, n, cfg),
            )
        np.savez_compressed(
            directory / "summary.npz",
            time=result.time,
            frequency=result.frequency,
            truth=target,
            mean=result.spectrum_summary["mean"].values[..., 0, 0].real,
            quantiles=result.quantiles(kind="real").values[..., 0, 0],
        )
        record["status"] = "ok"
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
    record["total_seconds"] = perf_counter() - started
    # macOS returns bytes, Linux KiB. Isolated process, inclusive of imports/JAX.
    record["peak_rss_mib"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    write_json(directory / "metrics.json", record)
    print(
        job["id"],
        record["status"],
        round(record["total_seconds"], 2),
        flush=True,
    )


def transform_rows(
    n: int, m: int, thin: int = 2, offset: int = 0, length: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = n if length is None else length
    raw = tang_moving_periodogram(np.zeros(length), m=m, thin=thin)
    centers = np.rint(raw["u"] * length).astype(int) - 1 + offset
    starts = centers - m
    rows = np.zeros((centers.size, n), complex)
    for k, (start, omega) in enumerate(zip(starts, raw["omega"], strict=True)):
        rows[k, start : start + 2 * m + 1] = np.exp(
            -1j * omega * np.arange(2 * m + 1)
        ) / np.sqrt(2 * np.pi * (2 * m + 1))
    return rows, centers, raw["omega"] / (2 * np.pi), np.full(centers.size, m)


def transform_checks(cfg: dict, out: Path) -> None:
    started = perf_counter()
    n = cfg["transform_n"]
    fixed = {f"fixed_m{m}": transform_rows(n, m) for m in (8, 32)}
    first = transform_rows(n, 8, length=n // 2)
    second = transform_rows(n, 32, length=n // 2, offset=n // 2)
    fixed["guarded_piecewise"] = tuple(
        np.concatenate([a, b]) for a, b in zip(first, second, strict=True)
    )
    starts = np.arange(0, n - 17 + 1, 8)
    rows, centers, frequency, halves = [], [], [], []
    for start in starts:
        for rung in range(1, 9):
            row = np.zeros(n, complex)
            row[start : start + 17] = np.exp(
                -2j * np.pi * rung * np.arange(17) / 17
            ) / np.sqrt(2 * np.pi * 17)
            rows.append(row)
            centers.append(start + 8)
            frequency.append(rung / 17)
            halves.append(8)
    fixed["naive_spectrogram"] = (
        np.array(rows),
        np.array(centers),
        np.array(frequency),
        np.array(halves),
    )
    # Inspect all exact covariances. MC uses the same externally fixed rows.
    rho = 0.7
    sigma_ar = toeplitz(rho ** np.arange(n)) / (1 - rho**2)
    scale = envelope(np.arange(n) / n, "mixed")
    covariances = {
        "white": np.eye(n),
        "ar1": sigma_ar,
        "mixed": scale[:, None] * sigma_ar * scale[None, :],
    }
    results = []
    rng = np.random.default_rng(cfg["transform_seed"])
    normals = rng.normal(size=(cfg["transform_replicates"], n))
    for family, sigma in covariances.items():
        x = normals @ np.linalg.cholesky(sigma).T
        for label, (a, centers, freq, halves) in fixed.items():
            c, p = a @ sigma @ a.conj().T, a @ sigma @ a.T
            mean = c.diagonal().real
            cov = abs(c) ** 2 + abs(p) ** 2
            variance = cov.diagonal()
            corr = cov / np.sqrt(variance[:, None] * variance[None, :])
            marginal = abs(x @ a.T) ** 2
            sample_mean = marginal.mean(axis=0)
            batches = marginal.reshape(20, -1, len(mean))
            sample_var = marginal.var(axis=0, ddof=1)
            var_batches = batches.var(axis=1, ddof=1)
            var_se = var_batches.std(axis=0, ddof=1) / np.sqrt(20)
            zmean = abs(sample_mean - mean) / np.sqrt(
                variance / cfg["transform_replicates"]
            )
            target = truth(family, (centers + 1) / n, freq, n).diagonal()
            np.fill_diagonal(corr, 0)
            between = (centers[:, None] < n // 2) != (
                centers[None, :] < n // 2
            )
            nearby = (abs(centers[:, None] - n // 2) < 64) & (
                abs(centers[None, :] - n // 2) < 64
            )
            eigenvalues = np.linalg.eigvalsh(corr + np.eye(len(mean)))
            i, j = np.unravel_index(corr.argmax(), corr.shape)
            batch_cov = np.array(
                [np.cov(batch[:, i], batch[:, j])[0, 1] for batch in batches]
            )
            report = {
                "family": family,
                "operator": label,
                "n_rows": len(mean),
                "max_offdiag_power_correlation": float(corr.max()),
                "rms_offdiag_power_correlation": float(
                    np.sqrt((corr**2).sum() / (len(mean) * (len(mean) - 1)))
                ),
                "cross_epoch_max": float(corr[between].max()),
                "near_transition_max": float(corr[nearby].max()),
                "power_correlation_eigen_min": float(eigenvalues.min()),
                "power_correlation_eigen_max": float(eigenvalues.max()),
                "effective_rank": float(
                    eigenvalues.sum() ** 2 / (eigenvalues**2).sum()
                ),
                "max_non_circularity": float((abs(p.diagonal()) / mean).max()),
                "mean_relative_bias_max": float(abs(mean / target - 1).max()),
                "mean_relative_bias_rms": float(
                    np.sqrt(np.mean((mean / target - 1) ** 2))
                ),
                "mc_mean_max_z": float(zmean.max()),
                "mc_mean_pass_5se": bool(np.all(zmean < 5)),
                "mc_var_max_z": float(
                    (abs(sample_var - variance) / var_se).max()
                ),
                "max_corr_pair_cov_exact": float(cov[i, j]),
                "max_corr_pair_cov_mc": float(batch_cov.mean()),
                "max_corr_pair_cov_mc_se": float(
                    batch_cov.std(ddof=1) / np.sqrt(20)
                ),
                "guarded_supports_disjoint": bool(
                    np.all(
                        (centers - halves < n // 2)
                        == (centers + halves < n // 2)
                    )
                )
                if label == "guarded_piecewise"
                else None,
            }
            results.append(report)
            np.savez_compressed(
                out / f"transform_{family}_{label}.npz",
                C=c,
                P=p,
                power_covariance=cov,
                mean=mean,
                local_target=target,
                mc_mean=sample_mean,
                mc_variance=sample_var,
                centers=centers,
                frequency=freq,
                support_start=centers - halves,
                support_stop=centers + halves + 1,
            )
    # Native white power -> physical integration, nonunit dt.
    dt = 0.25
    physical_variance = np.trapezoid(
        np.full(1001, 4 * np.pi * dt / (2 * np.pi)),
        np.linspace(0, 1 / (2 * dt), 1001),
    )
    write_json(
        out / "transform_checks.json",
        {
            "status": "ok"
            if all(r["mc_mean_pass_5se"] for r in results)
            else "failed",
            "manifest": manifest(cfg),
            "results": results,
            "physical_white_variance_integral": physical_variance,
            "total_seconds": perf_counter() - started,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for family in covariances:
        subset = [r for r in results if r["family"] == family]
        axes[0].plot(
            [r["operator"] for r in subset],
            [r["max_offdiag_power_correlation"] for r in subset],
            "o-",
            label=family,
        )
        axes[1].plot(
            [r["operator"] for r in subset],
            [r["mean_relative_bias_rms"] for r in subset],
            "o-",
            label=family,
        )
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Maximum off-diagonal power correlation")
    axes[1].set_ylabel("RMS relative finite-window mean bias")
    fig.savefig(out / "figures/transform_checks.png", dpi=160)
    plt.close(fig)
    print(
        "Transform diagnostics",
        len(results),
        "seconds",
        round(perf_counter() - started, 2),
        flush=True,
    )


def oracle_checks(cfg: dict, out: Path) -> None:
    model = build_model(cfg)
    u, f = model.time.grid, model.frequency.grid
    records = []
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7), layout="constrained", sharex=True, sharey=True
    )
    for family, ax in zip(cfg["families"], axes.flat, strict=True):
        surface = truth(family, u, f, cfg["n"])
        band = (f >= cfg["frequency_band"][0]) & (
            f <= cfg["frequency_band"][1]
        )
        logs = np.log(surface)
        projected = (
            model.time.basis
            @ (
                np.linalg.pinv(model.time.basis)
                @ logs
                @ np.linalg.pinv(model.frequency.basis.T)
            )
            @ model.frequency.basis.T
        )
        projection_rmse = float(
            np.sqrt(np.mean((projected[:, band] - logs[:, band]) ** 2))
        )
        centers = np.arange(
            max(cfg["windows"]), cfg["n"] - max(cfg["windows"]), 32
        )
        losses = []
        for center in centers:
            losses.append(
                [
                    float(
                        stationarity_loss(
                            np.log(
                                truth(
                                    family,
                                    (center + 1 + np.arange(-m, m + 1))
                                    / cfg["n"],
                                    np.linspace(*cfg["frequency_band"], 65),
                                    cfg["n"],
                                )
                            )
                        )
                    )
                    for m in cfg["windows"]
                ]
            )
        losses = np.array(losses)
        horizons = {}
        for eps in cfg["epsilons"]:
            raw = horizon_summary(
                losses[None, ...], np.array(cfg["windows"]), epsilon=eps
            )
            prefix = horizon_summary(
                losses[None, ...],
                np.array(cfg["windows"]),
                epsilon=eps,
                prefix=True,
            )
            horizons[str(eps)] = {"raw": raw, "prefix": prefix}
            ax.step(
                (centers + 1) / cfg["n"],
                raw["selected"],
                where="mid",
                label=f"epsilon={eps}",
            )
        ax.set_title(family)
        ax.set_xlabel("Full-record u")
        ax.set_ylabel("Truth maximum accepted m")
        ax.set_yticks([-1, 8, 16, 32, 64], ["none", "8", "16", "32", "64*"])
        ax.grid(alpha=0.2)
        np.savez_compressed(
            out / f"oracle_{family}.npz",
            centers=centers,
            losses=losses,
            truth=surface,
            projected=projected,
        )
        records.append(
            {
                "family": family,
                "basis_log_rmse_band": projection_rmse,
                "horizons": horizons,
            }
        )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Truth-only horizon: 64* is right-censored; none means no acceptable candidate"
    )
    fig.savefig(out / "figures/oracle_horizons.png", dpi=160)
    plt.close(fig)
    write_json(out / "oracle_checks.json", records)


def jobs(cfg: dict, mode: str) -> list[dict]:
    output = []
    if mode == "sweep":
        for family in cfg["families"]:
            for seed in cfg["seeds"]:
                for m in cfg["windows"]:
                    output.append(
                        dict(
                            id=f"{family}_seed{seed}_m{m}_diag",
                            family=family,
                            data_seed=seed,
                            m=m,
                            method="vi",
                            guide="diag",
                            inference_seed=cfg["inference_seed"],
                        )
                    )
        for family in ("mixed", "ls2", "resonance"):
            for m in (16, 64):
                output.append(
                    dict(
                        id=f"{family}_seed3101_m{m}_larger",
                        family=family,
                        data_seed=3101,
                        m=m,
                        method="vi",
                        guide="diag",
                        inference_seed=cfg["inference_seed"],
                        larger=True,
                    )
                )
    else:
        for family in cfg["calibration_families"]:
            for m in cfg["calibration_windows"]:
                for method, guide, seed in [
                    ("nuts", "none", 7101),
                    ("vi", "diag", 7101),
                    ("vi", "diag", 7102),
                    ("vi", "lowrank:10", 7101),
                ]:
                    output.append(
                        dict(
                            id=f"cal_{family}_m{m}_{method}_{guide.replace(':', '')}_init{seed}",
                            family=family,
                            data_seed=3101,
                            m=m,
                            method=method,
                            guide=guide,
                            inference_seed=seed,
                            calibration=True,
                        )
                    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["transforms", "sweep", "calibration", "job"],
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job", default="")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=true before Python starts")
    cfg = tomllib.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    if args.mode == "job":
        run_job(cfg, args.output, json.loads(args.job))
        return
    write_json(args.output / "config.json", cfg)
    write_json(args.output / "manifest.json", manifest(cfg))
    if args.mode == "transforms":
        transform_checks(cfg, args.output)
        oracle_checks(cfg, args.output)
        return
    job_list = jobs(cfg, args.mode)
    write_json(args.output / f"{args.mode}_jobs.json", job_list)
    started = perf_counter()
    for job in job_list:
        record = args.output / job["id"] / "metrics.json"
        if record.exists():
            print("Preserving existing run", job["id"], flush=True)
            continue
        log = args.output / f"{job['id']}.log"
        before = perf_counter()
        with log.open("w") as stream:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode",
                    "job",
                    "--config",
                    str(args.config),
                    "--output",
                    str(args.output),
                    "--job",
                    json.dumps(job),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        if not record.exists():
            write_json(
                record,
                {
                    "job": job,
                    "status": "process_failed",
                    "returncode": completed.returncode,
                    "log": str(log),
                },
            )
        print(
            job["id"],
            json.loads(record.read_text())["status"],
            round(perf_counter() - before, 2),
            flush=True,
        )
    write_json(
        args.output / f"{args.mode}_execution.json",
        {"elapsed_seconds": perf_counter() - started, "n_jobs": len(job_list)},
    )


if __name__ == "__main__":
    main()
