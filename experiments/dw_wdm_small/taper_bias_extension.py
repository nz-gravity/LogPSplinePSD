"""Paired Hann/control extension and coloured posterior calibration pilot.

All fits use the public package likelihood. Existing fits remain read-only.
The coloured ensemble is a new development configuration, never pooled with
the earlier broadband ensemble. No simulation truth enters fit initialization.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import arviz_stats as azs
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import gap_bias_audit as audit
from experiments.dw_wdm_small import gap_smoke as gap
from experiments.dw_wdm_small import taper_coloured_audit as coloured
from experiments.dw_wdm_small import taper_smoke as taper
from scipy import sparse

from log_psplines import (
    LogPSpline,
    PowerConfig,
    PowerData,
    PSDResult,
    SplineBasis,
    fit,
)
from log_psplines.example_datasets.ls2_data import LS2Data

ROOT = bench.ROOT / "taper_bias_extension"
GENERATORS = ("broadband", "LS2")
VARIANTS = ("rectangular", "boxcar_sparse", "hann_sparse", "wdm")
LABELS = {
    "rectangular": "Original rectangular",
    "boxcar_sparse": "Matched rectangular",
    "hann_sparse": "Hann",
    "wdm": "Guarded WDM",
}
COLOURS = {
    "rectangular": "#2866a2",
    "boxcar_sparse": "#777777",
    "hann_sparse": "#b46a1d",
    "wdm": "#697d35",
}
SEEDS = {"broadband": list(range(6101, 6111)), "LS2": list(range(8201, 8211))}
FIT_BAND = (0.04, 0.47)
BASIS = {"broadband": (8, 2), "LS2": (32, 6)}
MASKS = ("complete", "single")


def model(generator):
    nt, nf = BASIS[generator]
    return LogPSpline(
        SplineBasis.from_grid(np.linspace(0, 0.5, 65), nf),
        time=SplineBasis.from_grid(np.linspace(0, 1, 65), nt),
    )


def simulate(generator, seed):
    if generator == "broadband":
        return bench.simulate(seed)
    return LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=seed).data[:, 0]


def truth(generator, u, f):
    if generator == "broadband":
        return np.broadcast_to(
            2 * bench.DT * bench.variance(u)[:, None], (len(u), len(f))
        )
    b = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=8200).coefficient(
        u[:, None]
    )
    return (
        2
        * bench.DT
        * (1 + b**2 + 2 * b * np.cos(2 * np.pi * f[None, :] * bench.DT))
    )


def saved_mask(mask):
    return np.load(audit.ROOT / f"mask_{mask}.npz")


def observations(x, variant, mask, generator):
    saved = saved_mask(mask)
    observed, rho = saved["observed"], saved["rho"]
    if variant == "wdm":
        data = gap.wdm_data(x, observed, rho)
    elif variant == "rectangular":
        data = gap.dw_data(x, observed)
    else:
        data = taper.prepare(
            x, "hann" if variant == "hann_sparse" else "boxcar", observed
        )
    if generator == "LS2":
        # A declared development band; it keeps every sparse MP rung. WDM's
        # whole-column gate still tests all original interior frequencies.
        keep = (data.frequency >= FIT_BAND[0]) & (
            data.frequency <= FIT_BAND[1]
        )
        data = PowerData(
            data.power[:, keep] if data.is_grid else data.power[keep],
            data.counts[:, keep] if data.is_grid else data.counts[keep],
            data.frequency[keep],
            data.time if data.is_grid else data.time[keep],
            data.units,
        )
    return data


def existing(generator, mask, variant, seed):
    if generator != "broadband":
        return None
    method = "wdm" if variant == "wdm" else "dw"
    if variant in ("rectangular", "wdm"):
        return audit.selected_path(seed, method, mask, repaired=True)
    if mask == "single":
        group = "repairs" if seed == 6101 else "fits"
        return bench.ROOT / "taper_study" / group / variant / f"seed{seed}_dw"
    return None


def fit_directory(generator, mask, variant, seed, attempt):
    group = "LS2_diagonal" if generator == "LS2" else generator
    return (
        ROOT
        / "fits"
        / group
        / mask
        / variant
        / f"seed{seed}"
        / f"attempt{attempt}"
    )


def fit_path(generator, mask, variant, seed):
    original = existing(generator, mask, variant, seed)
    if original is not None:
        return original
    attempts = [
        fit_directory(generator, mask, variant, seed, a) for a in range(3)
    ]
    completed = [p for p in attempts if (p / "metrics.json").exists()]
    passing = [
        p
        for p in completed
        if audit.screen(audit.read(p / "metrics.json")["diagnostics"])
    ]
    return passing[-1] if passing else completed[-1] if completed else None


def diagnostics(result, master):
    u, f = np.linspace(0.05, 0.95, 9), np.linspace(0.05, 0.45, 5)
    w = result.posterior.weights.values
    logs = np.einsum(
        "ti,cdij,fj->cdtf",
        master.time.design_at(u),
        w,
        master.frequency.design_at(f),
        optimize=True,
    )
    summaries = [
        azs.summary(
            result.posterior[["s", "sigma_time", "sigma_freq"]],
            round_to="none",
        ),
        azs.summary(
            xr.Dataset({"log_psd": (("chain", "draw", "time", "freq"), logs)}),
            round_to="none",
        ),
    ]
    d = {
        "divergences": int(result.sample_stats.diverging.sum()),
        "depth_hits": int((result.sample_stats.n_steps >= 1023).sum()),
        "rhat_max": float(max(s.r_hat.max() for s in summaries)),
        "ess_bulk_min": float(min(s.ess_bulk.min() for s in summaries)),
        "ess_tail_min": float(min(s.ess_tail.min() for s in summaries)),
    }
    energy = result.sample_stats.energy.values
    d["ebfmi_min_descriptive"] = float(
        np.min(
            np.mean(np.diff(energy, axis=1) ** 2, axis=1)
            / np.var(energy, axis=1)
        )
    )
    d["pass"] = audit.screen(d)
    return d


def provenance(generator, mask, variant, seed, data, master):
    x, observed = simulate(generator, seed), saved_mask(mask)["observed"]
    return {
        "generator": generator,
        "mask": mask,
        "variant": variant,
        "seed": seed,
        "raw_sha256": bench.digest(x),
        "observed_raw_sha256": bench.digest(np.where(observed, x, 0.0)),
        "mask_sha256": bench.digest(observed),
        "observations_sha256": {
            k: bench.digest(np.asarray(getattr(data, k)))
            for k in ("time", "frequency", "counts", "power")
        },
        "model_hashes": {
            "time_knots": bench.digest(master.time.knots),
            "frequency_knots": bench.digest(master.frequency.knots),
            "time_penalty": bench.digest(np.asarray(master.time.penalty)),
            "frequency_penalty": bench.digest(
                np.asarray(master.frequency.penalty)
            ),
        },
        "likelihood_observations": int(np.count_nonzero(data.counts)),
        "real_component_count": float(data.counts.sum()),
    }


def run_fit(generator, mask, variant, seed):
    master = model(generator)
    x = simulate(generator, seed)
    data = observations(x, variant, mask, generator)
    original = existing(generator, mask, variant, seed)
    if original is not None:
        old = audit.read(original / "metrics.json")
        current = provenance(generator, mask, variant, seed, data, master)
        for key in (
            "raw_sha256",
            "model_hashes",
            "likelihood_observations",
            "real_component_count",
        ):
            assert old[key] == current[key], (original, key)
        if mask == "single":
            assert old["mask_sha256"] == current["mask_sha256"]
            assert old["observed_raw_sha256"] == current["observed_raw_sha256"]
        assert audit.screen(old["diagnostics"]), original
        return
    for attempt in range(3):
        path = fit_directory(generator, mask, variant, seed, attempt)
        if (path / "metrics.json").exists():
            old = audit.read(path / "metrics.json")
            assert old["raw_sha256"] == bench.digest(x)
            if audit.screen(old["diagnostics"]):
                return
            continue
        path.mkdir(parents=True, exist_ok=True)
        warmup = (
            (600, 1000, 1500)[attempt]
            if generator == "broadband"
            else (1000, 1500, 2000)[attempt]
        )
        samples = (
            (1000, 1500, 2500)[attempt]
            if generator == "broadband"
            else (1500, 2500, 4000)[attempt]
        )
        config = PowerConfig(
            n_interior_knots_time=BASIS[generator][0],
            n_interior_knots_freq=BASIS[generator][1],
            n_warmup=warmup,
            n_samples=samples,
            num_chains=2,
            centered=False,
            dense_mass=generator == "broadband",
            target_accept_prob=0.95 if attempt == 0 else 0.99,
            max_tree_depth=10,
            seed=seed + 10000,
            progress_bar=False,
            spectrum_draws=8,
            spectrum_chunk_size=8,
        )
        native_model = (
            bench.on_grid(master, data.time, data.frequency)
            if data.is_grid
            else master
        )
        start = perf_counter()
        print(
            json.dumps(
                {
                    "starting": [generator, mask, variant, seed, attempt],
                    "basis_shape": list(master.weights.shape),
                    "observations": int(np.count_nonzero(data.counts)),
                }
            ),
            flush=True,
        )
        result = fit(data, config, model=native_model)
        d = diagnostics(result, master)
        result.to_netcdf(path / "posterior.nc")
        row = {
            **provenance(generator, mask, variant, seed, data, master),
            "config": asdict(config),
            "diagnostics": d,
            "attempt": attempt,
            "inference_seconds": float(result.metadata["inference_seconds"]),
            "total_seconds": perf_counter() - start,
            "runner_sha256": audit.file_hash(__file__),
        }
        bench.write_json(path / "metrics.json", row)
        print(
            json.dumps(
                {
                    "completed": [generator, mask, variant, seed, attempt],
                    "seconds": row["total_seconds"],
                    "diagnostics": d,
                }
            ),
            flush=True,
        )
        del result
        jax.clear_caches()
        gc.collect()
        if d["pass"]:
            return
    print(
        json.dumps(
            {"unresolved": [generator, mask, variant, seed], "attempts": 3}
        ),
        flush=True,
    )


def sparse_rows(variant, observed):
    data = observations(
        np.zeros(bench.N),
        variant,
        "single" if not observed.all() else "complete",
        "LS2",
    )
    centres = np.rint(data.time * bench.N).astype(int)
    h = taper.weights("hann" if variant == "hann_sparse" else "boxcar")
    j = np.arange(len(h))
    values = (
        np.exp(-2j * np.pi * data.frequency[:, None] * bench.DT * j)
        * h
        / np.linalg.norm(h)
    )
    return sparse.csr_matrix(
        (
            values.ravel(),
            (
                np.repeat(np.arange(len(centres)), len(h)),
                (centres[:, None] - bench.M + j).ravel(),
            ),
        ),
        shape=(len(centres), bench.N),
    )


def wdm_all_pairs(mask, b):
    """Exact real Gaussian covariance via the finite MA innovation operator."""
    saved = saved_mask(mask)
    observed, rho = saved["observed"], saved["rho"]
    freqs = np.arange(1, bench.NF) / (2 * bench.NF * bench.DT)
    allowed = (freqs >= FIT_BAND[0]) & (freqs <= FIT_BAND[1])
    locations = np.argwhere(gap.wdm_keep(rho) & allowed[None, :])
    rows = np.empty((len(locations), bench.N))
    prototypes = {
        (f, parity): bench.wdm_row(bench.NT // 2 + parity, f + 1)
        for f in np.flatnonzero(allowed)
        for parity in (0, 1)
    }
    for i, (t, f) in enumerate(locations):
        actual_t = t + 1
        parity = actual_t % 2
        shift = (actual_t - (bench.NT // 2 + parity)) * bench.NF
        rows[i] = (
            np.roll(prototypes[f, parity], shift)
            * observed
            / np.sqrt(rho[t, f])
        )
    for i in np.linspace(0, len(rows) - 1, 5, dtype=int):
        t, f = locations[i]
        np.testing.assert_allclose(
            rows[i],
            bench.wdm_row(t + 1, f + 1) * observed / np.sqrt(rho[t, f]),
            atol=2e-12,
        )
    innovations = np.zeros((len(rows), bench.N + 1))
    innovations[:, 1:] = rows
    innovations[:, :-1] += rows * b
    variance = np.sum(innovations**2, axis=1)
    del rows
    maxima, sums, pairs = 0.0, np.zeros(len(variance)), 0
    for start in range(0, len(variance), 64):
        stop = min(start + 64, len(variance))
        covariance = innovations[start:stop] @ innovations[:stop].T
        corr = covariance**2 / (
            variance[start:stop, None] * variance[None, :stop]
        )
        for local, i in enumerate(range(start, stop)):
            values = corr[local, :i]
            if len(values):
                maxima = max(maxima, float(values.max()))
                sums[i] += values.sum()
                sums[:i] += values
            pairs += i
        if start % 1024 == 0:
            print(
                json.dumps(
                    {
                        "WDM_covariance": mask,
                        "rows_done": stop,
                        "total": len(variance),
                    }
                ),
                flush=True,
            )
    return {
        "rows": len(variance),
        "all_unordered_pairs_checked": pairs,
        "max_power_correlation": maxima,
        "max_sum_off_diagonal_power_correlations": float(sums.max()),
        "scope": "all retained pairs; finite MA innovations, actual masked energy-normalized real WDM kernels",
    }


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    candidates = []
    u, f = np.linspace(0.05, 0.95, 241), np.linspace(0.05, 0.45, 161)
    target = np.log(truth("LS2", u, f))
    for nt in (8, 12, 16, 20, 24, 32):
        for nf in (2, 6):
            bt = SplineBasis.from_grid(np.linspace(0, 1, 65), nt).design_at(u)
            bf = SplineBasis.from_grid(np.linspace(0, 0.5, 65), nf).design_at(
                f
            )
            w = np.linalg.lstsq(bt, target, rcond=None)[0] @ np.linalg.pinv(
                bf.T
            )
            e = bt @ w @ bf.T - target
            candidates.append(
                {
                    "interior_knots_time_frequency": [nt, nf],
                    "rms_log_representation_error": float(
                        np.sqrt(np.mean(e**2))
                    ),
                    "maximum_absolute": float(np.max(abs(e))),
                }
            )
    sources = [
        Path(__file__),
        bench.ROOT / "benchmark.py",
        bench.ROOT / "taper_smoke.py",
        bench.ROOT / "gap_smoke.py",
        bench.ROOT.parents[1] / "src/log_psplines/inference/power.py",
    ]
    protocol = {
        "stage": "development extension; no confirmation acceptance",
        "generators": SEEDS,
        "variants": {"broadband": list(VARIANTS), "LS2": list(VARIANTS[1:])},
        "masks": {
            mask: {
                "hash": bench.digest(saved_mask(mask)["observed"]),
                "intervals_half_open": audit.endpoints(
                    saved_mask(mask)["observed"]
                ).tolist(),
            }
            for mask in MASKS
        },
        "basis_interior_knots": BASIS,
        "basis_dimensions_time_frequency": {
            g: list(model(g).weights.shape) for g in GENERATORS
        },
        "coloured_fit_band_hz": FIT_BAND,
        "scoring_band_hz": [0.05, 0.45],
        "broadband_observation_band": "original complete positive MP/WDM schedule; matches archived fits",
        "cadence_seconds": bench.DT,
        "window_samples": 65,
        "Hann_convention": "symmetric numpy.hanning(65), sum squares=24; no empirical variance rescaling",
        "same_sparse_sites": "rungs 3,6,...,30 and complete-cycle mask gate; no cycle reset",
        "wdm_gate": "whole time column passes at every original interior frequency; >=99.99% energy retained",
        "candidate_basis_audit": candidates,
        "chosen_basis_reason": "primary-domain oracle RMS <0.002 and max <0.012 before fitting",
        "sampler_screen": "0 divergences/depth hits; Rhat<=1.01; bulk/tail ESS>=200",
        "sampler_mass_metric": {"broadband": "dense", "LS2": "diagonal"},
        "metric_validation": audit.read(ROOT / "metric_check/validation.json")
        if (ROOT / "metric_check/validation.json").exists()
        else None,
        "confidence": "fixed-site binomial intervals and whole-record paired bootstrap; exploratory ten-record development",
        "source_sha256": {
            str(p.relative_to(bench.ROOT.parents[1])): audit.file_hash(p)
            for p in sources
        },
        "source_plan_sha256": audit.file_hash(audit.ROOT / "study_plan.md"),
        "versions": {
            p: version(p)
            for p in ("jax", "numpy", "numpyro", "scipy", "wdm_transform")
        },
        "python": platform.python_version(),
    }
    chosen = next(
        c
        for c in candidates
        if c["interior_knots_time_frequency"] == list(BASIS["LS2"])
    )
    assert (
        chosen["rms_log_representation_error"] < 0.002
        and chosen["maximum_absolute"] < 0.012
    )
    target_path = ROOT / "protocol.json"
    if target_path.exists():
        old = audit.read(target_path)
        for key in (
            "generators",
            "basis_interior_knots",
            "coloured_fit_band_hz",
            "masks",
        ):
            assert json.loads(json.dumps(protocol[key])) == old[key], key
    bench.write_json(target_path, protocol)
    return protocol


def observation_audit():
    b = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=8200).coefficient(
        np.arange(bench.N) / bench.N
    )
    outputs = {}
    for mask in MASKS:
        observed = saved_mask(mask)["observed"]
        x = simulate("LS2", 8201)
        box = observations(x, "boxcar_sparse", mask, "LS2")
        hann = observations(x, "hann_sparse", mask, "LS2")
        for field in ("time", "frequency", "counts"):
            np.testing.assert_array_equal(
                getattr(box, field), getattr(hann, field)
            )
        for variant in VARIANTS[1:]:
            data = observations(x, variant, mask, "LS2")
            for value in (np.nan, 1e50, -1e50):
                np.testing.assert_array_equal(
                    observations(
                        np.where(observed, x, value), variant, mask, "LS2"
                    ).power,
                    data.power,
                )
            if variant == "wdm":
                uu, ff = np.meshgrid(data.time, data.frequency, indexing="ij")
                full_prediction = coloured.expected_wdm(
                    b, observed, saved_mask(mask)["rho"]
                )
                native_freq = np.arange(1, bench.NF) / (
                    2 * bench.NF * bench.DT
                )
                expected = full_prediction[
                    :,
                    (native_freq >= FIT_BAND[0])
                    & (native_freq <= FIT_BAND[1]),
                ]
                dependency = wdm_all_pairs(mask, b)
            else:
                uu, ff = data.time, data.frequency
                h = taper.weights(
                    "hann" if variant == "hann_sparse" else "boxcar"
                )
                expected = coloured.expected_dw(h, uu, ff, b)
                dependency = audit.gaussian_dependence(
                    sparse_rows(variant, observed), 1 + b**2, b
                )
            bb = LS2Data(
                n_samples=bench.N, fs=1 / bench.DT, seed=8200
            ).coefficient(uu)
            target = (
                2
                * bench.DT
                * (1 + bb**2 + 2 * bb * np.cos(2 * np.pi * ff * bench.DT))
            )
            keep = (
                (data.counts > 0)
                & (uu >= 0.05)
                & (uu <= 0.95)
                & (ff >= 0.05)
                & (ff <= 0.45)
            )
            e = np.log(expected[keep]) - np.log(target[keep])
            outputs[f"{mask}_{variant}"] = {
                "expected_mean_rms_log_mismatch": float(
                    np.sqrt(np.mean(e**2))
                ),
                "expected_mean_max_absolute_log_mismatch": float(
                    np.max(abs(e))
                ),
                "dependence": dependency,
                "hidden_values_invariance": True,
                "observations": int(np.count_nonzero(data.counts)),
            }
        bench.write_json(ROOT / "observation_audit.json", outputs)
    return outputs


def grid():
    lo, hi = audit.endpoints(saved_mask("single")["observed"])[0] / bench.N
    anchors = np.array(
        [0.2, lo - 65 / (2 * bench.N), hi + 65 / (2 * bench.N), 0.8]
    )
    u = np.unique(np.r_[np.linspace(0.05, 0.95, 241), anchors])
    f = np.unique(
        np.r_[np.linspace(0.05, 0.45, 57), 0.10, 0.15, 0.25, 0.30, 0.40]
    )
    return u, f, anchors


def reconstruct(path, generator):
    u, f, _ = grid()
    cache_dir = ROOT / "reconstructed"
    cache_dir.mkdir(exist_ok=True)
    cache = (
        cache_dir
        / f"{audit.file_hash(path / 'posterior.nc')}_{bench.digest(np.r_[u, f])}.npz"
    )
    result = PSDResult.from_netcdf(path / "posterior.nc")
    master = model(generator)
    np.testing.assert_array_equal(
        result.model_data.knots_time.values, master.time.knots
    )
    np.testing.assert_array_equal(
        result.model_data.knots_frequency.values, master.frequency.knots
    )
    nt, nf = master.weights.shape
    w = result.posterior.weights.values.reshape(-1, nt, nf)
    assert w.dtype == np.float64 and np.isfinite(w).all()
    bt, bf = master.time.design_at(u), master.frequency.design_at(f)
    if cache.exists():
        data = np.load(cache)
        q, bq = data["quantiles"], data["band_quantiles"]
    else:
        q, bq = np.empty((3, len(u), len(f))), np.empty((3, len(u), 3))
        for start in range(0, len(u), 12):
            logs = np.einsum(
                "ti,dij,fj->dtf", bt[start : start + 12], w, bf, optimize=True
            )
            q[:, start : start + 12] = np.quantile(
                logs, [0.05, 0.5, 0.95], axis=0
            )
            for bi, bounds in enumerate(audit.BANDS.values()):
                power = np.exp(logs) @ audit.band_weights(f, bounds)
                bq[:, start : start + 12, bi] = np.quantile(
                    power, [0.05, 0.5, 0.95], axis=0
                )
        np.savez_compressed(cache, quantiles=q, band_quantiles=bq)
    maximum = 0.0
    for ti, fi in (
        (0, 4),
        (60, 20),
        (120, 30),
        (180, 40),
        (len(u) - 1, len(f) - 1),
    ):
        draws = np.array([bt[ti] @ weight @ bf[fi] for weight in w])
        actual = np.quantile(draws, [0.05, 0.5, 0.95])
        maximum = max(maximum, float(np.max(abs(actual - q[:, ti, fi]))))
    assert maximum < 3e-12
    return q, bq, maximum


def summarize():
    u, f, anchors = grid()
    regions = audit.regions(u, saved_mask("single")["observed"], 327 / bench.N)
    resamples = np.random.default_rng(2026101301).integers(0, 10, (5000, 10))
    all_rows, coverage, ledger, arrays, checks = [], [], [], {}, []
    target_configs = {}
    for generator in GENERATORS:
        variants = VARIANTS if generator == "broadband" else VARIANTS[1:]
        target = truth(generator, u, f)
        true_band = np.column_stack(
            [
                target @ audit.band_weights(f, bounds)
                for bounds in audit.BANDS.values()
            ]
        )
        q, bq = {}, {}
        for variant in variants:
            for mask in MASKS:
                records, band_records = [], []
                for seed in SEEDS[generator]:
                    path = fit_path(generator, mask, variant, seed)
                    if path is None:
                        raise RuntimeError(
                            f"Missing fit: {generator} {mask} {variant} {seed}"
                        )
                    row = audit.read(path / "metrics.json")
                    target_configs.setdefault(
                        generator, audit.config_target(row["config"])
                    )
                    assert (
                        audit.config_target(row["config"])
                        == target_configs[generator]
                    ), (path, "target or sampler configuration mismatch")
                    expected = provenance(
                        generator,
                        mask,
                        variant,
                        seed,
                        observations(
                            simulate(generator, seed), variant, mask, generator
                        ),
                        model(generator),
                    )
                    for key in (
                        "raw_sha256",
                        "model_hashes",
                        "likelihood_observations",
                        "real_component_count",
                    ):
                        assert row[key] == expected[key], (path, key)
                    values, bands, error = reconstruct(path, generator)
                    records.append(values)
                    band_records.append(bands)
                    checks.append(
                        {
                            "generator": generator,
                            "variant": variant,
                            "mask": mask,
                            "seed": seed,
                            "path": str(path.relative_to(bench.ROOT)),
                            "maximum_quantile_error": error,
                            "sampler_pass": audit.screen(row["diagnostics"]),
                            "posterior_sha256": audit.file_hash(
                                path / "posterior.nc"
                            ),
                        }
                    )
                    ledger.append(
                        {
                            **expected,
                            "selected_path": str(path.relative_to(bench.ROOT)),
                            "config": row["config"],
                            "diagnostics": row["diagnostics"],
                            "reused": existing(generator, mask, variant, seed)
                            is not None,
                        }
                    )
                q[variant, mask], bq[variant, mask] = (
                    np.asarray(records),
                    np.asarray(band_records),
                )
                arrays[f"{generator}_{variant}_{mask}"] = q[variant, mask]
                arrays[f"{generator}_{variant}_{mask}_band"] = bq[
                    variant, mask
                ]
                included = (q[variant, mask][:, 0] <= np.log(target)) & (
                    np.log(target) <= q[variant, mask][:, 2]
                )
                for ai, time in enumerate(anchors):
                    ti = int(np.searchsorted(u, time))
                    for frequency in (0.1, 0.25, 0.4):
                        fi = int(np.searchsorted(f, frequency))
                        count = int(included[:, ti, fi].sum())
                        low, high = audit.wilson(count, 10)
                        coverage.append(
                            {
                                "generator": generator,
                                "variant": variant,
                                "mask": mask,
                                "anchor": (
                                    "far_left",
                                    "edge_left",
                                    "edge_right",
                                    "far_right",
                                )[ai],
                                "time": float(time),
                                "frequency_hz": frequency,
                                "truth_included": count,
                                "records": 10,
                                "coverage90": count / 10,
                                "wilson95_low": float(low),
                                "wilson95_high": float(high),
                                "interval_width_log": float(
                                    np.mean(
                                        q[variant, mask][:, 2, ti, fi]
                                        - q[variant, mask][:, 0, ti, fi]
                                    )
                                ),
                                "point_estimate_scatter_log_sd": float(
                                    np.std(
                                        q[variant, mask][:, 1, ti, fi], ddof=1
                                    )
                                ),
                                "formal_calibration": "development only; binomial bounds too wide for [0.85,0.95] acceptance",
                            }
                        )
        vectors, generator_rows = [], []
        for variant in variants:
            for kind in ("complete", "single", "added"):
                error = (
                    q[variant, "single"][:, 1] - q[variant, "complete"][:, 1]
                    if kind == "added"
                    else q[variant, kind][:, 1] - np.log(target)
                )
                linear = (
                    (
                        np.exp(q[variant, "single"][:, 1])
                        - np.exp(q[variant, "complete"][:, 1])
                    )
                    / target
                    if kind == "added"
                    else np.expm1(error)
                )
                band_error = (
                    (
                        bq[variant, "single"][:, 1]
                        - bq[variant, "complete"][:, 1]
                    )
                    / true_band
                    if kind == "added"
                    else bq[variant, kind][:, 1] / true_band - 1
                )
                for region, keep in regions.items():
                    tw = audit.axis_weights(u) * keep
                    tw /= tw.sum()
                    for bi, (band, bounds) in enumerate(audit.BANDS.items()):
                        fw = audit.band_weights(f, bounds)
                        fw /= fw.sum()
                        weights = np.outer(tw, fw)
                        record_errors = np.sum(error * weights, axis=(1, 2))
                        vector = record_errors
                        row = {
                            "generator": generator,
                            "variant": variant,
                            "kind": kind,
                            "region": region,
                            "band": band,
                            "records": 10,
                            "mean_log_bias": float(vector.mean()),
                            "rms_bias_map": float(
                                np.sqrt(
                                    np.sum(error.mean(axis=0) ** 2 * weights)
                                )
                            ),
                            "p95_absolute_bias_map": float(
                                np.quantile(
                                    abs(error.mean(axis=0)[keep][:, fw > 0]),
                                    0.95,
                                )
                            ),
                            "rmse_log": float(
                                np.sqrt(
                                    np.mean(
                                        np.sum(error**2 * weights, axis=(1, 2))
                                    )
                                )
                            ),
                            "mean_fractional_PSD_bias": float(
                                np.mean(np.sum(linear * weights, axis=(1, 2)))
                            ),
                            "mean_fractional_band_power_bias": float(
                                np.mean(band_error[:, :, bi] @ tw)
                            ),
                            "record_log_sd": float(vector.std(ddof=1)),
                        }
                        for key, vv in (
                            ("log", vector),
                            (
                                "fractional_PSD",
                                np.sum(linear * weights, axis=(1, 2)),
                            ),
                            ("band_power", band_error[:, :, bi] @ tw),
                        ):
                            lo, hi = np.quantile(
                                audit.resample_means(vv, resamples),
                                [0.025, 0.975],
                            )
                            (
                                row[f"{key}_bootstrap95_low"],
                                row[f"{key}_bootstrap95_high"],
                            ) = float(lo), float(hi)
                        generator_rows.append(row)
                        vectors.append(vector)
                arrays[f"{generator}_{variant}_{kind}_bias"] = error.mean(
                    axis=0
                )
        _, low, high, critical = audit.simultaneous_intervals(
            np.column_stack(vectors), resamples
        )
        for i, row in enumerate(generator_rows):
            row["simultaneous_log95_low"], row["simultaneous_log95_high"] = (
                float(low[i]),
                float(high[i]),
            )
            row["judgment"] = "development bound; no formal acceptance"
        all_rows.extend(generator_rows)
        np.savez_compressed(
            ROOT / f"{generator}_record_bias_vectors.npz",
            seeds=SEEDS[generator],
            vectors=np.column_stack(vectors),
            bootstrap_indices=resamples,
        )
        print(
            json.dumps(
                {
                    "summarized": generator,
                    "rows": len(generator_rows),
                    "bootstrap_max_standardized_error_critical": critical,
                }
            ),
            flush=True,
        )
        make_figures(generator, variants, u, f, arrays, coverage)
    attempts = [
        {"path": str(p.parent.relative_to(bench.ROOT)), **audit.read(p)}
        for p in sorted((ROOT / "fits").rglob("metrics.json"))
    ]
    payload = {
        "regional_bias": all_rows,
        "fixed_site_coverage": coverage,
        "fit_inventory": ledger,
        "posterior_validation": checks,
        "attempt_ledger": attempts,
        "primary_fits": len(ledger),
        "new_fit_attempts": len(attempts),
        "unresolved_fits": [
            r for r in ledger if not audit.screen(r["diagnostics"])
        ],
        "stage": "development; no new confirmation records and no LISA simulation",
        "target_configs": target_configs,
        "interrupted_attempts": audit.read(ROOT / "interrupted_attempts.json"),
        "dense_diagonal_check": audit.read(
            ROOT / "metric_check/validation.json"
        ),
        "native_netcdf_limit": "Package-native posterior files reload successfully; the writer emits its existing NetCDF compatibility warning.",
    }
    bench.write_json(ROOT / "summary.json", payload)
    np.savez_compressed(ROOT / "paired_surfaces.npz", time=u, freq=f, **arrays)
    for filename, rows in (
        ("regional_bias.csv", all_rows),
        ("fixed_site_coverage.csv", coverage),
    ):
        with (ROOT / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_results(payload)
    return payload


def make_figures(generator, variants, u, f, arrays, coverage):
    fig, axes = plt.subplots(
        len(variants),
        3,
        figsize=(11, 2.2 * len(variants)),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    extent = max(
        float(np.max(abs(arrays[f"{generator}_{v}_{k}_bias"])))
        for v in variants
        for k in ("complete", "single", "added")
    )
    edges = audit.endpoints(saved_mask("single")["observed"])[0] / bench.N
    for i, variant in enumerate(variants):
        for j, kind in enumerate(("complete", "single", "added")):
            image = axes[i, j].pcolormesh(
                u,
                f,
                arrays[f"{generator}_{variant}_{kind}_bias"].T,
                cmap="RdBu_r",
                vmin=-extent,
                vmax=extent,
                shading="auto",
            )
            for e in edges:
                axes[i, j].axvline(e, color="black", ls="--", lw=0.7)
            axes[i, j].set(title=f"{LABELS[variant]}: {kind}")
        axes[i, 0].set_ylabel("Frequency (Hz)")
    for ax in axes[-1]:
        ax.set_xlabel("Physical rescaled time")
    fig.colorbar(image, ax=axes, label="Across-record log bias", shrink=0.75)
    fig.suptitle(f"{generator}: ten paired development records")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"{generator}_bias_maps.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), layout="constrained", sharey=True
    )
    for ax, mask in zip(axes, MASKS, strict=True):
        for vi, variant in enumerate(variants):
            rows = [
                r
                for r in coverage
                if r["generator"] == generator
                and r["variant"] == variant
                and r["mask"] == mask
                and r["anchor"] in ("edge_left", "edge_right")
            ]
            x = np.arange(len(rows)) + (vi - (len(variants) - 1) / 2) * 0.13
            y = np.array([r["coverage90"] for r in rows])
            low = np.array([r["wilson95_low"] for r in rows])
            high = np.array([r["wilson95_high"] for r in rows])
            ax.errorbar(
                x,
                y,
                yerr=np.maximum(np.array([y - low, high - y]), 0),
                fmt="o",
                color=COLOURS[variant],
                label=LABELS[variant],
                capsize=2,
                ms=4,
            )
        ax.axhspan(0.85, 0.95, color="#769c66", alpha=0.18)
        ax.axhline(0.9, color="black", lw=0.7, ls="--")
        ax.set(
            xticks=np.arange(6),
            xticklabels=["L .10", "L .25", "L .40", "R .10", "R .25", "R .40"],
            title=mask,
            ylim=(0, 1.02),
            xlabel="Observed edge / frequency (Hz)",
        )
    axes[0].set_ylabel("Fixed-site truth inclusion across records")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"{generator}: nominal 90% posterior intervals; bars are 95% binomial Monte Carlo bounds"
    )
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"{generator}_coverage.{ext}", dpi=160)
    plt.close(fig)


def write_results(summary):
    """Write analysis reports; manuscript text is maintained in main.tex."""
    rows = summary["regional_bias"]
    widths = {band: hi - lo for band, (lo, hi) in audit.BANDS.items()}
    table = []
    for generator in GENERATORS:
        variants = VARIANTS if generator == "broadband" else VARIANTS[1:]
        for variant in variants:
            values = []
            for kind in ("single", "added"):
                for region in ("left_observed", "right_observed"):
                    rr = [
                        row
                        for row in rows
                        if row["generator"] == generator
                        and row["variant"] == variant
                        and row["kind"] == kind
                        and row["region"] == region
                    ]
                    values.append(
                        sum(
                            row["mean_log_bias"] * widths[row["band"]] / 0.4
                            for row in rr
                        )
                    )
            table.append(
                {
                    "generator": generator,
                    "variant": variant,
                    "left_total": values[0],
                    "right_total": values[1],
                    "left_added": values[2],
                    "right_added": values[3],
                }
            )
    bench.write_json(ROOT / "headline_table.json", table)
    report = [
        "# Paired Hann/control bias and coloured calibration extension",
        "",
        "Development pilot: ten independent records per generator, complete and central-gap masks. Details are in the manuscript appendix; no physical LISA simulation or confirmation claim is introduced.",
        "",
        f"- {summary['primary_fits']} selected fits; {summary['new_fit_attempts']} new fit attempts; {len(summary['unresolved_fits'])} unresolved selected fits.",
        "- Shared coloured 36-by-10 basis: primary oracle RMS about 0.00127. All coloured observation bands are 0.04–0.47 Hz; score 0.05–0.45 Hz. Every original sparse MP rung is retained.",
        "- Log, fractional PSD and draw-integrated band-power biases are reported separately. Bootstrap entire record vectors; never bootstrap pixels.",
        "- Fixed-site coverage includes 95% binomial Monte Carlo bounds. Ten records cannot establish coverage in [0.85,0.95].",
        "- Hann substantially improves the coloured observation-mean audit and properness, but frequency-specific signed bias remains. Near-zero frequency-averaged bias hides cancellation; no observed-edge bias equivalence is established.",
        "- Coherent complete-data bias is present too; a small added gap displacement cannot establish absolute accuracy. Regularisation and finite-information effects need a separate posterior-reference diagnosis.",
        "- Coloured sampling uses the validated diagonal NUTS metric. Four dense pilot fits and two interrupted jobs remain documented; three initial diagonal fits were extended without replacing their seeds. Native posterior reloads pass despite the package writer's existing NetCDF compatibility warning.",
        "",
        "| Generator | Method | Left total | Right total | Left added | Right added |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in table:
        report.append(
            f"| {row['generator']} | {LABELS[row['variant']]} | "
            + " | ".join(
                f"{row[k]:+.4f}"
                for k in (
                    "left_total",
                    "right_total",
                    "left_added",
                    "right_added",
                )
            )
            + " |"
        )
    report.extend(
        [
            "",
            "![Coloured bias maps](LS2_bias_maps.png)",
            "",
            "![Coloured fixed-site coverage](LS2_coverage.png)",
            "",
            "[Frozen protocol](protocol.json), [exact observation/covariance audit](observation_audit.json), [fit and failure ledger](summary.json), [regional bias CSV](regional_bias.csv), [fixed-site coverage CSV](fixed_site_coverage.csv). Full draws remain in NetCDF; per-record quantiles and paired bootstrap vectors are saved.",
            "",
            "Reproduce from the analysis repository:",
            "",
            "```bash",
            "export MPLCONFIGDIR=/tmp/psd-gap-bias-mpl OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=true",
            ".venv/bin/python -m experiments.dw_wdm_small.taper_bias_extension freeze",
            ".venv/bin/python -m experiments.dw_wdm_small.taper_bias_extension audit",
            ".venv/bin/python -m experiments.dw_wdm_small.taper_bias_extension fit",
            ".venv/bin/python -m experiments.dw_wdm_small.taper_bias_extension summarize",
            "```",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(report) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("freeze", "audit", "fit", "summarize")
    )
    parser.add_argument(
        "--generators", nargs="+", choices=GENERATORS, default=list(GENERATORS)
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "audit":
        observation_audit()
    elif args.phase == "summarize":
        summarize()
    else:
        assert (ROOT / "protocol.json").exists(), (
            "Freeze protocol before inference"
        )
        for generator in args.generators:
            for seed in args.seeds or SEEDS[generator]:
                for mask in MASKS:
                    for variant in args.variants:
                        if generator == "LS2" and variant == "rectangular":
                            continue
                        run_fit(generator, mask, variant, seed)


if __name__ == "__main__":
    main()
