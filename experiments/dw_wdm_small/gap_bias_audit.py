"""First-stage bias audit; reads existing fits and never launches inference.

Run from the repository root: .venv/bin/python -m
experiments.dw_wdm_small.gap_bias_audit. All original artifacts are read-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small import gap_smoke as gap
from experiments.dw_wdm_small import taper_coloured_audit as coloured
from experiments.dw_wdm_small.rerun_summary import axis_weights
from scipy import sparse
from scipy.signal import periodogram
from scipy.stats import norm

from log_psplines import PSDResult
from log_psplines.example_datasets.ls2_data import LS2Data

ROOT = bench.ROOT / "gap_bias_audit"
SEEDS = list(range(6101, 6111))
BANDS = {"low": (0.05, 0.15), "middle": (0.15, 0.30), "high": (0.30, 0.45)}
BOOTSTRAP_SEED = 20261012
BOOTSTRAPS = 5000
BASELINE_DURATION = (2 * bench.M + 1) * bench.DT


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoints(observed):
    """Half-open integer intervals, including gaps touching record boundaries."""
    changes = np.diff(np.r_[True, observed, True].astype(int))
    return np.column_stack(
        (np.flatnonzero(changes == -1), np.flatnonzero(changes == 1))
    )


def masks():
    fragmented = np.ones(bench.N, dtype=bool)
    for u, length in zip(
        (0.2, 0.4, 0.6, 0.8), (205, 205, 205, 204), strict=True
    ):
        centre = round(u * bench.N)
        start = centre - length // 2
        fragmented[start : start + length] = False
    single = gap.masks()["contiguous"]
    assert (~single).sum() == (~fragmented).sum() == 819
    assert len(endpoints(fragmented)) == 4
    return {
        "complete": np.ones(bench.N, dtype=bool),
        "single": single,
        "fragmented": fragmented,
    }


def regions(u, observed, radius):
    """Assign each observed site to its nearest edge, without duplication."""
    u = np.asarray(u)
    intervals = endpoints(observed) / bench.N
    edges = intervals.ravel()
    inside = ~observed[
        np.minimum(np.floor(u * bench.N).astype(int), bench.N - 1)
    ]
    if len(edges):
        nearest = np.argmin(abs(u[:, None] - edges), axis=1)
        distance = abs(u - edges[nearest])
        left = nearest % 2 == 0
    else:
        distance = np.full(len(u), np.inf)
        left = np.ones(len(u), dtype=bool)
    near = (distance <= radius) & ~inside
    return {
        "left_observed": near & left,
        "right_observed": near & ~left,
        "observed_far": ~near & ~inside,
        "gap_interior": inside,
    }


def config_target(config):
    """Remove sampler effort and runtime-only settings, preserving the target."""
    runtime = {
        "n_warmup",
        "n_samples",
        "target_accept_prob",
        "seed",
        "progress_bar",
        "spectrum_draws",
        "spectrum_chunk_size",
    }
    return {k: v for k, v in config.items() if k not in runtime}


def screen(d):
    return (
        d["divergences"] == 0
        and d["depth_hits"] == 0
        and d["rhat_max"] <= 1.01
        and d["ess_bulk_min"] >= 200
        and d["ess_tail_min"] >= 200
    )


def selected_path(seed, method, mask, repaired=False):
    if mask == "single":
        variant = "rectangular" if method == "dw" else "wdm"
        return (
            bench.ROOT / "taper_study/fits" / variant / f"seed{seed}_{method}"
        )
    root = bench.ROOT / "results/reuse_rerun"
    if repaired and seed in (6103, 6110):
        root = root / "repairs"
    return root / f"seed{seed}_{method}"


def inventory(mask_set):
    master = bench.master_model()
    hashes = {
        "time_knots": bench.digest(master.time.knots),
        "frequency_knots": bench.digest(master.frequency.knots),
        "time_penalty": bench.digest(np.asarray(master.time.penalty)),
        "frequency_penalty": bench.digest(
            np.asarray(master.frequency.penalty)
        ),
    }
    canonical = read(selected_path(6101, "dw", "complete") / "metrics.json")[
        "config"
    ]
    rows = []
    for p in sorted(bench.ROOT.rglob("metrics.json")):
        if ROOT in p.parents:
            continue
        m = read(p)
        if "seed" not in m or "config" not in m:
            continue
        issues = []
        if m.get("variant") in ("hann_sparse", "boxcar_sparse"):
            issues.append("exploratory_variant_not_primary_schedule")
        if m.get("model_hashes") != hashes:
            issues.append("basis_or_penalty_mismatch_or_unrecorded")
        if config_target(m["config"]) != config_target(canonical):
            issues.append("target_configuration_mismatch")
        if m["raw_sha256"] != bench.digest(bench.simulate(m["seed"])):
            issues.append("raw_record_hash_mismatch")
        if "mask_sha256" in m and m["mask_sha256"] != bench.digest(
            mask_set["single"]
        ):
            issues.append("not_protocol_single_mask")
        if not screen(m["diagnostics"]):
            issues.append("sampler_screen_failed")
        for name in ("posterior.nc", "surface.npz"):
            if not (p.parent / name).exists():
                issues.append(f"missing_{name}")
        selected = []
        for mask in ("complete", "single"):
            for repaired in (False, True):
                if p.parent == selected_path(
                    m["seed"], m["method"], mask, repaired
                ):
                    selected.append(
                        f"{mask}_{'repaired_sensitivity' if repaired else 'original'}"
                    )
        if "single_original" in selected and m.get(
            "observed_raw_sha256"
        ) != bench.digest(
            np.where(mask_set["single"], bench.simulate(m["seed"]), 0.0)
        ):
            issues.append("observed_record_hash_missing_or_mismatch")
        relative = str(p.parent.relative_to(bench.ROOT))
        lineage = (
            "same_target_extra_sampler_effort"
            if "repairs" in p.parts
            else "primary"
        )
        if "gaps/contiguous" in relative:
            issues.append("legacy_WDM_gate_for_wdm_only") if m[
                "method"
            ] == "wdm" else None
        rows.append(
            {
                "path": relative,
                "seed": m["seed"],
                "method": m["method"],
                "selected_for": selected,
                "issues": issues,
                "raw_sha256": m["raw_sha256"],
                "mask_sha256": m.get("mask_sha256"),
                "model_hashes": m.get("model_hashes"),
                "config": m["config"],
                "diagnostics": m["diagnostics"],
                "lineage": lineage,
                "artifact_sha256": {
                    name: file_hash(p.parent / name)
                    for name in ("metrics.json", "posterior.nc", "surface.npz")
                    if (p.parent / name).exists()
                },
                "variant": m.get("variant", "original"),
            }
        )
    frozen = read(bench.ROOT / "taper_study/protocol.json")
    source = frozen.get("final_source_sha256", frozen["source_sha256"])
    source_checks = {
        p: {
            "archived": h,
            "current": file_hash(bench.ROOT.parents[1] / p),
            "match": h == file_hash(bench.ROOT.parents[1] / p),
        }
        for p, h in source.items()
    }
    archived_power = (
        bench.ROOT / "results/reuse_rerun/optimized_power_source.py"
    )
    payload = {
        "grain": "one archived fit attempt; duplicates are retained, not independent records",
        "runs": rows,
        "selected_primary_fits": 40,
        "source_checks": source_checks,
        "complete_inference_snapshot_matches_current": file_hash(
            archived_power
        )
        == file_hash(
            bench.ROOT.parents[1] / "src/log_psplines/inference/power.py"
        ),
        "provenance_limits": [
            "No per-fit hashes of reconstructed observation arrays were archived; adapters and mask/input hashes are checked now.",
            "Original complete-data runner changed after inference; the archived optimized inference source and environment are retained.",
        ],
        "missing_new_inference": {
            "phase_B_exact_reference": 30,
            "phase_C_broadband_fragmented": 20,
            "phase_C_LS2_all_masks": 60,
            "phase_C_total_missing": 80,
            "confirmation": "Requires frozen settings and new seeds after development, not launched.",
            "LISA": "Conditional on controlled bias and calibration; not launched.",
        },
    }
    bench.write_json(ROOT / "inventory.json", payload)
    return payload


def mp_rows(observed):
    raw = bench.tang_moving_periodogram(
        np.zeros(bench.N), m=bench.M, thin=bench.THIN
    )
    centres = np.rint(raw["u"] * bench.N).astype(int) - 1
    keep = gap.dw_keep(observed)
    centres, omega = centres[keep], raw["omega"][keep]
    j = np.arange(2 * bench.M + 1)
    values = np.exp(-1j * omega[:, None] * j) / np.sqrt(len(j))
    rows = sparse.csr_matrix(
        (
            values.ravel(),
            (
                np.repeat(np.arange(len(centres)), len(j)),
                (centres[:, None] - bench.M + j).ravel(),
            ),
        ),
        shape=(len(centres), bench.N),
    )
    return rows


def gaussian_dependence(rows, diagonal, lower):
    """All potentially nonzero pairs, including disjoint adjacent MA windows."""
    covariance = sparse.diags(
        (lower[1:], diagonal, lower[1:]),
        (-1, 0, 1),
        shape=(len(diagonal), len(diagonal)),
        format="csr",
    )
    projected = rows @ covariance
    k = projected @ rows.conjugate().T
    p = projected @ rows.T
    marginal, pseudo = k.diagonal().real, p.diagonal()
    var_power = marginal**2 + abs(pseudo) ** 2
    power_cov = abs(k).power(2) + abs(p).power(2)
    scaling = sparse.diags(1 / np.sqrt(var_power))
    corr = (scaling @ power_cov @ scaling).tocsr()
    corr.setdiag(0)
    corr.eliminate_zeros()
    return {
        "rows": rows.shape[0],
        "unordered_potential_pairs": int(corr.nnz // 2),
        "max_power_correlation": float(corr.max()),
        "max_nonproperness": float(np.max(abs(pseudo) / marginal)),
        "max_sum_off_diagonal_power_correlations": float(
            np.max(np.asarray(corr.sum(axis=1)))
        ),
        "scope": "exhaustive sparse covariance projections; zero covariances outside covariance-connected supports",
    }


def oracle_basis(u, f, truth):
    """Unpenalized least squares; simulation truth used only in this diagnostic."""
    master = bench.master_model()
    bt, bf = master.time.design_at(u), master.frequency.design_at(f)
    coefficients = np.linalg.lstsq(bt, truth, rcond=None)[0] @ np.linalg.pinv(
        bf.T
    )
    error = bt @ coefficients @ bf.T - truth
    return {
        "rms_log_error": float(np.sqrt(np.mean(error**2))),
        "p95_absolute_log_error": float(np.quantile(abs(error), 0.95)),
        "max_absolute_log_error": float(np.max(abs(error))),
        "coefficient_shape": list(coefficients.shape),
    }, error


def band_weights(f, bounds):
    in_band = (f >= bounds[0]) & (f <= bounds[1])
    weights = np.zeros(len(f))
    weights[in_band] = axis_weights(f[in_band])
    return weights


def implementation_audit(mask_set):
    x = bench.simulate(6101)
    raw = bench.tang_moving_periodogram(x, m=bench.M, thin=bench.THIN)
    full_data = bench.prepare(x, "dw")
    coeff_checks, mask_info, mean_arrays = {}, {}, {}
    stationary = {}
    for name, observed in mask_set.items():
        keep = gap.dw_keep(observed)
        data = gap.dw_data(x, observed)
        np.testing.assert_array_equal(data.power, full_data.power[keep])
        for field in ("time", "frequency", "counts"):
            np.testing.assert_array_equal(
                getattr(data, field), getattr(full_data, field)[keep]
            )
        centres = np.rint(data.time * bench.N).astype(int)
        assert all(
            observed[c - bench.M : c + bench.M + 1].all() for c in centres
        )
        for value in (np.nan, 1e50, -1e50):
            hidden = np.where(observed, x, value)
            changed = bench.tang_moving_periodogram(
                np.where(observed, hidden, 0.0), m=bench.M, thin=bench.THIN
            )
            np.testing.assert_array_equal(
                changed["coeff"][keep], raw["coeff"][keep]
            )
            np.testing.assert_array_equal(
                gap.dw_data(hidden, observed).power, data.power
            )
        windows = np.lib.stride_tricks.sliding_window_view(x, 2 * bench.M + 1)[
            centres - bench.M
        ]
        _, reference = periodogram(
            windows,
            fs=1 / bench.DT,
            window="boxcar",
            detrend=False,
            scaling="density",
            axis=-1,
        )
        rungs = np.rint(data.frequency * (2 * bench.M + 1) * bench.DT).astype(
            int
        )
        err = float(
            np.max(
                abs(
                    data.power / data.counts
                    - reference[np.arange(len(rungs)), rungs]
                )
            )
        )
        assert err < 1e-10 and np.all(data.counts == 2)
        rho, broadband_full, broadband_masked = gap.kernel_expectations(
            observed
        )
        w = gap.wdm_data(x, observed, rho)
        for value in (np.nan, 1e50, -1e50):
            np.testing.assert_array_equal(
                w.power,
                gap.wdm_data(
                    np.where(observed, x, value), observed, rho
                ).power,
            )
        retained = gap.wdm_keep(rho)
        kept_times = w.time[retained.all(axis=1)]
        edges_u = endpoints(observed).ravel() / bench.N
        edge_radius = max(
            (np.min(abs(kept_times - e)) for e in edges_u), default=0.03
        )
        assert np.all(w.counts[retained] == 1)
        np.savez_compressed(
            ROOT / f"mask_{name}.npz", observed=observed, rho=rho
        )
        intervals = endpoints(observed)
        mask_info[name] = {
            "intervals_half_open_samples": intervals.tolist(),
            "missing_samples": int((~observed).sum()),
            "mask_sha256": bench.digest(observed),
            "mp_complex_ordinates": int(keep.sum()),
            "mp_real_components": int(2 * keep.sum()),
            "wdm_real_coefficients": int(retained.sum()),
            "wdm_retained_time_columns": int(retained.all(axis=1).sum()),
        }
        coeff_checks[name] = {
            "complex_and_power_invariance": True,
            "fixed_physical_coordinates": True,
            "scipy_density_max_absolute_error": err,
            "wdm_hidden_values_invariance": True,
        }
        simulator = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=8200)
        b = simulator.coefficient(simulator.rescaled_time)
        mw = coloured.expected_wdm(b, observed, rho)
        mu = coloured.expected_dw(
            np.ones(2 * bench.M + 1), data.time, data.frequency, b
        )
        vv = np.convolve(
            bench.variance(np.arange(bench.N) / bench.N),
            np.ones(2 * bench.M + 1) / (2 * bench.M + 1),
            "valid",
        )[centres - bench.M]
        wu, wf = np.meshgrid(w.time, w.frequency, indexing="ij")
        for generator, method, uu, ff, predicted, valid in (
            (
                "broadband",
                "dw",
                data.time,
                data.frequency,
                2 * bench.DT * vv,
                np.ones(len(vv), dtype=bool),
            ),
            (
                "broadband",
                "wdm",
                wu,
                wf,
                2
                * bench.DT
                * np.divide(
                    broadband_masked,
                    rho,
                    out=np.zeros_like(rho),
                    where=retained,
                ),
                retained,
            ),
            (
                "LS2",
                "dw",
                data.time,
                data.frequency,
                mu,
                np.ones(len(mu), dtype=bool),
            ),
            ("LS2", "wdm", wu, wf, mw, retained),
        ):
            target = (
                2 * bench.DT * bench.variance(uu)
                if generator == "broadband"
                else 2
                * bench.DT
                * (
                    1
                    + simulator.coefficient(uu) ** 2
                    + 2
                    * simulator.coefficient(uu)
                    * np.cos(2 * np.pi * ff * bench.DT)
                )
            )
            valid = (
                valid
                & (uu >= 0.05)
                & (uu <= 0.95)
                & (ff >= 0.05)
                & (ff <= 0.45)
            )
            delta = np.log(predicted[valid]) - np.log(target[valid])
            edges = intervals.ravel() / bench.N
            distance = (
                np.min(abs(uu[valid, None] - edges), axis=1)
                if len(edges)
                else np.full(len(delta), np.inf)
            )
            label = f"{generator}_{name}_{method}"
            near = distance <= edge_radius + 1e-12
            mean_arrays[label] = {
                "rms": float(np.sqrt(np.mean(delta**2))),
                "p95_absolute": float(np.quantile(abs(delta), 0.95)),
                "signed_mean": float(delta.mean()),
                "max_absolute": float(np.max(abs(delta))),
                "near_edge_rms": float(np.sqrt(np.mean(delta[near] ** 2)))
                if np.any(near)
                else None,
                "near_edge_radius_rescaled_time": edge_radius,
            }
            np.savez_compressed(
                ROOT / f"expected_{label}.npz",
                time=uu[valid],
                freq=ff[valid],
                delta=delta,
            )
        for b0 in (0.0, 0.7):
            constant = np.full(bench.N, b0)
            actual = coloured.expected_dw(
                np.ones(2 * bench.M + 1), data.time, data.frequency, constant
            )
            expected = (
                2
                * bench.DT
                * (
                    1
                    + b0**2
                    + 2
                    * b0
                    * np.cos(2 * np.pi * data.frequency * bench.DT)
                    * (2 * bench.M)
                    / (2 * bench.M + 1)
                )
            )
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            stationary[f"{name}_b{b0}"] = float(np.max(abs(actual - expected)))
            wdm_mu = coloured.expected_wdm(constant, observed, rho)
            errors = []
            for t, frequency in (
                (1, 1),
                (20, 8),
                (80, 16),
                (180, 24),
                (254, 31),
            ):
                if not retained[t - 1, frequency - 1]:
                    continue
                row = (
                    bench.wdm_row(t, frequency)
                    * observed
                    / np.sqrt(rho[t - 1, frequency - 1])
                )
                explicit = (
                    2
                    * bench.DT
                    * (
                        (1 + b0**2) * (row @ row)
                        + 2 * b0 * np.sum(row[1:] * row[:-1])
                    )
                )
                errors.append(abs(wdm_mu[t - 1, frequency - 1] - explicit))
            assert errors and max(errors) < 1e-10
            if b0 == 0:
                np.testing.assert_allclose(
                    wdm_mu[retained], 2 * bench.DT, atol=1e-12
                )
            stationary[f"{name}_WDM_b{b0}"] = float(max(errors))
    # Sensitivity uses deterministic mask translations and two distinct profiles.
    sensitivity = []
    single_intervals = endpoints(mask_set["single"])[0]
    for shift in (-bench.N // 20, 0, bench.N // 20):
        observed = np.ones(bench.N, dtype=bool)
        lo, hi = single_intervals + shift
        observed[lo:hi] = False
        data = gap.dw_data(np.zeros(bench.N), observed)
        centres = np.rint(data.time * bench.N).astype(int)
        for reversed_trend in (False, True):
            variance = bench.variance(np.arange(bench.N) / bench.N)
            if reversed_trend:
                variance = bench.variance(1 - np.arange(bench.N) / bench.N)
            mean = np.convolve(variance, np.ones(65) / 65, "valid")[
                centres - bench.M
            ]
            sensitivity.append(
                {
                    "shift_samples": int(shift),
                    "reversed_trend": reversed_trend,
                    "mask_sha256": bench.digest(observed),
                    "rms_mean_log_mismatch": float(
                        np.sqrt(np.mean(np.log(mean / variance[centres]) ** 2))
                    ),
                }
            )
    simulator = LS2Data(n_samples=bench.N, fs=1 / bench.DT, seed=8200)
    b = simulator.coefficient(simulator.rescaled_time)
    rows = mp_rows(mask_set["single"])
    dependence = {
        "broadband_MP": gaussian_dependence(
            rows,
            bench.variance(np.arange(bench.N) / bench.N),
            np.zeros(bench.N),
        ),
        "LS2_MP": gaussian_dependence(rows, 1 + b**2, b),
    }
    data = gap.dw_data(np.zeros(bench.N), mask_set["single"])
    scored = (
        (data.time >= 0.05)
        & (data.time <= 0.95)
        & (data.frequency >= 0.05)
        & (data.frequency <= 0.45)
    )
    dependence["LS2_MP_primary_domain"] = gaussian_dependence(
        rows[scored], 1 + b**2, b
    )
    # WDM's existing probes are retained with an explicit limit, not called exhaustive.
    dependence["WDM"] = {
        "broadband_existing_probe_audit": read(gap.ROOT / "audit.json")[
            "cases"
        ]["contiguous"]["dependence_probes"]["wdm"],
        "coloured_all_pairs": "unresolved; required before posterior-calibration claims",
    }
    u, f = np.linspace(0, 1, 401), np.linspace(0.025, 0.475, 241)
    truth_broadband = np.broadcast_to(
        np.log(2 * bench.DT * bench.variance(u))[:, None], (len(u), len(f))
    )
    bb = simulator.coefficient(u[:, None])
    truth_ls2 = np.log(
        2
        * bench.DT
        * (1 + bb**2 + 2 * bb * np.cos(2 * np.pi * f[None, :] * bench.DT))
    )
    basis, errors = {}, {}
    for name, truth in (("broadband", truth_broadband), ("LS2", truth_ls2)):
        basis[name], errors[name] = oracle_basis(u, f, truth)
        ut, ft = (u >= 0.05) & (u <= 0.95), (f >= 0.05) & (f <= 0.45)
        basis[name]["primary_domain"], _ = oracle_basis(
            u[ut], f[ft], truth[ut][:, ft]
        )
    np.savez_compressed(ROOT / "oracle_basis.npz", time=u, freq=f, **errors)
    legacy = gap.masks()["fragmented"]
    payload = {
        "normalization_and_mask_checks": coeff_checks,
        "masks": mask_info,
        "stationary_MP_limits": stationary,
        "expected_mean": mean_arrays,
        "MP_all_pair_dependence": dependence,
        "oracle_basis": basis,
        "mask_trend_sensitivity": sensitivity,
        "legacy_fragmented_missing_samples": int((~legacy).sum()),
        "legacy_fragmented_intervals": endpoints(legacy).tolist(),
        "dt_seconds": bench.DT,
        "basis_shape_time_frequency": [12, 6],
        "baseline_window_duration_seconds": BASELINE_DURATION,
    }
    bench.write_json(ROOT / "implementation_audit.json", payload)
    return payload


def reconstruct(path, u, f):
    result = PSDResult.from_netcdf(path / "posterior.nc")
    saved = np.load(path / "surface.npz")
    master = bench.master_model()
    weights = result.posterior.weights.values.reshape(-1, 12, 6)
    assert weights.dtype == np.float64 and np.isfinite(weights).all()
    # Check archived quantiles by independent per-draw matrix products.
    max_error = 0.0
    for ti, fi in ((0, 4), (20, 28), (30, 28), (40, 52), (60, 52)):
        bt = master.time.design_at(saved["time"][[ti]])[0]
        bf = master.frequency.design_at(saved["freq"][[fi]])[0]
        draws = np.asarray([bt @ w @ bf for w in weights])
        actual = np.quantile(draws, [0.05, 0.5, 0.95])
        expected = np.array(
            [saved[k][ti, fi] for k in ("low", "median", "high")]
        )
        np.testing.assert_allclose(actual, expected, atol=3e-12)
        max_error = max(max_error, float(np.max(abs(actual - expected))))
    bt, bf = master.time.design_at(u), master.frequency.design_at(f)
    cache_dir = ROOT / "reconstructed"
    cache_dir.mkdir(exist_ok=True)
    cache_file = (
        cache_dir
        / f"{file_hash(path / 'posterior.nc')}_{bench.digest(np.r_[len(u), u, len(f), f])}.npz"
    )
    if cache_file.exists():
        cached = np.load(cache_file)
        np.testing.assert_array_equal(cached["time"], u)
        np.testing.assert_array_equal(cached["freq"], f)
        return (
            cached["quantiles"],
            cached["band_quantiles"],
            {
                "quantile_check_max_error": max_error,
                "draws": len(weights),
                "path": str(path.relative_to(bench.ROOT)),
            },
        )
    quantiles = np.empty((3, len(u), len(f)))
    band_quantiles = np.empty((3, len(u), len(BANDS)))
    for start in range(0, len(u), 16):
        logs = np.einsum(
            "ti,dij,fj->dtf",
            bt[start : start + 16],
            weights,
            bf,
            optimize=True,
        )
        quantiles[:, start : start + 16] = np.quantile(
            logs, [0.05, 0.5, 0.95], axis=0
        )
        for j, (lo, hi) in enumerate(BANDS.values()):
            keep = (f >= lo) & (f <= hi)
            power = np.trapezoid(np.exp(logs[:, :, keep]), f[keep], axis=-1)
            band_quantiles[:, start : start + 16, j] = np.quantile(
                power, [0.05, 0.5, 0.95], axis=0
            )
    np.savez_compressed(
        cache_file,
        time=u,
        freq=f,
        quantiles=quantiles,
        band_quantiles=band_quantiles,
    )
    return (
        quantiles,
        band_quantiles,
        {
            "quantile_check_max_error": max_error,
            "draws": len(weights),
            "path": str(path.relative_to(bench.ROOT)),
        },
    )


def resample_means(vectors, indices):
    return np.asarray(vectors)[indices].mean(axis=1)


def simultaneous_intervals(vectors, indices):
    """Exploratory max-standardized-error bootstrap across whole record vectors."""
    vectors = np.asarray(vectors)
    mean = vectors.mean(axis=0)
    se = vectors.std(axis=0, ddof=1) / np.sqrt(len(vectors))
    boots = resample_means(vectors, indices)
    statistic = np.max(
        np.divide(
            abs(boots - mean), se, out=np.zeros_like(boots), where=se > 0
        ),
        axis=1,
    )
    critical = np.quantile(statistic, 0.95)
    return mean, mean - critical * se, mean + critical * se, float(critical)


def wilson(count, n, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    p = np.asarray(count) / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def paired_analysis(audit, inv):
    intervals = (
        np.asarray(audit["masks"]["single"]["intervals_half_open_samples"])
        / bench.N
    )
    lo, hi = intervals[0]
    # Geometry-only edge radius comes from the current WDM exclusion extent.
    mask = np.load(ROOT / "mask_single.npz")
    retained_t = (np.arange(1, bench.NT - 1) / bench.NT)[
        gap.wdm_keep(mask["rho"]).all(axis=1)
    ]
    radius = max(
        lo - retained_t[retained_t < lo].max(),
        retained_t[retained_t >= hi].min() - hi,
    )
    anchor_u = np.array(
        [
            0.2,
            lo - BASELINE_DURATION / (2 * bench.N * bench.DT),
            hi + BASELINE_DURATION / (2 * bench.N * bench.DT),
            0.8,
        ]
    )
    u = np.unique(np.r_[np.linspace(0.05, 0.95, 241), anchor_u])
    f = np.unique(
        np.r_[np.linspace(0.05, 0.45, 57), 0.10, 0.15, 0.25, 0.30, 0.40]
    )
    truth = np.broadcast_to(
        np.log(2 * bench.DT * bench.variance(u))[:, None], (len(u), len(f))
    )
    band_truth = np.column_stack(
        [
            2 * bench.DT * bench.variance(u) * (hi - lo)
            for lo, hi in BANDS.values()
        ]
    )
    region_masks = regions(u, mask["observed"], radius)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(SEEDS), size=(BOOTSTRAPS, len(SEEDS))
    )
    q, bq, checks, cache = {}, {}, [], {}
    for method in ("dw", "wdm"):
        for condition in ("complete", "single", "complete_repaired"):
            values, bands = [], []
            for seed in SEEDS:
                path = selected_path(
                    seed,
                    method,
                    "complete"
                    if condition == "complete_repaired"
                    else condition,
                    condition == "complete_repaired",
                )
                metrics = read(path / "metrics.json")
                assert metrics["raw_sha256"] == bench.digest(
                    bench.simulate(seed)
                )
                assert (
                    metrics["model_hashes"]
                    == read(
                        selected_path(seed, method, "single") / "metrics.json"
                    )["model_hashes"]
                )
                assert config_target(metrics["config"]) == config_target(
                    read(
                        selected_path(seed, method, "single") / "metrics.json"
                    )["config"]
                )
                key = str(path)
                if key not in cache:
                    cache[key] = reconstruct(path, u, f)
                    checks.append(cache[key][2])
                result, band, _ = cache[key]
                values.append(result)
                bands.append(band)
            q[method, condition], bq[method, condition] = (
                np.asarray(values),
                np.asarray(bands),
            )
    rows, summary_vectors = [], []
    for method in ("dw", "wdm"):
        for kind in ("complete", "single", "added", "added_repaired"):
            if kind.startswith("added"):
                reference = (
                    "complete_repaired"
                    if kind == "added_repaired"
                    else "complete"
                )
                errors = q[method, "single"][:, 1] - q[method, reference][:, 1]
                fractional = (
                    np.exp(q[method, "single"][:, 1])
                    - np.exp(q[method, reference][:, 1])
                ) / np.exp(truth)
                band_error = (
                    bq[method, "single"][:, 1] - bq[method, reference][:, 1]
                ) / band_truth
            else:
                errors = q[method, kind][:, 1] - truth
                fractional = np.expm1(errors)
                band_error = bq[method, kind][:, 1] / band_truth - 1
            for region, rmask in region_masks.items():
                for bi, (band, (flo, fhi)) in enumerate(BANDS.items()):
                    in_band = (f >= flo) & (f <= fhi)
                    fw = band_weights(f, (flo, fhi))
                    weights = np.outer(axis_weights(u) * rmask, fw)
                    weights /= weights.sum()
                    tw = axis_weights(u) * rmask
                    tw /= tw.sum()
                    vector = np.sum(errors * weights, axis=(1, 2))
                    linear_vector = np.sum(fractional * weights, axis=(1, 2))
                    band_vector = np.sum(band_error[:, :, bi] * tw, axis=1)
                    row = {
                        "method": method,
                        "kind": kind,
                        "region": region,
                        "frequency_band": band,
                        "mean_log_bias": float(vector.mean()),
                        "mean_fractional_bias": float(linear_vector.mean()),
                        "mean_band_power_fractional_bias": float(
                            band_vector.mean()
                        ),
                        "record_log_sd": float(vector.std(ddof=1)),
                        "n_records": len(SEEDS),
                    }
                    row["rmse_record_log_error"] = float(
                        np.sqrt(
                            np.mean(np.sum(errors**2 * weights, axis=(1, 2)))
                        )
                    )
                    row["rms_across_record_bias_map"] = float(
                        np.sqrt(np.sum(errors.mean(axis=0) ** 2 * weights))
                    )
                    row["p95_absolute_across_record_bias_map"] = float(
                        np.quantile(
                            abs(errors.mean(axis=0)[rmask][:, in_band]), 0.95
                        )
                    )
                    if kind in ("complete", "single"):
                        intervals = (
                            q[method, kind][:, 2] - q[method, kind][:, 0]
                        )
                        row["mean_log_credible_interval_width"] = float(
                            np.mean(np.sum(intervals * weights, axis=(1, 2)))
                        )
                    else:
                        row["mean_log_credible_interval_width"] = None
                    for label, vv in (
                        ("log", vector),
                        ("fractional", linear_vector),
                        ("band_power_fractional", band_vector),
                    ):
                        bounds = np.quantile(
                            resample_means(vv, indices), [0.025, 0.975]
                        )
                        (
                            row[f"{label}_pointwise_bootstrap_low"],
                            row[f"{label}_pointwise_bootstrap_high"],
                        ) = map(float, bounds)
                    rows.append(row)
                    summary_vectors.append(vector)
    family = np.column_stack(summary_vectors)
    np.savez_compressed(
        ROOT / "record_bias_vectors.npz",
        seeds=np.asarray(SEEDS),
        vectors=family,
    )
    _, lower, upper, critical = simultaneous_intervals(family, indices)
    for i, row in enumerate(rows):
        row.update(
            {
                "log_simultaneous_bootstrap_low": float(lower[i]),
                "log_simultaneous_bootstrap_high": float(upper[i]),
                "diagnostic_equivalence_0.02": "inside"
                if lower[i] > -0.02 and upper[i] < 0.02
                else "outside"
                if lower[i] > 0.02 or upper[i] < -0.02
                else "inconclusive",
                "formal_acceptance": "not_applicable_development_posthoc_summaries",
            }
        )
    anchor_t = np.searchsorted(u, anchor_u)
    anchor_f = np.searchsorted(f, [0.1, 0.25, 0.4])
    coverage = []
    for method in ("dw", "wdm"):
        for condition in ("complete", "single", "complete_repaired"):
            z = q[method, condition]
            included = (z[:, 0] <= truth) & (truth <= z[:, 2])
            for ti in anchor_t:
                for fi in anchor_f:
                    count = int(included[:, ti, fi].sum())
                    low_ci, high_ci = wilson(count, len(SEEDS))
                    coverage.append(
                        {
                            "method": method,
                            "condition": condition,
                            "time": float(u[ti]),
                            "frequency_hz": float(f[fi]),
                            "records_included": count,
                            "records": len(SEEDS),
                            "coverage": count / len(SEEDS),
                            "wilson95_low": float(low_ci),
                            "wilson95_high": float(high_ci),
                            "mean_log_interval_width": float(
                                np.mean(z[:, 2, ti, fi] - z[:, 0, ti, fi])
                            ),
                            "point_estimate_scatter_log_sd": float(
                                np.std(z[:, 1, ti, fi], ddof=1)
                            ),
                        }
                    )
    maps = {"time": u, "freq": f, "truth": truth}
    distance_curves = {}
    for method in ("dw", "wdm"):
        for condition in ("complete", "single", "complete_repaired"):
            maps[f"{method}_{condition}_quantiles"] = q[method, condition]
            maps[f"{method}_{condition}_band_quantiles"] = bq[
                method, condition
            ]
        for kind, values in (
            ("complete", q[method, "complete"][:, 1] - truth),
            ("single", q[method, "single"][:, 1] - truth),
            ("added", q[method, "single"][:, 1] - q[method, "complete"][:, 1]),
        ):
            maps[f"{method}_{kind}_bias"] = values.mean(axis=0)
            maps[f"{method}_{kind}_fractional_bias"] = (
                np.expm1(values).mean(axis=0)
                if kind != "added"
                else (
                    (
                        np.exp(q[method, "single"][:, 1])
                        - np.exp(q[method, "complete"][:, 1])
                    )
                    / np.exp(truth)
                ).mean(axis=0)
            )
            curves = values @ (axis_weights(f) / axis_weights(f).sum())
            boots = resample_means(curves, indices)
            distance_curves[f"{method}_{kind}"] = (
                curves.mean(axis=0),
                np.quantile(boots, [0.025, 0.975], axis=0),
            )
    np.savez_compressed(ROOT / "paired_surfaces.npz", **maps)
    np.savez_compressed(
        ROOT / "bootstrap_indices.npz", seed=BOOTSTRAP_SEED, indices=indices
    )
    with (ROOT / "regional_bias.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    family_multiplier = float(norm.ppf(1 - 0.05 / (2 * 2 * 3 * 3 * 3)))
    planning = {
        "desired_half_width_log": 0.01,
        "normal_Bonferroni_multiplier": family_multiplier,
        "largest_observed_primary_SD": max(
            r["record_log_sd"]
            for r in rows
            if r["region"] != "gap_interior"
            and r["kind"] in ("single", "added")
        ),
        "caveat": "Scatter-based planning only; no sample size removes fixed systematic bias.",
    }
    planning["suggested_records_for_bias_precision"] = max(
        100,
        int(
            np.ceil(
                (
                    family_multiplier
                    * planning["largest_observed_primary_SD"]
                    / 0.01
                )
                ** 2
            )
        ),
    )
    for count in range(100, 10001):
        low_ci, high_ci = wilson(round(0.9 * count), count, alpha=0.05 / 48)
        if low_ci >= 0.85 and high_ci <= 0.95:
            planning["illustrative_records_for_48_anchor_coverage_bounds"] = (
                count
            )
            break
    quality = [
        {
            "seed": seed,
            "method": method,
            "mask": "complete",
            "original": read(
                selected_path(seed, method, "complete") / "metrics.json"
            )["diagnostics"],
            "longer_rerun": read(
                selected_path(seed, method, "complete", True) / "metrics.json"
            )["diagnostics"],
        }
        for seed in (6103, 6110)
        for method in ("dw", "wdm")
    ]
    payload = {
        "development_records": SEEDS,
        "primary_original_fits": 40,
        "distinct_posteriors_reloaded": len(checks),
        "posterior_checks": checks,
        "quality_ledger": quality,
        "region_radius_rescaled_time": radius,
        "region_radius_seconds": radius * bench.N * bench.DT,
        "grid_points_time_frequency": [len(u), len(f)],
        "band_definitions_hz": BANDS,
        "estimator": "median of log PSD at each site; band-power medians from integrating each draw",
        "truth": "latent pointwise one-sided PSD, not expected window power",
        "bootstrap": {
            "record_resamples": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "family_size": family.shape[1],
            "exploratory_max_standardized_error_critical": critical,
            "scope": "Same whole-record resample used for every method, mask, band and region.",
        },
        "regional_bias": rows,
        "fixed_site_coverage": coverage,
        "planning": planning,
        "judgment": "Development audit only; negligible bias and calibrated uncertainty are not established; no formal confirmation acceptance.",
        "original_sampler_flags_retained": [6103, 6110],
        "longer_complete_reruns_used_only_for_sensitivity": True,
    }
    bench.write_json(ROOT / "paired_summary.json", payload)
    figures(u, f, maps, distance_curves, lo, hi)
    return payload


def figures(u, f, maps, curves, lo, hi):
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6.5), layout="constrained", sharex=True, sharey=True
    )
    extent = max(
        float(np.max(abs(maps[f"{method}_{kind}_bias"])))
        for method in ("dw", "wdm")
        for kind in ("complete", "single", "added")
    )
    for i, method in enumerate(("dw", "wdm")):
        for j, kind in enumerate(("complete", "single", "added")):
            ax = axes[i, j]
            image = ax.pcolormesh(
                u,
                f,
                maps[f"{method}_{kind}_bias"].T,
                cmap="RdBu_r",
                vmin=-extent,
                vmax=extent,
                shading="auto",
            )
            for edge in (lo, hi):
                ax.axvline(edge, color="black", lw=0.8, ls="--")
            ax.set(
                title=f"{'MP' if method == 'dw' else 'Guarded WDM'}: {kind.replace('single', 'single gap')}"
            )
    for ax in axes[-1]:
        ax.set_xlabel("Physical rescaled time u")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(
        image,
        ax=axes,
        label="Across-record signed log-PSD bias (10 records)",
        shrink=0.8,
    )
    fig.suptitle(
        "Total bias and paired displacement; dashed lines bound the missing interval"
    )
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"signed_bias_maps.{extension}", dpi=160)
    plt.close(fig)
    edge_figure(u, curves, lo, hi)


def report(audit, summary, inventory):
    rows = summary["regional_bias"]
    lines = [
        "# Moving-periodogram gap bias: first-stage study results",
        "",
        "**Date:** 12 October 2026. **Stage:** the supplied plan's first stopping point.",
        "",
        "**Finding:** the existing broadband pilot supports smaller gap-induced displacement for MP than for the specified guarded WDM implementation. It establishes neither negligible absolute edge bias nor calibrated uncertainty. The coloured LS2 configuration needs a common-basis revision before new inference.",
        "",
        "## Paired broadband bias",
        "",
        "Ten independent raw records, seeds 6101–6110; complete and central 819-sample masks; original rectangular MP and the 99.99% retained-energy whole-column WDM gate. These development records are reused, not new confirmation records.",
        "",
        "Both methods use cubic splines with **12 time coefficients and 6 frequency coefficients**, resulting from 8 and 2 interior knots. The shared HalfNormal roughness scale is 10; null precision is 0.0001. Physical cadence is 1 second, verified from the runner and archived environment. MP retains exact paired sites, without pooling or restarting its frequency schedule.",
        "",
        "The common domain is u=0.05–0.95 and f=0.05–0.45 Hz. Each observed edge region extends 327 seconds from the gap, based on the larger distance to the first retained WDM centre. Both methods are scored in that entire region, including the observed neighbourhood excluded by WDM.",
        "",
        "The table averages the posterior-median log errors over each observed edge region and the full frequency band. Added bias uses the paired single-gap minus complete estimate at identical sites.",
        "",
        "| Method | Left total log bias | Right total log bias | Left added log bias | Right added log bias |",
        "|---|---:|---:|---:|---:|",
    ]
    widths = {band: hi - lo for band, (lo, hi) in BANDS.items()}
    for method, label in (
        ("dw", "MP, original rectangular"),
        ("wdm", "WDM, whole-column guard"),
    ):
        values = []
        for kind in ("single", "added"):
            for region in ("left_observed", "right_observed"):
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["kind"] == kind
                    and r["region"] == region
                ]
                values.append(
                    sum(
                        r["mean_log_bias"] * widths[r["frequency_band"]] / 0.4
                        for r in selected
                    )
                )
        lines.append(
            f"| {label} | " + " | ".join(f"{v:+.4f}" for v in values) + " |"
        )
    lines.extend(
        [
            "",
            "- The proposed ±0.02 equivalence criterion is **not established** in either observed edge region. Bounds are too wide; failure to establish equivalence is not proof that every target violates the tolerance.",
            "- For example, left-edge middle-band total log bias is about -0.048 for MP and -0.099 for guarded WDM. Exploratory simultaneous bootstrap bounds are approximately [-0.106,+0.010] and [-0.170,-0.028], respectively. These are development diagnostics, not preregistered confirmation tests.",
            "- Complete-data bias is evaluated explicitly. A small added displacement cannot repair an already biased complete-data estimate.",
            "- Fractional PSD errors use the same point estimator. Band powers are integrated for every posterior draw before taking their medians. The [regional table](regional_bias.csv) also includes RMSE, RMS and upper-tail absolute bias-map summaries, interval widths and confidence bounds.",
            "- Fixed-site truth inclusion ranges from 8/10 to 10/10 across the chosen anchors. Even 9/10 has a 95% Wilson interval of approximately [0.60,0.98]; these records cannot establish coverage within [0.85,0.95]. Pixel coverage is not substituted for repeated-record coverage.",
            "",
            "![Bias and paired displacement versus observed edge distance](bias_by_edge_distance.png)",
            "",
            "The shaded intervals above are pointwise 95% Monte Carlo record-bootstrap intervals for bias. Green marks the proposed tolerance. They are different from posterior credible intervals and do not provide uniform coverage over the whole curve.",
            "",
            "![Signed bias surfaces](signed_bias_maps.png)",
            "",
            "Gap-interior maps describe model interpolation. No gap-interior recovery claim follows from observed-edge results. The pre-existing seed-6101 explanatory figure is retained in the [earlier study](../taper_study/REPORT.md).",
            "",
            "## Implementation and observation-model audit",
            "",
            "- Retained MP complex coefficients and powers are unchanged by hidden NaNs and ±1e50 substitutions and equal the complete-record quantities. All retained windows contain only observed samples; the original physical times and frequency schedule are preserved.",
            "- Independent SciPy periodograms confirm one-sided rectangular normalization. Complex MP observations use `power=2*I`, `count=2`; WDM observations have one real degree of freedom, their actual masked kernels and retained-energy correction. No coefficient count is interpreted as independent information.",
            "- White and stationary MA(1) expected means were checked against analytic MP formulas and explicit WDM covariance projections. LS2's simulator uses N+1 innovations, so the first sample has variance 1+b(0)^2 and innovations do not wrap around record boundaries.",
            "",
            "| Generator, single gap | MP RMS log mean mismatch | WDM RMS log mean mismatch |",
            "|---|---:|---:|",
            f"| Broadband | {audit['expected_mean']['broadband_single_dw']['rms']:.6f} | {audit['expected_mean']['broadband_single_wdm']['rms']:.6f} |",
            f"| Coloured LS2 | {audit['expected_mean']['LS2_single_dw']['rms']:.5f} | {audit['expected_mean']['LS2_single_wdm']['rms']:.5f} |",
            "",
            "These are exact observation-mean discrepancies against latent pointwise PSD, not fitted posterior biases. Complete, single and four-gap masks are separately audited; record-boundary effects are kept outside the scored interior.",
            "",
            f"- Broadband basis approximation RMS is {audit['oracle_basis']['broadband']['primary_domain']['rms_log_error']:.6f} on the primary domain. Coloured LS2's minimum unpenalized grid RMS in the frozen basis is **{audit['oracle_basis']['LS2']['primary_domain']['rms_log_error']:.5f}**, already larger than 0.02. Truth is used only in this diagnostic, never as a production initializer or correction.",
            "- Exhaustive MP Gaussian covariance projections include disjoint windows connected by the MA(1) lag. In the scored LS2 domain, maximum power correlation is 0.00903, maximum sum of off-diagonal correlations is 0.1854 and maximum nonproperness is 0.1700. Over the complete retained frequency schedule they rise to 0.1666 and 0.8935 for correlation and nonproperness. The likelihood also uses those endpoint-adjacent observations outside the scoring band. Mean correction alone would not establish an exponential marginal or independence.",
            "- WDM's archived broadband neighbour probes are verified provenance, **not an exhaustive coloured covariance audit**. That audit remains required. Smooth gap shifts and amplitude-trend reversal were checked separately for MP expected means; no opposite trends are pooled into a bias average.",
            "",
            "## Provenance, masks and sampler ledger",
            "",
            f"- [Inventory](inventory.json): {len(inventory['runs'])} archived attempts, with input/model/configuration checks and file hashes. Forty primary fits are paired; {summary['distinct_posteriors_reloaded']} distinct posteriors were reloaded including four longer complete-data sensitivity fits. Independent per-draw checks reproduce saved quantiles to below 3e-12.",
            "- Complete-data WDM seeds 6103 and 6110 failed the original uniform screen. They stay in the original ten-record summary. Longer paired MP/WDM reruns passed; the separate sensitivity changes added regional log bias by less than 0.001. No seed is silently dropped or replaced.",
            "- Current production-source hashes match the archived study, and the archived optimized complete-data likelihood source matches the current likelihood. Per-fit observation-array hashes were not originally saved; adapters, masks, raw-record hashes and model targets are checked now. Runtime claims are not re-estimated.",
            "- The new four-gap mask is `[1536,1741)`, `[3175,3380)`, `[4813,5018)`, `[6452,6656)`, totaling 819 missing samples. The legacy fragmented mask has sixteen gaps totaling 816 samples and is incompatible with the requested mask.",
            "- Single-gap MP retains 3,623 complex ordinates; WDM retains 6,479 real coefficients. Four-gap counts are 3,526 and 4,867. These describe retained observations only.",
            "",
            "## What new inference is required",
            "",
            "1. Complete the coloured WDM covariance audit and revise the **common** basis or narrow the coloured scope. Keep both methods on identical final basis/prior settings.",
            "2. Run Phase B's five-record exact variance reference: 30 fits. Its 1D model is separate from the 2D surface pilot.",
            "3. Add 20 broadband four-gap fits and 60 LS2 fits only after the basis/observation audit. With compatible existing complete/single broadband fits, Phase C currently lacks 80 fits; changing the common model would invalidate that reuse.",
            "4. Freeze a smaller justified primary family, linear/band-power tolerances and new confirmation seeds before confirmation. The present posthoc pilot summaries cannot be promoted to formal acceptance.",
            f"5. Precision planning from the largest pilot scatter gives roughly {summary['planning']['suggested_records_for_bias_precision']:,} records for a 0.01 simultaneous log-bias half-width over an illustrative 54-summary family. A separate illustrative 48-anchor coverage calculation needs about {summary['planning']['illustrative_records_for_48_anchor_coverage_bounds']} records. These are rough planning numbers, not a chosen final sample size or an instruction to launch that factorial. More records cannot remove fixed bias.",
            "",
            "The plan's first stopping point has been reached. Confirmation, exclusion/Fourier ablations and simulated LISA remain conditional on acceptable controlled bias and uncertainty. No new posterior fits, production likelihood changes or manuscript claims are introduced by this stage.",
            "",
            "The stationary scope of [The De-Biased Whittle Likelihood](https://arxiv.org/abs/1605.06718) does not validate the present nonstationary gapped estimator. [Gustafsson, Villani and Kohn](https://arxiv.org/abs/2411.14010) motivate the separate finite-sample posterior-reference comparison; the proposed variance-reference experiment has not yet been run.",
            "",
            "## Reproduce and inspect",
            "",
            "From the analysis repository, using the existing virtual environment:",
            "",
            "```bash",
            "export MPLCONFIGDIR=/tmp/psd-gap-bias-mpl",
            "export OPENBLAS_NUM_THREADS=1",
            "export JAX_ENABLE_X64=true",
            ".venv/bin/python -m experiments.dw_wdm_small.gap_bias_audit",
            ".venv/bin/python -m pytest experiments/dw_wdm_small/test_gap_bias_audit.py experiments/dw_wdm_small/test_gaps.py -q",
            "```",
            "",
            "- [Runner](../gap_bias_audit.py), [tests](../test_gap_bias_audit.py), [development manifest](protocol_manifest.json), [implementation audit](implementation_audit.json), [paired summary](paired_summary.json), [environment](environment.json).",
            "- `paired_surfaces.npz` contains each record's log-PSD and band-power quantiles on the common grid. `record_bias_vectors.npz` columns follow the regional-summary row order. `bootstrap_indices.npz` retains the exact paired record resamples. Full coefficient draws stay in the original NetCDF posteriors.",
            "- Cached reconstructions are identified by original posterior and grid hashes. The runner can be repeated without rerunning inference.",
            "",
            "**Verification:** 15 focused tests passed; stationary mean identities, mask/coordinate invariance and 44 posterior reload/quantile checks passed. Figures were opened and inspected. One existing JAX deprecation warning remains. Original source files and archived fits are preserved.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


def edge_figure(u, curves, lo, hi):
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True
    )
    for i, kind in enumerate(("single", "added")):
        for j, (edge, sign, title) in enumerate(
            ((lo, -1, "Before gap"), (hi, 1, "After gap"))
        ):
            ax = axes[i, j]
            distance = (
                sign * (u - edge) * bench.N * bench.DT / BASELINE_DURATION
            )
            keep = (distance >= 0) & (distance <= 12)
            order = np.argsort(distance[keep])
            for method, color, label in (
                ("dw", "#2866A2", "MP"),
                ("wdm", "#B46A1D", "Guarded WDM"),
            ):
                mean, ci = curves[f"{method}_{kind}"]
                xx = distance[keep][order]
                ax.plot(xx, mean[keep][order], color=color, label=label)
                ax.fill_between(
                    xx,
                    ci[0, keep][order],
                    ci[1, keep][order],
                    color=color,
                    alpha=0.18,
                )
            ax.axhspan(
                -0.02,
                0.02,
                color="#769c66",
                alpha=0.22,
                label="Proposed ±0.02 tolerance" if i == j == 0 else None,
            )
            ax.axhline(0, color="black", lw=0.6)
            ax.set(
                title=f"{title}: {'total bias' if kind == 'single' else 'added gap bias'}"
            )
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(
            "Observed distance from edge / 65-sample window duration"
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency-integrated signed log error")
    fig.suptitle(
        "Ten development records: shaded 95% pointwise record-bootstrap intervals\nMonte Carlo uncertainty in bias; these are not posterior credible bands"
    )
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"bias_by_edge_distance.{extension}", dpi=160)
    plt.close(fig)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    mask_set = masks()
    print("Inventorying existing fit attempts", flush=True)
    inv = inventory(mask_set)
    print(
        f"Found {len(inv['runs'])} attempts; auditing masks, exact means, dependence and spline representation",
        flush=True,
    )
    audit = implementation_audit(mask_set)
    print(
        "Reloading paired posterior draws and recomputing log/linear/band-power summaries",
        flush=True,
    )
    summary = paired_analysis(audit, inv)
    bench.write_json(
        ROOT / "environment.json",
        {
            "python": platform.python_version(),
            "versions": {
                p: version(p)
                for p in ("numpy", "scipy", "jax", "numpyro", "wdm_transform")
            },
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "runner_sha256": file_hash(__file__),
            "float64": True,
        },
    )
    bench.write_json(
        ROOT / "protocol_manifest.json",
        {
            "stage": "development_audit_first_stopping_point",
            "source_plan_sha256": file_hash(ROOT / "study_plan.md"),
            "masks": audit["masks"],
            "development_seeds": SEEDS,
            "basis_shape_time_frequency": [12, 6],
            "window_samples": 65,
            "wdm_nt": 256,
            "wdm_energy_retained": 0.9999,
            "dt_seconds": bench.DT,
            "baseline_window_duration_seconds": BASELINE_DURATION,
            "region_radius_seconds": summary["region_radius_seconds"],
            "evaluation_domain": {
                "u": [0.05, 0.95],
                "frequency_hz": [0.05, 0.45],
            },
            "estimator": summary["estimator"],
            "band_definitions_hz": BANDS,
            "epsilon_log_proposed": 0.02,
            "linear_and_band_power_tolerances": "not yet agreed; no acceptance evaluated",
            "confirmation_seeds": "not frozen until development configuration is resolved",
            "sampler_quality": "0 divergences; 0 depth hits; Rhat<=1.01; bulk/tail ESS>=200",
            "existing_target_config": read(
                selected_path(6101, "dw", "complete") / "metrics.json"
            )["config"],
            "posthoc_development_summaries": True,
        },
    )
    report(audit, summary, inv)
    print(
        json.dumps(
            {
                "oracle_basis": audit["oracle_basis"],
                "MP_dependence": audit["MP_all_pair_dependence"],
                "planning": summary["planning"],
                "judgment": summary["judgment"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
