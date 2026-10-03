"""Run stages A--C against shared prepared models, never a second fitter.

Each target runs in its own process to bound compilation/diagnostic memory.
Prior investigation targets are loaded from saved exact observations/bases.
Use --target to run one target; without it the CLI dispatches all targets.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from functools import partial
from inspect import getsource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpyro.distributions.transforms import biject_to
from numpyro.infer.initialization import init_to_uniform
from scipy.stats import kurtosis, skew

from log_psplines.basis import SplineBasis
from log_psplines.config import PowerConfig, StationaryConfig
from log_psplines.data.spectral import PowerData, WishartData
from log_psplines.diagnostics.stationarity import stationarity_loss
from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    compare_features,
    fingerprint,
    mmd_squared,
    nuts_reference_health,
    rebuild_guide,
    reference_transform,
    runtime_provenance,
)
from log_psplines.inference.model import (
    _blocked_channel_model,
    _sample_pspline_block,
    channel_model_kwargs,
    prepare_model,
)
from log_psplines.inference.nuts import run_nuts
from log_psplines.inference.power import (
    _collect_power_samples,
    prepare_power_model,
)
from log_psplines.inference.vi import fit_vi
from log_psplines.models.spectrum import LogPSpline
from log_psplines.results import PSDResult

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
DEFAULT_PREVIOUS = Path(
    "/Users/avi/Documents/projects/LogPSplinePSD/runs/dynamic-whittle-stages-0-3"
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def safe(x):
        if isinstance(x, dict):
            return {k: safe(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [safe(v) for v in x]
        if isinstance(x, np.ndarray):
            return safe(x.tolist())
        if isinstance(x, np.generic):
            return safe(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


@dataclass
class Target:
    model: object
    spline: object
    data: object
    config: object
    pair: object
    init: dict | None
    identity: str
    descriptor: dict


def basis():
    return LogPSpline(
        SplineBasis.from_grid(np.linspace(0, 0.5, 65), 2),
        time=SplineBasis.from_grid(np.linspace(0, 1, 129), 4),
    )


def build_target(name, cfg, previous, directory):
    rng = np.random.default_rng(cfg["exact_data_seed"])
    if name == "exact_stationary":
        freq = np.linspace(0.01, 0.49, 129)
        placeholder = WishartData(
            np.ones((129, 1, 1)), np.zeros((129, 1, 1)), freq, 129, 1, Nb=4
        )
        config = StationaryConfig(
            n_knots=6, verbose=False, roughness_scale=1.28
        )
        _, components = prepare_model(placeholder, config)
        spline = components.diagonal_models[0]
        coefficients = np.linspace(
            -2, -1, spline.basis.shape[1]
        ) + 0.15 * np.sin(np.arange(spline.basis.shape[1]))
        truth = np.exp(spline.basis @ coefficients)
        nu = np.full(len(freq), 8.0)
        powers = rng.gamma(nu / 2, 2 * truth)
        # Only sufficient statistics are encoded; these are not coefficient phases.
        data = WishartData(
            np.sqrt(powers[:, None, None] / 2),
            np.zeros((129, 1, 1)),
            freq,
            129,
            1,
            Nb=4,
        )
        kwargs, components = prepare_model(data, config)
        spline = components.diagonal_models[0]
        kwargs = channel_model_kwargs(kwargs, 0)
        model = partial(_blocked_channel_model, **kwargs)
        identity = fingerprint(
            kwargs,
            getsource(_blocked_channel_model),
            getsource(_sample_pspline_block),
        )
        np.savez_compressed(
            directory / "target.npz",
            frequency=freq,
            power=powers,
            counts=nu,
            truth=truth,
            truth_coefficients=coefficients,
            basis_frequency=spline.basis,
            penalty_frequency=spline.penalty_matrix,
        )
        return Target(
            model,
            spline,
            data,
            config,
            None,
            None,
            identity,
            {
                "tier": "exact_likelihood",
                "model": "stationary_univariate",
                "units": "coefficient variance",
                "generator": "P ~ Gamma(nu/2, 2*V); encoded Y=P/2; duration=1",
                "likelihood_scaling": "Nb=4, Nh=1, eta=1, ENBW=1",
                "truth_is_spline_surface": True,
            },
        )
    config = PowerConfig(progress_bar=False)
    spline = basis()
    if name == "exact_time_varying":
        time, freq = np.linspace(0, 1, 33), np.linspace(0.02, 0.48, 33)
        truth_coefficients = (
            -1.8
            + 0.35 * np.sin(np.linspace(0, 2 * np.pi, 8))[:, None]
            + 0.3 * np.cos(np.linspace(0, np.pi, 6))[None, :]
        )
        truth = np.exp(
            spline.time.design_at(time)
            @ truth_coefficients
            @ spline.frequency.design_at(freq).T
        )
        counts = np.full_like(truth, 2.0)
        data = PowerData(
            rng.gamma(counts / 2, 2 * truth),
            counts,
            freq,
            time,
            "coefficient variance",
        )
        # Inference is evaluated at this observation grid; same knots/prior.
        spline = LogPSpline(
            SplineBasis.from_grid(
                freq, interior_knots=spline.frequency.knots[4:-4]
            ),
            time=SplineBasis.from_grid(
                time, interior_knots=spline.time.knots[4:-4]
            ),
        )
        # Original surface must be generated from the intended inference basis.
        truth = np.exp(
            spline.time.basis @ truth_coefficients @ spline.frequency.basis.T
        )
        rng = np.random.default_rng(cfg["exact_data_seed"])
        data = PowerData(
            rng.gamma(counts / 2, 2 * truth),
            counts,
            freq,
            time,
            "coefficient variance",
        )
        descriptor = {
            "tier": "exact_likelihood",
            "model": "scalar_tensor_power",
            "units": data.units,
            "generator": "P ~ Gamma(nu/2, scale=2*V), independent cells",
            "truth_is_spline_surface": True,
        }
        np.savez_compressed(
            directory / "truth.npz",
            surface=truth,
            coefficients=truth_coefficients,
        )
    else:
        old_name = {
            "raw_ar1_m8": "cal_ar1_m8_nuts_none_init7101",
            "raw_slow_m32": "cal_slow_m32_nuts_none_init7101",
        }[name]
        source = previous / old_name / "result.nc"
        old = PSDResult.from_netcdf(source)
        observations = old.observed_data
        data = PowerData(
            observations["power"].values,
            observations["counts"].values,
            observations.frequency.values,
            observations.time.values,
            old.metadata["units"],
        )
        md = old.model_data
        spline = LogPSpline(
            SplineBasis.from_grid(
                md.frequency.values,
                interior_knots=md.knots_frequency.values[4:-4],
            ),
            time=SplineBasis.from_grid(
                md.time.values, interior_knots=md.knots_time.values[4:-4]
            ),
        )
        np.testing.assert_array_equal(
            spline.frequency.basis, md.basis_frequency.values
        )
        np.testing.assert_array_equal(spline.time.basis, md.basis_time.values)
        for field in (
            "roughness_scale",
            "null_precision",
            "ridge_eps",
            "centered",
        ):
            value = old.metadata[field]
            setattr(
                config,
                field,
                bool(value) if field == "centered" else float(value),
            )
        descriptor = {
            "tier": "raw_process_fixed_window",
            "source_result": str(source),
            "old_reference_status": "two-chain reference not reused as accepted reference",
            "units": data.units,
            "window_m": 8 if name == "raw_ar1_m8" else 32,
            "observations": "exact archived paired powers/counts and coordinates",
            "basis": "exact archived 8x6 cubic basis",
            "transform_dependence": "unresolved; outside algorithm comparison",
        }
    model, init, pair = prepare_power_model(data, spline, config)
    identity = fingerprint(
        data.power,
        data.counts,
        data.time,
        data.frequency,
        spline.time.basis,
        spline.frequency.basis,
        pair,
        {
            name: getattr(config, name)
            for name in (
                "roughness_scale",
                "null_precision",
                "ridge_eps",
                "centered",
            )
        },
        data.units,
        getsource(prepare_power_model),
    )
    np.savez_compressed(
        directory / "target.npz",
        power=data.power,
        counts=data.counts,
        time=data.time,
        frequency=data.frequency,
        basis_time=spline.time.basis,
        basis_frequency=spline.frequency.basis,
        penalty_time=spline.time.penalty,
        penalty_frequency=spline.frequency.penalty,
        knots_time=spline.time.knots,
        knots_frequency=spline.frequency.knots,
    )
    return Target(
        model, spline, data, config, pair, init, identity, descriptor
    )


def quadrature(grid):
    return np.r_[
        np.diff(grid)[0] / 2, (grid[2:] - grid[:-2]) / 2, np.diff(grid)[-1] / 2
    ]


def features(target, posterior):
    """All chain/draw coordinates enter compact physical features in chunks."""
    freq = np.linspace(0.08, 0.42, 33)
    if target.pair is None:
        weights = np.asarray(posterior["weights_delta_0"])
        bf = np.asarray(target.spline.basis)
        # Native frequency points declared before inference.
        indices = [23, 64, 104]
        logs = np.einsum("fk,cdk->cdf", bf, weights)
        selected = logs[..., indices]
        band = (target.data.freq >= 0.08) & (target.data.freq <= 0.42)
        power = np.einsum(
            "cdf,f->cd",
            np.exp(logs[..., band]),
            quadrature(target.data.freq[band]),
        )
        scales = np.log(np.asarray(posterior["sigma_delta_0"]))
        values = np.concatenate(
            [selected, power[..., None], scales[..., None]], axis=-1
        )
        names = [f"log_V_f{target.data.freq[i]:.5f}" for i in indices] + [
            "band_power",
            "log_sigma_delta_0",
        ]
        return values, names

    class View:
        pass

    view = View()
    view.posterior = posterior
    coefficients = _collect_power_samples(
        view, target.pair, target.config
    ).weights.values
    positions = [(0.25, 0.15), (0.5, 0.25), (0.75, 0.35)]
    names = [f"log_V_t{t}_f{f}" for t, f in positions]
    selected = np.stack(
        [
            np.einsum(
                "i,cdij,j->cd",
                target.spline.time.design_at(np.array([t]))[0],
                coefficients,
                target.spline.frequency.design_at(np.array([f]))[0],
            )
            for t, f in positions
        ],
        axis=-1,
    )
    bt = target.spline.time.design_at(np.array([0.25, 0.75]))
    bf = target.spline.frequency.design_at(freq)
    # Bounded-memory feature reconstruction: retain summaries, not giant surfaces.
    nc, nd = coefficients.shape[:2]
    powers = np.empty((nc, nd, 2))
    for start in range(0, nd, 128):
        logs = np.einsum(
            "ti,cdij,fj->cdtf",
            bt,
            coefficients[:, start : start + 128],
            bf,
            optimize=True,
        )
        powers[:, start : start + 128] = np.einsum(
            "cdtf,f->cdt", np.exp(logs), quadrature(freq)
        )
    time_contrast = np.einsum(
        "i,cdij,j->cd",
        bt[1] - bt[0],
        coefficients,
        target.spline.frequency.design_at(np.array([0.25]))[0],
    )
    extras = [
        powers[..., 0],
        powers[..., 1],
        np.log(posterior.sigma_time.values),
        np.log(posterior.sigma_freq.values),
        time_contrast,
    ]
    names += [
        "band_power_t.25",
        "band_power_t.75",
        "log_sigma_time",
        "log_sigma_freq",
        "time_contrast_logV_f.25",
    ]
    loss = []
    for center, half in ((0.38, 8), (0.68, 32)):
        times = center + np.arange(-half, half + 1) / 1024
        bt_window = target.spline.time.design_at(times)
        values = np.empty((nc, nd))
        for start in range(0, nd, 128):
            logs = np.einsum(
                "ti,cdij,fj->cdtf",
                bt_window,
                coefficients[:, start : start + 128],
                bf,
                optimize=True,
            )
            values[:, start : start + 128] = stationarity_loss(logs)
        loss.append(values)
        names.append(f"D_center{center}_m{half}")
    return np.concatenate(
        [selected, *[x[..., None] for x in extras + loss]], axis=-1
    ), names


def packed_reference(guide, posterior):
    nc, nd = posterior.sizes["chain"], posterior.sizes["draw"]
    values = []
    for name in guide._init_locs:
        site = guide.prototype_trace[name]
        unconstrained = biject_to(site["fn"].support).inv(
            jnp.asarray(posterior[name].values)
        )
        values.append(np.asarray(unconstrained).reshape(nc, nd, -1))
    return np.concatenate(values, axis=-1)


def posterior_geometry(guide, params, reference_packed):
    distribution = guide.get_posterior(params)
    mean = np.asarray(distribution.mean)
    if hasattr(distribution, "covariance_matrix"):
        covariance = np.asarray(distribution.covariance_matrix)
    else:
        covariance = np.diag(np.asarray(distribution.variance))
    whitening = np.linalg.inv(np.linalg.cholesky(covariance)).T
    transformed = (reference_packed.reshape(-1, len(mean)) - mean) @ whitening
    return {
        "nuts_mean_in_q_coordinates": transformed.mean(axis=0),
        "nuts_covariance_eigenvalues_in_q_coordinates": np.linalg.eigvalsh(
            np.cov(transformed, rowvar=False)
        ),
        "nuts_skew_in_q_coordinates": skew(transformed, axis=0),
        "nuts_excess_kurtosis_in_q_coordinates": kurtosis(transformed, axis=0),
    }


def mmd_comparison(q, p, cfg):
    """Reference transform trained on first 1/4; disjoint chain baselines."""
    nd = p.shape[1]
    train = p[:, : nd // 4].reshape(-1, p.shape[-1])
    remaining = p[:, nd // 4 :]
    transform = reference_transform(train)

    def project(x):
        return (x - transform["mean"]) @ transform["matrix"]

    n = min(
        cfg["mmd_count"], remaining[:2].size // p.shape[-1], q.shape[1] // 2
    )
    # No thinning/IID claim: consecutive chain segments yield descriptive baselines.
    p1 = project(remaining[:2].reshape(-1, p.shape[-1])[:n])
    p2 = project(remaining[2:].reshape(-1, p.shape[-1])[:n])
    q1 = project(q.reshape(-1, q.shape[-1])[:n])
    q2 = project(q.reshape(-1, q.shape[-1])[n : 2 * n])
    bandwidths = np.sqrt(p.shape[-1]) * np.asarray(
        cfg["mmd_bandwidth_multipliers"]
    )
    return {
        "status": "descriptive_autocorrelated_reference",
        "estimator": "unbiased_MMD_squared_diagonals_excluded",
        "bandwidths": bandwidths,
        "reference_transform": transform,
        "transform_training_count": len(train),
        "matched_count": n,
        "vi_vs_nuts": mmd_squared(q1, p1, bandwidths=bandwidths),
        "nuts_vs_nuts": mmd_squared(p1, p2, bandwidths=bandwidths),
        "vi_vs_vi": mmd_squared(q1, q2, bandwidths=bandwidths),
        "permutation_p_value": None,
        "sampling_design": "disjoint early transform-training subset; separate chains for NUTS baseline; no IID claim",
    }


def run_reference(target, cfg, directory, repair=False):
    prefix = "reference_repair" if repair else "reference"
    started = perf_counter()
    result = run_nuts(
        target.model,
        rng_key=jax.random.PRNGKey(cfg["nuts_seed"] + int(repair)),
        n_warmup=cfg["nuts_repair_warmup"] if repair else cfg["nuts_warmup"],
        n_samples=cfg["nuts_repair_draws"] if repair else cfg["nuts_draws"],
        num_chains=cfg["nuts_chains"],
        dense_mass=True,
        chain_method="sequential",
        target_accept_prob=cfg["nuts_repair_target_accept"]
        if repair
        else cfg["nuts_target_accept"],
        max_tree_depth=cfg["max_tree_depth"],
        init_strategy=init_to_uniform(radius=cfg["nuts_init_radius"]),
        progress_bar=False,
        extra_fields=(
            "energy",
            "potential_energy",
            "num_steps",
            "diverging",
            "accept_prob",
            "adapt_state.step_size",
        ),
    )
    inference_seconds = perf_counter() - started
    # Persist chains before analysis so a failed diagnostic cannot discard a reference.
    result.posterior.to_netcdf(
        directory / f"{prefix}_posterior.nc", engine="h5netcdf"
    )
    result.sample_stats.to_netcdf(
        directory / f"{prefix}_sample_stats.nc", engine="h5netcdf"
    )
    feature_started = perf_counter()
    values, names = features(target, result.posterior)
    latent_names = [
        name
        for name in result.posterior
        if not name.startswith("log_likelihood")
    ]
    latent = np.concatenate(
        [
            result.posterior[name].values.reshape(
                cfg["nuts_chains"],
                -1,
                int(np.prod(result.posterior[name].shape[2:])),
            )
            for name in latent_names
        ],
        axis=-1,
    )
    health = nuts_reference_health(
        np.concatenate([latent, values], axis=-1),
        result.sample_stats,
        max_tree_depth=cfg["max_tree_depth"],
    )
    record = {
        "status": health["status"],
        "target_fingerprint": target.identity,
        "health": health,
        "latent_names": latent_names,
        "feature_names": names,
        "inference_including_compile_seconds": inference_seconds,
        "feature_reconstruction_diagnostics_seconds": perf_counter()
        - feature_started,
        "wall_seconds": perf_counter() - started,
        "initialization": "four independently randomized uniform initializations; no VI",
        "provenance": runtime_provenance(),
        "dtypes": {
            name: str(var.dtype) for name, var in result.posterior.items()
        },
    }
    result.posterior.to_netcdf(
        directory / f"{prefix}_posterior.nc", engine="h5netcdf"
    )
    result.sample_stats.to_netcdf(
        directory / f"{prefix}_sample_stats.nc", engine="h5netcdf"
    )
    np.savez_compressed(
        directory / f"{prefix}_features.npz",
        values=values,
        names=np.array(names),
    )
    write_json(directory / f"{prefix}.json", record)
    return result, values, names, record


def run_target(name, cfg, previous, out):
    if not jax.config.x64_enabled:
        raise ValueError("this benchmark requires JAX_ENABLE_X64=true")
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        raise ValueError(f"refusing to overwrite completed target {directory}")
    started = perf_counter()
    manifest = {
        "target": name,
        "config": cfg,
        "provenance": runtime_provenance(),
        "status": "running",
    }
    write_json(directory / "manifest.json", manifest)
    try:
        target = build_target(name, cfg, previous, directory)
        manifest.update(
            target_fingerprint=target.identity,
            descriptor=target.descriptor,
            preparation_seconds=perf_counter() - started,
        )
        write_json(directory / "manifest.json", manifest)
        reference, p, names, health = run_reference(target, cfg, directory)
        if health["status"] != "accepted_screen":
            print(
                f"{name}: reference failed screens; retaining and running separate repair",
                flush=True,
            )
            reference, p, names, health = run_reference(
                target, cfg, directory, repair=True
            )
        print(f"{name}: reference {health['status']}", flush=True)
        diagnostic_config = VIDiagnosticConfig(
            seeds=tuple(cfg["diagnostic_seeds"]),
            num_particles=cfg["diagnostic_particles"],
            chunk_size=cfg["diagnostic_chunk"],
            evaluation_seeds=tuple(cfg["evaluation_seeds"]),
            evaluation_particles=cfg["evaluation_particles"],
            checkpoint_steps=tuple(cfg["checkpoints"]),
            target_fingerprint=target.identity,
            parameterization="centered"
            if target.pair is None
            else "noncentered",
        )
        events = []
        for index, feature in enumerate(names):
            if feature.startswith("D_"):
                events += [
                    (feature + f"_below{epsilon}", index, epsilon)
                    for epsilon in (0.005, 0.02, 0.08)
                ]
            elif feature == "time_contrast_logV_f.25":
                events.append(("positive_time_contrast", index, 0.0))
        rows = []
        for family in cfg["guides"]:
            for seed in cfg["optimization_seeds"]:
                job_name = f"{family.replace(':', '-')}_seed{seed}"
                job_dir = directory / job_name
                job_dir.mkdir(exist_ok=True)
                before = perf_counter()
                vi = fit_vi(
                    target.model,
                    rng_key=jax.random.PRNGKey(seed),
                    vi_steps=max(cfg["checkpoints"]),
                    optimizer_lr=cfg["vi_lr"],
                    guide=family,
                    posterior_draws=cfg["posterior_draws"],
                    init_values=target.init,
                    early_stopping=False,
                    diagnostics=diagnostic_config,
                )
                fit_seconds = perf_counter() - before
                vi.diagnostics.save(job_dir)
                vi.posterior.to_netcdf(
                    job_dir / "posterior.nc", engine="h5netcdf"
                )
                guide = rebuild_guide(
                    vi.diagnostics,
                    target.model,
                    target_fingerprint=target.identity,
                )
                pr = packed_reference(guide, reference.posterior)
                summaries = []
                for step in cfg["checkpoints"]:
                    checkpoint_started = perf_counter()
                    params = vi.diagnostics.checkpoints[str(step)]
                    draws = guide.get_posterior(params).sample(
                        jax.random.PRNGKey(8301), (cfg["posterior_draws"],)
                    )
                    constrained = guide._unpack_and_constrain(draws, params)
                    dataset = xr.Dataset(
                        {
                            key: xr.DataArray(
                                np.asarray(value)[None],
                                dims=(
                                    "chain",
                                    "draw",
                                    *[
                                        f"{key}_dim{i}"
                                        for i in range(np.ndim(value) - 1)
                                    ],
                                ),
                            )
                            for key, value in constrained.items()
                            if not key.startswith("log_likelihood")
                        }
                    )
                    q, _ = features(target, dataset)
                    comparison = compare_features(
                        q,
                        p,
                        names=names,
                        vi_fingerprint=target.identity,
                        reference_fingerprint=health["target_fingerprint"],
                        reference_status=health["status"],
                        events=events,
                    )
                    summaries.append(
                        {
                            "actual_steps": step,
                            "comparison": comparison,
                            "functional_mmd": mmd_comparison(q, p, cfg),
                            "latent_mmd": mmd_comparison(
                                np.asarray(draws)[None], pr, cfg
                            ),
                            "geometry": posterior_geometry(guide, params, pr),
                            "checkpoint_comparison_seconds": perf_counter()
                            - checkpoint_started,
                        }
                    )
                    np.savez_compressed(
                        job_dir / f"features_step{step}.npz",
                        values=q,
                        names=np.array(names),
                    )
                record = {
                    "target": name,
                    "guide": family,
                    "optimization_seed": seed,
                    "fit_workflow_seconds": fit_seconds,
                    "timings": vi.timings,
                    "optimization": vi.diagnostics.metadata["optimization"],
                    "native_psis": vi.diagnostics.metadata["native_psis"],
                    "weights": vi.diagnostics.metadata["weights"],
                    "checkpoints": summaries,
                    "provenance": vi.diagnostics.metadata["provenance"],
                    "predictive_adequacy": {"status": "not_run_stage_D"},
                    "calibration": {"status": "not_run_stage_E"},
                }
                write_json(job_dir / "comparison.json", record)
                rows.append(record)
                write_json(directory / "comparisons.json", rows)
                print(
                    f"{name} {job_name}: {fit_seconds:.2f}s, "
                    + str(
                        [
                            (r["k"], round(r.get("raw_ess_fraction", 0), 3))
                            for r in record["weights"]
                        ]
                    ),
                    flush=True,
                )
                del vi, guide, pr, summaries
                gc.collect()
                jax.clear_caches()
        manifest.update(
            status="complete",
            wall_seconds=perf_counter() - started,
            reference_status=health["status"],
            jobs_run=len(rows),
        )
        write_json(directory / "complete.json", manifest)
    except Exception as error:
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=perf_counter() - started,
        )
        write_json(directory / "failure.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/vi-nuts-validation"
    )
    parser.add_argument("--target")
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    if args.target:
        run_target(args.target, cfg, args.previous, args.out)
    else:
        start = perf_counter()
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(
            args.out / "execution.json",
            {
                "status": "running",
                "config": cfg,
                "provenance": runtime_provenance(),
            },
        )
        jobs = []
        for target in cfg["targets"]:
            before = perf_counter()
            with (args.out / f"{target}.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--target",
                        target,
                        "--config",
                        str(args.config),
                        "--previous",
                        str(args.previous),
                        "--out",
                        str(args.out),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            jobs.append(
                {
                    "target": target,
                    "exit_code": result.returncode,
                    "process_wall_seconds": perf_counter() - before,
                }
            )
            print(jobs[-1], flush=True)
        write_json(
            args.out / "execution.json",
            {
                "status": "complete"
                if all(j["exit_code"] == 0 for j in jobs)
                else "failures",
                "jobs": jobs,
                "config": cfg,
                "total_process_wall_seconds": perf_counter() - start,
                "provenance": runtime_provenance(),
            },
        )
        if any(job["exit_code"] for job in jobs):
            sys.exit(1)


if __name__ == "__main__":
    main()
