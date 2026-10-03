"""Stages 0--3: one native stationary AR(4) PSD target, two smoothing targets.

Reuses prepared stationary models, fit_vi, run_nuts and existing diagnostics.
Numerical targets are frozen once; reference repairs and all failures persist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
import tomllib
from functools import partial
from inspect import getsource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr
from arviz_stats.base import array_stats
from comparison import (
    compare_with_mc,
    mc_stats,
    paired_stability,
    primary_indices,
)
from comparison import (
    screen_interval as screen_interval,
)
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import log_density
from scipy.linalg import solve_discrete_lyapunov
from skfda.representation.basis import BSplineBasis
from study_common import (
    draw_dataset,
    learning_rate,
    mmd_comparison,
    packed_reference,
    quadrature,
    write_json,
)

from log_psplines.config import StationaryConfig
from log_psplines.data.spectral import WishartData
from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    compare_features,
    evaluate_objective,
    fingerprint,
    nuts_reference_health,
    rebuild_guide,
    require_same_target,
    runtime_provenance,
)
from log_psplines.inference.initialisation import init_weights
from log_psplines.inference.model import (
    _blocked_channel_model,
    _sample_pspline_block,
    channel_model_kwargs,
    prepare_model,
)
from log_psplines.inference.nuts import run_nuts
from log_psplines.inference.vi import fit_vi
from log_psplines.likelihoods.whittle import whittle_log_likelihood
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.models.reconstruction import reconstruct_stationary_spectrum
from log_psplines.preprocessing.periodogram import compute_wishart

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_memory_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def ar4_definition():
    # Existing VARMAData.ar(order=4) fixture, not the proposed fallback.
    phi = np.array([0.9, -0.8, 0.7, -0.6])
    companion = np.vstack((phi, np.c_[np.eye(3), np.zeros(3)]))
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if radius >= 1:
        raise ValueError("AR fixture is not stationary")
    unit = solve_discrete_lyapunov(companion, np.diag([1.0, 0, 0, 0]))
    innovation = 1 / unit[0, 0]
    covariance = unit * innovation
    return phi, companion, covariance, float(innovation), radius


def simulate_ar4(n, seed):
    phi, companion, covariance, innovation, _ = ar4_definition()
    rng = np.random.default_rng(seed)
    state = rng.multivariate_normal(np.zeros(4), covariance)
    initial = state.copy()
    noise = rng.normal(scale=np.sqrt(innovation), size=n)
    record = np.empty(n)
    for index, value in enumerate(noise):
        state = companion @ state
        state[0] += value
        record[index] = state[0]
    return record, initial, phi, innovation


def ar4_psd(frequency):
    phi, _, _, innovation, _ = ar4_definition()
    denominator = (
        1
        - np.exp(
            -2j * np.pi * np.asarray(frequency)[:, None] * np.arange(1, 5)
        )
        @ phi
    )
    return 2 * innovation / np.abs(denominator) ** 2


def design(spline, frequency, fit_frequency):
    # Native stationary knots are breakpoints, not clamped SciPy knots.
    coordinate = (np.asarray(frequency) - fit_frequency[0]) / (
        fit_frequency[-1] - fit_frequency[0]
    )
    basis = BSplineBasis(
        domain_range=[0, 1],
        order=spline.degree + 1,
        knots=spline.knots.tolist(),
    )
    return np.asarray(basis(coordinate)[:, :, 0].T)


def frequency_observations(raw):
    data = compute_wishart(
        np.asarray(raw)[:, None],
        fs=1.0,
        Nb=1,
        window=None,
        detrend=False,
        wishart_floor_fraction=None,
    )
    return data.apply_mask(data.freq < 0.5)


def native_config(coefficients):
    # In the native breakpoint convention cubic K = n_knots + 2.
    return StationaryConfig(
        n_knots=coefficients - 2,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": "uniform"},
        Nb=1,
        wishart_window=None,
        wishart_detrend=False,
        roughness_scale=1.28,
        smoothing_parameterization="centered",
        eta=1.0,
        verbose=False,
    )


def conditional_model(base, sigma):
    return numpyro.handlers.condition(
        base, data={"sigma_delta_0": jnp.asarray(sigma)}
    )


def ledger(out, event):
    path = out / "attempt_ledger.json"
    previous = json.loads(path.read_text()) if path.exists() else []
    previous.append(event)
    write_json(path, previous)


def prepare(out):
    if (out / "protocol.json").exists():
        verify_frozen(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    before = perf_counter()
    archive = ROOT / "runs/vi-optimization-exact-tv/optimization.toml"
    inherited = tomllib.loads(archive.read_text())
    schedule = next(
        x for x in inherited["schedules"] if x["name"] == "warmup_cosine"
    )
    cfg = {
        "stages": [0, 1, 2, 3],
        "data_seed": 60101,
        "reserved_data_seeds": [60102, 60103, 60104, 60105, 60106],
        "n": 4096,
        "dt": 1.0,
        "targets": ["fixed", "hierarchical"],
        "guides": ["diag", "mvn"],
        "optimization_seeds": [7101, 7102, 7103],
        "schedule": schedule,
        "schedule_source": str(archive),
        "schedule_source_sha256": file_hash(archive),
        "optimization_particles": 8,
        "updates": 40000,
        "checkpoint_steps": [5000, 10000, 20000, 30000, 35000, 40000],
        "checkpoint_draws": 4096,
        "final_draws": 16384,
        "checkpoint_draw_seed": 8301,
        "diagnostic_seeds": [8101, 8102, 8103],
        "diagnostic_particles": 4096,
        "diagnostic_chunk": 128,
        "evaluation_seeds": [8201, 8202, 8203, 8204, 8205, 8206, 8207, 8208],
        "evaluation_particles": 256,
        "nuts_chains": 4,
        "nuts_seed": 6101,
        "nuts_init_coefficient_jitter_scale": 0.2,
        "nuts_init_log_sigma_jitter_sd": 0.1,
        "reference_initial": {
            "warmup": 1000,
            "draws": 2000,
            "target_accept": 0.95,
            "max_tree_depth": 10,
        },
        "reference_repair": {
            "warmup": 2000,
            "draws": 4000,
            "target_accept": 0.99,
            "max_tree_depth": 10,
        },
        "max_reference_repairs_per_target": 1,
        "extra_diagnostic_draw_collections": 0,
        "mean_tolerance_sd": 0.1,
        "sd_ratio_limits": [0.9, 1.1],
        "reference_primary_mean_mcse_sd": 0.02,
        "stability_mean_tolerance_sd": 0.1,
        "stability_sd_relative_tolerance": 0.05,
        "objective_change_tolerance": 0.2,
        "mmd_count": 512,
        "mmd_bandwidth_multipliers": [0.5, 1.0, 2.0],
        "guide_initial_scale": 0.1,
        "clip_global_norm": 1.0,
        "optimizer": "optax.adam",
        "parameterization": "centered",
        "early_stopping": False,
        "provenance": runtime_provenance(),
    }
    write_json(
        out / "dry_run_manifest.json",
        {
            "status": "prepared_before_fitting",
            "nominal_reference_jobs": 2,
            "nominal_vi_jobs": 12,
            "maximum_reference_repairs": 2,
            "hierarchical_gate": "usable fixed reference and at least one late-stable VI fit",
            "forbidden_work": [
                "parameterization sweep",
                "new data seeds",
                "TV fits",
                "flows",
                "new likelihood",
                "80k extension",
            ],
        },
    )
    frequency = np.arange(1, 2048) / 4096
    dummy = WishartData(
        np.ones((2047, 1, 1)),
        np.zeros((2047, 1, 1)),
        frequency,
        2047,
        1,
        duration=4096,
    )
    checks = []
    dense_frequency = np.linspace(0.02, 0.48, 257)
    selected = None
    for count in (20, 30):
        _, components = prepare_model(dummy, native_config(count))
        spline = components.diagonal_models[0]
        matrix = design(spline, dense_frequency, frequency)
        log_truth = np.log(ar4_psd(dense_frequency))
        projected = np.linalg.lstsq(matrix, log_truth, rcond=None)[0]
        difference = matrix @ projected - log_truth
        weights = quadrature(dense_frequency)
        weights /= weights.sum()
        rms = float(np.sqrt(np.sum(weights * difference**2)))
        checks.append(
            {
                "coefficients": spline.n_basis,
                "rms_log_projection_error": rms,
                "maximum_local_error": float(np.max(abs(difference))),
            }
        )
        if rms < 0.03:
            selected = count
            break
    if selected is None:
        write_json(out / "capacity_failure.json", checks)
        raise ValueError(
            "neither authorized pre-fit basis passed the capacity screen"
        )
    # Only now generate/open the sole development observation record.
    raw, initial, phi, innovation = simulate_ar4(cfg["n"], cfg["data_seed"])
    data = frequency_observations(raw)
    model_kwargs, components = prepare_model(data, native_config(selected))
    spline = components.diagonal_models[0]
    if spline.n_basis != selected or spline.time is not None:
        raise ValueError("unexpected native frequency-only basis")
    observed = data.u_re[:, 0, 0] ** 2 + data.u_im[:, 0, 0] ** 2
    pilot = np.asarray(init_weights(np.log(observed / data.duration), spline))
    # Correct the explicit T convention in the data-based initializer; do not
    # use the truth projection or change the likelihood/prior.
    primary_frequency = np.arange(1, 10) / 20
    band_frequency = np.stack(
        [
            np.linspace(a, b, 257)
            for a, b in ((0.04, 0.18), (0.18, 0.32), (0.32, 0.46))
        ]
    )
    band_design = np.stack(
        [design(spline, x, data.freq) for x in band_frequency]
    )
    sigma_fixed = 0.6744897501960817 * 1.28
    path = out / "frozen_target"
    path.mkdir()
    np.savez_compressed(
        path / "arrays.npz",
        raw=raw,
        initial_state=initial,
        ar_coefficients=phi,
        innovation_variance=innovation,
        frequency=data.freq,
        u_re=data.u_re,
        u_im=data.u_im,
        retained_fourier=np.sqrt(2) * np.fft.rfft(raw)[1:-1],
        basis=spline.basis,
        penalty=spline.penalty_matrix,
        knots=spline.knots,
        pilot=pilot,
        primary_frequency=primary_frequency,
        primary_design=design(spline, primary_frequency, data.freq),
        band_frequency=band_frequency,
        band_design=band_design,
        band_quadrature=np.stack([quadrature(x) for x in band_frequency]),
        dense_frequency=dense_frequency,
        dense_design=design(spline, dense_frequency, data.freq),
        overlay_frequency=np.linspace(data.freq[0], data.freq[-1], 257),
        overlay_design=design(
            spline, np.linspace(data.freq[0], data.freq[-1], 257), data.freq
        ),
    )
    cfg.update(
        coefficients=selected,
        capacity_checks_before_data=checks,
        fixed_sigma=sigma_fixed,
        ar_coefficients=phi.tolist(),
        innovation_variance=innovation,
        companion_spectral_radius=ar4_definition()[-1],
        frozen_arrays_sha256=file_hash(path / "arrays.npz"),
    )
    source_identity = [
        getsource(x)
        for x in (
            _blocked_channel_model,
            _sample_pspline_block,
            wishart_log_likelihood,
            whittle_log_likelihood,
        )
    ]
    physical = {
        "observed_one_sided_periodogram": observed / data.duration,
        "frequency": data.freq,
        "basis": np.asarray(spline.basis),
        "penalty": np.asarray(spline.penalty_matrix),
        "roughness_scale": 1.28,
        "units": "one-sided x^2/Hz",
        "Nb": 1,
        "Nh": 1,
        "duration": 4096.0,
        "enbw": 1.0,
        "eta": 1.0,
    }
    cfg["target_fingerprints"] = {
        name: fingerprint(
            physical,
            source_identity,
            {"sigma": "HalfNormal(1.28)"}
            if name == "hierarchical"
            else {"conditioned_sigma": sigma_fixed},
        )
        for name in cfg["targets"]
    }
    cfg["coordinate_fingerprints"] = {
        name: fingerprint(
            identity,
            "centered",
            "log sigma support transform"
            if name == "hierarchical"
            else "conditioned scale excluded",
        )
        for name, identity in cfg["target_fingerprints"].items()
    }
    cfg["prepared_dtypes"] = {
        name: str(np.asarray(value).dtype)
        for name, value in model_kwargs.items()
        if isinstance(value, jax.Array)
    }
    write_json(out / "protocol.json", cfg)
    ledger(
        out,
        {
            "stage": 0,
            "status": "complete",
            "preparation_seconds": perf_counter() - before,
            "peak_memory_mib": peak_memory_mib(),
            "new_data_seed": 60101,
        },
    )
    audit(out, cfg)


def verify_frozen(out):
    cfg = json.loads((out / "protocol.json").read_text())
    if (
        file_hash(out / "frozen_target/arrays.npz")
        != cfg["frozen_arrays_sha256"]
    ):
        raise ValueError("frozen numerical inputs changed")
    return cfg


def load_target(out, target):
    cfg = verify_frozen(out)
    arrays = np.load(out / "frozen_target/arrays.npz")
    data = WishartData(
        arrays["u_re"],
        arrays["u_im"],
        arrays["frequency"],
        2047,
        1,
        duration=4096,
        Nb=1,
        Nh=1,
        enbw=1,
        scaling_factor=1.0,
    )
    kwargs, components = prepare_model(
        data, native_config(cfg["coefficients"])
    )
    spline = components.diagonal_models[0]
    np.testing.assert_array_equal(spline.basis, arrays["basis"])
    np.testing.assert_array_equal(spline.penalty_matrix, arrays["penalty"])
    physical = {
        "observed_one_sided_periodogram": (
            data.u_re[:, 0, 0] ** 2 + data.u_im[:, 0, 0] ** 2
        )
        / data.duration,
        "frequency": data.freq,
        "basis": np.asarray(spline.basis),
        "penalty": np.asarray(spline.penalty_matrix),
        "roughness_scale": 1.28,
        "units": "one-sided x^2/Hz",
        "Nb": 1,
        "Nh": 1,
        "duration": 4096.0,
        "enbw": 1.0,
        "eta": 1.0,
    }
    source_identity = [
        getsource(x)
        for x in (
            _blocked_channel_model,
            _sample_pspline_block,
            wishart_log_likelihood,
            whittle_log_likelihood,
        )
    ]
    current = fingerprint(
        physical,
        source_identity,
        {"sigma": "HalfNormal(1.28)"}
        if target == "hierarchical"
        else {"conditioned_sigma": cfg["fixed_sigma"]},
    )
    require_same_target(current, cfg["target_fingerprints"][target])
    kwargs = channel_model_kwargs(kwargs, 0)
    kwargs["eta"] = 1.0
    base = partial(_blocked_channel_model, **kwargs)
    model = (
        conditional_model(base, cfg["fixed_sigma"])
        if target == "fixed"
        else base
    )
    init = {"weights_delta_0": jnp.asarray(arrays["pilot"])}
    if target == "hierarchical":
        init["sigma_delta_0"] = jnp.asarray(cfg["fixed_sigma"])
    return cfg, arrays, data, components, model, init


def audit(out, cfg):
    text = f"""# Stationary AR(4) audit: completed stages 0–1 preparation

Source/backend/versions: see protocol.json. Actual NumPyro is {cfg["provenance"]["versions"]["numpyro"]}; JAX x64 is {cfg["provenance"]["x64_enabled"]} on {cfg["provenance"]["backend"]}. The worktree is dirty and preserved; each job also records its source/diff hash. No dependency changes were made.

Existing AR fixture phi={cfg["ar_coefficients"]}, companion radius {cfg["companion_spectral_radius"]:.8f}, innovation variance {cfg["innovation_variance"]:.8f}. Initial lag state is sampled from the stationary Lyapunov covariance, normalized to unit theoretical variance. Raw seed 60101 is preserved without realization normalization or burn-in. Seeds 60102–60106 are untouched.

Before generating observations, 20/30 native cubic coefficient capacity checks were {cfg["capacity_checks_before_data"]}. Selected K={cfg["coefficients"]} uniformly spaced breakpoint basis, one permitted pre-fit adjustment. The analytic projection is never an inference initializer.

Preprocessing uses native compute_wishart on the unchanged raw record: one full 4096-point rectangular FFT, detrend=False, no floor/standardization/segmentation/pooling. DC is dropped by that function; the real Nyquist endpoint is explicitly masked. Retained f=k/4096, k=1..2047. Native U stores sufficient statistics; original signed complex coefficients sqrt(2)*rfft(x)[1:-1] are saved separately. Nb=Nh=eta=ENBW=1; T=4096. For dt=1, Y=2|FFT(x)|², periodogram Y/T and modeled exp(Bc) are one-sided x²/Hz. Log L=-sum(log S+Y/(T*S)), with the existing ±80 exponent guard. This is the same working Whittle target for both algorithms, not a claim of finite-record Fourier independence.

The centered native prior is sigma~HalfNormal(1.28), c|sigma proportional to sigma^-K exp(-c'Pc/(2 sigma²)). P is the native integrated-second-derivative, max-normalized penalty plus 1e-6 I. Its weak null/intercept directions receive that same proper ridged prior; no separate intercept prior is introduced. The base Normal is corrected by weights_prior_weights_delta_0. Fixed smoothing conditions sigma at its prior median {cfg["fixed_sigma"]:.10f}, excluding it from latent packing; the hierarchy restores this site with its positive support transform. Physical target IDs differ; coordinate IDs are separate.

The pilot uses existing init_weights on log(Y/T). prepare_components stores an empirical log(Y) pilot, which differs by log(T); this experiment explicitly supplies the physically normalized data-based pilot to both samplers. No truth-based initialization. Four reference starts perturb that pilot by 0.2/sqrt(diag(B'B)); hierarchical sigma additionally gets independent 0.1-log-scale perturbations. VI initial guide scale is 0.1.

Inherited optimizer source: {cfg["schedule_source"]} (hash {cfg["schedule_source_sha256"]}); resolved schedule {cfg["schedule"]}. Eight full-data training particles, Adam/default beta/epsilon choices, clip global norm 1, no early stopping, 40k updates. Fixed evaluation 256 particles×8 keys plus three independent keys; PSIS 4096×3; final posterior summaries 16384 fresh draws; checkpoint comparisons 4096 draws with common keys. Guide checkpoints contain numerical parameters, not optimizer state. No exact resume or 80k extension is used.

Nominal budget: two independently diagnosed four-chain references and 12 VI fits. At most one repair per target; repair settings are frozen before fitting. Stage 3 requires a usable fixed reference and one reasonably late-stable VI fit. Failed references/checkpoints/logs remain in attempt_ledger.json. Primary features and all evaluation matrices are frozen in arrays.npz.
"""
    (out / "audit.md").write_text(text)


def physical_features(posterior, arrays, target):
    weights = np.asarray(posterior["weights_delta_0"])
    shape = weights.shape[:2]
    flat = weights.reshape(-1, weights.shape[-1])
    selected = flat @ arrays["primary_design"].T
    bands = np.empty((len(flat), 3))
    for first in range(0, len(flat), 256):
        logs = np.einsum(
            "nk,bfk->nbf", flat[first : first + 256], arrays["band_design"]
        )
        bands[first : first + 256] = np.einsum(
            "nbf,bf->nb", np.exp(logs), arrays["band_quadrature"]
        )
    values = [selected, np.log(bands), bands]
    names = (
        [f"log_S_f{x:.2f}" for x in arrays["primary_frequency"]]
        + [f"log_band_{i}" for i in range(3)]
        + [f"band_{i}" for i in range(3)]
    )
    if target == "hierarchical":
        sigma = np.asarray(posterior["sigma_delta_0"]).reshape(-1, 1)
        values += [np.log(sigma), sigma]
        names += ["log_sigma", "sigma"]
    return np.concatenate(values, axis=1).reshape(*shape, -1), names


def preflight(out):
    cfg, arrays, data, components, base, init = load_target(
        out, "hierarchical"
    )
    fixed = conditional_model(base, cfg["fixed_sigma"])
    c = jnp.asarray(arrays["pilot"])

    def hierarchy(value):
        return log_density(
            base,
            (),
            {},
            {
                "weights_delta_0": value,
                "sigma_delta_0": jnp.asarray(cfg["fixed_sigma"]),
            },
        )[0]

    def conditional(value):
        return log_density(fixed, (), {}, {"weights_delta_0": value})[0]

    delta = float(conditional(c) - hierarchy(c))
    gradient_error = float(
        np.max(
            abs(np.asarray(jax.grad(conditional)(c) - jax.grad(hierarchy)(c)))
        )
    )
    np.testing.assert_allclose(delta, 0, atol=1e-9)
    np.testing.assert_allclose(gradient_error, 0, atol=1e-9)
    _, trace = log_density(base, (), {}, init)
    actual = float(
        trace["likelihood_channel_0"]["fn"]
        .log_prob(trace["likelihood_channel_0"]["value"])
        .sum()
    )
    logs = arrays["basis"] @ np.asarray(c)
    explicit = float(
        -np.sum(
            logs
            + np.abs(arrays["retained_fourier"]) ** 2
            / (data.duration * np.exp(logs))
        )
    )
    np.testing.assert_allclose(actual, explicit, atol=1e-8, rtol=1e-12)
    datasets = {
        "weights_delta_0": xr.DataArray(
            np.stack([np.asarray(c), np.asarray(c) + 0.1])[None],
            dims=("chain", "draw", "coefficient"),
        )
    }
    reconstructed = (
        reconstruct_stationary_spectrum(xr.Dataset(datasets), components, data)
        .values[..., 0, 0]
        .real
    )
    np.testing.assert_allclose(
        reconstructed,
        np.exp(
            np.einsum(
                "cdk,fk->cdf",
                datasets["weights_delta_0"].values,
                arrays["basis"],
            )
        ),
        rtol=2e-12,
    )
    guides = {}
    for name, model in (("fixed", fixed), ("hierarchical", base)):
        for family, cls in (
            ("diag", AutoDiagonalNormal),
            ("mvn", AutoMultivariateNormal),
        ):
            guide = cls(model, init_loc_fn=init_to_value(values=init))
            numpyro.handlers.seed(guide, 0)()
            names = list(guide._init_locs)
            expected = (
                ["weights_delta_0"]
                if name == "fixed"
                else ["sigma_delta_0", "weights_delta_0"]
            )
            if names != expected:
                raise ValueError(f"unexpected guide latent sites: {names}")
            guides[f"{name}/{family}"] = {
                "latent_sites": names,
                "dimension": guide.latent_dim,
                "initial_scale": guide._init_scale,
            }
    record = {
        "status": "passed",
        "fixed_density_difference": delta,
        "fixed_gradient_max_difference": gradient_error,
        "likelihood_difference": actual - explicit,
        "complete_reconstruction_draws": 2,
        "guides": guides,
        "basis_dtype": str(arrays["basis"].dtype),
        "penalty_dtype": str(arrays["penalty"].dtype),
    }
    write_json(out / "preflight.json", record)
    ledger(out, {"stage": 0, "contract_tests": record})


def init_strategy(init, arrays, cfg):
    coefficient_scale = cfg["nuts_init_coefficient_jitter_scale"] / np.sqrt(
        np.sum(arrays["basis"] ** 2, axis=0)
    )

    def strategy(site):
        if site["type"] != "sample" or site.get("is_observed", False):
            return None
        key = site["kwargs"]["rng_key"]
        if site["name"] == "weights_delta_0":
            return init["weights_delta_0"] + jnp.asarray(
                coefficient_scale
            ) * jax.random.normal(key, shape=init["weights_delta_0"].shape)
        if site["name"] == "sigma_delta_0":
            return init["sigma_delta_0"] * jnp.exp(
                cfg["nuts_init_log_sigma_jitter_sd"] * jax.random.normal(key)
            )
        return None

    # NumPyro distinguishes a functools.partial callback from a strategy factory.
    return partial(strategy)


def reference(out, target, repair=False):
    cfg, arrays, data, components, model, init = load_target(out, target)
    directory = (
        out / "references" / target / ("repair_1" if repair else "initial")
    )
    if (directory / "complete.json").exists():
        return json.loads((directory / "complete.json").read_text())
    if directory.exists():
        raise ValueError(
            f"refusing to overwrite retained reference attempt {directory}"
        )
    directory.mkdir(parents=True)
    settings = cfg["reference_repair" if repair else "reference_initial"]
    record = {
        "target": target,
        "attempt": "repair_1" if repair else "initial",
        "settings": settings,
        "target_fingerprint": cfg["target_fingerprints"][target],
        "coordinate_fingerprint": cfg["coordinate_fingerprints"][target],
        "provenance": runtime_provenance(),
        "status": "running",
    }
    write_json(directory / "manifest.json", record)
    started = perf_counter()
    try:
        before = perf_counter()
        result = run_nuts(
            model,
            rng_key=jax.random.PRNGKey(
                cfg["nuts_seed"]
                + (100 if target == "hierarchical" else 0)
                + int(repair)
            ),
            n_warmup=settings["warmup"],
            n_samples=settings["draws"],
            num_chains=4,
            init_strategy=init_strategy(init, arrays, cfg),
            dense_mass=True,
            chain_method="sequential",
            target_accept_prob=settings["target_accept"],
            max_tree_depth=settings["max_tree_depth"],
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
        inference_seconds = perf_counter() - before
        # Preserve raw chains before any MC/statistical calculations.
        result.posterior.to_netcdf(
            directory / "posterior.nc", engine="h5netcdf"
        )
        result.sample_stats.to_netcdf(
            directory / "sample_stats.nc", engine="h5netcdf"
        )
        before = perf_counter()
        values, names = physical_features(result.posterior, arrays, target)
        weights = result.posterior.weights_delta_0.values
        latents = [weights]
        if target == "hierarchical":
            latents.append(result.posterior.sigma_delta_0.values[..., None])
        checked = np.concatenate([*latents, values], axis=-1)
        health = nuts_reference_health(
            checked,
            result.sample_stats,
            max_tree_depth=settings["max_tree_depth"],
        )
        stats = mc_stats(values)
        mean_precision = stats["mean_mcse"] / stats["sd"]
        limited = bool(
            np.max(mean_precision[primary_indices(names)])
            > cfg["reference_primary_mean_mcse_sd"]
        )
        status = (
            "failed_reference"
            if health["status"] != "accepted_screen"
            else "reference_precision_limited"
            if limited
            else "accepted_screen"
        )
        np.savez_compressed(
            directory / "features.npz", values=values, names=np.array(names)
        )
        write_json(directory / "feature_stats.json", stats)
        record.update(
            status=status,
            health=health,
            primary_mean_mcse_sd_max=float(
                np.max(mean_precision[primary_indices(names)])
            ),
            reference_precision_limited=limited,
            feature_names=names,
            inference_including_compile_seconds=inference_seconds,
            analysis_io_seconds=perf_counter() - before,
            wall_seconds=perf_counter() - started,
            peak_memory_mib=peak_memory_mib(),
            parameter_dtypes={
                name: str(var.dtype) for name, var in result.posterior.items()
            },
        )
        write_json(directory / "complete.json", record)
        print(
            f"{target} reference {record['attempt']}: {status}, divergences={health['divergences']}, MCSE/SD={record['primary_mean_mcse_sd_max']:.4f}",
            flush=True,
        )
        return record
    except Exception as error:
        record.update(
            status="execution_failed",
            error=repr(error),
            wall_seconds=perf_counter() - started,
            peak_memory_mib=peak_memory_mib(),
        )
        write_json(directory / "failure.json", record)
        raise


def stability(out, target):
    cfg = verify_frozen(out)
    refdir, _ = selected_reference(out, target)
    p = np.load(refdir / "features.npz")
    pc, cnames = coefficient_values(xr.load_dataset(refdir / "posterior.nc"))
    names = p["names"].tolist() + cnames
    reference_values = np.concatenate([p["values"], pc], axis=-1)
    records, fits, pairs = {}, [], []
    for family in cfg["guides"]:
        for seed in cfg["optimization_seeds"]:
            directory = out / "vi" / target / family / str(seed)
            if not (directory / "complete.json").exists():
                continue
            record = json.loads((directory / "complete.json").read_text())
            records[family, seed] = (directory, record)
            late = []
            for first, second in ((30000, 35000), (35000, 40000)):
                a, b = (
                    np.load(directory / f"checkpoint_{s}.npz")
                    for s in (first, second)
                )
                objectives = {
                    r["step"]: r["fixed_objective"]
                    for r in record["checkpoints"]
                }
                result = paired_stability(
                    np.concatenate([a["features"], a["coefficients"]], -1),
                    np.concatenate([b["features"], b["coefficients"]], -1),
                    reference_values,
                    names,
                    objectives[first],
                    objectives[second],
                    cfg,
                )
                late.append({"first": first, "second": second, **result})
            fits.append(
                {
                    "guide": family,
                    "seed": seed,
                    "late": late,
                    "reasonably_stable": all(
                        x["point_within_screen"]
                        and x["status"] != "outside_screen"
                        for x in late
                    ),
                }
            )
        for index, first in enumerate(cfg["optimization_seeds"]):
            for second in cfg["optimization_seeds"][index + 1 :]:
                if (family, first) not in records or (
                    family,
                    second,
                ) not in records:
                    continue
                da, ra = records[family, first]
                db, rb = records[family, second]
                a, b = (np.load(d / "checkpoint_40000.npz") for d in (da, db))
                oa, ob = (
                    r["checkpoints"][-1]["fixed_objective"] for r in (ra, rb)
                )
                result = paired_stability(
                    np.concatenate([a["features"], a["coefficients"]], -1),
                    np.concatenate([b["features"], b["coefficients"]], -1),
                    reference_values,
                    names,
                    oa,
                    ob,
                    cfg,
                )
                pairs.append(
                    {
                        "guide": family,
                        "first_seed": first,
                        "second_seed": second,
                        **result,
                    }
                )
    result = {
        "target": target,
        "fits": fits,
        "seed_pairs": pairs,
        "hierarchical_gate": any(x["reasonably_stable"] for x in fits),
        "gate_definition": "one completed fixed-target fit with both late point screens passing in primary physical features and coefficients, and no resolved outside-screen drift; all seeds/pairs retained separately",
    }
    write_json(out / f"stability_{target}.json", result)
    return result


def dispatch(out, arguments, name):
    """Isolate compilation/memory by inference job; retain every process log."""
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{name}.log"
    if path.exists():
        raise ValueError(f"refusing to replace an attempt log: {path}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(out),
        *arguments,
    ]
    ledger(out, {"job": name, "status": "dispatching", "command": command})
    print(f"dispatch {name}", flush=True)
    with path.open("w") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
    ledger(
        out,
        {
            "job": name,
            "status": "process_complete"
            if result.returncode == 0
            else "process_failed",
            "returncode": result.returncode,
            "log": str(path),
        },
    )
    print(f"finished {name}: exit {result.returncode}", flush=True)
    return result.returncode


def execute(out):
    prepare(out)
    cfg = verify_frozen(out)
    if not (out / "preflight.json").exists():
        dispatch(out, ["prepare"], "preflight")
    if not (out / "preflight.json").exists():
        raise ValueError("contract preflight failed; inference not launched")
    for target in cfg["targets"]:
        usable = None
        for attempt in ("initial", "repair_1"):
            directory = out / "references" / target / attempt
            if not directory.exists():
                if attempt == "repair_1":
                    previous = out / "references" / target / "initial"
                    reason_path = previous / (
                        "complete.json"
                        if (previous / "complete.json").exists()
                        else "failure.json"
                    )
                    reason = (
                        json.loads(reason_path.read_text())
                        if reason_path.exists()
                        else {"status": "incomplete_attempt"}
                    )
                    write_json(
                        out / "references" / target / "repair_reason.json",
                        {
                            "reason": reason,
                            "changes": cfg["reference_repair"],
                            "target_unchanged": True,
                            "authorized_repairs": 1,
                        },
                    )
                dispatch(
                    out,
                    [
                        "reference",
                        "--target",
                        target,
                        *(["--repair"] if attempt == "repair_1" else []),
                    ],
                    f"reference_{target}_{attempt}",
                )
            if (directory / "complete.json").exists():
                ref = json.loads((directory / "complete.json").read_text())
                if ref["status"] == "accepted_screen":
                    usable = ref
                    break
        if usable is None:
            ledger(
                out,
                {
                    "target": target,
                    "status": "stopped_reference_budget_exhausted",
                },
            )
            break
        write_json(
            out / "references" / target / "selection.json",
            {
                "attempt": usable["attempt"],
                "status": usable["status"],
                "target_fingerprint": usable["target_fingerprint"],
            },
        )
        for family in cfg["guides"]:
            for seed in cfg["optimization_seeds"]:
                directory = out / "vi" / target / family / str(seed)
                if not directory.exists():
                    dispatch(
                        out,
                        [
                            "vi",
                            "--target",
                            target,
                            "--guide",
                            family,
                            "--seed",
                            str(seed),
                        ],
                        f"vi_{target}_{family}_{seed}",
                    )
        result = stability(out, target)
        if target == "fixed" and not result["hierarchical_gate"]:
            ledger(
                out,
                {"stage": 3, "status": "unexecuted_fixed_optimization_gate"},
            )
            break
    summarize(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/vi-stationary-ar4"
    )
    parser.add_argument(
        "command", choices=["prepare", "reference", "vi", "summarize", "all"]
    )
    parser.add_argument(
        "--target", choices=["fixed", "hierarchical"], default="fixed"
    )
    parser.add_argument("--guide", choices=["diag", "mvn"], default="mvn")
    parser.add_argument(
        "--seed", type=int, choices=[7101, 7102, 7103], default=7101
    )
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        prepare(out)
        preflight(out)
    elif args.command == "reference":
        reference(out, args.target, args.repair)
    elif args.command == "vi":
        run_vi(out, args.target, args.guide, args.seed)
    elif args.command == "summarize":
        summarize(out)
    else:
        execute(out)


def selected_reference(out, target):
    selection = json.loads(
        (out / "references" / target / "selection.json").read_text()
    )
    directory = out / "references" / target / selection["attempt"]
    record = json.loads((directory / "complete.json").read_text())
    if record["status"] != "accepted_screen":
        raise ValueError(
            "reference is not usable within the frozen precision screen"
        )
    return directory, record


def coefficient_values(posterior):
    values = np.asarray(posterior["weights_delta_0"])
    return values, [f"coefficient_{i}" for i in range(values.shape[-1])]


def run_vi(out, target, family, seed):
    cfg, arrays, data, components, model, init = load_target(out, target)
    refdir, ref = selected_reference(out, target)
    require_same_target(
        cfg["target_fingerprints"][target], ref["target_fingerprint"]
    )
    directory = out / "vi" / target / family / str(seed)
    if (directory / "complete.json").exists():
        return
    if directory.exists():
        raise ValueError(
            f"refusing to overwrite retained VI attempt {directory}"
        )
    directory.mkdir(parents=True)
    record = {
        "target": target,
        "guide": family,
        "optimization_seed": seed,
        "target_fingerprint": cfg["target_fingerprints"][target],
        "coordinate_fingerprint": cfg["coordinate_fingerprints"][target],
        "provenance": runtime_provenance(),
        "status": "running",
    }
    write_json(directory / "manifest.json", record)
    started = perf_counter()
    try:
        diagnostics = VIDiagnosticConfig(
            seeds=tuple(cfg["diagnostic_seeds"]),
            num_particles=cfg["diagnostic_particles"],
            chunk_size=cfg["diagnostic_chunk"],
            evaluation_seeds=tuple(cfg["evaluation_seeds"]),
            evaluation_particles=cfg["evaluation_particles"],
            checkpoint_steps=tuple(cfg["checkpoint_steps"]),
            target_fingerprint=record["target_fingerprint"],
            parameterization="centered",
        )

        def preserve(step, params):
            np.savez_compressed(directory / f"parameters_{step}.npz", **params)

        before = perf_counter()
        vi = fit_vi(
            model,
            rng_key=jax.random.PRNGKey(seed),
            vi_steps=cfg["updates"],
            optimizer_lr=learning_rate(cfg["schedule"], cfg["updates"]),
            optimizer_lr_metadata=cfg["schedule"],
            optimization_particles=8,
            guide=family,
            posterior_draws=cfg["final_draws"],
            init_values=init,
            early_stopping=False,
            diagnostics=diagnostics,
            checkpoint_callback=preserve,
        )
        fit_seconds = perf_counter() - before
        vi.diagnostics.save(directory)
        vi.posterior.to_netcdf(directory / "posterior.nc", engine="h5netcdf")
        guide = rebuild_guide(
            vi.diagnostics,
            model,
            target_fingerprint=record["target_fingerprint"],
        )
        p = xr.load_dataset(refdir / "posterior.nc")
        reference_features = np.load(refdir / "features.npz")["values"]
        reference_coefficients, coefficient_names = coefficient_values(p)
        reference_packed = packed_reference(guide, p)
        checkpoints = []
        for step in cfg["checkpoint_steps"]:
            before = perf_counter()
            params = vi.diagnostics.checkpoints[str(step)]
            packed, draws = draw_dataset(
                guide,
                params,
                jax.random.PRNGKey(cfg["checkpoint_draw_seed"]),
                cfg["checkpoint_draws"],
            )
            q, names = physical_features(draws, arrays, target)
            coefficients, _ = coefficient_values(draws)
            np.savez_compressed(
                directory / f"checkpoint_{step}.npz",
                packed=packed,
                features=q,
                coefficients=coefficients,
                names=np.array(names),
            )
            checkpoints.append(
                {
                    "step": step,
                    "comparison": compare_features(
                        q,
                        reference_features,
                        names=names,
                        vi_fingerprint=record["target_fingerprint"],
                        reference_fingerprint=ref["target_fingerprint"],
                        reference_status="accepted_screen",
                    ),
                    "mc_uncertainty": compare_with_mc(
                        q, reference_features, names, cfg
                    ),
                    "fixed_objective": vi.diagnostics.metadata["optimization"][
                        "checkpoint_objectives"
                    ][str(step)],
                    "independent_objective": evaluate_objective(
                        model,
                        guide,
                        params,
                        diagnostics,
                        seeds=cfg["diagnostic_seeds"],
                    ),
                    "analysis_seconds": perf_counter() - before,
                }
            )
        before = perf_counter()
        final_q, names = physical_features(vi.posterior, arrays, target)
        final_c, _ = coefficient_values(vi.posterior)
        final_comparison = compare_features(
            final_q,
            reference_features,
            names=names,
            vi_fingerprint=record["target_fingerprint"],
            reference_fingerprint=ref["target_fingerprint"],
            reference_status="accepted_screen",
        )
        final_comparison["mc_uncertainty"] = compare_with_mc(
            final_q, reference_features, names, cfg
        )
        final_comparison["coefficient_comparison"] = compare_features(
            final_c,
            reference_coefficients,
            names=coefficient_names,
            vi_fingerprint=record["target_fingerprint"],
            reference_fingerprint=ref["target_fingerprint"],
            reference_status="accepted_screen",
        )
        final_comparison["coefficient_mc_uncertainty"] = compare_with_mc(
            final_c, reference_coefficients, coefficient_names, cfg
        )
        final_comparison["functional_mmd"] = mmd_comparison(
            final_q, reference_features, cfg
        )
        # Centered coefficient vector plus log sigma is nonredundant.
        packed_q, _ = draw_dataset(
            guide,
            vi.diagnostics.params,
            jax.random.PRNGKey(8401),
            cfg["final_draws"],
        )
        final_comparison["latent_mmd"] = mmd_comparison(
            packed_q[None], reference_packed, cfg
        )
        record.update(
            status="complete",
            fit_workflow_seconds=fit_seconds,
            timings=vi.timings,
            optimization=vi.diagnostics.metadata["optimization"],
            native_psis=vi.diagnostics.metadata["native_psis"],
            weights=vi.diagnostics.metadata["weights"],
            checkpoints=checkpoints,
            final=final_comparison,
            final_feature_analysis_seconds=perf_counter() - before,
            wall_seconds=perf_counter() - started,
            peak_memory_mib=peak_memory_mib(),
        )
        write_json(directory / "complete.json", record)
        print(
            f"completed {target}/{family}/{seed} in {record['wall_seconds']:.1f}s",
            flush=True,
        )
    except Exception as error:
        record.update(
            status="execution_failed",
            error=repr(error),
            wall_seconds=perf_counter() - started,
            saved_parameter_checkpoints=[
                p.name for p in directory.glob("parameters_*.npz")
            ],
        )
        write_json(directory / "failure.json", record)
        raise


def field_stats(posterior, matrix, iid):
    coefficients = np.asarray(posterior.weights_delta_0)
    flat = coefficients.reshape(-1, coefficients.shape[-1])
    logs = np.empty((len(flat), len(matrix)))
    for start in range(0, len(flat), 256):
        logs[start : start + 256] = flat[start : start + 256] @ matrix.T
    shaped = logs.reshape(*coefficients.shape[:2], -1)
    mean, sd = logs.mean(0), logs.std(0, ddof=1)
    mean_se = (
        sd / np.sqrt(len(flat))
        if iid
        else np.asarray(
            array_stats.mcse(shaped, chain_axis=0, draw_axis=1, method="mean")
        )
    )
    sd_se = (
        np.sqrt(
            np.maximum(np.mean((logs - mean) ** 4, axis=0) - sd**4, 0)
            / (4 * len(flat) * sd**2)
        )
        if iid
        else np.asarray(
            array_stats.mcse(shaped, chain_axis=0, draw_axis=1, method="sd")
        )
    )
    return {
        "log_mean": mean,
        "log_sd": sd,
        "log_mean_mcse": mean_se,
        "log_sd_mcse": sd_se,
        "psd_quantile_05_50_95": np.exp(
            np.quantile(logs, [0.05, 0.5, 0.95], axis=0)
        ),
    }


def field_comparison(q, p, frequency):
    delta = (q["log_mean"] - p["log_mean"]) / p["log_sd"]
    ratio = q["log_sd"] / p["log_sd"]
    delta_se = (
        np.sqrt(
            q["log_mean_mcse"] ** 2
            + p["log_mean_mcse"] ** 2
            + (delta * p["log_sd_mcse"]) ** 2
        )
        / p["log_sd"]
    )
    ratio_se = ratio * np.sqrt(
        (q["log_sd_mcse"] / q["log_sd"]) ** 2
        + (p["log_sd_mcse"] / p["log_sd"]) ** 2
    )
    w = quadrature(frequency)
    w /= w.sum()
    worst = int(np.argmax(abs(delta)))
    return {
        "frequency": frequency,
        "standardized_log_mean_difference": delta,
        "mean_mcse": delta_se,
        "sd_ratio": ratio,
        "sd_ratio_mcse": ratio_se,
        "physical_quadrature_weights": w,
        "rms_standardized_log_mean_difference": float(
            np.sqrt(np.sum(w * delta**2))
        ),
        "worst_standardized_log_mean_difference": float(abs(delta[worst])),
        "worst_frequency": float(frequency[worst]),
        "sd_ratio_range": [float(ratio.min()), float(ratio.max())],
    }


def value_range(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return f"{min(values):.3g}–{max(values):.3g}" if values else "unavailable"


def summarize(out):
    """Saved-draw analysis only: never starts inference or replaces failures."""
    started = perf_counter()
    cfg = verify_frozen(out)
    arrays = np.load(out / "frozen_target/arrays.npz")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    records, refs, stabilities, fields = {}, {}, {}, {}
    details, rows, reference_rows = [], [], []
    for target in cfg["targets"]:
        attempts = []
        for attempt in ("initial", "repair_1"):
            directory = out / "references" / target / attempt
            for name in ("complete.json", "failure.json"):
                path = directory / name
                if path.exists():
                    record = json.loads(path.read_text())
                    attempts.append(record)
                    reference_rows.append(
                        {
                            "target": target,
                            "attempt": attempt,
                            "status": record["status"],
                            "retained_draws_per_chain": record["settings"][
                                "draws"
                            ]
                            if (directory / "posterior.nc").exists()
                            else 0,
                            "error": record.get("error"),
                            "health": record.get("health"),
                            "primary_mean_mcse_sd_max": record.get(
                                "primary_mean_mcse_sd_max"
                            ),
                            "wall_seconds": record["wall_seconds"],
                        }
                    )
        try:
            refdir, ref = selected_reference(out, target)
        except (FileNotFoundError, ValueError):
            for family in cfg["guides"]:
                rows.append(
                    {
                        "target": target,
                        "guide": family,
                        "status": "unexecuted_no_usable_reference"
                        if attempts
                        else "unexecuted_prior_stage_gate",
                        "reference_status": attempts[-1]["status"]
                        if attempts
                        else "unexecuted",
                    }
                )
            continue
        refs[target] = (refdir, ref, xr.load_dataset(refdir / "posterior.nc"))
        stabilities[target] = stability(out, target)
        pfields = {
            name: field_stats(refs[target][2], arrays[f"{name}_design"], False)
            for name in ("dense", "overlay")
        }
        fields[target] = {"reference": pfields, "vi": {}}
        write_json(refdir / "field_stats.json", pfields)
        for family in cfg["guides"]:
            available = []
            for seed in cfg["optimization_seeds"]:
                directory = out / "vi" / target / family / str(seed)
                path = directory / "complete.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                records[target, family, seed] = (directory, record)
                available.append(record)
                posterior = xr.load_dataset(directory / "posterior.nc")
                qfields = {
                    name: field_stats(
                        posterior, arrays[f"{name}_design"], True
                    )
                    for name in ("dense", "overlay")
                }
                comparison = field_comparison(
                    qfields["dense"],
                    pfields["dense"],
                    arrays["dense_frequency"],
                )
                fields[target]["vi"][family, seed] = {
                    **qfields,
                    "comparison": comparison,
                }
                write_json(
                    directory / "field_stats.json",
                    {**qfields, "comparison": comparison},
                )
                # Geometry includes all centered coefficients and inferred log scale.
                pc, cnames = coefficient_values(refs[target][2])
                qc, _ = coefficient_values(posterior)
                if target == "hierarchical":
                    pc = np.concatenate(
                        [
                            pc,
                            np.log(refs[target][2].sigma_delta_0.values)[
                                ..., None
                            ],
                        ],
                        axis=-1,
                    )
                    qc = np.concatenate(
                        [
                            qc,
                            np.log(posterior.sigma_delta_0.values)[..., None],
                        ],
                        axis=-1,
                    )
                    cnames += ["log_sigma"]
                geometry = compare_features(
                    qc,
                    pc,
                    names=cnames,
                    vi_fingerprint=record["target_fingerprint"],
                    reference_fingerprint=ref["target_fingerprint"],
                    reference_status=ref["status"],
                )
                write_json(directory / "joint_geometry.json", geometry)
                saved_mc = {}
                for checkpoint in record["checkpoints"]:
                    saved = np.load(
                        directory / f"checkpoint_{checkpoint['step']}.npz"
                    )
                    uncertainty = checkpoint.get(
                        "mc_uncertainty"
                    ) or compare_with_mc(
                        saved["features"],
                        np.load(refdir / "features.npz")["values"],
                        saved["names"].tolist(),
                        cfg,
                    )
                    checkpoint["mc_uncertainty"] = uncertainty
                    coefficient_comparison = compare_features(
                        saved["coefficients"],
                        coefficient_values(refs[target][2])[0],
                        names=coefficient_values(refs[target][2])[1],
                        vi_fingerprint=record["target_fingerprint"],
                        reference_fingerprint=ref["target_fingerprint"],
                        reference_status=ref["status"],
                    )
                    coefficient_uncertainty = compare_with_mc(
                        saved["coefficients"],
                        coefficient_values(refs[target][2])[0],
                        coefficient_values(refs[target][2])[1],
                        cfg,
                    )
                    saved_mc[str(checkpoint["step"])] = {
                        "physical": uncertainty,
                        "coefficient_comparison": coefficient_comparison,
                        "coefficients": coefficient_uncertainty,
                    }
                    for feature, uncertainty_row in zip(
                        coefficient_comparison["features"],
                        coefficient_uncertainty,
                        strict=True,
                    ):
                        details.append(
                            {
                                "target": target,
                                "guide": family,
                                "seed": seed,
                                "step": checkpoint["step"],
                                "draw_collection": "checkpoint_common_4096",
                                **feature,
                                **{
                                    k: v
                                    for k, v in uncertainty_row.items()
                                    if not isinstance(v, dict)
                                },
                                **{
                                    f"interval_width_ratio_mcse_upper_bound_{level}": value
                                    for level, value in uncertainty_row[
                                        "interval_ratio_mcse_upper_bounds"
                                    ].items()
                                },
                            }
                        )
                write_json(
                    directory / "checkpoint_mc_uncertainty.json",
                    {
                        "analysis": "saved checkpoint draws only; no refit or additional draw collection",
                        "steps": saved_mc,
                    },
                )
                for checkpoint in record["checkpoints"] + [
                    {
                        "step": 40000,
                        "comparison": record["final"],
                        "mc_uncertainty": record["final"]["mc_uncertainty"],
                        "final_fresh_draws": True,
                    }
                ]:
                    uncertainty = {
                        r["name"]: r
                        for r in checkpoint.get("mc_uncertainty", [])
                    }
                    for feature in checkpoint["comparison"]["features"]:
                        details.append(
                            {
                                "target": target,
                                "guide": family,
                                "seed": seed,
                                "step": checkpoint["step"],
                                "draw_collection": "final_fresh_16384"
                                if checkpoint.get("final_fresh_draws")
                                else "checkpoint_common_4096",
                                **feature,
                                **{
                                    k: v
                                    for k, v in uncertainty.get(
                                        feature["name"], {}
                                    ).items()
                                    if not isinstance(v, dict)
                                },
                                **{
                                    f"interval_width_ratio_mcse_upper_bound_{level}": value
                                    for level, value in uncertainty.get(
                                        feature["name"], {}
                                    )
                                    .get(
                                        "interval_ratio_mcse_upper_bounds", {}
                                    )
                                    .items()
                                },
                            }
                        )
                for feature, uncertainty in zip(
                    record["final"]["coefficient_comparison"]["features"],
                    record["final"]["coefficient_mc_uncertainty"],
                    strict=True,
                ):
                    details.append(
                        {
                            "target": target,
                            "guide": family,
                            "seed": seed,
                            "step": 40000,
                            "draw_collection": "final_fresh_16384",
                            **feature,
                            **{
                                k: v
                                for k, v in uncertainty.items()
                                if not isinstance(v, dict)
                            },
                            **{
                                f"interval_width_ratio_mcse_upper_bound_{level}": value
                                for level, value in uncertainty[
                                    "interval_ratio_mcse_upper_bounds"
                                ].items()
                            },
                        }
                    )
            if not available:
                rows.append(
                    {
                        "target": target,
                        "guide": family,
                        "status": "no_completed_vi",
                        "reference_status": ref["status"],
                    }
                )
                continue
            primary = [
                r
                for fit in available
                for r in fit["final"]["mc_uncertainty"]
                if r["name"].startswith(("log_S", "log_band"))
                or r["name"] in ("log_sigma", "sigma")
            ]
            late = [
                x
                for fit in stabilities[target]["fits"]
                if fit["guide"] == family
                for x in fit["late"]
            ]
            pairs = [
                x
                for x in stabilities[target]["seed_pairs"]
                if x["guide"] == family
            ]
            all_stability = late + pairs
            mmd = [x["final"]["functional_mmd"] for x in available]
            native = [
                r for x in available for r in x["native_psis"].get("runs", [])
            ]
            weights = [r for x in available for r in x["weights"]]
            passes = [
                all(
                    r["point_within_mean_screen"]
                    and r["point_within_sd_screen"]
                    for r in x["final"]["mc_uncertainty"]
                    if r["name"].startswith(("log_S", "log_band"))
                    or r["name"] in ("log_sigma", "sigma")
                )
                for x in available
            ]
            row = {
                "target": target,
                "guide": family,
                "status": "complete" if len(available) == 3 else "incomplete",
                "reference_status": ref["status"],
                "reference_attempt": ref["attempt"],
                "completed_seeds": [x["optimization_seed"] for x in available],
                "primary_max_absolute_mean_difference": max(
                    abs(r["standardized_mean_difference"]) for r in primary
                ),
                "primary_sd_ratio_range": [
                    min(r["sd_ratio"] for r in primary),
                    max(r["sd_ratio"] for r in primary),
                ],
                "primary_point_pass_count": sum(passes),
                "primary_status_counts": {
                    key: sum(
                        r[s] == key
                        for r in primary
                        for s in ("mean_status", "sd_status")
                    )
                    for key in (
                        "within_screen",
                        "outside_screen",
                        "mc_precision_limited",
                        "unavailable",
                    )
                },
                "stability_point_pass": len(all_stability) == 9
                and all(x["point_within_screen"] for x in all_stability),
                "stability_status": "outside_screen"
                if any(x["status"] == "outside_screen" for x in all_stability)
                else "within_screen"
                if len(all_stability) == 9
                and all(x["status"] == "within_screen" for x in all_stability)
                else "mc_precision_limited",
                "stability_max_mean_change": max(
                    x["max_mean_change_reference_sd"] for x in all_stability
                ),
                "stability_max_sd_change": max(
                    x["max_relative_sd_change"] for x in all_stability
                ),
                "functional_mmd": mmd,
                "latent_mmd": [x["final"]["latent_mmd"] for x in available],
                "native_psis_runs": native,
                "packed_weights": weights,
                "vi_workflow_seconds": [x["wall_seconds"] for x in available],
                "peak_memory_mib": [x["peak_memory_mib"] for x in available],
                "dense_field": [
                    fields[target]["vi"][family, seed]["comparison"]
                    for seed in cfg["optimization_seeds"]
                    if (family, seed) in fields[target]["vi"]
                ],
            }
            rows.append(row)
    columns = sorted({k for row in details for k in row})
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(details)
    summary = {
        "reference_attempts": reference_rows,
        "comparisons": rows,
        "completed_vi_jobs": len(records),
        "analysis_seconds": perf_counter() - started,
        "storage_bytes": sum(
            p.stat().st_size for p in out.rglob("*") if p.is_file()
        ),
        "forbidden_jobs_executed": [],
        "reserved_data_seeds_generated": [],
        "extra_diagnostic_draw_collections": 0,
    }
    write_json(out / "summary.json", summary)
    plot_results(out, arrays, fields, records, refs, stabilities)
    write_report(out, cfg, summary, records, stabilities)


def plot_results(
    out,
    arrays,
    fields,
    records,
    refs,
    stabilities,
    late_labels=("30k→35k", "35k→40k", "Final seed pairs"),
    roughness_unexecuted_reason="Hierarchical target unexecuted\nFixed-stage stability gate not passed",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {"diag": "#c66a16", "mvn": "#2079ab"}
    styles = {7101: "-", 7102: "--", 7103: ":"}
    targets = list(fields)
    for kind in ("psd", "mean_error", "sd_ratio"):
        fig, axs = plt.subplots(
            max(1, len(targets)),
            1,
            figsize=(9, 4.8 * max(1, len(targets))),
            squeeze=False,
        )
        if not targets:
            axs[0, 0].text(
                0.5,
                0.5,
                "No usable matched reference",
                ha="center",
                transform=axs[0, 0].transAxes,
            )
        for index, target in enumerate(targets):
            ax = axs[index, 0]
            p = fields[target]["reference"]
            if kind == "psd":
                f = arrays["overlay_frequency"]
                interval = p["overlay"]["psd_quantile_05_50_95"]
                ax.fill_between(
                    f,
                    interval[0],
                    interval[2],
                    color="0.4",
                    alpha=0.18,
                    label="NUTS 90% interval",
                )
                ax.plot(
                    f,
                    interval[1],
                    color="black",
                    linewidth=1.5,
                    label="NUTS median",
                )
                ax.plot(
                    f,
                    ar4_psd(f),
                    color="#784d95",
                    linewidth=1.6,
                    label="AR(4) truth: context",
                )
                ax.set_yscale("log")
                ax.set_ylabel("One-sided PSD [x²/Hz]")
            else:
                f = arrays["dense_frequency"]
                limits = [-0.1, 0.1] if kind == "mean_error" else [0.9, 1.1]
                ax.axhspan(
                    *limits,
                    color="#46a276",
                    alpha=0.12,
                    label="Development screen",
                )
                ax.axhline(
                    0 if kind == "mean_error" else 1, color="0.4", linewidth=1
                )
                ax.set_ylabel(
                    "Mean difference / NUTS SD"
                    if kind == "mean_error"
                    else "VI SD / NUTS SD"
                )
            for (family, seed), stats in fields[target]["vi"].items():
                if kind == "psd":
                    interval = stats["overlay"]["psd_quantile_05_50_95"]
                    ax.fill_between(
                        f,
                        interval[0],
                        interval[2],
                        color=colors[family],
                        alpha=0.055,
                    )
                    y = interval[1]
                else:
                    comparison = stats["comparison"]
                    key = (
                        "standardized_log_mean_difference"
                        if kind == "mean_error"
                        else "sd_ratio"
                    )
                    se = comparison[
                        "mean_mcse"
                        if kind == "mean_error"
                        else "sd_ratio_mcse"
                    ]
                    y = comparison[key]
                    ax.fill_between(
                        f,
                        y - 2 * se,
                        y + 2 * se,
                        color=colors[family],
                        alpha=0.045,
                    )
                ax.plot(
                    f,
                    y,
                    color=colors[family],
                    linestyle=styles[seed],
                    linewidth=1.25,
                    label=f"{family}, seed {seed}",
                )
            ax.set_title(
                f"{target.capitalize()} smoothing — all completed seeds"
            )
            ax.set_xlabel("Frequency [cycles/sample]")
            ax.set_xlim(f[0], f[-1])
            ax.legend(fontsize=8, ncol=3, loc="best")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{kind}.png", dpi=170)
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    if "hierarchical" not in refs:
        for ax, label in zip(axs, ("Physical σ", "log σ"), strict=True):
            ax.set_axis_off()
            ax.text(
                0.5,
                0.62,
                label,
                ha="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.42,
                roughness_unexecuted_reason,
                ha="center",
                transform=ax.transAxes,
            )
    else:
        for index, ax in enumerate(axs):
            transform = np.log if index else np.asarray
            values = transform(
                refs["hierarchical"][2].sigma_delta_0.values
            ).ravel()
            bins = np.linspace(*np.quantile(values, [0.001, 0.999]), 60)
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color="black",
                linewidth=2,
                label="NUTS",
            )
            for (target, family, seed), (directory, _) in records.items():
                if target == "hierarchical":
                    q = transform(
                        xr.load_dataset(
                            directory / "posterior.nc"
                        ).sigma_delta_0.values
                    ).ravel()
                    ax.hist(
                        q,
                        bins=bins,
                        density=True,
                        histtype="step",
                        color=colors[family],
                        linestyle=styles[seed],
                        label=f"{family} {seed}",
                    )
            ax.set_xlabel("log σ" if index else "σ")
            ax.set_ylabel("Posterior density")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures/roughness.png", dpi=170)
    plt.close(fig)
    fig, axs = plt.subplots(
        max(1, len(stabilities)),
        2,
        figsize=(10, 4.5 * max(1, len(stabilities))),
        squeeze=False,
    )
    for index, (target, result) in enumerate(stabilities.items()):
        for family in ("diag", "mvn"):
            for seed in styles:
                fits = [
                    x
                    for x in result["fits"]
                    if x["guide"] == family and x["seed"] == seed
                ]
                if not fits:
                    continue
                for column, key in enumerate(
                    ("max_mean_change_reference_sd", "max_relative_sd_change")
                ):
                    axs[index, column].plot(
                        [0, 1],
                        [x[key] for x in fits[0]["late"]],
                        marker="o",
                        color=colors[family],
                        linestyle=styles[seed],
                        label=f"{family} {seed}",
                    )
            seed_pairs = [
                x for x in result["seed_pairs"] if x["guide"] == family
            ]
            for column, key in enumerate(
                ("max_mean_change_reference_sd", "max_relative_sd_change")
            ):
                axs[index, column].scatter(
                    np.full(len(seed_pairs), 2)
                    + (0.05 if family == "mvn" else -0.05),
                    [x[key] for x in seed_pairs],
                    color=colors[family],
                    marker="x",
                    s=40,
                )
        for column, ax in enumerate(axs[index]):
            ax.axhline(
                0.1 if column == 0 else 0.05,
                color="0.4",
                linestyle="--",
                label="Screen",
            )
            ax.set_xticks([0, 1, 2], late_labels)
            ax.set_ylabel(
                "Max |mean drift| / NUTS SD"
                if column == 0
                else "Max relative SD drift"
            )
            ax.set_title(
                f"{target.capitalize()}: primary features and coefficients"
            )
            ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figures/stability.png", dpi=170)
    plt.close(fig)


def write_report(out, cfg, summary, records, stabilities):
    complete = [
        row
        for row in summary["comparisons"]
        if row.get("status") == "complete"
    ]
    successful = [
        row
        for row in complete
        if row["primary_point_pass_count"] == 3 and row["stability_point_pass"]
    ]
    fixed_good = [
        row
        for row in complete
        if row["target"] == "fixed" and row["primary_point_pass_count"] == 3
    ]
    lead = (
        "A tested recipe meets the primary point and repeatability screens on this development target; Monte Carlo classifications and joint diagnostics below delimit that statement."
        if successful
        else "No tested recipe completes the full declared validation milestone. "
        + (
            "At 40k, full covariance reproduces the fixed-smoothing PSD and band-power uncertainty across all three seeds, but the two-interval stability gate is not met. Inferred smoothing remains unexecuted."
            if fixed_good
            else "The executed comparisons and unresolved sampling/optimization gates are reported below."
        )
    )
    text = [
        "# Stationary AR(4): VI versus NUTS, stages 0–3",
        "",
        lead,
        "",
        "This report describes executed development work on data seed 60101, not the proposed protocol or a confirmation study. `audit.md`, `protocol.json`, `dry_run_manifest.json`, numerical `frozen_target/arrays.npz` and the append-only attempt ledger precede the fits. The native frequency-only centered model, likelihood, data, prior and basis are identical between algorithms within each target. Fixed and hierarchical physical target IDs are distinct.",
        "",
        "| Target / guide | Reference | Primary mean error / SD; SD ratios, all seeds | Stability | Functional MMD² / NUTS baseline | Native / packed k; packed smoothed ESS fraction | VI workflow seconds per seed | Limitation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["comparisons"]:
        if row.get("status") != "complete":
            text.append(
                f"| {row['target']} / {row['guide']} | {row['reference_status']} | unavailable | unexecuted | unavailable | unavailable | unavailable | {row['status']} |"
            )
            continue
        k = value_range([x.get("k") for x in row["native_psis_runs"]])
        packed_k = value_range([x.get("k") for x in row["packed_weights"]])
        ess = value_range(
            [x.get("smoothed_ess_fraction") for x in row["packed_weights"]]
        )
        mmd = value_range([x["vi_vs_nuts"] for x in row["functional_mmd"]])
        base = value_range([x["nuts_vs_nuts"] for x in row["functional_mmd"]])
        sd = value_range(row["primary_sd_ratio_range"])
        text.append(
            f"| {row['target']} / {row['guide']} | accepted ({row['reference_attempt']}) | max {row['primary_max_absolute_mean_difference']:.4f}; {sd} | {row['stability_status']}; point pass {row['stability_point_pass']} | {mmd} / {base} | {k} / {packed_k}; {ess} | {value_range(row['vi_workflow_seconds'])} | {row['primary_point_pass_count']}/3 seeds pass final primary point screens; status counts {row['primary_status_counts']} |"
        )
    text += [
        "",
        "Primary features are nine log PSD values plus three log band powers, and physical/log roughness when inferred. Raw bands are secondary. All final comparisons use 16,384 fresh full joint guide draws. Checkpoints use 4,096 common-key draws; their precision limits are retained. `comparison.csv` records every frozen scalar, checkpoint, seed, coefficients, 50/90/95% width ratio, Wasserstein and combined MC uncertainty. Mean and SD screen statuses use a two-MCSE uncertainty interval; borderline estimates remain `mc_precision_limited`. Reference MCSE estimates account for chain autocorrelation.",
        "",
        "## References, attempts and exact scope",
        "",
    ]
    for ref in summary["reference_attempts"]:
        line = f"- {ref['target']}/{ref['attempt']}: {ref['status']}; four planned chains; {ref['retained_draws_per_chain']} retained draws per chain; {ref['wall_seconds']:.2f} s."
        if ref.get("health"):
            h = ref["health"]
            line += f" Divergences {h['divergences']}, cap hits {h['depth_saturation']}, maximum rank split R-hat {max(h['rhat']):.5f}, minimum bulk/tail ESS {min(h['ess_bulk']):.0f}/{min(h['ess_tail']):.0f}, minimum BFMI {min(h['bfmi']):.3f}, primary mean MCSE/SD max {ref['primary_mean_mcse_sd_max']:.5f}."
        if ref.get("error"):
            line += f" Error retained: `{ref['error']}`."
        text.append(line)
    text += [
        "",
        f"Completed VI jobs: {summary['completed_vi_jobs']}. Executed seeds: "
        + ", ".join(f"{t}/{g}/{s}" for t, g, s in sorted(records))
        + ". Every completed fit used 40,000 actual updates and eight full-data training particles. Objective evaluation uses 256 particles×8 fixed keys and three separate keys; PSIS uses 4,096 joint draws×3 diagnostic seeds. Numeric guide checkpoints exist at 5k, 10k, 20k, 30k, 35k and 40k, saved before expensive analysis. These contain guide parameters, not optimizer state.",
        "",
        "The fixed initial attempt failed before sampling because the experimental independent-initialization callback was passed as a function rather than the partial expected by NumPyro. That failure and traceback are retained. A deterministic initialization test verifies the corrected callback and four distinct starts. The single authorized fixed-target repair used its already frozen 2k warmup/4k draws per chain, acceptance 0.99, depth 10 and dense mass; no observations or target components changed. `repair_reason.json` was written before launch. This exhausts the fixed-target reference repair budget. The original planned settings were 1k/2k, acceptance 0.95. No original posterior is claimed.",
        "",
        "The first controller reached the end of the six fixed fits before its offline summary function was available and failed at report generation. Its log is retained separately; subsequent summarization reads saved draws only and does not refit or replace inference attempts.",
        "",
        "No hierarchical reference or fit ran when the fixed stability gate failed. Seeds 60102–60106 were neither generated nor inspected. No parameterisation sweep, time-varying investigation, new likelihood, flows, 80k tail, optimizer-state continuation, reweighting or extra diagnostic precision collection ran. The routine suite includes existing small stationary and time-varying regression fits, as required when changing shared inference.",
        "The runner also made one separately keyed 16,384-draw packed-guide collection per fit (seed 8401) for nonredundant parameter MMD; only the first 512+512 samples enter its comparison/baseline. This collection is part of the executed diagnostic recipe, rather than a precision extension or an optimizer run. Posterior features use the distinct fresh draws from fit_vi; checkpoint collections use seed 8301. These actual draw counts and seeds are recorded in executed_draw_design.json alongside the original pre-fit protocol.",
        "",
        "## Posterior agreement and stability",
        "",
    ]
    for row in complete:
        text.append(
            f"- {row['target']}/{row['guide']}: dense-field standardized log-mean RMS {value_range([x['rms_standardized_log_mean_difference'] for x in row['dense_field']])}; worst-location mean discrepancy {value_range([x['worst_standardized_log_mean_difference'] for x in row['dense_field']])}; dense SD range {value_range([y for x in row['dense_field'] for y in x['sd_ratio_range']])}. Physical quadrature is normalized over the frozen 257-node [.02,.48] grid. The full fit domain is shown separately in the PSD overlay."
        )
        coefficients = [
            r
            for (t, g, _), (_, fit) in records.items()
            if t == row["target"] and g == row["guide"]
            for r in fit["final"]["coefficient_mc_uncertainty"]
        ]
        text.append(
            f"  Complete coefficient means: max discrepancy {max(abs(r['standardized_mean_difference']) for r in coefficients):.4f} reference SD; SD ratios {value_range([r['sd_ratio'] for r in coefficients])}. Primary final point screen passes {row['primary_point_pass_count']}/3 seeds. Geometry/correlation matrices and compact nonredundant latent MMD are retained per seed."
        )
        text.append(
            f"  Across all six late interval comparisons and three final seed pairs, worst mean drift {row['stability_max_mean_change']:.4f} reference SD and worst SD drift {row['stability_max_sd_change']:.2%}. All paired negative-ELBO changes, their MCSE and independent-key objective estimates are saved; a flat objective alone does not certify posterior stability."
        )
    if "fixed" in stabilities:
        result = stabilities["fixed"]
        text += [
            "",
            "The hierarchy gate requires one fit passing both 30k→35k and 35k→40k point screens in the declared primary physical features and every coefficient, without resolved outside-screen drift. The same thresholds were retained after seeing the results. All seed pairs and both late intervals are assessed separately; end-point seed agreement does not erase earlier movement.",
        ]
        for family in cfg["guides"]:
            fits = [x for x in result["fits"] if x["guide"] == family]
            for interval in (0, 1):
                late = [x["late"][interval] for x in fits]
                text.append(
                    f"- {family}, {'30k→35k' if interval == 0 else '35k→40k'}: max mean drift across seeds {value_range([x['max_mean_change_reference_sd'] for x in late])}; max relative SD drift {value_range([x['max_relative_sd_change'] for x in late])}; statuses {[x['status'] for x in late]}."
                )
        text += [
            "",
            "At 40k the full-covariance Gaussian has close selected-functional and complete coefficient agreement with the diagnosed conditional reference. Diagonal VI can match pointwise spectral uncertainty while missing integrated band uncertainty and coefficient covariance. Its measured discrepancy is specific to this guide, coordinates and recipe; it is not a proof of the irreducibly best diagonal approximation. Fixed-smoothing agreement does not establish inferred-smoothing agreement or explain a hierarchical funnel.",
        ]
    text += [
        "",
        "## Joint proposal diagnostics and interpretation",
        "",
        "Native NumPyro k and the tested packed-density adapter use the complete prior/factor/model density, guide density and exactly one support Jacobian in common unconstrained coordinates. Fixed sigma is excluded from packing. All raw/smoothed ESS values, ESS fractions, maximum weights, native/adapter status and nonfinite cases remain in complete.json/guide metadata. A k≥0.7 warns about importance sampling; it is not the primary mean/SD screen, and low k does not prove equality. No weights were used to alter these posterior summaries.",
        "Native NumPyro splits each diagnostic seed into per-particle guide traces; the packed adapter samples one vectorized posterior collection. They therefore use different realized particles even for the same seed, so their finite-sample k values need not coincide. Both repeated ranges are shown, with adapter weights/ESS labelled separately; no discrepancy is hidden by averaging k.",
        "",
        "Functional and nonredundant latent MMD use 512 matched draws, whitening trained on a disjoint reference subset and the frozen three bandwidths. VI–VI and NUTS–NUTS baselines are retained, including negative unbiased values. These are descriptive comparisons with autocorrelated reference draws, without IID p-values, universal thresholds or an equality claim.",
        "",
        "## Cost, verification and reproduction",
        "",
        f"NumPyro {cfg['provenance']['versions']['numpyro']}; JAX {cfg['provenance']['versions']['jax']}; x64 {cfg['provenance']['x64_enabled']}; backend {cfg['provenance']['backend']}. Repository SHA {cfg['provenance']['repository_sha']}; dirty-source hashes and installed versions recorded for each job. No dependency update was required. Frozen native basis and penalty are float64; reference/guide parameter dtypes are recorded.",
        "",
        f"Inherited schedule: `{cfg['schedule']}` from `{cfg['schedule_source']}` with its saved hash. Adam default beta/epsilon choices and clipping norm 1 are unchanged. Initialization/first compiled update, optimization, checkpoint objective/PSIS diagnostics, reconstruction and I/O timings are retained in each fit; workflow seconds in the table include diagnostic work and cannot imply speed superiority. Peak fit memory MiB {value_range([fit['peak_memory_mib'] for _, fit in records.values()])}; current artifact bytes {summary['storage_bytes']:,}. Offline field/report timing is separate.",
        "",
        "Runnable commands from the audited worktree:",
        "",
        "```bash",
        "PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary.py prepare",
        "PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary.py all",
        "PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary.py summarize",
        "PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python -m pytest -q",
        "```",
        "",
        "`all` preserves completed jobs and failed attempts and applies the frozen repair/gate budget; it never resumes optimizer state. `summarize` is offline. The complete routine suite passed 139 tests with 29 warnings in 70.89 seconds, including marked slow regressions. Artifact verification also passed: all six saved guides reproduce the stored log joint and log guide exactly on checked saved points; all 36 parameter checkpoints round-trip; 1,890 comparison rows contain MC uncertainty. Tests, retained initial failures, and visual review are recorded in verification.json and logs. Figures: PSD median/90% intervals with truth only as context, standardized mean difference, SD ratio, explicit unexecuted roughness panel, and late/seed stability. Mean/SD curve shading represents two pointwise MCSE, not simultaneous confidence bands. Native complete reconstruction is checked against exp(Bc); scalar likelihood, conditional gradients, fixed packing, AR stationarity/sign/normalization, independent starts and persistence-before-analysis failure are tested.",
        "",
        "## Next supported experiment, proposed only",
        "",
        "The supported next experiment is a bounded refinement of full-covariance optimization on this same fixed-smoothing target, using the accepted reference and all three optimization seeds. Predeclare a short low-rate tail beyond 40k and two new late checkpoints; preserve the inherited schedule prefix and rerun it because guide checkpoints do not store optimizer state. Check the same coefficient/primary mean and SD drifts and paired objectives. Do not infer an intrinsic guide-shape error from the earlier movement. Proceed to the matched hierarchical target only after that stability milestone, without demanding diagonal VI pass.",
        "",
        "This study stops here. It does not establish whitening, coverage, SBC, model adequacy, robustness across AR records, a speed advantage, inferred smoothing, or TV validation. The truth overlay is descriptive, not calibration.",
    ]
    if "fixed" in stabilities and records:
        first = next(
            directory
            for (target, _, _), (directory, _) in records.items()
            if target == "fixed"
        )
        geometry = json.loads((first / "joint_geometry.json").read_text())
        adjacent = np.diag(np.asarray(geometry["nuts_correlation"]), 1)
        text += [
            "",
            f"The conditional NUTS coefficient geometry shows adjacent correlations from {adjacent.min():.3f} to {adjacent.max():.3f}; coefficient 15–16 correlation is {geometry['nuts_correlation'][15][16]:.3f}. Diagonal guides cannot retain those correlations. The recorded band SD inflation and reduced coefficient SDs are consistent with that missing covariance, while the full-covariance fit closely reproduces them. This is a measured covariance diagnostic, not a proof of an optimizer-independent family optimum.",
        ]
    text += ["", "## Figures", ""]
    for name, label in (
        ("psd", "Full-domain PSD"),
        ("mean_error", "Dense log-PSD mean discrepancy"),
        ("sd_ratio", "Dense log-PSD SD ratios"),
        ("roughness", "Roughness stage status"),
        ("stability", "Checkpoint and seed stability"),
    ):
        text += [f"![{label}]({out / 'figures' / (name + '.png')})", ""]
    (out / "report.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
