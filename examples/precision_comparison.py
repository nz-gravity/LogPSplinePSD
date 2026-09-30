"""Bounded stationary univariate precision study; see docs/precision.md.

Shared preparation is host float64. Sampling workers select X64 at startup,
never inside a model or fit. This is an experiment, not a public precision API.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

# Set once before imports; worker environments explicitly supply 0 or 1.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import arviz_stats as azs
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import potential_energy
from scipy.special import digamma, polygamma
from scipy.stats import norm

from log_psplines.basis import SplineBasis
from log_psplines.inference.model import _blocked_channel_model
from log_psplines.likelihoods.whittle import (
    power_whittle_log_likelihood,
    whittle_log_likelihood,
)
from log_psplines.models.spectrum import build_spline

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2718, 2719, 2720, 2721, 2722)
# Frozen before pilot: engineering limits, not universal precision guarantees.
DELTA_LIMIT = 1e-2
GRAD_ABS_LIMIT = 5e-4
GRAD_SCALED_LIMIT = 5e-6
MEAN_MARGIN = 0.1
SD_RATIO_BOUNDS = (0.95, 1.05)


def constant_model(power: jax.Array, count: int) -> None:
    """Test-only IG(4,3) prior, proper complex likelihood, duration=ENBW=1."""
    spectrum = numpyro.sample(
        "spectrum",
        dist.InverseGamma(
            jnp.asarray(4.0, power.dtype), jnp.asarray(3.0, power.dtype)
        ),
    )
    numpyro.factor(
        "likelihood",
        whittle_log_likelihood(
            jnp.log(spectrum)[None], power[None], count=count
        ),
    )


def channel_kwargs(fixture: dict, dtype: Any) -> dict:
    """Use the production channel-zero model, with explicit floating inputs."""
    nf = fixture["basis"].shape[0]
    return dict(
        channel_index=0,
        u_re_channel=jnp.asarray(fixture["u_re"], dtype=dtype),
        u_im_channel=jnp.asarray(fixture["u_im"], dtype=dtype),
        u_re_prev=jnp.zeros((nf, 0, 1), dtype=dtype),
        u_im_prev=jnp.zeros((nf, 0, 1), dtype=dtype),
        basis_delta=jnp.asarray(fixture["basis"], dtype=dtype),
        penalty_delta=jnp.asarray(fixture["penalty"], dtype=dtype),
        basis_theta_re_by_component=(),
        penalty_theta_re_by_component=(),
        basis_theta_im_by_component=(),
        penalty_theta_im_by_component=(),
        duration=1.0,
        Nb=int(fixture["count"]),
        Nh=1,
        enbw=1.0,
        eta=1.0,
        roughness_scale=1.28,
        smoothing_parameterization="centered",
    )


def fixture(seed: int, nf: int = 96, knots: int = 6) -> dict:
    """Proper complex draws CN(0,S), compressed to a scalar sufficient factor."""
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Shared basis preparation requires JAX_ENABLE_X64=1"
        )
    basis = SplineBasis.from_knots(
        np.linspace(0.0, 1.0, nf), np.linspace(0.0, 1.0, knots)
    )
    b, p = np.asarray(basis.basis), np.asarray(basis.penalty)
    w = 0.5 * np.sin(np.linspace(0.0, 2 * np.pi, b.shape[1]))
    truth = np.exp(b @ w)
    rng = np.random.default_rng(seed)
    x = np.sqrt(truth[:, None] / 2) * (
        rng.normal(size=(nf, 4)) + 1j * rng.normal(size=(nf, 4))
    )
    return dict(
        basis=b,
        penalty=p,
        truth=truth,
        u_re=np.sqrt(np.sum(np.abs(x) ** 2, axis=1))[:, None],
        u_im=np.zeros((nf, 1)),
        count=np.array(4),
        initial_weights=np.linalg.lstsq(
            b, np.log(np.mean(np.abs(x) ** 2, axis=1)), rcond=None
        )[0],
    )


def tree_dtypes(tree: Any) -> dict[str, str]:
    return {
        jax.tree_util.keystr(path): str(leaf.dtype)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        if hasattr(leaf, "dtype")
    }


def direct_target(z: dict, kw: dict) -> tuple[float, dict]:
    """Independent NumPy expression for centered target plus sigma Jacobian."""
    w = np.asarray(z["weights_delta_0"], dtype=np.float64)
    s = float(z["sigma_delta_0"])
    b, p = (
        np.asarray(kw["basis_delta"], float),
        np.asarray(kw["penalty_delta"], float),
    )
    power = np.sum(
        np.asarray(kw["u_re_channel"], float) ** 2
        + np.asarray(kw["u_im_channel"], float) ** 2,
        axis=1,
    )
    log_s = b @ w
    n = kw["Nb"] * kw["Nh"]
    roughness = w @ p @ w
    scale = float(kw["roughness_scale"])
    ell = -n * np.sum(log_s) - np.sum(power * np.exp(-log_s))
    ell += (
        0.5 * np.log(2 / np.pi)
        - np.log(scale)
        - 0.5 * np.exp(2 * s) / scale**2
    )
    ell += (1 - w.size) * s - 0.5 * np.exp(-2 * s) * roughness
    grad = dict(
        sigma_delta_0=np.array(
            1 - w.size - np.exp(2 * s) / scale**2 + np.exp(-2 * s) * roughness
        ),
        weights_delta_0=b.T @ (power * np.exp(-log_s) - n)
        - np.exp(-2 * s) * p @ w,
    )
    return float(ell), grad


def mixed_target(z: dict, kw: dict, penalty64: jax.Array) -> jax.Array:
    """Diagnostic only: float32 fields/residuals, retained float64 prior and sums.

    Gradients retain input dtype. This does not certify a mixed NUTS policy.
    """
    w32 = z["weights_delta_0"]
    log_s = build_spline(kw["basis_delta"], w32)
    variance64 = jnp.exp(jnp.clip(log_s, -80.0, 80.0)).astype(jnp.float64)
    power32 = jnp.sum(
        kw["u_re_channel"] ** 2 + kw["u_im_channel"] ** 2, axis=-1
    )
    ell = -jnp.asarray(kw["Nb"] * kw["Nh"], jnp.float64) * jnp.sum(
        jnp.log(variance64)
    )
    ell -= jnp.sum(power32.astype(jnp.float64) / variance64)
    w, s = w32.astype(jnp.float64), z["sigma_delta_0"].astype(jnp.float64)
    p = penalty64
    ell += (
        0.5 * jnp.log(2 / jnp.pi)
        - jnp.log(1.28)
        - 0.5 * jnp.exp(2 * s) / 1.28**2
    )
    return ell + (1 - w.size) * s - 0.5 * jnp.exp(-2 * s) * (w @ p @ w)


def preparation_audit() -> dict:
    """Audit the actual default path separately from explicit policies."""
    from log_psplines import StationaryConfig
    from log_psplines.example_datasets.varma_data import VARMAData
    from log_psplines.inference.model import (
        channel_model_kwargs,
        prepare_model,
    )
    from log_psplines.logger import set_level
    from log_psplines.preprocessing.spectral import preprocess_to_freq_domain

    set_level("WARNING")
    config = StationaryConfig(n_knots=5, Nb=2)
    example = VARMAData.ar(order=1, n_samples=256, fs=64.0, seed=8)
    data = preprocess_to_freq_domain(example.ts, config)
    kw, spline = prepare_model(data, config)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(_blocked_channel_model, jax.random.PRNGKey(31))
    ).get_trace(**channel_model_kwargs(kw, 0))
    return dict(
        normalized_observations=str(data.u_re.dtype),
        basis_before_prepare=str(spline.diagonal_models[0].basis.dtype),
        prepared=tree_dtypes(kw),
        sites={
            name: str(site["value"].dtype)
            for name, site in trace.items()
            if site["type"] in ("sample", "deterministic")
            and hasattr(site["value"], "dtype")
        },
    )


def target_errors(
    f: dict, smoothing_log: float = 0.0, points: list[dict] | None = None
) -> dict:
    """Separate rounding, arithmetic and total local target errors."""
    kws = {
        name: channel_kwargs(f, dtype)
        for name, dtype in (("fp64", jnp.float64), ("fp32", jnp.float32))
    }
    rounded_kw = dict(kws["fp32"])
    for name, value in rounded_kw.items():
        if hasattr(value, "dtype") and jnp.issubdtype(
            value.dtype, jnp.floating
        ):
            rounded_kw[name] = value.astype(jnp.float64)
    funcs = {
        name: jax.jit(
            jax.value_and_grad(
                lambda z, kw=kw: (
                    -potential_energy(_blocked_channel_model, (), kw, z)
                )
            )
        )
        for name, kw in (*kws.items(), ("rounded64", rounded_kw))
    }
    # Binary fractions isolate parameter arithmetic from parameter rounding.
    k = f["basis"].shape[1]
    points = points or [
        dict(
            sigma_delta_0=np.array(smoothing_log + ds),
            weights_delta_0=(np.arange(k) % 5 - 2) / 16 + dw,
        )
        for ds, dw in ((0.0, 0.0), (1 / 16, 1 / 128), (-1 / 8, -1 / 32))
    ]
    funcs["mixed_diagnostic"] = jax.jit(
        jax.value_and_grad(
            lambda z: mixed_target(
                z, kws["fp32"], kws["fp64"]["penalty_delta"]
            )
        )
    )
    values = {name: [] for name in funcs}
    max_abs = max_scaled = max_recon = direct_delta = 0.0
    mixed_abs = mixed_scaled = 0.0
    ref_direct = None
    grad_dtypes = {}
    for z in points:
        results = {}
        for name, fn in funcs.items():
            dtype = (
                jnp.float32
                if name in ("fp32", "mixed_diagnostic")
                else jnp.float64
            )
            source = (
                jax.tree.map(
                    lambda x: np.asarray(x, np.float32).astype(np.float64), z
                )
                if name == "rounded64"
                else z
            )
            zp = jax.tree.map(
                lambda x, dtype=dtype: jnp.asarray(x, dtype=dtype), source
            )
            value, grad = fn(zp)
            values[name].append(float(value))
            results[name] = grad
            grad_dtypes[name] = tree_dtypes(grad)
            assert value.dtype == (
                jnp.float64 if name == "mixed_diagnostic" else dtype
            )
        direct, gradient = direct_target(z, kws["fp64"])
        if ref_direct is None:
            ref_direct = direct
        direct_delta = max(
            direct_delta,
            abs(
                values["fp64"][-1] - values["fp64"][0] - (direct - ref_direct)
            ),
        )
        for name in gradient:
            g64 = np.asarray(results["fp64"][name], float)
            np.testing.assert_allclose(
                g64, gradient[name], rtol=2e-11, atol=1e-8
            )
            error = np.abs(np.asarray(results["fp32"][name], float) - g64)
            mixed_error = np.abs(
                np.asarray(results["mixed_diagnostic"][name], float) - g64
            )
            mixed_abs = max(mixed_abs, float(np.max(mixed_error)))
            mixed_scaled = max(
                mixed_scaled,
                float(np.max(mixed_error / np.maximum(1.0, np.abs(g64)))),
            )
            max_abs = max(max_abs, float(np.max(error)))
            max_scaled = max(
                max_scaled, float(np.max(error / np.maximum(1.0, np.abs(g64))))
            )
        s32 = np.asarray(
            jnp.exp(
                build_spline(
                    kws["fp32"]["basis_delta"],
                    jnp.asarray(z["weights_delta_0"], jnp.float32),
                )
            ),
            float,
        )
        s64 = np.exp(np.asarray(f["basis"], float) @ z["weights_delta_0"])
        assert np.isfinite(s32).all() and (s32 > 0).all()
        max_recon = max(max_recon, float(np.max(np.abs(s32 / s64 - 1))))
    delta = {name: np.asarray(v, float) - v[0] for name, v in values.items()}
    errors = dict(
        local_total=float(np.max(np.abs(delta["fp32"] - delta["fp64"]))),
        local_arithmetic=float(
            np.max(np.abs(delta["fp32"] - delta["rounded64"]))
        ),
        local_rounding=float(
            np.max(np.abs(delta["rounded64"] - delta["fp64"]))
        ),
        local_mixed_diagnostic=float(
            np.max(np.abs(delta["mixed_diagnostic"] - delta["fp64"]))
        ),
        mixed_gradient_abs=mixed_abs,
        mixed_gradient_scaled=mixed_scaled,
        gradient_abs=max_abs,
        gradient_scaled=max_scaled,
        reconstruction_relative=max_recon,
        independent_fp64_delta=direct_delta,
        gradient_dtypes=grad_dtypes,
    )
    errors["classification"] = (
        "within_budget"
        if errors["local_total"] < DELTA_LIMIT
        and max_abs < GRAD_ABS_LIMIT
        and max_scaled < GRAD_SCALED_LIMIT
        else "outside_budget"
    )
    return errors


def deterministic() -> dict:
    """Independent analytic checks and explicitly labelled stress probes."""
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Use JAX_ENABLE_X64=1 for deterministic comparisons"
        )
    analytic = {}
    for dtype in (jnp.float32, jnp.float64):
        power = jnp.asarray(12.0, dtype)
        fn = jax.jit(
            jax.value_and_grad(
                lambda v, power=power: (
                    -potential_energy(
                        constant_model,
                        (),
                        dict(power=power, count=8),
                        {"spectrum": v},
                    )
                )
            )
        )
        points = (0.0, 1 / 8, -1 / 4)
        vals = [fn(jnp.asarray(v, dtype)) for v in points]
        expected = np.array([-12 * v - 15 * np.exp(-v) for v in points])
        tol = 2e-5 if dtype == jnp.float32 else 1e-12
        np.testing.assert_allclose(
            [float(v[0]) - float(vals[0][0]) for v in vals],
            expected - expected[0],
            rtol=0.0,
            atol=tol,
        )
        np.testing.assert_allclose(
            [float(v[1]) for v in vals],
            [-12 + 15 * np.exp(-v) for v in points],
            rtol=0.0,
            atol=tol,
        )
        assert all(v.dtype == dtype and g.dtype == dtype for v, g in vals)
        # Real-component counts=2 correspond to one proper complex ordinate.
        log_s = jnp.asarray([-0.25, 0.0, 0.5], dtype)
        powers = jnp.asarray([0.5, 1.5, 2.0], dtype)
        scalar = whittle_log_likelihood(log_s, powers)
        real = power_whittle_log_likelihood(2 * powers, jnp.full(3, 2), log_s)
        np.testing.assert_allclose(scalar, real, rtol=0.0, atol=tol)
        # Pool exactly constant covariance; masks have finite model predictions.
        pooled = power_whittle_log_likelihood(
            jnp.sum(2 * powers), jnp.array(6), log_s[1]
        )
        full = power_whittle_log_likelihood(
            2 * powers, jnp.full(3, 2), jnp.zeros(3, dtype)
        )
        np.testing.assert_allclose(pooled, full, atol=tol)
        masked_grad = jax.grad(
            lambda v, dtype=dtype: power_whittle_log_likelihood(
                jnp.asarray([1.0, 0.0, 3.0], dtype), jnp.array([1, 0, 2]), v
            )
        )(log_s)
        assert masked_grad[1] == 0
        analytic[str(np.dtype(dtype))] = "passed"
    ordinary = target_errors(fixture(SEEDS[0]))
    assert ordinary["classification"] == "within_budget", ordinary
    assert ordinary["reconstruction_relative"] < 1e-6
    # No clipping/ridge/prior changes to rescue these artificial stress probes.
    high = fixture(SEEDS[0])
    high["count"] = np.array(2**24 + 1)
    high["u_re"] = np.sqrt(high["truth"] * int(high["count"]))[:, None]
    count32 = float(jnp.asarray(int(high["count"]), jnp.float32))
    assert count32 == 2**24 and count32 != int(high["count"])
    # Physical units: normalize BEFORE casting/squaring coefficients.
    physical = np.array([1e-21, 2e-21, 3e-21])
    unit_scale = np.sqrt(np.mean(physical**2))
    normalized = (physical / unit_scale).astype(np.float32)
    reconstructed = np.asarray(normalized, float) ** 2 * unit_scale**2
    np.testing.assert_allclose(reconstructed / physical**2, 1.0, rtol=2e-7)
    # Float32 can represent amplitudes here but raw outer products underflow
    # at 1e-30 amplitude. The actual pipeline's host normalization avoids it.
    tiny = physical * 1e-9
    assert np.all(tiny.astype(np.float32) ** 2 == 0)
    return dict(
        analytic=analytic,
        ordinary=ordinary,
        high_count={
            **target_errors(high),
            "exact_count": int(high["count"]),
            "float32_count": count32,
            "artificial_scaling": True,
        },
        strong_smoothing=target_errors(fixture(SEEDS[0]), -10.0),
        larger_grid=target_errors(fixture(SEEDS[0], 4096, 24)),
        physical_normalization="passed",
        raw_tiny_power="underflow",
    )


def summary_stats(draws: np.ndarray) -> dict:
    """Rank split Rhat, ESS and functional MCSE in host float64."""
    x = np.asarray(draws, dtype=np.float64)
    return dict(
        mean=float(x.mean()),
        sd=float(x.std(ddof=1)),
        rhat=float(azs.rhat(x, method="rank")),
        ess_bulk=float(azs.ess(x, method="bulk")),
        ess_tail=float(azs.ess(x, method="tail", prob=0.05)),
        mcse_mean=float(azs.mcse(x, method="mean")),
        mcse_sd=float(azs.mcse(x, method="sd")),
        q05=float(np.quantile(x, 0.05)),
        q95=float(np.quantile(x, 0.95)),
    )


def worker(args: argparse.Namespace) -> None:
    """One policy/case per process; production target and NumPyro NUTS."""
    dtype = jnp.float32 if args.policy == "fp32" else jnp.float64
    assert jax.config.x64_enabled == (args.policy == "fp64")
    started = perf_counter()
    f = dict(np.load(args.fixture)) if args.case == "spline" else {}
    if args.case == "constant":
        model = constant_model
        kwargs = dict(power=jnp.asarray(12.0, dtype), count=8)
        init = {
            "spectrum": jnp.asarray(np.linspace(-0.3, 0.3, args.chains), dtype)
        }
    else:
        model, kwargs = _blocked_channel_model, channel_kwargs(f, dtype)
        rng = np.random.default_rng(8675309)
        init = dict(
            sigma_delta_0=jnp.asarray(
                np.linspace(-0.3, 0.3, args.chains), dtype
            ),
            weights_delta_0=jnp.asarray(
                f["initial_weights"][None, :]
                + rng.normal(0.0, 0.1, (args.chains, f["basis"].shape[1])),
                dtype,
            ),
        )
    if args.chains == 1:
        init = jax.tree.map(lambda a: a[0], init)
    z = jax.tree.map(lambda a: a[0] if args.chains > 1 else a, init)
    vg = jax.jit(
        jax.value_and_grad(lambda z: potential_energy(model, (), kwargs, z))
    )
    preparation_s = perf_counter() - started
    t = perf_counter()
    lowered = vg.lower(z)
    compiled_float_types = sorted(
        set(re.findall(r"\bf(?:32|64)\b", lowered.as_text()))
    )
    compiled = lowered.compile()
    jax.block_until_ready(compiled(z))
    compile_s = perf_counter() - t
    times = []
    for _ in range(5):
        t = perf_counter()
        for _ in range(100):
            jax.block_until_ready(compiled(z))
        times.append((perf_counter() - t) / 100)
    value, gradient = compiled(z)
    kernel = NUTS(
        model,
        dense_mass=True,
        target_accept_prob=args.target_accept,
        max_tree_depth=10,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.draws,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=False,
    )
    fields = (
        "potential_energy",
        "energy",
        "num_steps",
        "accept_prob",
        "adapt_state.step_size",
    )
    t = perf_counter()
    mcmc.warmup(
        jax.random.PRNGKey(args.sampler_seed),
        init_params=init,
        extra_fields=fields,
        **kwargs,
    )
    jax.block_until_ready(mcmc.last_state)
    warmup_s = perf_counter() - t
    adapted_dtypes = tree_dtypes(mcmc.last_state)
    t = perf_counter()
    mcmc.run(
        jax.random.PRNGKey(args.sampler_seed + 1),
        extra_fields=fields,
        **kwargs,
    )
    jax.block_until_ready(mcmc.last_state)
    sampling_s = perf_counter() - t
    samples = {
        k: np.asarray(v)
        for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    extra = {
        k: np.asarray(v)
        for k, v in mcmc.get_extra_fields(group_by_chain=True).items()
    }
    # Repeat the already compiled sampling operation for timing; discard draws.
    t = perf_counter()
    mcmc.run(
        jax.random.PRNGKey(args.sampler_seed + 2),
        extra_fields=fields,
        **kwargs,
    )
    jax.block_until_ready(mcmc.last_state)
    warmed_sampling_s = perf_counter() - t
    t = perf_counter()
    if args.case == "constant":
        summaries = dict(
            log_spectrum=np.log(samples["spectrum"].astype(float)),
            spectrum=samples["spectrum"].astype(float),
        )
        truth = dict(
            log_spectrum=float(np.log(15.0) - digamma(12)), spectrum=15 / 11
        )
        true_sd = dict(
            log_spectrum=float(np.sqrt(polygamma(1, 12))),
            spectrum=float(15 / (11 * np.sqrt(10))),
        )
    else:
        weights = samples["weights_delta_0"].astype(float)
        logs = np.einsum("fk,cdk->cdf", f["basis"], weights)
        indices = (0, len(f["basis"]) // 2, len(f["basis"]) - 1)
        summaries = {f"log_psd_{i}": logs[:, :, i] for i in indices}
        summaries["band_power"] = np.trapezoid(
            np.exp(logs), dx=1 / (logs.shape[-1] - 1), axis=-1
        )
        summaries["sigma"] = samples["sigma_delta_0"].astype(float)
        truth = {f"log_psd_{i}": float(np.log(f["truth"][i])) for i in indices}
        truth["band_power"] = float(
            np.trapezoid(f["truth"], dx=1 / (len(f["truth"]) - 1))
        )
        true_sd = {}
    reconstruction_s = perf_counter() - t
    stats = {name: summary_stats(x) for name, x in summaries.items()}
    for name, s in stats.items():
        s["ess_per_sampling_second"] = s["ess_bulk"] / warmed_sampling_s
        if name in truth:
            s["truth"] = truth[name]
            s["mean_error_posterior_sd"] = (s["mean"] - truth[name]) / s["sd"]
        if name in true_sd:
            s["analytic_sd"] = true_sd[name]
    floating = {v for v in adapted_dtypes.values() if v.startswith("float")}
    assert floating == {str(np.dtype(dtype))}, adapted_dtypes
    assert value.dtype == dtype
    assert all(np.isfinite(v).all() for v in samples.values())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timings = dict(
        preparation=preparation_s,
        target_compile_first_call=compile_s,
        target_warmed_median=float(np.median(times)),
        target_warmed_repeats=times,
        warmup_with_compile=warmup_s,
        sampling_first_call=sampling_s,
        sampling_warmed=warmed_sampling_s,
        reconstruction_float64=reconstruction_s,
        total_with_timing_repeat=perf_counter() - started,
    )
    posterior_points = []
    if args.case == "spline":
        for draw in (args.draws // 4, args.draws // 2, 3 * args.draws // 4):
            posterior_points.append(
                dict(
                    sigma_delta_0=float(
                        np.log(samples["sigma_delta_0"][0, draw])
                    ),
                    weights_delta_0=samples["weights_delta_0"][0, draw]
                    .astype(float)
                    .tolist(),
                )
            )
    output = dict(
        settings=dict(
            chains=args.chains,
            warmup=args.warmup,
            draws=args.draws,
            target_accept=args.target_accept,
            dense_mass=True,
            chain_method="sequential",
        ),
        posterior_points=posterior_points,
        policy=args.policy,
        case=args.case,
        seed=args.sampler_seed,
        x64=jax.config.x64_enabled,
        summaries=stats,
        timings_s=timings,
        divergences=int(np.sum(extra["diverging"])),
        trajectory_steps_median=float(np.median(extra["num_steps"])),
        trajectory_steps_max=int(np.max(extra["num_steps"])),
        max_depth_hits=int(np.sum(extra["num_steps"] >= 2**10 - 1)),
        target_dtype=str(value.dtype),
        compiled_float_types=compiled_float_types,
        gradient_dtypes=tree_dtypes(gradient),
        adapted_state_dtypes=adapted_dtypes,
        sample_dtypes={k: str(v.dtype) for k, v in samples.items()},
        estimated_input_bytes=sum(
            v.size * v.dtype.itemsize
            for v in jax.tree.leaves(kwargs)
            if hasattr(v, "dtype")
        ),
        estimated_sample_bytes=sum(v.nbytes for v in samples.values()),
        peak_process_rss_bytes=int(
            rss if sys.platform == "darwin" else rss * 1024
        ),
    )
    Path(args.output).write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


def compare(a: dict, b: dict, k: float) -> dict:
    """Simultaneous approximate MC intervals; crossing a margin is inconclusive."""
    comparisons = {}
    good = all(
        r["divergences"] == 0
        and r["max_depth_hits"] == 0
        and all(
            s["rhat"] < 1.01 and min(s["ess_bulk"], s["ess_tail"]) >= 400
            for s in r["summaries"].values()
        )
        for r in (a, b)
    )
    for name, x in a["summaries"].items():
        y = b["summaries"][name]
        d = y["mean"] - x["mean"]
        se = np.hypot(x["mcse_mean"], y["mcse_mean"])
        margin = MEAN_MARGIN * x["sd"]
        log_ratio = np.log(y["sd"] / x["sd"])
        se_sd = np.hypot(x["mcse_sd"] / x["sd"], y["mcse_sd"] / y["sd"])
        mean_ci = [float(d - k * se), float(d + k * se)]
        sd_ci = [
            float(np.exp(log_ratio - k * se_sd)),
            float(np.exp(log_ratio + k * se_sd)),
        ]
        equiv = (
            abs(d) + k * se < margin and sd_ci[0] > 0.95 and sd_ci[1] < 1.05
        )
        outside = (
            mean_ci[0] > margin
            or mean_ci[1] < -margin
            or sd_ci[1] < 0.95
            or sd_ci[0] > 1.05
        )
        classification = (
            "equivalent"
            if equiv
            else "non_equivalent"
            if outside
            else "inconclusive"
        )
        if not good:
            classification = "inconclusive_diagnostics"
        comparisons[name] = dict(
            classification=classification,
            mean_shift_reference_sd=float(d / x["sd"]),
            mean_difference_ci=mean_ci,
            mean_margin=margin,
            sd_ratio=float(np.exp(log_ratio)),
            sd_ratio_ci=sd_ci,
        )
    return dict(diagnostics_satisfactory=good, summaries=comparisons)


def manifest() -> dict:
    return dict(
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "src/log_psplines/inference/model.py",
            )
        },
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        python=sys.version,
        platform=platform.platform(),
        versions={
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "jax",
                "jaxlib",
                "numpyro",
                "scipy",
                "arviz-stats",
                "pytest",
            )
        },
        devices=[str(d) for d in jax.devices()],
        backend=jax.default_backend(),
        x64=jax.config.x64_enabled,
        matmul=str(jax.config.jax_default_matmul_precision),
        command=sys.argv,
        seeds=SEEDS,
        policy="float64 preparation; explicit inference cast",
        thresholds=dict(
            local_nats=DELTA_LIMIT,
            gradient_abs=GRAD_ABS_LIMIT,
            gradient_scaled=GRAD_SCALED_LIMIT,
            mean_sd=MEAN_MARGIN,
            sd_ratio=SD_RATIO_BOUNDS,
            rhat=1.01,
            ess=400,
        ),
        missing="TV univariate, multivariate, LISA applications and accelerator precision untested",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("deterministic", "smoke", "analytic", "pilot"),
        default="deterministic",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tests/test-output/precision"
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--policy", choices=("fp32", "fp64"))
    parser.add_argument(
        "--case", choices=("constant", "spline"), default="spline"
    )
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--sampler-seed", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.9)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(
        manifest=manifest(),
        preparation_audit=preparation_audit(),
        deterministic=deterministic(),
    )
    if args.mode != "deterministic":
        if args.mode == "smoke":
            args.chains, args.warmup, args.draws = 2, 30, 40
        elif args.mode == "analytic":
            args.chains, args.warmup, args.draws = 4, 500, 5000
        cases = [("constant", SEEDS[0])]
        if args.mode != "analytic":
            cases += [
                ("spline", seed)
                for seed in (SEEDS if args.mode == "pilot" else SEEDS[:1])
            ]
        runs = []
        for index, (case, seed) in enumerate(cases):
            f = (
                fixture(seed)
                if case == "spline"
                else dict(
                    count=np.array(8), power=np.array(12.0, dtype=np.float64)
                )
            )
            filename = (
                f"fixture_{seed}.npz"
                if case == "spline"
                else "fixture_constant.npz"
            )
            path = args.output / filename
            np.savez(path, **f)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            policies = (
                ("fp64", "fp32", "fp64")
                if args.mode == "pilot"
                and case == "spline"
                and seed == SEEDS[0]
                else ("fp64", "fp32")
            )
            for repeat, policy in enumerate(policies):
                output = args.output / f"{case}_{seed}_{policy}_{repeat}.json"
                stream = 10000 + index * 100 + repeat * 10
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--policy",
                    policy,
                    "--case",
                    case,
                    "--fixture",
                    str(path),
                    "--output",
                    str(output),
                    "--sampler-seed",
                    str(stream),
                    "--chains",
                    str(args.chains),
                    "--warmup",
                    str(args.warmup),
                    "--draws",
                    str(args.draws),
                    "--target-accept",
                    str(args.target_accept),
                ]
                env = dict(
                    os.environ, JAX_ENABLE_X64="1" if policy == "fp64" else "0"
                )
                subprocess.run(cmd, env=env, cwd=ROOT, check=True)
                r = json.loads(output.read_text())
                r.update(
                    data_seed=seed if case == "spline" else None,
                    data_sha256=digest,
                    repeat=repeat,
                )
                if policy == "fp64" and repeat == 0 and case == "spline":
                    r["posterior_region_errors"] = target_errors(
                        f,
                        points=[
                            {
                                name: np.asarray(value, float)
                                for name, value in point.items()
                            }
                            for point in r["posterior_points"]
                        ],
                    )
                runs.append(r)
                print(
                    f"{case} seed={seed} {policy}: divergences={r['divergences']}, "
                    f"max Rhat={max(s['rhat'] for s in r['summaries'].values()):.4f}",
                    flush=True,
                )
        pairs = [
            (runs[i], runs[i + 1])
            for i in range(len(runs) - 1)
            if runs[i]["repeat"] == 0 and runs[i + 1]["repeat"] == 1
        ]
        repeats = [
            (runs[i], runs[i + 2])
            for i in range(len(runs) - 2)
            if runs[i + 2]["repeat"] == 2
        ]
        m = 2 * sum(len(a["summaries"]) for a, _ in pairs + repeats)
        k = float(norm.ppf(1 - 0.05 / (2 * m)))
        result.update(
            runs=runs,
            simultaneous_comparisons=m,
            interval_multiplier=k,
            sampler=dict(
                chains=args.chains,
                warmup=args.warmup,
                draws=args.draws,
                target_accept=args.target_accept,
                dense_mass=True,
                chain_method="sequential",
            ),
            comparisons=[
                dict(
                    case=a["case"],
                    data_seed=a["data_seed"],
                    policies=[a["policy"], b["policy"]],
                    **compare(a, b, k),
                )
                for a, b in pairs + repeats
            ],
        )
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
