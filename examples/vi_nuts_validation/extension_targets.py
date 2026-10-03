"""Two explicit native targets; frozen observations and bases are never refitted."""

import json
from functools import partial
from inspect import getsource
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer.util import log_density
from scipy.linalg import solve_discrete_lyapunov
from skfda.representation.basis import BSplineBasis
from study_common import Target, quadrature

from log_psplines.basis import SplineBasis
from log_psplines.config import PowerConfig, StationaryConfig
from log_psplines.data.spectral import PowerData, WishartData
from log_psplines.diagnostics.variational import (
    fingerprint,
    require_same_target,
)
from log_psplines.inference.model import (
    _blocked_channel_model,
    _sample_pspline_block,
    channel_model_kwargs,
    prepare_model,
)
from log_psplines.inference.power import (
    _collect_power_samples,
    prepare_power_model,
)
from log_psplines.models.matrix import SpectralMatrix
from log_psplines.models.spectrum import LogPSpline
from log_psplines.preprocessing.periodogram import compute_wishart

ROOT = Path(__file__).resolve().parents[2]
TV_SOURCE = ROOT / "runs/vi-nuts-validation-v2/exact_time_varying"
COMPONENTS = ["delta_0", "delta_1", "theta_re_1_0", "theta_im_1_0"]


def load_time_varying(directory=TV_SOURCE):
    """Exact archived loader, extracted from the former optimizer driver."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["target"] != "exact_time_varying":
        raise ValueError("only frozen exact time-varying target is allowed")
    arrays = np.load(directory / "target.npz")
    data = PowerData(
        arrays["power"],
        arrays["counts"],
        arrays["frequency"],
        arrays["time"],
        manifest["descriptor"]["units"],
    )
    spline = LogPSpline(
        SplineBasis.from_grid(
            data.frequency, interior_knots=arrays["knots_frequency"][4:-4]
        ),
        time=SplineBasis.from_grid(
            data.time, interior_knots=arrays["knots_time"][4:-4]
        ),
    )
    for axis in ("time", "frequency"):
        np.testing.assert_array_equal(
            getattr(spline, axis).basis, arrays[f"basis_{axis}"]
        )
        np.testing.assert_array_equal(
            getattr(spline, axis).penalty, arrays[f"penalty_{axis}"]
        )
    config = PowerConfig(progress_bar=False)
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
    require_same_target(identity, manifest["target_fingerprint"])
    return Target(
        model,
        spline,
        data,
        config,
        pair,
        init,
        identity,
        manifest["descriptor"],
    )


def matrix_config():
    return StationaryConfig(
        n_knots=6,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": "uniform"},
        Nb=8,
        wishart_window=None,
        wishart_detrend=False,
        roughness_scale=1.28,
        smoothing_parameterization="centered",
        eta=1.0,
        verbose=False,
    )


def freeze_matrix(path):
    """Generate just the declared development record, initialized stationarily."""
    rng = np.random.default_rng(62001)
    transition = np.array([[0.55, 0.12], [-0.08, 0.35]])
    innovation = np.array([[0.4, 0.12], [0.12, 0.3]])
    covariance = solve_discrete_lyapunov(transition, innovation)
    state = rng.multivariate_normal(np.zeros(2), covariance)
    initial = state.copy()
    raw = np.empty((4096, 2))
    for i, noise in enumerate(
        rng.multivariate_normal(np.zeros(2), innovation, size=len(raw))
    ):
        state = transition @ state + noise
        raw[i] = state
    data = compute_wishart(
        raw,
        fs=1.0,
        Nb=8,
        window=None,
        detrend=False,
        wishart_floor_fraction=None,
    )
    data = data.apply_mask(data.freq < 0.5)
    kwargs, components = prepare_model(data, matrix_config())
    models = components.diagonal_models + [
        components.get_theta_model("re", 1, 0),
        components.get_theta_model("im", 1, 0),
    ]
    saved = dict(
        raw=raw,
        initial_state=initial,
        transition=transition,
        innovation_covariance=innovation,
        u_re=data.u_re,
        u_im=data.u_im,
        frequency=data.freq,
        duration=np.array(data.duration),
    )
    for name, model in zip(COMPONENTS, models, strict=True):
        saved.update(
            {
                f"basis_{name}": np.asarray(model.basis),
                f"penalty_{name}": np.asarray(model.penalty_matrix),
                f"knots_{name}": np.asarray(model.knots),
                f"init_{name}": np.asarray(model.weights),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **saved)


def load_matrix(path):
    a = np.load(path)
    data = WishartData(
        a["u_re"],
        a["u_im"],
        a["frequency"],
        len(a["frequency"]),
        2,
        Nb=8,
        duration=float(a["duration"]),
    )
    kwargs, components = prepare_model(data, matrix_config())
    models = components.diagonal_models + [
        components.get_theta_model("re", 1, 0),
        components.get_theta_model("im", 1, 0),
    ]
    init = {}
    for name, model in zip(COMPONENTS, models, strict=True):
        for key, actual in (
            ("basis", model.basis),
            ("penalty", model.penalty_matrix),
            ("knots", model.knots),
            ("init", model.weights),
        ):
            np.testing.assert_array_equal(
                a[f"{key}_{name}"], np.asarray(actual)
            )
        init[f"weights_{name}"] = jnp.asarray(a[f"init_{name}"])
        if name.startswith("delta_"):
            init[f"weights_{name}"] -= np.log(data.duration)
        init[f"sigma_{name}"] = jnp.asarray(0.6744897501960817 * 1.28)
    model = partial(joint_matrix_model, kwargs=kwargs)
    identity = fingerprint(
        kwargs,
        getsource(_blocked_channel_model),
        getsource(_sample_pspline_block),
        getsource(joint_matrix_model),
    )
    return Target(
        model,
        components,
        data,
        matrix_config(),
        kwargs,
        init,
        identity,
        {
            "model": "native_stationary_modified_Cholesky",
            "generator": "stationary_VAR1",
            "data_seed": 62001,
            "n": 4096,
            "channels": 2,
            "Nb": 8,
            "parameterization": "centered",
            "coefficients_per_block": 8,
            "units": "native one-sided PSD",
            "guide": "one joint full-covariance Gaussian; native channel likelihoods factorize",
        },
    )


def joint_matrix_model(*, kwargs):
    for channel in range(kwargs["n_channels"]):
        _blocked_channel_model(**channel_model_kwargs(kwargs, channel))


def select_target(base, smoothing):
    if smoothing == "hierarchical":
        return base
    names = [name for name in base.init if name.startswith("sigma_")]
    fixed = {
        name: jnp.asarray(base.config.roughness_scale * 0.6744897501960817)
        for name in names
    }
    model = numpyro.handlers.condition(base.model, data=fixed)
    init = {
        name: value for name, value in base.init.items() if name not in fixed
    }
    if isinstance(base.config, PowerConfig):
        # Noncentered coordinates depend on sigma: remap the native PLS physical
        # coefficients to the fixed target. This changes initialization only.
        pilot = {
            name: np.asarray(value)[None, None]
            for name, value in base.init.items()
        }
        import xarray as xr

        ds = xr.Dataset(
            {
                name: (
                    (
                        "chain",
                        "draw",
                        *[f"d{i}" for i in range(np.ndim(value) - 2)],
                    ),
                    value,
                )
                for name, value in pilot.items()
            }
        )
        weights = physical_coefficients(base, ds)[0, 0]
        pair = base.pair
        eig = pair["U_time"].T @ weights @ pair["U_freq"]
        sigma = float(next(iter(fixed.values())))
        precision = sigma**-2 * (
            pair["lam_time"][:, None] + pair["lam_freq"][None, :]
        )
        inverse_scale = np.where(
            pair["joint_null"],
            np.sqrt(base.config.null_precision),
            np.sqrt(precision + base.config.ridge_eps),
        )
        init["s"] = jnp.asarray((eig * inverse_scale).reshape(-1))
    descriptor = {
        **base.descriptor,
        "conditioned_sigma": {k: float(v) for k, v in fixed.items()},
    }
    return Target(
        model,
        base.spline,
        base.data,
        base.config,
        base.pair,
        init,
        fingerprint(base.identity, "conditioned_native_sigma", fixed),
        descriptor,
    )


def stationary_design(spline, frequency, fit_frequency):
    coordinate = (np.asarray(frequency) - fit_frequency[0]) / (
        fit_frequency[-1] - fit_frequency[0]
    )
    basis = BSplineBasis(
        domain_range=[0, 1],
        order=spline.degree + 1,
        knots=spline.knots.tolist(),
    )
    return np.asarray(basis(coordinate)[:, :, 0].T)


def physical_coefficients(target, posterior):
    if isinstance(target.config, PowerConfig):
        p = posterior.copy()
        for name, value in target.descriptor.get(
            "conditioned_sigma", {}
        ).items():
            p[name] = (
                ("chain", "draw"),
                np.full((p.sizes["chain"], p.sizes["draw"]), value),
            )
        return _collect_power_samples(
            SimpleNamespace(posterior=p), target.pair, target.config
        )["weights"].values
    return np.concatenate(
        [posterior[f"weights_{name}"].values for name in COMPONENTS], -1
    )


def feature_values(target, posterior):
    """All-draw physical functionals; quadrature occurs inside each joint draw."""
    shape = (posterior.sizes["chain"], posterior.sizes["draw"])
    weights = physical_coefficients(target, posterior)
    columns, names = [], []

    def add(name, value):
        names.append(name)
        columns.append(np.asarray(value).reshape(shape))

    points = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    bands = ((0.04, 0.18), (0.18, 0.32), (0.32, 0.46))
    if isinstance(target.config, PowerConfig):
        times = np.array([0.25, 0.5, 0.75])
        bt = np.asarray(target.spline.time.design_at(times))
        bf = np.asarray(target.spline.frequency.design_at(points))
        logs = np.einsum("ti,cdij,fj->cdtf", bt, weights, bf, optimize=True)
        for t, time in enumerate(times):
            for f, frequency in enumerate(points):
                add(f"log_S_t{time:g}_f{frequency:g}", logs[..., t, f])
            for b, (lo, hi) in enumerate(bands):
                x = np.linspace(lo, hi, 129)
                design = np.asarray(target.spline.frequency.design_at(x))
                surface = np.einsum(
                    "i,cdij,fj->cdf", bt[t], weights, design, optimize=True
                )
                add(
                    f"log_band_t{time:g}_b{b}",
                    np.log(np.exp(surface) @ quadrature(x)),
                )
        for f, frequency in enumerate(points):
            add(
                f"temporal_contrast_f{frequency:g}",
                logs[..., 2, f] - logs[..., 0, f],
            )
        add(
            "roughness_time",
            np.einsum(
                "cdij,ik,cdkj->cd",
                weights,
                target.spline.time.penalty,
                weights,
            ),
        )
        add(
            "roughness_freq",
            np.einsum(
                "cdij,jk,cdik->cd",
                weights,
                target.spline.frequency.penalty,
                weights,
            ),
        )
        coefficients = weights.reshape(*shape, -1)
    else:
        models = target.spline.diagonal_models + [
            target.spline.get_theta_model("re", 1, 0),
            target.spline.get_theta_model("im", 1, 0),
        ]

        def spectrum(x):
            evaluated = [
                posterior[f"weights_{name}"].values
                @ stationary_design(m, x, target.data.freq).T
                for name, m in zip(COMPONENTS, models, strict=True)
            ]
            return SpectralMatrix(2)(
                np.stack(evaluated[:2], -1),
                evaluated[2][..., None],
                evaluated[3][..., None],
            )

        values = spectrum(points)
        if (
            not np.isfinite(values).all()
            or np.linalg.eigvalsh(values).min() <= 0
        ):
            raise ValueError("nonfinite or nonpositive spectral matrix")
        np.testing.assert_allclose(
            values, values.conj().swapaxes(-1, -2), rtol=1e-12, atol=1e-12
        )
        coherence = SpectralMatrix.coherence(values)[..., 1, 0]
        if np.any(coherence < 0) or np.any(coherence > 1 + 1e-12):
            raise ValueError("invalid coherence")
        for f, frequency in enumerate(points):
            for channel in range(2):
                add(
                    f"log_S_c{channel}_f{frequency:g}",
                    np.log(values[..., f, channel, channel].real),
                )
            add(f"csd_re_f{frequency:g}", values[..., f, 1, 0].real)
            add(f"csd_im_f{frequency:g}", values[..., f, 1, 0].imag)
            add(f"coherence_f{frequency:g}", coherence[..., f])
        for b, (lo, hi) in enumerate(bands):
            x = np.linspace(lo, hi, 129)
            value = spectrum(x)
            for channel in range(2):
                add(
                    f"log_band_c{channel}_b{b}",
                    np.log(value[..., channel, channel].real @ quadrature(x)),
                )
        for name, model in zip(COMPONENTS, models, strict=True):
            w = posterior[f"weights_{name}"].values
            add(
                f"roughness_{name}",
                np.einsum("cdi,ij,cdj->cd", w, model.penalty_matrix, w),
            )
        coefficients = weights
    for name in posterior.data_vars:
        if name.startswith("sigma_") and name not in target.descriptor.get(
            "conditioned_sigma", {}
        ):
            add(name, posterior[name].values)
            add("log_" + name, np.log(posterior[name].values))
    for i in range(coefficients.shape[-1]):
        add(f"coefficient_{i}", coefficients[..., i])
    return np.stack(columns, -1), names


def density_preflight(base, target):
    """Conditioning retains native weights prior; matrix wrapper is exact."""
    fixed = {
        k: jnp.asarray(v)
        for k, v in target.descriptor.get("conditioned_sigma", {}).items()
    }
    point = {**target.init, **fixed}
    a = log_density(target.model, (), {}, target.init)[0]
    b = log_density(base.model, (), {}, point)[0]
    if not np.isfinite(float(a)) or not np.isfinite(float(b)):
        raise ValueError(
            "nonfinite native density at empirical initialization"
        )
    np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-12)
    result = {"conditional_vs_native_log_density_error": float(abs(a - b))}
    if isinstance(base.config, StationaryConfig):
        blocks = sum(
            log_density(
                partial(
                    _blocked_channel_model,
                    **channel_model_kwargs(base.pair, j),
                ),
                (),
                {},
                point,
            )[0]
            for j in range(2)
        )
        np.testing.assert_allclose(b, blocks, atol=1e-8, rtol=1e-12)
        result["joint_vs_native_block_sum_error"] = float(abs(b - blocks))
    return result
