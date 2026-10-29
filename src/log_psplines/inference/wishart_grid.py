"""Blocked tensor-spline fitting for complex grid or paired-point statistics."""

from collections.abc import Callable
from dataclasses import asdict
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr

from log_psplines.config import PowerConfig
from log_psplines.data.wishart_grid import WishartGridData
from log_psplines.inference.collapsed_grid import (
    collect_grid_samples,
    grid_design,
    grid_field_design,
    grid_field_scales,
    grid_theta_conditional,
    grid_theta_scales,
    prepare_grid_prior,
    prepare_grid_theta,
    recover_grid_theta,
    sample_grid_field,
    sample_grid_scales,
)
from log_psplines.inference.nuts import _mapping_to_dataset, _suffix, run_nuts
from log_psplines.inference.power import power_floor
from log_psplines.inference.power_results import power_result_spectra
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.logger import logger
from log_psplines.models.reconstruction import wishart_grid_draws
from log_psplines.models.spectrum import LogPSpline
from log_psplines.results import PSDResult


def row_field_labels(channel: int) -> list[str]:
    """One log-innovation field and 2*j real/imaginary regression fields."""
    return [f"delta_{channel}"] + [
        f"theta_{part}_{channel}_{previous}"
        for previous in range(channel)
        for part in ("re", "im")
    ]


def _initialize_field(marginal, counts, field_design, pair, config):
    """Weighted, penalized tensor initialization in bounded observation chunks."""
    count = np.asarray(counts).reshape(-1)
    active = count > 0
    mean = np.asarray(marginal).reshape(-1)[active] / count[active]
    target = np.log(mean + power_floor(mean))
    design = np.asarray(field_design)[active]
    weights = count[active] / count[active].max()
    penalty = (
        config.init_penalty_time * pair["lam_time"][:, None]
        + config.init_penalty_freq * pair["lam_freq"][None, :]
    ).reshape(-1)
    system = np.diag(penalty)
    rhs = np.zeros(len(penalty))
    for start in range(0, len(target), 4096):
        section = slice(start, start + 4096)
        block = design[section]
        system += block.T @ (weights[section, None] * block)
        rhs += block.T @ (weights[section] * target[section])
    result = {
        "s": np.linalg.solve(system, rhs),
        "sigma_freq": config.roughness_scale,
    }
    if pair["time_active"]:
        result["sigma_time"] = config.roughness_scale
    return result


def _initial_sampling_values(init, pair, config, label):
    values = dict(init)
    if not config.centered:
        values["z"] = values.pop("s") / np.asarray(
            grid_field_scales(
                pair, values["sigma_freq"], values.get("sigma_time")
            )
        )
    return {f"{name}_{label}": value for name, value in values.items()}


def prepare_wishart_grid_row(
    data: WishartGridData,
    spline: LogPSpline,
    config: PowerConfig,
    channel: int,
    *,
    pair: dict | None = None,
    design: tuple | None = None,
) -> tuple[Callable, dict]:
    """Sample delta tensor coefficients and scales, integrating theta tensors."""
    if not 0 <= channel < data.p:
        raise ValueError("channel must index a Cholesky row")
    pair = prepare_grid_prior(spline) if pair is None else pair
    design = grid_design(data, spline, pair) if design is None else design
    field = grid_field_design(design[2], design[3], paired=data.is_scattered)
    basis = jnp.asarray(field)
    real = jnp.asarray(data.u_re.reshape(-1, data.p, data.u_re.shape[-1]))
    imag = jnp.asarray(data.u_im.reshape(real.shape))
    counts = jnp.asarray(data.counts.reshape(-1))
    labels = row_field_labels(channel)
    prepared = (
        prepare_grid_theta(data, pair, design, channel) if channel else None
    )

    def model() -> None:
        coefficients = sample_grid_field(pair, config, labels[0])
        logs = basis @ coefficients
        if channel:
            sigmas_freq, sigmas_time = [], []
            for label in labels[1:]:
                sf, st = sample_grid_scales(pair, config, label)
                sigmas_freq.append(sf)
                sigmas_time.append(st)
            scales = grid_theta_scales(pair, sigmas_freq, sigmas_time)
            likelihood, _, _, _ = grid_theta_conditional(
                logs, scales, prepared
            )
        else:
            empty = jnp.empty((len(logs), 0))
            likelihood = wishart_log_likelihood(
                logs,
                empty,
                empty,
                real[:, channel],
                imag[:, channel],
                real[:, :channel],
                imag[:, :channel],
                counts=counts,
                enbw=data.enbw,
            )
            numpyro.deterministic(
                f"log_likelihood_block_{channel}", likelihood
            )
        numpyro.factor(f"wishart_{channel}", likelihood)

    marginal = np.sum(
        np.asarray(real[:, channel]) ** 2 + np.asarray(imag[:, channel]) ** 2,
        axis=-1,
    )
    initial = _initialize_field(marginal, data.counts, field, pair, config)
    if channel:
        scales = grid_theta_scales(
            pair,
            np.full(2 * channel, config.roughness_scale),
            np.full(2 * channel, config.roughness_scale),
        )
        _, beta, _, _ = grid_theta_conditional(
            basis @ initial["s"], scales, prepared
        )
        residual_re = prepared["observed_re"] - jnp.einsum(
            "frk,k->fr", prepared["design_re"], beta
        )
        residual_im = prepared["observed_im"] - jnp.einsum(
            "frk,k->fr", prepared["design_im"], beta
        )
        innovation = np.asarray(
            jnp.sum(residual_re**2 + residual_im**2, axis=-1)
        )
        initial = _initialize_field(
            innovation, data.counts, field, pair, config
        )
    init = _initial_sampling_values(initial, pair, config, labels[0])
    for label in labels[1:]:
        init[f"sigma_freq_{label}"] = config.roughness_scale
        if pair["time_active"]:
            init[f"sigma_time_{label}"] = config.roughness_scale
    return model, init


def fit_wishart_grid(
    data: WishartGridData,
    spline: LogPSpline,
    config: PowerConfig,
    *,
    true_psd: np.ndarray | None = None,
) -> PSDResult:
    """Fit independent tensor Cholesky rows with NUTS on reference grids."""
    if config.method != "nuts":
        raise ValueError(
            "WishartGridData currently supports method='nuts' only"
        )
    if config.structure != "tensor" or not isinstance(spline, LogPSpline):
        raise ValueError(
            "WishartGridData requires LogPSpline and structure='tensor'"
        )
    if not np.any(data.counts > 0):
        raise ValueError("at least one active complex observation is required")
    output_time = (
        data.reference_time if spline.time is None else spline.time.grid
    )
    output_frequency = spline.frequency.grid
    truth = None
    if true_psd is not None:
        truth = np.asarray(true_psd)
        expected = (len(output_time), len(output_frequency), data.p, data.p)
        if truth.shape != expected or not np.isfinite(truth).all():
            raise ValueError(
                f"true_psd must be finite with output-grid shape {expected}"
            )
    pair = prepare_grid_prior(spline)
    design = grid_design(data, spline, pair)
    keys = jax.random.split(jax.random.PRNGKey(config.seed), data.p)
    coefficient_size = len(pair["lam_freq"]) * len(pair["lam_time"])
    row_sizes = [coefficient_size * (1 + 2 * j) for j in range(data.p)]
    logger.info(
        f"GridTV reconstructed coefficients per row: {row_sizes}; marginal sampling retains {coefficient_size} delta coefficients per row (plus smoothing scales)"
    )
    posterior_parts, stats_parts, likelihood_parts, runtimes = [], [], [], []
    for channel, row_key in enumerate(keys):
        key, conditional_key = jax.random.split(row_key)
        model, init = prepare_wishart_grid_row(
            data, spline, config, channel, pair=pair, design=design
        )
        logger.info(f"GridTV sampling Cholesky row {channel + 1}/{data.p}")
        start = perf_counter()
        sampled = run_nuts(
            model,
            rng_key=key,
            init_values=init,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            num_chains=config.num_chains,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            progress_bar=config.progress_bar,
            extra_fields=(
                "potential_energy",
                "energy",
                "num_steps",
                "accept_prob",
                "adapt_state.step_size",
                "diverging",
            ),
        )
        if channel:
            prepared = prepare_grid_theta(data, pair, design, channel)
            recovered = recover_grid_theta(
                {
                    name: value.values
                    for name, value in sampled.posterior.data_vars.items()
                },
                pair,
                config,
                prepared,
                conditional_key,
            )
            likelihood_name = f"log_likelihood_block_{channel}"
            sampled.log_likelihood = _mapping_to_dataset(
                {likelihood_name: recovered.pop(likelihood_name)},
                num_chains=config.num_chains,
            )
            sampled.posterior = xr.merge(
                [
                    sampled.posterior,
                    _mapping_to_dataset(
                        recovered, num_chains=config.num_chains
                    ),
                ]
            )
        runtimes.append(perf_counter() - start)
        logger.info(
            f"GridTV row {channel + 1}/{data.p} sampling and conditional recovery: {runtimes[-1]:.1f} s"
        )
        posterior_parts.append(
            collect_grid_samples(
                sampled.posterior, pair, labels=row_field_labels(channel)
            )
        )
        stats_parts.append(_suffix(sampled.sample_stats, channel))
        likelihood_parts.append(sampled.log_likelihood)
    posterior = xr.merge(posterior_parts)
    # The existing frequency-chunked reducer retains an explicit draw preview.
    # Its final array still scales with T*F*C^2; estimate before allocating it.
    keep = (
        config.n_samples
        if config.spectrum_draws is None
        else min(config.n_samples, config.spectrum_draws)
    )
    final_bytes = (
        16
        * config.num_chains
        * keep
        * len(output_time)
        * len(output_frequency)
        * data.p**2
    )
    logger.info(
        f"GridTV materialized spectrum: {final_bytes / 2**20:.2f} MiB; draws per chain={keep}"
    )
    output_bt = (
        np.ones((len(output_time), 1))
        if spline.time is None
        else np.asarray(spline.time.basis)
    )
    output_bf = np.asarray(spline.frequency.basis)
    start = perf_counter()
    spectrum, summary = power_result_spectra(
        posterior,
        lambda section: wishart_grid_draws(
            posterior, output_bt, output_bf[section], data.p
        ),
        output_time,
        output_frequency,
        np.arange(data.p),
        config,
        matrix=True,
    )
    reconstruction_seconds = perf_counter() - start
    dims = ("time", "frequency", "channel", "channel_aux")
    observation_dims = (
        ("time", "frequency") if data.is_grid else ("observation",)
    )
    observation_coords = (
        {"time": data.time, "frequency": data.frequency}
        if data.is_grid
        else {
            "time": ("observation", data.time),
            "frequency": ("observation", data.frequency),
        }
    )
    result = PSDResult(
        posterior=posterior,
        sample_stats=xr.merge(stats_parts),
        log_likelihood=xr.merge(likelihood_parts),
        spectrum=spectrum,
        spectrum_summary=summary,
        truth=None
        if truth is None
        else xr.DataArray(
            truth,
            dims=dims,
            coords={name: spectrum.coords[name] for name in dims},
        ),
        observed_data=xr.Dataset(
            {
                "u_re": (
                    (*observation_dims, "channel", "factor"),
                    data.u_re,
                ),
                "u_im": (
                    (*observation_dims, "channel", "factor"),
                    data.u_im,
                ),
                "counts": (observation_dims, data.counts),
            },
            coords={
                **observation_coords,
                "channel": np.arange(data.p),
            },
            attrs={
                "units": data.units,
                "normalization": data.normalization,
                "enbw": data.enbw,
                **data.preprocessing,
            },
        ),
        model_data=xr.Dataset(
            {
                "basis_time": (("time", "time_coefficient"), output_bt),
                "basis_frequency": (
                    ("frequency", "frequency_coefficient"),
                    output_bf,
                ),
            },
            coords={"time": output_time, "frequency": output_frequency},
        ),
        metadata={
            **{
                key: value
                for key, value in asdict(config).items()
                if key
                not in ("interaction_scale", "null_precision", "ridge_eps")
            },
            "data_type": "multivariate_gridtv",
            "structure": "tensor",
            "observation_geometry": "rectangular"
            if data.is_grid
            else "paired",
            "units": data.units,
            "normalization": data.normalization,
            "counts_convention": "independent_proper_complex_observations; Y=sum(x x^H)",
            "tensor_prior": "Pt/sigma_time^2 kron I + I kron Pf/sigma_freq^2; supplied Pf includes its stationary ridge",
            "time_smoothing_active": pair["time_active"],
            "row_coefficient_counts": row_sizes,
            "row_latent_dimensions": [
                coefficient_size + (1 + int(pair["time_active"])) * (1 + 2 * j)
                for j in range(data.p)
            ],
            "theta_parameterization": "marginalized",
            "inference_dtype": "float64"
            if jax.config.x64_enabled
            else "float32",
            "enbw": data.enbw,
            "mean_removal": data.preprocessing.get("mean_removal", "unknown"),
            "window": data.preprocessing.get("window", "unknown"),
            "row_fit_seconds": runtimes,
            "reconstruction_seconds": reconstruction_seconds,
            "spectrum_bytes": final_bytes,
        },
    )
    return result
