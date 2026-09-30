"""Thin blocked ANOVA integration for rectangular proper complex data."""

from collections.abc import Callable
from dataclasses import asdict
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr

from log_psplines.config import PowerConfig
from log_psplines.data.spectral import PowerData
from log_psplines.data.wishart_grid import WishartGridData
from log_psplines.inference.anova_power import (
    anova_init_values,
    collect_anova_samples,
    initialize_anova,
    prepare_anova_prior,
    sample_anova_field,
)
from log_psplines.inference.nuts import _suffix, run_nuts
from log_psplines.inference.power_results import power_result_spectra
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.logger import logger
from log_psplines.models.anova import ANOVALogPSpline, anova_components
from log_psplines.models.reconstruction import wishart_grid_draws
from log_psplines.results import PSDResult


def row_field_labels(channel: int) -> list[str]:
    """Row-major names for one diagonal and 2*j signed theta fields."""
    return [f"delta_{channel}"] + [
        f"theta_{part}_{channel}_{previous}"
        for previous in range(channel)
        for part in ("re", "im")
    ]


def prepare_wishart_grid_row(
    data: WishartGridData,
    spline: ANOVALogPSpline,
    config: PowerConfig,
    channel: int,
    *,
    pair: dict[str, np.ndarray] | None = None,
    design: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
) -> tuple[Callable, dict[str, np.ndarray | float]]:
    """Prepare an independently callable row fit, without matrix inversions.

    Shared designs contain raw time/frequency and their penalty-eigenbasis
    versions: (T,Kt), (F,Kf), (T,Kt), (F,Kf).
    Each scalar field samples its own ANOVA coefficients and hyperparameters.
    """
    if not 0 <= channel < data.p:
        raise ValueError("channel must index a Cholesky row")
    if not np.array_equal(data.reference_time, spline.time.grid):
        raise ValueError(
            "ANOVA time.grid must match data.reference_time for fixed centring"
        )
    pair = prepare_anova_prior(spline, config) if pair is None else pair
    if design is None:
        bt, bf = spline.design(data.time, data.frequency)
        design = bt, bf, bt @ pair["U_time"], bf @ pair["U_freq"]
    bt, bf, bt_eig, bf_eig = design
    bt_eig, bf_eig = jnp.asarray(bt_eig), jnp.asarray(bf_eig)
    # Each independent row needs only itself and preceding channels. Keep the
    # real/imaginary factors separate rather than rebuilding the full complex U.
    u_re = jnp.asarray(data.u_re[..., channel, :])
    u_im = jnp.asarray(data.u_im[..., channel, :])
    prev_re = jnp.asarray(data.u_re[..., :channel, :])
    prev_im = jnp.asarray(data.u_im[..., :channel, :])
    counts = jnp.asarray(data.counts)
    labels = row_field_labels(channel)

    def model() -> None:
        fields = []
        for label in labels:
            g, eta = sample_anova_field(
                pair, config, spline.sigma_eta_prior, label=label
            )
            mean, deviation = anova_components(bt_eig, bf_eig, g, eta)
            fields.append(mean[None, :] + deviation)
        logs = fields[0]
        theta_re = (
            jnp.stack(fields[1::2], axis=-1)
            if channel
            else jnp.empty((*logs.shape, 0))
        )
        theta_im = (
            jnp.stack(fields[2::2], axis=-1)
            if channel
            else jnp.empty((*logs.shape, 0))
        )
        likelihood = wishart_log_likelihood(
            logs,
            theta_re,
            theta_im,
            u_re,
            u_im,
            prev_re,
            prev_im,
            counts=counts,
            clip_log_psd=False,
        )
        numpyro.deterministic(f"log_likelihood_block_{channel}", likelihood)
        numpyro.factor(f"wishart_{channel}", likelihood)

    # Reuse scalar log-variance initialization only. The inference observations
    # remain complex multichannel factors, including every cross-spectrum.
    marginal = np.sum(
        data.u_re[..., channel, :] ** 2 + data.u_im[..., channel, :] ** 2,
        axis=-1,
    )
    diagonal_init = initialize_anova(
        PowerData(2 * marginal, 2 * data.counts, data.frequency, data.time),
        bt,
        bf,
        spline.time_penalty,
        np.asarray(spline.frequency.penalty),
        pair,
        config,
    )
    init = anova_init_values(diagonal_init, pair, config, label=labels[0])
    for label in labels[1:]:
        init.update(
            anova_init_values(
                {
                    "g": np.zeros(len(pair["lam_freq"])),
                    "eta": np.zeros(pair["eta_scale"].size),
                    "sigma_g": float(diagonal_init["sigma_g"]),
                    "sigma_eta": float(diagonal_init["sigma_eta"]),
                },
                pair,
                config,
                label=label,
            )
        )
    return model, init


def fit_wishart_grid(
    data: WishartGridData,
    spline: ANOVALogPSpline,
    config: PowerConfig,
    *,
    true_psd: np.ndarray | None = None,
) -> PSDResult:
    """Execute row fits and reconstruct the product posterior on reference grids."""
    if config.structure != "anova" or not isinstance(spline, ANOVALogPSpline):
        raise ValueError(
            "WishartGridData requires ANOVALogPSpline and structure='anova'"
        )
    if not np.any(data.counts > 0):
        raise ValueError("at least one active complex observation is required")
    output_time, output_frequency = spline.time.grid, spline.frequency.grid
    truth = None
    if true_psd is not None:
        truth = np.asarray(true_psd)
        expected = (len(output_time), len(output_frequency), data.p, data.p)
        if truth.shape != expected or not np.isfinite(truth).all():
            raise ValueError(
                f"true_psd must be finite with output-grid shape {expected}"
            )
    pair = prepare_anova_prior(spline, config)
    bt, bf = spline.design(data.time, data.frequency)
    design = bt, bf, bt @ pair["U_time"], bf @ pair["U_freq"]
    keys = jax.random.split(jax.random.PRNGKey(config.seed), data.p)
    coefficient_size = len(pair["lam_freq"]) * (1 + len(pair["lam_time"]))
    row_sizes = [coefficient_size * (1 + 2 * j) for j in range(data.p)]
    logger.info(
        f"GridTV coefficients per row: {row_sizes}; total={sum(row_sizes)} (plus {2 * data.p**2} hyperparameters)"
    )
    posterior_parts, stats_parts, likelihood_parts, runtimes, timings = (
        [],
        [],
        [],
        [],
        [],
    )
    for channel, key in enumerate(keys):
        model, init = prepare_wishart_grid_row(
            data, spline, config, channel, pair=pair, design=design
        )
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
            record_timing=True,
        )
        runtimes.append(perf_counter() - start)
        timings.append(sampled.timings)
        posterior_parts.append(
            collect_anova_samples(
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
    output_bt, output_bf = spline.design()
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
                    ("time", "frequency", "channel", "factor"),
                    data.u_re,
                ),
                "u_im": (
                    ("time", "frequency", "channel", "factor"),
                    data.u_im,
                ),
                "counts": (("time", "frequency"), data.counts),
            },
            coords={
                "time": data.time,
                "frequency": data.frequency,
                "channel": np.arange(data.p),
            },
            attrs={"units": data.units, "normalization": data.normalization},
        ),
        model_data=xr.Dataset(
            {
                "basis_time": (("time", "time_coefficient"), output_bt),
                "basis_frequency": (
                    ("frequency", "frequency_coefficient"),
                    output_bf,
                ),
                "time_transform": (
                    ("original_time_coefficient", "time_coefficient"),
                    spline.time_transform,
                ),
            },
            coords={"time": output_time, "frequency": output_frequency},
        ),
        metadata={
            **asdict(config),
            "data_type": "multivariate_gridtv",
            "structure": "anova",
            "units": data.units,
            "normalization": data.normalization,
            "counts_convention": "independent_proper_complex_observations; Y=sum(x x^H)",
            "centring": "fixed_reference_time_grid_before_masking_and_pooling",
            "row_coefficient_counts": row_sizes,
            "row_latent_dimensions": [
                size + 2 * (1 + 2 * j) for j, size in enumerate(row_sizes)
            ],
            "row_fit_seconds": runtimes,
            "row_warmup_including_compilation_seconds": [
                timing["warmup_including_compilation_seconds"]
                for timing in timings
            ],
            "row_sampling_including_compilation_seconds": [
                timing["sampling_including_compilation_seconds"]
                for timing in timings
            ],
            "reconstruction_seconds": reconstruction_seconds,
            "spectrum_bytes": final_bytes,
            "interaction_scale": spline.sigma_eta_prior,
        },
    )
    return result
