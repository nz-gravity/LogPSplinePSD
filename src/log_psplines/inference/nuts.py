"""NumPyro NUTS helpers with native xarray outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import xarray as xr
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from log_psplines.inference.model import (
    _blocked_channel_model,
    channel_model_kwargs,
)


@dataclass
class MCMCResult:
    """Posterior samples and sampler diagnostics from one or more NUTS fits."""

    posterior: xr.Dataset
    sample_stats: xr.Dataset | None = None
    log_likelihood: xr.Dataset | None = None


def _mapping_to_dataset(
    values: Mapping[str, Any],
    *,
    num_chains: int,
    prefix: str = "",
) -> xr.Dataset:
    """Convert chain/draw arrays to a Dataset without depending on ArviZ."""
    data_vars = {}
    for name, value in values.items():
        array = np.asarray(value)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.shape[0] != num_chains:
            array = array.reshape(num_chains, -1, *array.shape[1:])
        tail = tuple(f"{prefix}{name}_dim_{i}" for i in range(array.ndim - 2))
        data_vars[str(name)] = xr.DataArray(
            array,
            dims=("chain", "draw", *tail),
        )
    return xr.Dataset(data_vars)


def run_nuts(
    model: Callable,
    *,
    rng_key: jax.Array,
    model_kwargs: dict | None = None,
    init_values: dict | None = None,
    n_warmup: int,
    n_samples: int,
    num_chains: int = 1,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    dense_mass: bool = False,
    chain_method: str | None = None,
    progress_bar: bool = False,
    extra_fields: tuple[str, ...] = (),
) -> MCMCResult:
    """Execute NumPyro NUTS and return native samples/statistics."""
    kernel_options = dict(
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
    )
    if init_values is not None:
        kernel_options["init_strategy"] = init_to_value(values=init_values)
    chain_options = (
        {} if chain_method is None else {"chain_method": chain_method}
    )
    mcmc = MCMC(
        NUTS(model, **kernel_options),
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        **chain_options,
    )
    mcmc.run(rng_key, extra_fields=extra_fields, **(model_kwargs or {}))

    samples = mcmc.get_samples(group_by_chain=True)
    log_likelihood = {
        name: value
        for name, value in samples.items()
        if str(name).startswith("log_likelihood")
    }
    posterior = {
        name: value
        for name, value in samples.items()
        if not str(name).startswith("log_likelihood")
    }
    raw_stats = dict(mcmc.get_extra_fields(group_by_chain=True))
    stat_names = {
        "accept_prob": "acceptance_rate",
        "num_steps": "n_steps",
        "adapt_state.step_size": "step_size",
    }
    stats = {
        stat_names.get(str(name), str(name)): value
        for name, value in raw_stats.items()
    }
    if "potential_energy" in stats and "lp" not in stats:
        stats["lp"] = -np.asarray(stats["potential_energy"])

    return MCMCResult(
        posterior=_mapping_to_dataset(posterior, num_chains=num_chains),
        sample_stats=(
            _mapping_to_dataset(stats, num_chains=num_chains, prefix="stat_")
            if stats
            else None
        ),
        log_likelihood=(
            _mapping_to_dataset(
                log_likelihood, num_chains=num_chains, prefix="ll_"
            )
            if log_likelihood
            else None
        ),
    )


def _suffix(dataset: xr.Dataset | None, channel: int) -> xr.Dataset | None:
    if dataset is None:
        return None
    suffix = f"_channel_{channel}"
    return xr.Dataset(
        {f"{name}{suffix}": var for name, var in dataset.data_vars.items()}
    )


def _channel_setting(values, default, channel_index):
    """Use a per-channel override when present."""
    if values is None:
        return default
    try:
        return values[int(channel_index)]
    except (IndexError, TypeError, ValueError):
        return default


def run_multivariate_nuts(
    model_kwargs: dict[str, Any],
    *,
    rng_key: jax.Array,
    n_warmup: int,
    n_samples: int,
    num_chains: int = 1,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    dense_mass: bool = True,
    chain_method: str | None = None,
    eta: float = 1.0,
    target_accept_prob_by_channel: list[float] | None = None,
    max_tree_depth_by_channel: list[int] | None = None,
    verbose: bool = False,
) -> MCMCResult:
    """Run one NUTS fit for each multivariate Cholesky factor."""
    kwargs = dict(model_kwargs)
    kwargs["eta"] = eta
    n_channels = int(kwargs["n_channels"])
    keys = jax.random.split(rng_key, n_channels)
    posterior_parts = []
    stats_parts = []
    log_likelihood_parts = []

    for channel_index in range(n_channels):
        result = run_nuts(
            _blocked_channel_model,
            rng_key=keys[channel_index],
            model_kwargs=channel_model_kwargs(kwargs, channel_index),
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            dense_mass=dense_mass,
            target_accept_prob=float(
                _channel_setting(
                    target_accept_prob_by_channel,
                    target_accept_prob,
                    channel_index,
                )
            ),
            max_tree_depth=int(
                _channel_setting(
                    max_tree_depth_by_channel, max_tree_depth, channel_index
                )
            ),
            chain_method=chain_method,
            progress_bar=verbose,
            extra_fields=(
                "potential_energy",
                "energy",
                "num_steps",
                "accept_prob",
                "adapt_state.step_size",
                "diverging",
            ),
        )
        posterior_parts.append(result.posterior)
        stats = _suffix(result.sample_stats, channel_index)
        if stats is not None:
            stats_parts.append(stats)
        if result.log_likelihood is not None:
            log_likelihood_parts.append(result.log_likelihood)

    return MCMCResult(
        posterior=xr.merge(posterior_parts),
        sample_stats=xr.merge(stats_parts) if stats_parts else None,
        log_likelihood=(
            xr.merge(log_likelihood_parts) if log_likelihood_parts else None
        ),
    )


__all__ = ["MCMCResult", "run_nuts", "run_multivariate_nuts"]
