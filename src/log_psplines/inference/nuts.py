"""NumPyro NUTS helpers with native xarray outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value


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
) -> MCMC:
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
    stats = dict(mcmc.get_extra_fields(group_by_chain=True))
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


def _channel_model_kwargs(
    model_kwargs: dict[str, Any],
    channel_index: int,
) -> dict[str, Any]:
    """Extract kwargs for one multivariate Cholesky likelihood factor."""
    j = int(channel_index)
    return {
        "channel_index": j,
        "u_re_channel": model_kwargs["u_re"][:, j, :],
        "u_im_channel": model_kwargs["u_im"][:, j, :],
        "u_re_prev": model_kwargs["u_re"][:, :j, :],
        "u_im_prev": model_kwargs["u_im_prev"][:, :j, :]
        if "u_im_prev" in model_kwargs
        else model_kwargs["u_im"][:, :j, :],
        "basis_delta": model_kwargs["bases_delta"][j],
        "penalty_delta": model_kwargs["penalties_delta"][j],
        "basis_theta_re_by_component": tuple(model_kwargs["bases_theta_re"][j]),
        "penalty_theta_re_by_component": tuple(
            model_kwargs["penalties_theta_re"][j]
        ),
        "basis_theta_im_by_component": tuple(model_kwargs["bases_theta_im"][j]),
        "penalty_theta_im_by_component": tuple(
            model_kwargs["penalties_theta_im"][j]
        ),
        "alpha_phi": model_kwargs["alpha_phi"],
        "beta_phi": model_kwargs["beta_phi"],
        "alpha_phi_theta": model_kwargs["alpha_phi_theta"],
        "beta_phi_theta": model_kwargs["beta_phi_theta"],
        "alpha_delta": model_kwargs["alpha_delta"],
        "beta_delta": model_kwargs["beta_delta"],
        "duration": model_kwargs["duration"],
        "Nb": model_kwargs["Nb"],
        "Nh": model_kwargs["Nh"],
        "design_weights": model_kwargs.get("design_weights"),
        "tau": model_kwargs.get("tau"),
        "enbw": model_kwargs.get("enbw", 1.0),
        "eta": model_kwargs.get("eta", 1.0),
    }


def _init_values_for_channel(
    init_values: dict[str, jnp.ndarray] | None,
    channel_index: int,
) -> dict[str, jnp.ndarray] | None:
    """Return VI initial values belonging to one Cholesky channel block."""
    if not init_values:
        return None
    j = int(channel_index)
    prefixes = (
        f"delta_{j}",
        f"phi_delta_{j}",
        f"weights_delta_{j}",
        f"delta_theta_re_{j}_",
        f"phi_theta_re_{j}_",
        f"weights_theta_re_{j}_",
        f"delta_theta_im_{j}_",
        f"phi_theta_im_{j}_",
        f"weights_theta_im_{j}_",
    )
    values = {
        name: value
        for name, value in init_values.items()
        if any(str(name).startswith(prefix) for prefix in prefixes)
    }
    return values or None


def _suffix(dataset: xr.Dataset | None, channel: int) -> xr.Dataset | None:
    if dataset is None:
        return None
    suffix = f"_channel_{channel}"
    return xr.Dataset(
        {f"{name}{suffix}": var for name, var in dataset.data_vars.items()}
    )


@dataclass
class FactorizedMultivarNUTSStage:
    """Run independent NUTS chains for each multivariate Cholesky factor."""

    n_samples: int = 1000
    n_warmup: int = 500
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    num_chains: int = 1
    eta: float = 1.0
    chain_method: str | None = None
    target_accept_prob_by_channel: list[float] | None = None
    max_tree_depth_by_channel: list[int] | None = None

    def _channel_target_accept(self, channel_index: int) -> float:
        values = self.target_accept_prob_by_channel
        if values is None:
            return float(self.target_accept_prob)
        try:
            return float(values[int(channel_index)])
        except (IndexError, TypeError, ValueError):
            return float(self.target_accept_prob)

    def _channel_max_tree_depth(self, channel_index: int) -> int:
        values = self.max_tree_depth_by_channel
        if values is None:
            return int(self.max_tree_depth)
        try:
            return int(values[int(channel_index)])
        except (IndexError, TypeError, ValueError):
            return int(self.max_tree_depth)

    def run(
        self,
        model_kwargs: dict[str, Any],
        init_values: dict[str, jnp.ndarray] | None = None,
        *,
        rng_key: jax.Array,
        verbose: bool = False,
    ) -> MCMCResult:
        from log_psplines.inference.model import _blocked_channel_model

        kwargs = dict(model_kwargs)
        kwargs["eta"] = self.eta
        n_channels = int(kwargs["n_channels"])
        keys = jax.random.split(rng_key, n_channels)
        posterior_parts = []
        stats_parts = []
        log_likelihood_parts = []

        for channel_index in range(n_channels):
            result = run_nuts(
                _blocked_channel_model,
                rng_key=keys[channel_index],
                model_kwargs=_channel_model_kwargs(kwargs, channel_index),
                init_values=_init_values_for_channel(init_values, channel_index),
                n_warmup=self.n_warmup,
                n_samples=self.n_samples,
                num_chains=self.num_chains,
                dense_mass=self.dense_mass,
                target_accept_prob=self._channel_target_accept(channel_index),
                max_tree_depth=self._channel_max_tree_depth(channel_index),
                chain_method=self.chain_method,
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
            ll = _suffix(result.log_likelihood, channel_index)
            if ll is not None:
                log_likelihood_parts.append(ll)

        return MCMCResult(
            posterior=xr.merge(posterior_parts),
            sample_stats=xr.merge(stats_parts) if stats_parts else None,
            log_likelihood=(
                xr.merge(log_likelihood_parts)
                if log_likelihood_parts
                else None
            ),
        )


__all__ = ["MCMCResult", "run_nuts", "FactorizedMultivarNUTSStage"]
