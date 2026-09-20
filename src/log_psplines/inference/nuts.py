"""Pipeline stage primitives for VI and NUTS inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import arviz_base as az
import jax
import jax.numpy as jnp
import xarray as xr
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value


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
    """Execute NumPyro NUTS with explicit settings; no spectral mathematics."""
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
    return mcmc


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
        "u_im_prev": model_kwargs["u_im"][:, :j, :],
        "basis_delta": model_kwargs["bases_delta"][j],
        "penalty_delta": model_kwargs["penalties_delta"][j],
        "basis_theta_re_by_component": tuple(
            model_kwargs["bases_theta_re"][j]
        ),
        "penalty_theta_re_by_component": tuple(
            model_kwargs["penalties_theta_re"][j]
        ),
        "basis_theta_im_by_component": tuple(
            model_kwargs["bases_theta_im"][j]
        ),
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
    channel_values = {
        name: value
        for name, value in init_values.items()
        if any(str(name).startswith(prefix) for prefix in prefixes)
    }
    return channel_values or None


def _posterior_vars_without_log_likelihood(dataset: xr.Dataset) -> xr.Dataset:
    keep = {
        name: var
        for name, var in dataset.data_vars.items()
        if not str(name).startswith("log_likelihood_block_")
    }
    return xr.Dataset(keep, attrs=dataset.attrs)


def _log_likelihood_vars_from_posterior(dataset: xr.Dataset) -> xr.Dataset:
    keep = {
        name: var
        for name, var in dataset.data_vars.items()
        if str(name).startswith("log_likelihood_block_")
    }
    return xr.Dataset(keep, attrs=dataset.attrs)


def _suffix_sample_stats(
    dataset: xr.Dataset, channel_index: int
) -> xr.Dataset:
    suffix = f"_channel_{int(channel_index)}"
    return xr.Dataset(
        {f"{name}{suffix}": var for name, var in dataset.data_vars.items()},
        attrs=dataset.attrs,
    )


def _merge_factor_idatas(idatas: list[xr.DataTree]) -> xr.DataTree:
    """Merge independently sampled channel DataTrees into one tree."""
    merged = xr.DataTree()
    posterior_parts = []
    sample_stats_parts = []
    log_likelihood_parts = []

    for channel_index, idata in enumerate(idatas):
        posterior = idata["posterior"].dataset
        posterior_parts.append(
            _posterior_vars_without_log_likelihood(posterior)
        )
        log_likelihood = _log_likelihood_vars_from_posterior(posterior)
        if log_likelihood.data_vars:
            log_likelihood_parts.append(log_likelihood)

        if "sample_stats" in idata.children:
            sample_stats_parts.append(
                _suffix_sample_stats(
                    idata["sample_stats"].dataset,
                    channel_index,
                )
            )

    if posterior_parts:
        merged["posterior"] = xr.DataTree(dataset=xr.merge(posterior_parts))
    if sample_stats_parts:
        merged["sample_stats"] = xr.DataTree(
            dataset=xr.merge(sample_stats_parts)
        )
    if log_likelihood_parts:
        merged["log_likelihood"] = xr.DataTree(
            dataset=xr.merge(log_likelihood_parts)
        )
    return merged


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
    ) -> xr.DataTree:
        from log_psplines.inference.model import _blocked_channel_model

        kwargs = dict(model_kwargs)
        kwargs["eta"] = self.eta
        n_channels = int(kwargs["n_channels"])
        keys = jax.random.split(rng_key, n_channels)
        idatas: list[xr.DataTree] = []

        for channel_index in range(n_channels):
            channel_kwargs = _channel_model_kwargs(kwargs, channel_index)
            channel_init = _init_values_for_channel(
                init_values,
                channel_index,
            )
            mcmc = run_nuts(
                _blocked_channel_model,
                rng_key=keys[channel_index],
                model_kwargs=channel_kwargs,
                init_values=channel_init,
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
                ),
            )
            idata = az.from_numpyro(mcmc)
            stats = idata["sample_stats"].dataset
            if (
                stats is not None
                and "lp" not in stats
                and "potential_energy" in stats
            ):
                stats["lp"] = -stats["potential_energy"]
            idatas.append(idata)

        merged = _merge_factor_idatas(idatas)
        merged.attrs["factorized"] = True
        merged.attrs["n_factors"] = n_channels
        return merged


__all__ = [
    "run_nuts",
    "FactorizedMultivarNUTSStage",
]
