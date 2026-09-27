"""Variational inference for Cholesky channel models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
)

from log_psplines.inference.model import (
    _blocked_channel_model,
    channel_model_kwargs,
)


def _values_to_dataset(
    values: dict[str, Any] | None,
    *,
    values_are_draws: bool = True,
) -> xr.Dataset | None:
    if not values:
        return None
    data_vars = {}
    for name, value in values.items():
        array = np.asarray(value)
        if values_are_draws:
            if array.ndim == 0:
                raise ValueError(f"Samples for '{name}' require a draw axis")
            array = array[None, ...]
        else:
            array = array[None, None, ...]
        tail = tuple(f"{name}_dim_{i}" for i in range(array.ndim - 2))
        data_vars[name] = xr.DataArray(array, dims=("chain", "draw", *tail))
    return xr.Dataset(data_vars)


@dataclass
class VIResult:
    """VI posterior draws and optimization diagnostics."""

    posterior: xr.Dataset
    losses: jnp.ndarray
    guide_name: str
    losses_per_block: list[jnp.ndarray] | None = None



def resolve_guide(
    guide: str | Callable[..., Any] | None,
    model: Callable[..., Any],
) -> tuple[Any, str]:
    """Instantiate an autoguide for ``model``.

    Parameters
    ----------
    guide:
        Either a string identifier (``"diag"``, ``"mvn"``, ``"lowrank"`` or
        ``"lowrank:<rank>"``) or a callable producing an autoguide when called
        with ``model``. When ``None`` the diagonal guide is used.
    model:
        NumPyro model callable.

    Returns
    -------
    (guide_instance, guide_name)
        Instantiated autoguide and a human-readable name for diagnostics.
    """

    if guide is None:
        guide = "diag"

    if isinstance(guide, str):
        key = guide.lower()
        if key == "diag":
            return AutoDiagonalNormal(model), "diag"
        if key == "mvn":
            return (
                AutoMultivariateNormal(model),
                "mvn",
            )
        if key.startswith("lowrank"):
            rank = 10
            parts = key.split(":", 1)
            if len(parts) == 2 and parts[1]:
                rank = int(parts[1])
            guide_instance = AutoLowRankMultivariateNormal(
                model, rank=rank
            )
            return guide_instance, f"lowrank:{rank}"
        if key.startswith("flow"):
            layers = 1
            parts = key.split(":", 1)
            if len(parts) == 2 and parts[1]:
                layers = int(parts[1])
            # Use IAF for speed; allow switching to BNAF by prefix.
            if key.startswith("flowbnaf"):
                guide_instance = AutoBNAFNormal(
                    model, num_flows=layers
                )
                return guide_instance, f"flowbnaf:{layers}"
            guide_instance = AutoIAFNormal(
                model, num_flows=layers
            )
            return guide_instance, f"flow:{layers}"
        raise ValueError(f"Unknown VI guide specifier: {guide}")

    if isinstance(guide, type):
        instance = guide(model)
        return instance, getattr(guide, "__name__", guide.__class__.__name__)

    if callable(guide):
        instance = guide(model)
        return instance, getattr(guide, "__name__", "custom_guide")

    raise TypeError(
        "Guide must be a string identifier or a callable returning an autoguide"
    )


def _run_svi_with_early_stop(
    svi: SVI,
    rng_key: jax.Array,
    vi_steps: int,
    model_args: tuple,
    model_kwargs: dict[str, Any],
    *,
    progress_bar: bool = False,
    chunk_size: int = 100,
    patience: int = 3,
    rtol: float = 1e-4,
):
    """Run SVI in chunks, stopping early when ELBO converges.

    Convergence: the relative ELBO improvement over the last ``chunk_size``
    steps is below ``rtol`` for ``patience`` consecutive chunks.
    """
    if vi_steps <= chunk_size:
        result = svi.run(
            rng_key,
            vi_steps,
            *model_args,
            progress_bar=progress_bar,
            **model_kwargs,
        )
        return result.params, jnp.asarray(result.losses), result.state

    state = svi.init(rng_key, *model_args, **model_kwargs)
    all_losses: list[float] = []
    stale_count = 0
    step = 0

    def _run_chunk(state, n):
        def body(_, s):
            s, loss = svi.update(s, *model_args, **model_kwargs)
            return s

        return jax.lax.fori_loop(0, n, body, state)

    _run_full_chunk = jax.jit(lambda s: _run_chunk(s, chunk_size))

    @jax.jit
    def _evaluate(state):
        return svi.evaluate(state, *model_args, **model_kwargs)

    while step < vi_steps:
        n = min(chunk_size, vi_steps - step)
        if n == chunk_size:
            state = _run_full_chunk(state)
        else:
            state = jax.jit(lambda s, m=n: _run_chunk(s, m))(state)
        step += n

        loss_val = float(_evaluate(state))
        all_losses.append(loss_val)

        if len(all_losses) >= 2:
            prev = all_losses[-2]
            curr = all_losses[-1]
            denom = max(abs(prev), 1.0)
            if abs(prev - curr) / denom < rtol:
                stale_count += 1
            else:
                stale_count = 0
            if stale_count >= patience:
                break

    params = svi.get_params(state)
    losses = jnp.array(all_losses)
    return params, losses, state


def fit_vi(
    model: Callable[..., Any],
    *,
    rng_key: jax.Array,
    vi_steps: int,
    optimizer_lr: float,
    model_args: Iterable[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
    guide: str | Callable[..., Any] | None = "diag",
    posterior_draws: int = 256,
    progress_bar: bool = False,
) -> VIResult:
    """Run SVI and return posterior draws with loss diagnostics."""

    if model_kwargs is None:
        model_kwargs = {}
    else:
        model_kwargs = dict(model_kwargs)

    if vi_steps <= 0:
        raise ValueError("vi_steps must be positive")

    guide_obj, guide_name = resolve_guide(guide, model)
    # Gradient clipping helps avoid NaNs when the ELBO has very steep regions
    # (common for spectral models with exp/log transforms).
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optimizer_lr),
    )
    svi = SVI(model, guide_obj, optimizer, loss=Trace_ELBO())

    params, losses, final_state = _run_svi_with_early_stop(
        svi,
        rng_key,
        vi_steps,
        model_args=tuple(model_args),
        model_kwargs=model_kwargs,
        progress_bar=progress_bar,
    )

    state_key = getattr(final_state, "rng_key", rng_key)
    if posterior_draws > 0:
        sample_key, _ = jax.random.split(state_key)
        posterior_dist = guide_obj.get_posterior(params)
        latent_samples = posterior_dist.sample(
            sample_key, sample_shape=(posterior_draws,)
        )
        samples = guide_obj._unpack_and_constrain(latent_samples, params)
        posterior = _values_to_dataset(samples)
    else:
        median = guide_obj.median(params)
        posterior = _values_to_dataset(median, values_are_draws=False)
    if posterior is None:
        raise RuntimeError("VI produced no posterior values")
    return VIResult(
        posterior=posterior,
        losses=losses,
        guide_name=guide_name,
    )


def run_multivariate_vi(
    model_kwargs: dict[str, Any],
    *,
    rng_key: jax.Array,
    steps: int = 1500,
    lr: float = 1e-2,
    guide: str = "diag",
    posterior_draws: int = 256,
    eta: float = 1.0,
    verbose: bool = False,
) -> VIResult:
    """Run VI for each Cholesky channel and merge posterior draws."""
    kwargs = dict(model_kwargs)
    kwargs["eta"] = eta
    n_channels = int(kwargs["n_channels"])
    keys = jax.random.split(rng_key, n_channels)

    posterior_parts: list[xr.Dataset] = []
    losses_per_block: list[jnp.ndarray] = []
    guide_names: list[str] = []
    for channel_index in range(n_channels):
        result = fit_vi(
            _blocked_channel_model,
            rng_key=keys[channel_index],
            vi_steps=steps,
            optimizer_lr=lr,
            model_kwargs=channel_model_kwargs(kwargs, channel_index),
            guide=guide,
            posterior_draws=posterior_draws,
            progress_bar=verbose,
        )
        posterior_parts.append(result.posterior)
        losses_per_block.append(result.losses)
        guide_names.append(result.guide_name)

    nonempty_losses = [
        losses for losses in losses_per_block if int(losses.size) > 0
    ]
    if nonempty_losses:
        n_common = min(int(losses.size) for losses in nonempty_losses)
        losses = jnp.sum(
            jnp.stack(
                [losses[:n_common] for losses in nonempty_losses], axis=0
            ),
            axis=0,
        )
    else:
        losses = jnp.asarray([])
    guide_name = guide_names[0] if len(set(guide_names)) == 1 else "mixed"
    return VIResult(
        posterior=xr.merge(posterior_parts),
        losses=losses,
        guide_name=guide_name,
        losses_per_block=losses_per_block,
    )


__all__ = ["VIResult", "resolve_guide", "fit_vi", "run_multivariate_vi"]
