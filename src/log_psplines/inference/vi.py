"""NumPyro variational inference for scalar and Cholesky channel models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
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
from numpyro.infer.initialization import init_to_value

from log_psplines.inference.model import (
    _blocked_channel_model,
    channel_model_kwargs,
)


def _values_to_dataset(
    values: Mapping[str, Any], *, values_are_draws: bool = True
) -> xr.Dataset:
    """Give constrained model values the result's chain and draw axes."""
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
        data_vars[name] = (("chain", "draw", *tail), array)
    if not data_vars:
        raise RuntimeError("VI produced no posterior values")
    return xr.Dataset(data_vars)


@dataclass
class VIResult:
    """Constrained draws, optimization losses and wall-clock timings.

    Fits of at most 100 updates retain every training loss. Longer fits
    retain a loss evaluation after each 100-update chunk and the final chunk.
    A flat loss trace alone does not establish posterior accuracy.
    """

    posterior: xr.Dataset | None
    losses: jnp.ndarray | np.ndarray
    guide_name: str
    losses_per_block: list[jnp.ndarray | np.ndarray] | None = None
    timings: dict[str, float | int] = field(default_factory=dict)


def resolve_guide(
    guide: object,
    model: Callable[..., Any],
    *,
    init_values: Mapping[str, Any] | None = None,
) -> tuple[Any, str]:
    """Build diag, mvn, lowrank[:rank], flow[:layers] or flowbnaf[:layers].

    A callable may instead construct a NumPyro guide from ``model``.
    ``init_values`` initializes model sites for the built-in guides.
    """
    if guide is None:
        guide = "diag"
    if callable(guide):
        return guide(model), getattr(guide, "__name__", "custom_guide")
    if not isinstance(guide, str):
        raise TypeError("Guide must be a string or a callable guide factory")

    kind, _, size = guide.lower().partition(":")
    options = (
        {}
        if init_values is None
        else {"init_loc_fn": init_to_value(values=dict(init_values))}
    )
    factories = {
        "diag": AutoDiagonalNormal,
        "mvn": AutoMultivariateNormal,
        "lowrank": AutoLowRankMultivariateNormal,
        "flow": AutoIAFNormal,
        "flowbnaf": AutoBNAFNormal,
    }
    if kind not in factories:
        raise ValueError(f"Unknown VI guide specifier: {guide}")
    factory = factories[kind]
    if kind == "lowrank":
        rank = int(size) if size else 10
        options["rank"] = rank
        kind = f"lowrank:{rank}"
    elif kind.startswith("flow"):
        layers = int(size) if size else 1
        options["num_flows"] = layers
        kind = f"{kind}:{layers}"
    return factory(model, **options), kind


def _run_svi_with_early_stop(
    svi: SVI,
    rng_key: jax.Array,
    vi_steps: int,
    model_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
    *,
    progress_bar: bool,
    early_stopping: bool,
    timings: dict[str, float | int],
    chunk_size: int = 100,
    patience: int = 3,
    rtol: float = 1e-4,
) -> tuple[Any, jax.Array, Any]:
    """Keep the existing 100-update relative-loss stopping rule."""
    started = perf_counter()
    if vi_steps <= chunk_size:
        result = svi.run(
            rng_key,
            vi_steps,
            *model_args,
            progress_bar=progress_bar,
            **model_kwargs,
        )
        jax.block_until_ready(result.losses)
        timings["svi_run_including_compile_seconds"] = perf_counter() - started
        timings["steps_run"] = vi_steps
        return result.params, jnp.asarray(result.losses), result.state

    state = svi.init(rng_key, *model_args, **model_kwargs)
    jax.block_until_ready(state)
    timings["initialization_seconds"] = perf_counter() - started
    all_losses: list[float] = []
    stale_count = 0
    step = 0

    def run_chunk(state: Any, n: int) -> Any:
        def update(_: Any, state: Any) -> Any:
            return svi.update(state, *model_args, **model_kwargs)[0]

        return jax.lax.fori_loop(0, n, update, state)

    run_full_chunk = jax.jit(lambda state: run_chunk(state, chunk_size))
    evaluate = jax.jit(
        lambda state: svi.evaluate(state, *model_args, **model_kwargs)
    )
    loop_started = perf_counter()
    while step < vi_steps:
        n = min(chunk_size, vi_steps - step)
        state = (
            run_full_chunk(state)
            if n == chunk_size
            else jax.jit(lambda state, size=n: run_chunk(state, size))(state)
        )
        step += n
        all_losses.append(float(evaluate(state)))
        if step == n:
            timings["first_chunk_including_compile_seconds"] = (
                perf_counter() - loop_started
            )
            remaining_started = perf_counter()
        if early_stopping and len(all_losses) >= 2:
            previous, current = all_losses[-2:]
            stable = abs(previous - current) / max(abs(previous), 1.0) < rtol
            stale_count = stale_count + 1 if stable else 0
            if stale_count >= patience:
                break

    timings["remaining_optimization_seconds"] = (
        perf_counter() - remaining_started
    )
    timings["steps_run"] = step
    return svi.get_params(state), jnp.asarray(all_losses), state


def fit_vi(
    model: Callable[..., Any],
    *,
    rng_key: jax.Array,
    vi_steps: int,
    optimizer_lr: float | Callable,
    model_args: Iterable[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
    guide: str | Callable[..., Any] | None = "diag",
    posterior_draws: int = 256,
    progress_bar: bool = False,
    init_values: Mapping[str, Any] | None = None,
    early_stopping: bool = True,
    optimization_particles: int = 1,
) -> VIResult:
    """Fit a NumPyro model with SVI, clipped Adam and Trace_ELBO.

    ``optimizer_lr`` accepts a positive rate or an Optax schedule.
    ``optimization_particles`` controls the training ELBO sample count.
    ``early_stopping=False`` runs the complete update budget; otherwise three
    consecutive 100-update chunks with relative loss change below 1e-4 stop
    training. ``posterior_draws <= 0`` returns the guide median as one draw.
    """
    if vi_steps <= 0:
        raise ValueError("vi_steps must be positive")
    if (
        isinstance(optimization_particles, bool)
        or not isinstance(optimization_particles, int)
        or optimization_particles < 1
    ):
        raise ValueError("optimization_particles must be a positive integer")
    if not callable(optimizer_lr) and (
        not np.isfinite(optimizer_lr) or optimizer_lr <= 0
    ):
        raise ValueError("optimizer_lr must be positive and finite")

    model_args = tuple(model_args)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    guide_obj, guide_name = resolve_guide(
        guide, model, init_values=init_values
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(optimizer_lr)
    )
    svi = SVI(
        model,
        guide_obj,
        optimizer,
        loss=Trace_ELBO(num_particles=optimization_particles),
    )
    timings: dict[str, float | int] = {}
    params, losses, state = _run_svi_with_early_stop(
        svi,
        rng_key,
        vi_steps,
        model_args,
        model_kwargs,
        progress_bar=progress_bar,
        early_stopping=early_stopping,
        timings=timings,
    )
    started = perf_counter()
    if posterior_draws > 0:
        sample_key, _ = jax.random.split(state.rng_key)
        latent = guide_obj.get_posterior(params).sample(
            sample_key, sample_shape=(posterior_draws,)
        )
        # Retain the existing draw RNG stream across the wrapper simplification.
        values = guide_obj._unpack_and_constrain(latent, params)
        posterior = _values_to_dataset(values)
    else:
        posterior = _values_to_dataset(
            guide_obj.median(params), values_are_draws=False
        )
    timings["posterior_draw_seconds"] = perf_counter() - started
    return VIResult(posterior, losses, guide_name, timings=timings)


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
    early_stopping: bool = True,
) -> VIResult:
    """Fit independent Cholesky channels and merge constrained draws.

    Timings sum the sequential fits; ``steps_run`` sums their SVI updates.
    ``block_<index>_*`` timing entries retain each channel's measurements.
    """
    kwargs = {**model_kwargs, "eta": eta}
    n_channels = int(kwargs["n_channels"])
    keys = jax.random.split(rng_key, n_channels)
    results = [
        fit_vi(
            _blocked_channel_model,
            rng_key=keys[index],
            vi_steps=steps,
            optimizer_lr=lr,
            model_kwargs=channel_model_kwargs(kwargs, index),
            guide=guide,
            posterior_draws=posterior_draws,
            progress_bar=verbose,
            early_stopping=early_stopping,
        )
        for index in range(n_channels)
    ]
    losses_per_block = [result.losses for result in results]
    n_common = min(int(losses.size) for losses in losses_per_block)
    losses = jnp.sum(
        jnp.stack([losses[:n_common] for losses in losses_per_block]), axis=0
    )
    timings: dict[str, float | int] = {
        "steps_run": sum(result.timings["steps_run"] for result in results),
        "num_blocks": n_channels,
    }
    for index, result in enumerate(results):
        for name, value in result.timings.items():
            timings[f"block_{index}_{name}"] = value
            if name.endswith("_seconds"):
                timings[name] = timings.get(name, 0) + value
    names = {result.guide_name for result in results}
    return VIResult(
        posterior=xr.merge([result.posterior for result in results]),
        losses=losses,
        guide_name=names.pop() if len(names) == 1 else "mixed",
        losses_per_block=losses_per_block,
        timings=timings,
    )


__all__ = ["VIResult", "resolve_guide", "fit_vi", "run_multivariate_vi"]
