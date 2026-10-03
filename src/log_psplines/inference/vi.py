"""Shared variational inference for NumPyro model callables."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
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

from log_psplines.diagnostics.variational import (
    VIDiagnosticConfig,
    VIDiagnosticState,
    combine_factor_ratios,
    diagnose_guide,
    evaluate_objective,
    fingerprint,
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

    posterior: xr.Dataset | None
    losses: jnp.ndarray | np.ndarray
    guide_name: str
    losses_per_block: list[jnp.ndarray | np.ndarray] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    diagnostics: VIDiagnosticState | None = None
    diagnostics_per_block: list[VIDiagnosticState] | None = None


def resolve_guide(
    guide: object,
    model: Callable[..., Any],
    *,
    init_values: Mapping[str, Any] | None = None,
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
        options = (
            {}
            if init_values is None
            else {"init_loc_fn": init_to_value(values=dict(init_values))}
        )
        key = guide.lower()
        if key == "diag":
            return AutoDiagonalNormal(model, **options), "diag"
        if key == "mvn":
            return (
                AutoMultivariateNormal(model, **options),
                "mvn",
            )
        if key.startswith("lowrank"):
            rank = 10
            parts = key.split(":", 1)
            if len(parts) == 2 and parts[1]:
                rank = int(parts[1])
            guide_instance = AutoLowRankMultivariateNormal(
                model, rank=rank, **options
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
                    model, num_flows=layers, **options
                )
                return guide_instance, f"flowbnaf:{layers}"
            guide_instance = AutoIAFNormal(model, num_flows=layers, **options)
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
    timings: dict[str, float] | None = None,
    early_stopping: bool = True,
    checkpoint_steps: tuple[int, ...] = (),
    checkpoint_callback: Callable | None = None,
    audit: dict | None = None,
    diagnostic_config: VIDiagnosticConfig | None = None,
):
    """Run SVI in chunks, stopping early when ELBO converges.

    Convergence: the relative ELBO improvement over the last ``chunk_size``
    steps is below ``rtol`` for ``patience`` consecutive chunks.
    """
    timings = {} if timings is None else timings
    audit = {} if audit is None else audit
    audit.update(
        checkpoint_steps=[],
        loss_step_indices=[],
        stopping_reason="step_limit",
        early_stopping=early_stopping,
        stopping_rule="paired_objective_noise_and_location"
        if diagnostic_config
        else "legacy_relative_loss",
    )
    started = perf_counter()
    if vi_steps <= chunk_size and not checkpoint_steps:
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
        audit["loss_step_indices"] = list(range(1, vi_steps + 1))
        if checkpoint_callback is not None:
            checkpoint_callback(vi_steps, result.params)
            audit["checkpoint_steps"].append(vi_steps)
        return result.params, jnp.asarray(result.losses), result.state

    state = svi.init(rng_key, *model_args, **model_kwargs)
    jax.block_until_ready(state)
    timings["initialization_seconds"] = perf_counter() - started
    loop_started = perf_counter()
    all_losses: list[float] = []
    stale_count = 0
    step = 0
    previous_objective = None
    previous_location = None

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
        boundaries = [value for value in checkpoint_steps if value > step]
        next_boundary = min(boundaries) if boundaries else vi_steps
        n = min(chunk_size, vi_steps - step, next_boundary - step)
        if n == chunk_size:
            state = _run_full_chunk(state)
        else:
            state = jax.jit(lambda s, m=n: _run_chunk(s, m))(state)
        step += n

        loss_val = float(_evaluate(state))
        all_losses.append(loss_val)
        audit["loss_step_indices"].append(step)
        if checkpoint_callback is not None and (
            step in checkpoint_steps or step == vi_steps
        ):
            checkpoint_callback(step, svi.get_params(state))
            audit["checkpoint_steps"].append(step)
        if step == n:
            timings["first_chunk_including_compile_seconds"] = (
                perf_counter() - loop_started
            )
            remaining_started = perf_counter()

        if early_stopping and diagnostic_config is not None:
            current_params = svi.get_params(state)
            objective = evaluate_objective(
                svi.model,
                svi.guide,
                current_params,
                diagnostic_config,
                model_args=model_args,
                model_kwargs=model_kwargs,
            )
            distribution = svi.guide.get_posterior(current_params)
            location = np.asarray(distribution.mean)
            if previous_objective is not None:
                changes = np.asarray(objective["values"]) - np.asarray(
                    previous_objective["values"]
                )
                change_se = changes.std(ddof=1) / np.sqrt(len(changes))
                location_change = np.max(
                    np.abs(location - previous_location)
                    / np.sqrt(np.asarray(distribution.variance))
                )
                stable = (
                    abs(changes.mean()) <= 2 * change_se
                    and location_change < 0.01
                )
                stale_count = stale_count + 1 if stable else 0
                if stale_count >= patience:
                    audit["stopping_reason"] = "early_stop"
                    break
            previous_objective, previous_location = objective, location
        elif early_stopping and len(all_losses) >= 2:
            prev = all_losses[-2]
            curr = all_losses[-1]
            denom = max(abs(prev), 1.0)
            if abs(prev - curr) / denom < rtol:
                stale_count += 1
            else:
                stale_count = 0
            if stale_count >= patience:
                audit["stopping_reason"] = "early_stop"
                break

    params = svi.get_params(state)
    if (
        checkpoint_callback is not None
        and step not in audit["checkpoint_steps"]
    ):
        checkpoint_callback(step, params)
        audit["checkpoint_steps"].append(step)
    timings["remaining_optimization_seconds"] = (
        perf_counter() - remaining_started
    )
    timings["steps_run"] = step
    losses = jnp.array(all_losses)
    return params, losses, state


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
    diagnostics: VIDiagnosticConfig | Mapping[str, Any] | None = None,
    optimization_particles: int = 1,
    optimizer_lr_metadata: Mapping[str, Any] | None = None,
    checkpoint_callback: Callable | None = None,
) -> VIResult:
    """Run SVI and return constrained draws, losses and measured timings.

    init_values optionally supplies the existing model's initial sites for
    built-in guides. First-chunk time includes compilation and optimization.
    optimizer_lr may be an Optax schedule. optimizer_lr_metadata describes
    that schedule for numerical checkpoints; it does not rebuild executable
    code. optimization_particles controls training, independently of the
    diagnostic and objective-evaluation particle counts.
    checkpoint_callback receives actual step and numerical parameters before
    objective/density analysis, allowing an experiment to preserve completed
    checkpoints even if later analysis fails. It does not resume optimization.
    """

    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model_args = tuple(model_args)
    if isinstance(diagnostics, Mapping):
        diagnostics = VIDiagnosticConfig(**diagnostics)
    checkpoints, checkpoint_objectives, audit = {}, {}, {}

    def record_checkpoint(step, params):
        checkpoints[str(step)] = jax.tree.map(np.asarray, params)
        if checkpoint_callback is not None:
            checkpoint_callback(step, checkpoints[str(step)])
        checkpoint_objectives[str(step)] = evaluate_objective(
            model,
            guide_obj,
            params,
            diagnostics,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )

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

    guide_obj, guide_name = resolve_guide(
        guide, model, init_values=init_values
    )
    timings: dict[str, float] = {}
    # Gradient clipping helps avoid NaNs when the ELBO has very steep regions
    # (common for spectral models with exp/log transforms).
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optimizer_lr),
    )
    svi = SVI(
        model,
        guide_obj,
        optimizer,
        loss=Trace_ELBO(num_particles=optimization_particles),
    )

    params, losses, final_state = _run_svi_with_early_stop(
        svi,
        rng_key,
        vi_steps,
        model_args=model_args,
        model_kwargs=model_kwargs,
        progress_bar=progress_bar,
        timings=timings,
        early_stopping=early_stopping,
        checkpoint_steps=diagnostics.checkpoint_steps if diagnostics else (),
        checkpoint_callback=record_checkpoint if diagnostics else None,
        audit=audit,
        diagnostic_config=(
            diagnostics
            if diagnostics and diagnostics.stopping_rule == "noise_aware"
            else None
        ),
    )

    draw_started = perf_counter()
    state_key = getattr(final_state, "rng_key", rng_key)
    audit.update(
        training_rng_key=np.asarray(jax.random.key_data(rng_key)).tolist(),
        optimizer={
            "type": "optax.adam",
            "learning_rate": "callable_schedule"
            if callable(optimizer_lr)
            else float(optimizer_lr),
            "schedule_metadata": dict(optimizer_lr_metadata or {}),
            "clip_global_norm": 1.0,
        },
        optimization_particles=optimization_particles,
    )
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
    timings["posterior_draw_seconds"] = perf_counter() - draw_started
    diagnostic_state = None
    if diagnostics is not None:
        diagnostic_started = perf_counter()
        diagnostic_state = diagnose_guide(
            model,
            guide_obj,
            params,
            guide_name,
            diagnostics,
            model_args=model_args,
            model_kwargs=model_kwargs,
            checkpoints=checkpoints,
            optimization={
                **audit,
                "steps_run": int(timings["steps_run"]),
                "checkpoint_objectives": checkpoint_objectives,
            },
        )
        timings["diagnostics_seconds"] = perf_counter() - diagnostic_started
        diagnostic_state.metadata["timings"] = dict(timings)
    return VIResult(
        posterior=posterior,
        losses=losses,
        guide_name=guide_name,
        timings=timings,
        diagnostics=diagnostic_state,
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
    early_stopping: bool = True,
    diagnostics: VIDiagnosticConfig | Mapping[str, Any] | None = None,
) -> VIResult:
    """Run VI for each Cholesky channel and merge posterior draws."""
    if isinstance(diagnostics, Mapping):
        diagnostics = VIDiagnosticConfig(**diagnostics)
    block_diagnostics = []
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
            early_stopping=early_stopping,
            diagnostics=(
                replace(
                    diagnostics,
                    seeds=tuple(
                        seed + channel_index * 10000
                        for seed in diagnostics.seeds
                    ),
                    target_fingerprint=fingerprint(
                        diagnostics.target_fingerprint,
                        channel_model_kwargs(kwargs, channel_index),
                    ),
                )
                if diagnostics
                else None
            ),
        )
        posterior_parts.append(result.posterior)
        losses_per_block.append(result.losses)
        guide_names.append(result.guide_name)
        if result.diagnostics is not None:
            block_diagnostics.append(result.diagnostics)

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
    joint = None
    if block_diagnostics:
        names = [
            site["name"]
            for block in block_diagnostics
            for site in block.metadata.get("latent_schema", [])
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                "shared latent sites invalidate independent Cholesky factor diagnostics"
            )
        joint = VIDiagnosticState(
            params={},
            metadata={
                "reconstruction_status": "unsupported_joint_recipe",
                "target_fingerprint": diagnostics.target_fingerprint,
                "coordinate_system": "aligned_independent_factor_particles",
                "weights": [],
            },
        )
        for repeat in range(len(diagnostics.seeds)):
            ratios = [
                block.arrays[
                    f"seed_{block.metadata['diagnostic_seeds'][repeat]}_log_ratios"
                ]
                for block in block_diagnostics
            ]
            combined, weights = combine_factor_ratios(
                ratios, factorization_verified=True
            )
            joint.arrays[f"repeat_{repeat}_log_ratios"] = combined
            joint.metadata["weights"].append({"repeat": repeat, **weights})
    return VIResult(
        posterior=xr.merge(posterior_parts),
        losses=losses,
        guide_name=guide_name,
        losses_per_block=losses_per_block,
        diagnostics=joint,
        diagnostics_per_block=block_diagnostics or None,
    )


__all__ = ["VIResult", "resolve_guide", "fit_vi", "run_multivariate_vi"]
