"""Variational inference helpers for sampler initialisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
)

GuideSpecifier = Any  # Accept strings or callables provided by the caller.


@dataclass
class VIResult:
    """Container for VI runs used to seed MCMC initial states."""

    means: Dict[str, jnp.ndarray]
    scales: Dict[str, jnp.ndarray]
    losses: jnp.ndarray
    params: Dict[str, Any]
    guide_name: str


def resolve_guide(
    guide: GuideSpecifier, model: Callable[..., Any]
) -> Tuple[Any, str]:
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
            return AutoMultivariateNormal(model), "mvn"
        if key.startswith("lowrank"):
            rank = 10
            parts = key.split(":", 1)
            if len(parts) == 2 and parts[1]:
                rank = int(parts[1])
            guide_instance = AutoLowRankMultivariateNormal(model, rank=rank)
            return guide_instance, f"lowrank:{rank}"
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


def _reduce_tree(
    samples: Mapping[str, jnp.ndarray],
    reducer: Callable[[jnp.ndarray], jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    return {
        name: reducer(jnp.asarray(array)) for name, array in samples.items()
    }


def fit_vi(
    model: Callable[..., Any],
    *,
    rng_key: jax.Array,
    vi_steps: int,
    optimizer_lr: float,
    model_args: Iterable[Any] = (),
    model_kwargs: Optional[Mapping[str, Any]] = None,
    guide: GuideSpecifier = "diag",
    posterior_draws: int = 256,
    progress_bar: bool = False,
) -> VIResult:
    """Run stochastic variational inference for ``model``.

    Returns posterior means and (sample-based) standard deviations for each
    latent site. These can be fed to ``init_to_value`` when initialising NUTS.
    """

    if model_kwargs is None:
        model_kwargs = {}

    if vi_steps <= 0:
        raise ValueError("vi_steps must be positive")

    guide_obj, guide_name = resolve_guide(guide, model)
    optimizer = optax.adam(optimizer_lr)
    svi = SVI(model, guide_obj, optimizer, loss=Trace_ELBO())

    run_result = svi.run(
        rng_key,
        vi_steps,
        *model_args,
        progress_bar=progress_bar,
        **model_kwargs,
    )
    params = run_result.params
    losses = jnp.asarray(run_result.losses)

    state_key = getattr(run_result.state, "rng_key", rng_key)
    if posterior_draws and posterior_draws > 0:
        sample_key, _ = jax.random.split(state_key)
        vi_samples = guide_obj.sample_posterior(
            sample_key,
            params,
            sample_shape=(posterior_draws,),
            *model_args,
            **model_kwargs,
        )
        means = _reduce_tree(vi_samples, lambda value: jnp.mean(value, axis=0))
        scales = _reduce_tree(vi_samples, lambda value: jnp.std(value, axis=0))
    else:
        posterior_sample = guide_obj.median(
            params, *model_args, **model_kwargs
        )
        means = {
            name: jnp.asarray(value)
            for name, value in posterior_sample.items()
        }
        scales = {name: jnp.zeros_like(array) for name, array in means.items()}

    return VIResult(
        means=means,
        scales=scales,
        losses=losses,
        params=params,
        guide_name=guide_name,
    )
