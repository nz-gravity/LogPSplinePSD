"""Pipeline stage primitives for VI and NUTS inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import arviz_base as az
import jax
import jax.numpy as jnp
import xarray as xr
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from ..samplers.vi_init.core import fit_vi


@dataclass
class StageResult:
    """Output of a VIStage run."""

    init_values: Optional[Dict[str, jnp.ndarray]]
    losses: Optional[jnp.ndarray]
    khat: Optional[float]
    guide_name: Optional[str]
    runtime: float


@dataclass
class VIStage:
    """Variational inference stage wrapping :func:`fit_vi`."""

    steps: int = 1500
    lr: float = 1e-2
    guide: str = "diag"
    posterior_draws: int = 256
    eta: float = 1.0

    def run(
        self,
        model_fn: Callable[..., Any],
        model_kwargs: Dict[str, Any],
        init_values: Optional[Dict[str, jnp.ndarray]] = None,
        *,
        rng_key: jax.Array,
        verbose: bool = False,
    ) -> StageResult:
        kwargs = dict(model_kwargs)
        kwargs["eta"] = self.eta

        t0 = time.time()
        result = fit_vi(
            model_fn,
            rng_key=rng_key,
            vi_steps=self.steps,
            optimizer_lr=self.lr,
            model_kwargs=kwargs,
            guide=self.guide,
            posterior_draws=self.posterior_draws,
            progress_bar=verbose,
            init_values=init_values,
        )
        runtime = time.time() - t0

        return StageResult(
            init_values=result.means,
            losses=result.losses,
            khat=None,
            guide_name=result.guide_name,
            runtime=runtime,
        )


@dataclass
class NUTSStage:
    """NUTS sampling stage wrapping NumPyro MCMC."""

    n_samples: int = 1000
    n_warmup: int = 500
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    num_chains: int = 1
    eta: float = 1.0

    def run(
        self,
        model_fn: Callable[..., Any],
        model_kwargs: Dict[str, Any],
        init_values: Optional[Dict[str, jnp.ndarray]] = None,
        *,
        rng_key: jax.Array,
        verbose: bool = False,
    ) -> xr.DataTree:
        kwargs = dict(model_kwargs)
        kwargs["eta"] = self.eta

        kernel_kwargs: Dict[str, Any] = dict(
            target_accept_prob=self.target_accept_prob,
            max_tree_depth=self.max_tree_depth,
            dense_mass=self.dense_mass,
        )
        if init_values is not None:
            kernel_kwargs["init_strategy"] = init_to_value(values=init_values)

        kernel = NUTS(model_fn, **kernel_kwargs)
        mcmc = MCMC(
            kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.num_chains,
            progress_bar=verbose,
        )
        mcmc.run(rng_key, **kwargs)
        return az.from_numpyro(mcmc)


__all__ = ["StageResult", "VIStage", "NUTSStage"]
