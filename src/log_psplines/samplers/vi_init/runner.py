"""Core VI runners used by univariate and multivariate initialisation adapters."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import init_to_value

from ...diagnostics.vi_results import (
    _build_multivar_vi_diagnostics,
    _build_univar_vi_diagnostics,
)
from ...logger import logger
from ..pspline_block import (
    build_log_density_fn,
    pspline_hyperparameter_initials,
)
from .core import fit_vi
from .guide import (
    suggest_guide_block,
    suggest_guide_multivar,
    suggest_guide_univar,
)
from .mixin import VIInitialisationArtifacts


def _hyperparameter_defaults(
    *,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
    divide_phi_by_delta: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(delta_0, log_phi_0)`` from the prior hypers."""
    delta_0, phi_0 = pspline_hyperparameter_initials(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        divide_phi_by_delta=divide_phi_by_delta,
    )
    return jnp.asarray(delta_0), jnp.log(jnp.asarray(phi_0))


def _assign_theta_component_init_values(
    init_values: Dict[str, jnp.ndarray],
    *,
    prefix: str,
    weights: tuple[jnp.ndarray, jnp.ndarray],
    delta_init: jnp.ndarray,
    log_phi_init: jnp.ndarray,
) -> None:
    """Populate delta/phi/weight init values for one theta component."""
    init_values[f"delta_theta_re_{prefix}"] = delta_init
    init_values[f"phi_theta_re_{prefix}"] = log_phi_init
    init_values[f"weights_theta_re_{prefix}"] = weights[0]
    init_values[f"delta_theta_im_{prefix}"] = delta_init
    init_values[f"phi_theta_im_{prefix}"] = log_phi_init
    init_values[f"weights_theta_im_{prefix}"] = weights[1]


def default_init_values_univar(
    spline_model,
    *,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
) -> Dict[str, jnp.ndarray]:
    """Return default init values for univariate samplers."""
    delta_0, log_phi_0 = _hyperparameter_defaults(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        divide_phi_by_delta=True,
    )
    return {
        "delta": delta_0,
        "phi": log_phi_0,
        "weights": jnp.asarray(spline_model.weights),
    }


def default_init_values_multivar(
    spline_model,
    *,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
) -> Dict[str, jnp.ndarray]:
    """Return default init values for multivariate samplers."""
    delta_0, log_phi_0 = _hyperparameter_defaults(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        divide_phi_by_delta=False,
    )

    init_values: Dict[str, jnp.ndarray] = {}
    for idx, model in enumerate(spline_model.diagonal_models):
        init_values[f"delta_{idx}"] = delta_0
        init_values[f"phi_delta_{idx}"] = log_phi_0
        init_values[f"weights_delta_{idx}"] = jnp.asarray(model.weights)

    if spline_model.n_theta > 0:
        for j, l in spline_model.theta_pairs:
            re_model = spline_model.get_theta_model("re", j, l)
            im_model = spline_model.get_theta_model("im", j, l)
            _assign_theta_component_init_values(
                init_values,
                prefix=f"{j}_{l}",
                weights=(
                    jnp.asarray(re_model.weights),
                    jnp.asarray(im_model.weights),
                ),
                delta_init=delta_0,
                log_phi_init=log_phi_0,
            )

    return init_values


def select_vi_or_default_init(
    *,
    vi_values: Optional[Dict[str, jnp.ndarray]],
    default_values: Dict[str, jnp.ndarray],
    log_posterior_fn: Callable[[Dict[str, jnp.ndarray]], float],
    log_prefix: str = "VI init",
):
    """Compare median VI init vs deterministic default; return the better one."""
    if vi_values is None:
        return init_to_value(values=default_values)

    vi_lp = float(log_posterior_fn(vi_values))
    det_lp = float(log_posterior_fn(default_values))
    delta_lp = vi_lp - det_lp
    delta_label = (
        f"improved by {delta_lp:.3f}"
        if np.isfinite(delta_lp) and delta_lp >= 0.0
        else f"worse by {abs(delta_lp):.3f}"
    )
    logger.info(
        f"{log_prefix}: VI log-post={vi_lp:.3f}, default log-post={det_lp:.3f} "
        f"({delta_label})"
    )
    if np.isfinite(vi_lp) and vi_lp > det_lp:
        return init_to_value(values=vi_values)
    return init_to_value(values=default_values)


def _univar_model_args(sampler) -> tuple:
    """Build the positional args tuple for bayesian_model."""
    return (
        sampler.log_pdgrm,
        sampler.basis_matrix,
        sampler.penalty_matrix,
        sampler.log_parametric,
        sampler.Nh,
        sampler.config.alpha_phi,
        sampler.config.beta_phi,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
    )


def _univar_default_init(sampler) -> Dict[str, jnp.ndarray]:
    """Build default init values for univariate VI/NUTS."""
    return default_init_values_univar(
        sampler.spline_model,
        alpha_phi=sampler.config.alpha_phi,
        beta_phi=sampler.config.beta_phi,
        alpha_delta=sampler.config.alpha_delta,
        beta_delta=sampler.config.beta_delta,
    )


def compute_vi_artifacts_univar(
    sampler,
    *,
    model: Callable[..., Any],
    init_values: Optional[Dict[str, jnp.ndarray]] = None,
) -> VIInitialisationArtifacts:
    """Run VI for univariate samplers and return initialisation artifacts."""

    guide_spec = sampler.config.vi_guide or suggest_guide_univar(
        sampler.n_weights + 2
    )

    def _postprocess(vi_result):
        means = {name: jnp.asarray(v) for name, v in vi_result.means.items()}
        diagnostics = _build_univar_vi_diagnostics(sampler, vi_result)
        return means, diagnostics

    return sampler._run_vi_initialisation(
        model=model,
        model_args=_univar_model_args(sampler),
        guide=guide_spec,
        init_values=init_values or _univar_default_init(sampler),
        postprocess=_postprocess,
    )


def compute_vi_artifacts_multivar(
    sampler,
    *,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """Run VI for fully coupled multivariate samplers."""

    total_latents = sum(
        m.n_basis + 2 for m in sampler.spline_model.diagonal_models
    )
    if sampler.n_theta > 0:
        for pair in sampler.spline_model.theta_pairs:
            total_latents += (
                sampler.spline_model.offdiag_re_models[pair].n_basis + 2
            )
            total_latents += (
                sampler.spline_model.offdiag_im_models[pair].n_basis + 2
            )
    guide_spec = sampler.config.vi_guide or suggest_guide_multivar(
        total_latents
    )

    def _postprocess(vi_result):
        init_values = {
            name: jnp.asarray(v) for name, v in vi_result.means.items()
        }
        diagnostics = _build_multivar_vi_diagnostics(sampler, vi_result)
        return init_values, diagnostics

    return sampler._run_vi_initialisation(
        model=model,
        model_args=(
            sampler.u_re,
            sampler.u_im,
            sampler.duration,
            sampler.Nb,
            sampler.all_bases,
            sampler.all_penalties,
            sampler.Nh,
            sampler.config.alpha_phi,
            sampler.config.beta_phi,
            sampler.config.alpha_delta,
            sampler.config.beta_delta,
        ),
        guide=guide_spec,
        init_values=default_init_values_multivar(
            sampler.spline_model,
            alpha_phi=sampler.config.alpha_phi,
            beta_phi=sampler.config.beta_phi,
            alpha_delta=sampler.config.alpha_delta,
            beta_delta=sampler.config.beta_delta,
        ),
        postprocess=_postprocess,
    )


def build_block_model_args(
    sampler,
    channel_index: int,
    alpha_phi_theta: float,
    beta_phi_theta: float,
) -> tuple[tuple, dict]:
    """Build positional and keyword args for blocked-channel model."""
    theta_re_basis, theta_re_penalty = (
        sampler._theta_component_arrays_for_channel(channel_index, part="re")
    )
    theta_im_basis, theta_im_penalty = (
        sampler._theta_component_arrays_for_channel(channel_index, part="im")
    )
    model_args = (
        channel_index,
        sampler.u_re[:, channel_index, :],
        sampler.u_im[:, channel_index, :],
        sampler.u_re[:, :channel_index, :],
        sampler.u_im[:, :channel_index, :],
        sampler.all_bases[channel_index],
        sampler.all_penalties[channel_index],
        theta_re_basis,
        theta_re_penalty,
        theta_im_basis,
        theta_im_penalty,
        sampler.config.alpha_phi,
        sampler.config.beta_phi,
        alpha_phi_theta,
        beta_phi_theta,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
        sampler.duration,
        sampler.Nb,
        sampler.Nh,
    )
    eta_vi = min(1.0, 1.0 / float(sampler.Nb * sampler.Nh))
    model_kwargs = {
        "enbw": float(sampler.enbw),
        "eta": eta_vi,
    }
    return model_args, model_kwargs


def build_block_init_values(
    *,
    sampler,
    channel_index: int,
    theta_count: int,
    delta_weights_init: jnp.ndarray,
    alpha_phi_theta: float,
    beta_phi_theta: float,
) -> Dict[str, jnp.ndarray]:
    """Build initial values for one blocked channel model."""
    alpha_phi_delta = sampler.config.alpha_phi
    beta_phi_delta = sampler.config.beta_phi
    delta_init, phi_delta_init = pspline_hyperparameter_initials(
        alpha_phi_delta,
        beta_phi_delta,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
        divide_phi_by_delta=True,
    )
    _, phi_theta_init = pspline_hyperparameter_initials(
        alpha_phi_theta,
        beta_phi_theta,
        sampler.config.alpha_delta,
        sampler.config.beta_delta,
        divide_phi_by_delta=True,
    )
    init_values = {
        f"delta_{channel_index}": jnp.asarray(delta_init),
        f"phi_delta_{channel_index}": jnp.log(jnp.asarray(phi_delta_init)),
        f"weights_delta_{channel_index}": delta_weights_init,
    }
    if theta_count > 0:
        for theta_idx in range(theta_count):
            re_model = sampler.spline_model.get_theta_model(
                "re", channel_index, theta_idx
            )
            im_model = sampler.spline_model.get_theta_model(
                "im", channel_index, theta_idx
            )
            pfx = f"{channel_index}_{theta_idx}"
            _assign_theta_component_init_values(
                init_values,
                prefix=pfx,
                weights=(
                    jnp.asarray(re_model.weights),
                    jnp.asarray(im_model.weights),
                ),
                delta_init=jnp.asarray(delta_init),
                log_phi_init=jnp.log(jnp.asarray(phi_theta_init)),
            )
    return init_values


def run_single_block_vi(
    *,
    sampler,
    block_model: Callable[..., Any],
    vi_key,
    model_args: tuple[Any, ...],
    model_kwargs: Dict[str, Any],
    guide_spec: str,
    progress_bar: bool,
    init_values: Dict[str, jnp.ndarray],
):
    """Execute one VI run with a guarded retry for unstable ELBOs."""
    vi_result = fit_vi(
        model=block_model,
        rng_key=vi_key,
        vi_steps=sampler.config.vi_steps,
        optimizer_lr=sampler.config.vi_lr,
        model_args=model_args,
        model_kwargs=model_kwargs,
        guide=guide_spec,
        posterior_draws=sampler.config.vi_posterior_draws,
        progress_bar=progress_bar,
        init_values=init_values,
    )
    losses_arr = np.asarray(vi_result.losses)
    if not (losses_arr.size and not np.isfinite(losses_arr[-1])):
        return vi_result, losses_arr

    logger.warning(
        f"VI returned a non-finite ELBO (guide={vi_result.guide_name}); retrying with diag guide."
    )
    vi_result = fit_vi(
        model=block_model,
        rng_key=vi_key,
        vi_steps=min(int(sampler.config.vi_steps), 2000),
        optimizer_lr=min(float(sampler.config.vi_lr), 1e-3),
        model_args=model_args,
        model_kwargs=model_kwargs,
        guide="diag",
        posterior_draws=sampler.config.vi_posterior_draws,
        progress_bar=progress_bar,
        init_values=init_values,
    )
    losses_arr = np.asarray(vi_result.losses)
    return vi_result, losses_arr


def build_block_log_posterior_fn(
    block_model: Callable[..., Any],
    model_args: tuple[Any, ...],
    model_kwargs: Dict[str, Any],
):
    """Build callable log posterior for blocked VI init model selection."""
    return build_log_density_fn(
        partial(block_model, *model_args),
        model_kwargs,
    )
