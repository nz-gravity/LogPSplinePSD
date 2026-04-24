"""Core VI runners used by univariate and multivariate initialisation adapters."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import init_to_value

from ...diagnostics.vi_results import (
    _build_multivar_vi_diagnostics,
    _build_univar_vi_diagnostics,
)
from ...logger import logger
from ..pspline_block import pspline_hyperparameter_initials
from .common import suggest_guide_multivar, suggest_guide_univar
from .core import VIInitialisationArtifacts, VIResult, fit_vi
from .loo import compute_vi_loo_approximate_posterior


def _run_vi_initialisation(
    config,
    rng_key: jax.Array,
    *,
    model: Callable[..., Any],
    model_args: Tuple[Any, ...],
    model_kwargs: Optional[Dict[str, Any]] = None,
    guide: Optional[str],
    init_values: Optional[Dict[str, Any]] = None,
    postprocess: Callable[
        [VIResult], Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]
    ],
    log_likelihood_fn: Optional[Callable] = None,
) -> VIInitialisationArtifacts:
    """Run VI and return init strategy plus diagnostics."""
    if not getattr(config, "init_from_vi", False):
        return VIInitialisationArtifacts(None, rng_key, None)

    key_vi, key_run = jax.random.split(rng_key)
    progress_cfg = getattr(config, "vi_progress_bar", None)
    if progress_cfg is None:
        progress_bar = bool(getattr(config, "verbose", False))
    else:
        progress_bar = bool(progress_cfg)
    model_kwargs = model_kwargs or {}

    try:
        vi_result = fit_vi(
            model=model,
            rng_key=key_vi,
            vi_steps=config.vi_steps,
            optimizer_lr=config.vi_lr,
            model_args=model_args,
            model_kwargs=model_kwargs,
            guide=guide,
            posterior_draws=config.vi_posterior_draws,
            progress_bar=progress_bar,
            init_values=init_values,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        if getattr(config, "verbose", False):
            logger.warning(
                f"VI initialisation failed ({exc}) - using default init."
            )
        return VIInitialisationArtifacts(None, key_run, None)

    init_vals, diagnostics = postprocess(vi_result)
    init_strategy = init_to_value(values=init_vals)

    diagnostics = diagnostics or {}
    diagnostics.setdefault("guide", vi_result.guide_name)
    diagnostics.setdefault("losses", jnp.asarray(vi_result.losses))

    if vi_result.scales:
        scales_np = {
            name: np.asarray(jax.device_get(value))
            for name, value in vi_result.scales.items()
        }
        summary = sorted(
            (
                (name, float(np.mean(np.abs(val))), float(np.max(np.abs(val))))
                for name, val in scales_np.items()
            ),
            key=lambda x: x[2],
            reverse=True,
        )
        diagnostics["guide_scale_summary"] = summary[:5]

    losses_np = np.asarray(jax.device_get(vi_result.losses))
    if losses_np.size >= 2:
        window = min(200, losses_np.size)
        start = losses_np[-window]
        end = losses_np[-1]
        slope = (end - start) / max(1, window - 1)
        diagnostics["elbo_slope_recent"] = float(slope)
        if getattr(config, "verbose", False) and abs(slope) > 1e-3:
            logger.info(
                f"VI ELBO trend over last {window} steps: Δ={end - start:.3f}, slope/step={slope:.4f}"
            )

    loo_diag = compute_vi_loo_approximate_posterior(
        log_likelihood=(
            None
            if log_likelihood_fn is None
            else log_likelihood_fn(vi_result.samples or {})
        ),
        model=model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        guide=vi_result.guide,
        guide_params=vi_result.params,
        vi_samples=vi_result.samples,
        latent_samples=vi_result.latent_samples,
    )
    if loo_diag is not None:
        diagnostics.update(loo_diag)
        khat_max = float(loo_diag["pareto_k_max"])
        good_k = float(loo_diag["good_k"])
        if getattr(config, "verbose", False):
            logger.info(
                f"VI Pareto-k max = {khat_max:.3f} (threshold={good_k:.3f})"
            )
        if loo_diag["warning"]:
            logger.warning(
                "VI Pareto-k exceeds the ArviZ approximate-posterior threshold. "
                "Consider revisiting the guide or VI settings."
            )

    return VIInitialisationArtifacts(
        init_strategy,
        key_run,
        diagnostics,
        means=vi_result.means,
        posterior_draws=vi_result.samples,
    )


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

    log_ll_fn = getattr(sampler, "_build_vi_log_likelihood_dataset", None)
    return _run_vi_initialisation(
        sampler.config,
        sampler.rng_key,
        model=model,
        model_args=_univar_model_args(sampler),
        guide=guide_spec,
        init_values=init_values
        or default_init_values_univar(
            sampler.spline_model,
            alpha_phi=sampler.config.alpha_phi,
            beta_phi=sampler.config.beta_phi,
            alpha_delta=sampler.config.alpha_delta,
            beta_delta=sampler.config.beta_delta,
        ),
        postprocess=_postprocess,
        log_likelihood_fn=log_ll_fn,
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

    log_ll_fn = getattr(sampler, "_build_vi_log_likelihood_dataset", None)
    return _run_vi_initialisation(
        sampler.config,
        sampler.rng_key,
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
        log_likelihood_fn=log_ll_fn,
    )
