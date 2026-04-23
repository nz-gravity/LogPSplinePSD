"""Thin blocked multivariate VI adapters, including coarse-to-fine flow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ...datatypes.multivar_utils import _interp_complex_matrix
from ...diagnostics.psd_compare import compute_multivar_riae_diagnostics
from ...diagnostics.vi_results import (
    _ensure_positive_definite_psd,
    _extract_multivar_design_psd,
    _extract_psd_q50,
    _extract_true_psd,
    _reconstruct_psd_quantiles_from_draws,
    _rescale_multivar_psd_for_diagnostics,
)
from ...logger import logger
from ..pspline_block import (
    build_log_density_fn,
    pspline_hyperparameter_initials,
)
from .bridge import transfer_block_init_values
from .common import (
    _median_vi_values,
    _sanitize_vi_init_values,
    _strip_coarse_vi_plot_arrays,
    _to_np,
    _to_np_dict,
    _vi_means_are_usable,
    suggest_guide_block,
)
from .core import fit_vi
from .loo import compute_vi_loo_approximate_posterior
from .plan import (
    VIWarmStartPlan,
    build_coarse_sampler_from_plan,
    coarse_vi_metadata,
    mark_coarse_vi,
)
from .runner import (
    _assign_theta_component_init_values,
    select_vi_or_default_init,
)


@dataclass
class BlockVIArtifacts:
    """Container holding per-block VI outputs for blocked samplers."""

    init_strategies: List[Optional[Callable[[Any], Any]]]
    mcmc_keys: List[jax.Array]
    rng_key: jax.Array
    diagnostics: Optional[Dict[str, Any]]


def _coarse_block_plot_payload(
    *,
    coarse_diag: Dict[str, Any],
    coarse_sampler,
    sampler,
    coarse_design: np.ndarray,
) -> Dict[str, Any]:
    """Return coarse-grid plotting diagnostics for blocked VI flows."""
    coarse_freq = np.asarray(coarse_sampler.freq)
    coarse_plot_psd = _extract_psd_q50(coarse_diag)
    if coarse_plot_psd is not None:
        coarse_psd = np.real(coarse_plot_psd)
        coarse_label = "Coarse-Grid VI Posterior Median"
    else:
        coarse_psd = np.real(
            _rescale_multivar_psd_for_diagnostics(
                coarse_sampler,
                np.asarray(coarse_design, dtype=np.complex128),
            )
        )
        coarse_label = "Coarse-Grid VI Mean"

    payload: Dict[str, Any] = {
        "psd_matrix_complex": coarse_design,
        "psd_matrix": np.real(
            _rescale_multivar_psd_for_diagnostics(sampler, coarse_design)
        ),
        "coarse_vi_nfreq": int(coarse_freq.size),
        "coarse_vi_freq": coarse_freq,
        "coarse_vi_psd": coarse_psd,
        "coarse_vi_label": coarse_label,
    }
    coarse_losses = coarse_diag.get("losses")
    if coarse_losses is not None:
        payload["coarse_losses"] = np.asarray(coarse_losses)
    coarse_per_block = coarse_diag.get("losses_per_block")
    if coarse_per_block is not None:
        payload["coarse_losses_per_block"] = np.asarray(coarse_per_block)
    return payload


def _prepare_block_accum(sampler) -> Dict[str, Any]:
    posterior_draws = (
        sampler.config.vi_posterior_draws
        if getattr(sampler.config, "vi_posterior_draws", 0) > 0
        else 0
    )
    store_draws = sampler.config.init_from_vi and posterior_draws > 0
    theta_shape = (posterior_draws, sampler.N, sampler.n_theta)
    return {
        "posterior_draws": posterior_draws,
        "store_draws": store_draws,
        "log_delta_draws": (
            np.zeros((posterior_draws, sampler.N, sampler.p), dtype=np.float32)
            if store_draws
            else None
        ),
        "theta_re_draws": (
            np.zeros(theta_shape, dtype=np.float32)
            if store_draws and sampler.n_theta > 0
            else None
        ),
        "theta_im_draws": (
            np.zeros(theta_shape, dtype=np.float32)
            if store_draws and sampler.n_theta > 0
            else None
        ),
        "draws_missing": False,
        "draws_recorded": 0,
        "vi_samples": {},
        "vi_losses_blocks": [],
        "vi_guides": [],
        "vi_log_delta_means": [],
        "vi_theta_re_mean": None,
        "vi_theta_im_mean": None,
        "pareto_k_blocks": [],
        "pareto_k_max_per_block": [],
        "pareto_good_k_per_block": [],
        "loo_warning_per_block": [],
    }


def _aggregate_psis_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    pareto_k_blocks: List[np.ndarray],
    pareto_k_max_per_block: List[float],
    pareto_good_k_per_block: List[float],
    loo_warning_per_block: List[bool],
    verbose: bool,
) -> None:
    if pareto_k_max_per_block:
        khat_array = np.asarray(pareto_k_max_per_block, dtype=float)
        khat_max = float(np.nanmax(khat_array))
        good_k = (
            np.nan
            if not pareto_good_k_per_block
            else float(np.nanmax(np.asarray(pareto_good_k_per_block)))
        )
        warning = bool(np.any(np.asarray(loo_warning_per_block, dtype=bool)))
        diagnostics.update(
            pareto_k_per_block=khat_array,
            pareto_k=(
                None
                if not pareto_k_blocks
                else np.concatenate(
                    [
                        np.asarray(block, dtype=np.float64).reshape(-1)
                        for block in pareto_k_blocks
                    ]
                )
            ),
            pareto_k_good_k=good_k,
            pareto_k_max=khat_max,
            loo_warning=warning,
            psis_khat_per_block=khat_array,
            psis_khat_max=khat_max,
            psis_khat_status=(
                "unknown"
                if not np.isfinite(khat_max)
                else "warn" if warning else "ok"
            ),
            psis_khat_threshold=good_k,
            psis_flag_warn=warning,
            psis_flag_critical=warning,
        )
        if verbose:
            logger.info(
                f"VI Pareto-k max (blocked) = {khat_max:.3f} (threshold={good_k:.3f})"
            )
        if warning:
            logger.warning(
                "VI Pareto-k exceeds the ArviZ approximate-posterior threshold. "
                "Consider adjusting the guide or VI settings."
            )


def _accumulate_block_vi_diagnostics(
    *,
    sampler,
    vi_result,
    block_model,
    model_args: tuple,
    model_kwargs: Dict[str, Any],
    channel_index: int,
    theta_start: int,
    theta_count: int,
    delta_basis,
    accum: Dict[str, Any],
) -> None:
    accum["vi_losses_blocks"].append(_to_np(vi_result.losses))
    accum["vi_guides"].append(vi_result.guide_name)
    accum["vi_samples"].update(_to_np_dict(vi_result.samples))
    full_log_likelihood = sampler._build_vi_log_likelihood_dataset(
        accum["vi_samples"]
    )
    block_log_likelihood = (
        None
        if full_log_likelihood is None
        else full_log_likelihood[[f"log_likelihood_block_{channel_index}"]]
    )

    loo_diag = compute_vi_loo_approximate_posterior(
        log_likelihood=block_log_likelihood,
        model=block_model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        guide=vi_result.guide,
        guide_params=vi_result.params,
        vi_samples=vi_result.samples,
        latent_samples=vi_result.latent_samples,
        var_name=f"log_likelihood_block_{channel_index}",
    )
    if loo_diag is not None:
        pareto_k = loo_diag.get("pareto_k")
        if pareto_k is not None:
            accum["pareto_k_blocks"].append(
                np.asarray(pareto_k, dtype=np.float64)
            )
        accum["pareto_k_max_per_block"].append(float(loo_diag["pareto_k_max"]))
        accum["pareto_good_k_per_block"].append(float(loo_diag["good_k"]))
        accum["loo_warning_per_block"].append(bool(loo_diag["warning"]))

    weights_delta_name = f"weights_delta_{channel_index}"
    weights_delta = vi_result.means.get(weights_delta_name)
    if weights_delta is not None:
        accum["vi_log_delta_means"].append(
            _to_np(jnp.einsum("nk,k->n", delta_basis, weights_delta))
        )

    if sampler.n_theta > 0 and theta_count > 0:
        for key in ("vi_theta_re_mean", "vi_theta_im_mean"):
            if accum[key] is None:
                accum[key] = np.zeros(
                    (sampler.N, sampler.n_theta), dtype=np.float32
                )
        theta_slice = slice(theta_start, theta_start + theta_count)
        for part, accum_key in (
            ("re", "vi_theta_re_mean"),
            ("im", "vi_theta_im_mean"),
        ):
            accum[accum_key][:, theta_slice] = np.stack(
                [
                    _to_np(
                        jnp.einsum(
                            "nk,k->n",
                            jnp.asarray(
                                sampler.spline_model.get_theta_model(
                                    part, channel_index, theta_idx
                                ).basis
                            ),
                            vi_result.means[
                                f"weights_theta_{part}_{channel_index}_{theta_idx}"
                            ],
                        )
                    )
                    for theta_idx in range(theta_count)
                ],
                axis=1,
            )

    if not accum["store_draws"] or vi_result.samples is None:
        return

    def _record_draws(weights_samples, basis, target_buf, col):
        if weights_samples is None:
            accum["draws_missing"] = True
            return
        weights_samples = jnp.asarray(weights_samples)
        n = min(accum["posterior_draws"], weights_samples.shape[0])
        accum["draws_recorded"] = (
            n
            if accum["draws_recorded"] == 0
            else min(accum["draws_recorded"], n)
        )
        target_buf[:n, :, col] = _to_np(
            weights_samples[:n]
            @ jnp.asarray(basis, dtype=weights_samples.dtype).T
        )

    _record_draws(
        vi_result.samples.get(weights_delta_name),
        delta_basis,
        accum["log_delta_draws"],
        channel_index,
    )
    if (
        sampler.n_theta > 0
        and theta_count > 0
        and accum["theta_re_draws"] is not None
    ):
        for theta_idx in range(theta_count):
            column = theta_start + theta_idx
            for part, buf_key in (
                ("re", "theta_re_draws"),
                ("im", "theta_im_draws"),
            ):
                _record_draws(
                    vi_result.samples.get(
                        f"weights_theta_{part}_{channel_index}_{theta_idx}"
                    ),
                    jnp.asarray(
                        sampler.spline_model.get_theta_model(
                            part, channel_index, theta_idx
                        ).basis
                    ),
                    accum[buf_key],
                    column,
                )


def _assemble_blocked_vi_diagnostics(
    *,
    sampler,
    accum: Dict[str, Any],
    _rescale_psd: Callable[[np.ndarray], np.ndarray],
) -> Optional[Dict[str, Any]]:
    diagnostics = None
    if sampler.config.init_from_vi and accum["vi_log_delta_means"]:
        log_delta_vi_np = (
            np.stack(accum["vi_log_delta_means"], axis=1)
            if len(accum["vi_log_delta_means"]) == sampler.p
            else None
        )
        vi_psd = vi_psd_np = None
        if log_delta_vi_np is not None:
            theta_re_vi = (
                accum["vi_theta_re_mean"]
                if sampler.n_theta > 0
                else np.zeros((sampler.N, 0), dtype=np.float32)
            )
            theta_im_vi = (
                accum["vi_theta_im_mean"]
                if sampler.n_theta > 0
                else np.zeros((sampler.N, 0), dtype=np.float32)
            )
            vi_psd = sampler.spline_model.reconstruct_psd_matrix(
                jnp.asarray(log_delta_vi_np)[None, ...],
                jnp.asarray(theta_re_vi)[None, ...],
                jnp.asarray(theta_im_vi)[None, ...],
                n_samples_max=1,
            )[0]
            vi_psd_np = _rescale_psd(np.asarray(vi_psd))

        psd_quantiles = coherence_quantiles = None
        log_delta_draws = accum["log_delta_draws"]
        if (
            accum["store_draws"]
            and not accum["draws_missing"]
            and log_delta_draws is not None
        ):
            available = min(
                accum["draws_recorded"] or accum["posterior_draws"],
                log_delta_draws.shape[0],
            )
            if available > 0:
                theta_re = accum["theta_re_draws"]
                tr_s = (
                    jnp.asarray(theta_re[:available], dtype=jnp.float32)
                    if sampler.n_theta > 0 and theta_re is not None
                    else jnp.zeros(
                        (available, sampler.N, 0), dtype=jnp.float32
                    )
                )
                ti_s = (
                    jnp.asarray(
                        accum["theta_im_draws"][:available], dtype=jnp.float32
                    )
                    if sampler.n_theta > 0 and theta_re is not None
                    else jnp.zeros_like(tr_s)
                )
                psd_quantiles, coherence_quantiles, vi_psd_np = (
                    _reconstruct_psd_quantiles_from_draws(
                        spline_model=sampler.spline_model,
                        config=sampler.config,
                        log_delta_samples=jnp.asarray(
                            log_delta_draws[:available], dtype=jnp.float32
                        ),
                        theta_re_samples=tr_s,
                        theta_im_samples=ti_s,
                        p=sampler.p,
                        rescale_fn=_rescale_psd,
                    )
                )

        valid_losses = [
            arr
            for arr in accum["vi_losses_blocks"]
            if arr.size and np.isfinite(arr).all()
        ]
        losses_stack = (
            None
            if not valid_losses
            else np.stack(
                [
                    arr[-min(arr.shape[0] for arr in valid_losses) :]
                    for arr in valid_losses
                ],
                axis=0,
            )
        )
        diagnostics = {
            "losses": (
                np.asarray([])
                if losses_stack is None
                else losses_stack.mean(axis=0)
            ),
            "losses_per_block": losses_stack,
            "guide": (
                ",".join(sorted(set(accum["vi_guides"])))
                if accum["vi_guides"]
                else "vi"
            ),
            "psd_matrix": vi_psd_np,
            "psd_matrix_complex": (
                None
                if log_delta_vi_np is None
                else np.asarray(vi_psd, dtype=np.complex128)
            ),
            "true_psd": _extract_true_psd(sampler),
        }
        if psd_quantiles is not None:
            diagnostics["psd_quantiles"] = psd_quantiles
        if coherence_quantiles is not None:
            diagnostics["coherence_quantiles"] = coherence_quantiles
        if diagnostics["true_psd"] is not None and vi_psd_np is not None:
            diagnostics.update(
                compute_multivar_riae_diagnostics(
                    vi_psd_np,
                    np.real(np.asarray(diagnostics["true_psd"])),
                    np.asarray(sampler.freq, dtype=np.float64),
                    psd_quantiles=psd_quantiles,
                )
            )
    if accum["vi_samples"]:
        diagnostics = diagnostics or {}
        diagnostics["vi_samples"] = accum["vi_samples"]
    if diagnostics is not None and (
        accum["pareto_k_blocks"] or accum["pareto_k_max_per_block"]
    ):
        _aggregate_psis_diagnostics(
            diagnostics,
            pareto_k_blocks=accum["pareto_k_blocks"],
            pareto_k_max_per_block=accum["pareto_k_max_per_block"],
            pareto_good_k_per_block=accum["pareto_good_k_per_block"],
            loo_warning_per_block=accum["loo_warning_per_block"],
            verbose=sampler.config.verbose,
        )
    return diagnostics


def _block_site_names(channel_index: int) -> list[str]:
    return [f"weights_delta_{channel_index}"] + [
        f"weights_{prefix}_{channel_index}_{theta_idx}"
        for theta_idx in range(channel_index)
        for prefix in ("theta_re", "theta_im")
    ]


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


def prepare_block_vi(
    sampler,
    *,
    rng_key: jax.Array,
    block_model: Callable[..., Any],
    init_overrides: Optional[Dict[int, Dict[str, jnp.ndarray]]] = None,
) -> BlockVIArtifacts:
    """Run VI per block for blocked multivariate samplers."""

    guide_cfg = getattr(sampler.config, "vi_guide", None) or "auto(block)"
    steps = int(getattr(sampler.config, "vi_steps", 0) or 0)
    draws = int(getattr(sampler.config, "vi_posterior_draws", 0) or 0)

    if sampler.config.init_from_vi:
        logger.info(
            "Running VI initialisation per block "
            f"(p={sampler.p}, guide={guide_cfg}, steps={steps}, Lr={sampler.config.vi_lr}, posterior_draws={draws})..."
        )
    else:
        logger.info(
            f"Skipping VI initialisation per block (p={sampler.p}); using default NUTS initialisation."
        )

    p = sampler.p
    init_strategies: List[Optional[Callable[[Any], Any]]] = [None] * p
    mcmc_keys: List[jax.Array] = [jax.random.PRNGKey(0)] * p

    rescale_psd = lambda arr: _rescale_multivar_psd_for_diagnostics(
        sampler, arr
    )
    accum = _prepare_block_accum(sampler)
    current_key = rng_key

    for channel_index in range(p):
        current_key, block_key = jax.random.split(current_key)

        if sampler.config.init_from_vi:
            vi_key, mcmc_key = jax.random.split(block_key)
        else:
            vi_key = None
            mcmc_key = block_key

        mcmc_keys[channel_index] = mcmc_key

        if not sampler.config.init_from_vi or vi_key is None:
            continue

        delta_basis = sampler.all_bases[channel_index]
        delta_model = sampler.spline_model.diagonal_models[channel_index]
        delta_weights_init = jnp.asarray(delta_model.weights)

        theta_start = channel_index * (channel_index - 1) // 2
        theta_count = channel_index
        theta_basis_cols = 0
        if theta_count > 0:
            theta_basis_cols = max(
                int(
                    sampler.spline_model.get_theta_model(
                        "re", channel_index, theta_idx
                    ).basis.shape[1]
                )
                for theta_idx in range(theta_count)
            )

        guide_spec = sampler.config.vi_guide or suggest_guide_block(
            delta_basis.shape[1], theta_count, theta_basis_cols
        )
        progress_bar = (
            sampler.config.vi_progress_bar
            if sampler.config.vi_progress_bar is not None
            else sampler.config.verbose
        )

        try:
            alpha_phi_theta = getattr(
                sampler.config, "alpha_phi_theta", sampler.config.alpha_phi
            )
            beta_phi_theta = getattr(
                sampler.config, "beta_phi_theta", sampler.config.beta_phi
            )

            init_values = build_block_init_values(
                sampler=sampler,
                channel_index=channel_index,
                theta_count=theta_count,
                delta_weights_init=delta_weights_init,
                alpha_phi_theta=alpha_phi_theta,
                beta_phi_theta=beta_phi_theta,
            )
            if init_overrides and channel_index in init_overrides:
                init_values.update(init_overrides[channel_index])

            model_args, model_kwargs = build_block_model_args(
                sampler, channel_index, alpha_phi_theta, beta_phi_theta
            )
            block_log_posterior_fn = build_block_log_posterior_fn(
                block_model, model_args, model_kwargs
            )

            vi_result, losses_arr = run_single_block_vi(
                sampler=sampler,
                block_model=block_model,
                vi_key=vi_key,
                model_args=model_args,
                model_kwargs=model_kwargs,
                guide_spec=guide_spec,
                progress_bar=progress_bar,
                init_values=init_values,
            )

            vi_means = {
                name: jnp.asarray(value)
                for name, value in (vi_result.means or {}).items()
            }
            default_init_values = dict(init_values)
            vi_median = _median_vi_values(
                draws={
                    name: jnp.asarray(value)
                    for name, value in (vi_result.samples or {}).items()
                },
            )
            vi_candidate = _sanitize_vi_init_values(vi_median)
            if not _vi_means_are_usable(vi_candidate or {}):
                vi_candidate = _sanitize_vi_init_values(vi_means)
            if losses_arr.size and not np.isfinite(losses_arr[-1]):
                logger.warning(
                    f"VI returned a non-finite ELBO for block {channel_index} "
                    f"(guide={vi_result.guide_name}); skipping VI-based init."
                )
            elif vi_candidate is None or not _vi_means_are_usable(
                vi_candidate
            ):
                logger.warning(
                    f"VI produced invalid init parameters for block {channel_index} "
                    f"(guide={vi_result.guide_name}); skipping VI-based init."
                )
            else:

                def _block_log_posterior(
                    vals: Dict[str, jnp.ndarray],
                ) -> float:
                    return float(
                        block_log_posterior_fn(
                            {k: jnp.asarray(v) for k, v in vals.items()}
                        )
                    )

                init_strategies[channel_index] = select_vi_or_default_init(
                    vi_values=vi_candidate,
                    default_values=default_init_values,
                    log_posterior_fn=_block_log_posterior,
                    log_prefix=f"Blocked VI init channel {channel_index}",
                )

            _accumulate_block_vi_diagnostics(
                sampler=sampler,
                vi_result=vi_result,
                block_model=block_model,
                model_args=model_args,
                model_kwargs=model_kwargs,
                channel_index=channel_index,
                theta_start=theta_start,
                theta_count=theta_count,
                delta_basis=delta_basis,
                accum=accum,
            )

        except Exception as exc:  # pragma: no cover
            if sampler.config.verbose:
                logger.warning(
                    f"VI block initialisation failed [channel {channel_index}]: {exc}"
                )
            if accum["store_draws"]:
                accum["draws_missing"] = True

    diagnostics = _assemble_blocked_vi_diagnostics(
        sampler=sampler,
        accum=accum,
        _rescale_psd=rescale_psd,
    )

    return BlockVIArtifacts(
        init_strategies=init_strategies,
        mcmc_keys=mcmc_keys,
        rng_key=current_key,
        diagnostics=diagnostics,
    )


def prepare_coarse_block_vi(
    sampler,
    *,
    warm_start_plan: VIWarmStartPlan,
    block_model: Callable[..., Any],
) -> BlockVIArtifacts:
    """Run blocked VI on a coarse grid then transfer to fine model."""
    metadata = coarse_vi_metadata(warm_start_plan)
    coarse_sampler = build_coarse_sampler_from_plan(sampler, warm_start_plan)
    coarse_setup = prepare_block_vi(
        coarse_sampler,
        rng_key=sampler.rng_key,
        block_model=block_model,
    )
    diagnostics = mark_coarse_vi(
        coarse_setup.diagnostics,
        metadata,
        attempted=True,
        success=False,
    )

    coarse_design = _extract_multivar_design_psd(diagnostics)
    if coarse_design is None:
        logger.warning(
            "Coarse blocked VI did not produce a valid PSD matrix warm "
            "start; using default init."
        )
        return BlockVIArtifacts(
            init_strategies=[None] * sampler.p,
            mcmc_keys=coarse_setup.mcmc_keys,
            rng_key=coarse_setup.rng_key,
            diagnostics=_strip_coarse_vi_plot_arrays(diagnostics),
        )

    try:
        fine_freq = np.asarray(sampler.freq, dtype=np.float64)
        coarse_freq = np.asarray(coarse_sampler.freq, dtype=np.float64)

        fine_design = _interp_complex_matrix(
            coarse_freq.astype(np.float64),
            fine_freq.astype(np.float64),
            np.asarray(coarse_design, dtype=np.complex128),
        )
        fine_design = _ensure_positive_definite_psd(fine_design)

        coarse_vi_samples = (coarse_setup.diagnostics or {}).get(
            "vi_samples"
        ) or {}
        init_overrides: Dict[int, Dict[str, jnp.ndarray]] = {}
        for channel_index in range(sampler.p):
            key = f"weights_delta_{channel_index}"
            if key not in coarse_vi_samples:
                continue
            coarse_means = _median_vi_values(
                draws={
                    name: jnp.asarray(value)
                    for name, value in coarse_vi_samples.items()
                },
                site_names=_block_site_names(channel_index),
            )
            if not coarse_means:
                continue
            default_vals = build_block_init_values(
                sampler=sampler,
                channel_index=channel_index,
                theta_count=channel_index,
                delta_weights_init=jnp.asarray(
                    sampler.spline_model.diagonal_models[channel_index].weights
                ),
                alpha_phi_theta=getattr(
                    sampler.config,
                    "alpha_phi_theta",
                    sampler.config.alpha_phi,
                ),
                beta_phi_theta=getattr(
                    sampler.config,
                    "beta_phi_theta",
                    sampler.config.beta_phi,
                ),
            )
            transferred = transfer_block_init_values(
                draw_values=coarse_means,
                channel_index=channel_index,
                coarse_sampler=coarse_sampler,
                fine_sampler=sampler,
                coarse_freq=coarse_freq,
                fine_freq=fine_freq,
                default_init_values=default_vals,
            )
            init_overrides[channel_index] = transferred

        coarse_only = bool(getattr(sampler.config, "vi_coarse_only", False))

        if coarse_only:
            coarse_diag = coarse_setup.diagnostics or {}
            merged_diagnostics = mark_coarse_vi(
                coarse_diag,
                metadata,
                attempted=True,
                success=True,
            )
            merged_diagnostics.update(
                _coarse_block_plot_payload(
                    coarse_diag=coarse_diag,
                    coarse_sampler=coarse_sampler,
                    sampler=sampler,
                    coarse_design=fine_design,
                )
            )
            return BlockVIArtifacts(
                init_strategies=[None] * sampler.p,
                mcmc_keys=coarse_setup.mcmc_keys,
                rng_key=coarse_setup.rng_key,
                diagnostics=merged_diagnostics,
            )

        fine_setup = prepare_block_vi(
            sampler,
            rng_key=coarse_setup.rng_key,
            block_model=block_model,
            init_overrides=init_overrides or None,
        )

        coarse_diag = coarse_setup.diagnostics or {}
        merged_diagnostics = mark_coarse_vi(
            fine_setup.diagnostics,
            metadata,
            attempted=True,
            success=True,
        )
        merged_diagnostics.update(
            _coarse_block_plot_payload(
                coarse_diag=coarse_diag,
                coarse_sampler=coarse_sampler,
                sampler=sampler,
                coarse_design=fine_design,
            )
        )

        return BlockVIArtifacts(
            init_strategies=fine_setup.init_strategies,
            mcmc_keys=fine_setup.mcmc_keys,
            rng_key=fine_setup.rng_key,
            diagnostics=merged_diagnostics,
        )
    except Exception as exc:
        logger.warning(
            f"Could not transfer coarse blocked VI warm start to full "
            f"grid: {exc}"
        )
        return BlockVIArtifacts(
            init_strategies=[None] * sampler.p,
            mcmc_keys=coarse_setup.mcmc_keys,
            rng_key=coarse_setup.rng_key,
            diagnostics=_strip_coarse_vi_plot_arrays(diagnostics),
        )


__all__ = ["BlockVIArtifacts", "prepare_block_vi", "prepare_coarse_block_vi"]
