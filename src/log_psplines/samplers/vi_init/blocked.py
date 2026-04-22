"""Thin blocked multivariate VI adapters, including coarse-to-fine flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ...logger import logger
from .bridge import transfer_block_init_values
from .diagnostics import (
    _accumulate_block_vi_diagnostics,
    _assemble_blocked_vi_diagnostics,
    _block_site_names,
    _ensure_positive_definite_psd,
    _extract_multivar_design_psd,
    _extract_psd_q50,
    _median_vi_values,
    _prepare_block_accum,
    _rescale_multivar_psd_for_diagnostics,
    _sanitize_vi_init_values,
    _strip_coarse_vi_plot_arrays,
    _to_np,
    _vi_means_are_usable,
    interp_design_psd_to_fine,
)
from .runner import (
    build_block_init_values,
    build_block_log_posterior_fn,
    build_block_model_args,
    run_single_block_vi,
    select_vi_or_default_init,
)
from .transfer import coarse_vi_metadata, mark_coarse_vi


@dataclass
class BlockVIArtifacts:
    """Container holding per-block VI outputs for blocked samplers."""

    init_strategies: List[Optional[Callable[[Any], Any]]]
    mcmc_keys: List[jax.Array]
    rng_key: jax.Array
    diagnostics: Optional[Dict[str, Any]]


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

        from .guide import suggest_guide_block

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
    coarse_sampler,
    block_model: Callable[..., Any],
) -> BlockVIArtifacts:
    """Run blocked VI on a coarse grid then transfer to fine model."""
    metadata = coarse_vi_metadata(sampler)
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

        fine_design = interp_design_psd_to_fine(
            coarse_freq,
            fine_freq,
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
            merged_diagnostics = dict(coarse_diag)
            merged_diagnostics.update(metadata)
            merged_diagnostics["coarse_vi_attempted"] = 1
            merged_diagnostics["coarse_vi_success"] = 1
            merged_diagnostics["psd_matrix_complex"] = fine_design
            merged_diagnostics["psd_matrix"] = np.real(
                _rescale_multivar_psd_for_diagnostics(sampler, fine_design)
            )
            merged_diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
            merged_diagnostics["coarse_vi_freq"] = np.asarray(coarse_freq)
            coarse_losses = coarse_diag.get("losses")
            coarse_plot_psd = _extract_psd_q50(coarse_diag)
            if coarse_plot_psd is not None:
                merged_diagnostics["coarse_vi_psd"] = np.real(coarse_plot_psd)
                merged_diagnostics["coarse_vi_label"] = (
                    "Coarse-Grid VI Posterior Median"
                )
            else:
                merged_diagnostics["coarse_vi_psd"] = np.real(
                    _rescale_multivar_psd_for_diagnostics(
                        coarse_sampler,
                        np.asarray(coarse_design, dtype=np.complex128),
                    )
                )
                merged_diagnostics["coarse_vi_label"] = "Coarse-Grid VI Mean"
            if coarse_losses is not None:
                merged_diagnostics["coarse_losses"] = np.asarray(coarse_losses)
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

        merged_diagnostics = dict(fine_setup.diagnostics or {})
        merged_diagnostics.update(metadata)
        merged_diagnostics["coarse_vi_attempted"] = 1
        merged_diagnostics["coarse_vi_success"] = 1
        merged_diagnostics["psd_matrix_complex"] = fine_design
        merged_diagnostics["psd_matrix"] = np.real(
            _rescale_multivar_psd_for_diagnostics(sampler, fine_design)
        )
        merged_diagnostics["coarse_vi_nfreq"] = int(coarse_freq.size)
        merged_diagnostics["coarse_vi_freq"] = np.asarray(coarse_freq)
        coarse_diag = coarse_setup.diagnostics or {}
        coarse_plot_psd = _extract_psd_q50(coarse_diag)
        if coarse_plot_psd is not None:
            merged_diagnostics["coarse_vi_psd"] = np.real(coarse_plot_psd)
            merged_diagnostics["coarse_vi_label"] = (
                "Coarse-Grid VI Posterior Median"
            )
        else:
            merged_diagnostics["coarse_vi_psd"] = np.real(
                _rescale_multivar_psd_for_diagnostics(
                    coarse_sampler,
                    np.asarray(coarse_design, dtype=np.complex128),
                )
            )
            merged_diagnostics["coarse_vi_label"] = "Coarse-Grid VI Mean"
        coarse_losses = coarse_diag.get("losses")
        if coarse_losses is not None:
            merged_diagnostics["coarse_losses"] = np.asarray(coarse_losses)
        coarse_per_block = coarse_diag.get("losses_per_block")
        if coarse_per_block is not None:
            merged_diagnostics["coarse_losses_per_block"] = np.asarray(
                coarse_per_block
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
