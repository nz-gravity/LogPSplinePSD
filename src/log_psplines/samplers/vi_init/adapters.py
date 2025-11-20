"""Adapters that encapsulate VI initialisation for samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import init_to_value

from ...logger import logger
from .core import fit_vi
from .guide import (
    suggest_guide_block,
    suggest_guide_multivar,
    suggest_guide_univar,
)
from .mixin import VIInitialisationArtifacts


def _cap_psd_draws(config, desired: int) -> int:
    """Limit PSD reconstructions to a manageable number of draws."""
    max_cfg = int(getattr(config, "vi_psd_max_draws", 0) or 0)
    capped = int(desired)
    if max_cfg > 0:
        capped = min(capped, max_cfg)
    return max(1, capped)


def compute_vi_artifacts_univar(
    sampler,
    *,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """Run VI for univariate samplers and return initialisation artifacts."""

    guide_spec = sampler.config.vi_guide or suggest_guide_univar(
        sampler.n_weights + 2
    )

    def _postprocess(vi_result):
        init_values = {
            name: jnp.asarray(value) for name, value in vi_result.means.items()
        }

        weights = vi_result.means.get("weights")
        weights_np = None
        vi_psd = None
        psd_quantiles = None
        scaling = float(getattr(sampler.config, "scaling_factor", 1.0) or 1.0)

        if weights is not None:
            weights_np = np.asarray(jax.device_get(weights))
            ln_psd = sampler.spline_model(vi_result.means["weights"])
            vi_psd = np.asarray(jax.device_get(jnp.exp(ln_psd))) * scaling

        if vi_result.samples is not None:
            weights_draws = vi_result.samples.get("weights")
            if weights_draws is not None and weights_draws.size:
                ln_psd_draws = jax.vmap(sampler.spline_model)(
                    jnp.asarray(weights_draws)
                )
                psd_draws = jnp.exp(ln_psd_draws)
                psd_draws_np = np.asarray(jax.device_get(psd_draws)) * scaling
                q05, q50, q95 = np.percentile(
                    psd_draws_np, [5, 50, 95], axis=0
                )
                psd_quantiles = {
                    "q05": q05,
                    "q50": q50,
                    "q95": q95,
                }

        true_psd = None
        if sampler.config.true_psd is not None:
            true_psd = np.asarray(jax.device_get(sampler.config.true_psd))

        diagnostics = {
            "weights": weights_np,
            "psd": vi_psd,
            "true_psd": true_psd,
        }
        if psd_quantiles is not None:
            diagnostics["psd_quantiles"] = psd_quantiles
        if vi_result.samples is not None:
            diagnostics["vi_samples"] = {
                name: np.asarray(jax.device_get(value))
                for name, value in vi_result.samples.items()
            }
        return init_values, diagnostics

    return sampler._run_vi_initialisation(
        model=model,
        model_args=(
            sampler.log_pdgrm,
            sampler.basis_matrix,
            sampler.penalty_matrix,
            sampler.log_parametric,
            sampler.freq_weights,
            sampler.config.alpha_phi,
            sampler.config.beta_phi,
            sampler.config.alpha_delta,
            sampler.config.beta_delta,
        ),
        guide=guide_spec,
        postprocess=_postprocess,
    )


def _count_multivar_latents(sampler) -> int:
    total_latents = 0
    for model in sampler.spline_model.diagonal_models:
        total_latents += model.n_basis + 2

    if sampler.n_theta > 0:
        total_latents += sampler.spline_model.offdiag_re_model.n_basis + 2
        total_latents += sampler.spline_model.offdiag_im_model.n_basis + 2
    return total_latents


def compute_vi_artifacts_multivar(
    sampler,
    *,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """Run VI for fully coupled multivariate samplers."""

    total_latents = _count_multivar_latents(sampler)
    guide_spec = sampler.config.vi_guide or suggest_guide_multivar(
        total_latents
    )

    def _postprocess(vi_result):
        init_values = {
            name: jnp.asarray(value) for name, value in vi_result.means.items()
        }

        scaling = float(getattr(sampler.config, "scaling_factor", 1.0) or 1.0)
        channel_stds = getattr(sampler.fft_data, "channel_stds", None)
        if channel_stds is not None:
            channel_stds = np.asarray(channel_stds, dtype=np.float32)
            scale_matrix = np.outer(channel_stds, channel_stds).astype(
                np.float32
            )
            factor_matrix = scale_matrix
            scalar_factor = None
        else:
            factor_matrix = None
            scalar_factor = scaling

        def _rescale_psd(arr: np.ndarray) -> np.ndarray:
            if factor_matrix is not None:
                return arr * factor_matrix
            return arr * scalar_factor

        vi_psd_np = None
        psd_quantiles = None
        coherence_quantiles = None
        try:
            log_delta_terms: List[jnp.ndarray] = []
            for channel_index in range(sampler.n_channels):
                weights_name = f"weights_delta_{channel_index}"
                weights = vi_result.means.get(weights_name)
                if weights is None:
                    raise KeyError(weights_name)
                basis = sampler.all_bases[channel_index]
                component_eval = jnp.einsum("nk,k->n", basis, weights)
                log_delta_terms.append(component_eval)
            log_delta_sq = jnp.stack(log_delta_terms, axis=1)

            if sampler.n_theta > 0:
                basis_theta = sampler.all_bases[sampler.n_channels]
                weights_theta_re = vi_result.means.get("weights_theta_re")
                weights_theta_im = vi_result.means.get("weights_theta_im")
                if weights_theta_re is None or weights_theta_im is None:
                    raise KeyError("theta weights")
                theta_re_base = jnp.einsum(
                    "nk,k->n", basis_theta, weights_theta_re
                )
                theta_im_base = jnp.einsum(
                    "nk,k->n", basis_theta, weights_theta_im
                )
                theta_re = jnp.tile(
                    theta_re_base[:, None], (1, max(1, sampler.n_theta))
                )
                theta_im = jnp.tile(
                    theta_im_base[:, None], (1, max(1, sampler.n_theta))
                )
            else:
                theta_re = jnp.zeros((sampler.n_freq, 0))
                theta_im = jnp.zeros((sampler.n_freq, 0))

            vi_psd = sampler.spline_model.reconstruct_psd_matrix(
                log_delta_sq[None, ...],
                theta_re[None, ...],
                theta_im[None, ...],
                n_samples_max=1,
            )[0]
            vi_psd_np = _rescale_psd(np.asarray(vi_psd))

            samples_tree = vi_result.samples or {}
            if samples_tree:
                log_delta_draws = []
                n_draws = None
                for channel_index in range(sampler.n_channels):
                    weights_name = f"weights_delta_{channel_index}"
                    weights_samples = samples_tree.get(weights_name)
                    if weights_samples is None:
                        log_delta_draws = []
                        break
                    weights_samples = jnp.asarray(weights_samples)
                    if n_draws is None:
                        n_draws = weights_samples.shape[0]
                    basis = sampler.all_bases[channel_index]
                    log_delta_draws.append(
                        weights_samples
                        @ jnp.asarray(basis, dtype=weights_samples.dtype).T
                    )

                if log_delta_draws:
                    log_delta_samples = jnp.stack(log_delta_draws, axis=2)
                    if sampler.n_theta > 0:
                        basis_theta = sampler.all_bases[sampler.n_channels]
                        weights_theta_re_samples = samples_tree.get(
                            "weights_theta_re"
                        )
                        weights_theta_im_samples = samples_tree.get(
                            "weights_theta_im"
                        )
                        if (
                            weights_theta_re_samples is not None
                            and weights_theta_im_samples is not None
                        ):
                            weights_theta_re_samples = jnp.asarray(
                                weights_theta_re_samples
                            )
                            weights_theta_im_samples = jnp.asarray(
                                weights_theta_im_samples
                            )
                            theta_re_base = (
                                weights_theta_re_samples
                                @ jnp.asarray(
                                    basis_theta,
                                    dtype=weights_theta_re_samples.dtype,
                                ).T
                            )
                            theta_im_base = (
                                weights_theta_im_samples
                                @ jnp.asarray(
                                    basis_theta,
                                    dtype=weights_theta_im_samples.dtype,
                                ).T
                            )
                            theta_re_samples = jnp.repeat(
                                theta_re_base[:, :, None],
                                sampler.n_theta,
                                axis=2,
                            )
                            theta_im_samples = jnp.repeat(
                                theta_im_base[:, :, None],
                                sampler.n_theta,
                                axis=2,
                            )
                        else:
                            theta_re_samples = jnp.zeros(
                                (
                                    log_delta_samples.shape[0],
                                    sampler.n_freq,
                                    sampler.n_theta,
                                )
                            )
                            theta_im_samples = jnp.zeros_like(theta_re_samples)
                    else:
                        theta_re_samples = jnp.zeros(
                            (log_delta_samples.shape[0], sampler.n_freq, 0)
                        )
                        theta_im_samples = jnp.zeros_like(theta_re_samples)

                    n_psd_draws = _cap_psd_draws(
                        sampler.config,
                        min(
                            log_delta_samples.shape[0],
                            max(1, sampler.config.vi_posterior_draws),
                        ),
                    )
                    if n_psd_draws < log_delta_samples.shape[0]:
                        logger.debug(
                            "Capping VI PSD reconstruction to %d draws "
                            "(limit=%d).",
                            n_psd_draws,
                            getattr(sampler.config, "vi_psd_max_draws", 0),
                        )
                    log_delta_samples = log_delta_samples[:n_psd_draws]
                    theta_re_samples = theta_re_samples[:n_psd_draws]
                    theta_im_samples = theta_im_samples[:n_psd_draws]
                    psd_real_q, psd_imag_q, coh_percentiles = (
                        sampler.spline_model.compute_psd_quantiles(
                            log_delta_samples,
                            theta_re_samples,
                            theta_im_samples,
                            percentiles=[5.0, 50.0, 95.0],
                            n_samples_max=n_psd_draws,
                            compute_coherence=sampler.n_channels > 1,
                        )
                    )
                    psd_real_q = _rescale_psd(psd_real_q)
                    psd_imag_q = _rescale_psd(psd_imag_q)

                    psd_quantiles = {
                        "real": {
                            "q05": psd_real_q[0],
                            "q50": psd_real_q[1],
                            "q95": psd_real_q[2],
                        },
                        "imag": {
                            "q05": psd_imag_q[0],
                            "q50": psd_imag_q[1],
                            "q95": psd_imag_q[2],
                        },
                    }
                    vi_psd_np = psd_quantiles["real"]["q50"]

                    if coh_percentiles is not None:
                        coh_percentiles *= 1.0  # already dimensionless
                        coherence_quantiles = {
                            "q05": coh_percentiles[0],
                            "q50": coh_percentiles[1],
                            "q95": coh_percentiles[2],
                        }
        except OverflowError as err:  # pragma: no cover - defensive fallback
            if sampler.config.verbose:
                logger.warning(f"Could not build VI PSD diagnostics: {err}")
                # print full traceback
                import traceback

                traceback.print_exc()

        true_psd = None
        if sampler.config.true_psd is not None:
            true_psd = np.asarray(jax.device_get(sampler.config.true_psd))

        diagnostics = {
            "psd_matrix": vi_psd_np,
            "true_psd": true_psd,
        }
        if psd_quantiles is not None:
            diagnostics["psd_quantiles"] = psd_quantiles
        if coherence_quantiles is not None:
            diagnostics["coherence_quantiles"] = coherence_quantiles
        if vi_result.samples is not None:
            diagnostics["vi_samples"] = {
                name: np.asarray(jax.device_get(value))
                for name, value in vi_result.samples.items()
            }
        return init_values, diagnostics

    return sampler._run_vi_initialisation(
        model=model,
        model_args=(
            sampler.u_re,
            sampler.u_im,
            sampler.nu,
            sampler.all_bases,
            sampler.all_penalties,
            sampler.freq_weights,
            sampler.config.alpha_phi,
            sampler.config.beta_phi,
            sampler.config.alpha_delta,
            sampler.config.beta_delta,
        ),
        guide=guide_spec,
        postprocess=_postprocess,
    )


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
) -> BlockVIArtifacts:
    """Run VI per block for blocked multivariate samplers."""

    logger.info("Running VI initialisation per block...")
    n_channels = sampler.n_channels
    init_strategies: List[Optional[Callable[[Any], Any]]] = [None] * n_channels
    mcmc_keys: List[jax.Array] = [jax.random.PRNGKey(0)] * n_channels

    scaling = float(getattr(sampler.config, "scaling_factor", 1.0) or 1.0)
    channel_stds = getattr(sampler.fft_data, "channel_stds", None)
    if channel_stds is not None:
        channel_stds = np.asarray(channel_stds, dtype=np.float32)
        scale_matrix = np.outer(channel_stds, channel_stds).astype(np.float32)
        factor_matrix = scale_matrix
        scalar_factor = None
    else:
        factor_matrix = None
        scalar_factor = scaling

    def _rescale_psd(arr: np.ndarray) -> np.ndarray:
        if factor_matrix is not None:
            return arr * factor_matrix
        return arr * scalar_factor

    vi_losses_blocks: List[np.ndarray] = []
    vi_guides: List[str] = []
    vi_log_delta_means: List[np.ndarray] = []
    vi_theta_re_mean: Optional[np.ndarray] = None
    vi_theta_im_mean: Optional[np.ndarray] = None

    posterior_draws = (
        sampler.config.vi_posterior_draws
        if getattr(sampler.config, "vi_posterior_draws", 0) > 0
        else 0
    )
    store_draws = sampler.config.init_from_vi and posterior_draws > 0
    vi_samples: Dict[str, np.ndarray] = {}
    log_delta_draws = None
    theta_re_draws = None
    theta_im_draws = None
    draws_missing = False
    draws_recorded = 0
    if store_draws:
        log_delta_draws = np.zeros(
            (posterior_draws, sampler.n_freq, sampler.n_channels),
            dtype=np.float32,
        )
        if sampler.n_theta > 0:
            theta_re_draws = np.zeros(
                (posterior_draws, sampler.n_freq, sampler.n_theta),
                dtype=np.float32,
            )
            theta_im_draws = np.zeros_like(theta_re_draws)

    current_key = rng_key

    for channel_index in range(n_channels):
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
        delta_penalty = sampler.all_penalties[channel_index]

        theta_start = channel_index * (channel_index - 1) // 2
        theta_count = channel_index

        guide_spec = sampler.config.vi_guide or suggest_guide_block(
            delta_basis.shape[1], theta_count, sampler._theta_basis.shape[1]
        )
        progress_bar = (
            sampler.config.vi_progress_bar
            if sampler.config.vi_progress_bar is not None
            else sampler.config.verbose
        )

        try:
            u_re_channel = sampler.u_re[:, channel_index, :]
            u_im_channel = sampler.u_im[:, channel_index, :]
            u_re_prev = sampler.u_re[:, :channel_index, :]
            u_im_prev = sampler.u_im[:, :channel_index, :]

            vi_result = fit_vi(
                model=block_model,
                rng_key=vi_key,
                vi_steps=sampler.config.vi_steps,
                optimizer_lr=sampler.config.vi_lr,
                model_args=(
                    channel_index,
                    u_re_channel,
                    u_im_channel,
                    u_re_prev,
                    u_im_prev,
                    delta_basis,
                    delta_penalty,
                    sampler._theta_basis,
                    sampler._theta_penalty,
                    sampler.config.alpha_phi,
                    sampler.config.beta_phi,
                    sampler.config.alpha_delta,
                    sampler.config.beta_delta,
                    sampler.nu,
                    sampler.freq_weights,
                ),
                guide=guide_spec,
                posterior_draws=sampler.config.vi_posterior_draws,
                progress_bar=progress_bar,
            )

            init_values = {
                name: jnp.asarray(value)
                for name, value in vi_result.means.items()
            }
            init_strategies[channel_index] = init_to_value(values=init_values)

            losses_arr = np.asarray(jax.device_get(vi_result.losses))
            vi_losses_blocks.append(losses_arr)
            vi_guides.append(vi_result.guide_name)

            if vi_result.samples is not None:
                for name, value in vi_result.samples.items():
                    vi_samples[name] = np.asarray(jax.device_get(value))

            weights_delta_name = f"weights_delta_{channel_index}"
            weights_delta = vi_result.means.get(weights_delta_name)
            if weights_delta is not None:
                log_delta_vi = jnp.einsum(
                    "nk,k->n", delta_basis, weights_delta
                )
                vi_log_delta_means.append(
                    np.asarray(jax.device_get(log_delta_vi))
                )

            if sampler.n_theta > 0 and theta_count > 0:
                if vi_theta_re_mean is None:
                    vi_theta_re_mean = np.zeros(
                        (sampler.n_freq, sampler.n_theta), dtype=np.float32
                    )
                    vi_theta_im_mean = np.zeros(
                        (sampler.n_freq, sampler.n_theta), dtype=np.float32
                    )

                theta_re_components: List[np.ndarray] = []
                theta_im_components: List[np.ndarray] = []

                for theta_idx in range(theta_count):
                    prefix = f"{channel_index}_{theta_idx}"
                    weights_theta_re = vi_result.means.get(
                        f"weights_theta_re_{prefix}"
                    )
                    weights_theta_im = vi_result.means.get(
                        f"weights_theta_im_{prefix}"
                    )
                    if weights_theta_re is None or weights_theta_im is None:
                        raise KeyError(f"theta weights {prefix}")

                    theta_re_eval = jnp.einsum(
                        "nk,k->n", sampler._theta_basis, weights_theta_re
                    )
                    theta_im_eval = jnp.einsum(
                        "nk,k->n", sampler._theta_basis, weights_theta_im
                    )
                    theta_re_components.append(
                        np.asarray(jax.device_get(theta_re_eval))
                    )
                    theta_im_components.append(
                        np.asarray(jax.device_get(theta_im_eval))
                    )

                if theta_re_components:
                    theta_re_block = np.stack(theta_re_components, axis=1)
                    theta_im_block = np.stack(theta_im_components, axis=1)
                else:
                    theta_re_block = np.zeros((sampler.n_freq, theta_count))
                    theta_im_block = np.zeros((sampler.n_freq, theta_count))

                theta_slice = slice(theta_start, theta_start + theta_count)
                vi_theta_re_mean[:, theta_slice] = theta_re_block
                vi_theta_im_mean[:, theta_slice] = theta_im_block

            if store_draws and vi_result.samples is not None:
                weights_delta_samples = vi_result.samples.get(
                    weights_delta_name
                )
                if weights_delta_samples is not None:
                    weights_delta_samples = jnp.asarray(weights_delta_samples)
                    draw_count = min(
                        posterior_draws, weights_delta_samples.shape[0]
                    )
                    draws_recorded = (
                        draw_count
                        if draws_recorded == 0
                        else min(draws_recorded, draw_count)
                    )
                    delta_basis_jnp = jnp.asarray(
                        delta_basis, dtype=weights_delta_samples.dtype
                    )
                    log_delta_samples = (
                        weights_delta_samples[:draw_count] @ delta_basis_jnp.T
                    )
                    log_delta_draws[:draw_count, :, channel_index] = (
                        np.asarray(jax.device_get(log_delta_samples))
                    )
                else:
                    draws_missing = True

                if (
                    sampler.n_theta > 0
                    and theta_count > 0
                    and theta_re_draws is not None
                ):
                    theta_basis_jnp = jnp.asarray(
                        sampler._theta_basis, dtype=jnp.float32
                    )
                    for theta_idx in range(theta_count):
                        prefix = f"{channel_index}_{theta_idx}"
                        weights_theta_re_samples = vi_result.samples.get(
                            f"weights_theta_re_{prefix}"
                        )
                        weights_theta_im_samples = vi_result.samples.get(
                            f"weights_theta_im_{prefix}"
                        )
                        if (
                            weights_theta_re_samples is None
                            or weights_theta_im_samples is None
                        ):
                            draws_missing = True
                            continue

                        weights_theta_re_samples = jnp.asarray(
                            weights_theta_re_samples
                        )
                        weights_theta_im_samples = jnp.asarray(
                            weights_theta_im_samples
                        )
                        theta_basis_jnp = jnp.asarray(
                            sampler._theta_basis,
                            dtype=weights_theta_re_samples.dtype,
                        )
                        draw_count = min(
                            posterior_draws, weights_theta_re_samples.shape[0]
                        )
                        draws_recorded = (
                            draw_count
                            if draws_recorded == 0
                            else min(draws_recorded, draw_count)
                        )
                        theta_re_eval = (
                            weights_theta_re_samples[:draw_count]
                            @ theta_basis_jnp.T
                        )
                        theta_im_eval = (
                            weights_theta_im_samples[:draw_count]
                            @ theta_basis_jnp.T
                        )
                        column = theta_start + theta_idx
                        theta_re_draws[:draw_count, :, column] = np.asarray(
                            jax.device_get(theta_re_eval)
                        )
                        theta_im_draws[:draw_count, :, column] = np.asarray(
                            jax.device_get(theta_im_eval)
                        )

        except Exception as exc:  # pragma: no cover - defensive fallback
            if sampler.config.verbose:
                logger.warning(
                    f"VI block initialisation failed [channel {channel_index}]: {exc}"
                )
            if store_draws:
                draws_missing = True

    diagnostics = None

    if sampler.config.init_from_vi and vi_log_delta_means:
        if len(vi_log_delta_means) == n_channels:
            log_delta_vi_np = np.stack(vi_log_delta_means, axis=1)
        else:
            log_delta_vi_np = None

        if log_delta_vi_np is not None:
            if sampler.n_theta > 0:
                assert (
                    vi_theta_re_mean is not None
                    and vi_theta_im_mean is not None
                )
                theta_re_vi_np = vi_theta_re_mean
                theta_im_vi_np = vi_theta_im_mean
            else:
                theta_re_vi_np = np.zeros(
                    (sampler.n_freq, 0), dtype=np.float32
                )
                theta_im_vi_np = np.zeros(
                    (sampler.n_freq, 0), dtype=np.float32
                )

            vi_psd = sampler.spline_model.reconstruct_psd_matrix(
                jnp.asarray(log_delta_vi_np)[None, ...],
                jnp.asarray(theta_re_vi_np)[None, ...],
                jnp.asarray(theta_im_vi_np)[None, ...],
                n_samples_max=1,
            )[0]
            vi_psd_np = _rescale_psd(np.asarray(vi_psd))
        else:
            vi_psd_np = None

        psd_quantiles = None
        coherence_quantiles = None
        if store_draws and not draws_missing and log_delta_draws is not None:
            desired_draws = draws_recorded or posterior_draws
            draws_available = min(desired_draws, log_delta_draws.shape[0])
            if draws_available == 0:
                logger.debug(
                    "No VI PSD draws recorded; skipping PSD reconstruction."
                )
            else:
                draws_used = _cap_psd_draws(sampler.config, draws_available)
                if draws_used < draws_available:
                    logger.debug(
                        f"Capping VI PSD reconstruction to {draws_used} draws "
                        f"(limit={getattr(sampler.config, 'vi_psd_max_draws', 0)})."
                    )
                log_delta_samples = jnp.asarray(
                    log_delta_draws[:draws_used], dtype=jnp.float32
                )
                if sampler.n_theta > 0 and theta_re_draws is not None:
                    theta_re_samples = jnp.asarray(
                        theta_re_draws[:draws_used], dtype=jnp.float32
                    )
                    theta_im_samples = jnp.asarray(
                        theta_im_draws[:draws_used], dtype=jnp.float32
                    )
                else:
                    theta_re_samples = jnp.zeros(
                        (draws_used, sampler.n_freq, 0), dtype=jnp.float32
                    )
                    theta_im_samples = jnp.zeros_like(theta_re_samples)

                logger.debug("Reconstructing PSD samples from VI draws...")
                psd_real_q, psd_imag_q, coh_percentiles = (
                    sampler.spline_model.compute_psd_quantiles(
                        log_delta_samples,
                        theta_re_samples,
                        theta_im_samples,
                        percentiles=[5.0, 50.0, 95.0],
                        n_samples_max=draws_used,
                        compute_coherence=sampler.n_channels > 1,
                    )
                )
                psd_real_q = _rescale_psd(psd_real_q)
                psd_imag_q = _rescale_psd(psd_imag_q)

                psd_quantiles = {
                    "real": {
                        "q05": psd_real_q[0],
                        "q50": psd_real_q[1],
                        "q95": psd_real_q[2],
                    },
                    "imag": {
                        "q05": psd_imag_q[0],
                        "q50": psd_imag_q[1],
                        "q95": psd_imag_q[2],
                    },
                }
                vi_psd_np = psd_quantiles["real"]["q50"]

                if coh_percentiles is not None:
                    coh_percentiles *= 1.0
                    coherence_quantiles = {
                        "q05": coh_percentiles[0],
                        "q50": coh_percentiles[1],
                        "q95": coh_percentiles[2],
                    }

        valid_losses = [arr for arr in vi_losses_blocks if arr.size]
        losses_mean = None
        losses_stack = None
        if valid_losses:
            min_len = min(arr.shape[0] for arr in valid_losses)
            if min_len > 0:
                losses_stack = np.stack(
                    [arr[-min_len:] for arr in valid_losses], axis=0
                )
                losses_mean = losses_stack.mean(axis=0)

        true_psd = None
        if sampler.config.true_psd is not None:
            true_psd = np.asarray(jax.device_get(sampler.config.true_psd))

        guide_label = ",".join(sorted(set(vi_guides))) if vi_guides else "vi"

        diagnostics = {
            "losses": (
                losses_mean if losses_mean is not None else np.asarray([])
            ),
            "losses_per_block": losses_stack,
            "guide": guide_label,
            "psd_matrix": vi_psd_np,
            "true_psd": true_psd,
        }
        if psd_quantiles is not None:
            diagnostics["psd_quantiles"] = psd_quantiles
        if coherence_quantiles is not None:
            diagnostics["coherence_quantiles"] = coherence_quantiles

    if vi_samples:
        diagnostics = diagnostics or {}
        diagnostics["vi_samples"] = vi_samples

    return BlockVIArtifacts(
        init_strategies=init_strategies,
        mcmc_keys=mcmc_keys,
        rng_key=current_key,
        diagnostics=diagnostics,
    )
