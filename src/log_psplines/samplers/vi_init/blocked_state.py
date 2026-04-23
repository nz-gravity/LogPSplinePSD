"""Blocked VI state accumulation and result assembly."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from ...diagnostics.psd_compare import compute_multivar_riae_diagnostics
from ...diagnostics.vi_psis import _compute_psis_khat, _interpret_khat
from ...diagnostics.vi_results import (
    _extract_true_psd,
    _reconstruct_psd_quantiles_from_draws,
)
from ...logger import logger
from .common import _to_np, _to_np_dict


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
        "psis_khat_values": [],
        "psis_hyper_entries": [],
        "psis_weight_blocks": [],
        "psis_corr_summary": {},
        "psis_thresholds": None,
    }


def _aggregate_psis_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    psis_khat_values: List[float],
    psis_hyper_entries: List[Dict[str, Any]],
    psis_weight_blocks: List[Dict[str, Any]],
    psis_corr_summary: Dict[str, Any],
    psis_thresholds: Optional[Dict[str, float]],
    verbose: bool,
) -> None:
    if psis_khat_values:
        khat_array = np.asarray(psis_khat_values, dtype=float)
        khat_max = float(np.nanmax(khat_array))
        status, status_msg = _interpret_khat(khat_max)
        diagnostics.update(
            psis_khat_per_block=khat_array,
            psis_khat_max=khat_max,
            psis_khat_status=status,
            psis_status_message=status_msg,
            psis_khat_threshold=0.7,
            psis_flag_warn=status in ("warn", "fail"),
            psis_flag_critical=status == "fail",
        )
        if verbose or status in ("warn", "fail"):
            logger.info(
                f"VI PSIS k-hat max (blocked) = {khat_max:.3f} ({status_msg})"
            )
        if status == "fail":
            logger.warning(
                "VI PSIS diagnostic (blocked) indicates poor posterior fit. "
                "Consider adjusting the guide or VI settings."
            )
    if psis_hyper_entries or psis_weight_blocks or psis_corr_summary:
        summary: Dict[str, Any] = {}
        if psis_hyper_entries:
            summary["hyperparameters"] = psis_hyper_entries
        if psis_weight_blocks:
            summary["weights_by_block"] = psis_weight_blocks
            fns: Dict[str, Callable[[Any], Any]] = {
                "var_ratio_min": np.nanmin,
                "var_ratio_median": np.nanmedian,
                "var_ratio_max": np.nanmax,
                "frac_outside": np.nanmax,
                "bias_median_abs": np.nanmedian,
                "bias_max_abs": np.nanmax,
                "bias_abs_median": np.nanmedian,
                "bias_abs_max": np.nanmax,
            }
            summary["weights"] = {
                **{
                    k: float(
                        fn([e.get(k, np.nan) for e in psis_weight_blocks])
                    )
                    for k, fn in fns.items()
                },
                "n_weights": int(
                    np.sum([e.get("n_weights", 0) for e in psis_weight_blocks])
                ),
            }
        if psis_thresholds:
            summary["thresholds"] = psis_thresholds
        if summary:
            diagnostics["psis_moment_summary"] = summary
        if psis_corr_summary:
            diagnostics["psis_correlation_summary"] = psis_corr_summary


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

    psis_diag = _compute_psis_khat(
        model=block_model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        guide=vi_result.guide,
        guide_params=vi_result.params,
        vi_samples=vi_result.samples,
        latent_samples=vi_result.latent_samples,
    )
    if psis_diag is not None:
        accum["psis_khat_values"].append(psis_diag["psis_khat_max"])
        moment_summary = psis_diag.get("psis_moment_summary") or {}
        accum["psis_thresholds"] = accum[
            "psis_thresholds"
        ] or moment_summary.get("thresholds")
        for entry in moment_summary.get("hyperparameters") or []:
            accum["psis_hyper_entries"].append(
                {
                    **entry,
                    "param": f"block_{channel_index}:{entry.get('param', 'param')}",
                }
            )
        if moment_summary.get("weights"):
            accum["psis_weight_blocks"].append(
                {"block": channel_index, **moment_summary["weights"]}
            )
        for label, stats in (
            psis_diag.get("psis_correlation_summary") or {}
        ).items():
            if stats:
                accum["psis_corr_summary"][
                    f"{label}_block_{channel_index}"
                ] = stats

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
        accum["psis_khat_values"]
        or accum["psis_hyper_entries"]
        or accum["psis_weight_blocks"]
        or accum["psis_corr_summary"]
    ):
        _aggregate_psis_diagnostics(
            diagnostics,
            psis_khat_values=accum["psis_khat_values"],
            psis_hyper_entries=accum["psis_hyper_entries"],
            psis_weight_blocks=accum["psis_weight_blocks"],
            psis_corr_summary=accum["psis_corr_summary"],
            psis_thresholds=accum["psis_thresholds"],
            verbose=sampler.config.verbose,
        )
    return diagnostics


def _block_site_names(channel_index: int) -> list[str]:
    return [f"weights_delta_{channel_index}"] + [
        f"weights_{prefix}_{channel_index}_{theta_idx}"
        for theta_idx in range(channel_index)
        for prefix in ("theta_re", "theta_im")
    ]
