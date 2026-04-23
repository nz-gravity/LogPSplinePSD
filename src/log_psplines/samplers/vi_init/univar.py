"""Thin univariate VI adapters, including coarse-to-fine warm-start flow."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ...diagnostics.vi_results import _extract_psd_q50, _get_scaling_factor
from ...logger import logger
from .bridge import transfer_univar_weights
from .common import (
    _strip_coarse_vi_plot_arrays,
    _validate_positive_finite_psd,
)
from .mixin import VIInitialisationArtifacts
from .plan import (
    VIWarmStartPlan,
    add_coarse_plot_diagnostics,
    build_coarse_sampler_from_plan,
    coarse_vi_metadata,
    mark_coarse_vi,
)
from .runner import compute_vi_artifacts_univar, default_init_values_univar


def _fallback_to_fine_vi(
    sampler, metadata: dict[str, Any], model: Callable[..., Any]
):
    fine_artifacts = compute_vi_artifacts_univar(sampler, model=model)
    diagnostics = mark_coarse_vi(
        fine_artifacts.diagnostics, metadata, attempted=True, success=False
    )
    return VIInitialisationArtifacts(
        fine_artifacts.init_strategy,
        fine_artifacts.rng_key,
        _strip_coarse_vi_plot_arrays(diagnostics),
        means=fine_artifacts.means,
        posterior_draws=fine_artifacts.posterior_draws,
    )


def compute_coarse_vi_artifacts_univar(
    sampler,
    *,
    warm_start_plan: VIWarmStartPlan,
    model: Callable[..., Any],
) -> VIInitialisationArtifacts:
    """VI-coarse -> transfer -> VI-fine for univariate models."""
    metadata = coarse_vi_metadata(warm_start_plan)
    coarse_sampler = build_coarse_sampler_from_plan(sampler, warm_start_plan)
    coarse_artifacts = compute_vi_artifacts_univar(coarse_sampler, model=model)
    coarse_diag = coarse_artifacts.diagnostics or {}

    coarse_psd = _extract_psd_q50(coarse_diag)
    coarse_label = "Coarse-Grid VI Posterior Median"
    if coarse_psd is None:
        coarse_psd = coarse_diag.get("psd")
        coarse_label = "Coarse-Grid VI Mean"

    if coarse_psd is None or not _validate_positive_finite_psd(coarse_psd):
        logger.warning(
            "Coarse-grid VI did not produce a valid PSD; "
            "falling back to standard fine-grid VI."
        )
        return _fallback_to_fine_vi(sampler, metadata, model)

    try:
        coarse_freq = np.asarray(coarse_sampler.periodogram.freqs, dtype=float)
        fine_freq = np.asarray(sampler.periodogram.freqs, dtype=float)

        coarse_means = coarse_artifacts.means or {}
        if "weights" not in coarse_means:
            raise ValueError("Coarse VI did not produce mean weights")

        fine_weights = transfer_univar_weights(
            coarse_weights=np.asarray(jax.device_get(coarse_means["weights"])),
            coarse_spline_model=coarse_sampler.spline_model,
            coarse_freq=coarse_freq,
            coarse_scaling=_get_scaling_factor(
                coarse_sampler.periodogram, coarse_sampler.config
            ),
            fine_spline_model=sampler.spline_model,
            fine_freq=fine_freq,
            fine_scaling=_get_scaling_factor(
                sampler.periodogram, sampler.config
            ),
        )
        transferred_init = default_init_values_univar(
            sampler.spline_model,
            alpha_phi=sampler.config.alpha_phi,
            beta_phi=sampler.config.beta_phi,
            alpha_delta=sampler.config.alpha_delta,
            beta_delta=sampler.config.beta_delta,
        )
        transferred_init["weights"] = jnp.asarray(fine_weights)

        coarse_only = bool(getattr(sampler.config, "vi_coarse_only", False))

        if coarse_only:
            diagnostics = add_coarse_plot_diagnostics(
                mark_coarse_vi(
                    coarse_diag,
                    metadata,
                    attempted=True,
                    success=True,
                ),
                coarse_freq=coarse_freq,
                coarse_psd=coarse_psd,
                coarse_label=coarse_label,
                coarse_losses=coarse_diag.get("losses"),
            )
            return VIInitialisationArtifacts(
                init_strategy=None,
                rng_key=coarse_artifacts.rng_key,
                diagnostics=diagnostics,
                means=coarse_artifacts.means,
                posterior_draws=coarse_artifacts.posterior_draws,
            )

        fine_artifacts = compute_vi_artifacts_univar(
            sampler,
            model=model,
            init_values=transferred_init,
        )

        diagnostics = add_coarse_plot_diagnostics(
            mark_coarse_vi(
                fine_artifacts.diagnostics,
                metadata,
                attempted=True,
                success=True,
            ),
            coarse_freq=coarse_freq,
            coarse_psd=coarse_psd,
            coarse_label=coarse_label,
            coarse_losses=coarse_diag.get("losses"),
        )

        return VIInitialisationArtifacts(
            fine_artifacts.init_strategy,
            fine_artifacts.rng_key,
            diagnostics,
            means=fine_artifacts.means,
            posterior_draws=fine_artifacts.posterior_draws,
        )
    except Exception as exc:
        logger.warning(
            f"Coarse-to-fine transfer failed ({exc}); "
            "falling back to standard fine-grid VI."
        )
        return _fallback_to_fine_vi(sampler, metadata, model)


__all__ = [
    "compute_coarse_vi_artifacts_univar",
    "compute_vi_artifacts_univar",
]
