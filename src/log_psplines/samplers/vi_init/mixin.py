"""Shared helpers for seeding samplers with variational inference results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import init_to_value, log_density

from ...diagnostics.vi_psis import (
    _compute_correlation_diagnostics,
    _compute_psis_khat,
    _compute_psis_moment_checks,
    _emit_moment_warnings,
    _interpret_khat,
)
from ...logger import logger
from .core import VIResult, fit_vi


@dataclass
class VIInitialisationArtifacts:
    """Summary of a VI run used for sampler initialisation."""

    init_strategy: Optional[Callable[[Any], Any]]
    rng_key: jax.Array
    diagnostics: Optional[Dict[str, Any]]
    means: Optional[Dict[str, jnp.ndarray]] = None
    posterior_draws: Optional[Dict[str, jnp.ndarray]] = None


class VIInitialisationMixin:
    """Mixin providing a reusable ``_run_vi_initialisation`` helper."""

    config: Any  # subclasses expose sampler-specific configs
    rng_key: jax.Array

    def _run_vi_initialisation(
        self,
        *,
        model: Callable[..., Any],
        model_args: Tuple[Any, ...],
        model_kwargs: Optional[Dict[str, Any]] = None,
        guide: Optional[str],
        init_values: Optional[Dict[str, Any]] = None,
        postprocess: Callable[
            [VIResult], Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]
        ],
    ) -> VIInitialisationArtifacts:
        """Run VI and return an init strategy plus diagnostics.

        Parameters
        ----------
        model:
            NumPyro model to pass to :func:`fit_vi`.
        model_args:
            Positional arguments to forward to the model.
        model_kwargs:
            Keyword arguments forwarded to the model (used for both VI and
            diagnostics).
        guide:
            Explicit autoguide specifier, or ``None`` to let ``fit_vi`` choose.
        init_values:
            Optional initial latent values passed to the autoguide to start VI
            near the same mode used for NUTS initialisation.
        postprocess:
            Callback mapping the :class:`VIResult` to ``(init_values, diagnostics)``.
            ``init_values`` must be a pytree compatible with ``init_to_value``.

        Returns
        -------
        VIInitialisationArtifacts
            Bundle containing the ``init_strategy`` (or ``None`` on failure), the
            updated RNG key, and optional diagnostics for plotting/logging.
        """

        if not getattr(self.config, "init_from_vi", False):
            return VIInitialisationArtifacts(None, self.rng_key, None)

        key_vi, key_run = jax.random.split(self.rng_key)
        progress_bar = (
            getattr(self.config, "vi_progress_bar", None)
            if getattr(self.config, "vi_progress_bar", None) is not None
            else getattr(self.config, "verbose", False)
        )
        model_kwargs = model_kwargs or {}

        try:
            vi_result = fit_vi(
                model=model,
                rng_key=key_vi,
                vi_steps=self.config.vi_steps,
                optimizer_lr=self.config.vi_lr,
                model_args=model_args,
                model_kwargs=model_kwargs,
                guide=guide,
                posterior_draws=self.config.vi_posterior_draws,
                progress_bar=progress_bar,
                init_values=init_values,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if getattr(self.config, "verbose", False):
                logger.warning(
                    f"VI initialisation failed ({exc}) - using default init."
                )
            return VIInitialisationArtifacts(None, key_run, None)

        init_values, diagnostics = postprocess(vi_result)
        init_strategy = init_to_value(values=init_values)

        diagnostics = diagnostics or {}
        diagnostics.setdefault("guide", vi_result.guide_name)
        diagnostics.setdefault("losses", jnp.asarray(vi_result.losses))

        # Add simple guide scale diagnostics for quick inspection
        if vi_result.scales:
            scales_np = {
                name: np.asarray(jax.device_get(value))
                for name, value in vi_result.scales.items()
            }
            summary = sorted(
                (
                    (
                        name,
                        float(np.mean(np.abs(val))),
                        float(np.max(np.abs(val))),
                    )
                    for name, val in scales_np.items()
                ),
                key=lambda x: x[2],
                reverse=True,
            )
            diagnostics["guide_scale_summary"] = summary[:5]

        # Track ELBO slope over the recent window
        losses_np = np.asarray(jax.device_get(vi_result.losses))
        if losses_np.size >= 2:
            window = min(200, losses_np.size)
            start = losses_np[-window]
            end = losses_np[-1]
            slope = (end - start) / max(1, window - 1)
            diagnostics["elbo_slope_recent"] = float(slope)
            if getattr(self.config, "verbose", False) and abs(slope) > 1e-3:
                logger.info(
                    f"VI ELBO trend over last {window} steps: Δ={end - start:.3f}, slope/step={slope:.4f}"
                )

        psis_diag = _compute_psis_khat(
            model=model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            guide=vi_result.guide,
            guide_params=vi_result.params,
            vi_samples=vi_result.samples,
            latent_samples=vi_result.latent_samples,
        )
        if psis_diag is not None:
            diagnostics.update(psis_diag)
            status, status_msg = _interpret_khat(psis_diag["psis_khat_max"])
            diagnostics["psis_khat_status"] = status
            diagnostics["psis_status_message"] = status_msg
            diagnostics["psis_khat_threshold"] = 0.7
            diagnostics["psis_flag_warn"] = status in ("warn", "fail")
            diagnostics["psis_flag_critical"] = status == "fail"
            should_log = getattr(self.config, "verbose", False) or status in (
                "warn",
                "fail",
            )
            if should_log:
                logger.info(
                    f"VI PSIS k-hat max = {psis_diag['psis_khat_max']:.3f} ({status_msg})"
                )
            if status == "fail":
                logger.warning(
                    "VI PSIS diagnostic indicates poor posterior fit (k-hat > 0.7). "
                    "Consider revisiting parametrization or VI configuration."
                )
            moment_summary = psis_diag.get("psis_moment_summary")
            if moment_summary:
                _emit_moment_warnings(moment_summary)

        return VIInitialisationArtifacts(
            init_strategy,
            key_run,
            diagnostics,
            means=vi_result.means,
            posterior_draws=vi_result.samples,
        )
