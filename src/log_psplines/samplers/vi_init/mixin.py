"""Shared helpers for seeding samplers with variational inference results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from numpyro.infer.util import init_to_value

from .core import VIResult, fit_vi


@dataclass
class VIInitialisationArtifacts:
    """Summary of a VI run used for sampler initialisation."""

    init_strategy: Optional[Callable[[Any], Any]]
    rng_key: jax.Array
    diagnostics: Optional[Dict[str, Any]]


class VIInitialisationMixin:
    """Mixin providing a reusable ``_run_vi_initialisation`` helper."""

    config: Any  # subclasses expose sampler-specific configs
    rng_key: jax.Array

    def _run_vi_initialisation(
        self,
        *,
        model: Callable[..., Any],
        model_args: Tuple[Any, ...],
        guide: Optional[str],
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
        guide:
            Explicit autoguide specifier, or ``None`` to let ``fit_vi`` choose.
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

        try:
            vi_result = fit_vi(
                model=model,
                rng_key=key_vi,
                vi_steps=self.config.vi_steps,
                optimizer_lr=self.config.vi_lr,
                model_args=model_args,
                guide=guide,
                posterior_draws=self.config.vi_posterior_draws,
                progress_bar=progress_bar,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if getattr(self.config, "verbose", False):
                print(
                    f"VI initialisation failed ({exc}) - using default init."
                )
            return VIInitialisationArtifacts(None, key_run, None)

        init_values, diagnostics = postprocess(vi_result)
        init_strategy = init_to_value(values=init_values)

        diagnostics = diagnostics or {}
        diagnostics.setdefault("guide", vi_result.guide_name)
        diagnostics.setdefault("losses", jnp.asarray(vi_result.losses))

        return VIInitialisationArtifacts(init_strategy, key_run, diagnostics)
