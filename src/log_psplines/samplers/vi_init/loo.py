"""ArviZ approximate-posterior LOO helpers for VI diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from arviz_stats import loo_approximate_posterior
from numpyro.infer.util import log_density

from ...logger import logger


def _trim_log_likelihood_dataset(
    dataset: xr.Dataset, n_draws: int
) -> xr.Dataset:
    if "draw" not in dataset.dims:
        return dataset
    return dataset.isel(draw=slice(0, n_draws))


def _make_log_likelihood_tree(dataset: xr.Dataset) -> xr.DataTree:
    tree = xr.DataTree()
    tree["log_likelihood"] = xr.DataTree(dataset=dataset)
    return tree


def _reduce_log_q(log_q: jax.Array) -> np.ndarray:
    arr = np.asarray(jax.device_get(log_q), dtype=np.float64)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    return arr.reshape(arr.shape[0], -1).sum(axis=1)


def _compute_log_p(
    *,
    model,
    model_args: Tuple[Any, ...],
    model_kwargs: Dict[str, Any],
    vi_samples: Dict[str, jnp.ndarray],
) -> np.ndarray:
    def _log_posterior(sample):
        log_prob, _ = log_density(model, model_args, model_kwargs, sample)
        return log_prob

    return np.asarray(
        jax.device_get(jax.vmap(_log_posterior)(vi_samples)), dtype=np.float64
    )


def compute_vi_loo_approximate_posterior(
    *,
    log_likelihood: Optional[xr.Dataset],
    model,
    model_args: Tuple[Any, ...],
    model_kwargs: Dict[str, Any],
    guide,
    guide_params: Dict[str, Any],
    vi_samples: Optional[Dict[str, jnp.ndarray]],
    latent_samples: Optional[jnp.ndarray],
    var_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return ArviZ approximate-posterior LOO diagnostics for VI draws."""

    if log_likelihood is None or not vi_samples or latent_samples is None:
        return None

    try:
        sample_tree = {
            name: jnp.asarray(array) for name, array in vi_samples.items()
        }
        n_draws = min(
            int(latent_samples.shape[0]),
            *[
                int(array.shape[0])
                for array in sample_tree.values()
                if getattr(array, "ndim", 0) > 0
            ],
        )
        if n_draws <= 0:
            return None

        trimmed_samples = {k: v[:n_draws] for k, v in sample_tree.items()}
        trimmed_latents = jnp.asarray(latent_samples)[:n_draws]
        trimmed_ll = _trim_log_likelihood_dataset(log_likelihood, n_draws)
        if var_name is None:
            try:
                var_name = next(iter(trimmed_ll.data_vars))
            except StopIteration:
                return None

        log_p = _compute_log_p(
            model=model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            vi_samples=trimmed_samples,
        )
        posterior = guide.get_posterior(guide_params)
        log_q = _reduce_log_q(posterior.log_prob(trimmed_latents))
        elpd = loo_approximate_posterior(
            data=_make_log_likelihood_tree(trimmed_ll),
            log_p=log_p,
            log_q=log_q,
            var_name=var_name,
            pointwise=True,
        )

        pareto_k = (
            None
            if getattr(elpd, "pareto_k", None) is None
            else np.asarray(elpd.pareto_k.values, dtype=np.float64)
        )
        elpd_i = (
            None
            if getattr(elpd, "elpd_i", None) is None
            else np.asarray(elpd.elpd_i.values, dtype=np.float64)
        )
        pareto_k_max = (
            np.nan
            if pareto_k is None or pareto_k.size == 0
            else float(np.nanmax(pareto_k))
        )
        good_k = float(getattr(elpd, "good_k", np.nan))
        warning = bool(getattr(elpd, "warning", False))
        status = (
            "unknown"
            if not np.isfinite(pareto_k_max)
            else "warn" if warning else "ok"
        )

        return {
            "kind": str(elpd.kind),
            "elpd": float(elpd.elpd),
            "se": float(elpd.se),
            "p": float(elpd.p),
            "n_samples": int(elpd.n_samples),
            "n_data_points": int(elpd.n_data_points),
            "scale": str(elpd.scale),
            "warning": warning,
            "good_k": good_k,
            "elpd_i": elpd_i,
            "pareto_k": pareto_k,
            "approx_posterior": bool(elpd.approx_posterior),
            "pareto_k_max": pareto_k_max,
            "loo_warning": warning,
            "psis_khat_max": pareto_k_max,
            "psis_khat_status": status,
            "psis_khat_threshold": good_k,
            "psis_flag_warn": warning,
            "psis_flag_critical": warning,
        }
    except Exception as exc:  # pragma: no cover - diagnostics are best-effort
        logger.debug(f"Could not compute ArviZ VI LOO diagnostics: {exc}")
        return None


__all__ = ["compute_vi_loo_approximate_posterior"]
