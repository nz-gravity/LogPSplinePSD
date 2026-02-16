"""PSIS-based diagnostics shared across VI and sampling entry points."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

from ..logger import logger

MIN_PSIS_DRAWS = 200


def _interpret_khat(khat_max: float) -> Tuple[str, str]:
    """Map PSIS k-hat to a compact status label and message."""
    if not np.isfinite(khat_max):
        return "unknown", "PSIS not available"
    if khat_max < 0.5:
        return "ok", "VI reliable"
    if khat_max <= 0.7:
        return "warn", "usable but imperfect"
    return "fail", "not trustworthy"


def _normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    """Convert log-weights to normalized importance weights."""
    log_w = np.asarray(log_w, dtype=np.float64)
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(w)
    return w / total


def _compute_psis_moment_checks(
    samples: Dict[str, jnp.ndarray],
    weights: np.ndarray,
    *,
    bias_eps: float = 1e-10,
    bias_threshold: float = 0.05,
    var_low: float = 0.7,
    var_high: float = 1.3,
) -> Dict[str, Any]:
    """Compute PSIS-corrected means/variances and compare to raw VI moments."""

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        return {}
    w = w / max(np.sum(w), 1e-12)
    n_draws = w.size

    weight_var_ratios: list[np.ndarray] = []
    weight_bias_abs: list[np.ndarray] = []
    weight_bias_abs_raw: list[np.ndarray] = []
    hyper_entries: list[Dict[str, Any]] = []

    for name, array in samples.items():
        arr = np.asarray(array)
        if arr.shape[0] != n_draws:
            continue
        flat = arr.reshape(n_draws, -1)
        vi_mean = flat.mean(axis=0)
        psis_mean = w @ flat
        bias_raw = psis_mean - vi_mean
        bias = bias_raw / (np.abs(vi_mean) + bias_eps)

        vi_var = flat.var(axis=0)
        centered = flat - psis_mean
        psis_var = (w[:, None] * centered * centered).sum(axis=0)
        var_ratio = psis_var / (vi_var + bias_eps)

        if "weights" in name:
            weight_var_ratios.append(var_ratio)
            weight_bias_abs.append(np.abs(bias))
            weight_bias_abs_raw.append(np.abs(bias_raw))
            continue

        if name.startswith(("phi", "delta")):
            vi_std = np.sqrt(vi_var + bias_eps)
            psis_std = np.sqrt(psis_var)
            entries = []
            for idx in range(var_ratio.size):
                label = name if var_ratio.size == 1 else f"{name}[{idx}]"
                entries.append(
                    dict(
                        param=label,
                        vi_mean=float(vi_mean.flat[idx]),
                        psis_mean=float(psis_mean.flat[idx]),
                        bias_pct=float(bias.flat[idx] * 100.0),
                        bias_abs=float(bias_raw.flat[idx]),
                        vi_std=float(vi_std.flat[idx]),
                        psis_std=float(psis_std.flat[idx]),
                        var_ratio=float(var_ratio.flat[idx]),
                    )
                )
            hyper_entries.extend(entries)

    weight_stats = None
    if weight_var_ratios:
        vr = np.concatenate(weight_var_ratios)
        bias_abs = (
            np.concatenate(weight_bias_abs) if weight_bias_abs else vr * 0
        )
        bias_abs_raw = (
            np.concatenate(weight_bias_abs_raw)
            if weight_bias_abs_raw
            else vr * 0
        )
        weight_stats = {
            "var_ratio_min": float(np.nanmin(vr)),
            "var_ratio_median": float(np.nanmedian(vr)),
            "var_ratio_max": float(np.nanmax(vr)),
            "frac_outside": float(np.mean((vr < var_low) | (vr > var_high))),
            "bias_median_abs": float(np.nanmedian(bias_abs)),
            "bias_max_abs": float(np.nanmax(bias_abs)),
            "bias_abs_median": float(np.nanmedian(bias_abs_raw)),
            "bias_abs_max": float(np.nanmax(bias_abs_raw)),
            "n_weights": int(vr.size),
        }

    return {
        "weights": weight_stats,
        "hyperparameters": hyper_entries,
        "thresholds": {
            "bias_threshold": bias_threshold,
            "var_low": var_low,
            "var_high": var_high,
        },
    }


def _emit_moment_warnings(summary: Dict[str, Any]) -> None:
    """Log warnings for PSIS moment discrepancies."""

    if not summary:
        return
    thresholds = summary.get("thresholds", {})
    bias_thr = thresholds.get("bias_threshold", 0.05)
    var_low = thresholds.get("var_low", 0.7)
    var_high = thresholds.get("var_high", 1.3)

    issues: list[str] = []
    weight_stats = summary.get("weights") or {}
    if weight_stats and weight_stats.get("frac_outside", 0.0) > 0:
        issues.append(
            f"{weight_stats['frac_outside']*100:.1f}% of weights var_ratio outside [{var_low},{var_high}]"
        )
    if weight_stats and weight_stats.get("bias_max_abs", 0.0) > bias_thr:
        issues.append(
            f"weights bias up to {weight_stats['bias_max_abs']*100:.1f}% (> {bias_thr*100:.0f}%)"
        )
    hyper_params = summary.get("hyperparameters") or []
    for entry in hyper_params:
        bias_pct = abs(entry.get("bias_pct", 0.0)) / 100.0
        var_ratio = entry.get("var_ratio", 1.0)
        name = entry.get("param", "param")
        if bias_pct > bias_thr:
            issues.append(f"{name} bias {bias_pct*100:.1f}%")
        if var_ratio < var_low or var_ratio > var_high:
            issues.append(f"{name} var_ratio={var_ratio:.2f}")

    if issues:
        logger.warning(
            "PSIS moment checks flagged potential bias/dispersion issues: "
            + "; ".join(issues)
        )


def _corr_metrics(matrix: np.ndarray) -> Optional[Dict[str, float]]:
    """Return basic correlation statistics for a correlation matrix."""
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return None
    upper = np.abs(matrix[np.triu_indices(matrix.shape[0], k=1)])
    if upper.size == 0:
        return None
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return None
    return {
        "max_abs": float(np.max(upper)),
        "median_abs": float(np.median(upper)),
        "mean_abs": float(np.mean(upper)),
        "n_params": int(matrix.shape[0]),
    }


def _compute_correlation_diagnostics(
    samples: Dict[str, jnp.ndarray],
    reference_corr: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute guide-structure correlation diagnostics for VI samples."""

    def _build_matrix(names_prefix: Tuple[str, ...]):
        cols = []
        for name, arr in samples.items():
            if not any(name.startswith(pfx) for pfx in names_prefix):
                continue
            a = np.asarray(arr)
            if a.ndim <= 1:
                col = a.reshape(a.shape[0], 1)
            else:
                col = a.reshape(a.shape[0], -1)
            cols.append(col)
        if not cols:
            return None
        mat = np.concatenate(cols, axis=1)
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            return None
        # Drop near-constant columns to avoid NaNs in corr
        std = mat.std(axis=0)
        mask = std > 0
        if not np.any(mask) or mask.sum() < 2:
            return None
        return mat[:, mask]

    results: Dict[str, Any] = {}

    weights_mat = _build_matrix(("weights",))
    if weights_mat is not None:
        corr = np.corrcoef(weights_mat, rowvar=False)
        metrics = _corr_metrics(corr)
        if metrics:
            results["weights"] = metrics
            if reference_corr and reference_corr.get("weights") is not None:
                ref = reference_corr["weights"]
                if ref.shape == corr.shape:
                    diff = np.abs(corr - ref)
                    results["weights"]["mean_corr_diff"] = float(
                        np.mean(diff[np.triu_indices(diff.shape[0], k=1)])
                    )

    hyper_mat = _build_matrix(("phi", "delta"))
    if hyper_mat is not None:
        corr = np.corrcoef(hyper_mat, rowvar=False)
        metrics = _corr_metrics(corr)
        if metrics:
            results["hyperparameters"] = metrics
            if (
                reference_corr
                and reference_corr.get("hyperparameters") is not None
            ):
                ref = reference_corr["hyperparameters"]
                if ref.shape == corr.shape:
                    diff = np.abs(corr - ref)
                    results["hyperparameters"]["mean_corr_diff"] = float(
                        np.mean(diff[np.triu_indices(diff.shape[0], k=1)])
                    )

    return results


def _compute_psis_khat(
    *,
    model: Callable[..., Any],
    model_args: Tuple[Any, ...],
    model_kwargs: Dict[str, Any],
    guide: Any,
    guide_params: Dict[str, Any],
    vi_samples: Optional[Dict[str, jnp.ndarray]],
    latent_samples: Optional[jnp.ndarray],
) -> Optional[Dict[str, Any]]:
    """Compute PSIS k-hat on VI posterior draws to assess joint mismatch."""

    if not vi_samples or latent_samples is None:
        logger.debug("Skipping PSIS k-hat: no VI samples or latent latents.")
        return None

    try:
        sample_tree = {
            name: jnp.asarray(array) for name, array in vi_samples.items()
        }
        n_draws = None
        for array in sample_tree.values():
            size = int(array.shape[0]) if array.ndim > 0 else 0
            n_draws = size if n_draws is None else min(n_draws, size)

        latent_arr = jnp.asarray(latent_samples)
        if latent_arr.ndim == 1:
            latent_arr = latent_arr[:, None]
        if n_draws is None:
            n_draws = int(latent_arr.shape[0]) if latent_arr.ndim > 0 else 0
        n_draws = min(n_draws, int(latent_arr.shape[0]))
        if n_draws <= 0:
            return None
        if n_draws < MIN_PSIS_DRAWS:
            logger.warning(
                f"Skipping PSIS k-hat: n_draws={int(n_draws)} < {MIN_PSIS_DRAWS}. "
                "Increase vi_posterior_draws for a stable estimate."
            )
            return None

        sample_tree = {k: v[:n_draws] for k, v in sample_tree.items()}
        latent_arr = latent_arr[:n_draws]

        def _log_posterior(sample):
            log_prob, _ = log_density(model, model_args, model_kwargs, sample)
            return log_prob

        log_posterior = jax.vmap(_log_posterior)(sample_tree)

        latent_name = f"_{getattr(guide, 'prefix', 'auto')}_latent"
        guide_params = dict(guide_params or {})

        def _log_guide(latent):
            params = dict(guide_params)
            params[latent_name] = latent
            log_prob, _ = log_density(guide, model_args, model_kwargs, params)
            return log_prob

        log_guide = jax.vmap(_log_guide)(latent_arr)

        log_r = np.asarray(
            jax.device_get(log_posterior - log_guide), dtype=np.float64
        )
        lw, k_hat = az.psislw(log_r)
        lw = np.asarray(lw, dtype=np.float64)
        k_hat = np.asarray(k_hat, dtype=np.float64)
        khat_max = float(np.max(k_hat)) if k_hat.size else np.nan
        psis_weights = _normalize_log_weights(lw)
        moment_summary = _compute_psis_moment_checks(sample_tree, psis_weights)
        corr_summary = _compute_correlation_diagnostics(sample_tree)
        logger.debug(
            f"Computed PSIS k-hat from VI draws: max={khat_max:.3f}, draws={int(n_draws)}"
        )

        return {
            "psis_log_weights": log_r,
            "psis_log_weights_smoothed": lw,
            "psis_weights": psis_weights,
            "psis_khat": k_hat,
            "psis_khat_max": khat_max,
            "psis_moment_summary": moment_summary,
            "psis_correlation_summary": corr_summary,
        }
    except Exception as exc:  # pragma: no cover - diagnostic best-effort
        logger.debug(f"Could not compute PSIS k-hat for VI samples: {exc}")
        return None


__all__ = [
    "_compute_psis_khat",
    "_interpret_khat",
    "_compute_psis_moment_checks",
    "_emit_moment_warnings",
    "_compute_correlation_diagnostics",
    "_normalize_log_weights",
]
