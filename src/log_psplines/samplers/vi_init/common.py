"""Small shared helpers for VI warm-start flows."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def _to_np(x) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def _to_np_dict(d: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    return {} if not d else {name: _to_np(value) for name, value in d.items()}


def _median_vi_values(
    draws: Dict[str, jnp.ndarray],
    site_names: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, jnp.ndarray]]:
    if not draws:
        return None
    out = {}
    for name, value in draws.items():
        if site_names is None or name in site_names:
            arr = jnp.asarray(value)
            out[name] = jnp.median(arr, axis=0) if arr.ndim >= 1 else arr
    return out or None


def _strip_coarse_vi_plot_arrays(
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(diagnostics)
    for key in (
        "psd",
        "weights",
        "psd_quantiles",
        "psd_matrix",
        "psd_matrix_complex",
        "coherence_quantiles",
        "coarse_vi_label",
    ):
        out.pop(key, None)
    return out


def _validate_positive_finite_psd(psd: np.ndarray) -> bool:
    arr = np.asarray(psd, dtype=np.float64)
    return bool(arr.size and np.isfinite(arr).all() and (arr > 0.0).all())


def _vi_means_are_usable(means: Dict[str, Any]) -> bool:
    if not means:
        return False
    for name, value in means.items():
        try:
            arr = _to_np(value)
        except Exception:
            return False
        if arr.size and not np.isfinite(arr).all():
            return False
        if str(name).startswith("delta_") and arr.size and not (arr > 0).all():
            return False
    return True


def _sanitize_vi_init_values(
    values: Optional[Dict[str, Any]],
    *,
    delta_floor: float = 1e-10,
) -> Optional[Dict[str, jnp.ndarray]]:
    if not values:
        return None
    floor = jnp.asarray(delta_floor)
    out = {}
    for name, value in values.items():
        arr = jnp.asarray(value)
        out[name] = (
            jnp.maximum(arr, floor.astype(arr.dtype))
            if str(name).startswith("delta_")
            else arr
        )
    return out
