from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def coarse_grain_univar(
    power: jnp.ndarray,
    *,
    mask_low: jnp.ndarray,
    mask_high: jnp.ndarray,
    sort_indices: jnp.ndarray,
    bin_indices_sorted: jnp.ndarray,
    bin_counts: jnp.ndarray,
    n_low: int,
    n_bins_high: int,
    freqs: jnp.ndarray,
    f_coarse: jnp.ndarray,
    bin_widths: jnp.ndarray,
    fine_spacing: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Coarse-grain a power spectrum using precomputed masks/bins."""

    if n_bins_high <= 0:
        raise ValueError("Coarse-graining spec has no bins.")

    power_high = power[mask_high]
    power_high_sorted = power_high[sort_indices]

    bin_counts = jnp.asarray(bin_counts)
    sum_power = jax.ops.segment_sum(
        power_high_sorted,
        bin_indices_sorted,
        n_bins_high,
    )

    power_coarse = sum_power
    weights = bin_counts.astype(sum_power.dtype)

    return power_coarse, weights
