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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Coarse-grain a power spectrum using precomputed masks/bins."""

    power_low = power[mask_low]
    weights_low = jnp.ones_like(power_low)

    if n_bins_high == 0:
        power_coarse = power_low
        weights = weights_low
    else:
        power_high = power[mask_high]
        power_high_sorted = power_high[sort_indices]

        bin_counts = jnp.asarray(bin_counts)
        sum_power = jax.ops.segment_sum(
            power_high_sorted,
            bin_indices_sorted,
            n_bins_high,
        )
        counts = jax.ops.segment_sum(
            jnp.ones_like(power_high_sorted),
            bin_indices_sorted,
            n_bins_high,
        )

        counts = jnp.where(counts > 0, counts, 1.0)
        mean_power = sum_power / counts

        non_empty = bin_counts > 0
        mean_power = mean_power[non_empty]
        counts = counts[non_empty]

        power_coarse = jnp.concatenate((power_low, mean_power))
        weights = jnp.concatenate((weights_low, counts))

    return power_coarse, weights
