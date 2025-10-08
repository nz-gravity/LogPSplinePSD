from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class CoarseGrainSpec:
    """Static binning information for coarse-grained frequency data."""

    f_coarse: np.ndarray
    selection_mask: np.ndarray
    mask_low: np.ndarray
    mask_high: np.ndarray
    bin_indices: np.ndarray
    sort_indices: np.ndarray
    bin_counts: np.ndarray
    n_low: int
    n_bins_high: int


def compute_binning_structure(
    freqs: np.ndarray,
    *,
    f_transition: float,
    n_log_bins: int,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> CoarseGrainSpec:
    """Compute coarse-graining bins for a monotonically increasing frequency grid."""

    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1-D array")
    if not np.all(np.diff(freqs) >= 0):
        raise ValueError("freqs must be monotonically increasing")
    if n_log_bins <= 0:
        raise ValueError("n_log_bins must be positive")
    if f_transition <= 0:
        raise ValueError("f_transition must be positive")

    f_min = freqs[0] if f_min is None else max(f_min, freqs[0])
    f_max = freqs[-1] if f_max is None else min(f_max, freqs[-1])
    if f_max <= f_min:
        raise ValueError("f_max must be greater than f_min after clamping")

    in_range = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(in_range):
        raise ValueError("No frequencies fall within the requested range")

    selected_freqs = freqs[in_range]
    mask_low_full = (freqs >= f_min) & (freqs <= f_transition)
    mask_high_full = (freqs > f_transition) & (freqs <= f_max)

    mask_low = mask_low_full[in_range]
    mask_high = mask_high_full[in_range]

    n_low = int(mask_low.sum())

    high_freqs = selected_freqs[mask_high]
    if high_freqs.size == 0:
        # No high-frequency bins; only low frequencies retained
        return CoarseGrainSpec(
            f_coarse=selected_freqs,
            mask_low=mask_low,
            mask_high=mask_high,
            bin_indices=np.array([], dtype=np.int32),
            sort_indices=np.array([], dtype=np.int32),
            bin_counts=np.array([], dtype=np.int32),
            n_low=n_low,
            n_bins_high=0,
        )

    # Build logarithmic bin edges for the high-frequency region
    high_min = max(f_transition, high_freqs[0])
    edges = np.logspace(
        np.log10(high_min),
        np.log10(f_max),
        num=n_log_bins + 1,
        base=10.0,
    )
    # Ensure edges cover the entire high-frequency range
    edges[0] = min(edges[0], high_min)
    edges[-1] = max(edges[-1], f_max)

    # Assign bins (0-indexed). Points below first edge get bin -1; clamp to 0.
    raw_indices = np.digitize(high_freqs, edges[1:-1], right=False)
    unique_bins, reindexed = np.unique(raw_indices, return_inverse=True)
    n_bins_high = int(unique_bins.size)

    sort_indices = np.argsort(reindexed, kind="stable")
    sorted_bins = reindexed[sort_indices]

    # Count points per bin
    bin_counts = np.bincount(sorted_bins, minlength=n_bins_high)

    # Compute representative frequency per bin (mean of members)
    bin_sums = np.zeros(n_bins_high, dtype=np.float64)
    np.add.at(bin_sums, sorted_bins, high_freqs[sort_indices])
    bin_means = bin_sums / np.where(bin_counts > 0, bin_counts, 1)

    f_coarse = np.concatenate((selected_freqs[:n_low], bin_means))

    return CoarseGrainSpec(
        f_coarse=f_coarse,
        selection_mask=in_range,
        mask_low=mask_low,
        mask_high=mask_high,
        bin_indices=reindexed,
        sort_indices=sort_indices,
        bin_counts=bin_counts,
        n_low=n_low,
        n_bins_high=n_bins_high,
    )


def compute_gaussian_bin_statistics(
    freqs: np.ndarray,
    power: np.ndarray,
    spec: CoarseGrainSpec,
    *,
    epsilon: float = 1e-20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coarse-grained Gaussian statistics per Caprini et al."""

    selected_freqs = np.asarray(freqs)[spec.selection_mask]
    selected_power = np.asarray(power)[spec.selection_mask]

    low_freqs = selected_freqs[spec.mask_low]
    low_power = selected_power[spec.mask_low]
    low_sigma = np.sqrt(np.maximum(low_power**2, epsilon))

    if spec.n_bins_high == 0:
        return low_freqs, low_power, low_sigma

    high_freqs = selected_freqs[spec.mask_high]
    high_power = selected_power[spec.mask_high]
    high_freqs_sorted = high_freqs[spec.sort_indices]
    high_power_sorted = high_power[spec.sort_indices]
    bins_sorted = spec.bin_indices[spec.sort_indices]

    inv_var = 1.0 / (np.maximum(high_power_sorted**2, epsilon))

    sum_inv_var = np.bincount(
        bins_sorted,
        weights=inv_var,
        minlength=spec.n_bins_high,
    )
    sum_weighted_power = np.bincount(
        bins_sorted,
        weights=high_power_sorted * inv_var,
        minlength=spec.n_bins_high,
    )
    sum_weighted_freqs = np.bincount(
        bins_sorted,
        weights=high_freqs_sorted * inv_var,
        minlength=spec.n_bins_high,
    )

    sum_inv_var = np.maximum(sum_inv_var, epsilon)
    coarse_power_high = sum_weighted_power / sum_inv_var
    coarse_freq_high = sum_weighted_freqs / sum_inv_var
    sigma_high = 1.0 / np.sqrt(sum_inv_var)

    coarse_freqs = np.concatenate((low_freqs, coarse_freq_high))
    coarse_power = np.concatenate((low_power, coarse_power_high))
    sigma = np.concatenate((low_sigma, sigma_high))

    return coarse_freqs, coarse_power, sigma


def apply_coarse_graining_univar(
    power: np.ndarray,
    spec: CoarseGrainSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply coarse graining to a power array using ``CoarseGrainSpec``."""

    if power.ndim != 1:
        raise ValueError("power must be 1-D")
    if power.size != spec.mask_low.size:
        raise ValueError("power length must match selected frequency count")

    power_low = power[spec.mask_low]

    if spec.n_bins_high == 0:
        weights = np.ones_like(power_low, dtype=np.float64)
        return power_low, weights

    power_high = power[spec.mask_high]
    power_high_sorted = power_high[spec.sort_indices]
    sorted_bins = spec.bin_indices[spec.sort_indices]

    sum_power = np.bincount(
        sorted_bins,
        weights=power_high_sorted,
        minlength=spec.n_bins_high,
    )
    counts = spec.bin_counts.astype(np.float64)
    means = np.divide(
        sum_power,
        np.where(counts > 0, counts, 1),
        out=np.zeros_like(sum_power),
        where=counts > 0,
    )

    power_coarse = np.concatenate((power_low, means))
    weights = np.concatenate(
        (
            np.ones_like(power_low, dtype=np.float64),
            counts,
        )
    )
    return power_coarse, weights
