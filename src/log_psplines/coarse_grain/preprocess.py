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
    bin_widths: np.ndarray
    n_low: int
    n_bins_high: int
    fine_spacing: float


def compute_binning_structure(
    freqs: np.ndarray,
    *,
    n_bins: Optional[int] = None,
    n_freqs_per_bin: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> CoarseGrainSpec:
    """Compute full-band linear coarse-graining bins for a frequency grid.

    Args:
        freqs: Fine frequency grid (monotonically increasing).
        n_bins: Number of coarse bins spanning the retained grid. Bin sizes are
            as equal as possible and must be odd.
        n_freqs_per_bin: Optional fixed-membership mode: aim for bins containing
            exactly this many (odd) fine-grid frequencies each. When the
            selected frequency count is not divisible by ``n_freqs_per_bin``,
            the final coarse bin is adjusted to absorb the remainder while
            preserving odd bin sizes.
        f_min: Optional lower bound on retained frequencies.
        f_max: Optional upper bound on retained frequencies.
    """

    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1-D array")
    if not np.all(np.diff(freqs) >= 0):
        raise ValueError("freqs must be monotonically increasing")
    if (n_bins is None) == (n_freqs_per_bin is None):
        raise ValueError(
            "Exactly one of n_bins or n_freqs_per_bin must be provided."
        )
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError("n_bins must be positive when provided.")
    if n_freqs_per_bin is not None:
        n_freqs_per_bin = int(n_freqs_per_bin)
        if n_freqs_per_bin <= 0:
            raise ValueError("n_freqs_per_bin must be positive when provided")
        if n_freqs_per_bin % 2 == 0:
            raise ValueError(
                "n_freqs_per_bin must be odd to define a midpoint Fourier frequency."
            )

    freq_min = float(freqs[0])
    freq_max = float(freqs[-1])
    f_min = freq_min if f_min is None else float(f_min)
    f_max = freq_max if f_max is None else float(f_max)
    f_min = min(max(f_min, freq_min), freq_max)
    f_max = min(max(f_max, freq_min), freq_max)
    if f_max < f_min:
        f_max = f_min

    in_range = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(in_range):
        raise ValueError("No frequencies fall within the requested range")

    selected_freqs = freqs[in_range]
    mask_low = np.zeros_like(selected_freqs, dtype=bool)
    mask_high = np.ones_like(selected_freqs, dtype=bool)
    n_low = 0

    if selected_freqs.size < 2:
        fine_spacing = 1.0
    else:
        freq_diffs = np.diff(selected_freqs)
        positive_diffs = freq_diffs[freq_diffs > 0]
        if positive_diffs.size == 0:
            raise ValueError(
                "Selected frequencies must contain increasing values."
            )
        fine_spacing = float(np.median(positive_diffs))

    n_high = int(selected_freqs.size)
    if n_high <= 0:
        raise ValueError("No retained frequencies for coarse graining.")

    if n_freqs_per_bin is not None:
        # Fixed-membership bins on the fine grid (paper style).
        # We keep all bins odd-sized so each has a well-defined midpoint
        # Fourier frequency. When the selection is not divisible by
        # n_freqs_per_bin, we adjust the final bin to absorb the remainder.
        n_full = int(n_high // n_freqs_per_bin)
        remainder = int(n_high - n_full * n_freqs_per_bin)

        if n_full == 0:
            if n_high % 2 == 0:
                raise ValueError(
                    f"Selected frequencies ({n_high}) cannot be partitioned into odd-sized bins "
                    f"when n_freqs_per_bin ({n_freqs_per_bin}) exceeds the selection. "
                    "Adjust f_min/f_max or use n_bins mode."
                )
            n_bins_high = 1
            bin_counts = np.array([n_high], dtype=np.int32)
        else:
            if remainder == 0:
                n_bins_high = n_full
                bin_counts = np.full(
                    (n_bins_high,), n_freqs_per_bin, dtype=np.int32
                )
            elif remainder % 2 == 1:
                n_bins_high = n_full + 1
                bin_counts = np.concatenate(
                    [
                        np.full((n_full,), n_freqs_per_bin, dtype=np.int32),
                        np.array([remainder], dtype=np.int32),
                    ],
                    axis=0,
                )
            else:
                # Even remainder: fold it into the final full bin (odd+even=odd).
                n_bins_high = n_full
                bin_counts = np.full(
                    (n_bins_high,), n_freqs_per_bin, dtype=np.int32
                )
                bin_counts[-1] = int(bin_counts[-1] + remainder)
    else:
        n_bins_high = int(n_bins)
        if n_bins_high > n_high:
            raise ValueError(
                f"n_bins ({n_bins_high}) exceeds retained frequency count ({n_high})."
            )
        if (n_high % 2) != (n_bins_high % 2):
            raise ValueError(
                f"Cannot partition {n_high} frequencies into {n_bins_high} bins with all bin sizes odd "
                "(parity mismatch). Choose a different n_bins or adjust f_min/f_max."
            )
        base = n_high // n_bins_high
        if base % 2 == 0:
            base -= 1
        if base < 1:
            raise ValueError(
                "n_bins is too large to form odd-sized bins (minimum bin size would be < 1)."
            )
        remainder = n_high - (base * n_bins_high)
        if remainder < 0 or (remainder % 2) != 0:
            raise ValueError(
                "Internal error constructing odd-sized bins; check n_bins and frequency selection."
            )
        n_plus_two = remainder // 2
        bin_counts = np.full((n_bins_high,), base, dtype=np.int32)
        if n_plus_two:
            bin_counts[:n_plus_two] += 2

    if np.any(bin_counts <= 0) or np.any(bin_counts % 2 == 0):
        raise ValueError("All bin sizes must be positive and odd.")
    if int(np.sum(bin_counts)) != n_high:
        raise ValueError(
            "Bin counts must sum to the retained frequency count."
        )

    sort_indices = np.arange(n_high, dtype=np.int32)
    bin_indices = np.repeat(
        np.arange(n_bins_high, dtype=np.int32),
        repeats=bin_counts,
    ).astype(np.int32)
    if bin_indices.shape[0] != n_high:
        raise ValueError(
            "Bin construction failed to cover all retained frequencies."
        )

    starts = np.concatenate(([0], np.cumsum(bin_counts)[:-1]))
    mids = starts + (bin_counts // 2)
    f_rep = selected_freqs[mids].astype(np.float64)

    bin_widths = bin_counts.astype(np.float64) * float(fine_spacing)
    f_coarse = np.asarray(f_rep, dtype=np.float64)

    return CoarseGrainSpec(
        f_coarse=f_coarse,
        selection_mask=in_range,
        mask_low=mask_low,
        mask_high=mask_high,
        bin_indices=bin_indices,
        sort_indices=sort_indices,
        bin_counts=bin_counts,
        bin_widths=bin_widths,
        n_low=n_low,
        n_bins_high=int(n_bins_high),
        fine_spacing=fine_spacing,
    )


def apply_coarse_graining_univar(
    power: np.ndarray,
    spec: CoarseGrainSpec,
    freqs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply coarse graining to a power array using ``CoarseGrainSpec``."""

    if power.ndim != 1:
        raise ValueError("power must be 1-D")
    if power.size != spec.mask_low.size:
        raise ValueError("power length must match selected frequency count")

    if freqs is not None:
        freqs = np.asarray(freqs, dtype=np.float64)
        if freqs.ndim != 1:
            raise ValueError("freqs must be 1-D when provided")
        if freqs.size != spec.selection_mask.sum():
            raise ValueError(
                "freqs length must match selected frequency count"
            )

    if spec.n_bins_high <= 0:
        raise ValueError("Coarse-graining spec has no bins.")

    power_high = power[spec.mask_high]
    power_high_sorted = power_high[spec.sort_indices]
    sorted_bins = spec.bin_indices[spec.sort_indices]

    sum_power = np.bincount(
        sorted_bins,
        weights=power_high_sorted,
        minlength=spec.n_bins_high,
    )
    counts = spec.bin_counts.astype(np.float64)
    if sum_power.shape[0] != counts.shape[0]:
        raise ValueError("Coarse-graining bins have inconsistent sizes.")

    power_coarse = sum_power.astype(np.float64, copy=False)
    weights = counts
    return power_coarse, weights
