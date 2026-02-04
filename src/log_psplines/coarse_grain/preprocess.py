from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

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
    f_transition: float,
    n_log_bins: int,
    binning: Literal["log", "linear"] = "linear",
    representative: Literal["mean", "middle"] = "middle",
    keep_low: bool = False,
    n_freqs_per_bin: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> CoarseGrainSpec:
    """Compute coarse-graining bins for a monotonically increasing frequency grid.

    Args:
        freqs: Fine frequency grid (monotonically increasing).
        f_transition: When ``keep_low=True``, frequencies <= f_transition are
            retained as-is and only the high band is binned.
        n_log_bins: Number of coarse bins (name retained for backwards
            compatibility).
        binning: Bin spacing ("log" or "linear").
        representative: Representative frequency per coarse bin. ``"middle"``
            picks the midpoint Fourier frequency (middle member on the discrete
            grid) when possible.
        keep_low: When True, retain the low-frequency region unbinned.
            When False (paper-style), bin all retained frequencies.
        n_freqs_per_bin: Optional paper-exact mode for linear binning: force
            equal-size bins containing exactly this many (odd) fine-grid
            frequencies each.
        f_min: Optional lower bound on retained frequencies.
        f_max: Optional upper bound on retained frequencies.
    """

    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1-D array")
    if not np.all(np.diff(freqs) >= 0):
        raise ValueError("freqs must be monotonically increasing")
    if n_log_bins <= 0:
        raise ValueError("n_log_bins must be positive")
    if f_transition <= 0:
        raise ValueError("f_transition must be positive")
    if binning not in {"log", "linear"}:
        raise ValueError("binning must be 'log' or 'linear'")
    if representative not in {"mean", "middle"}:
        raise ValueError("representative must be 'mean' or 'middle'")
    if n_freqs_per_bin is not None:
        n_freqs_per_bin = int(n_freqs_per_bin)
        if n_freqs_per_bin <= 0:
            raise ValueError("n_freqs_per_bin must be positive when provided")
        if n_freqs_per_bin % 2 == 0:
            raise ValueError(
                "n_freqs_per_bin must be odd to define a midpoint Fourier frequency"
            )
        if binning != "linear":
            raise ValueError(
                "n_freqs_per_bin is only supported for linear binning"
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
    if keep_low:
        mask_low_full = (freqs >= f_min) & (freqs <= f_transition)
        mask_high_full = (freqs > f_transition) & (freqs <= f_max)
        mask_low = mask_low_full[in_range]
        mask_high = mask_high_full[in_range]
        n_low = int(mask_low.sum())
    else:
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

    high_freqs = selected_freqs[mask_high]
    if high_freqs.size == 0:
        # No high-frequency bins; only low frequencies retained
        return CoarseGrainSpec(
            f_coarse=selected_freqs,
            selection_mask=in_range,
            mask_low=mask_low,
            mask_high=mask_high,
            bin_indices=np.array([], dtype=np.int32),
            sort_indices=np.array([], dtype=np.int32),
            bin_counts=np.array([], dtype=np.int32),
            bin_widths=np.array([], dtype=np.float64),
            n_low=n_low,
            n_bins_high=0,
            fine_spacing=fine_spacing,
        )

    if binning == "linear" and n_freqs_per_bin is not None:
        # Fixed-size, equal-length bins on the fine grid (paper style).
        n_high = int(high_freqs.size)
        if n_high % n_freqs_per_bin != 0:
            raise ValueError(
                f"Selected high-band frequencies ({n_high}) must be divisible by "
                f"n_freqs_per_bin ({n_freqs_per_bin}) for equal-length bins."
            )
        n_bins_high = int(n_high // n_freqs_per_bin)
        # High freqs are already in increasing order; no sorting needed.
        sort_indices = np.arange(n_high, dtype=np.int32)
        bin_indices = np.repeat(
            np.arange(n_bins_high, dtype=np.int32),
            repeats=n_freqs_per_bin,
        )
        bin_counts = np.full((n_bins_high,), n_freqs_per_bin, dtype=np.int32)
        # Exact midpoint Fourier frequency (middle member).
        mid_offset = n_freqs_per_bin // 2
        f_rep = np.array(
            [
                high_freqs[b * n_freqs_per_bin + mid_offset]
                for b in range(n_bins_high)
            ],
            dtype=np.float64,
        )
        # For equal-spacing grids, this makes weights == bin_counts in univariate mode.
        bin_widths = bin_counts.astype(np.float64) * float(fine_spacing)
        f_coarse = np.concatenate((selected_freqs[:n_low], f_rep))
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
            n_bins_high=n_bins_high,
            fine_spacing=fine_spacing,
        )

    if (not keep_low) and binning == "linear":
        # Paper-style binning across the full retained frequency grid:
        # partition into n_log_bins consecutive, disjoint subsets J_h, each with
        # an odd number of Fourier frequencies so that the "middle" member is a
        # well-defined midpoint Fourier frequency.
        n_bins = int(n_log_bins)
        n_high = int(high_freqs.size)
        if n_bins <= 0:
            raise ValueError("n_log_bins must be positive")
        # If the requested number of bins exceeds the number of retained
        # frequencies, fall back to the identity partition (no aggregation).
        n_bins = min(n_bins, n_high)
        # Sum of odd bin sizes has parity == n_bins (mod 2), so require parity match.
        if (n_high % 2) != (n_bins % 2):
            raise ValueError(
                f"Cannot partition {n_high} frequencies into {n_bins} bins with all bin sizes odd "
                "(parity mismatch). Choose a different n_log_bins or adjust f_min/f_max."
            )

        base = n_high // n_bins
        if base % 2 == 0:
            base -= 1
        if base < 1:
            raise ValueError(
                "n_log_bins is too large to form odd-sized bins (minimum bin size would be < 1)."
            )
        remainder = n_high - (base * n_bins)
        if remainder < 0 or (remainder % 2) != 0:
            raise ValueError(
                "Internal error constructing odd-sized bins; check n_log_bins and frequency selection."
            )
        n_plus_two = remainder // 2

        bin_counts = np.full((n_bins,), base, dtype=np.int32)
        if n_plus_two:
            bin_counts[:n_plus_two] += 2

        sort_indices = np.arange(n_high, dtype=np.int32)
        bin_indices = np.repeat(
            np.arange(n_bins, dtype=np.int32),
            repeats=bin_counts,
        ).astype(np.int32)

        # Representative frequency: middle member (paper), or mean (diagnostics/back-compat).
        starts = np.concatenate(([0], np.cumsum(bin_counts)[:-1]))
        mids = starts + (bin_counts // 2)
        if representative == "mean":
            f_rep = np.empty((n_bins,), dtype=np.float64)
            for b in range(n_bins):
                start = int(starts[b])
                end = start + int(bin_counts[b])
                f_rep[b] = float(np.mean(high_freqs[start:end]))
        else:
            f_rep = high_freqs[mids].astype(np.float64)

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
            n_bins_high=int(n_bins),
            fine_spacing=fine_spacing,
        )

    # Otherwise: bin by edges (log or linear), allowing variable bin membership.
    high_min = (
        float(high_freqs[0])
        if not keep_low
        else max(f_transition, float(high_freqs[0]))
    )
    if binning == "log":
        if high_min <= 0:
            raise ValueError(
                "Log binning requires positive frequencies in the high band."
            )
        edges = np.logspace(
            np.log10(high_min),
            np.log10(f_max),
            num=n_log_bins + 1,
            base=10.0,
        )
    else:
        edges = np.linspace(high_min, f_max, num=n_log_bins + 1)
    edges[0] = min(edges[0], high_min)
    edges[-1] = max(edges[-1], f_max)

    raw_indices = np.digitize(high_freqs, edges[1:-1], right=False)
    unique_bins, reindexed = np.unique(raw_indices, return_inverse=True)
    n_bins_high = int(unique_bins.size)

    sort_indices = np.argsort(reindexed, kind="stable")
    sorted_bins = reindexed[sort_indices]

    bin_counts = np.bincount(sorted_bins, minlength=n_bins_high)

    high_sorted = high_freqs[sort_indices]
    if representative == "mean":
        bin_sums = np.zeros(n_bins_high, dtype=np.float64)
        np.add.at(bin_sums, sorted_bins, high_sorted)
        f_rep = bin_sums / np.where(bin_counts > 0, bin_counts, 1)
    else:
        # Choose the middle member frequency within each bin. For an odd number
        # of members this is the exact midpoint Fourier frequency; for an even
        # number of members we take the lower of the two central frequencies.
        f_rep = np.zeros(n_bins_high, dtype=np.float64)
        pos = 0
        for b in range(n_bins_high):
            count = int(bin_counts[b])
            if count <= 0:
                continue
            mid = pos + ((count - 1) // 2)
            f_rep[b] = high_sorted[mid]
            pos += count

    raw_widths = edges[unique_bins + 1] - edges[unique_bins]
    min_width = np.array(fine_spacing, dtype=np.float64)
    bin_widths = np.maximum(raw_widths, min_width)

    f_coarse = np.concatenate((selected_freqs[:n_low], f_rep))

    return CoarseGrainSpec(
        f_coarse=f_coarse,
        selection_mask=in_range,
        mask_low=mask_low,
        mask_high=mask_high,
        bin_indices=reindexed,
        sort_indices=sort_indices,
        bin_counts=bin_counts,
        bin_widths=bin_widths,
        n_low=n_low,
        n_bins_high=n_bins_high,
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

    power_low = power[spec.mask_low]

    fine_spacing = spec.fine_spacing
    if freqs is not None:
        freqs = np.asarray(freqs, dtype=np.float64)
        if freqs.ndim != 1:
            raise ValueError("freqs must be 1-D when provided")
        if freqs.size != spec.selection_mask.sum():
            raise ValueError(
                "freqs length must match selected frequency count"
            )
        if freqs.size >= 2:
            diff_freqs = np.diff(freqs)
            positive_diffs = diff_freqs[diff_freqs > 0]
            if positive_diffs.size > 0:
                fine_spacing = float(np.median(positive_diffs))

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

    bin_widths = spec.bin_widths.astype(np.float64, copy=False)
    if bin_widths.size != spec.n_bins_high:
        raise ValueError("bin_widths length must equal n_bins_high")

    if fine_spacing <= 0:
        raise ValueError("fine_spacing must be positive")

    new_weights_high = bin_widths / fine_spacing
    # Renormalize to match exact count of fine frequencies for consistency
    total_expected = float(spec.bin_counts.sum())
    total_current = (
        float(new_weights_high.sum()) if new_weights_high.size else 0.0
    )
    if total_current > 0 and total_expected > 0:
        new_weights_high *= total_expected / total_current
    # Do not enforce monotonicity in weights used for likelihood; small
    # non-monotonicity due to edge quantization is expected and correct.

    power_coarse = np.concatenate((power_low, means))
    weights = np.concatenate(
        (
            np.ones_like(power_low, dtype=np.float64),
            new_weights_high,
        )
    )
    return power_coarse, weights
