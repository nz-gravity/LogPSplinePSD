from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class CoarseGrainConfig:
    """Configuration for frequency-domain coarse graining."""

    enabled: bool = False
    # Exactly one of n_bins or n_freqs_per_bin must be provided.
    n_bins: Optional[int] = 1000
    # If provided, enforce equal-size bins containing this many fine-grid
    # frequencies each. Must be odd and divide the retained frequency count.
    n_freqs_per_bin: Optional[int] = None
    f_min: Optional[float] = None
    f_max: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.n_bins is None) == (self.n_freqs_per_bin is None):
            raise ValueError(
                "Exactly one of n_bins or n_freqs_per_bin must be set."
            )
        if self.n_bins is not None:
            self.n_bins = int(self.n_bins)
            if self.n_bins <= 0:
                raise ValueError("n_bins must be positive when provided.")
        if self.n_freqs_per_bin is not None:
            self.n_freqs_per_bin = int(self.n_freqs_per_bin)
            if self.n_freqs_per_bin <= 0:
                raise ValueError(
                    "n_freqs_per_bin must be positive when provided."
                )
            if self.n_freqs_per_bin % 2 == 0:
                raise ValueError(
                    "n_freqs_per_bin must be odd to define a midpoint frequency."
                )

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "n_bins": self.n_bins,
            "n_freqs_per_bin": self.n_freqs_per_bin,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }
