from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(slots=True)
class CoarseGrainConfig:
    """Configuration for frequency-domain coarse graining."""

    enabled: bool = False
    # When True, keep frequencies <= f_transition unbinned; otherwise bin all
    # retained frequencies into disjoint intervals (paper-style).
    keep_low: bool = False
    f_transition: float = 1e-3
    n_log_bins: int = 1000
    binning: Literal["log", "linear"] = "linear"
    # Representative frequency for each coarse bin:
    # - "mean": arithmetic mean of member frequencies
    # - "middle": middle Fourier frequency (requires odd bin size for exact "midpoint")
    representative: Literal["mean", "middle"] = "middle"
    # If provided (and binning="linear"), enforce equal-size bins containing this
    # many fine-grid frequencies each. Must be odd to match the paper exactly.
    n_freqs_per_bin: Optional[int] = None
    f_min: Optional[float] = None
    f_max: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "keep_low": self.keep_low,
            "f_transition": self.f_transition,
            "n_log_bins": self.n_log_bins,
            "binning": self.binning,
            "representative": self.representative,
            "n_freqs_per_bin": self.n_freqs_per_bin,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }
