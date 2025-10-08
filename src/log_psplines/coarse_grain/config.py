from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class CoarseGrainConfig:
    """Configuration for frequency-domain coarse graining."""

    enabled: bool = False
    f_transition: float = 1e-3
    n_log_bins: int = 1000
    f_min: Optional[float] = None
    f_max: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "f_transition": self.f_transition,
            "n_log_bins": self.n_log_bins,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }
