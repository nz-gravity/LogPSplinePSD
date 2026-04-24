"""Pipeline warm-start plan dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT


@dataclass(frozen=True)
class VIWarmStartPlan:
    """Describe coarse-VI warm-start inputs independent of sampler classes."""

    strategy: str
    processed_data: Periodogram | MultivarFFT
    scaled_true_psd: np.ndarray | None
    metadata: dict[str, Any]
    model_n_knots: Any
    model_degree: int
    model_diff_matrix_order: int
    model_knot_kwargs: dict[str, Any]
    model_parametric_model: Any
    model_analytical_psd: Any


__all__ = ["VIWarmStartPlan"]
