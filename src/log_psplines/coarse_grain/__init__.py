"""Coarse-graining utilities for frequency-domain PSD data."""

from .config import CoarseGrainConfig
from .plotting import plot_coarse_vs_original
from .preprocess import (
    CoarseGrainSpec,
    apply_coarse_graining_univar,
    compute_binning_structure,
    compute_gaussian_bin_statistics,
)

__all__ = [
    "CoarseGrainConfig",
    "CoarseGrainSpec",
    "compute_binning_structure",
    "apply_coarse_graining_univar",
    "compute_gaussian_bin_statistics",
    "plot_coarse_vs_original",
]
