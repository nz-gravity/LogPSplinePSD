"""Bayesian PSD estimation with shared scalar spline components."""

try:
    from ._version import __commit_id__, __version__
except ImportError:
    __version__ = "0+unknown"
    __commit_id__ = None

from .basis import SplineBasis
from .config import PowerConfig, StationaryConfig
from .data import TimeSeries, WishartData
from .data.spectral import PowerData
from .fit import fit
from .models.anova import ANOVALogPSpline
from .models.matrix import SpectralMatrix
from .models.parametric import ParametricSpectrum
from .models.spectrum import LogPSpline
from .preprocessing.moving_periodogram import (
    moving_periodogram,
    scattered_moving_periodogram,
)
from .preprocessing.power_partition import (
    PowerPartition,
    coarse_grain_power,
    mask_power,
    select_power_partition,
)
from .results import PSDResult

__all__ = [
    "fit",
    "StationaryConfig",
    "PSDResult",
    "SplineBasis",
    "LogPSpline",
    "ANOVALogPSpline",
    "ParametricSpectrum",
    "SpectralMatrix",
    "TimeSeries",
    "WishartData",
    "PowerConfig",
    "PowerData",
    "moving_periodogram",
    "scattered_moving_periodogram",
    "PowerPartition",
    "coarse_grain_power",
    "mask_power",
    "select_power_partition",
]
