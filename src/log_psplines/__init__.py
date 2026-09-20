"""Bayesian PSD estimation with shared scalar spline components."""
try:
    from ._version import __commit_id__, __version__
except ImportError:
    __version__ = "0+unknown"
    __commit_id__ = None

from .basis import SplineBasis
from .models.spectrum import LogPSpline
from .models.matrix import SpectralMatrix
from .data import TimeSeries, WishartData
from .data.spectral import PowerSpectrum
from .config import PipelineConfig, PowerSplineConfig
from .pipeline import fit, make_pipeline
from .results import PSDResult

__all__ = ["fit", "make_pipeline", "PipelineConfig", "PSDResult", "SplineBasis",
           "LogPSpline", "SpectralMatrix", "TimeSeries", "WishartData",
           "PowerSplineConfig", "PowerSpectrum"]
