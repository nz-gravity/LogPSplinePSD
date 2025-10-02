"""Variational-initialisation helpers shared across samplers."""

from .core import VIResult, fit_vi, resolve_guide
from .mixin import VIInitialisationArtifacts, VIInitialisationMixin

__all__ = [
    "fit_vi",
    "resolve_guide",
    "VIResult",
    "VIInitialisationMixin",
    "VIInitialisationArtifacts",
]
