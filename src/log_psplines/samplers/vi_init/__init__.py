"""Variational-initialisation helpers shared across samplers."""

from .core import VIInitialisationArtifacts, VIResult, fit_vi, resolve_guide
from .plan import VIWarmStartPlan, build_coarse_sampler_from_plan

__all__ = [
    "fit_vi",
    "resolve_guide",
    "VIResult",
    "VIInitialisationArtifacts",
    "VIWarmStartPlan",
    "build_coarse_sampler_from_plan",
]
