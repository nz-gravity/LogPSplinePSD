"""Backward-compatible import wrapper for time-domain moment utilities."""

from .diagnostics.time_domain_moments import (  # noqa: F401
    compute_empirical_covariances,
    compute_empirical_variances,
    compute_psd_covariances,
    compute_psd_variances,
)

__all__ = [
    "compute_empirical_covariances",
    "compute_empirical_variances",
    "compute_psd_covariances",
    "compute_psd_variances",
]
