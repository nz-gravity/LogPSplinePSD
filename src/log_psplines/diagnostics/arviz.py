"""ArviZ interoperability for sampling diagnostics only."""

from __future__ import annotations

from arviz_base import from_dict


def to_arviz(result):
    """Build a minimal ArviZ-compatible diagnostics view of a PSDResult."""
    groups = {"posterior": result.posterior}
    if result.sample_stats is not None:
        groups["sample_stats"] = result.sample_stats
    if result.log_likelihood is not None:
        groups["log_likelihood"] = result.log_likelihood
    return from_dict(groups, attrs=result.metadata)


__all__ = ["to_arviz"]
