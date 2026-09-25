"""ArviZ interoperability for sampling diagnostics only."""

from __future__ import annotations

from arviz_base import from_dict


def _dataset_values(dataset):
    return {name: variable.values for name, variable in dataset.data_vars.items()}


def to_arviz(result):
    """Build a minimal ArviZ-compatible diagnostics object from a PSDResult.

    Only posterior samples and sampler statistics cross this boundary.
    Spectral reconstruction, model storage and persistence remain
    LogPSplinePSD responsibilities.
    """
    kwargs = {"posterior": _dataset_values(result.posterior)}
    if result.sample_stats is not None:
        kwargs["sample_stats"] = _dataset_values(result.sample_stats)
    return from_dict(**kwargs)


__all__ = ["to_arviz"]
