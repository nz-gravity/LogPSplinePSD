"""ArviZ interoperability for sampling diagnostics only."""

from __future__ import annotations

from arviz_base import from_dict


_STAT_NAMES = {
    "accept_prob": "acceptance_rate",
    "num_steps": "n_steps",
    "adapt_state.step_size": "step_size",
}


def _dataset_values(dataset, *, rename=None):
    rename = rename or {}
    return {
        rename.get(name, name): variable.values
        for name, variable in dataset.data_vars.items()
    }


def to_arviz(result):
    """Build a minimal ArviZ-compatible diagnostics view of a PSDResult."""
    kwargs = {"posterior": _dataset_values(result.posterior)}
    if result.sample_stats is not None:
        kwargs["sample_stats"] = _dataset_values(
            result.sample_stats, rename=_STAT_NAMES
        )
    if result.log_likelihood is not None:
        kwargs["log_likelihood"] = _dataset_values(result.log_likelihood)
    return from_dict(**kwargs)


__all__ = ["to_arviz"]
