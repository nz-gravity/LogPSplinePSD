"""Helpers for working with explicit per-factor diagnostics inputs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import xarray as xr
from arviz_base import from_dict

from ..arviz_utils._datatree import require_dataset as _require_dataset

_POSTERIOR_BLOCK_PATTERNS = (
    re.compile(r"^(?:weights_delta|phi|delta)_(\d+)$"),
    re.compile(r"^(?:weights_theta_(?:re|im)|theta_(?:re|im))_(\d+)_\d+$"),
)
_SAMPLE_STATS_BLOCK_PATTERN = re.compile(r"^(?P<base>.+)_channel_(\d+)$")
_LOG_LIKELIHOOD_BLOCK_PATTERN = re.compile(
    r"^(?P<base>log_likelihood_block)_(\d+)$"
)


def _posterior_factor_index(name: str) -> int | None:
    for pattern in _POSTERIOR_BLOCK_PATTERNS:
        match = pattern.match(name)
        if match:
            return int(match.group(1))
    return None


def _sample_stats_factor_key(name: str) -> tuple[str, int] | None:
    match = _SAMPLE_STATS_BLOCK_PATTERN.match(name)
    if match is None:
        return None
    return match.group("base"), int(match.group(2))


def _log_likelihood_factor_key(name: str) -> tuple[str, int] | None:
    match = _LOG_LIKELIHOOD_BLOCK_PATTERN.match(name)
    if match is None:
        return None
    return "log_likelihood", int(match.group(2))


def _factor_labels_from_combined_idata(idata: xr.DataTree) -> list[str]:
    labels: set[int] = set()

    try:
        posterior = _require_dataset(idata, "posterior")
    except (KeyError, TypeError):
        posterior = None
    if posterior is not None:
        for name in posterior.data_vars:
            idx = _posterior_factor_index(str(name))
            if idx is not None:
                labels.add(idx)

    try:
        sample_stats = _require_dataset(idata, "sample_stats")
    except (KeyError, TypeError):
        sample_stats = None
    if sample_stats is not None:
        for name in sample_stats.data_vars:
            key = _sample_stats_factor_key(str(name))
            if key is not None:
                labels.add(key[1])

    return [str(idx) for idx in sorted(labels)]


def _copy_factor_attrs(idata: xr.DataTree, factor: str) -> dict[str, Any]:
    attrs = dict(getattr(idata, "attrs", {}) or {})
    factor_idx = int(factor)

    max_tree_depth_by_channel = attrs.get("max_tree_depth_by_channel")
    if max_tree_depth_by_channel is not None:
        try:
            attrs["max_tree_depth"] = int(
                max_tree_depth_by_channel[factor_idx]
            )
        except (IndexError, TypeError, ValueError):
            pass

    target_accept_by_channel = attrs.get("target_accept_prob_by_channel")
    if target_accept_by_channel is not None:
        try:
            attrs["target_accept_prob"] = float(
                target_accept_by_channel[factor_idx]
            )
        except (IndexError, TypeError, ValueError):
            pass

    attrs["factor"] = factor
    return attrs


def _factor_tree_from_combined_idata(
    idata: xr.DataTree, factor: str
) -> xr.DataTree:
    factor_idx = int(factor)
    factor_tree = xr.DataTree()
    factor_tree.attrs.update(_copy_factor_attrs(idata, factor))

    try:
        posterior = _require_dataset(idata, "posterior")
    except (KeyError, TypeError):
        posterior = None
    if posterior is not None:
        posterior_vars = {
            str(name): var
            for name, var in posterior.data_vars.items()
            if _posterior_factor_index(str(name)) == factor_idx
        }
        if posterior_vars:
            factor_tree["posterior"] = xr.DataTree(
                dataset=xr.Dataset(posterior_vars, attrs=posterior.attrs)
            )

    try:
        sample_stats = _require_dataset(idata, "sample_stats")
    except (KeyError, TypeError):
        sample_stats = None
    if sample_stats is not None:
        sample_stats_vars = {}
        for name, var in sample_stats.data_vars.items():
            key = _sample_stats_factor_key(str(name))
            if key is None:
                continue
            base, idx = key
            if idx == factor_idx:
                sample_stats_vars[base] = var
        if sample_stats_vars:
            factor_tree["sample_stats"] = xr.DataTree(
                dataset=xr.Dataset(sample_stats_vars, attrs=sample_stats.attrs)
            )

    try:
        log_likelihood = _require_dataset(idata, "log_likelihood")
    except (KeyError, TypeError):
        log_likelihood = None
    if log_likelihood is not None:
        ll_vars = {}
        for name, var in log_likelihood.data_vars.items():
            key = _log_likelihood_factor_key(str(name))
            if key is None:
                continue
            base, idx = key
            if idx == factor_idx:
                ll_vars[base] = var
        if ll_vars:
            factor_tree["log_likelihood"] = xr.DataTree(
                dataset=xr.Dataset(ll_vars, attrs=log_likelihood.attrs)
            )

    return factor_tree


def factor_idatas(
    idata_or_factors: (
        xr.DataTree | Mapping[str, xr.DataTree] | Sequence[xr.DataTree]
    ),
) -> dict[str, xr.DataTree]:
    """Return a normalized ``factor -> DataTree`` mapping."""

    if isinstance(idata_or_factors, xr.DataTree):
        labels = _factor_labels_from_combined_idata(idata_or_factors)
        if not labels:
            return {"0": idata_or_factors}
        return {
            label: _factor_tree_from_combined_idata(idata_or_factors, label)
            for label in labels
        }

    if isinstance(idata_or_factors, Mapping):
        return {str(key): value for key, value in idata_or_factors.items()}

    if isinstance(idata_or_factors, Sequence):
        return {str(idx): value for idx, value in enumerate(idata_or_factors)}

    raise TypeError(
        "Expected a DataTree, factor mapping, or sequence of factor DataTrees."
    )


def vi_factor_idatas(
    idata_or_factors: (
        xr.DataTree | Mapping[str, xr.DataTree] | Sequence[xr.DataTree]
    ),
) -> dict[str, Any]:
    """Return per-factor VI objects using posterior/log_likelihood groups."""

    if not isinstance(idata_or_factors, xr.DataTree):
        return factor_idatas(idata_or_factors)

    idata = idata_or_factors
    labels = _factor_labels_from_combined_idata(idata)
    if not labels:
        labels = ["0"]

    out: dict[str, xr.DataTree] = {}
    vi_posterior = idata.children.get("vi_posterior")
    vi_log_likelihood = idata.children.get("vi_log_likelihood")
    vi_sample_stats = idata.children.get("vi_sample_stats")

    for label in labels:
        factor_idx = int(label)
        posterior_ds = None
        log_likelihood_ds = None

        if vi_posterior is not None and vi_posterior.dataset is not None:
            posterior_vars = {
                str(name): var
                for name, var in vi_posterior.dataset.data_vars.items()
                if _posterior_factor_index(str(name)) == factor_idx
            }
            if posterior_vars:
                posterior_ds = xr.Dataset(
                    posterior_vars, attrs=vi_posterior.dataset.attrs
                )

        if (
            vi_log_likelihood is not None
            and vi_log_likelihood.dataset is not None
        ):
            ll_vars = {}
            for name, var in vi_log_likelihood.dataset.data_vars.items():
                key = _log_likelihood_factor_key(str(name))
                if key is None:
                    continue
                _, idx = key
                if idx == factor_idx:
                    ll_vars["log_likelihood"] = var
            if ll_vars:
                log_likelihood_ds = xr.Dataset(
                    ll_vars, attrs=vi_log_likelihood.dataset.attrs
                )

        if posterior_ds is not None and log_likelihood_ds is not None:
            out[label] = from_dict(
                {
                    "posterior": posterior_ds,
                    "log_likelihood": log_likelihood_ds,
                },
                attrs=_copy_factor_attrs(idata, label),
            )
        else:
            factor_tree = xr.DataTree()
            factor_tree.attrs.update(_copy_factor_attrs(idata, label))
            if posterior_ds is not None:
                factor_tree["posterior"] = xr.DataTree(dataset=posterior_ds)
            if log_likelihood_ds is not None:
                factor_tree["log_likelihood"] = xr.DataTree(
                    dataset=log_likelihood_ds
                )
            if (
                vi_sample_stats is not None
                and vi_sample_stats.dataset is not None
            ):
                factor_tree["vi_sample_stats"] = xr.DataTree(
                    dataset=vi_sample_stats.dataset
                )
            out[label] = factor_tree

    return out
