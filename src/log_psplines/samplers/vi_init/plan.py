"""Warm-start planning utilities for sampler-level VI initialisation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, cast

import numpy as np

from ...datatypes import Periodogram
from ...datatypes.multivar import MultivarFFT
from ...psplines import LogPSplines, MultivariateLogPSplines


@dataclass(frozen=True)
class VIWarmStartPlan:
    """Describe sampler warm-start inputs without carrying a sampler instance."""

    strategy: str
    processed_data: Periodogram | MultivarFFT
    scaled_true_psd: np.ndarray | None
    metadata: dict[str, Any]
    model_n_knots: Any
    model_degree: int
    model_diff_matrix_order: int
    model_knot_kwargs: dict[str, Any]
    model_parametric_model: Any
    model_analytical_psd: Any


def build_coarse_sampler_from_plan(
    sampler,
    warm_start_plan: VIWarmStartPlan,
):
    """Build a transient coarse sampler from a warm-start plan."""
    if warm_start_plan.strategy != "coarse_vi":
        raise ValueError(
            "Only coarse_vi warm-start plans are currently supported."
        )

    processed_data = warm_start_plan.processed_data
    if isinstance(processed_data, Periodogram):
        n_knots = int(warm_start_plan.model_n_knots)
        coarse_model = LogPSplines.from_periodogram(
            processed_data,
            n_knots=n_knots,
            degree=warm_start_plan.model_degree,
            diffMatrixOrder=warm_start_plan.model_diff_matrix_order,
            parametric_model=warm_start_plan.model_parametric_model,
            knot_kwargs=dict(warm_start_plan.model_knot_kwargs),
        )
    elif isinstance(processed_data, MultivarFFT):
        n_knots = warm_start_plan.model_n_knots
        if isinstance(n_knots, dict):
            n_knots = cast(Mapping[object, object], n_knots)
        coarse_model = MultivariateLogPSplines.from_multivar_fft(
            processed_data,
            n_knots=n_knots,
            degree=warm_start_plan.model_degree,
            diffMatrixOrder=warm_start_plan.model_diff_matrix_order,
            knot_kwargs=dict(warm_start_plan.model_knot_kwargs),
            analytical_psd=warm_start_plan.model_analytical_psd,
        )
    else:
        raise TypeError(
            "Unsupported processed_data type in warm-start plan: "
            f"{type(processed_data).__name__}."
        )

    coarse_scaling = float(
        getattr(
            processed_data, "scaling_factor", sampler.config.scaling_factor
        )
        or 1.0
    )
    coarse_channel_stds = getattr(processed_data, "channel_stds", None)

    coarse_config = replace(
        sampler.config,
        scaling_factor=coarse_scaling,
        true_psd=warm_start_plan.scaled_true_psd,
        channel_stds=coarse_channel_stds,
    )
    return type(sampler)(processed_data, coarse_model, coarse_config)


__all__ = ["VIWarmStartPlan", "build_coarse_sampler_from_plan"]
