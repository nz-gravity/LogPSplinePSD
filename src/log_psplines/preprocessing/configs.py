from __future__ import annotations

import warnings
from abc import ABCMeta
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from ..pipeline.config import PipelineConfig
from .coarse_grain import CoarseGrainConfig

SamplerName = Literal[
    "nuts",
    "multivar_blocked_nuts",
]
TruePSDInput = Union[
    None,
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    list,
    dict,
]
FrequencyBand = Tuple[float, float]


@dataclass(frozen=True)
class ModelConfig:
    n_knots: int | dict[str, int] = 10
    degree: int = 3
    diffMatrixOrder: int = 2
    knot_kwargs: dict[str, Any] = field(default_factory=dict)
    parametric_model: Optional[jnp.ndarray] = None
    analytical_psd: Optional[np.ndarray] = None
    true_psd: TruePSDInput = None
    fmin: Optional[float] = None
    fmax: Optional[float] = None
    exclude_freq_bands: tuple[FrequencyBand, ...] = field(
        default_factory=tuple
    )


@dataclass(frozen=True)
class DiagnosticsConfig:
    verbose: bool = True
    outdir: Optional[str] = None
    compute_lnz: Optional[bool] = None


@dataclass(frozen=True)
class VIConfig:
    only_vi: bool = False
    init_from_vi: bool = True
    vi_steps: int = 1500
    vi_lr: float = 1e-2
    vi_guide: Optional[str] = None
    vi_posterior_draws: int = 50
    vi_progress_bar: Optional[bool] = None
    vi_psd_max_draws: int = 50
    coarse_grain_config_vi: Optional[CoarseGrainConfig | dict] = None
    auto_coarse_vi: bool = False
    auto_coarse_vi_target_nfreq: int = 192
    auto_coarse_vi_min_full_nfreq: int = 512
    use_coarse_vi_for_init: bool = True
    vi_coarse_only: bool = False


@dataclass(frozen=True)
class NUTSConfigOverride:
    target_accept_prob: float = 0.8
    target_accept_prob_by_channel: Optional[list[float]] = None
    max_tree_depth: int = 10
    max_tree_depth_by_channel: Optional[list[int]] = None
    dense_mass: bool = True
    alpha_phi_theta: Optional[float] = None
    beta_phi_theta: Optional[float] = None
    design_from_vi: bool = False
    design_from_vi_tau: float = 10.0


def _merge_legacy_config(
    pipeline_kwargs: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        source = dict(value)
    elif is_dataclass(value):
        source = {
            item.name: getattr(value, item.name) for item in fields(value)
        }
    else:
        raise TypeError(
            f"{key} must be a dataclass instance, dict, or None; "
            f"got {type(value).__name__}."
        )
    for field_name, field_value in source.items():
        if field_name in pipeline_kwargs:
            raise TypeError(
                f"Received both '{field_name}' and legacy '{key}' values."
            )
        pipeline_kwargs[field_name] = field_value


def _build_pipeline_config_from_legacy(
    *args: Any,
    **kwargs: Any,
) -> PipelineConfig:
    if args:
        raise TypeError(
            "RunMCMCConfig no longer accepts positional arguments."
        )

    pipeline_kwargs = dict(kwargs)
    _merge_legacy_config(
        pipeline_kwargs, "model", pipeline_kwargs.pop("model", None)
    )
    _merge_legacy_config(
        pipeline_kwargs,
        "diagnostics",
        pipeline_kwargs.pop("diagnostics", None),
    )
    _merge_legacy_config(
        pipeline_kwargs, "vi", pipeline_kwargs.pop("vi", None)
    )
    _merge_legacy_config(
        pipeline_kwargs, "nuts", pipeline_kwargs.pop("nuts", None)
    )

    allowed = {item.name for item in fields(PipelineConfig)}
    unknown = sorted(set(pipeline_kwargs) - allowed)
    if unknown:
        raise TypeError(
            f"Unexpected RunMCMCConfig keyword arguments: {unknown}"
        )

    return PipelineConfig(**pipeline_kwargs)


class RunMCMCConfig(metaclass=ABCMeta):
    """Deprecated compatibility shim for the flat ``PipelineConfig``."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "RunMCMCConfig is deprecated; use PipelineConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _build_pipeline_config_from_legacy(*args, **kwargs)


RunMCMCConfig.register(PipelineConfig)
