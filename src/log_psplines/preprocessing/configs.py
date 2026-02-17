from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from .._jaxtypes import Complex, Float
from .._typecheck import runtime_typecheck
from ..datatypes import Periodogram
from ..datatypes.multivar import (
    EmpiricalPSD,
    MultivarFFT,
    MultivariateTimeseries,
)
from ..datatypes.multivar_utils import _interp_frequency_indexed_array
from ..datatypes.univar import Timeseries
from ..logger import logger
from ..psplines import LogPSplines, MultivariateLogPSplines
from ..samplers import (
    MultivarBlockedNUTSConfig,
    MultivarBlockedNUTSSampler,
    NUTSConfig,
    NUTSSampler,
)
from .coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    apply_coarse_graining_univar,
    compute_binning_structure,
)

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


@dataclass(frozen=True)
class ModelConfig:
    n_knots: int = 10
    degree: int = 3
    diffMatrixOrder: int = 2
    knot_kwargs: dict[str, Any] = field(default_factory=dict)
    parametric_model: Optional[jnp.ndarray] = None
    true_psd: TruePSDInput = None
    fmin: Optional[float] = None
    fmax: Optional[float] = None


@dataclass(frozen=True)
class DiagnosticsConfig:
    verbose: bool = True
    outdir: Optional[str] = None
    compute_lnz: bool = False


@dataclass(frozen=True)
class VIConfig:
    only_vi: bool = False
    init_from_vi: bool = True
    vi_steps: int = 1500
    vi_lr: float = 1e-2
    vi_guide: Optional[str] = None
    vi_posterior_draws: int = 256
    vi_progress_bar: Optional[bool] = None
    vi_psd_max_draws: int = 64


@dataclass(frozen=True)
class NUTSConfigOverride:
    target_accept_prob: float = 0.8
    target_accept_prob_by_channel: Optional[list[float]] = None
    max_tree_depth: int = 10
    max_tree_depth_by_channel: Optional[list[int]] = None
    dense_mass: bool = True
    alpha_phi_theta: Optional[float] = None
    beta_phi_theta: Optional[float] = None


@dataclass(frozen=True)
class RunMCMCConfig:
    """MCMC and data-processing configuration.

    Sampler type is automatically inferred from input data:
    - Timeseries (1D) → 'nuts'
    - MultivariateTimeseries (P-D) → 'multivar_blocked_nuts'
    """

    n_samples: int = 1000
    n_warmup: int = 500
    num_chains: int = 1
    chain_method: Optional[Literal["parallel", "vectorized", "sequential"]] = (
        None
    )
    alpha_phi: float = 1.0
    beta_phi: float = 1.0
    alpha_delta: float = 1e-4
    beta_delta: float = 1e-4
    rng_key: int = 42
    coarse_grain_config: Optional[CoarseGrainConfig | dict] = None
    Nb: int = 1
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    welch_window: str = "hann"
    model: ModelConfig = field(default_factory=ModelConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    vi: VIConfig = field(default_factory=VIConfig)
    nuts: NUTSConfigOverride = field(default_factory=NUTSConfigOverride)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerFactoryConfig:
    sampler_type: SamplerName
    run_config: RunMCMCConfig
    scaling_factor: float
    true_psd: Optional[np.ndarray]
    channel_stds: Optional[np.ndarray]
    extra_empirical_psd: list[EmpiricalPSD] | None = None
    extra_empirical_labels: list[str] | None = None
    extra_empirical_styles: list[dict] | None = None
