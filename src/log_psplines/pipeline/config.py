from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from ..preprocessing.coarse_grain import CoarseGrainConfig

TruePSDInput = Union[
    None,
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    list,
    dict,
]
FrequencyBand = Tuple[float, float]


@dataclass(frozen=True)
class PipelineConfig:
    """Flat configuration for preprocessing, VI, and MCMC pipeline setup."""

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
    wishart_window: Optional[str | tuple] = None
    wishart_detrend: str | bool = "constant"
    wishart_floor_fraction: Optional[float] = None
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    welch_window: str = "hann"

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

    verbose: bool = True
    outdir: Optional[str] = None
    compute_lnz: Optional[bool] = None

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

    target_accept_prob: float = 0.8
    target_accept_prob_by_channel: Optional[list[float]] = None
    max_tree_depth: int = 10
    max_tree_depth_by_channel: Optional[list[int]] = None
    dense_mass: bool = True
    alpha_phi_theta: Optional[float] = None
    beta_phi_theta: Optional[float] = None
    design_from_vi: bool = False
    design_from_vi_tau: float = 10.0

    eta: float = 1.0

    extra_kwargs: dict[str, Any] = field(default_factory=dict)


__all__ = ["PipelineConfig"]
