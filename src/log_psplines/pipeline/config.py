from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np

from ..preprocessing.coarse_grain import CoarseGrainConfig

TruePSDInput = Union[
    None,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    list,
    dict,
]
FrequencyBand = tuple[float, float]


@dataclass(frozen=True)
class PipelineConfig:
    """Flat configuration for preprocessing, VI, and MCMC pipeline setup."""

    n_samples: int = 1000
    n_warmup: int = 500
    num_chains: int = 1
    chain_method: Literal["parallel", "vectorized", "sequential"] | None = None
    alpha_phi: float = 1.0
    beta_phi: float = 1.0
    alpha_delta: float = 1e-4
    beta_delta: float = 1e-4
    rng_key: int = 42
    coarse_grain_config: CoarseGrainConfig | dict | None = None
    Nb: int = 1
    wishart_window: str | tuple | None = None
    wishart_detrend: str | bool = "constant"
    wishart_floor_fraction: float | None = None
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    welch_window: str = "hann"

    n_knots: int | dict[str, int] = 10
    degree: int = 3
    diffMatrixOrder: int = 2
    knot_kwargs: dict[str, Any] = field(default_factory=dict)
    analytical_psd: np.ndarray | None = None
    true_psd: TruePSDInput = None
    fmin: float | None = None
    fmax: float | None = None
    exclude_freq_bands: tuple[FrequencyBand, ...] = field(
        default_factory=tuple
    )

    verbose: bool = True
    outdir: str | None = None
    compute_lnz: bool | None = None

    only_vi: bool = False
    init_from_vi: bool = True
    vi_steps: int = 1500
    vi_lr: float = 1e-2
    vi_guide: str | None = None
    vi_posterior_draws: int = 50
    vi_progress_bar: bool | None = None
    vi_psd_max_draws: int = 50
    coarse_grain_config_vi: CoarseGrainConfig | dict | None = None
    auto_coarse_vi: bool = False
    auto_coarse_vi_target_nfreq: int = 192
    auto_coarse_vi_min_full_nfreq: int = 512
    use_coarse_vi_for_init: bool = True
    vi_coarse_only: bool = False

    target_accept_prob: float = 0.8
    target_accept_prob_by_channel: list[float] | None = None
    max_tree_depth: int = 10
    max_tree_depth_by_channel: list[int] | None = None
    dense_mass: bool = True
    alpha_phi_theta: float | None = None
    beta_phi_theta: float | None = None
    design_from_vi: bool = False
    design_from_vi_tau: float = 10.0

    eta: float = 1.0

    extra_kwargs: dict[str, Any] = field(default_factory=dict)


__all__ = ["PipelineConfig"]
