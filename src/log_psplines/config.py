from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np

from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

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

    method: Literal["nuts", "vi"] = "nuts"
    vi_steps: int = 1500
    vi_lr: float = 1e-2
    vi_guide: str | None = None
    vi_posterior_draws: int = 50
    vi_progress_bar: bool | None = None
    vi_psd_max_draws: int = 50

    target_accept_prob: float = 0.8
    target_accept_prob_by_channel: list[float] | None = None
    max_tree_depth: int = 10
    max_tree_depth_by_channel: list[int] | None = None
    dense_mass: bool = True
    alpha_phi_theta: float | None = None
    beta_phi_theta: float | None = None

    eta: float = 1.0

    extra_kwargs: dict[str, Any] = field(default_factory=dict)


__all__ = ["PipelineConfig", "PowerSplineConfig"]


@dataclass
class PowerSplineConfig:
    """WDM power/count prior and NUTS settings, separate from Wishart priors.

    ``phi_time`` and ``phi_freq`` posterior sites contain log precisions.
    Gamma parameters use the shape/rate convention of the source model.
    """

    alpha_phi: float = 2.0
    beta_phi: float = 1.0
    phi_log_base_scale: float = 1.0
    null_precision: float = 1e-4
    ridge_eps: float = 1e-6
    init_penalty_time: float = 0.05
    init_penalty_freq: float = 0.05
    centered: bool = False
    n_warmup: int = 250
    n_samples: int = 300
    num_chains: int = 1
    seed: int = 7
    max_tree_depth: int = 10
    target_accept_prob: float = 0.85
    progress_bar: bool = True

    def __post_init__(self) -> None:
        for name in (
            "alpha_phi",
            "beta_phi",
            "phi_log_base_scale",
            "null_precision",
            "ridge_eps",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("init_penalty_time", "init_penalty_freq"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("n_warmup", "n_samples", "num_chains", "max_tree_depth"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < (
                0 if name == "n_warmup" else 1
            ):
                raise ValueError(f"invalid {name}")
        if not 0 < self.target_accept_prob < 1:
            raise ValueError("target_accept_prob must lie in (0,1)")
