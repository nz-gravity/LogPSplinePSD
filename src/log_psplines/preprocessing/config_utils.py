from __future__ import annotations

from typing import Any

from ..logger import logger
from .configs import (
    DiagnosticsConfig,
    ModelConfig,
    NUTSConfigOverride,
    RunMCMCConfig,
    VIConfig,
)


def _normalize_run_config(config: RunMCMCConfig | None) -> RunMCMCConfig:
    if config is None:
        return RunMCMCConfig()
    if not isinstance(config, RunMCMCConfig):
        raise TypeError("config must be a RunMCMCConfig instance or None.")
    return config


def _build_config_from_kwargs(**kwargs) -> RunMCMCConfig:
    """
    Route kwargs to appropriate config classes.

    Accepts legacy-style kwargs and constructs RunMCMCConfig with nested
    ModelConfig, VIConfig, DiagnosticsConfig, and NUTSConfigOverride.

    Note: 'sampler' is automatically inferred from input data type
    (univariate → 'nuts', multivariate → 'multivar_blocked_nuts').
    If 'sampler' is provided in kwargs, it will be ignored with a warning.
    """
    # Map of kwarg names to their target config classes and fields
    model_fields = {
        "n_knots",
        "degree",
        "diffMatrixOrder",
        "knot_kwargs",
        "fmin",
        "fmax",
        "true_psd",
        "parametric_model",
    }
    nuts_fields = {
        "target_accept_prob",
        "target_accept_prob_by_channel",
        "max_tree_depth",
        "max_tree_depth_by_channel",
        "dense_mass",
        "alpha_phi_theta",
        "beta_phi_theta",
    }
    vi_fields = {
        "init_from_vi",
        "vi_steps",
        "vi_guide",
        "vi_psd_max_draws",
        "vi_lr",
        "vi_posterior_draws",
        "only_vi",
        "vi_progress_bar",
    }
    diagnostics_fields = {
        "verbose",
        "outdir",
        "compute_lnz",
    }
    run_mcmc_fields = {
        "n_samples",
        "n_warmup",
        "num_chains",
        "chain_method",
        "rng_key",
        "alpha_phi",
        "beta_phi",
        "alpha_delta",
        "beta_delta",
        "coarse_grain_config",
        "Nb",
        "welch_nperseg",
        "welch_noverlap",
        "welch_window",
    }
    sampler_config_fields = {
        "posterior_psd_max_draws",
        "compute_coherence_quantiles",
    }

    # Warn if user provided 'sampler' (now inferred automatically)
    if "sampler" in kwargs:
        logger.warning(
            "The 'sampler' parameter is now inferred automatically from input data type "
            "(univariate → 'nuts', multivariate → 'multivar_blocked_nuts'). "
            "Your provided value will be ignored."
        )

    # Build config dicts by routing kwargs directly (no pre-renaming)
    model_dict = {k: v for k, v in kwargs.items() if k in model_fields}
    nuts_dict = {k: v for k, v in kwargs.items() if k in nuts_fields}
    vi_dict = {k: v for k, v in kwargs.items() if k in vi_fields}
    diagnostics_dict = {
        k: v for k, v in kwargs.items() if k in diagnostics_fields
    }
    run_mcmc_dict = {k: v for k, v in kwargs.items() if k in run_mcmc_fields}
    sampler_dict = {
        k: v for k, v in kwargs.items() if k in sampler_config_fields
    }

    # Collect unknown kwargs for extra_kwargs
    routed = (
        model_fields
        | nuts_fields
        | vi_fields
        | diagnostics_fields
        | run_mcmc_fields
        | sampler_config_fields
    )
    extra_kwargs = {
        k: v for k, v in kwargs.items() if k not in routed and k != "sampler"
    }
    # Add sampler-specific fields to extra_kwargs so they reach the sampler config
    extra_kwargs.update(sampler_dict)

    run_mcmc_dict["extra_kwargs"] = extra_kwargs

    return RunMCMCConfig(
        model=ModelConfig(**model_dict),
        nuts=NUTSConfigOverride(**nuts_dict),
        vi=VIConfig(**vi_dict),
        diagnostics=DiagnosticsConfig(**diagnostics_dict),
        **run_mcmc_dict,
    )
