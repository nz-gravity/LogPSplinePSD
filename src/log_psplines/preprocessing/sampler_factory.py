from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Union

import numpy as np

from ..datatypes import Periodogram
from ..datatypes.multivar import EmpiricalPSD, MultivarFFT
from ..logger import logger
from ..psplines import LogPSplines, MultivariateLogPSplines
from ..samplers import (
    MultivarBlockedNUTSConfig,
    MultivarBlockedNUTSSampler,
    NUTSConfig,
    NUTSSampler,
)
from .configs import RunMCMCConfig, SamplerFactoryConfig, SamplerName


def _build_model_from_data(
    processed_data: Union[Periodogram, MultivarFFT],
    model_config,
):
    if isinstance(processed_data, Periodogram):
        return LogPSplines.from_periodogram(
            processed_data,
            n_knots=model_config.n_knots,
            degree=model_config.degree,
            diffMatrixOrder=model_config.diffMatrixOrder,
            parametric_model=model_config.parametric_model,
            knot_kwargs=model_config.knot_kwargs,
        )
    if isinstance(processed_data, MultivarFFT):
        return MultivariateLogPSplines.from_multivar_fft(
            processed_data,
            n_knots=model_config.n_knots,
            degree=model_config.degree,
            diffMatrixOrder=model_config.diffMatrixOrder,
            knot_kwargs=model_config.knot_kwargs,
        )
    raise ValueError(
        f"Unsupported processed data type: {type(processed_data)}."
    )


def _build_sampler_inputs(
    processed_data: Union[Periodogram, MultivarFFT],
    config: RunMCMCConfig,
    sampler_type: SamplerName,
    scaled_true_psd: np.ndarray | None,
    extra_empirical_psd: list[EmpiricalPSD] | None,
    extra_empirical_labels: list[str] | None,
    extra_empirical_styles: list[dict] | None,
) -> SamplerFactoryConfig:
    scaling_factor = (
        processed_data.scaling_factor
        if hasattr(processed_data, "scaling_factor")
        else 1.0
    )
    channel_stds = (
        processed_data.channel_stds
        if hasattr(processed_data, "channel_stds")
        else None
    )
    return SamplerFactoryConfig(
        sampler_type=sampler_type,
        run_config=config,
        scaling_factor=float(scaling_factor or 1.0),
        true_psd=scaled_true_psd,
        channel_stds=channel_stds,
        extra_empirical_psd=extra_empirical_psd,
        extra_empirical_labels=extra_empirical_labels,
        extra_empirical_styles=extra_empirical_styles,
    )


def _build_common_sampler_kwargs(
    config: SamplerFactoryConfig,
) -> dict[str, Any]:
    run = config.run_config
    return {
        "alpha_phi": run.alpha_phi,
        "beta_phi": run.beta_phi,
        "alpha_delta": run.alpha_delta,
        "beta_delta": run.beta_delta,
        "num_chains": run.num_chains,
        "chain_method": run.chain_method,
        "rng_key": run.rng_key,
        "verbose": run.diagnostics.verbose,
        "outdir": run.diagnostics.outdir,
        "compute_psis": True,
        "compute_lnz": run.diagnostics.compute_lnz,
        "scaling_factor": config.scaling_factor,
        "channel_stds": config.channel_stds,
        "true_psd": config.true_psd,
        "vi_psd_max_draws": run.vi.vi_psd_max_draws,
        "only_vi": run.vi.only_vi,
        "extra_empirical_psd": config.extra_empirical_psd,
        "extra_empirical_labels": config.extra_empirical_labels,
        "extra_empirical_styles": config.extra_empirical_styles,
    }


def _validate_sampler_selection(
    data: Union[Periodogram, MultivarFFT],
    sampler_type: SamplerName,
    verbose: bool,
) -> SamplerName:
    if isinstance(data, Periodogram):
        if sampler_type != "nuts":
            raise ValueError(
                f"Unknown sampler_type '{sampler_type}' for univariate data. Choose 'nuts'."
            )
        return sampler_type

    allowed_types = {"nuts", "multivar_blocked_nuts"}
    if sampler_type not in allowed_types:
        if verbose:
            allowed = ", ".join(sorted(allowed_types))
            logger.warning(
                f"Multivariate analysis supports {allowed}. Using NUTS instead of {sampler_type}"
            )
        sampler_type = "nuts"

    if sampler_type == "nuts":
        if verbose:
            logger.info(
                "Mapping multivariate sampler 'nuts' to 'multivar_blocked_nuts'."
            )
        return "multivar_blocked_nuts"
    return sampler_type


def _build_univar_sampler(
    data: Periodogram,
    model,
    sampler_type: SamplerName,
    config: SamplerFactoryConfig,
    common_kwargs: dict[str, Any],
):
    run = config.run_config
    if sampler_type != "nuts":
        raise ValueError(
            f"Unknown sampler_type '{sampler_type}' for univariate data. Choose 'nuts'."
        )

    nuts_extra_kwargs = _validate_extra_kwargs(NUTSConfig, run.extra_kwargs)
    nuts_config = NUTSConfig(
        **common_kwargs,
        target_accept_prob=run.nuts.target_accept_prob,
        max_tree_depth=run.nuts.max_tree_depth,
        dense_mass=run.nuts.dense_mass,
        init_from_vi=run.vi.init_from_vi,
        vi_steps=run.vi.vi_steps,
        vi_lr=run.vi.vi_lr,
        vi_guide=run.vi.vi_guide,
        vi_posterior_draws=run.vi.vi_posterior_draws,
        vi_progress_bar=run.vi.vi_progress_bar,
        **nuts_extra_kwargs,
    )
    return NUTSSampler(data, model, nuts_config)


def _build_multivar_blocked_sampler(
    data: MultivarFFT,
    model,
    config: SamplerFactoryConfig,
    common_kwargs: dict[str, Any],
):
    run = config.run_config
    blocked_extra_kwargs = _validate_extra_kwargs(
        MultivarBlockedNUTSConfig, run.extra_kwargs
    )
    blocked_config = MultivarBlockedNUTSConfig(
        **common_kwargs,
        target_accept_prob=run.nuts.target_accept_prob,
        target_accept_prob_by_channel=run.nuts.target_accept_prob_by_channel,
        max_tree_depth=run.nuts.max_tree_depth,
        max_tree_depth_by_channel=run.nuts.max_tree_depth_by_channel,
        dense_mass=run.nuts.dense_mass,
        init_from_vi=run.vi.init_from_vi,
        vi_steps=run.vi.vi_steps,
        vi_lr=run.vi.vi_lr,
        vi_guide=run.vi.vi_guide,
        vi_posterior_draws=run.vi.vi_posterior_draws,
        vi_progress_bar=run.vi.vi_progress_bar,
        alpha_phi_theta=run.nuts.alpha_phi_theta,
        beta_phi_theta=run.nuts.beta_phi_theta,
        **blocked_extra_kwargs,
    )
    return MultivarBlockedNUTSSampler(data, model, blocked_config)


def _validate_extra_kwargs(
    config_cls: type,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if not extra_kwargs:
        return {}
    if not is_dataclass(config_cls):
        return dict(extra_kwargs)

    allowed = {item.name for item in fields(config_cls)}
    unknown = sorted(set(extra_kwargs) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported keyword arguments for {config_cls.__name__}: {unknown}"
        )
    return dict(extra_kwargs)


def _create_sampler(
    data: Union[Periodogram, MultivarFFT],
    model,
    config: SamplerFactoryConfig,
):
    """Factory function to create sampler instances from a config object."""
    sampler_type = _validate_sampler_selection(
        data,
        config.sampler_type,
        config.run_config.diagnostics.verbose,
    )
    common_kwargs = _build_common_sampler_kwargs(config)

    if isinstance(data, Periodogram):
        return _build_univar_sampler(
            data,
            model,
            sampler_type,
            config,
            common_kwargs,
        )

    if sampler_type == "multivar_blocked_nuts":
        return _build_multivar_blocked_sampler(
            data,
            model,
            config,
            common_kwargs,
        )
    raise ValueError(
        f"Unknown sampler_type '{sampler_type}' for multivariate data. Choose 'multivar_blocked_nuts'."
    )
