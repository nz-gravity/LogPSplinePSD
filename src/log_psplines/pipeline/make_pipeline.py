"""Factory function for building an InferencePipeline from pre-processed data."""

from __future__ import annotations

from typing import Optional, Union

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..samplers.univar.nuts import bayesian_model
from .config import PipelineConfig
from .model_kwargs import _joint_multivar_model, build_model_kwargs
from .pipeline import InferencePipeline
from .stages import NUTSStage, VIStage


def _coarse_model_kwargs(
    data: Union[Periodogram, MultivarFFT],
    config: PipelineConfig,
) -> Optional[dict]:
    """Return coarse model kwargs when auto_coarse_vi is enabled, else None."""
    if not config.auto_coarse_vi:
        return None

    # Import here to avoid a top-level circular dependency.
    from ..preprocessing.data_prep import (
        _coarse_grain_processed_data,
        _normalize_coarse_grain_config,
    )
    from ..preprocessing.preprocessing import _derive_vi_coarse_grain_config

    vi_cg_config, _ = _derive_vi_coarse_grain_config(data, config)
    if vi_cg_config is None or not vi_cg_config.enabled:
        return None

    coarse_data, _ = _coarse_grain_processed_data(
        data, _normalize_coarse_grain_config(vi_cg_config), None
    )
    if coarse_data is None:
        return None

    return build_model_kwargs(coarse_data, config, coarse=True)


def make_pipeline(
    data: Union[Periodogram, MultivarFFT],
    config: Optional[PipelineConfig] = None,
) -> InferencePipeline:
    """Build an InferencePipeline for pre-processed frequency-domain data.

    Parameters
    ----------
    data:
        Pre-processed data: ``Periodogram`` for univariate, ``MultivarFFT``
        for multivariate inference.
    config:
        Pipeline configuration.  Defaults to :class:`PipelineConfig` with all
        default values.

    Returns
    -------
    InferencePipeline
        Ready-to-run pipeline.  Call ``.run()`` to execute it.
    """
    if config is None:
        config = PipelineConfig()

    model_fn = (
        bayesian_model
        if isinstance(data, Periodogram)
        else _joint_multivar_model
    )

    full_kwargs = build_model_kwargs(data, config, coarse=False)
    coarse_kwargs = _coarse_model_kwargs(data, config)

    eta = float(config.eta)
    vi_stage = VIStage(
        steps=config.vi_steps,
        lr=config.vi_lr,
        guide=config.vi_guide or "diag",
        posterior_draws=config.vi_posterior_draws,
        eta=eta,
    )
    nuts_stage = NUTSStage(
        n_samples=config.n_samples,
        n_warmup=config.n_warmup,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
        dense_mass=config.dense_mass,
        num_chains=config.num_chains,
        eta=eta,
    )

    return InferencePipeline(
        model_fn=model_fn,
        full_model_kwargs=full_kwargs,
        coarse_model_kwargs=coarse_kwargs,
        vi_stage=vi_stage,
        nuts_stage=nuts_stage,
        rng_key=config.rng_key,
        verbose=config.verbose,
        only_vi=config.only_vi,
    )
