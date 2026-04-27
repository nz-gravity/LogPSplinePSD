"""Factory function for building an InferencePipeline from data."""

from __future__ import annotations

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..preprocessing.checks import _save_preprocessing_plot
from .config import PipelineConfig
from .model_kwargs import _joint_multivar_model, build_model_kwargs_and_spline
from .models import bayesian_model
from .pipeline import InferencePipeline
from .preprocessing import coarse_vi_freq_domain, preprocess_to_freq_domain
from .stages import (
    FactorizedMultivarNUTSStage,
    FactorizedMultivarVIStage,
    NUTSStage,
    VIStage,
)


def make_pipeline(
    data,
    config: PipelineConfig | None = None,
) -> InferencePipeline:
    """Build an InferencePipeline from data and config.

    Parameters
    ----------
    data:
        Time-domain (``Timeseries`` / ``MultivariateTimeseries``) or
        pre-processed frequency-domain (``Periodogram`` / ``MultivarFFT``).
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

    if not isinstance(data, Periodogram | MultivarFFT):
        data = preprocess_to_freq_domain(data, config)

    model_fn = (
        bayesian_model
        if isinstance(data, Periodogram)
        else _joint_multivar_model
    )

    full_kwargs, spline_model = build_model_kwargs_and_spline(
        data,
        config,
        coarse=False,
    )
    if isinstance(data, MultivarFFT) and config.outdir is not None:
        _save_preprocessing_plot(data, config, spline_model=spline_model)

    coarse_data = (
        coarse_vi_freq_domain(data, config)
        if config.init_from_vi and config.use_coarse_vi_for_init
        else None
    )
    coarse_kwargs = (
        build_model_kwargs_and_spline(coarse_data, config, coarse=True)[0]
        if coarse_data is not None
        else None
    )

    eta = float(config.eta)
    vi_stage = (
        VIStage(
            steps=config.vi_steps,
            lr=config.vi_lr,
            guide=config.vi_guide or "diag",
            posterior_draws=config.vi_posterior_draws,
            eta=eta,
        )
        if isinstance(data, Periodogram)
        else FactorizedMultivarVIStage(
            steps=config.vi_steps,
            lr=config.vi_lr,
            guide=config.vi_guide or "diag",
            posterior_draws=config.vi_posterior_draws,
            eta=eta,
        )
    )
    nuts_stage = (
        NUTSStage(
            n_samples=config.n_samples,
            n_warmup=config.n_warmup,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            num_chains=config.num_chains,
            eta=eta,
        )
        if isinstance(data, Periodogram)
        else FactorizedMultivarNUTSStage(
            n_samples=config.n_samples,
            n_warmup=config.n_warmup,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            num_chains=config.num_chains,
            eta=eta,
            target_accept_prob_by_channel=config.target_accept_prob_by_channel,
            max_tree_depth_by_channel=config.max_tree_depth_by_channel,
        )
    )

    return InferencePipeline(
        model_fn=model_fn,
        full_model_kwargs=full_kwargs,
        coarse_model_kwargs=coarse_kwargs,
        data=data,
        spline_model=spline_model,
        config=config,
        vi_stage=vi_stage,
        nuts_stage=nuts_stage,
        rng_key=config.rng_key,
        verbose=config.verbose,
        vi_progress_bar=config.vi_progress_bar,
        only_vi=config.only_vi,
        init_from_vi=config.init_from_vi,
        vi_coarse_only=config.vi_coarse_only,
    )
