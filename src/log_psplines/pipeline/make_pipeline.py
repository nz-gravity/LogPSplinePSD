"""Factory function for building an InferencePipeline from data."""

from __future__ import annotations

from math import ceil
from typing import Optional, Union

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from .config import PipelineConfig
from .model_kwargs import _joint_multivar_model, build_model_kwargs
from .models import bayesian_model
from .pipeline import InferencePipeline
from .stages import NUTSStage, VIStage


def _preprocess_to_freq_domain(
    data,
    config: PipelineConfig,
) -> Union[Periodogram, MultivarFFT]:
    """Convert time-domain data to frequency-domain using config settings."""
    from ..datatypes.multivar import MultivariateTimeseries
    from ..datatypes.univar import Timeseries
    from ..preprocessing.data_prep import (
        _coarse_grain_processed_data,
        _normalize_coarse_grain_config,
        _normalize_excluded_frequency_bands,
    )
    from ..preprocessing.preprocessing import _apply_frequency_exclusion

    standardized = data.standardise_for_psd()
    if isinstance(data, Timeseries):
        freq_data = standardized.to_periodogram(
            fmin=config.fmin, fmax=config.fmax
        )
    else:
        freq_data = standardized.to_wishart_stats(
            Nb=config.Nb,
            fmin=config.fmin,
            fmax=config.fmax,
            window=config.wishart_window,
            detrend=config.wishart_detrend,
            wishart_floor_fraction=config.wishart_floor_fraction,
        )

    cg_config = _normalize_coarse_grain_config(config.coarse_grain_config)
    freq_data, _ = _coarse_grain_processed_data(freq_data, cg_config, None)

    excl_bands = _normalize_excluded_frequency_bands(config.exclude_freq_bands)
    if excl_bands:
        freq_data = _apply_frequency_exclusion(freq_data, excl_bands)

    return freq_data


def _coarse_vi_freq_domain(
    data: Union[Periodogram, MultivarFFT],
    config: PipelineConfig,
) -> Optional[Union[Periodogram, MultivarFFT]]:
    """Return coarse-grained data for VI warm-start, or None."""
    if not config.auto_coarse_vi:
        return None

    from ..preprocessing.coarse_grain import (
        CoarseGrainConfig,
        _smallest_divisor_geq,
    )
    from ..preprocessing.data_prep import _coarse_grain_processed_data

    full_nfreq = (
        len(data.freqs) if isinstance(data, Periodogram) else len(data.freq)
    )
    n_knots = (
        config.n_knots
        if isinstance(config.n_knots, int)
        else max(config.n_knots.values())
    )
    k_basis = int(n_knots + config.degree - 1)
    min_required = max(
        1, int(config.auto_coarse_vi_min_full_nfreq), 20 * k_basis
    )
    if full_nfreq < min_required:
        return None

    target = max(1, min(config.auto_coarse_vi_target_nfreq, full_nfreq - 1))
    min_nh = max(2, int(ceil(full_nfreq / float(target))))
    nh = _smallest_divisor_geq(full_nfreq, min_nh)
    if nh is None or nh <= 1:
        return None

    cg = CoarseGrainConfig(enabled=True, Nc=None, Nh=nh)
    coarse_data, _ = _coarse_grain_processed_data(data, cg, None)
    return coarse_data


def make_pipeline(
    data,
    config: Optional[PipelineConfig] = None,
) -> InferencePipeline:
    """Build an InferencePipeline from data and config.

    Parameters
    ----------
    data:
        Time-domain (``Timeseries`` / ``MultivariateTimeseries``) or
        pre-processed frequency-domain (``Periodogram`` / ``MultivarFFT``) data.
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

    if not isinstance(data, (Periodogram, MultivarFFT)):
        data = _preprocess_to_freq_domain(data, config)

    model_fn = (
        bayesian_model
        if isinstance(data, Periodogram)
        else _joint_multivar_model
    )

    full_kwargs = build_model_kwargs(data, config, coarse=False)

    coarse_data = _coarse_vi_freq_domain(data, config)
    coarse_kwargs = (
        build_model_kwargs(coarse_data, config, coarse=True)
        if coarse_data is not None
        else None
    )

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
