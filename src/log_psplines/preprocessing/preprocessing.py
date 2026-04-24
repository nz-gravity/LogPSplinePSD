from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Optional, Union

import numpy as np

from .._jaxtypes import Complex, Float
from .._typecheck import runtime_typecheck
from ..datatypes import Periodogram
from ..datatypes.multivar import EmpiricalPSD, MultivarFFT
from ..datatypes.multivar_utils import _interp_frequency_indexed_array
from ..logger import logger
from ..samplers.vi_init import VIWarmStartPlan
from .checks import _run_preprocessing_checks, _save_preprocessing_plot
from .coarse_grain import (
    CoarseGrainConfig,
    _closest_divisor,
    _smallest_divisor_geq,
)
from .configs import (
    DiagnosticsConfig,
    ModelConfig,
    NUTSConfigOverride,
    RunMCMCConfig,
    SamplerFactoryConfig,
    SamplerName,
    TruePSDInput,
    VIConfig,
)
from .data_prep import (
    _build_welch_overlay,
    _coarse_grain_processed_data,
    _normalize_coarse_grain_config,
    _normalize_excluded_frequency_bands,
    _prepare_processed_data,
)

__all__ = [
    "PreprocessedMCMCInput",
]


@dataclass(frozen=True)
class PreprocessedMCMCInput:
    processed_data: Union[Periodogram, MultivarFFT]
    scaled_true_psd: Optional[np.ndarray]
    sampler_type: SamplerName
    extra_empirical_psd: list[EmpiricalPSD] | None
    extra_empirical_labels: list[str] | None
    extra_empirical_styles: list[dict] | None
    run_config: RunMCMCConfig
    vi_warm_start_plan: Optional[VIWarmStartPlan] = None


def _normalize_run_config(config: RunMCMCConfig | None) -> RunMCMCConfig:
    if config is None:
        return RunMCMCConfig()
    if not isinstance(config, RunMCMCConfig):
        raise TypeError("config must be a RunMCMCConfig instance or None.")
    return config


def _build_config_from_kwargs(**kwargs) -> RunMCMCConfig:
    """Route legacy kwargs into the nested run configuration objects."""
    model_fields = {
        "n_knots",
        "degree",
        "diffMatrixOrder",
        "knot_kwargs",
        "fmin",
        "fmax",
        "exclude_freq_bands",
        "true_psd",
        "parametric_model",
        "analytical_psd",
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
        "coarse_grain_config_vi",
        "auto_coarse_vi",
        "auto_coarse_vi_target_nfreq",
        "auto_coarse_vi_min_full_nfreq",
        "use_coarse_vi_for_init",
        "vi_coarse_only",
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
        "wishart_window",
        "wishart_detrend",
        "wishart_floor_fraction",
        "welch_nperseg",
        "welch_noverlap",
        "welch_window",
    }
    sampler_config_fields = {
        "posterior_psd_max_draws",
        "compute_coherence_quantiles",
    }

    if "sampler" in kwargs:
        logger.warning(
            "The 'sampler' parameter is now inferred automatically from input data type "
            "(univariate -> 'nuts', multivariate -> 'multivar_blocked_nuts'). "
            "Your provided value will be ignored."
        )

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
    extra_kwargs.update(sampler_dict)
    run_mcmc_dict["extra_kwargs"] = extra_kwargs

    return RunMCMCConfig(
        model=ModelConfig(**model_dict),
        nuts=NUTSConfigOverride(**nuts_dict),
        vi=VIConfig(**vi_dict),
        diagnostics=DiagnosticsConfig(**diagnostics_dict),
        **run_mcmc_dict,
    )


def _unpack_true_psd(
    true_psd: TruePSDInput,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(freq, psd)`` from accepted ``true_psd`` formats."""
    if true_psd is None:
        return None, None

    if isinstance(true_psd, dict):
        freq = true_psd.get("freq")
        psd = true_psd.get("psd")
        if psd is None:
            raise ValueError(
                "true_psd dict must contain a 'psd' entry (optional 'freq')."
            )
        freq_arr = np.asarray(freq) if freq is not None else None
        return freq_arr, np.asarray(psd)

    if isinstance(true_psd, (tuple, list)) and len(true_psd) == 2:
        freq = np.asarray(true_psd[0]) if true_psd[0] is not None else None
        return freq, np.asarray(true_psd[1])

    return None, np.asarray(true_psd)


@runtime_typecheck
def _interp_psd_array(
    psd: Complex[np.ndarray, "f_src ..."] | Float[np.ndarray, "f_src ..."],
    freq_src: Float[np.ndarray, "f_src"],
    freq_tgt: Float[np.ndarray, "f_tgt"],
) -> Complex[np.ndarray, "f_tgt ..."] | Float[np.ndarray, "f_tgt ..."]:
    """Interpolate PSD arrays (real or complex) onto target frequencies."""
    return _interp_frequency_indexed_array(
        freq_src,
        freq_tgt,
        psd,
        sort_and_dedup=True,
    )


def _prepare_true_psd_for_freq(
    true_psd: TruePSDInput,
    freq_target: Optional[Float[np.ndarray, "f_tgt"]],
) -> Optional[np.ndarray]:
    """Resample supplied true PSD onto the target frequency grid."""
    if true_psd is None:
        return None

    freq_src, psd_array = _unpack_true_psd(true_psd)
    if psd_array is None:
        return None
    if freq_target is None:
        return np.asarray(psd_array)

    psd_array = np.asarray(psd_array)
    freq_target = np.asarray(freq_target)

    if freq_src is None:
        if psd_array.shape[0] == freq_target.size:
            return psd_array
        logger.warning(
            "true_psd length {} does not match target frequencies {}; assuming uniform spacing for interpolation.",
            psd_array.shape[0],
            freq_target.size,
        )
        freq_src = np.linspace(
            freq_target[0], freq_target[-1], psd_array.shape[0]
        )
    else:
        freq_src = np.asarray(freq_src)
        if freq_src.ndim != 1:
            raise ValueError(
                "true_psd frequency grid must be one-dimensional."
            )
        if freq_src.shape[0] != psd_array.shape[0]:
            raise ValueError(
                "true_psd frequency and value arrays must have matching lengths."
            )

    return _interp_psd_array(psd_array, freq_src, freq_target)


def _align_true_psd_to_freq(
    true_psd: TruePSDInput,
    processed_data,
) -> Optional[np.ndarray]:
    """Align ``true_psd`` to the frequency grid of the processed data."""
    if true_psd is None:
        return None
    if processed_data is None:
        _, psd = _unpack_true_psd(true_psd)
        return None if psd is None else np.asarray(psd)

    if isinstance(processed_data, Periodogram):
        freq_target = np.asarray(processed_data.freqs)
    elif isinstance(processed_data, MultivarFFT):
        freq_target = np.asarray(processed_data.freq)
    else:
        _, psd = _unpack_true_psd(true_psd)
        return None if psd is None else np.asarray(psd)

    aligned = _prepare_true_psd_for_freq(true_psd, freq_target)
    if aligned is not None:
        return aligned
    _, psd = _unpack_true_psd(true_psd)
    return None if psd is None else np.asarray(psd)


def _get_frequency_count(data: Union[Periodogram, MultivarFFT]) -> int:
    if isinstance(data, Periodogram):
        return int(len(data.freqs))
    return int(len(data.freq))


def _max_config_n_knots(n_knots: int | dict[str, int]) -> int:
    """Return the largest knot count implied by a model config."""
    if isinstance(n_knots, int):
        return int(n_knots)
    return max(int(value) for value in n_knots.values())


def _get_frequency_axis(
    data: Union[Periodogram, MultivarFFT],
) -> np.ndarray:
    if isinstance(data, Periodogram):
        return np.asarray(data.freqs, dtype=float)
    return np.asarray(data.freq, dtype=float)


def _apply_frequency_exclusion(
    data: Union[Periodogram, MultivarFFT],
    excl_bands: tuple[tuple[float, float], ...],
) -> Union[Periodogram, MultivarFFT]:
    """Remove excluded frequency bands from processed data before inference."""
    if not excl_bands:
        return data

    freq = _get_frequency_axis(data)
    keep = np.ones(freq.shape, dtype=bool)
    for f_lo, f_hi in excl_bands:
        keep &= ~((freq >= f_lo) & (freq <= f_hi))

    n_excl = int((~keep).sum())
    if n_excl == 0:
        return data

    logger.info(
        f"Null-band excision: removing {n_excl} bins across {len(excl_bands)} band(s). "
        f"{int(np.count_nonzero(keep))} bins retained."
    )
    return data.apply_mask(keep)


def _derive_vi_coarse_grain_config(
    processed_data: Union[Periodogram, MultivarFFT],
    run_config: RunMCMCConfig,
):
    vi_cfg = run_config.vi
    full_nfreq = _get_frequency_count(processed_data)
    max_n_knots = _max_config_n_knots(run_config.model.n_knots)
    k_basis = int(max_n_knots + run_config.model.degree - 1)

    explicit_cfg = _normalize_coarse_grain_config(
        vi_cfg.coarse_grain_config_vi
    )
    if explicit_cfg.enabled:
        metadata: dict[str, object] = {
            "coarse_vi_mode": "config",
            "coarse_vi_full_nfreq": full_nfreq,
        }

        if explicit_cfg.Nc is not None:
            requested_nc = int(explicit_cfg.Nc)
            if full_nfreq % requested_nc != 0:
                adjusted_nc = _closest_divisor(full_nfreq, requested_nc)
                logger.info(
                    "Adjusting explicit coarse VI bins: "
                    f"N_full={full_nfreq} not divisible by Nc={requested_nc}; "
                    f"using Nc={adjusted_nc}."
                )
                explicit_cfg = CoarseGrainConfig(
                    enabled=True,
                    Nc=adjusted_nc,
                    Nh=None,
                )
                metadata["coarse_vi_requested_Nc"] = requested_nc
                metadata["coarse_vi_adjusted_Nc"] = adjusted_nc
        elif explicit_cfg.Nh is not None:
            requested_nh = int(explicit_cfg.Nh)
            if full_nfreq % requested_nh != 0:
                adjusted_nh = _closest_divisor(full_nfreq, requested_nh)
                logger.info(
                    "Adjusting explicit coarse VI bin width: "
                    f"N_full={full_nfreq} not divisible by Nh={requested_nh}; "
                    f"using Nh={adjusted_nh}."
                )
                explicit_cfg = CoarseGrainConfig(
                    enabled=True,
                    Nc=None,
                    Nh=adjusted_nh,
                )
                metadata["coarse_vi_requested_Nh"] = requested_nh
                metadata["coarse_vi_adjusted_Nh"] = adjusted_nh

        return explicit_cfg, metadata

    if not vi_cfg.auto_coarse_vi:
        return None, None

    min_full_nfreq = int(vi_cfg.auto_coarse_vi_min_full_nfreq)
    min_required_nfreq = max(1, min_full_nfreq, 20 * k_basis)
    if full_nfreq < min_required_nfreq:
        return None, None

    target_nfreq = max(
        1, min(vi_cfg.auto_coarse_vi_target_nfreq, full_nfreq - 1)
    )
    min_nh = max(2, int(ceil(full_nfreq / float(target_nfreq))))
    nh = _smallest_divisor_geq(full_nfreq, min_nh)
    if nh is None or nh <= 1:
        return None, None
    target_nfreq = full_nfreq // nh

    return _normalize_coarse_grain_config(
        {
            "enabled": True,
            "Nc": None,
            "Nh": nh,
        }
    ), {
        "coarse_vi_mode": "auto",
        "coarse_vi_full_nfreq": full_nfreq,
        "coarse_vi_target_nfreq": target_nfreq,
        "coarse_vi_min_required_nfreq": min_required_nfreq,
        "coarse_vi_basis_target_floor": 10 * k_basis,
    }


def _build_coarse_vi_warm_start_plan_from_preprocessed(
    preproc_input: PreprocessedMCMCInput,
    *,
    vi_cg_config: CoarseGrainConfig,
    metadata: dict[str, object] | None,
) -> VIWarmStartPlan:
    """Build a coarse-VI context from the final analysis grid.

    This applies additional coarse graining to the already-preprocessed
    analysis data. Doing so keeps the coarse-VI grid consistent with any
    post-coarse-grain frequency exclusions already present in
    ``preproc_input.processed_data``.
    """
    coarse_processed_data, coarse_scaled_true_psd = (
        _coarse_grain_processed_data(
            preproc_input.processed_data,
            _normalize_coarse_grain_config(vi_cg_config),
            preproc_input.scaled_true_psd,
        )
    )
    assert (
        coarse_processed_data is not None
    ), "Coarse graining should preserve non-None processed data."
    coarse_scaled_true_psd = _align_true_psd_to_freq(
        coarse_scaled_true_psd,
        coarse_processed_data,
    )
    model_cfg = preproc_input.run_config.model
    parametric_model_arr: np.ndarray | None = None
    if model_cfg.parametric_model is not None:
        parametric_model_arr = np.asarray(model_cfg.parametric_model)

    analytical_psd_arr: np.ndarray | None = None
    if model_cfg.analytical_psd is not None:
        analytical_psd_arr = np.asarray(model_cfg.analytical_psd)

    return VIWarmStartPlan(
        strategy="coarse_vi",
        processed_data=coarse_processed_data,
        scaled_true_psd=coarse_scaled_true_psd,
        metadata={
            **(metadata or {}),
            "coarse_vi_attempted": 0,
            "coarse_vi_success": 0,
            "coarse_vi_nfreq": _get_frequency_count(coarse_processed_data),
        },
        model_n_knots=model_cfg.n_knots,
        model_degree=int(model_cfg.degree),
        model_diff_matrix_order=int(model_cfg.diffMatrixOrder),
        model_knot_kwargs=dict(model_cfg.knot_kwargs or {}),
        model_parametric_model=parametric_model_arr,
        model_analytical_psd=analytical_psd_arr,
    )


def _preprocess_with_run_config(
    data,
    run_config: RunMCMCConfig,
    *,
    include_overlays: bool,
) -> PreprocessedMCMCInput:
    """Preprocess data for one concrete run configuration."""
    fft_data, raw_ts, sampler_type = _prepare_processed_data(
        data,
        run_config,
    )
    scaled_true_psd = _align_true_psd_to_freq(
        run_config.model.true_psd,
        fft_data,
    )

    fft_data_cg, scaled_true_psd = _coarse_grain_processed_data(
        fft_data,
        _normalize_coarse_grain_config(run_config.coarse_grain_config),
        scaled_true_psd,
    )
    assert (
        fft_data_cg is not None
    ), "Coarse graining should preserve non-None fft_data"
    fft_data = fft_data_cg
    # Frequency-band exclusion is applied after coarse graining so CG divisibility
    # constraints are unaffected.
    exclude_bands = _normalize_excluded_frequency_bands(
        run_config.model.exclude_freq_bands
    )
    fft_data = _apply_frequency_exclusion(
        fft_data,
        exclude_bands,
    )
    scaled_true_psd = _align_true_psd_to_freq(scaled_true_psd, fft_data)

    if include_overlays:
        (
            extra_empirical_psd,
            extra_empirical_labels,
            extra_empirical_styles,
        ) = _build_welch_overlay(raw_ts, fft_data, run_config)
    else:
        extra_empirical_psd = None
        extra_empirical_labels = None
        extra_empirical_styles = None

    # _run_preprocessing_checks(fft_data, run_config)
    return PreprocessedMCMCInput(
        processed_data=fft_data,
        scaled_true_psd=scaled_true_psd,
        sampler_type=sampler_type,
        extra_empirical_psd=extra_empirical_psd,
        extra_empirical_labels=extra_empirical_labels,
        extra_empirical_styles=extra_empirical_styles,
        run_config=run_config,
    )


def _preprocess_data(data, config=None, **kwargs) -> PreprocessedMCMCInput:
    if kwargs:
        if config is not None:
            raise ValueError("Cant use both 'config' and kwargs")
        config = _build_config_from_kwargs(**kwargs)

    # This collects all arguments and config options, applies defaults, and performs validation
    run_config = _normalize_run_config(config)
    preproc_input = _preprocess_with_run_config(
        data,
        run_config,
        include_overlays=True,
    )
    _save_preprocessing_plot(
        preproc_input.processed_data,
        run_config,
        spline_model=None,
    )

    vi_warm_start_plan = None
    vi_cfg = run_config.vi
    if vi_cfg.init_from_vi and vi_cfg.use_coarse_vi_for_init:
        vi_cg_config, metadata = _derive_vi_coarse_grain_config(
            preproc_input.processed_data,
            run_config,
        )
        if vi_cg_config is not None and vi_cg_config.enabled:
            vi_warm_start_plan = (
                _build_coarse_vi_warm_start_plan_from_preprocessed(
                    preproc_input,
                    vi_cg_config=vi_cg_config,
                    metadata=metadata,
                )
            )
            if run_config.diagnostics.verbose:
                coarse_nfreq = vi_warm_start_plan.metadata["coarse_vi_nfreq"]
                logger.info(
                    "Using a separate coarse VI grid for warm start: "
                    f"mode={vi_warm_start_plan.metadata['coarse_vi_mode']}, "
                    f"N_full={vi_warm_start_plan.metadata['coarse_vi_full_nfreq']}, "
                    f"N_vi={coarse_nfreq}"
                )

    return replace(preproc_input, vi_warm_start_plan=vi_warm_start_plan)


# ---------------------------------------------------------------------------
# Sampler factory helpers (previously in sampler_factory.py)
# ---------------------------------------------------------------------------

from dataclasses import fields, is_dataclass  # noqa: E402
from typing import Any  # noqa: E402

from ..psplines import LogPSplines, MultivariateLogPSplines  # noqa: E402
from ..samplers import (  # noqa: E402
    MultivarBlockedNUTSConfig,
    MultivarBlockedNUTSSampler,
    NUTSConfig,
    NUTSSampler,
)


def _resolve_compute_lnz(
    sampler_type: SamplerName,
    compute_lnz: bool | None,
) -> bool:
    if compute_lnz is not None:
        return compute_lnz
    return sampler_type == "nuts"


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
            analytical_psd=model_config.analytical_psd,
        )
    raise ValueError(
        f"Unsupported processed data type: {type(processed_data)}."
    )


def _build_sampler_inputs(
    processed_data: Union[Periodogram, MultivarFFT],
    config: RunMCMCConfig,
    sampler_type: SamplerName,
    scaled_true_psd: np.ndarray | None,
    extra_empirical_psd,
    extra_empirical_labels,
    extra_empirical_styles,
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
    compute_lnz = _resolve_compute_lnz(
        config.sampler_type,
        run.diagnostics.compute_lnz,
    )
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
        "compute_lnz": compute_lnz,
        "scaling_factor": config.scaling_factor,
        "channel_stds": config.channel_stds,
        "true_psd": config.true_psd,
        "vi_psd_max_draws": run.vi.vi_psd_max_draws,
        "only_vi": run.vi.only_vi,
        "extra_empirical_psd": config.extra_empirical_psd,
        "extra_empirical_labels": config.extra_empirical_labels,
        "extra_empirical_styles": config.extra_empirical_styles,
    }


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
    nuts_config_kwargs = {
        **common_kwargs,
        "target_accept_prob": run.nuts.target_accept_prob,
        "max_tree_depth": run.nuts.max_tree_depth,
        "dense_mass": run.nuts.dense_mass,
        "init_from_vi": run.vi.init_from_vi,
        "vi_steps": run.vi.vi_steps,
        "vi_lr": run.vi.vi_lr,
        "vi_guide": run.vi.vi_guide,
        "vi_posterior_draws": run.vi.vi_posterior_draws,
        "vi_progress_bar": run.vi.vi_progress_bar,
        **nuts_extra_kwargs,
    }
    nuts_config = NUTSConfig(**nuts_config_kwargs)
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
    blocked_config_kwargs = {
        **common_kwargs,
        "target_accept_prob": run.nuts.target_accept_prob,
        "target_accept_prob_by_channel": run.nuts.target_accept_prob_by_channel,
        "max_tree_depth": run.nuts.max_tree_depth,
        "max_tree_depth_by_channel": run.nuts.max_tree_depth_by_channel,
        "dense_mass": run.nuts.dense_mass,
        "init_from_vi": run.vi.init_from_vi,
        "vi_steps": run.vi.vi_steps,
        "vi_lr": run.vi.vi_lr,
        "vi_guide": run.vi.vi_guide,
        "vi_posterior_draws": run.vi.vi_posterior_draws,
        "vi_progress_bar": run.vi.vi_progress_bar,
        "alpha_phi_theta": run.nuts.alpha_phi_theta,
        "beta_phi_theta": run.nuts.beta_phi_theta,
        "design_from_vi": run.nuts.design_from_vi,
        "design_from_vi_tau": run.nuts.design_from_vi_tau,
        **blocked_extra_kwargs,
    }
    blocked_config = MultivarBlockedNUTSConfig(**blocked_config_kwargs)
    return MultivarBlockedNUTSSampler(data, model, blocked_config)


def _create_sampler(
    data: Union[Periodogram, MultivarFFT],
    model,
    config: SamplerFactoryConfig,
):
    common_kwargs = _build_common_sampler_kwargs(config)
    if isinstance(data, Periodogram):
        return _build_univar_sampler(
            data, model, "nuts", config, common_kwargs
        )
    if isinstance(data, MultivarFFT):
        return _build_multivar_blocked_sampler(
            data, model, config, common_kwargs
        )
    raise ValueError(
        f"Unsupported data type: {type(data)}. Expected Periodogram or MultivarFFT."
    )
