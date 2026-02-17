from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from ._jaxtypes import Complex, Float
from ._typecheck import runtime_typecheck
from .coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_grain_multivar_fft,
    apply_coarse_graining_univar,
    compute_binning_structure,
)
from .datatypes import Periodogram
from .datatypes.multivar import (
    EmpiricalPSD,
    MultivarFFT,
    MultivariateTimeseries,
)
from .datatypes.multivar_utils import _interp_frequency_indexed_array
from .datatypes.univar import Timeseries
from .logger import logger
from .psplines import LogPSplines, MultivariateLogPSplines
from .samplers import (
    MultivarBlockedNUTSConfig,
    MultivarBlockedNUTSSampler,
    NUTSConfig,
    NUTSSampler,
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
    sampler: SamplerName = "nuts"
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


def _unpack_true_psd(
    true_psd: TruePSDInput,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (freq, psd) pair from accepted true_psd formats."""
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


def _normalize_coarse_grain_config(
    coarse_grain_config: Optional[CoarseGrainConfig | dict],
) -> CoarseGrainConfig:
    if coarse_grain_config is None:
        return CoarseGrainConfig()
    if isinstance(coarse_grain_config, dict):
        return CoarseGrainConfig(**coarse_grain_config)
    return coarse_grain_config


def _coarse_grain_processed_data(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    cg_config: CoarseGrainConfig,
    scaled_true_psd: Optional[np.ndarray],
) -> Tuple[
    Optional[Union[Periodogram, MultivarFFT]],
    Optional[np.ndarray],
]:
    """Apply coarse graining to the already-processed data if configured."""
    if processed_data is None or not cg_config.enabled:
        return processed_data, scaled_true_psd

    if isinstance(processed_data, Periodogram):
        spec = compute_binning_structure(
            processed_data.freqs,
            Nc=cg_config.Nc,
            Nh=cg_config.Nh,
            f_min=cg_config.f_min,
            f_max=cg_config.f_max,
        )

        selection_mask = spec.selection_mask
        power_selected = np.asarray(processed_data.power[selection_mask])
        freqs_selected = processed_data.freqs[selection_mask]
        power_coarse = apply_coarse_graining_univar(
            power_selected, spec, freqs_selected
        )

        processed_data = Periodogram(
            spec.f_coarse,
            power_coarse,
            scaling_factor=processed_data.scaling_factor,
            Nh=int(spec.Nh),
        )

        logger.info(f"Coarse-grained periodogram: {spec}")

        if scaled_true_psd is not None:
            try:
                true_selected = np.asarray(scaled_true_psd)[selection_mask]
                true_coarse = apply_coarse_graining_univar(
                    true_selected, spec, freqs_selected
                )
                scaled_true_psd = true_coarse
            except Exception:
                logger.warning(
                    "Could not coarse-grain provided true_psd; leaving unchanged."
                )

        return processed_data, scaled_true_psd

    if isinstance(processed_data, MultivarFFT):
        spec = compute_binning_structure(
            processed_data.freq,
            Nc=cg_config.Nc,
            Nh=cg_config.Nh,
            f_min=cg_config.f_min,
            f_max=cg_config.f_max,
        )
        processed_data = apply_coarse_grain_multivar_fft(processed_data, spec)
        logger.info(f"Coarse-grained multivariate FFT: {spec}")
        return processed_data, scaled_true_psd

    return processed_data, scaled_true_psd


def _truncate_frequency_range(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    fmin: Optional[float],
    fmax: Optional[float],
) -> Optional[Union[Periodogram, MultivarFFT]]:
    if processed_data is None or (fmin is None and fmax is None):
        return processed_data

    freq_attr = "freqs" if isinstance(processed_data, Periodogram) else "freq"
    freqs = np.asarray(getattr(processed_data, freq_attr), dtype=float)
    if freqs.size == 0:
        raise ValueError("Processed data contains no frequencies.")

    freq_min = float(freqs[0])
    freq_max = float(freqs[-1])
    lower = freq_min if fmin is None else float(fmin)
    upper = freq_max if fmax is None else float(fmax)

    lower = min(max(lower, freq_min), freq_max)
    upper = min(max(upper, freq_min), freq_max)
    if upper < lower:
        upper = lower

    truncated = processed_data.cut(lower, upper)
    n_points = (
        truncated.n if isinstance(truncated, Periodogram) else truncated.N
    )
    if n_points == 0:
        raise ValueError(
            "Frequency truncation removed all data points. Check fmin/fmax."
        )
    return truncated


def _prepare_processed_data(
    data: Union[Timeseries, MultivariateTimeseries],
    config: RunMCMCConfig,
) -> tuple[
    Union[Periodogram, MultivarFFT],
    Optional[MultivariateTimeseries],
    SamplerName,
]:
    sampler = config.sampler
    raw_multivar_ts: Optional[MultivariateTimeseries] = None

    standardized_ts = data.standardise_for_psd()

    if isinstance(data, Timeseries):
        processed = standardized_ts.to_periodogram(
            fmin=config.model.fmin,
            fmax=config.model.fmax,
        )
    else:
        raw_multivar_ts = data
        processed = standardized_ts.to_wishart_stats(
            Nb=config.Nb,
            fmin=config.model.fmin,
            fmax=config.model.fmax,
        )

    if config.diagnostics.verbose:
        logger.info(
            f"Standardized data: original scale ~{processed.scaling_factor:.2e}"
        )

    if isinstance(processed, MultivarFFT):
        allowed = {"nuts", "multivar_blocked_nuts"}
        if sampler not in allowed:
            if config.diagnostics.verbose:
                allowed_str = ", ".join(sorted(allowed))
                logger.warning(
                    f"Multivariate analysis supports {allowed_str}. Using NUTS instead of {sampler}"
                )
            sampler = "nuts"

    processed = _truncate_frequency_range(
        processed,
        config.model.fmin,
        config.model.fmax,
    )
    if processed is None:
        raise ValueError("Processed data unexpectedly None.")
    return processed, raw_multivar_ts, sampler


def _build_welch_overlay(
    raw_multivar_ts: Optional[MultivariateTimeseries],
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    config: RunMCMCConfig,
) -> tuple[
    list[EmpiricalPSD] | None,
    list[str] | None,
    list[dict] | None,
]:
    if raw_multivar_ts is None:
        return None, None, None
    if not isinstance(processed_data, MultivarFFT):
        return None, None, None

    try:
        welch_emp = raw_multivar_ts.get_empirical_psd(
            nperseg=config.welch_nperseg,
            noverlap=config.welch_noverlap,
            window=config.welch_window,
        )
        freq_target = np.asarray(processed_data.freq, dtype=float)
        f_lo = float(freq_target[0])
        f_hi = float(freq_target[-1])
        keep = (
            (welch_emp.freq > 0.0)
            & (welch_emp.freq >= f_lo)
            & (welch_emp.freq <= f_hi)
        )
        if not np.any(keep):
            if config.diagnostics.verbose:
                logger.warning(
                    "Welch overlay requested but produced no in-range positive-frequency bins; skipping."
                )
            return None, None, None

        overlay = EmpiricalPSD(
            freq=np.asarray(welch_emp.freq)[keep],
            psd=np.asarray(welch_emp.psd)[keep],
            coherence=np.asarray(welch_emp.coherence)[keep],
            channels=welch_emp.channels,
        )
        return (
            [overlay],
            ["Welch"],
            [
                {
                    "color": "0.25",
                    "lw": 0.9,
                    "alpha": 0.18,
                    "ls": ":",
                    "zorder": -10,
                }
            ],
        )
    except Exception as exc:
        if config.diagnostics.verbose:
            logger.warning(f"Could not compute Welch overlay: {exc}")
        return None, None, None


def _align_true_psd_to_freq(
    true_psd: TruePSDInput,
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
) -> Optional[np.ndarray]:
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
        return true_psd

    aligned = _prepare_true_psd_for_freq(true_psd, freq_target)
    return aligned if aligned is not None else true_psd


def _run_preprocessing_checks(
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
    config: RunMCMCConfig,
) -> None:
    if not isinstance(processed_data, MultivarFFT):
        return
    if processed_data.raw_psd is None:
        logger.warning(
            "Skipping eigenvalue separation check: processed_data.raw_psd is missing."
        )
        return

    try:
        from .diagnostics.preprocessing import (
            eigenvalue_separation_diagnostics,
            save_eigenvalue_separation_plot,
        )

        # Use hardcoded defaults
        min_lambda1_quantile = 0.0
        warn_threshold = 0.8
        warn_frac = 0.25

        diag = eigenvalue_separation_diagnostics(
            freq=np.asarray(processed_data.freq, dtype=float),
            matrix=np.asarray(processed_data.raw_psd),
            min_lambda1_quantile=min_lambda1_quantile,
        )
        p = int(diag.eigvals_desc.shape[1])
        if p < 2:
            return
        summaries = None
        if config.diagnostics.verbose:
            summaries = diag.ratio_summary(warn_threshold=warn_threshold)
            if diag.lambda1_cutoff is not None:
                kept = int(np.count_nonzero(diag.mask))
                logger.info(
                    f"Eigenvalue separation mask: keep λ1 > {diag.lambda1_cutoff:.3e} ({kept}/{diag.mask.size} bins)."
                )

        for key, ratio in diag.ratios.items():
            ratio_m = np.asarray(ratio)[diag.mask]
            ratio_m = ratio_m[np.isfinite(ratio_m)]
            frac = (
                float(np.mean(ratio_m > warn_threshold))
                if ratio_m.size
                else 0.0
            )
            if frac >= warn_frac:
                if summaries is None:
                    msg = (
                        f"Eigenvalue separation {key}: "
                        f"frac(>{warn_threshold:.2f})={frac*100:.1f}%"
                    )
                else:
                    msg = f"Eigenvalue separation {summaries[key]}"
                logger.warning(msg)
                worst = diag.worst_frequencies(top_k=10).get(key, [])
                if worst:
                    joined = ", ".join([f"{f:.4g}:{v:.3f}" for f, v in worst])
                    logger.warning(
                        f"Worst-separated frequencies for {key}: {joined}"
                    )
                continue
            if config.diagnostics.verbose and summaries is not None:
                logger.info(f"Eigenvalue separation {summaries[key]}")

        # Always save preprocessing plots
        if config.diagnostics.outdir is None:
            logger.warning("Skipping preprocessing plot save: outdir is None.")
            return

        out_path = (
            Path(config.diagnostics.outdir)
            / "preprocessing_eigenvalue_ratios.png"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_eigenvalue_separation_plot(
            diag,
            str(out_path),
            warn_threshold=warn_threshold,
        )
        if config.diagnostics.verbose:
            logger.info(f"Saved preprocessing eigenvalue plot to {out_path}")
    except Exception as exc:
        logger.warning(f"Eigenvalue separation check failed: {exc}")


def _build_model_from_data(
    processed_data: Union[Periodogram, MultivarFFT],
    model_config: ModelConfig,
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
    scaled_true_psd: Optional[np.ndarray],
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
        "sampler",
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
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in routed}
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
