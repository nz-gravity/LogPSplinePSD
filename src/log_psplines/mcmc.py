from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import arviz as az
import jax.numpy as jnp
import numpy as np

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
from .datatypes.univar import Timeseries
from .logger import logger
from .psplines import LogPSplines, MultivariateLogPSplines
from .samplers import (
    MetropolisHastingsConfig,
    MetropolisHastingsSampler,
    MultivarBlockedNUTSConfig,
    MultivarBlockedNUTSSampler,
    MultivarNUTSConfig,
    MultivarNUTSSampler,
    NUTSConfig,
    NUTSSampler,
)

SamplerName = Literal[
    "nuts",
    "mh",
    "multivar_blocked_nuts",
    "multivar_nuts",
]


@dataclass(frozen=True)
class ModelConfig:
    n_knots: int = 10
    degree: int = 3
    diffMatrixOrder: int = 2
    knot_kwargs: dict[str, Any] = field(default_factory=dict)
    parametric_model: Optional[jnp.ndarray] = None
    true_psd: Optional[jnp.ndarray] = None
    fmin: Optional[float] = None
    fmax: Optional[float] = None


@dataclass(frozen=True)
class DiagnosticsConfig:
    verbose: bool = True
    outdir: Optional[str] = None
    compute_psis: bool = True
    skip_plot_diagnostics: bool = False
    diagnostics_summary_mode: Literal["off", "light", "full"] = "light"
    diagnostics_summary_position: Literal["start", "end"] = "end"
    save_preprocessing_plots: bool = False
    preprocessing_plot_path: Optional[str] = None
    preprocessing_warn_threshold: float = 0.8
    preprocessing_warn_frac: float = 0.25
    preprocessing_min_lambda1_quantile: float = 0.0
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
class MHConfigOverride:
    target_accept_rate: float = 0.44
    adaptation_window: int = 50


@dataclass(frozen=True)
class NoiseFloorConfig:
    use_noise_floor: bool = False
    mode: str = "constant"
    constant: float = 0.0
    scale: float = 1e-4
    array: Optional[jnp.ndarray] = None
    theory_psd: Optional[jnp.ndarray] = None
    blocks: Optional[list[int] | str] = None


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
    plot_welch_overlay: bool | None = None
    welch_nperseg: int | None = None
    welch_noverlap: int | None = None
    welch_window: str = "hann"
    model: ModelConfig = field(default_factory=ModelConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    vi: VIConfig = field(default_factory=VIConfig)
    nuts: NUTSConfigOverride = field(default_factory=NUTSConfigOverride)
    mh: MHConfigOverride = field(default_factory=MHConfigOverride)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerFactoryConfig:
    sampler_type: SamplerName
    run_config: RunMCMCConfig
    scaling_factor: float
    true_psd: Optional[np.ndarray]
    freq_weights: Optional[np.ndarray]
    channel_stds: Optional[np.ndarray]
    extra_empirical_psd: list[EmpiricalPSD] | None = None
    extra_empirical_labels: list[str] | None = None
    extra_empirical_styles: list[dict] | None = None


def _unpack_true_psd(
    true_psd: Union[
        None,
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        list,
        dict,
    ],
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


def _interp_psd_array(
    psd: np.ndarray,
    freq_src: np.ndarray,
    freq_tgt: np.ndarray,
) -> np.ndarray:
    """Interpolate PSD arrays (real or complex) onto target frequencies."""
    if psd.shape[0] != freq_src.size:
        raise ValueError("psd and freq_src must have matching lengths.")

    freq_src = np.asarray(freq_src)
    freq_tgt = np.asarray(freq_tgt)
    flat = psd.reshape(psd.shape[0], -1)
    real_part = np.vstack(
        [
            np.interp(
                freq_tgt,
                freq_src,
                flat[:, idx].real,
                left=flat[0, idx].real,
                right=flat[-1, idx].real,
            )
            for idx in range(flat.shape[1])
        ]
    ).T

    if np.iscomplexobj(psd):
        imag_part = np.vstack(
            [
                np.interp(
                    freq_tgt,
                    freq_src,
                    flat[:, idx].imag,
                    left=flat[0, idx].imag,
                    right=flat[-1, idx].imag,
                )
                for idx in range(flat.shape[1])
            ]
        ).T
        resampled = real_part + 1j * imag_part
    else:
        resampled = real_part

    return resampled.reshape((freq_tgt.size,) + psd.shape[1:])


def _prepare_true_psd_for_freq(
    true_psd: Union[
        None,
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        list,
        dict,
    ],
    freq_target: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Resample supplied true PSD onto the target frequency grid."""
    if true_psd is None or freq_target is None:
        return true_psd

    freq_src, psd_array = _unpack_true_psd(true_psd)
    if psd_array is None:
        return None

    psd_array = np.asarray(psd_array)
    freq_target = np.asarray(freq_target)

    if psd_array.shape[0] == freq_target.size:
        return psd_array

    if freq_src is None:
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

    order = np.argsort(freq_src)
    freq_src_sorted = freq_src[order]
    psd_sorted = psd_array[order, ...]

    return _interp_psd_array(psd_sorted, freq_src_sorted, freq_target)


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
    Optional[np.ndarray],
]:
    """Apply coarse graining to the already-processed data if configured."""
    if processed_data is None or not cg_config.enabled:
        return processed_data, None, scaled_true_psd

    freq_weights = None

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
        power_coarse, weights = apply_coarse_graining_univar(
            power_selected, spec, freqs_selected
        )

        processed_data = Periodogram(
            spec.f_coarse,
            power_coarse,
            scaling_factor=processed_data.scaling_factor,
            weights=weights,
        )
        freq_weights = weights.astype(np.float32)

        logger.info(f"Coarse-grained periodogram: {spec}")

        if scaled_true_psd is not None:
            try:
                true_selected = np.asarray(scaled_true_psd)[selection_mask]
                true_coarse, _ = apply_coarse_graining_univar(
                    true_selected, spec, freqs_selected
                )
                scaled_true_psd = true_coarse
            except Exception:
                logger.warning(
                    "Could not coarse-grain provided true_psd; leaving unchanged."
                )

        return processed_data, freq_weights, scaled_true_psd

    if isinstance(processed_data, MultivarFFT):
        spec = compute_binning_structure(
            processed_data.freq,
            Nc=cg_config.Nc,
            Nh=cg_config.Nh,
            f_min=cg_config.f_min,
            f_max=cg_config.f_max,
        )
        processed_data, weights = apply_coarse_grain_multivar_fft(
            processed_data, spec
        )
        freq_weights = weights.astype(np.float32)
        logger.info(f"Coarse-grained multivariate FFT: {spec}")
        return processed_data, freq_weights, scaled_true_psd

    return processed_data, freq_weights, scaled_true_psd


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
    data: Union[Timeseries, MultivariateTimeseries, Periodogram, MultivarFFT],
    config: RunMCMCConfig,
) -> tuple[
    Union[Periodogram, MultivarFFT],
    Optional[MultivariateTimeseries],
    SamplerName,
]:
    sampler = config.sampler
    raw_multivar_ts: Optional[MultivariateTimeseries] = None

    if isinstance(data, (Periodogram, MultivarFFT)):
        processed = _truncate_frequency_range(
            data,
            config.model.fmin,
            config.model.fmax,
        )
        return processed, raw_multivar_ts, sampler

    if not isinstance(data, (Timeseries, MultivariateTimeseries)):
        raise ValueError(f"Unsupported input data type: {type(data)}")

    standardized_ts = data.standardise_for_psd()

    if isinstance(data, Timeseries):
        processed = standardized_ts.to_periodogram(
            fmin=config.model.fmin,
            fmax=config.model.fmax,
        )
    else:
        raw_multivar_ts = data
        if sampler == "multivar_nuts":
            if config.Nb != 1 and config.diagnostics.verbose:
                logger.warning(
                    "multivar_nuts ignores Nb; using the full-periodogram likelihood."
                )
            processed = standardized_ts.to_cross_spectral_density(
                fmin=config.model.fmin,
                fmax=config.model.fmax,
            )
        else:
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
        allowed = {"nuts", "multivar_blocked_nuts", "multivar_nuts"}
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
    return processed, raw_multivar_ts, sampler


def _maybe_build_welch_overlay(
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
    if config.diagnostics.outdir is None:
        return None, None, None

    want_welch = (
        config.plot_welch_overlay
        if config.plot_welch_overlay is not None
        else True
    )
    if not want_welch:
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
    true_psd: Optional[np.ndarray],
    processed_data: Optional[Union[Periodogram, MultivarFFT]],
) -> Optional[np.ndarray]:
    if true_psd is None or processed_data is None:
        return true_psd

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

        diag = eigenvalue_separation_diagnostics(
            freq=np.asarray(processed_data.freq, dtype=float),
            matrix=np.asarray(processed_data.raw_psd),
            min_lambda1_quantile=float(
                config.diagnostics.preprocessing_min_lambda1_quantile
            ),
        )
        p = int(diag.eigvals_desc.shape[1])
        if p < 2:
            return

        warn_threshold = float(config.diagnostics.preprocessing_warn_threshold)
        warn_frac = float(config.diagnostics.preprocessing_warn_frac)
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

        if not config.diagnostics.save_preprocessing_plots:
            return

        out_path = None
        if config.diagnostics.preprocessing_plot_path is not None:
            out_path = Path(config.diagnostics.preprocessing_plot_path)
        elif config.diagnostics.outdir is not None:
            out_path = (
                Path(config.diagnostics.outdir)
                / "preprocessing_eigenvalue_ratios.png"
            )

        if out_path is None:
            logger.warning(
                "Skipping preprocessing plot save: outdir is None and preprocessing_plot_path is not set."
            )
            return

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
    freq_weights: Optional[np.ndarray],
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
        scaling_factor=float(scaling_factor),
        true_psd=scaled_true_psd,
        freq_weights=freq_weights,
        channel_stds=channel_stds,
        extra_empirical_psd=extra_empirical_psd,
        extra_empirical_labels=extra_empirical_labels,
        extra_empirical_styles=extra_empirical_styles,
    )


def _normalize_run_config(
    config: RunMCMCConfig | SamplerName | None,
    legacy_kwargs: dict[str, Any],
) -> RunMCMCConfig:
    if isinstance(config, str):
        legacy_kwargs = dict(legacy_kwargs)
        legacy_kwargs.setdefault("sampler", config)
        config = None

    if config is not None:
        if legacy_kwargs:
            raise ValueError(
                "Pass either a RunMCMCConfig or keyword options, not both."
            )
        return config

    kwargs = dict(legacy_kwargs)

    model_cfg = ModelConfig(
        n_knots=kwargs.pop("n_knots", 10),
        degree=kwargs.pop("degree", 3),
        diffMatrixOrder=kwargs.pop("diffMatrixOrder", 2),
        knot_kwargs=kwargs.pop("knot_kwargs", {}),
        parametric_model=kwargs.pop("parametric_model", None),
        true_psd=kwargs.pop("true_psd", None),
        fmin=kwargs.pop("fmin", None),
        fmax=kwargs.pop("fmax", None),
    )
    diagnostics_cfg = DiagnosticsConfig(
        verbose=kwargs.pop("verbose", True),
        outdir=kwargs.pop("outdir", None),
        compute_psis=kwargs.pop("compute_psis", True),
        skip_plot_diagnostics=kwargs.pop("skip_plot_diagnostics", False),
        diagnostics_summary_mode=kwargs.pop(
            "diagnostics_summary_mode", "light"
        ),
        diagnostics_summary_position=kwargs.pop(
            "diagnostics_summary_position", "end"
        ),
        save_preprocessing_plots=kwargs.pop("save_preprocessing_plots", False),
        preprocessing_plot_path=kwargs.pop("preprocessing_plot_path", None),
        preprocessing_warn_threshold=kwargs.pop(
            "preprocessing_warn_threshold", 0.8
        ),
        preprocessing_warn_frac=kwargs.pop("preprocessing_warn_frac", 0.25),
        preprocessing_min_lambda1_quantile=kwargs.pop(
            "preprocessing_min_lambda1_quantile", 0.0
        ),
        compute_lnz=kwargs.pop("compute_lnz", False),
    )
    vi_cfg = VIConfig(
        only_vi=kwargs.pop("only_vi", False),
        init_from_vi=kwargs.pop("init_from_vi", True),
        vi_steps=kwargs.pop("vi_steps", 1500),
        vi_lr=kwargs.pop("vi_lr", 1e-2),
        vi_guide=kwargs.pop("vi_guide", None),
        vi_posterior_draws=kwargs.pop("vi_posterior_draws", 256),
        vi_progress_bar=kwargs.pop("vi_progress_bar", None),
        vi_psd_max_draws=kwargs.pop("vi_psd_max_draws", 64),
    )
    nuts_cfg = NUTSConfigOverride(
        target_accept_prob=kwargs.pop("target_accept_prob", 0.8),
        target_accept_prob_by_channel=kwargs.pop(
            "target_accept_prob_by_channel", None
        ),
        max_tree_depth=kwargs.pop("max_tree_depth", 10),
        max_tree_depth_by_channel=kwargs.pop(
            "max_tree_depth_by_channel", None
        ),
        dense_mass=kwargs.pop("dense_mass", True),
        alpha_phi_theta=kwargs.pop("alpha_phi_theta", None),
        beta_phi_theta=kwargs.pop("beta_phi_theta", None),
    )
    mh_cfg = MHConfigOverride(
        target_accept_rate=kwargs.pop("target_accept_rate", 0.44),
        adaptation_window=kwargs.pop("adaptation_window", 50),
    )
    noise_floor_cfg = NoiseFloorConfig(
        use_noise_floor=kwargs.pop("use_noise_floor", False),
        mode=kwargs.pop("noise_floor_mode", "constant"),
        constant=kwargs.pop("noise_floor_constant", 0.0),
        scale=kwargs.pop("noise_floor_scale", 1e-4),
        array=kwargs.pop("noise_floor_array", None),
        theory_psd=kwargs.pop("theory_psd", None),
        blocks=kwargs.pop("noise_floor_blocks", None),
    )

    return RunMCMCConfig(
        sampler=kwargs.pop("sampler", "nuts"),
        n_samples=kwargs.pop("n_samples", 1000),
        n_warmup=kwargs.pop("n_warmup", 500),
        num_chains=kwargs.pop("num_chains", 1),
        chain_method=kwargs.pop("chain_method", None),
        alpha_phi=kwargs.pop("alpha_phi", 1.0),
        beta_phi=kwargs.pop("beta_phi", 1.0),
        alpha_delta=kwargs.pop("alpha_delta", 1e-4),
        beta_delta=kwargs.pop("beta_delta", 1e-4),
        rng_key=kwargs.pop("rng_key", 42),
        coarse_grain_config=kwargs.pop("coarse_grain_config", None),
        Nb=kwargs.pop("Nb", 1),
        plot_welch_overlay=kwargs.pop("plot_welch_overlay", None),
        welch_nperseg=kwargs.pop("welch_nperseg", None),
        welch_noverlap=kwargs.pop("welch_noverlap", None),
        welch_window=kwargs.pop("welch_window", "hann"),
        model=model_cfg,
        diagnostics=diagnostics_cfg,
        vi=vi_cfg,
        nuts=nuts_cfg,
        mh=mh_cfg,
        extra_kwargs=kwargs,
    )


def run_mcmc(
    data: Union[Timeseries, MultivariateTimeseries, Periodogram, MultivarFFT],
    config: RunMCMCConfig | SamplerName | None = None,
    **legacy_kwargs,
) -> az.InferenceData:
    """Unified MCMC entrypoint for univariate and multivariate PSD estimation."""
    run_config = _normalize_run_config(config, legacy_kwargs)

    if run_config.vi.only_vi:
        vi_capable = {"nuts", "multivar_blocked_nuts", "multivar_nuts"}
        if run_config.sampler not in vi_capable:
            raise ValueError(
                f"Sampler '{run_config.sampler}' does not support variational-only execution."
            )

    processed_data, raw_multivar_ts, sampler_type = _prepare_processed_data(
        data,
        run_config,
    )
    scaled_true_psd = run_config.model.true_psd

    processed_data, freq_weights, scaled_true_psd = (
        _coarse_grain_processed_data(
            processed_data,
            _normalize_coarse_grain_config(run_config.coarse_grain_config),
            scaled_true_psd,
        )
    )

    (
        extra_empirical_psd,
        extra_empirical_labels,
        extra_empirical_styles,
    ) = _maybe_build_welch_overlay(raw_multivar_ts, processed_data, run_config)

    scaled_true_psd = _align_true_psd_to_freq(scaled_true_psd, processed_data)
    _run_preprocessing_checks(processed_data, run_config)

    model = _build_model_from_data(processed_data, run_config.model)
    sampler_inputs = _build_sampler_inputs(
        processed_data,
        run_config,
        sampler_type,
        scaled_true_psd,
        freq_weights,
        extra_empirical_psd,
        extra_empirical_labels,
        extra_empirical_styles,
    )

    sampler_obj = _create_sampler(
        data=processed_data,
        model=model,
        config=sampler_inputs,
    )
    return sampler_obj.sample(
        n_samples=run_config.n_samples,
        n_warmup=run_config.n_warmup,
        only_vi=run_config.vi.only_vi,
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
        "compute_psis": run.diagnostics.compute_psis,
        "skip_plot_diagnostics": run.diagnostics.skip_plot_diagnostics,
        "diagnostics_summary_mode": run.diagnostics.diagnostics_summary_mode,
        "diagnostics_summary_position": run.diagnostics.diagnostics_summary_position,
        "compute_lnz": run.diagnostics.compute_lnz,
        "scaling_factor": config.scaling_factor,
        "channel_stds": config.channel_stds,
        "true_psd": config.true_psd,
        "freq_weights": config.freq_weights,
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
        if sampler_type not in {"nuts", "mh"}:
            raise ValueError(
                f"Unknown sampler_type '{sampler_type}' for univariate data. Choose 'nuts' or 'mh'."
            )
        return sampler_type

    allowed_types = {"nuts", "multivar_blocked_nuts", "multivar_nuts"}
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
    if sampler_type == "nuts":
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
            **run.extra_kwargs,
        )
        return NUTSSampler(data, model, nuts_config)

    mh_config = MetropolisHastingsConfig(
        **common_kwargs,
        target_accept_rate=run.mh.target_accept_rate,
        adaptation_window=run.mh.adaptation_window,
        adaptation_start=100,
        step_size_factor=1.1,
        min_step_size=1e-6,
        max_step_size=10.0,
        **run.extra_kwargs,
    )
    return MetropolisHastingsSampler(data, model, mh_config)


def _build_multivar_blocked_sampler(
    data: MultivarFFT,
    model,
    config: SamplerFactoryConfig,
    common_kwargs: dict[str, Any],
):
    run = config.run_config
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
        use_noise_floor=run.noise_floor.use_noise_floor,
        noise_floor_mode=run.noise_floor.mode,
        noise_floor_constant=run.noise_floor.constant,
        noise_floor_scale=run.noise_floor.scale,
        noise_floor_array=run.noise_floor.array,
        theory_psd=run.noise_floor.theory_psd,
        noise_floor_blocks=(
            run.noise_floor.blocks
            if run.noise_floor.blocks is not None
            else MultivarBlockedNUTSConfig.noise_floor_blocks
        ),
        **run.extra_kwargs,
    )
    return MultivarBlockedNUTSSampler(data, model, blocked_config)


def _build_multivar_coupled_sampler(
    data: MultivarFFT,
    model,
    config: SamplerFactoryConfig,
    common_kwargs: dict[str, Any],
):
    run = config.run_config
    coupled_config = MultivarNUTSConfig(
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
        **run.extra_kwargs,
    )
    return MultivarNUTSSampler(data, model, coupled_config)


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

    return _build_multivar_coupled_sampler(
        data,
        model,
        config,
        common_kwargs,
    )
