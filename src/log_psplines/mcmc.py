from typing import Literal, Optional, Tuple, Union

import arviz as az
import jax.numpy as jnp
import numpy as np
from loguru import logger

from .coarse_grain import (
    CoarseGrainConfig,
    apply_coarse_graining_univar,
    coarse_grain_multivar_fft,
    compute_binning_structure,
)
from .datatypes import Periodogram
from .datatypes.multivar import MultivarFFT, MultivariateTimeseries
from .datatypes.univar import Timeseries
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

    resampled = _interp_psd_array(psd_sorted, freq_src_sorted, freq_target)
    return resampled


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
            f_transition=cg_config.f_transition,
            n_log_bins=cg_config.n_log_bins,
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

        logger.info(
            "Coarse-grained periodogram: selected={} -> n_freq={} (n_bins_high={})",
            int(spec.selection_mask.sum()),
            processed_data.n,
            int(spec.n_bins_high),
        )

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
            f_transition=cg_config.f_transition,
            n_log_bins=cg_config.n_log_bins,
            f_min=cg_config.f_min,
            f_max=cg_config.f_max,
        )

        processed_data, weights = coarse_grain_multivar_fft(
            processed_data, spec
        )
        freq_weights = weights.astype(np.float32)

        logger.info(
            "Coarse-grained multivariate FFT: selected={} -> n_freq={} (n_bins_high={})",
            int(spec.selection_mask.sum()),
            processed_data.n_freq,
            int(spec.n_bins_high),
        )

        return processed_data, freq_weights, scaled_true_psd

    return processed_data, freq_weights, scaled_true_psd


def run_mcmc(
    data: Union[Timeseries, MultivariateTimeseries, Periodogram, MultivarFFT],
    sampler: Literal[
        "nuts",
        "mh",
        "multivar_blocked_nuts",
        "multivar_nuts",
    ] = "nuts",
    n_samples: int = 1000,
    n_warmup: int = 500,
    num_chains: int = 1,
    # Model parameters
    n_knots: int = 10,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    knot_kwargs: dict = {},
    parametric_model: Optional[jnp.ndarray] = None,
    true_psd: Optional[jnp.ndarray] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    # Sampler parameters
    alpha_phi: float = 1.0,
    beta_phi: float = 1.0,
    alpha_delta: float = 1e-4,
    beta_delta: float = 1e-4,
    rng_key: int = 42,
    verbose: bool = True,
    outdir: Optional[str] = None,
    compute_lnz: bool = False,
    only_vi: bool = False,
    # NUTS specific
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    init_from_vi: bool = True,
    vi_steps: int = 1500,
    vi_lr: float = 1e-2,
    vi_guide: Optional[str] = None,
    vi_posterior_draws: int = 256,
    vi_progress_bar: Optional[bool] = None,
    vi_psd_max_draws: int = 64,
    coarse_grain_config: Optional[CoarseGrainConfig | dict] = None,
    n_time_blocks: int = 1,
    alpha_phi_theta: Optional[float] = None,
    beta_phi_theta: Optional[float] = None,
    # MH specific
    target_accept_rate: float = 0.44,
    adaptation_window: int = 50,
    **kwargs,
) -> az.InferenceData:
    """
    Unified MCMC interface for both univariate and multivariate PSD estimation.

    Parameters
    ----------
    data : Timeseries, MultivariateTimeseries, Periodogram or MultivarFFT
        Input data for analysis. When timeseries data is provided, it will be
        automatically standardized for numerical stability, and posterior samples
        will be rescaled to original units.
    sampler : {"nuts", "mh", "multivar_blocked_nuts", "multivar_nuts"}
        MCMC sampler type. Multivariate analysis supports the blocked sampler
        ("multivar_blocked_nuts", the default) and the coupled sampler
        ("multivar_nuts").
    n_samples : int, default=1000
        Number of posterior samples to collect
    n_warmup : int, default=500
        Number of warmup/burn-in samples
    num_chains : int, default=1
        Number of MCMC chains to run for convergence diagnostics
    n_knots : int, default=10
        Number of knots for B-spline basis
    degree : int, default=3
        Degree of B-spline basis functions
    diffMatrixOrder : int, default=2
        Order of difference penalty matrix
    knot_kwargs : dict, default={}
        Additional keyword arguments for knot allocation
    parametric_model : Optional[jnp.ndarray], default=None
        Known parametric component (univariate only)
    alpha_phi : float, default=1.0
        Alpha parameter for precision prior
    beta_phi : float, default=1.0
        Beta parameter for precision prior
    alpha_delta : float, default=1e-4
        Alpha parameter for smoothing prior
    beta_delta : float, default=1e-4
        Beta parameter for smoothing prior
    fmin, fmax : float, optional
        Lower/upper frequency bounds to retain. Only frequencies within the
        inclusive range ``[fmin, fmax]`` are used for inference. When either
        bound is ``None`` the full available range is kept on that side.
    rng_key : int, default=42
        Random number generator key
    verbose : bool, default=True
        Whether to print progress information
    outdir : Optional[str], default=None
        Directory to save output files
    compute_lnz : bool, default=False
        Whether to compute log evidence
    only_vi : bool, default=False
        When ``True`` the samplers that support variational initialisation
        (NUTS variants) skip the MCMC phase and return an approximation based on
        VI posterior draws.
    target_accept_prob : float, default=0.8
        Target acceptance probability for NUTS
    max_tree_depth : int, default=10
        Maximum tree depth for NUTS
    coarse_grain_config : CoarseGrainConfig or dict, optional
        Optional frequency coarse-graining configuration for univariate periodograms
        and multivariate FFT statistics.
    n_time_blocks : int, default=1
        Number of equal-length segments used to form block-averaged (Wishart)
        periodogram statistics for multivariate timeseries input. ``1`` reduces
        to the standard full-length periodogram.
    target_accept_rate : float, default=0.44
        Target acceptance rate for MH
    adaptation_window : int, default=50
        Adaptation window for MH
    **kwargs
        Additional keyword arguments

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object with MCMC results
    """

    # Map any supported aliases onto canonical sampler names
    sampler_aliases = {}
    sampler = sampler_aliases.get(sampler, sampler)

    if only_vi:
        vi_capable = {"nuts", "multivar_blocked_nuts", "multivar_nuts"}
        if sampler not in vi_capable:
            raise ValueError(
                f"Sampler '{sampler}' does not support variational-only execution."
            )

    if coarse_grain_config is None:
        cg_config = CoarseGrainConfig()
    elif isinstance(coarse_grain_config, dict):
        cg_config = CoarseGrainConfig(**coarse_grain_config)
    else:
        cg_config = coarse_grain_config

    # Keep true_psd on the original data scale. Any standardisation applied
    # to the observed data is tracked separately via the scaling_factor and
    # consistently undone inside ArviZ conversion before comparisons.
    scaled_true_psd = true_psd

    # Handle raw timeseries input - standardize automatically
    processed_data = None
    freq_weights = None

    if isinstance(data, (Timeseries, MultivariateTimeseries)):
        # Standardize the raw timeseries for numerical stability
        standardized_ts = data.standardise_for_psd()

        # Convert to processed format for existing pipeline
        if isinstance(data, Timeseries):
            processed_data = standardized_ts.to_periodogram(
                fmin=fmin, fmax=fmax
            )
        else:  # MultivariateTimeseries
            if sampler == "multivar_nuts":
                if n_time_blocks != 1 and verbose:
                    logger.warning(
                        "multivar_nuts ignores n_time_blocks; using the full-periodogram likelihood."
                    )
                processed_data = standardized_ts.to_cross_spectral_density(
                    fmin=fmin, fmax=fmax
                )
            else:
                processed_data = standardized_ts.to_wishart_stats(
                    n_blocks=n_time_blocks,
                    fmin=fmin,
                    fmax=fmax,
                )

        if verbose:
            logger.info(
                f"Standardized data: original scale ~{processed_data.scaling_factor:.2e}"
            )

        # Validate sampler for standardized data
        if isinstance(processed_data, MultivarFFT):
            allowed_multivar_samplers = {
                "nuts",
                "multivar_blocked_nuts",
                "multivar_nuts",
            }
            if sampler not in allowed_multivar_samplers:
                if verbose:
                    allowed = ", ".join(sorted(allowed_multivar_samplers))
                    logger.warning(
                        f"Multivariate analysis supports {allowed}. Using NUTS instead of {sampler}"
                    )
                sampler = "nuts"

    if isinstance(data, (Periodogram, MultivarFFT)):
        processed_data = data  # Use as is

    # Apply frequency truncation when raw processed data is provided
    if processed_data is not None and (fmin is not None or fmax is not None):
        freq_attr = (
            "freqs" if isinstance(processed_data, Periodogram) else "freq"
        )
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

        processed_data = processed_data.cut(lower, upper)

        n_points = (
            processed_data.n
            if isinstance(processed_data, Periodogram)
            else processed_data.n_freq
        )
        if n_points == 0:
            raise ValueError(
                "Frequency truncation removed all data points. Check fmin/fmax."
            )

    processed_data, freq_weights, scaled_true_psd = (
        _coarse_grain_processed_data(
            processed_data, cg_config, scaled_true_psd
        )
    )

    # Align true_psd (if provided) to the processed frequency grid
    if scaled_true_psd is not None and processed_data is not None:
        if isinstance(processed_data, Periodogram):
            freq_target = np.asarray(processed_data.freqs)
        elif isinstance(processed_data, MultivarFFT):
            freq_target = np.asarray(processed_data.freq)
        else:
            freq_target = None
        if freq_target is not None:
            aligned_true_psd = _prepare_true_psd_for_freq(
                scaled_true_psd, freq_target
            )
            if aligned_true_psd is not None:
                scaled_true_psd = aligned_true_psd

    # Create model based on processed data type
    if isinstance(processed_data, Periodogram):
        # Univariate case
        model = LogPSplines.from_periodogram(
            processed_data,
            n_knots=n_knots,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            parametric_model=(
                parametric_model if parametric_model is not None else None
            ),
            knot_kwargs=knot_kwargs,
        )
    elif isinstance(processed_data, MultivarFFT):
        # Multivariate case
        model = MultivariateLogPSplines.from_multivar_fft(
            processed_data,
            n_knots=n_knots,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            knot_kwargs=knot_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported processed data type: {type(processed_data)}."
        )

    # Create sampler
    # Always use processed_data (the standardized Periodogram or MultivarFFT) for the sampler and model
    sampler_obj = create_sampler(
        data=processed_data,  # Always use processed_data, which has correct scaling_factor
        model=model,
        sampler_type=sampler,
        num_chains=num_chains,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        rng_key=rng_key,
        verbose=verbose,
        outdir=outdir,
        compute_lnz=compute_lnz,
        only_vi=only_vi,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_from_vi=init_from_vi,
        vi_steps=vi_steps,
        vi_lr=vi_lr,
        vi_guide=vi_guide,
        vi_posterior_draws=vi_posterior_draws,
        vi_progress_bar=vi_progress_bar,
        vi_psd_max_draws=vi_psd_max_draws,
        target_accept_rate=target_accept_rate,
        adaptation_window=adaptation_window,
        scaling_factor=(
            processed_data.scaling_factor
            if processed_data and hasattr(processed_data, "scaling_factor")
            else 1.0
        ),  # Pass scaling info to sampler
        true_psd=scaled_true_psd,
        freq_weights=freq_weights,
        channel_stds=(
            processed_data.channel_stds
            if processed_data is not None
            and hasattr(processed_data, "channel_stds")
            else None
        ),
        alpha_phi_theta=alpha_phi_theta,
        beta_phi_theta=beta_phi_theta,
        **kwargs,
    )

    return sampler_obj.sample(
        n_samples=n_samples, n_warmup=n_warmup, only_vi=only_vi
    )


def create_sampler(
    data: Union[Periodogram, MultivarFFT],
    model,
    sampler_type: Literal[
        "nuts",
        "mh",
        "multivar_blocked_nuts",
        "multivar_nuts",
    ] = "nuts",
    num_chains: int = 1,
    alpha_phi: float = 1.0,
    beta_phi: float = 1.0,
    alpha_delta: float = 1e-4,
    beta_delta: float = 1e-4,
    rng_key: int = 42,
    verbose: bool = True,
    outdir: Optional[str] = None,
    compute_lnz: bool = False,
    only_vi: bool = False,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    init_from_vi: bool = True,
    vi_steps: int = 1500,
    vi_lr: float = 1e-2,
    vi_guide: Optional[str] = None,
    vi_posterior_draws: int = 256,
    vi_progress_bar: Optional[bool] = None,
    vi_psd_max_draws: int = 64,
    target_accept_rate: float = 0.44,
    adaptation_window: int = 50,
    scaling_factor: float = 1.0,
    true_psd: Optional[jnp.ndarray] = None,
    freq_weights: Optional[np.ndarray] = None,
    channel_stds: Optional[np.ndarray] = None,
    alpha_phi_theta: Optional[float] = None,
    beta_phi_theta: Optional[float] = None,
    **kwargs,
):
    """Factory function to create appropriate sampler."""

    sampler_aliases = {}
    sampler_type = sampler_aliases.get(sampler_type, sampler_type)

    common_config_kwargs = {
        "alpha_phi": alpha_phi,
        "beta_phi": beta_phi,
        "alpha_delta": alpha_delta,
        "beta_delta": beta_delta,
        "num_chains": num_chains,
        "rng_key": rng_key,
        "verbose": verbose,
        "outdir": outdir,
        "compute_lnz": compute_lnz,
        "scaling_factor": scaling_factor,
        "channel_stds": channel_stds,
        "true_psd": true_psd,
        "freq_weights": freq_weights,
        "vi_psd_max_draws": vi_psd_max_draws,
        "only_vi": only_vi,
    }

    if isinstance(data, Periodogram):
        # Univariate case
        if sampler_type == "nuts":
            config = NUTSConfig(
                **common_config_kwargs,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                init_from_vi=init_from_vi,
                vi_steps=vi_steps,
                vi_lr=vi_lr,
                vi_guide=vi_guide,
                vi_posterior_draws=vi_posterior_draws,
                vi_progress_bar=vi_progress_bar,
            )
            return NUTSSampler(data, model, config)
        elif sampler_type == "mh":
            config = MetropolisHastingsConfig(
                **common_config_kwargs,
                target_accept_rate=target_accept_rate,
                adaptation_window=adaptation_window,
                adaptation_start=100,
                step_size_factor=1.1,
                min_step_size=1e-6,
                max_step_size=10.0,
            )
            return MetropolisHastingsSampler(data, model, config)
        else:
            raise ValueError(
                f"Unknown sampler_type '{sampler_type}' for univariate data. Choose 'nuts' or 'mh'."
            )

    elif isinstance(data, MultivarFFT):
        # Multivariate case (NUTS only for now)
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
            sampler_type = "multivar_blocked_nuts"

        if sampler_type == "multivar_blocked_nuts":
            config = MultivarBlockedNUTSConfig(
                **common_config_kwargs,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                init_from_vi=init_from_vi,
                vi_steps=vi_steps,
                vi_lr=vi_lr,
                vi_guide=vi_guide,
                vi_posterior_draws=vi_posterior_draws,
                vi_progress_bar=vi_progress_bar,
                alpha_phi_theta=alpha_phi_theta,
                beta_phi_theta=beta_phi_theta,
            )
            return MultivarBlockedNUTSSampler(data, model, config)

        config = MultivarNUTSConfig(
            **common_config_kwargs,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            init_from_vi=init_from_vi,
            vi_steps=vi_steps,
            vi_lr=vi_lr,
            vi_guide=vi_guide,
            vi_posterior_draws=vi_posterior_draws,
            vi_progress_bar=vi_progress_bar,
        )
        return MultivarNUTSSampler(data, model, config)

    else:
        raise ValueError(
            f"Unsupported data type: {type(data).__name__}. Expected Periodogram or MultivarFFT."
        )
