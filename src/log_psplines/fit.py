"""Public fitting orchestration for stationary and time-frequency spectra."""

from __future__ import annotations

import jax
import numpy as np
import xarray as xr

from log_psplines.basis import SplineBasis
from log_psplines.config import PowerConfig, StationaryConfig
from log_psplines.data.spectral import (
    PowerData,
    WishartData,
)
from log_psplines.inference.log_likelihood import compute_pointwise_lnl
from log_psplines.inference.model import prepare_model
from log_psplines.inference.nuts import run_multivariate_nuts
from log_psplines.inference.power import fit_power
from log_psplines.inference.vi import run_multivariate_vi
from log_psplines.models.anova import ANOVALogPSpline
from log_psplines.models.parametric import ParametricSpectrum
from log_psplines.models.reconstruction import reconstruct_stationary_spectrum
from log_psplines.models.spectrum import LogPSpline
from log_psplines.preprocessing.checks import _save_preprocessing_plot
from log_psplines.preprocessing.spectral import (
    align_true_psd_to_freq,
    preprocess_to_freq_domain,
)
from log_psplines.results import PSDResult, observed_wishart_data

from .logger import logger


def _fit_stationary(data, config: StationaryConfig) -> PSDResult:
    """Preprocess, sample, reconstruct and optionally save a Wishart fit."""
    if not isinstance(data, WishartData):
        data = preprocess_to_freq_domain(data, config)
    model_kwargs, spline_model = prepare_model(data, config)
    if config.outdir is not None:
        _save_preprocessing_plot(data, config, spline_model=spline_model)

    rng = (
        jax.random.PRNGKey(config.rng_key)
        if isinstance(config.rng_key, int)
        else config.rng_key
    )
    _, key = jax.random.split(rng)
    if config.method == "vi":
        vi = run_multivariate_vi(
            model_kwargs,
            rng_key=key,
            steps=config.vi_steps,
            lr=config.vi_lr,
            guide=config.vi_guide or "diag",
            posterior_draws=config.vi_posterior_draws,
            eta=float(config.eta),
            early_stopping=config.vi_early_stopping,
            verbose=(
                config.verbose
                if config.vi_progress_bar is None
                else config.vi_progress_bar
            ),
        )
        posterior = vi.posterior
        sample_stats = None
        log_likelihood = None
    else:
        logger.info(f"Spline model: {spline_model}")
        mcmc = run_multivariate_nuts(
            model_kwargs,
            rng_key=key,
            n_samples=config.n_samples,
            n_warmup=config.n_warmup,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            num_chains=config.num_chains,
            chain_method=config.chain_method,
            eta=float(config.eta),
            target_accept_prob_by_channel=config.target_accept_prob_by_channel,
            max_tree_depth_by_channel=config.max_tree_depth_by_channel,
            verbose=config.verbose,
        )
        log_likelihood = compute_pointwise_lnl(
            posterior=mcmc.posterior,
            data=data,
            model_kwargs=model_kwargs,
        )
        posterior = mcmc.posterior
        sample_stats = mcmc.sample_stats
        vi = None
    result = PSDResult(
        posterior=posterior,
        sample_stats=sample_stats,
        spectrum=reconstruct_stationary_spectrum(
            posterior, spline_model, data
        ),
        metadata={
            "data_type": "multivariate",
            "scaling_factor": float(data.scaling_factor or 1.0),
            "channel_stds": (
                None
                if data.channel_stds is None
                else np.asarray(data.channel_stds)
            ),
            "max_tree_depth": int(config.max_tree_depth),
            "max_tree_depth_by_channel": config.max_tree_depth_by_channel,
            "eta": float(config.eta),
            "sampling_eta": float(config.eta),
        },
        vi=vi,
        log_likelihood=log_likelihood,
        observed_data=observed_wishart_data(data),
    )
    result.spectrum_summary = xr.Dataset(
        {
            "quantiles": result.quantiles(),
            "coherence_quantiles": result.quantiles(kind="coherence"),
            "magnitude_quantiles": result.quantiles(kind="magnitude"),
        }
    )
    if config.outdir is not None:
        result.save(
            config.outdir,
            true_psd=align_true_psd_to_freq(config.true_psd, data),
        )
    return result


def fit(
    data,
    config=None,
    *,
    model=None,
    partition=None,
    reference=None,
    true_psd=None,
) -> PSDResult:
    """Fit stationary Wishart data or time-frequency powers.

    ``config.method`` selects NUTS (default) or NumPyro VI for spline models.
    Both use the same likelihood, prior and spectrum reconstruction.
    Scalar powers use the tensor or ANOVA structure in PowerConfig.
    ParametricSpectrum also supports joint independent channel powers.
    A partition may pool rectangular powers while retaining native-grid
    reconstruction. Scattered ordinates are evaluated at their exact points.
    For grid power data, reference is a fixed positive native-grid spectrum in
    the same units as the powers. The fitted spline models its log correction.
    true_psd is used only for post-fit analysis, never for inference.
    """
    if isinstance(data, PowerData):
        config = PowerConfig() if config is None else config
        if not isinstance(config, PowerConfig):
            raise TypeError("PowerData requires PowerConfig")
        if isinstance(model, ParametricSpectrum):
            if reference is not None or partition is not None:
                raise ValueError(
                    "ParametricSpectrum supplies its own projected and pooled spectrum"
                )
            from log_psplines.inference.parametric_power import (
                fit_parametric_power,
            )

            return fit_parametric_power(data, model, config, true_psd=true_psd)
        if data.power.ndim == 3:
            raise ValueError(
                "Fit scalar splines per channel; joint diagonal powers require ParametricSpectrum"
            )
        if model is None:

            def basis(grid, axis):
                knots = getattr(config, f"interior_knots_{axis}")
                kwargs = (
                    {"interior_knots": knots}
                    if knots is not None
                    else {
                        "n_interior_knots": getattr(
                            config, f"n_interior_knots_{axis}"
                        )
                    }
                )
                return SplineBasis.from_grid(
                    grid,
                    degree=getattr(config, f"degree_{axis}"),
                    penalty_order=getattr(config, f"penalty_order_{axis}"),
                    **kwargs,
                )

            if data.is_grid:
                time_grid, frequency_grid = data.time, data.frequency
            else:
                time_grid = np.unique(data.time)
                frequency_grid = np.unique(data.frequency)
            frequency_basis, time_basis = (
                basis(frequency_grid, "freq"),
                basis(time_grid, "time"),
            )
            model = (
                ANOVALogPSpline(
                    frequency_basis,
                    time_basis,
                    sigma_eta_prior=config.interaction_scale,
                )
                if config.structure == "anova"
                else LogPSpline(frequency_basis, time=time_basis)
            )
        return fit_power(
            data,
            model,
            config,
            partition=partition,
            reference=reference,
            true_psd=true_psd,
        )
    if reference is not None:
        raise ValueError("reference requires grid PowerData")
    if true_psd is not None:
        raise ValueError("true_psd requires grid PowerData")
    if partition is not None:
        raise ValueError("partition requires PowerData")
    if model is not None:
        raise ValueError("explicit model requires PowerData")
    config = StationaryConfig() if config is None else config
    if not isinstance(config, StationaryConfig):
        raise TypeError("stationary fitting requires StationaryConfig")
    return _fit_stationary(data, config)
