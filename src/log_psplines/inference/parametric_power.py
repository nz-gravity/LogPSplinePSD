"""Generic independent-power inference for supplied parametric spectra."""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr

from log_psplines.inference.power import _run_power_nuts
from log_psplines.inference.power_results import power_result_spectra
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.results import PSDResult, observed_power_data


def prepare_parametric_power_model(data, model):
    """Construct the shared Whittle likelihood around a deterministic spectrum."""
    if not data.is_grid:
        raise ValueError("ParametricSpectrum requires grid PowerData")
    initial = np.asarray(model.spectrum(model.initial_values, slice(None)))
    if (
        initial.shape != data.power.shape
        or not np.isfinite(initial).all()
        or np.any(initial <= 0)
    ):
        raise ValueError(
            "initial spectrum must be finite, positive and match powers"
        )
    power, counts = jnp.asarray(data.power), jnp.asarray(data.counts)

    def bayesian_model():
        parameters = {
            name: numpyro.sample(name, prior)
            for name, prior in model.priors.items()
        }
        variance = model.spectrum(parameters, slice(None))
        log_like = power_whittle_log_likelihood(
            power, counts, jnp.log(variance)
        )
        numpyro.deterministic("log_likelihood", log_like)
        numpyro.factor("whittle", log_like)

    return bayesian_model


def fit_parametric_power(data, model, config, *, true_psd=None):
    """Sample joint parameters; retain chains and all-draw spectral summaries."""
    if config.method != "nuts":
        raise NotImplementedError(
            "ParametricSpectrum currently supports NUTS only"
        )
    truth = None
    if true_psd is not None:
        values = np.asarray(true_psd, dtype=float)
        if (
            values.shape != data.power.shape
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            raise ValueError(
                "true_psd must be finite, positive and match powers"
            )
        dims = ("time", "frequency") + (
            ("channel",) if data.power.ndim == 3 else ()
        )
        coords = {"time": data.time, "frequency": data.frequency}
        if data.channels is not None:
            coords["channel"] = list(data.channels)
        truth = xr.DataArray(values, dims=dims, coords=coords)
    result = _run_power_nuts(
        prepare_parametric_power_model(data, model),
        dict(model.initial_values),
        config,
    )
    samples = {
        name: jnp.asarray(result.posterior[name]) for name in model.priors
    }

    def evaluate(frequency_slice):
        values = jax.vmap(
            jax.vmap(lambda pars: model.spectrum(pars, frequency_slice))
        )(samples)
        return values[..., None] if data.power.ndim == 2 else values

    spectrum, summary = power_result_spectra(
        result.posterior,
        evaluate,
        data.time,
        data.frequency,
        data.channels or ("0",),
        config,
    )
    return PSDResult(
        posterior=result.posterior,
        spectrum=spectrum,
        spectrum_summary=summary,
        sample_stats=result.sample_stats,
        log_likelihood=result.log_likelihood,
        observed_data=observed_power_data(data),
        truth=truth,
        metadata={
            **asdict(config),
            "data_type": "power",
            "model": "parametric",
            "likelihood": "diagonal_power_whittle",
            "units": data.units,
            "prior_distributions": str(dict(model.priors)),
        },
    )
