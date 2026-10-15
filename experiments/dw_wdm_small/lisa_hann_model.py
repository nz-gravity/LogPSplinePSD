"""Small paired-site adapter for the existing scalar ANOVA prior/likelihood.

Only the predictor geometry differs from rectangular GridTV. Initialization
uses a per-cycle grid; the likelihood uses every ordinate's exact coordinate.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from log_psplines import ANOVALogPSpline, PowerConfig, PowerData
from log_psplines.inference.anova_power import (
    _mean_scale,
    initialize_anova,
    prepare_anova_prior,
    sample_anova_field,
)
from log_psplines.inference.power import sample_eigen_coefficients
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood


def prepare_paired_model(
    data: PowerData,
    spline: ANOVALogPSpline,
    config: PowerConfig,
    cycle_time: np.ndarray,
    *,
    noncentered_eta: bool = False,
):
    """Return the unchanged ANOVA prior, paired predictor, and data-only start."""
    assert not data.is_grid and config.centered
    bt, _ = spline.design(data.time, data.frequency)
    frequency, first, inverse = np.unique(
        data.frequency, return_index=True, return_inverse=True
    )
    bf = np.asarray(spline.frequency.design_at(frequency))
    pair = prepare_anova_prior(spline, config)
    bt_eigen = jnp.asarray(bt @ pair["U_time"])
    bf_eigen = jnp.asarray(bf @ pair["U_freq"])
    frequency_index = jnp.asarray(inverse)
    power, counts = jnp.asarray(data.power), jnp.asarray(data.counts)

    def evaluate(g, eta):
        mean = bf_eigen @ g
        projection = bf_eigen @ eta.T
        return mean[frequency_index] + jnp.sum(
            bt_eigen * projection[frequency_index], axis=1
        )

    def model():
        if noncentered_eta:
            sigma_g = numpyro.sample(
                "sigma_g", dist.HalfNormal(config.roughness_scale)
            )
            g = sample_eigen_coefficients(
                "g",
                _mean_scale(pair, config, sigma_g),
                (len(pair["lam_freq"]),),
                config,
            )
            sigma_eta = numpyro.sample(
                "sigma_eta", dist.HalfNormal(spline.sigma_eta_prior)
            )
            scale = sigma_eta * jnp.asarray(pair["eta_scale"])
            physical = sample_eigen_coefficients(
                "z_eta",
                scale.reshape(-1),
                (scale.size,),
                replace(config, centered=False),
            )
            eta = numpyro.deterministic("eta", physical).reshape(scale.shape)
        else:
            g, eta = sample_anova_field(pair, config, spline.sigma_eta_prior)
        likelihood = power_whittle_log_likelihood(
            power, counts, evaluate(g, eta)
        )
        numpyro.deterministic("log_likelihood", likelihood)
        numpyro.factor("whittle", likelihood)

    shape = (len(cycle_time), len(frequency))
    assert len(data.power) == np.prod(shape)
    grid = PowerData(
        np.asarray(data.power).reshape(shape),
        np.asarray(data.counts).reshape(shape),
        frequency,
        cycle_time,
    )
    bt_init, bf_init = spline.design(cycle_time, frequency)
    init = initialize_anova(
        grid,
        bt_init,
        bf_init,
        spline.time_penalty,
        np.asarray(spline.frequency.penalty),
        pair,
        config,
    )
    if noncentered_eta:
        init["z_eta"] = init.pop("eta") / (
            init["sigma_eta"] * pair["eta_scale"].reshape(-1)
        )
    return model, pair, init, evaluate
