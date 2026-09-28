"""GridTV ANOVA power model with separate mean and deviation priors.

In the eigenbases of the marginal penalties, ``sigma_g ~ HalfNormal(10)``
by default. Frequency mean coefficients have standard deviation
``sigma_g / sqrt(lambda_f + ridge_eps * sigma_g**2)``, except penalty null
modes use ``null_precision**-1/2``. Deviation coefficients have standard
deviation ``sigma_eta * (lambda_t + lambda_f + ridge_eps)**-1/2``; joint null
modes use ``sigma_eta``. ``sigma_eta ~ HalfNormal(0.5)`` by default. The
deviation uses a centered coefficient hierarchy, independent of the tensor
model's ``PowerConfig.centered`` setting.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from log_psplines.basis.penalty import whiten_penalty_pair
from log_psplines.config import PowerConfig
from log_psplines.data.spectral import PowerData
from log_psplines.inference.power import (
    _mean_power_for_masked_initialization,
    power_floor,
)
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.models.anova import ANOVALogPSpline


def initialize_anova(
    data: PowerData,
    basis_time: np.ndarray,
    basis_freq: np.ndarray,
    penalty_time: np.ndarray,
    penalty_freq: np.ndarray,
    pair: dict[str, np.ndarray],
    config: PowerConfig,
) -> dict[str, np.ndarray | float]:
    """Split log mean power, then fit g and eta with penalized least squares."""
    mean_power = _mean_power_for_masked_initialization(data.power, data.counts)
    target = np.log(mean_power + power_floor(mean_power))
    weights = np.asarray(data.counts, dtype=float)
    denominator = weights.sum(axis=0)
    g_target = np.divide(
        (target * weights).sum(axis=0),
        denominator,
        out=np.mean(target, axis=0),
        where=denominator > 0,
    )
    bf_eig = basis_freq @ pair["U_freq"]
    lam_f = pair["lam_freq"]
    g_system = (
        bf_eig.T @ bf_eig
        + config.init_penalty_freq * np.diag(lam_f)
        + config.ridge_eps * np.eye(len(lam_f))
    )
    g = np.linalg.solve(g_system, bf_eig.T @ g_target)
    sigma_g = min(10.0, np.sqrt((float(np.sum(lam_f * g**2)) + 1e-6) / g.size))
    residual = target - (bf_eig @ g)[None, :]
    kt, kf = basis_time.shape[1], basis_freq.shape[1]
    system = (
        np.kron(basis_freq.T @ basis_freq, basis_time.T @ basis_time)
        + config.init_penalty_time * np.kron(np.eye(kf), penalty_time)
        + config.init_penalty_freq * np.kron(penalty_freq, np.eye(kt))
        + config.ridge_eps * np.eye(kt * kf)
    )
    rhs = (basis_time.T @ residual @ basis_freq).reshape(-1, order="F")
    coefficients = np.linalg.solve(system, rhs).reshape((kt, kf), order="F")
    eta = pair["U_time"].T @ coefficients @ pair["U_freq"]
    eta_surface = basis_time @ coefficients @ basis_freq.T
    return {
        "g": g,
        "eta": eta.reshape(-1),
        "sigma_g": sigma_g,
        "sigma_eta": float(np.clip(np.std(eta_surface), 0.05, 1.0)),
    }


def prepare_anova_power_model(
    data: PowerData, spline: ANOVALogPSpline, config: PowerConfig
) -> tuple[Callable, dict[str, np.ndarray], dict[str, np.ndarray | float]]:
    """Build the centered GridTV likelihood and independent priors."""
    if not data.is_grid:
        raise ValueError(
            "ANOVALogPSpline requires rectangular GridTV PowerData"
        )
    for basis, grid in (
        (spline.time, data.time),
        (spline.frequency, data.frequency),
    ):
        if (
            not np.array_equal(grid, basis.grid)
            and basis.knot_convention != "clamped"
        ):
            raise ValueError(
                "partitioned ANOVA requires SplineBasis.from_grid bases"
            )
    bt = (
        spline.time_basis
        if np.array_equal(data.time, spline.time.grid)
        else np.asarray(spline.time.design_at(data.time))
        @ spline.time_transform
    )
    bf = np.asarray(
        spline.frequency.basis
        if np.array_equal(data.frequency, spline.frequency.grid)
        else spline.frequency.design_at(data.frequency)
    )
    pair = whiten_penalty_pair(spline.time_penalty, spline.frequency.penalty)
    bt_eig = jnp.asarray(bt @ pair["U_time"])
    bf_eig = jnp.asarray(bf @ pair["U_freq"])
    lam_t = jnp.asarray(pair["lam_time"])
    lam_f = jnp.asarray(pair["lam_freq"])
    null_f = jnp.asarray(
        pair["lam_freq"] <= 1e-10 * max(pair["lam_freq"].max(), 1.0)
    )
    joint_null = jnp.asarray(pair["joint_null"])
    power = jnp.asarray(data.power)
    counts = jnp.asarray(data.counts)

    def model() -> None:
        # The frequency null space retains a fixed scale.
        sigma_g = numpyro.sample(
            "sigma_g", dist.HalfNormal(config.roughness_scale)
        )
        scale_g = jnp.where(
            null_f,
            config.null_precision**-0.5,
            sigma_g / jnp.sqrt(lam_f + config.ridge_eps * sigma_g**2),
        )
        with numpyro.plate("g_plate", len(pair["lam_freq"])):
            g = numpyro.sample("g", dist.Normal(0.0, scale_g))
        scale_eta = jnp.where(
            joint_null,
            1.0,
            (lam_t[:, None] + lam_f[None, :] + config.ridge_eps) ** -0.5,
        )
        sigma_eta = numpyro.sample(
            "sigma_eta", dist.HalfNormal(spline.sigma_eta_prior)
        )
        with numpyro.plate("eta_plate", scale_eta.size):
            eta = numpyro.sample(
                "eta", dist.Normal(0.0, (sigma_eta * scale_eta).reshape(-1))
            )
        eta = eta.reshape(scale_eta.shape)
        correction = (bf_eig @ g)[None, :] + jnp.einsum(
            "ti,ij,fj->tf", bt_eig, eta, bf_eig, optimize="optimal"
        )
        log_like = power_whittle_log_likelihood(power, counts, correction)
        numpyro.deterministic("log_likelihood", log_like)
        numpyro.factor("whittle", log_like)

    init = initialize_anova(
        data,
        bt,
        bf,
        spline.time_penalty,
        np.asarray(spline.frequency.penalty),
        pair,
        config,
    )
    return model, pair, init


def collect_anova_samples(
    posterior: xr.Dataset, pair: dict[str, np.ndarray]
) -> xr.Dataset:
    """Expose reconstruction-grid coefficients in the original spline bases."""
    output = posterior.copy()
    g = np.asarray(output["g"])
    eta = np.asarray(output["eta"]).reshape(
        *g.shape[:2], len(pair["lam_time"]), len(pair["lam_freq"])
    )
    output["weights_g"] = xr.DataArray(
        np.einsum("ij,cdj->cdi", pair["U_freq"], g),
        dims=("chain", "draw", "frequency_coefficient"),
    )
    output["weights_eta"] = xr.DataArray(
        np.einsum("ia,cdab,jb->cdij", pair["U_time"], eta, pair["U_freq"]),
        dims=("chain", "draw", "time_coefficient", "frequency_coefficient"),
    )
    return output
