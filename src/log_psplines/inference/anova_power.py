"""GridTV ANOVA power model with separate mean and deviation priors.

In the eigenbases of the marginal penalties, ``sigma_g ~ HalfNormal(10)``
by default. Frequency mean coefficients have standard deviation
``sigma_g / sqrt(lambda_f + ridge_eps * sigma_g**2)``, except penalty null
modes use ``null_precision**-1/2``. Deviation coefficients have standard
deviation ``sigma_eta * (lambda_t + lambda_f + ridge_eps)**-1/2``; joint null
modes use ``sigma_eta``. ``sigma_eta ~ HalfNormal(0.5)`` by default.
Scalar ANOVA retains centered g/eta sites. Complex GridTV fields honor
``PowerConfig.centered`` with the same physical prior in either coordinate system.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

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
    sample_eigen_coefficients,
)
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.models.anova import ANOVALogPSpline, anova_components


def prepare_anova_prior(
    spline: ANOVALogPSpline, config: PowerConfig
) -> dict[str, np.ndarray]:
    """Factor the fixed penalties once; random field hyperparameters stay independent."""
    pair = whiten_penalty_pair(spline.time_penalty, spline.frequency.penalty)
    lam_t, lam_f = pair["lam_time"], pair["lam_freq"]
    pair["null_freq"] = lam_f <= 1e-10 * max(lam_f.max(), 1.0)
    pair["eta_scale"] = np.where(
        pair["joint_null"],
        1.0,
        (lam_t[:, None] + lam_f[None, :] + config.ridge_eps) ** -0.5,
    )
    return pair


def _mean_scale(
    pair: dict[str, np.ndarray],
    config: PowerConfig,
    sigma_g: float | jnp.ndarray,
) -> jnp.ndarray:
    """Conditional mean-field scale, including the fixed frequency null modes."""
    return jnp.where(
        jnp.asarray(pair["null_freq"]),
        config.null_precision**-0.5,
        sigma_g
        / jnp.sqrt(
            jnp.asarray(pair["lam_freq"]) + config.ridge_eps * sigma_g**2
        ),
    )


def anova_init_values(
    init: dict[str, np.ndarray | float],
    pair: dict[str, np.ndarray],
    config: PowerConfig,
    *,
    label: str = "",
) -> dict[str, np.ndarray | float]:
    """Map physical eigen-coefficients into the selected sampling coordinates."""
    suffix = f"_{label}" if label else ""
    values = dict(init)
    if not config.centered:
        values["z_g"] = values.pop("g") / np.asarray(
            _mean_scale(pair, config, init["sigma_g"])
        )
        values["z_eta"] = values.pop("eta") / (
            init["sigma_eta"] * pair["eta_scale"].reshape(-1)
        )
    return {f"{name}{suffix}": value for name, value in values.items()}


def sample_anova_field(
    pair: dict[str, np.ndarray],
    config: PowerConfig,
    interaction_scale: float,
    *,
    label: str = "",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample one independent scalar field in the existing ANOVA eigenbasis.

    Returns g (Kf,), eta (Kt,Kf). Labels suffix every random site and plate;
    only deterministic penalties/designs are shared between Cholesky fields.
    """
    suffix = f"_{label}" if label else ""
    sigma_g = numpyro.sample(
        f"sigma_g{suffix}", dist.HalfNormal(config.roughness_scale)
    )
    scale_g = _mean_scale(pair, config, sigma_g)
    g = sample_eigen_coefficients(
        f"{'g' if config.centered else 'z_g'}{suffix}",
        scale_g,
        (len(pair["lam_freq"]),),
        config,
    )
    if not config.centered:
        numpyro.deterministic(f"g{suffix}", g)
    sigma_eta = numpyro.sample(
        f"sigma_eta{suffix}", dist.HalfNormal(interaction_scale)
    )
    scale_eta = jnp.asarray(pair["eta_scale"])
    eta = sample_eigen_coefficients(
        f"{'eta' if config.centered else 'z_eta'}{suffix}",
        (sigma_eta * scale_eta).reshape(-1),
        (scale_eta.size,),
        config,
    )
    if not config.centered:
        numpyro.deterministic(f"eta{suffix}", eta)
    return jnp.asarray(g), jnp.asarray(eta).reshape(scale_eta.shape)


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
    """Build the scalar GridTV model, retaining centered g/eta sample sites."""
    if not data.is_grid:
        raise ValueError(
            "ANOVALogPSpline requires rectangular GridTV PowerData"
        )
    bt, bf = spline.design(data.time, data.frequency)
    pair = prepare_anova_prior(spline, config)
    bt_eig = jnp.asarray(bt @ pair["U_time"])
    bf_eig = jnp.asarray(bf @ pair["U_freq"])
    power = jnp.asarray(data.power)
    counts = jnp.asarray(data.counts)

    scalar_config = replace(config, centered=True)

    def model() -> None:
        g, eta = sample_anova_field(
            pair, scalar_config, spline.sigma_eta_prior
        )
        mean, deviation = anova_components(bt_eig, bf_eig, g, eta)
        correction = mean[None, :] + deviation
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
    posterior: xr.Dataset,
    pair: dict[str, np.ndarray],
    *,
    labels: Sequence[str] = ("",),
) -> xr.Dataset:
    """Add original-basis coefficients without copying existing draw arrays."""
    output = posterior.copy(deep=False)
    for label in labels:
        suffix = f"_{label}" if label else ""
        g = np.asarray(posterior[f"g{suffix}"])
        eta = np.asarray(posterior[f"eta{suffix}"]).reshape(
            *g.shape[:2], len(pair["lam_time"]), len(pair["lam_freq"])
        )
        output[f"weights_g{suffix}"] = xr.DataArray(
            np.einsum("ij,cdj->cdi", pair["U_freq"], g),
            dims=("chain", "draw", "frequency_coefficient"),
        )
        output[f"weights_eta{suffix}"] = xr.DataArray(
            np.einsum("ia,cdab,jb->cdij", pair["U_time"], eta, pair["U_freq"]),
            dims=(
                "chain",
                "draw",
                "time_coefficient",
                "frequency_coefficient",
            ),
        )
    return output
