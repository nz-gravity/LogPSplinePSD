"""Tensor-spline Gaussian priors and collapsed complex matrix regressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from jax.scipy.linalg import solve_triangular

from log_psplines.basis.penalty import whiten_penalty_pair
from log_psplines.config import PowerConfig
from log_psplines.data.wishart_grid import WishartGridData
from log_psplines.inference.collapsed import gaussian_linear_conditional
from log_psplines.models.spectrum import LogPSpline


def prepare_grid_prior(spline: LogPSpline) -> dict:
    """Diagonalize Q = Pt/sigma_time² (x) I + I (x) Pf/sigma_freq².

    Pf is the supplied positive-definite frequency penalty, including its
    existing ridge and normalization. No extra ridge or null-space prior is
    added. With time=None, Bt=1 and Pt=0: Q is exactly Pf/sigma_freq², the
    stationary prior when basis and roughness hyperprior also match.
    """
    pf = np.asarray(spline.frequency.penalty, dtype=float)
    np.linalg.cholesky(pf)
    pt = (
        np.zeros((1, 1))
        if spline.time is None
        else np.asarray(spline.time.penalty, dtype=float)
    )
    pair = whiten_penalty_pair(pt, pf)
    if np.any(pair["lam_freq"] <= 0):
        raise ValueError(
            "matrix tensor frequency penalty must be positive definite"
        )
    pair["time_active"] = bool(np.any(pt))
    return pair


def _basis_at(basis, coordinates):
    coordinates = np.asarray(coordinates)
    if np.array_equal(coordinates, basis.grid):
        return np.asarray(basis.basis)
    indices = np.searchsorted(basis.grid, coordinates)
    if np.all(indices < len(basis.grid)) and np.array_equal(
        basis.grid[indices], coordinates
    ):
        return np.asarray(basis.basis)[indices]
    if basis.knot_convention != "clamped":
        raise ValueError("off-grid matrix splines require clamped bases")
    if coordinates.min() < basis.grid[0] or coordinates.max() > basis.grid[-1]:
        raise ValueError(
            "evaluation coordinates must be inside the spline domain"
        )
    return np.asarray(basis.design_at(coordinates))


def grid_design(data, spline, pair):
    """Raw and penalty-eigenbasis designs at the exact observation coordinates."""
    bt = (
        np.ones((len(data.time), 1))
        if spline.time is None
        else _basis_at(spline.time, data.time)
    )
    bf = _basis_at(spline.frequency, data.frequency)
    return bt, bf, bt @ pair["U_time"], bf @ pair["U_freq"]


def grid_field_design(bt, bf, *, paired: bool) -> np.ndarray:
    """Tensor design with time-major flattened coefficient ordering."""
    if paired:
        return (bt[:, :, None] * bf[:, None, :]).reshape(len(bt), -1)
    return (bt[:, None, :, None] * bf[None, :, None, :]).reshape(
        len(bt) * len(bf), -1
    )


def grid_field_scales(pair, sigma_freq, sigma_time=None):
    """Eigen-coefficient scales of the supplied proper tensor precision."""
    precision = jnp.asarray(pair["lam_freq"])[None, :] / sigma_freq**2
    precision = jnp.broadcast_to(
        precision, (len(pair["lam_time"]), len(pair["lam_freq"]))
    )
    if pair["time_active"]:
        precision = (
            precision + jnp.asarray(pair["lam_time"])[:, None] / sigma_time**2
        )
    return jax.lax.rsqrt(precision).reshape(-1)


def sample_grid_scales(pair, config, label):
    sigma_freq = numpyro.sample(
        f"sigma_freq_{label}", dist.HalfNormal(config.roughness_scale)
    )
    sigma_time = None
    if pair["time_active"]:
        sigma_time = numpyro.sample(
            f"sigma_time_{label}", dist.HalfNormal(config.roughness_scale)
        )
    return sigma_freq, sigma_time


def grid_coordinate_scales(prior_scales, information):
    """Smoothly interpolate between prior and likelihood scales per mode.

    The fixed information is the diagonal expected log-variance Fisher
    information, sum(count * basis**2) / enbw. This is only a change of
    sampling coordinates: the normal density below includes its Jacobian.
    """
    return jax.lax.rsqrt(prior_scales**-2 + information)


def sample_grid_field(pair, config, label, information):
    """Sample delta in information-scaled coordinates with its exact prior."""
    sigma_freq, sigma_time = sample_grid_scales(pair, config, label)
    scales = grid_field_scales(pair, sigma_freq, sigma_time)
    coordinate_scales = grid_coordinate_scales(scales, information)
    z = numpyro.sample(
        f"z_{label}", dist.Normal(0.0, scales / coordinate_scales).to_event(1)
    )
    return numpyro.deterministic(f"s_{label}", z * coordinate_scales)


def grid_theta_scales(pair, sigma_freq, sigma_time=None):
    return jnp.concatenate(
        [
            grid_field_scales(
                pair,
                sigma_freq[index],
                None if sigma_time is None else sigma_time[index],
            )
            for index in range(len(sigma_freq))
        ]
    )


def prepare_grid_theta(
    data: WishartGridData, pair: dict, design: tuple, channel: int
) -> dict:
    """Form fixed real/imaginary designs for jointly integrated theta fields."""
    if not jax.config.x64_enabled:
        raise ValueError(
            "Multivariate Gaussian marginalization requires JAX_ENABLE_X64=true"
        )
    field = grid_field_design(design[2], design[3], paired=data.is_scattered)
    real = data.u_re.reshape(-1, data.p, data.u_re.shape[-1])
    imag = data.u_im.reshape(real.shape)
    designs_re, designs_im, labels = [], [], []
    for previous in range(channel):
        for part in ("re", "im"):
            labels.append(f"theta_{part}_{channel}_{previous}")
            designs_re.append(
                (real[:, previous] if part == "re" else -imag[:, previous])[
                    :, :, None
                ]
                * field[:, None, :]
            )
            designs_im.append(
                (imag[:, previous] if part == "re" else real[:, previous])[
                    :, :, None
                ]
                * field[:, None, :]
            )
    return {
        "channel": channel,
        "field_basis": jnp.asarray(field),
        "design_re": jnp.asarray(np.concatenate(designs_re, axis=-1)),
        "design_im": jnp.asarray(np.concatenate(designs_im, axis=-1)),
        "observed_re": jnp.asarray(real[:, channel]),
        "observed_im": jnp.asarray(imag[:, channel]),
        "counts": jnp.asarray(data.counts.reshape(-1)),
        "enbw": data.enbw,
        "theta_labels": tuple(labels),
        "field_size": field.shape[1],
    }


def grid_theta_conditional(log_variance, prior_scales, prepared):
    inverse_scale = 1 / jnp.asarray(prior_scales)
    return gaussian_linear_conditional(
        jnp.asarray(log_variance).reshape(-1),
        prepared["counts"] / prepared["enbw"],
        design_re=prepared["design_re"],
        design_im=prepared["design_im"],
        observed_re=prepared["observed_re"],
        observed_im=prepared["observed_im"],
        prior_precision=jnp.diag(inverse_scale**2),
        prior_whitener=jnp.diag(inverse_scale),
        prior_logdet=2 * jnp.sum(jnp.log(inverse_scale)),
        precision_multiplier=1 / prepared["enbw"],
    )


def recover_grid_theta(
    posterior: Mapping,
    pair: dict,
    config: PowerConfig,
    prepared: dict,
    rng_key: jax.Array,
) -> dict[str, jax.Array]:
    """Recover one joint theta coefficient draw per retained marginal draw."""
    channel = prepared["channel"]
    delta = jnp.asarray(posterior[f"s_delta_{channel}"])
    shape = delta.shape[:2]
    delta = delta.reshape(-1, prepared["field_size"])
    sigma_freq = jnp.stack(
        [
            jnp.asarray(posterior[f"sigma_freq_{label}"]).reshape(-1)
            for label in prepared["theta_labels"]
        ],
        axis=-1,
    )
    sigma_time = (
        jnp.stack(
            [
                jnp.asarray(posterior[f"sigma_time_{label}"]).reshape(-1)
                for label in prepared["theta_labels"]
            ],
            axis=-1,
        )
        if pair["time_active"]
        else jnp.zeros_like(sigma_freq)
    )
    keys = jax.random.split(rng_key, len(delta))

    def draw(inputs):
        coefficients, sf, st, key = inputs
        logs = prepared["field_basis"] @ coefficients
        scales = grid_theta_scales(pair, sf, st)
        _, mean, chol, diagonal = grid_theta_conditional(
            logs, scales, prepared
        )
        beta = (
            mean
            + solve_triangular(
                chol.T,
                jax.random.normal(key, mean.shape, dtype=mean.dtype),
                lower=False,
            )
            / diagonal
        )
        active = prepared["counts"] > 0
        logs = jnp.where(active, logs, 0.0)
        residual_re = prepared["observed_re"] - jnp.einsum(
            "frk,k->fr", prepared["design_re"], beta
        )
        residual_im = prepared["observed_im"] - jnp.einsum(
            "frk,k->fr", prepared["design_im"], beta
        )
        power = jnp.where(
            active, jnp.sum(residual_re**2 + residual_im**2, axis=-1), 0.0
        )
        likelihood = (
            -jnp.sum(prepared["counts"] * logs + power * jnp.exp(-logs))
            / prepared["enbw"]
        )
        return beta, beta / scales, likelihood

    beta, standardized, likelihood = jax.lax.map(
        draw, (delta, sigma_freq, sigma_time, keys)
    )
    output = {f"log_likelihood_block_{channel}": likelihood.reshape(shape)}
    size = prepared["field_size"]
    for index, label in enumerate(prepared["theta_labels"]):
        output[f"s_{label}"] = beta[
            :, index * size : (index + 1) * size
        ].reshape(*shape, size)
        output[f"z_{label}"] = standardized[
            :, index * size : (index + 1) * size
        ].reshape(*shape, size)
    return output


def collect_grid_samples(
    posterior: xr.Dataset, pair: dict, *, labels: Sequence[str]
) -> xr.Dataset:
    """Rotate physical eigen-coefficients into original tensor spline weights."""
    output = posterior.copy(deep=False)
    kt, kf = len(pair["lam_time"]), len(pair["lam_freq"])
    for label in labels:
        coefficients = np.asarray(posterior[f"s_{label}"]).reshape(
            *posterior[f"s_{label}"].shape[:2], kt, kf
        )
        output[f"weights_{label}"] = xr.DataArray(
            np.einsum(
                "ia,cdab,jb->cdij",
                pair["U_time"],
                coefficients,
                pair["U_freq"],
            ),
            dims=(
                "chain",
                "draw",
                "time_coefficient",
                "frequency_coefficient",
            ),
        )
    return output
