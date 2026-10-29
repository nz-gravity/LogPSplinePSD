"""Gaussian integration and conditional recovery for ANOVA cross spectra."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from log_psplines.config import PowerConfig
from log_psplines.data.wishart_grid import WishartGridData
from log_psplines.inference.anova_power import _mean_scale
from log_psplines.inference.collapsed import gaussian_linear_conditional


def grid_field_design(bt, bf, *, paired: bool) -> np.ndarray:
    """Observation-by-coefficient design for [g, flattened eta] in eigenbases."""
    if paired:
        mean = bf
        interaction = (bt[:, :, None] * bf[:, None, :]).reshape(len(bt), -1)
    else:
        mean = np.broadcast_to(bf[None, :, :], (len(bt), *bf.shape)).reshape(
            -1, bf.shape[1]
        )
        interaction = (bt[:, None, :, None] * bf[None, :, None, :]).reshape(
            len(bt) * len(bf), -1
        )
    return np.column_stack((mean, interaction))


def prepare_grid_theta(
    data: WishartGridData,
    pair: dict[str, np.ndarray],
    design: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    channel: int,
) -> dict:
    """Build real/imaginary regression designs without any observation-square matrix.

    Fields are ordered theta_re(j,0), theta_im(j,0), theta_re(j,1), ...;
    coefficients within each field are its frequency g followed by flattened eta.
    """
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
        "frequency_size": len(pair["lam_freq"]),
        "field_size": field.shape[1],
    }


def grid_theta_scales(pair, config, sigma_g, sigma_eta):
    """Concatenate the existing independent ANOVA coefficient standard deviations."""
    return jnp.concatenate(
        [
            jnp.concatenate(
                (
                    _mean_scale(pair, config, sigma_g[index]),
                    sigma_eta[index]
                    * jnp.asarray(pair["eta_scale"]).reshape(-1),
                )
            )
            for index in range(len(sigma_g))
        ]
    )


def grid_theta_conditional(log_variance, prior_scales, prepared):
    """Return marginal log likelihood and the joint Gaussian theta conditional."""
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
        clip_log_psd=False,
    )


def recover_grid_theta(
    posterior: Mapping,
    pair: dict[str, np.ndarray],
    config: PowerConfig,
    prepared: dict,
    rng_key: jax.Array,
) -> dict[str, jax.Array]:
    """One joint conditional coefficient draw per retained marginal posterior draw."""
    channel = prepared["channel"]
    g = jnp.asarray(posterior[f"g_delta_{channel}"])
    eta = jnp.asarray(posterior[f"eta_delta_{channel}"])
    shape = g.shape[:2]
    coefficients = jnp.concatenate((g, eta), axis=-1).reshape(
        -1, prepared["field_size"]
    )
    sigma_g = jnp.stack(
        [
            jnp.asarray(posterior[f"sigma_g_{label}"]).reshape(-1)
            for label in prepared["theta_labels"]
        ],
        axis=-1,
    )
    sigma_eta = jnp.stack(
        [
            jnp.asarray(posterior[f"sigma_eta_{label}"]).reshape(-1)
            for label in prepared["theta_labels"]
        ],
        axis=-1,
    )
    keys = jax.random.split(rng_key, len(coefficients))

    def draw(inputs):
        delta, sg, se, key = inputs
        logs = prepared["field_basis"] @ delta
        scales = grid_theta_scales(pair, config, sg, se)
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
        draw, (coefficients, sigma_g, sigma_eta, keys)
    )
    output = {f"log_likelihood_block_{channel}": likelihood.reshape(shape)}
    size, kf = prepared["field_size"], prepared["frequency_size"]
    for index, label in enumerate(prepared["theta_labels"]):
        field = beta[:, index * size : (index + 1) * size]
        output[f"g_{label}"] = field[:, :kf].reshape(*shape, kf)
        output[f"eta_{label}"] = field[:, kf:].reshape(*shape, size - kf)
        if not config.centered:
            field = standardized[:, index * size : (index + 1) * size]
            output[f"z_g_{label}"] = field[:, :kf].reshape(*shape, kf)
            output[f"z_eta_{label}"] = field[:, kf:].reshape(*shape, size - kf)
    return output
