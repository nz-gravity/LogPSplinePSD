"""Exact Gaussian integration of stationary Cholesky regression coefficients.

Conditional on a row's log innovation spectrum and smoothing scales, its
complex regression coefficients form a Gaussian linear model. Integrating
those coefficients reduces the state for NUTS and VI without changing the
posterior target.
Independent conditional draws recover the ordinary coefficient posterior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular
from scipy.linalg import block_diag

from log_psplines.inference.model import _sample_pspline_block
from log_psplines.likelihoods.whittle import LOG_SPECTRUM_LIMIT


def prepare_collapsed_channel(channel_kwargs: Mapping[str, Any]) -> dict:
    """Prepare fixed design operators for a row with preceding channels.

    ``channel_kwargs`` is the output of ``channel_model_kwargs``. Every
    real/imaginary component may have its own basis and coefficient count.
    Preparation runs on the host; its returned dictionary is model kwargs.
    The first Cholesky row has no regression to integrate and stays explicit.
    """
    if not jax.config.x64_enabled:
        raise ValueError(
            "Multivariate Gaussian marginalization requires JAX_ENABLE_X64=true"
        )
    j = int(channel_kwargs["channel_index"])
    if j < 1:
        raise ValueError("collapsed rows require a preceding channel")
    previous_re = np.asarray(channel_kwargs["u_re_prev"])
    previous_im = np.asarray(channel_kwargs["u_im_prev"])
    names, sizes, penalties, designs_re, designs_im = [], [], [], [], []
    for previous in range(j):
        for part in ("re", "im"):
            basis = np.asarray(
                channel_kwargs[f"basis_theta_{part}_by_component"][previous]
            )
            penalty = np.asarray(
                channel_kwargs[f"penalty_theta_{part}_by_component"][previous],
                dtype=np.float64,
            )
            if not np.isfinite(penalty).all():
                raise ValueError("theta penalties must be finite")
            # Cholesky verifies positive definiteness without silently changing
            # the Gaussian prior with an extra numerical ridge.
            np.linalg.cholesky(penalty)
            names.append(f"theta_{part}_{j}_{previous}")
            sizes.append(penalty.shape[0])
            penalties.append(penalty)
            real = previous_re[:, previous, :]
            imag = previous_im[:, previous, :]
            designs_re.append(
                (real if part == "re" else -imag)[:, :, None]
                * basis[:, None, :]
            )
            designs_im.append(
                (imag if part == "re" else real)[:, :, None]
                * basis[:, None, :]
            )
    penalty = block_diag(*penalties)
    penalty_cholesky = np.linalg.cholesky(penalty)
    return {
        "channel_index": j,
        "basis_delta": channel_kwargs["basis_delta"],
        "penalty_delta": channel_kwargs["penalty_delta"],
        "roughness_scale": channel_kwargs["roughness_scale"],
        "smoothing_parameterization": channel_kwargs.get(
            "smoothing_parameterization", "centered"
        ),
        "duration": channel_kwargs["duration"],
        "Nb": channel_kwargs["Nb"],
        "Nh": channel_kwargs["Nh"],
        "enbw": channel_kwargs.get("enbw", 1.0),
        "eta": channel_kwargs.get("eta", 1.0),
        "design_re": jnp.asarray(np.concatenate(designs_re, axis=-1)),
        "design_im": jnp.asarray(np.concatenate(designs_im, axis=-1)),
        "observed_re": jnp.asarray(channel_kwargs["u_re_channel"]),
        "observed_im": jnp.asarray(channel_kwargs["u_im_channel"]),
        "theta_names": tuple(names),
        "theta_sizes": tuple(sizes),
        "theta_penalty": jnp.asarray(penalty),
        "theta_penalty_cholesky": jnp.asarray(penalty_cholesky),
        "theta_logdet_penalty": float(
            2 * np.sum(np.log(np.diag(penalty_cholesky)))
        ),
    }


def gaussian_theta_conditional(
    log_delta_sq: jax.Array,
    sigmas: jax.Array,
    *,
    design_re: jax.Array,
    design_im: jax.Array,
    observed_re: jax.Array,
    observed_im: jax.Array,
    theta_sizes: tuple[int, ...],
    theta_penalty: jax.Array,
    theta_penalty_cholesky: jax.Array,
    theta_logdet_penalty: float,
    Nb: int,
    Nh: int,
    duration: float,
    enbw: float = 1.0,
    eta: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return marginal log L, conditional mean, scaled Cholesky, and scales.

    With prior precision Q and weighted design Gram matrix A, the conditional
    precision is H=Q+2A. We factor D^-1 H D^-1 with D=sqrt(diag(H)) to improve
    conditioning. Its conditional covariance is D^-1 L^-T L^-1 D^-1.

    The marginal quadratic is evaluated at the conditional mean as residual
    power plus prior quadratic. This avoids subtracting two large quadratic
    forms when a LISA channel is almost determined by preceding channels.
    """
    log_variance = jnp.clip(
        log_delta_sq, -LOG_SPECTRUM_LIMIT, LOG_SPECTRUM_LIMIT
    )
    likelihood_scale = eta / enbw
    frequency_precision = likelihood_scale * jnp.exp(-log_variance) / duration
    expanded_sigma = jnp.repeat(
        sigmas,
        jnp.asarray(theta_sizes),
        total_repeat_length=theta_penalty.shape[0],
    )
    prior_precision = theta_penalty / (
        expanded_sigma[:, None] * expanded_sigma[None, :]
    )
    gram = jnp.einsum(
        "frk,frl,f->kl", design_re, design_re, frequency_precision
    ) + jnp.einsum("frk,frl,f->kl", design_im, design_im, frequency_precision)
    linear = 2 * (
        jnp.einsum("frk,fr,f->k", design_re, observed_re, frequency_precision)
        + jnp.einsum(
            "frk,fr,f->k", design_im, observed_im, frequency_precision
        )
    )
    precision = prior_precision + 2 * gram
    scale = jnp.sqrt(jnp.diag(precision))
    balanced = precision / (scale[:, None] * scale[None, :])
    cholesky = jnp.linalg.cholesky(balanced)
    intermediate = solve_triangular(cholesky, linear / scale, lower=True)
    mean = solve_triangular(cholesky.T, intermediate, lower=False) / scale
    residual_re = observed_re - jnp.einsum("frk,k->fr", design_re, mean)
    residual_im = observed_im - jnp.einsum("frk,k->fr", design_im, mean)
    residual_power = jnp.sum(
        (residual_re**2 + residual_im**2) * frequency_precision[:, None]
    )
    logdet_prior = theta_logdet_penalty - 2 * jnp.sum(
        jnp.asarray(theta_sizes) * jnp.log(sigmas)
    )
    logdet_conditional = 2 * (
        jnp.sum(jnp.log(scale)) + jnp.sum(jnp.log(jnp.diag(cholesky)))
    )
    marginal = (
        -likelihood_scale * Nb * Nh * jnp.sum(log_variance)
        - residual_power
        - 0.5
        * jnp.sum((theta_penalty_cholesky.T @ (mean / expanded_sigma)) ** 2)
        + 0.5 * (logdet_prior - logdet_conditional)
    )
    return marginal, mean, cholesky, scale


def _conditional_kwargs(prepared: Mapping[str, Any]) -> dict:
    return {
        key: prepared[key]
        for key in (
            "design_re",
            "design_im",
            "observed_re",
            "observed_im",
            "theta_sizes",
            "theta_penalty",
            "theta_penalty_cholesky",
            "theta_logdet_penalty",
            "Nb",
            "Nh",
            "duration",
            "enbw",
            "eta",
        )
    }


def collapsed_channel_model(
    channel_index: int,
    basis_delta: jax.Array,
    penalty_delta: jax.Array,
    roughness_scale: float,
    theta_names: tuple[str, ...],
    smoothing_parameterization: str = "centered",
    **conditional_kwargs: Any,
) -> None:
    """Sample delta coefficients and all scales, integrating theta weights."""
    weights = _sample_pspline_block(
        f"sigma_delta_{channel_index}",
        f"weights_delta_{channel_index}",
        penalty_delta,
        roughness_scale,
        smoothing_parameterization,
    )
    sigmas = jnp.stack(
        [
            numpyro.sample(f"sigma_{name}", dist.HalfNormal(roughness_scale))
            for name in theta_names
        ]
    )
    marginal, _, _, _ = gaussian_theta_conditional(
        basis_delta @ weights, sigmas, **conditional_kwargs
    )
    numpyro.factor(f"likelihood_channel_{channel_index}", marginal)


def sample_conditional_theta(
    posterior: Mapping[str, Any],
    prepared_kwargs: Mapping[str, Any],
    rng_key: jax.Array,
) -> dict[str, jax.Array]:
    """Recover theta weights and full likelihood for every retained draw.

    Inputs have arbitrary leading sample dimensions, conventionally
    (chain, draw, coefficient). Each hyperparameter draw gets one independent
    conditional Gaussian draw. A sequential ``lax.map`` limits intermediate
    memory to one precision matrix rather than chains*draws such matrices.
    """
    j = int(prepared_kwargs["channel_index"])
    weights = jnp.asarray(posterior[f"weights_delta_{j}"])
    leading_shape = weights.shape[:-1]
    flat_weights = weights.reshape(-1, weights.shape[-1])
    names = prepared_kwargs["theta_names"]
    sigmas = jnp.stack(
        [
            jnp.asarray(posterior[f"sigma_{name}"]).reshape(-1)
            for name in names
        ],
        axis=-1,
    )
    log_delta = flat_weights @ jnp.asarray(prepared_kwargs["basis_delta"]).T
    keys = jax.random.split(rng_key, flat_weights.shape[0])
    conditional_kwargs = _conditional_kwargs(prepared_kwargs)

    def draw(inputs):
        logd, scales, key = inputs
        _, mean, cholesky, diagonal = gaussian_theta_conditional(
            logd, scales, **conditional_kwargs
        )
        beta = (
            mean
            + solve_triangular(
                cholesky.T,
                jax.random.normal(key, mean.shape, dtype=mean.dtype),
                lower=False,
            )
            / diagonal
        )
        residual_re = prepared_kwargs["observed_re"] - jnp.einsum(
            "frk,k->fr", prepared_kwargs["design_re"], beta
        )
        residual_im = prepared_kwargs["observed_im"] - jnp.einsum(
            "frk,k->fr", prepared_kwargs["design_im"], beta
        )
        logd = jnp.clip(logd, -LOG_SPECTRUM_LIMIT, LOG_SPECTRUM_LIMIT)
        full_log_likelihood = (
            (
                -prepared_kwargs["Nb"] * prepared_kwargs["Nh"] * jnp.sum(logd)
                - jnp.sum(
                    (residual_re**2 + residual_im**2) * jnp.exp(-logd)[:, None]
                )
                / prepared_kwargs["duration"]
            )
            * prepared_kwargs["eta"]
            / prepared_kwargs["enbw"]
        )
        return beta, full_log_likelihood

    beta, log_likelihood = jax.lax.map(draw, (log_delta, sigmas, keys))
    output = {}
    offset = 0
    for name, size in zip(names, prepared_kwargs["theta_sizes"], strict=True):
        output[f"weights_{name}"] = beta[:, offset : offset + size].reshape(
            *leading_shape, size
        )
        offset += size
    output[f"log_likelihood_block_{j}"] = log_likelihood.reshape(leading_shape)
    return output
