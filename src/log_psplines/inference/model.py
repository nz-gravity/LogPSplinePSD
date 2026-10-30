"""Shared stationary model selection, data operators, and initial values."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from log_psplines.config import StationaryConfig
from log_psplines.data.spectral import WishartData
from log_psplines.inference.collapsed import (
    collapsed_channel_model,
    prepare_collapsed_channel,
)
from log_psplines.inference.components import SpectralComponents
from log_psplines.inference.explicit import explicit_channel_model
from log_psplines.inference.initialisation import prepare_components


def channel_model_kwargs(
    model_kwargs: dict[str, Any],
    channel_index: int,
) -> dict[str, Any]:
    """Extract kwargs for one multivariate Cholesky likelihood factor."""
    j = int(channel_index)
    return {
        "channel_index": j,
        "u_re_channel": model_kwargs["u_re"][:, j, :],
        "u_im_channel": model_kwargs["u_im"][:, j, :],
        "u_re_prev": model_kwargs["u_re"][:, :j, :],
        "u_im_prev": model_kwargs["u_im_prev"][:, :j, :]
        if "u_im_prev" in model_kwargs
        else model_kwargs["u_im"][:, :j, :],
        "basis_delta": model_kwargs["bases_delta"][j],
        "penalty_delta": model_kwargs["penalties_delta"][j],
        "basis_theta_re_by_component": tuple(
            model_kwargs["bases_theta_re"][j]
        ),
        "penalty_theta_re_by_component": tuple(
            model_kwargs["penalties_theta_re"][j]
        ),
        "basis_theta_im_by_component": tuple(
            model_kwargs["bases_theta_im"][j]
        ),
        "penalty_theta_im_by_component": tuple(
            model_kwargs["penalties_theta_im"][j]
        ),
        "roughness_scale": model_kwargs["roughness_scale"],
        "theta_backend": model_kwargs.get("theta_backend", "auto"),
        "smoothing_parameterization": model_kwargs.get(
            "smoothing_parameterization", "centered"
        ),
        "duration": model_kwargs["duration"],
        "Nb": model_kwargs["Nb"],
        "Nh": model_kwargs["Nh"],
        "enbw": model_kwargs.get("enbw", 1.0),
        "eta": model_kwargs.get("eta", 1.0),
    }


def empirical_init_values(kwargs: dict[str, Any], channel: int) -> dict:
    """Initialize sampled sites from empirical conditional-spectrum pilots.

    The pilot selects starting values only; it does not change any prior.
    Marginalised rows need theta scales only. Explicit rows also initialize
    their theta weights, with the same whitening used by the spline prior.
    """
    blocks = [
        (
            f"delta_{channel}",
            kwargs["penalties_delta"][channel],
            kwargs["initial_weights_delta"][channel],
        )
    ]
    for previous in range(channel):
        for part in ("re", "im"):
            blocks.append(
                (
                    f"theta_{part}_{channel}_{previous}",
                    kwargs[f"penalties_theta_{part}"][channel][previous],
                    kwargs[f"initial_weights_theta_{part}"][channel][previous],
                )
            )
    values = {}
    for label, penalty, weights in blocks:
        penalty, weights = np.asarray(penalty), np.asarray(weights)
        sigma = float(
            np.clip(
                np.sqrt(
                    max(float(weights @ penalty @ weights), 0.0) / weights.size
                ),
                min(1e-3, float(kwargs["roughness_scale"])),
                float(kwargs["roughness_scale"]),
            )
        )
        values[f"sigma_{label}"] = sigma
        if label.startswith("delta") or not kwargs.get(
            "marginalize_theta", True
        ):
            if kwargs.get("smoothing_parameterization") == "noncentered":
                values[f"weights_{label}_raw"] = jnp.asarray(
                    np.linalg.cholesky(penalty).T @ weights / sigma
                )
            else:
                values[f"weights_{label}"] = jnp.asarray(weights)
    return values


def prepare_channel_model(
    model_kwargs: dict[str, Any],
    channel_index: int,
) -> tuple[Callable[..., None], dict[str, Any], dict[str, Any]]:
    """Shared stationary row model and initialization for NUTS and VI.

    The explicit model also handles the first row, which has no theta blocks.
    Only marginalised rows prepare Gram caches and conditional-draw operators.
    """
    kwargs = channel_model_kwargs(model_kwargs, channel_index)
    initial = empirical_init_values(model_kwargs, channel_index)
    if channel_index > 0 and model_kwargs.get("marginalize_theta", True):
        return (
            collapsed_channel_model,
            prepare_collapsed_channel(kwargs),
            initial,
        )
    return explicit_channel_model, kwargs, initial


def prepare_model(
    data: WishartData,
    config: StationaryConfig,
) -> tuple[dict, SpectralComponents]:
    if not np.isfinite(config.roughness_scale) or config.roughness_scale <= 0:
        raise ValueError("roughness_scale must be finite and positive")
    if config.smoothing_parameterization not in ("centered", "noncentered"):
        raise ValueError("Unknown smoothing_parameterization")
    if config.theta_backend not in ("auto", "dense", "banded"):
        raise ValueError("theta_backend must be auto, dense or banded")
    if data.p > 1 and not jax.config.x64_enabled:
        raise ValueError(
            "Multivariate stationary fitting requires JAX_ENABLE_X64=true"
        )
    spline = prepare_components(
        data,
        n_knots=config.n_knots,
        degree=config.degree,
        diffMatrixOrder=config.diffMatrixOrder,
        knot_kwargs=config.knot_kwargs or {},
        analytical_psd=config.analytical_psd,
    )

    p = data.p
    # Respect configured JAX precision; scientific CI explicitly enables x64.
    u_re = jnp.asarray(data.u_re)
    u_im = jnp.asarray(data.u_im)

    bases_delta = []
    penalties_delta = []
    for j in range(p):
        m = spline.diagonal_models[j]
        bases_delta.append(jnp.asarray(m.basis))
        penalties_delta.append(jnp.asarray(m.penalty_matrix))

    bases_theta_re: list[list] = []
    penalties_theta_re: list[list] = []
    bases_theta_im: list[list] = []
    penalties_theta_im: list[list] = []
    for j in range(p):
        br, pr, bi, pi = [], [], [], []
        for previous_channel in range(j):
            m_re = spline.get_theta_model("re", j, previous_channel)
            m_im = spline.get_theta_model("im", j, previous_channel)
            br.append(jnp.asarray(m_re.basis))
            pr.append(jnp.asarray(m_re.penalty_matrix))
            bi.append(jnp.asarray(m_im.basis))
            pi.append(jnp.asarray(m_im.penalty_matrix))
        bases_theta_re.append(br)
        penalties_theta_re.append(pr)
        bases_theta_im.append(bi)
        penalties_theta_im.append(pi)

    kwargs = {
        "u_re": u_re,
        "u_im": u_im,
        "n_channels": p,
        "bases_delta": bases_delta,
        "penalties_delta": penalties_delta,
        "bases_theta_re": bases_theta_re,
        "penalties_theta_re": penalties_theta_re,
        "bases_theta_im": bases_theta_im,
        "penalties_theta_im": penalties_theta_im,
        "roughness_scale": float(config.roughness_scale),
        "theta_backend": config.theta_backend,
        "marginalize_theta": config.marginalize_theta,
        "smoothing_parameterization": config.smoothing_parameterization,
        "initial_weights_delta": tuple(
            m.weights for m in spline.diagonal_models
        ),
        "initial_weights_theta_re": tuple(
            tuple(spline.get_theta_model("re", j, i).weights for i in range(j))
            for j in range(p)
        ),
        "initial_weights_theta_im": tuple(
            tuple(spline.get_theta_model("im", j, i).weights for i in range(j))
            for j in range(p)
        ),
        "duration": float(getattr(data, "duration", 1.0) or 1.0),
        "Nb": int(data.Nb),
        "Nh": int(data.Nh),
        "enbw": float(getattr(data, "enbw", 1.0)),
    }
    return kwargs, spline
