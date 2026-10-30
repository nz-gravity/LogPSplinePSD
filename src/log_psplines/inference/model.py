"""Modified-Cholesky channel models and model argument preparation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from log_psplines.config import StationaryConfig
from log_psplines.data.spectral import WishartData
from log_psplines.inference.components import SpectralComponents
from log_psplines.inference.coordinates import (
    coordinate_scales,
    spline_coordinates,
)
from log_psplines.inference.initialisation import prepare_components
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.models.spectrum import build_spline


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
        "delta_coordinates": spline_coordinates(
            model_kwargs["bases_delta"][j],
            model_kwargs["penalties_delta"][j],
            (
                model_kwargs.get("counts")
                if model_kwargs.get("counts") is not None
                else model_kwargs["Nb"] * model_kwargs["Nh"]
            )
            * model_kwargs.get("eta", 1.0)
            / model_kwargs.get("enbw", 1.0),
        ),
        "duration": model_kwargs["duration"],
        "Nb": model_kwargs["Nb"],
        "Nh": model_kwargs["Nh"],
        "counts": model_kwargs.get("counts"),
        "enbw": model_kwargs.get("enbw", 1.0),
        "eta": model_kwargs.get("eta", 1.0),
    }


def _sample_pspline_block(
    sigma_name: str,
    weights_name: str,
    roughness_scale: float,
    coordinates: dict,
) -> jnp.ndarray:
    """Return physical weights under the unchanged proper Gaussian prior."""
    sigma = numpyro.sample(sigma_name, dist.HalfNormal(roughness_scale))
    prior_scale = sigma / jnp.sqrt(coordinates["eigenvalues"])
    scale = coordinate_scales(prior_scale, coordinates["information"])
    raw = numpyro.sample(
        f"{weights_name}_raw",
        dist.Normal(0.0, prior_scale / scale).to_event(1),
    )
    weights = coordinates["rotation"] @ (raw * scale)
    return numpyro.deterministic(weights_name, weights)


def _scalar_channel_model(
    channel_index: int,
    u_re_channel: jnp.ndarray,
    u_im_channel: jnp.ndarray,
    u_re_prev: jnp.ndarray,
    u_im_prev: jnp.ndarray,
    basis_delta: jnp.ndarray,
    penalty_delta: jnp.ndarray,
    duration: float,
    Nb: int,
    Nh: int,
    roughness_scale: float,
    delta_coordinates: dict,
    enbw: float = 1.0,
    eta: float = 1.0,
    counts: jnp.ndarray | None = None,
    **unused_theta_kwargs: Any,
) -> None:
    """The first Cholesky row has no cross-spectrum weights to integrate."""
    if channel_index != 0:
        raise ValueError("The scalar channel model requires channel_index=0")
    weights = _sample_pspline_block(
        "sigma_delta_0",
        "weights_delta_0",
        roughness_scale,
        delta_coordinates,
    )
    log_delta_sq = build_spline(basis_delta, weights)
    empty_theta = jnp.zeros((u_re_channel.shape[0], 0))
    likelihood = wishart_log_likelihood(
        log_delta_sq,
        empty_theta,
        empty_theta,
        u_re_channel,
        u_im_channel,
        u_re_prev,
        u_im_prev,
        Nb=Nb,
        Nh=Nh,
        counts=counts,
        duration=duration,
        enbw=enbw,
        eta=eta,
    )
    numpyro.factor("likelihood_channel_0", likelihood)
    numpyro.deterministic("log_likelihood_block_0", likelihood)


def empirical_init_values(
    kwargs: dict[str, Any], channel: int, coordinates: dict
) -> dict:
    """Initialize the reduced state from empirical conditional-spectrum pilots.

    The pilot selects starting values only; it does not change any prior.
    Theta weights are integrated out, so only their smoothing scales enter
    the numerical inference state.
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
        if label.startswith("delta"):
            prior_scale = sigma / jnp.sqrt(coordinates["eigenvalues"])
            scale = coordinate_scales(prior_scale, coordinates["information"])
            values[f"weights_{label}_raw"] = (
                coordinates["rotation"].T @ weights / scale
            )
    return values


def prepare_channel_model(
    model_kwargs: dict[str, Any],
    channel_index: int,
) -> tuple[Callable[..., None], dict[str, Any], dict[str, Any]]:
    """Shared stationary row model and initialization for NUTS and VI.

    Every row with predecessors integrates all of its theta coefficients.
    The first row remains a scalar log-spectrum model.
    """
    kwargs = channel_model_kwargs(model_kwargs, channel_index)
    initial = empirical_init_values(
        model_kwargs, channel_index, kwargs["delta_coordinates"]
    )
    if channel_index == 0:
        return _scalar_channel_model, kwargs, initial
    # Local import keeps the spline-prior helper available to collapsed.py
    # without a module initialization cycle.
    from log_psplines.inference.collapsed import (
        collapsed_channel_model,
        prepare_collapsed_channel,
    )

    return collapsed_channel_model, prepare_collapsed_channel(kwargs), initial


def prepare_model(
    data: WishartData,
    config: StationaryConfig,
) -> tuple[dict, SpectralComponents]:
    if not np.isfinite(config.roughness_scale) or config.roughness_scale <= 0:
        raise ValueError("roughness_scale must be finite and positive")
    if data.p > 1 and not jax.config.x64_enabled:
        raise ValueError(
            "Multivariate Gaussian marginalization requires JAX_ENABLE_X64=true"
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
        "counts": jnp.asarray(data.counts),
        "enbw": float(getattr(data, "enbw", 1.0)),
    }
    return kwargs, spline
