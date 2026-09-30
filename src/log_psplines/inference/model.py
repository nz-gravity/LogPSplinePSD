"""Modified-Cholesky channel models and model argument preparation."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular

from log_psplines.config import StationaryConfig
from log_psplines.data.spectral import WishartData
from log_psplines.inference.components import SpectralComponents
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
        "smoothing_parameterization": model_kwargs.get(
            "smoothing_parameterization", "centered"
        ),
        "duration": model_kwargs["duration"],
        "Nb": model_kwargs["Nb"],
        "Nh": model_kwargs["Nh"],
        "enbw": model_kwargs.get("enbw", 1.0),
        "eta": model_kwargs.get("eta", 1.0),
    }


def _sample_pspline_block(
    sigma_name: str,
    weights_name: str,
    penalty_matrix: jnp.ndarray,
    roughness_scale: float,
    smoothing_parameterization: str = "centered",
) -> jnp.ndarray:
    """Draw spline weights; return shape ``(K,)`` under a HalfNormal scale."""
    # Keep explicitly supplied low-precision operators low precision even
    # when the process permits float64 arrays.
    dtype = penalty_matrix.dtype
    sigma = numpyro.sample(
        sigma_name, dist.HalfNormal(jnp.asarray(roughness_scale, dtype=dtype))
    )

    k = penalty_matrix.shape[0]
    base_normal = (
        dist.Normal(
            jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype)
        )
        .expand((k,))
        .to_event(1)
    )
    if smoothing_parameterization == "noncentered":
        raw_weights = numpyro.sample(f"{weights_name}_raw", base_normal)
        cholesky = jnp.linalg.cholesky(penalty_matrix)
        weights = sigma * solve_triangular(
            cholesky.T, raw_weights, lower=False
        )
        numpyro.deterministic(weights_name, weights)
    else:
        weights = numpyro.sample(weights_name, base_normal)

    if smoothing_parameterization == "centered":
        wPw = jnp.dot(weights, jnp.dot(penalty_matrix, weights))
        precision = sigma**-2
        log_prior_w = 0.5 * k * jnp.log(precision) - 0.5 * precision * wPw
        base_log_prob = base_normal.log_prob(weights)
        numpyro.factor(
            f"weights_prior_{weights_name}", log_prior_w - base_log_prob
        )

    return weights


__all__ = [
    "_blocked_channel_model",
]


def _blocked_channel_model(
    channel_index: int,
    u_re_channel: jnp.ndarray,
    u_im_channel: jnp.ndarray,
    u_re_prev: jnp.ndarray,
    u_im_prev: jnp.ndarray,
    basis_delta: jnp.ndarray,
    penalty_delta: jnp.ndarray,
    basis_theta_re_by_component: tuple[jnp.ndarray, ...],
    penalty_theta_re_by_component: tuple[jnp.ndarray, ...],
    basis_theta_im_by_component: tuple[jnp.ndarray, ...],
    penalty_theta_im_by_component: tuple[jnp.ndarray, ...],
    duration: float,
    Nb: int,
    Nh: int,
    roughness_scale: float,
    smoothing_parameterization: str = "centered",
    enbw: float = 1.0,
    eta: float = 1.0,
) -> None:
    """NumPyro model for a single blocked multivariate Cholesky channel."""
    channel_label = f"{channel_index}"

    # --- Spline model for log(δ²_{jh}) ---
    # Sample P-spline weights and evaluate: log_delta_sq[h] = log(δ²_{jh})
    # δ²_{jh} is the j-th diagonal of D_h (the noise variance for this channel
    # at coarse bin h in the Cholesky factorisation S^{-1} = T* D^{-1} T).
    delta_weights = _sample_pspline_block(
        sigma_name=f"sigma_delta_{channel_label}",
        weights_name=f"weights_delta_{channel_label}",
        penalty_matrix=penalty_delta,
        roughness_scale=roughness_scale,
        smoothing_parameterization=smoothing_parameterization,
    )
    # log_delta_sq[h] = B_h @ w  →  log(δ²_{jh}), shape (n_coarse_bins,)
    log_delta_sq = build_spline(basis_delta, delta_weights)

    n_freq = u_re_channel.shape[0]
    # channel_index == j means there are j preceding channels (l = 0, …, j-1)
    # whose Cholesky coefficients θ_{jl} couple into this channel's residual.
    n_theta_block = channel_index

    # --- Spline models for the complex Cholesky off-diagonal coefficients θ_{jl} ---
    # For each preceding channel l < j, sample Re(θ_{jl}^(h)) and Im(θ_{jl}^(h))
    # independently as P-splines over the coarse-bin axis h.
    if n_theta_block > 0:
        theta_re_components = []
        theta_im_components = []

        for theta_idx in range(n_theta_block):
            basis_theta_re = basis_theta_re_by_component[theta_idx]
            penalty_theta_re = penalty_theta_re_by_component[theta_idx]
            theta_prefix = f"theta_re_{channel_label}_{theta_idx}"
            theta_re_block = _sample_pspline_block(
                sigma_name=f"sigma_{theta_prefix}",
                weights_name=f"weights_{theta_prefix}",
                penalty_matrix=penalty_theta_re,
                roughness_scale=roughness_scale,
                smoothing_parameterization=smoothing_parameterization,
            )
            # Re(θ_{jl}^(h)) evaluated at each coarse bin, shape (n_coarse_bins,)
            theta_re_components.append(
                build_spline(basis_theta_re, theta_re_block)
            )

            basis_theta_im = basis_theta_im_by_component[theta_idx]
            penalty_theta_im = penalty_theta_im_by_component[theta_idx]
            theta_im_prefix = f"theta_im_{channel_label}_{theta_idx}"
            theta_im_block = _sample_pspline_block(
                sigma_name=f"sigma_{theta_im_prefix}",
                weights_name=f"weights_{theta_im_prefix}",
                penalty_matrix=penalty_theta_im,
                roughness_scale=roughness_scale,
                smoothing_parameterization=smoothing_parameterization,
            )
            # Im(θ_{jl}^(h)) evaluated at each coarse bin, shape (n_coarse_bins,)
            theta_im_components.append(
                build_spline(basis_theta_im, theta_im_block)
            )

        # theta_re/im: shape (n_coarse_bins, n_theta_block=j)
        # theta_re[h, l] = Re(θ_{jl}^(h)),  theta_im[h, l] = Im(θ_{jl}^(h))
        theta_re = jnp.stack(theta_re_components, axis=1)
        theta_im = jnp.stack(theta_im_components, axis=1)
    else:
        # Channel 0 has no preceding channels; residual equals the observation.
        theta_re = jnp.zeros((n_freq, 0), dtype=log_delta_sq.dtype)
        theta_im = jnp.zeros((n_freq, 0), dtype=log_delta_sq.dtype)

    log_likelihood = wishart_log_likelihood(
        log_delta_sq,
        theta_re,
        theta_im,
        u_re_channel,
        u_im_channel,
        u_re_prev,
        u_im_prev,
        Nb=Nb,
        Nh=Nh,
        duration=duration,
        enbw=enbw,
        eta=eta,
    )
    numpyro.factor(f"likelihood_channel_{channel_label}", log_likelihood)
    numpyro.deterministic(
        f"log_likelihood_block_{channel_label}", log_likelihood
    )


def prepare_model(
    data: WishartData,
    config: StationaryConfig,
) -> tuple[dict, SpectralComponents]:
    if not np.isfinite(config.roughness_scale) or config.roughness_scale <= 0:
        raise ValueError("roughness_scale must be finite and positive")
    if config.smoothing_parameterization not in ("centered", "noncentered"):
        raise ValueError("Unknown smoothing_parameterization")
    spline = prepare_components(
        data,
        n_knots=config.n_knots,
        degree=config.degree,
        diffMatrixOrder=config.diffMatrixOrder,
        knot_kwargs=config.knot_kwargs or {},
        analytical_psd=config.analytical_psd,
    )

    p = data.p
    u_re = jnp.asarray(data.u_re, dtype=jnp.float32)
    u_im = jnp.asarray(data.u_im, dtype=jnp.float32)

    bases_delta = []
    penalties_delta = []
    for j in range(p):
        m = spline.diagonal_models[j]
        bases_delta.append(jnp.asarray(m.basis, dtype=jnp.float32))
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
            br.append(jnp.asarray(m_re.basis, dtype=jnp.float32))
            pr.append(jnp.asarray(m_re.penalty_matrix))
            bi.append(jnp.asarray(m_im.basis, dtype=jnp.float32))
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
        "smoothing_parameterization": config.smoothing_parameterization,
        "duration": float(getattr(data, "duration", 1.0) or 1.0),
        "Nb": int(data.Nb),
        "Nh": int(data.Nh),
        "enbw": float(getattr(data, "enbw", 1.0)),
    }
    return kwargs, spline
