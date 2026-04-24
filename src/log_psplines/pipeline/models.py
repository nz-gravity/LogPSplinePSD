"""Pipeline-owned NumPyro model definitions.

These models are used by the pipeline runtime and intentionally avoid
dependencies on sampler implementations.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ..psplines import build_spline


def _sample_pspline_block(
    delta_name: str,
    phi_name: str,
    weights_name: str,
    penalty_matrix: jnp.ndarray,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
    factor_name: str | None = None,
    w_design: Optional[jnp.ndarray] = None,
    tau: Optional[float] = None,
) -> dict[str, Any]:
    """Draw hierarchical Gamma-Normal P-spline weights and record log priors."""
    log_delta_base = dist.Normal(0.0, 1.0)
    log_delta = numpyro.sample(delta_name, log_delta_base)
    delta = jnp.exp(log_delta)
    delta_dist = dist.Gamma(concentration=alpha_delta, rate=beta_delta)
    log_prior_delta = delta_dist.log_prob(delta) + log_delta
    numpyro.factor(
        f"{delta_name}_prior",
        log_prior_delta - log_delta_base.log_prob(log_delta),
    )

    log_phi_base = dist.Normal(0.0, 1.0)
    log_phi = numpyro.sample(phi_name, log_phi_base)
    phi = jnp.exp(log_phi)
    phi_rate = jnp.asarray(beta_phi, dtype=delta.dtype) * delta
    phi_dist = dist.Gamma(
        concentration=jnp.asarray(alpha_phi, dtype=delta.dtype),
        rate=phi_rate,
    )
    log_prior_phi = phi_dist.log_prob(phi) + log_phi
    numpyro.factor(
        f"{phi_name}_prior",
        log_prior_phi - log_phi_base.log_prob(log_phi),
    )

    k = penalty_matrix.shape[0]
    base_normal = dist.Normal(0.0, 1.0).expand((k,)).to_event(1)
    weights = numpyro.sample(weights_name, base_normal)

    residual = weights if w_design is None else weights - w_design
    wPw = jnp.dot(residual, jnp.dot(penalty_matrix, residual))
    log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
    if tau is not None and w_design is not None:
        log_prior_w += -0.5 * jnp.sum(residual**2) / tau**2
    base_log_prob = base_normal.log_prob(weights)

    if factor_name is None:
        factor_name = f"weights_prior_{weights_name}"
    numpyro.factor(factor_name, log_prior_w - base_log_prob)

    return {
        "weights": weights,
        "delta": delta,
        "phi": phi,
    }


def _univar_log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
    Nh: int,
) -> jnp.ndarray:
    """Univariate Whittle log-likelihood for spline coefficients."""
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    nh = jnp.asarray(Nh, dtype=ln_model.dtype)
    sum_log_det = nh * jnp.sum(ln_model)
    quad = jnp.sum(jnp.exp(log_pdgrm - ln_model))
    return -0.5 * (sum_log_det + quad)


def bayesian_model(
    log_pdgrm: jnp.ndarray,
    lnspline_basis: jnp.ndarray,
    penalty_matrix: jnp.ndarray,
    ln_parametric: jnp.ndarray,
    Nh: int,
    alpha_phi,
    beta_phi,
    alpha_delta,
    beta_delta,
    eta: float = 1.0,
) -> None:
    """NumPyro model for univariate PSD estimation."""
    block = _sample_pspline_block(
        delta_name="delta",
        phi_name="phi",
        weights_name="weights",
        penalty_matrix=penalty_matrix,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        factor_name="ln_prior",
    )

    lnl = _univar_log_likelihood(
        block["weights"],
        log_pdgrm,
        lnspline_basis,
        ln_parametric,
        Nh,
    )
    numpyro.factor("ln_likelihood", eta * lnl)


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
    alpha_phi: float,
    beta_phi: float,
    alpha_phi_theta: float,
    beta_phi_theta: float,
    alpha_delta: float,
    beta_delta: float,
    duration: float,
    Nb: int,
    Nh: int,
    design_weights: dict | None = None,
    tau: Optional[float] = None,
    enbw: float = 1.0,
    eta: float = 1.0,
) -> None:
    """NumPyro model for a single blocked multivariate Cholesky channel."""
    channel_label = f"{channel_index}"
    _dw = design_weights or {}

    delta_block = _sample_pspline_block(
        delta_name=f"delta_{channel_label}",
        phi_name=f"phi_delta_{channel_label}",
        weights_name=f"weights_delta_{channel_label}",
        penalty_matrix=penalty_delta,
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        w_design=_dw.get(f"delta_{channel_index}"),
        tau=tau,
    )
    log_delta_sq = jnp.einsum("nk,k->n", basis_delta, delta_block["weights"])
    log_delta_sq_safe = jnp.clip(log_delta_sq, a_min=-80.0, a_max=80.0)

    n_freq = u_re_channel.shape[0]
    n_theta_block = channel_index

    if n_theta_block > 0:
        theta_re_components = []
        theta_im_components = []

        for theta_idx in range(n_theta_block):
            basis_theta_re = basis_theta_re_by_component[theta_idx]
            penalty_theta_re = penalty_theta_re_by_component[theta_idx]
            theta_prefix = f"theta_re_{channel_label}_{theta_idx}"
            theta_re_block = _sample_pspline_block(
                delta_name=f"delta_{theta_prefix}",
                phi_name=f"phi_{theta_prefix}",
                weights_name=f"weights_{theta_prefix}",
                penalty_matrix=penalty_theta_re,
                alpha_phi=alpha_phi_theta,
                beta_phi=beta_phi_theta,
                alpha_delta=alpha_delta,
                beta_delta=beta_delta,
                w_design=_dw.get(f"theta_re_{channel_index}_{theta_idx}"),
                tau=tau,
            )
            theta_re_components.append(
                jnp.einsum(
                    "nk,k->n", basis_theta_re, theta_re_block["weights"]
                )
            )

            basis_theta_im = basis_theta_im_by_component[theta_idx]
            penalty_theta_im = penalty_theta_im_by_component[theta_idx]
            theta_im_prefix = f"theta_im_{channel_label}_{theta_idx}"
            theta_im_block = _sample_pspline_block(
                delta_name=f"delta_{theta_im_prefix}",
                phi_name=f"phi_{theta_im_prefix}",
                weights_name=f"weights_{theta_im_prefix}",
                penalty_matrix=penalty_theta_im,
                alpha_phi=alpha_phi_theta,
                beta_phi=beta_phi_theta,
                alpha_delta=alpha_delta,
                beta_delta=beta_delta,
                w_design=_dw.get(f"theta_im_{channel_index}_{theta_idx}"),
                tau=tau,
            )
            theta_im_components.append(
                jnp.einsum(
                    "nk,k->n", basis_theta_im, theta_im_block["weights"]
                )
            )

        theta_re = jnp.stack(theta_re_components, axis=1)
        theta_im = jnp.stack(theta_im_components, axis=1)
    else:
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))

    delta_eff_sq = jnp.exp(log_delta_sq_safe)
    nh = jnp.asarray(Nh, dtype=log_delta_sq.dtype)
    sum_log_det = -float(Nb) * nh * jnp.sum(jnp.log(delta_eff_sq))

    if n_theta_block > 0:
        contrib_re = jnp.einsum(
            "fl,flr->fr", theta_re, u_re_prev
        ) - jnp.einsum("fl,flr->fr", theta_im, u_im_prev)
        contrib_im = jnp.einsum(
            "fl,flr->fr", theta_re, u_im_prev
        ) + jnp.einsum("fl,flr->fr", theta_im, u_re_prev)
        u_re_resid = u_re_channel - contrib_re
        u_im_resid = u_im_channel - contrib_im
    else:
        u_re_resid = u_re_channel
        u_im_resid = u_im_channel

    residual_power = u_re_resid**2 + u_im_resid**2
    residual_power_sum = jnp.sum(residual_power, axis=1)
    duration_scale = jnp.asarray(duration, dtype=log_delta_sq.dtype)
    log_likelihood = sum_log_det - jnp.sum(
        residual_power_sum / (duration_scale * delta_eff_sq)
    )
    log_likelihood = log_likelihood / jnp.asarray(
        enbw, dtype=log_delta_sq.dtype
    )
    log_likelihood = log_likelihood * jnp.asarray(
        eta, dtype=log_delta_sq.dtype
    )

    numpyro.factor(f"likelihood_channel_{channel_label}", log_likelihood)
    numpyro.deterministic(
        f"log_likelihood_block_{channel_label}", log_likelihood
    )
