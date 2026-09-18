"""Pipeline-owned NumPyro model definitions.

These models are used by the pipeline runtime and intentionally avoid
dependencies on sampler implementations.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from log_psplines.config import PipelineConfig
from log_psplines.data.spectral import WishartData
from log_psplines.inference.components import SpectralComponents
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.models.spectrum import build_spline


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
    w_design: jnp.ndarray | None = None,
    tau: float | None = None,
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
    tau: float | None = None,
    enbw: float = 1.0,
    eta: float = 1.0,
) -> None:
    """NumPyro model for a single blocked multivariate Cholesky channel."""
    channel_label = f"{channel_index}"
    _dw = design_weights or {}

    # --- Spline model for log(δ²_{jh}) ---
    # Sample P-spline weights and evaluate: log_delta_sq[h] = log(δ²_{jh})
    # δ²_{jh} is the j-th diagonal of D_h (the noise variance for this channel
    # at coarse bin h in the Cholesky factorisation S^{-1} = T* D^{-1} T).
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
    # log_delta_sq[h] = B_h @ w  →  log(δ²_{jh}), shape (n_coarse_bins,)
    log_delta_sq = build_spline(basis_delta, delta_block["weights"])

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
            # Re(θ_{jl}^(h)) evaluated at each coarse bin, shape (n_coarse_bins,)
            theta_re_components.append(
                build_spline(basis_theta_re, theta_re_block["weights"])
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
            # Im(θ_{jl}^(h)) evaluated at each coarse bin, shape (n_coarse_bins,)
            theta_im_components.append(
                build_spline(basis_theta_im, theta_im_block["weights"])
            )

        # theta_re/im: shape (n_coarse_bins, n_theta_block=j)
        # theta_re[h, l] = Re(θ_{jl}^(h)),  theta_im[h, l] = Im(θ_{jl}^(h))
        theta_re = jnp.stack(theta_re_components, axis=1)
        theta_im = jnp.stack(theta_im_components, axis=1)
    else:
        # Channel 0 has no preceding channels; residual equals the observation.
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))

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


def _joint_multivar_model(
    u_re: jnp.ndarray,
    u_im: jnp.ndarray,
    n_channels: int,
    bases_delta: list,
    penalties_delta: list,
    bases_theta_re: list,
    penalties_theta_re: list,
    bases_theta_im: list,
    penalties_theta_im: list,
    alpha_phi: float,
    beta_phi: float,
    alpha_phi_theta: float,
    beta_phi_theta: float,
    alpha_delta: float,
    beta_delta: float,
    duration: float,
    Nb: int,
    Nh: int,
    enbw: float,
    eta: float = 1.0,
    design_weights=None,
    tau=None,
) -> None:
    """Joint NumPyro model that calls _blocked_channel_model for every channel.

    All channels are sampled in a single NumPyro model context, making it
    compatible with the generic VIStage / NUTSStage interface.  Production
    code uses factorized NUTS which runs independent
    per-channel chains.
    """
    for j in range(n_channels):
        _blocked_channel_model(
            channel_index=j,
            u_re_channel=u_re[:, j, :],
            u_im_channel=u_im[:, j, :],
            u_re_prev=u_re[:, :j, :],
            u_im_prev=u_im[:, :j, :],
            basis_delta=bases_delta[j],
            penalty_delta=penalties_delta[j],
            basis_theta_re_by_component=tuple(bases_theta_re[j]),
            penalty_theta_re_by_component=tuple(penalties_theta_re[j]),
            basis_theta_im_by_component=tuple(bases_theta_im[j]),
            penalty_theta_im_by_component=tuple(penalties_theta_im[j]),
            alpha_phi=alpha_phi,
            beta_phi=beta_phi,
            alpha_phi_theta=alpha_phi_theta,
            beta_phi_theta=beta_phi_theta,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
            duration=duration,
            Nb=Nb,
            Nh=Nh,
            design_weights=design_weights,
            tau=tau,
            enbw=enbw,
            eta=eta,
        )


def prepare_model(
    data: WishartData,
    config: PipelineConfig,
) -> tuple[dict, SpectralComponents]:
    spline = SpectralComponents.from_multivar_fft(
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
        for l in range(j):
            m_re = spline.get_theta_model("re", j, l)
            m_im = spline.get_theta_model("im", j, l)
            br.append(jnp.asarray(m_re.basis, dtype=jnp.float32))
            pr.append(jnp.asarray(m_re.penalty_matrix))
            bi.append(jnp.asarray(m_im.basis, dtype=jnp.float32))
            pi.append(jnp.asarray(m_im.penalty_matrix))
        bases_theta_re.append(br)
        penalties_theta_re.append(pr)
        bases_theta_im.append(bi)
        penalties_theta_im.append(pi)

    alpha_phi_theta = (
        config.alpha_phi_theta
        if config.alpha_phi_theta is not None
        else config.alpha_phi
    )
    beta_phi_theta = (
        config.beta_phi_theta
        if config.beta_phi_theta is not None
        else config.beta_phi
    )

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
        "alpha_phi": float(config.alpha_phi),
        "beta_phi": float(config.beta_phi),
        "alpha_phi_theta": float(alpha_phi_theta),
        "beta_phi_theta": float(beta_phi_theta),
        "alpha_delta": float(config.alpha_delta),
        "beta_delta": float(config.beta_delta),
        "duration": float(getattr(data, "duration", 1.0) or 1.0),
        "Nb": int(data.Nb),
        "Nh": int(data.Nh),
        "enbw": float(getattr(data, "enbw", 1.0)),
        "design_weights": None,
        "tau": None,
    }
    return kwargs, spline
