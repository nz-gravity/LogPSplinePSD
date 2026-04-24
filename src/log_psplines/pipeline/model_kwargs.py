"""Build model_kwargs dicts for VIStage / NUTSStage.

This is the only place in the codebase that branches on the data type
(Periodogram vs MultivarFFT) for inference purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import jax.numpy as jnp
import numpy as np
import numpyro

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..psplines import LogPSplines, MultivariateLogPSplines
from ..samplers.multivar.multivar_blocked_nuts import _blocked_channel_model

if TYPE_CHECKING:
    from .config import PipelineConfig


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
    code should prefer MultivarBlockedNUTSSampler which runs independent
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


def _build_univar_kwargs(
    data: Periodogram,
    config: "PipelineConfig",
) -> dict:
    spline = LogPSplines.from_periodogram(
        data,
        n_knots=config.n_knots,
        degree=config.degree,
        diffMatrixOrder=config.diffMatrixOrder,
        parametric_model=config.parametric_model,
        knot_kwargs=config.knot_kwargs or {},
    )
    return {
        "log_pdgrm": jnp.log(jnp.asarray(data.power, dtype=jnp.float32)),
        "lnspline_basis": jnp.asarray(spline.basis, dtype=jnp.float32),
        "penalty_matrix": jnp.asarray(spline.penalty_matrix),
        "ln_parametric": jnp.asarray(spline.log_parametric_model),
        "Nh": int(data.Nh),
        "alpha_phi": float(config.alpha_phi),
        "beta_phi": float(config.beta_phi),
        "alpha_delta": float(config.alpha_delta),
        "beta_delta": float(config.beta_delta),
    }


def _build_multivar_kwargs(
    data: MultivarFFT,
    config: "PipelineConfig",
) -> dict:
    spline = MultivariateLogPSplines.from_multivar_fft(
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
        m = spline.component_specs[spline.delta_key(j)].model
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

    return {
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


def build_model_kwargs(
    data: Union[Periodogram, MultivarFFT],
    config: "PipelineConfig",
    coarse: bool = False,
) -> dict:
    """Return model kwargs for the appropriate inference model.

    Parameters
    ----------
    data:
        Pre-processed frequency-domain data.  ``Periodogram`` triggers the
        univariate path; ``MultivarFFT`` triggers the multivariate path.
    config:
        Pipeline configuration.
    coarse:
        Set to ``True`` when ``data`` is already-coarse-grained.  Currently
        this is informational only — the kwargs are built identically.
    """
    if isinstance(data, Periodogram):
        return _build_univar_kwargs(data, config)
    return _build_multivar_kwargs(data, config)
