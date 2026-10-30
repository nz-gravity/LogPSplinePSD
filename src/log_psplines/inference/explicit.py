"""Stationary Cholesky rows with all spline coefficients sampled jointly."""

from typing import Any

import jax
import jax.numpy as jnp
import numpyro

from log_psplines.inference.priors import sample_pspline_block
from log_psplines.likelihoods.wishart import wishart_log_likelihood
from log_psplines.models.spectrum import build_spline


def explicit_channel_model(
    channel_index: int,
    u_re_channel: jax.Array,
    u_im_channel: jax.Array,
    u_re_prev: jax.Array,
    u_im_prev: jax.Array,
    basis_delta: jax.Array,
    penalty_delta: jax.Array,
    basis_theta_re_by_component: tuple,
    penalty_theta_re_by_component: tuple,
    basis_theta_im_by_component: tuple,
    penalty_theta_im_by_component: tuple,
    duration: float,
    Nb: int,
    Nh: int,
    roughness_scale: float,
    smoothing_parameterization: str = "centered",
    enbw: float = 1.0,
    eta: float = 1.0,
    **unused_backend_kwargs: Any,
) -> None:
    """Sample one row, including the scalar first row with no theta blocks.

    Every component uses the common spline prior and Wishart likelihood.
    The algebra backend setting applies only to marginalised rows.
    """

    def sample_curve(label, basis, penalty):
        weights = sample_pspline_block(
            f"sigma_{label}",
            f"weights_{label}",
            penalty,
            roughness_scale,
            smoothing_parameterization,
        )
        return build_spline(basis, weights)

    log_delta_sq = sample_curve(
        f"delta_{channel_index}", basis_delta, penalty_delta
    )
    theta = []
    for part, bases, penalties in (
        ("re", basis_theta_re_by_component, penalty_theta_re_by_component),
        ("im", basis_theta_im_by_component, penalty_theta_im_by_component),
    ):
        curves = [
            sample_curve(f"theta_{part}_{channel_index}_{i}", basis, penalty)
            for i, (basis, penalty) in enumerate(
                zip(bases, penalties, strict=True)
            )
        ]
        theta.append(
            jnp.stack(curves, axis=-1)
            if curves
            else jnp.zeros((u_re_channel.shape[0], 0))
        )
    likelihood = wishart_log_likelihood(
        log_delta_sq,
        *theta,
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
    numpyro.factor(f"likelihood_channel_{channel_index}", likelihood)
    numpyro.deterministic(f"log_likelihood_block_{channel_index}", likelihood)
