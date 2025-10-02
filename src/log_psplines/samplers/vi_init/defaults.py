"""Default initial value helpers for sampler initialisation."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from ..utils import pspline_hyperparameter_initials


def default_init_values_univar(
    spline_model,
    *,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
) -> Dict[str, jnp.ndarray]:
    """Return default init values for univariate samplers."""

    delta_0, phi_0 = pspline_hyperparameter_initials(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        divide_phi_by_delta=True,
    )
    return {
        "delta": jnp.asarray(delta_0),
        "phi": jnp.log(jnp.asarray(phi_0)),
        "weights": jnp.asarray(spline_model.weights),
    }


def default_init_values_multivar(
    spline_model,
    *,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
) -> Dict[str, jnp.ndarray]:
    """Return default init values for multivariate samplers."""

    delta_init, phi_init = pspline_hyperparameter_initials(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
    )

    init_values: Dict[str, jnp.ndarray] = {}

    for idx, model in enumerate(spline_model.diagonal_models):
        init_values[f"delta_{idx}"] = jnp.asarray(delta_init)
        init_values[f"phi_delta_{idx}"] = jnp.log(jnp.asarray(phi_init))
        init_values[f"weights_delta_{idx}"] = jnp.asarray(model.weights)

    if spline_model.n_theta > 0:
        init_values["delta_theta_re"] = jnp.asarray(delta_init)
        init_values["phi_theta_re"] = jnp.log(jnp.asarray(phi_init))
        init_values["weights_theta_re"] = jnp.asarray(
            spline_model.offdiag_re_model.weights
        )

        init_values["delta_theta_im"] = jnp.asarray(delta_init)
        init_values["phi_theta_im"] = jnp.log(jnp.asarray(phi_init))
        init_values["weights_theta_im"] = jnp.asarray(
            spline_model.offdiag_im_model.weights
        )

    return init_values
