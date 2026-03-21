"""Default initial value helpers for sampler initialisation."""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp

from ..pspline_block import pspline_hyperparameter_initials


def _hyperparameter_defaults(
    *,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
    divide_phi_by_delta: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(delta_0, log_phi_0)`` from the prior hypers."""
    delta_0, phi_0 = pspline_hyperparameter_initials(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        divide_phi_by_delta=divide_phi_by_delta,
    )
    return jnp.asarray(delta_0), jnp.log(jnp.asarray(phi_0))


def default_init_values_univar(
    spline_model,
    *,
    alpha_phi: float,
    beta_phi: float,
    alpha_delta: float,
    beta_delta: float,
) -> Dict[str, jnp.ndarray]:
    """Return default init values for univariate samplers."""
    delta_0, log_phi_0 = _hyperparameter_defaults(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        divide_phi_by_delta=True,
    )
    return {
        "delta": delta_0,
        "phi": log_phi_0,
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
    delta_0, log_phi_0 = _hyperparameter_defaults(
        alpha_phi=alpha_phi,
        beta_phi=beta_phi,
        alpha_delta=alpha_delta,
        beta_delta=beta_delta,
        divide_phi_by_delta=False,
    )

    init_values: Dict[str, jnp.ndarray] = {}
    for idx, model in enumerate(spline_model.diagonal_models):
        init_values[f"delta_{idx}"] = delta_0
        init_values[f"phi_delta_{idx}"] = log_phi_0
        init_values[f"weights_delta_{idx}"] = jnp.asarray(model.weights)

    if spline_model.n_theta > 0:
        for prefix, model in (
            ("theta_re", spline_model.offdiag_re_model),
            ("theta_im", spline_model.offdiag_im_model),
        ):
            init_values[f"delta_{prefix}"] = delta_0
            init_values[f"phi_{prefix}"] = log_phi_0
            init_values[f"weights_{prefix}"] = jnp.asarray(model.weights)

    return init_values
