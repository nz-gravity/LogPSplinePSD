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
        for j, l in spline_model.theta_pairs:
            re_model = spline_model.get_theta_model("re", j, l)
            im_model = spline_model.get_theta_model("im", j, l)
            init_values[f"delta_theta_re_{j}_{l}"] = delta_0
            init_values[f"phi_theta_re_{j}_{l}"] = log_phi_0
            init_values[f"weights_theta_re_{j}_{l}"] = jnp.asarray(
                re_model.weights
            )
            init_values[f"delta_theta_im_{j}_{l}"] = delta_0
            init_values[f"phi_theta_im_{j}_{l}"] = log_phi_0
            init_values[f"weights_theta_im_{j}_{l}"] = jnp.asarray(
                im_model.weights
            )

        # Backward-compatible shared aliases (first theta component), used by
        # legacy fully-coupled multivariate paths.
        first_j, first_l = spline_model.theta_pair_from_index(0)
        init_values["delta_theta_re"] = delta_0
        init_values["phi_theta_re"] = log_phi_0
        init_values["weights_theta_re"] = jnp.asarray(
            spline_model.get_theta_model("re", first_j, first_l).weights
        )
        init_values["delta_theta_im"] = delta_0
        init_values["phi_theta_im"] = log_phi_0
        init_values["weights_theta_im"] = jnp.asarray(
            spline_model.get_theta_model("im", first_j, first_l).weights
        )

    return init_values
