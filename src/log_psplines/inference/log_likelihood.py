"""Pointwise stationary likelihood draws for ArviZ diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from log_psplines.data.spectral import WishartData


def _posterior_coords(posterior: xr.Dataset) -> dict[str, np.ndarray]:
    return {
        "chain": np.asarray(posterior.coords["chain"].values),
        "draw": np.asarray(posterior.coords["draw"].values),
    }


def _channel_theta_array(
    posterior: xr.Dataset,
    basis_list: list[Any],
    channel_index: int,
    param_type: str,
) -> np.ndarray:
    """Evaluate per-draw theta spline values for one multivariate channel."""
    theta_parts: list[np.ndarray] = []
    for theta_idx, theta_basis in enumerate(basis_list):
        weights_name = (
            f"weights_theta_{param_type}_{channel_index}_{theta_idx}"
        )
        weights = np.asarray(posterior[weights_name].values, dtype=np.float64)
        basis = np.asarray(theta_basis, dtype=np.float64)
        theta_eval = np.einsum("fk,cdk->cdf", basis, weights)
        theta_parts.append(theta_eval)
    return np.stack(theta_parts, axis=-1)


def _pointwise_multivar_log_likelihood(
    posterior: xr.Dataset,
    data: WishartData,
    model_kwargs: dict[str, Any],
) -> xr.Dataset:
    """Return blocked multivariate pointwise log-likelihood draws by frequency."""
    u_re = np.asarray(model_kwargs["u_re"], dtype=np.float64)
    u_im = np.asarray(model_kwargs["u_im"], dtype=np.float64)
    bases_delta = model_kwargs["bases_delta"]
    bases_theta_re = model_kwargs["bases_theta_re"]
    bases_theta_im = model_kwargs["bases_theta_im"]
    nb = float(model_kwargs["Nb"])
    nh = float(model_kwargs["Nh"])
    duration = float(model_kwargs["duration"])
    enbw = float(model_kwargs.get("enbw", 1.0))

    freq = np.asarray(data.freq, dtype=np.float64)
    p = int(model_kwargs["n_channels"])
    coords = _posterior_coords(posterior)
    coords["freq"] = freq

    channel_terms: dict[str, xr.DataArray] = {}
    total_pointwise: np.ndarray | None = None

    for channel_index in range(p):
        weights_delta = np.asarray(
            posterior[f"weights_delta_{channel_index}"].values,
            dtype=np.float64,
        )
        basis_delta = np.asarray(bases_delta[channel_index], dtype=np.float64)
        log_delta_sq = np.einsum("fk,cdk->cdf", basis_delta, weights_delta)
        log_delta_sq = np.clip(log_delta_sq, a_min=-80.0, a_max=80.0)
        delta_eff_sq = np.exp(log_delta_sq)

        pointwise = -nb * nh * np.log(delta_eff_sq)

        u_re_channel = u_re[:, channel_index, :]
        u_im_channel = u_im[:, channel_index, :]
        if channel_index > 0:
            theta_re = _channel_theta_array(
                posterior,
                list(bases_theta_re[channel_index]),
                channel_index,
                "re",
            )
            theta_im = _channel_theta_array(
                posterior,
                list(bases_theta_im[channel_index]),
                channel_index,
                "im",
            )
            u_re_prev = u_re[:, :channel_index, :]
            u_im_prev = u_im[:, :channel_index, :]

            contrib_re = np.einsum(
                "cdfl,flr->cdfr", theta_re, u_re_prev
            ) - np.einsum("cdfl,flr->cdfr", theta_im, u_im_prev)
            contrib_im = np.einsum(
                "cdfl,flr->cdfr", theta_re, u_im_prev
            ) + np.einsum("cdfl,flr->cdfr", theta_im, u_re_prev)
            u_re_resid = u_re_channel[None, None, :, :] - contrib_re
            u_im_resid = u_im_channel[None, None, :, :] - contrib_im
        else:
            u_re_resid = u_re_channel[None, None, :, :]
            u_im_resid = u_im_channel[None, None, :, :]

        residual_power_sum = np.sum(u_re_resid**2 + u_im_resid**2, axis=-1)
        pointwise = pointwise - residual_power_sum / (duration * delta_eff_sq)
        pointwise = pointwise / enbw

        channel_terms[f"log_likelihood_channel_{channel_index}"] = (
            xr.DataArray(
                pointwise,
                dims=("chain", "draw", "freq"),
                coords=coords,
            )
        )
        total_pointwise = (
            pointwise
            if total_pointwise is None
            else total_pointwise + pointwise
        )

    if total_pointwise is None:
        raise ValueError(
            "No multivariate channels available for pointwise log-likelihood."
        )

    return xr.Dataset(
        {
            "log_likelihood": xr.DataArray(
                total_pointwise,
                dims=("chain", "draw", "freq"),
                coords=coords,
            ),
            **channel_terms,
        }
    )


def compute_pointwise_lnl(
    *,
    posterior: xr.Dataset,
    data: WishartData,
    model_kwargs: dict[str, Any],
) -> xr.Dataset:
    """Compute pointwise log-likelihood contributions for PSIS-LOO.

    Each retained frequency bin is treated as one observation. For multivariate
    fits, the dataset includes a total ``log_likelihood`` variable plus
    per-channel diagnostics.
    """
    return _pointwise_multivar_log_likelihood(posterior, data, model_kwargs)


__all__ = ["compute_pointwise_lnl"]
