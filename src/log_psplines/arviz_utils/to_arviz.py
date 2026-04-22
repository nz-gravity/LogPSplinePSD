from __future__ import annotations

"""Helpers for ArviZ-compatible DataTree packing and PSD reconstruction."""
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from xarray import DataArray, Dataset

from log_psplines.datatypes import MultivarFFT

from ..logger import logger

if TYPE_CHECKING:
    from ..samplers.base_sampler import SamplerConfig


def _pack_spline_model(spline_model) -> Dataset:
    """Pack univariate spline model parameters into an xarray Dataset."""
    data: Dict[str, Any] = {
        "degree": spline_model.degree,
        "diffMatrixOrder": spline_model.diffMatrixOrder,
        "n": spline_model.n,
    }
    payload, coords = spline_model.to_storage_payload(
        include_linear_operators=False
    )
    data.update(payload)

    return Dataset(
        {
            key: (
                DataArray(value[1], dims=value[0])
                if isinstance(value, tuple)
                else DataArray(value)
            )
            for key, value in data.items()
        },
        coords=coords,
    )


def _pack_model_component(
    model, prefix: str, data: Dict[str, Any], coords: Dict[str, Any]
) -> None:
    """Pack a single multivariate component into storage dictionaries."""
    payload, component_coords = model.to_storage_payload(
        prefix=prefix, include_linear_operators=False
    )
    data.update(payload)
    coords.update(component_coords)


def _pack_spline_model_multivar(spline_model) -> Dataset:
    """Pack multivariate spline model parameters into an xarray Dataset."""
    data = {
        "degree": spline_model.degree,
        "diffMatrixOrder": spline_model.diffMatrixOrder,
        "N": spline_model.N,
        "p": spline_model.p,
        "n_theta": spline_model.n_theta,
    }

    coords: Dict[str, np.ndarray] = {}

    for i, diag_model in enumerate(spline_model.diagonal_models):
        _pack_model_component(diag_model, f"diag_{i}", data, coords)

    for j, l in spline_model.theta_pairs:
        _pack_model_component(
            spline_model.get_theta_model("re", j, l),
            f"theta_re_{j}_{l}",
            data,
            coords,
        )
        _pack_model_component(
            spline_model.get_theta_model("im", j, l),
            f"theta_im_{j}_{l}",
            data,
            coords,
        )

    return Dataset(
        {
            key: (
                DataArray(value[1], dims=value[0])
                if isinstance(value, tuple)
                else DataArray(value)
            )
            for key, value in data.items()
        },
        coords=coords,
    )


@jax.jit
def batch_spline_eval(
    basis: jnp.ndarray, weights_batch: jnp.ndarray
) -> jnp.ndarray:
    """JIT-compiled batch spline evaluation over multiple weight vectors."""
    return jnp.sum(basis[None, :, :] * weights_batch[:, None, :], axis=-1)


def _select_evenly_spaced_indices(
    n_total: int, n_keep: int
) -> np.ndarray | None:
    """Return evenly spaced indices for a capped posterior subset."""
    if n_total <= 0 or n_keep <= 0 or n_total <= n_keep:
        return None
    return np.unique(
        np.linspace(0, n_total - 1, num=n_keep, dtype=int, endpoint=True)
    )


def _flatten_posterior_draws(array: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Flatten leading chain/draw axes into a single sample axis."""
    arr = jnp.asarray(array)
    if arr.ndim <= 1:
        return arr
    if arr.ndim == 2:
        return arr
    return arr.reshape((-1,) + tuple(arr.shape[2:]))


def _subset_weight_samples_for_psd(
    samples: Dict[str, jnp.ndarray], n_keep: int
) -> Dict[str, jnp.ndarray]:
    """Return flattened weight samples capped to the draws needed for PSD summaries."""
    weight_keys = [key for key in samples if str(key).startswith("weights_")]
    if not weight_keys:
        return {}

    first_weights = jnp.asarray(samples[weight_keys[0]])
    if first_weights.ndim >= 3:
        n_total = int(first_weights.shape[0]) * int(first_weights.shape[1])
    else:
        n_total = int(first_weights.shape[0])
    keep_idx = _select_evenly_spaced_indices(n_total, int(n_keep))

    subset: Dict[str, jnp.ndarray] = {}
    for key in weight_keys:
        flat = _flatten_posterior_draws(samples[key])
        if keep_idx is not None:
            flat = flat[keep_idx]
        subset[str(key)] = flat
    return subset


def _flatten_and_cap_posterior_array(
    array: Optional[jnp.ndarray], n_keep: int
) -> Optional[jnp.ndarray]:
    """Flatten chain/draw axes and cap the sample axis when requested."""
    if array is None:
        return None
    flat = _flatten_posterior_draws(array)
    if flat.ndim == 0:
        return flat
    keep_idx = _select_evenly_spaced_indices(int(flat.shape[0]), int(n_keep))
    if keep_idx is not None:
        flat = flat[keep_idx]
    return flat


def _compute_posterior_predictive_multivar(
    samples: Dict[str, jnp.ndarray],
    sample_stats: Dict[str, jnp.ndarray],
    spline_model,
    fft_data: MultivarFFT,
    config: Optional["SamplerConfig"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute PSD percentiles (and optional coherence percentiles) from samples."""
    # Keep concise logging
    logger.debug("_compute_posterior_predictive_multivar: entry")

    # Fast-diagnostics controls from config
    n_draw_cap = 50
    compute_coh = fft_data.p > 1
    if config is not None:
        try:
            value = getattr(config, "posterior_psd_max_draws", None)
            if value is not None:
                value_int = int(value)
                if value_int > 0:
                    n_draw_cap = value_int
        except Exception:
            pass
        try:
            value = getattr(config, "compute_coherence_quantiles", None)
            if value is not None:
                compute_coh = bool(value)
        except Exception:
            pass

    log_delta_sq = _flatten_and_cap_posterior_array(
        sample_stats.get("log_delta_sq"), n_draw_cap
    )
    theta_re = _flatten_and_cap_posterior_array(
        sample_stats.get("theta_re"), n_draw_cap
    )
    theta_im = _flatten_and_cap_posterior_array(
        sample_stats.get("theta_im"), n_draw_cap
    )

    if log_delta_sq is None or theta_re is None or theta_im is None:
        samples_for_psd = _subset_weight_samples_for_psd(samples, n_draw_cap)
        if log_delta_sq is None:
            log_delta_sq = _reconstruct_log_delta_sq(
                samples_for_psd, spline_model, fft_data
            )
        if theta_re is None:
            theta_re = _reconstruct_theta_params(
                samples_for_psd, spline_model, fft_data, "re"
            )
        if theta_im is None:
            theta_im = _reconstruct_theta_params(
                samples_for_psd, spline_model, fft_data, "im"
            )

    percentiles = np.array([5.0, 50.0, 95.0], dtype=np.float64)
    psd_real_q, psd_imag_q, coh_q = spline_model.compute_psd_quantiles(
        log_delta_sq,
        theta_re,
        theta_im,
        percentiles=percentiles,
        n_samples_max=n_draw_cap,
        compute_coherence=compute_coh,
    )
    return percentiles, psd_real_q, psd_imag_q, coh_q


def _compute_prior_predictive_multivar(
    spline_model: "MultivariateLogPSplines",
    fft_data: MultivarFFT,
    config: "SamplerConfig",
    n_prior_draws: int = 500,
    seed: int = 42,
    log_delta_sq_clip: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw PSD matrices from the shrinkage prior and return quantiles.

    This mirrors the posterior predictive computation but samples weights
    from the P-spline prior (with design weights and tau shrinkage)
    instead of using MCMC draws.

    Returns
    -------
    psd_real_q, psd_imag_q : np.ndarray
        Shape ``(3, N, p, p)`` — the 5/50/95% quantiles in the
        **standardized** (channel-whitened) parameterization.
    """
    rng = np.random.default_rng(seed)
    N = spline_model.N
    p = spline_model.p

    # Resolve design weights from config
    design_weights: dict = {}
    design_psd_raw = getattr(config, "design_psd", None)
    if design_psd_raw is not None:
        # Re-compute design weights at model frequencies.
        # The sampler already did this, but we don't have a handle on its
        # internal _design_weights dict, so recompute cheaply here.
        if isinstance(design_psd_raw, tuple):
            src_freq, src_psd = design_psd_raw
            src_freq = np.asarray(src_freq)
            src_psd = np.asarray(src_psd, dtype=np.complex128)
            model_freq = np.asarray(fft_data.freq)
            aligned = np.stack(
                [
                    [
                        np.interp(model_freq, src_freq, src_psd[:, j, l].real)
                        + 1j
                        * np.interp(
                            model_freq, src_freq, src_psd[:, j, l].imag
                        )
                        for l in range(p)
                    ]
                    for j in range(p)
                ],
                axis=0,
            )
            design_psd = np.moveaxis(aligned, [0, 1, 2], [1, 2, 0])
        else:
            design_psd = np.asarray(design_psd_raw, dtype=np.complex128)

        # Apply channel standardization (the model works on standardized data)
        channel_stds = getattr(config, "channel_stds", None)
        if channel_stds is not None:
            stds = np.asarray(channel_stds, dtype=np.float64)
            scale_matrix = np.outer(stds, stds)
            design_psd = design_psd / scale_matrix[np.newaxis, :, :]

        design_weights = spline_model.compute_design_weights(design_psd)

    tau = getattr(config, "tau", None)
    alpha_phi = float(getattr(config, "alpha_phi", 1.0))
    beta_phi = float(getattr(config, "beta_phi", 1e-4))
    alpha_delta = float(getattr(config, "alpha_delta", 1.0))
    beta_delta = float(getattr(config, "beta_delta", 1.0))

    log_delta_sq_all = np.zeros((n_prior_draws, N, p))
    n_theta = p * (p - 1) // 2
    theta_re_all = np.zeros((n_prior_draws, N, n_theta))
    theta_im_all = np.zeros((n_prior_draws, N, n_theta))

    for draw_idx in range(n_prior_draws):
        for j in range(p):
            diag_model = spline_model.diagonal_models[j]
            basis = np.asarray(diag_model.basis)
            penalty = np.asarray(diag_model.penalty_matrix)
            k = basis.shape[1]

            delta = rng.gamma(shape=alpha_delta, scale=1.0 / beta_delta)
            phi = rng.gamma(
                shape=alpha_phi,
                scale=1.0 / (beta_phi * max(delta, 1e-12)),
            )

            w_d = np.asarray(design_weights.get(f"delta_{j}", np.zeros(k)))
            precision = phi * penalty + 1e-6 * np.eye(k)
            if tau is not None and f"delta_{j}" in design_weights:
                precision += np.eye(k) / tau**2
            cov = np.linalg.inv(precision)
            cov = 0.5 * (cov + cov.T)
            weights = rng.multivariate_normal(w_d, cov)
            log_delta_sq_all[draw_idx, :, j] = np.clip(
                basis @ weights, -log_delta_sq_clip, log_delta_sq_clip
            )

        if p > 1:
            theta_idx = 0
            for j_ch in range(1, p):
                for l_ch in range(j_ch):
                    for part, arr in [
                        ("re", theta_re_all),
                        ("im", theta_im_all),
                    ]:
                        theta_model = spline_model.get_theta_model(
                            part, j_ch, l_ch
                        )
                        theta_basis = np.asarray(theta_model.basis)
                        theta_penalty = np.asarray(theta_model.penalty_matrix)
                        k_theta = theta_basis.shape[1]
                        delta_t = rng.gamma(
                            shape=alpha_delta, scale=1.0 / beta_delta
                        )
                        phi_t = rng.gamma(
                            shape=alpha_phi,
                            scale=1.0 / (beta_phi * max(delta_t, 1e-12)),
                        )
                        key = f"theta_{part}_{j_ch}_{l_ch}"
                        w_d_t = np.asarray(
                            design_weights.get(key, np.zeros(k_theta))
                        )
                        prec_t = phi_t * theta_penalty + 1e-6 * np.eye(k_theta)
                        if tau is not None and key in design_weights:
                            prec_t += np.eye(k_theta) / tau**2
                        cov_t = np.linalg.inv(prec_t)
                        cov_t = 0.5 * (cov_t + cov_t.T)
                        w_t = rng.multivariate_normal(w_d_t, cov_t)
                        arr[draw_idx, :, theta_idx] = theta_basis @ w_t
                    theta_idx += 1

    percentiles = np.array([5.0, 50.0, 95.0], dtype=np.float64)
    psd_real_q, psd_imag_q, _ = spline_model.compute_psd_quantiles(
        log_delta_sq_all,
        theta_re_all,
        theta_im_all,
        percentiles=percentiles,
        n_samples_max=n_prior_draws,
        compute_coherence=False,
    )
    return (
        np.asarray(psd_real_q, dtype=np.float64),
        np.asarray(psd_imag_q, dtype=np.float64),
    )


def _reconstruct_log_delta_sq(
    samples: Dict[str, jnp.ndarray], spline_model, fft_data: MultivarFFT
) -> jnp.ndarray:
    """Reconstruct log_delta_sq from individual diagonal component samples."""
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    sample_key = next(
        (key for key in samples if str(key).startswith("weights_delta_")),
        next(iter(samples.keys())),
    )
    first_sample = _flatten_posterior_draws(samples[sample_key])
    n_samples = int(first_sample.shape[0]) if first_sample.ndim else 1
    log_delta_components = []

    for j in range(fft_data.p):
        weights_key = f"weights_delta_{j}"
        if weights_key in samples:
            weights = _flatten_posterior_draws(samples[weights_key])
            log_delta_j = batch_spline_eval(all_bases[j], weights)
            log_delta_components.append(log_delta_j)

    if log_delta_components:
        return jnp.stack(log_delta_components, axis=2)
    else:
        return jnp.zeros((n_samples, fft_data.N, fft_data.p))


def _reconstruct_theta_params(
    samples: Dict[str, jnp.ndarray],
    spline_model,
    fft_data: MultivarFFT,
    param_type: str,
) -> jnp.ndarray:
    """Reconstruct theta parameters from samples."""
    sample_key = next(
        (key for key in samples if str(key).startswith("weights_")),
        next(iter(samples.keys())),
    )
    first_sample = _flatten_posterior_draws(samples[sample_key])
    n_samples = int(first_sample.shape[0]) if first_sample.ndim else 1
    theta = jnp.zeros((n_samples, fft_data.N, max(1, spline_model.n_theta)))
    found = False

    if spline_model.n_theta > 0:
        for theta_idx, (j, l) in enumerate(spline_model.theta_pairs):
            key = f"weights_theta_{param_type}_{j}_{l}"
            if key not in samples:
                continue
            weights = _flatten_posterior_draws(samples[key])
            basis = jnp.asarray(
                spline_model.get_theta_model(param_type, j, l).basis
            )
            theta_eval = batch_spline_eval(basis, weights)
            theta = theta.at[:, :, theta_idx].set(theta_eval)
            found = True

    if found:
        return theta

    key = f"weights_theta_{param_type}"
    if key in samples and spline_model.n_theta > 0:
        weights = _flatten_posterior_draws(samples[key])
        first_j, first_l = spline_model.theta_pair_from_index(0)
        basis = jnp.asarray(
            spline_model.get_theta_model(param_type, first_j, first_l).basis
        )
        theta_base = batch_spline_eval(basis, weights)
        return jnp.tile(
            theta_base[:, :, None], (1, 1, max(1, spline_model.n_theta))
        )

    return theta
