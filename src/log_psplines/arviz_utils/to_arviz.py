from __future__ import annotations

from log_psplines.inference.initialisation import fit_design_weights

"""Helpers for ArviZ-compatible DataTree packing and PSD reconstruction."""
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from log_psplines.models.reconstruction import compute_psd_quantiles
from log_psplines.arviz_utils.spline_storage import to_storage_payload
from log_psplines.data import WishartData

if TYPE_CHECKING:
    from log_psplines.config import PipelineConfig, PowerSplineConfig
    from log_psplines.data.spectral import PowerSpectrum, ScatteredPowerSpectrum
    from log_psplines.inference.components import SpectralComponents
    from log_psplines.inference.vi import StageResult
    from log_psplines.models.spectrum import LogPSpline

SamplerConfig = Any


def _pack_model_component(
    model, prefix: str, data: dict[str, Any], coords: dict[str, Any]
) -> None:
    """Pack a single multivariate component into storage dictionaries."""
    payload, component_coords = to_storage_payload(
        model, prefix=prefix, include_linear_operators=False
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

    coords: dict[str, np.ndarray] = {}

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


def _compute_prior_predictive_multivar(
    spline_model: SpectralComponents,
    fft_data: WishartData,
    config: SamplerConfig,
    n_prior_draws: int = 500,
    seed: int = 42,
    log_delta_sq_clip: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
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

        design_weights = fit_design_weights(spline_model, design_psd)

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
    psd_real_q, psd_imag_q, _ = compute_psd_quantiles(
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
    samples: dict[str, jnp.ndarray], spline_model, fft_data: WishartData
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
    samples: dict[str, jnp.ndarray],
    spline_model,
    fft_data: WishartData,
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


def _losses_per_block_array(
    losses_per_block: list[jnp.ndarray] | None,
) -> np.ndarray:
    if not losses_per_block:
        return np.asarray([], dtype=float)

    arrays = [
        np.asarray(losses, dtype=float).reshape(-1)
        for losses in losses_per_block
    ]
    max_len = max((arr.size for arr in arrays), default=0)
    if max_len == 0:
        return np.asarray([], dtype=float)

    padded = np.full((len(arrays), max_len), np.nan, dtype=float)
    for idx, arr in enumerate(arrays):
        padded[idx, : arr.size] = arr
    return padded


def _vi_result_to_idata(result: StageResult) -> xr.DataTree:
    """Wrap VI posterior draws into a minimal xr.DataTree."""
    has_samples = result.samples is not None
    values = result.samples if has_samples else result.init_values
    if not values:
        return xr.DataTree()
    ds = _posterior_values_to_dataset(values, values_are_draws=has_samples)
    return xr.DataTree(children={"posterior": xr.DataTree(dataset=ds)})


def _posterior_values_to_dataset(
    values: dict[str, jnp.ndarray],
    *,
    values_are_draws: bool,
) -> xr.Dataset:
    """Pack posterior-like values using ``chain``/``draw`` leading dims."""
    data_vars = {}
    draw_count: int | None = None
    for name, value in values.items():
        array = np.asarray(value)
        if values_are_draws:
            if array.ndim == 0:
                raise ValueError(
                    f"Posterior samples for '{name}' must include a draw axis."
                )
            array = array[None, ...]
        else:
            array = array[None, None, ...]
        if draw_count is None:
            draw_count = int(array.shape[1])
        elif int(array.shape[1]) != draw_count:
            raise ValueError(
                f"Posterior value '{name}' has {array.shape[1]} draws; "
                f"expected {draw_count}."
            )

        tail_dims = tuple(
            f"{name}_dim_{axis}" for axis in range(array.ndim - 2)
        )
        data_vars[name] = xr.DataArray(
            array,
            dims=("chain", "draw", *tail_dims),
        )

    n_draws = int(draw_count or 0)
    return xr.Dataset(
        data_vars,
        coords={"chain": [0], "draw": np.arange(n_draws)},
    )


def _init_values_to_dataset(values: dict[str, jnp.ndarray]) -> xr.Dataset:
    """Pack VI point estimates using variable-specific trailing dimensions."""
    data_vars = {}
    for name, value in values.items():
        array = np.asarray(value)[None, None, ...]
        tail_dims = tuple(
            f"{name}_dim_{axis}" for axis in range(array.ndim - 2)
        )
        data_vars[name] = xr.DataArray(
            array,
            dims=("chain", "draw", *tail_dims),
        )
    return xr.Dataset(
        data_vars,
        coords={"chain": [0], "draw": [0]},
    )


def _observed_data_dataset(data: WishartData) -> xr.Dataset:
    freq = np.asarray(data.freq, dtype=float)
    channel_coords = np.arange(int(data.p))
    coords = {
        "freq": freq,
        "channels": channel_coords,
        "channels_aux": channel_coords,
    }
    dims = ("freq", "channels", "channels_aux")
    variables = {}
    if data.raw_psd is not None:
        variables["periodogram"] = xr.DataArray(
            np.asarray(data.raw_psd, dtype=np.complex128),
            dims=dims,
            coords=coords,
        )
    return xr.Dataset(variables, coords=coords)


def _vi_posterior_dataset(vi: StageResult) -> xr.Dataset:
    has_samples = vi.samples is not None
    values = vi.samples if has_samples else vi.init_values
    if not values:
        return xr.Dataset()
    return _posterior_values_to_dataset(
        values,
        values_are_draws=has_samples,
    )


def pack_stationary_result(
    idata: xr.DataTree,
    data: WishartData,
    spline_model: SpectralComponents,
    config: PipelineConfig,
    sampling_eta: float,
    vi: StageResult | None,
) -> xr.DataTree:
    """Attach model/data groups needed by diagnostics and plotting."""
    spline_ds = _pack_spline_model_multivar(spline_model)
    attrs = {
        "data_type": "multivariate",
        "scaling_factor": float(data.scaling_factor or 1.0),
        "channel_stds": (
            None
            if data.channel_stds is None
            else np.asarray(data.channel_stds)
        ),
        "sampler": "factorized_multivar_nuts",
    }

    attrs.update(
        {
            "max_tree_depth": int(config.max_tree_depth),
            "posterior_psd_max_draws": int(config.vi_psd_max_draws),
            "vi_psd_max_draws": int(config.vi_psd_max_draws),
            "alpha_phi": float(config.alpha_phi),
            "beta_phi": float(config.beta_phi),
            "alpha_delta": float(config.alpha_delta),
            "beta_delta": float(config.beta_delta),
            "eta": float(config.eta),
            "sampling_eta": float(sampling_eta),
        }
    )
    attrs["compute_lnz"] = bool(config.compute_lnz)
    if config.target_accept_prob_by_channel is not None:
        attrs["target_accept_prob_by_channel"] = list(
            config.target_accept_prob_by_channel
        )
    if config.max_tree_depth_by_channel is not None:
        attrs["max_tree_depth_by_channel"] = list(
            config.max_tree_depth_by_channel
        )
    for channel_index in range(int(data.p)):
        attrs[f"sampling_eta_channel_{channel_index}"] = float(sampling_eta)
    idata.attrs.update(attrs)
    idata["observed_data"] = xr.DataTree(dataset=_observed_data_dataset(data))
    idata["spline_model"] = xr.DataTree(dataset=spline_ds)

    if vi is not None:
        idata["vi_posterior"] = xr.DataTree(dataset=_vi_posterior_dataset(vi))
        losses = (
            np.asarray(vi.losses, dtype=float)
            if vi.losses is not None
            else np.asarray([], dtype=float)
        )
        vi_stats = xr.Dataset(
            {
                "losses": xr.DataArray(
                    losses,
                    dims=("draw",),
                    coords={"draw": np.arange(losses.size)},
                )
            }
        )
        losses_per_block = _losses_per_block_array(vi.losses_per_block)
        if losses_per_block.size:
            vi_stats["losses_per_block"] = xr.DataArray(
                losses_per_block,
                dims=("factor", "draw_per_factor"),
                coords={
                    "factor": np.arange(losses_per_block.shape[0]),
                    "draw_per_factor": np.arange(losses_per_block.shape[1]),
                },
            )
        idata["vi_sample_stats"] = xr.DataTree(dataset=vi_stats)
        # Pointwise VI likelihoods are not computed here. Omit the group
        # instead of presenting zero placeholders as observations.
    return idata


def pack_power_result(
    data: PowerSpectrum,
    spline: LogPSpline,
    config: PowerSplineConfig,
    samples: dict[str, np.ndarray],
    stats: dict[str, jnp.ndarray],
) -> xr.DataTree:
    """Store compact coefficients and explicit grids for scalar power fits."""
    from dataclasses import asdict

    posterior = {}
    for name, value in samples.items():
        dims = ("chain", "draw")
        if name == "weights":
            dims += ("time_coefficient", "frequency_coefficient")
        elif np.ndim(value) > 2:
            dims += ("eigen_coefficient",)
        posterior[name] = (dims, value)
    observed = xr.Dataset(
        {
            "power": (("time", "frequency"), data.power),
            "counts": (("time", "frequency"), data.counts),
        },
        coords={"time": data.time, "frequency": data.frequency},
        attrs={"units": data.units},
    )
    basis = xr.Dataset(
        {
            "basis_time": (("time", "time_coefficient"), spline.time.basis),
            "basis_frequency": (
                ("frequency", "frequency_coefficient"),
                spline.basis,
            ),
            "penalty_time": (
                ("time_coefficient", "time_coefficient_aux"),
                spline.time.penalty,
            ),
            "penalty_frequency": (
                ("frequency_coefficient", "frequency_coefficient_aux"),
                spline.frequency.penalty,
            ),
            "knots_time": (("time_knot",), spline.time.knots),
            "knots_frequency": (("frequency_knot",), spline.frequency.knots),
            "grid_time": (("time",), spline.time.grid),
            "grid_frequency": (("frequency",), spline.frequency.grid),
        },
        attrs={
            "degree_time": spline.time.degree,
            "degree_frequency": spline.degree,
            "penalty_order_time": spline.time.penalty_order,
            "penalty_order_frequency": spline.frequency.penalty_order,
            **{
                f"{name}_{axis}": getattr(basis, name)
                for axis, basis in (
                    ("time", spline.time),
                    ("frequency", spline.frequency),
                )
                for name in (
                    "penalty_normalization",
                    "penalty_ridge",
                    "knot_convention",
                )
            },
        },
    )
    idata = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                attrs={
                    **asdict(config),
                    "likelihood": "power_whittle",
                    "units": data.units,
                }
            ),
            "posterior": xr.Dataset(
                posterior,
                coords={
                    "chain": np.arange(config.num_chains),
                    "draw": np.arange(config.n_samples),
                },
            ),
            "observed_data": observed,
            "power_basis": basis,
            "sample_stats": xr.Dataset(
                {
                    name: (("chain", "draw"), np.asarray(value))
                    for name, value in stats.items()
                }
            ),
        }
    )
    return idata


def pack_scattered_power_result(
    data: "ScatteredPowerSpectrum",
    spline: LogPSpline,
    config: PowerSplineConfig,
    samples: dict[str, np.ndarray],
    stats: dict[str, jnp.ndarray],
) -> xr.DataTree:
    """Store compact coefficients and scattered ordinates for point fits.

    Companion to :func:`pack_power_result` for data without a shared
    time/frequency axis (e.g. raw Tang moving-periodogram ordinates). The
    ``power_basis`` group still holds the spline's own reconstruction grid
    (``spline.time.grid``/``spline.frequency.grid``), independent of where
    the ordinates were observed.
    """
    from dataclasses import asdict

    posterior = {}
    for name, value in samples.items():
        dims = ("chain", "draw")
        if name == "weights":
            dims += ("time_coefficient", "frequency_coefficient")
        elif np.ndim(value) > 2:
            dims += ("eigen_coefficient",)
        posterior[name] = (dims, value)
    observed = xr.Dataset(
        {
            "power": (("ordinate",), data.power),
            "counts": (("ordinate",), data.counts),
            "time": (("ordinate",), data.time),
            "frequency": (("ordinate",), data.frequency),
        },
        coords={"ordinate": np.arange(data.power.size)},
        attrs={"units": data.units},
    )
    basis = xr.Dataset(
        {
            "basis_time": (("time", "time_coefficient"), spline.time.basis),
            "basis_frequency": (
                ("frequency", "frequency_coefficient"),
                spline.basis,
            ),
            "penalty_time": (
                ("time_coefficient", "time_coefficient_aux"),
                spline.time.penalty,
            ),
            "penalty_frequency": (
                ("frequency_coefficient", "frequency_coefficient_aux"),
                spline.frequency.penalty,
            ),
            "knots_time": (("time_knot",), spline.time.knots),
            "knots_frequency": (("frequency_knot",), spline.frequency.knots),
            "grid_time": (("time",), spline.time.grid),
            "grid_frequency": (("frequency",), spline.frequency.grid),
        },
        attrs={
            "degree_time": spline.time.degree,
            "degree_frequency": spline.degree,
            "penalty_order_time": spline.time.penalty_order,
            "penalty_order_frequency": spline.frequency.penalty_order,
            **{
                f"{name}_{axis}": getattr(basis, name)
                for axis, basis in (
                    ("time", spline.time),
                    ("frequency", spline.frequency),
                )
                for name in (
                    "penalty_normalization",
                    "penalty_ridge",
                    "knot_convention",
                )
            },
        },
    )
    idata = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                attrs={
                    **asdict(config),
                    "likelihood": "power_whittle",
                    "units": data.units,
                }
            ),
            "posterior": xr.Dataset(
                posterior,
                coords={
                    "chain": np.arange(config.num_chains),
                    "draw": np.arange(config.n_samples),
                },
            ),
            "observed_data": observed,
            "power_basis": basis,
            "sample_stats": xr.Dataset(
                {
                    name: (("chain", "draw"), np.asarray(value))
                    for name, value in stats.items()
                }
            ),
        }
    )
    return idata
