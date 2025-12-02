"""Convert inference results to ArviZ InferenceData format."""

import warnings
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson
from xarray import DataArray, Dataset

from log_psplines.datatypes import MultivarFFT, Periodogram

from ..logger import logger
from ..spectrum_utils import wishart_u_to_psd

warnings.filterwarnings("ignore", module="arviz")


def results_to_arviz(
    samples: Dict[str, jnp.ndarray],
    sample_stats: Dict[str, Any],
    config: "SamplerConfig",
    data: Union[Periodogram, MultivarFFT],
    model: Union["LogPSplines", "MultivariateLogPSplines"],
    attributes: Dict[str, Any],
) -> az.InferenceData:
    """Unified ArviZ conversion for both univar and multivar cases."""
    logger.debug(f"results_to_arviz: entry")
    logger.debug(f"results_to_arviz: samples keys={list(samples.keys())}")
    logger.debug(
        f"results_to_arviz: sample_stats keys={list(sample_stats.keys())}"
    )

    if isinstance(data, Periodogram):
        logger.debug("results_to_arviz: detected Periodogram")
        idata = _create_univar_inference_data(
            samples, sample_stats, config, data, model, attributes
        )
    elif isinstance(data, MultivarFFT):
        logger.debug("results_to_arviz: detected MultivarFFT")
        idata = _create_multivar_inference_data(
            samples, sample_stats, config, data, model, attributes
        )
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Compute diagnostics if true_psd is provided (using posterior_psd from idata)
    if config.true_psd is not None:
        logger.debug(
            "results_to_arviz: computing PSD diagnostics (true_psd provided)"
        )
        idata.attrs.update(
            _compute_psd_diagnostics(idata, config, data, model)
        )

    logger.debug("results_to_arviz: exit")
    return idata


def _add_chain_dim(
    data_dict: Dict[str, Any], num_chains: int
) -> Dict[str, Any]:
    """Ensure sample arrays include an explicit chain dimension."""
    result = {}
    for k, v in data_dict.items():
        v_array = np.array(v)
        # If the leading dimension already matches the configured number of chains,
        # keep the array as-is; otherwise, add a singleton chain dimension.
        if v_array.ndim >= 2 and v_array.shape[0] == num_chains:
            result[k] = v_array
        elif v_array.ndim == 0:
            result[k] = v_array
        else:
            result[k] = v_array[None, ...]
    return result


def _prepare_samples_and_stats(
    samples: Dict[str, Any], sample_stats: Dict[str, Any], num_chains: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare samples and sample_stats by adding chain dimensions."""
    logger.debug("_prepare_samples_and_stats: ensuring chain dims present")
    samples = _add_chain_dim(samples, num_chains)
    sample_stats = _add_chain_dim(sample_stats, num_chains)
    return samples, sample_stats


def _handle_log_posterior(sample_stats: Dict[str, Any]) -> None:
    """Add log posterior to sample_stats if likelihood and prior exist."""
    if {"log_likelihood", "log_prior"}.issubset(sample_stats.keys()):
        try:
            sample_stats["lp"] = (
                sample_stats["log_likelihood"] + sample_stats["log_prior"]
            )
        except Exception:
            logger.exception("_handle_log_posterior: failed to compute lp")


def _prepare_attributes_and_dims(
    config: "SamplerConfig",
    attributes: Dict[str, Any],
    samples: Dict[str, Any],
    coords: Dict[str, Any],
    dims: Dict[str, Any],
    sample_stats: Dict[str, Any],
    data,
    model,
) -> None:
    """Prepare attributes, coordinates, and dimensions for InferenceData."""
    # Convert config to attributes (handle booleans)
    config_attrs = {
        k: (int(v) if isinstance(v, bool) else v)
        for k, v in asdict(config).items()
        if v is not None
    }

    true_psd_attr = config_attrs.pop("true_psd", None)
    if true_psd_attr is not None:
        config_attrs["true_psd_provided"] = 1

    attributes.update(config_attrs)

    # Add ESS calculation (best-effort; keep small set for speed)
    try:
        ess_vars = list(samples.keys())
        if len(ess_vars) > 10:
            ess_vars = ess_vars[:10]
        summary = az.summary(samples, var_names=ess_vars, round_to=2)
        ess_vals = summary["ess_bulk"].values
        attributes.update(dict(ess=ess_vals))
    except Exception:
        logger.exception(
            "_prepare_attributes_and_dims: ESS computation failed"
        )
        attributes.update(dict(ess=[]))

    # Base coords
    first_key = list(samples.keys())[0]
    coords.update(
        {
            "chain": range(samples[first_key].shape[0]),
            "draw": range(samples[first_key].shape[1]),
        }
    )

    # Add log posterior to dims if it was added to sample_stats
    if "lp" in sample_stats:
        dims["lp"] = ["chain", "draw"]


def _pack_spline_model(spline_model) -> Dataset:
    """Pack univariate spline model parameters into xarray Dataset."""
    data = {
        "knots": (["knots_dim"], np.array(spline_model.knots)),
        "degree": spline_model.degree,
        "diffMatrixOrder": spline_model.diffMatrixOrder,
        "n": spline_model.n,
        "basis": (["freq", "weights_dim"], np.array(spline_model.basis)),
        "penalty_matrix": (
            ["weights_dim_row", "weights_dim_col"],
            np.array(spline_model.penalty_matrix),
        ),
        "parametric_model": (
            ["freq"],
            np.array(spline_model.parametric_model),
        ),
    }

    coords = {
        "knots_dim": np.arange(len(spline_model.knots)),
        "weights_dim": np.arange(spline_model.basis.shape[1]),
        "weights_dim_row": np.arange(spline_model.penalty_matrix.shape[0]),
        "weights_dim_col": np.arange(spline_model.penalty_matrix.shape[1]),
        "freq": np.arange(spline_model.basis.shape[0]),
    }

    return Dataset(
        {
            k: (
                DataArray(v[1], dims=v[0])
                if isinstance(v, tuple)
                else DataArray(v)
            )
            for k, v in data.items()
        },
        coords=coords,
    )


def _create_univar_inference_data(
    samples: dict,
    sample_stats: dict,
    config: "SamplerConfig",
    periodogram: "Periodogram",
    spline_model: "LogPSplines",
    attributes: Dict[str, Any],
) -> az.InferenceData:
    """Create InferenceData for univariate case."""
    # Prepare samples and stats with chain dimensions
    samples, sample_stats = _prepare_samples_and_stats(
        samples, sample_stats, config.num_chains
    )
    logger.debug("_create_univar_inference_data: entry")

    # Extract dimensions
    n_chains, n_draws, n_weights = samples["weights"].shape

    # Create posterior predictive samples
    weights_chain0 = samples["weights"][0]  # First chain
    n_pp = min(500, n_draws)
    pp_idx = (
        np.random.choice(n_draws, n_pp, replace=False)
        if n_draws > n_pp
        else slice(None)
    )

    percentiles = np.array([5.0, 50.0, 95.0], dtype=np.float32)

    # Evaluate spline_model on a subset of draws
    pp_eval = [spline_model(w) for w in weights_chain0[pp_idx]]
    pp_samples = np.array(pp_eval, dtype=np.float64)
    pp_samples = np.exp(pp_samples)

    observed_power = np.array(periodogram.power, dtype=np.float64)
    pp_samples_rescaled = pp_samples * config.scaling_factor
    psd_percentiles = np.percentile(
        pp_samples_rescaled, percentiles, axis=0
    ).astype(np.float32)
    observed_power_rescaled = observed_power * config.scaling_factor

    _handle_log_posterior(sample_stats)

    coords = {
        "percentile": percentiles,
        "weight_dim": range(n_weights),
        "freq": periodogram.freqs,
    }

    dims = {
        "phi": ["chain", "draw"],
        "delta": ["chain", "draw"],
        "weights": ["chain", "draw", "weight_dim"],
        **{k: ["chain", "draw"] for k in sample_stats.keys()},
        "periodogram": ["freq"],
    }

    _prepare_attributes_and_dims(
        config,
        attributes,
        samples,
        coords,
        dims,
        sample_stats,
        periodogram,
        spline_model,
    )

    idata = az.from_dict(
        posterior=samples,
        sample_stats=sample_stats,
        observed_data={"periodogram": observed_power_rescaled},
        dims={k: v for k, v in dims.items() if k not in ["psd", "lp"]},
        coords=coords,
        attrs=attributes,
    )

    idata.add_groups(
        posterior_psd=Dataset(
            {"psd": DataArray(psd_percentiles, dims=["percentile", "freq"])},
            coords={
                "percentile": coords["percentile"],
                "freq": coords["freq"],
            },
        )
    )

    idata.add_groups(spline_model=_pack_spline_model(spline_model))
    return idata


def _create_multivar_inference_data(
    samples: dict,
    sample_stats: dict,
    config: "SamplerConfig",
    fft_data: MultivarFFT,
    spline_model: "MultivariateLogPSplines",
    attributes: Dict[str, Any],
) -> az.InferenceData:
    """Create InferenceData for multivariate case."""
    # Prepare samples and stats with chain dimensions
    samples, sample_stats = _prepare_samples_and_stats(
        samples, sample_stats, config.num_chains
    )
    logger.debug("_create_multivar_inference_data: entry")

    # Extract dimensions from a standard sample
    first_sample_key = next(
        (k for k, v in samples.items() if "weights_" in k),
        next(iter(samples.keys())),
    )
    sample_shape = samples[first_sample_key].shape
    n_chains, n_draws = sample_shape[:2]

    # Create posterior predictive summaries
    percentiles, psd_real_q, psd_imag_q, coh_q = (
        _compute_posterior_predictive_multivar(
            samples, sample_stats, spline_model, fft_data
        )
    )
    psd_real_q = np.asarray(psd_real_q, dtype=np.float32)
    psd_imag_q = np.asarray(psd_imag_q, dtype=np.float32)
    coh_q = np.asarray(coh_q, dtype=np.float32) if coh_q is not None else None

    channel_stds = getattr(config, "channel_stds", None)
    factor_matrix = None
    sf = float(getattr(fft_data, "scaling_factor", 1.0) or 1.0)
    if channel_stds is not None:
        channel_stds = np.asarray(channel_stds, dtype=np.float32)
        if channel_stds.shape[0] != fft_data.n_dim:
            raise ValueError(
                "channel_stds length must match number of channels in FFT data."
            )
        scale_matrix = np.outer(channel_stds, channel_stds).astype(np.float32)
        factor_matrix = scale_matrix / sf if sf != 0 else scale_matrix
        factor_4d = factor_matrix[None, None, :, :]
        psd_real_q_rescaled = psd_real_q * factor_4d
        psd_imag_q_rescaled = psd_imag_q * factor_4d
    else:
        psd_real_q_rescaled = psd_real_q
        psd_imag_q_rescaled = psd_imag_q
    coherence_q_rescaled = coh_q
    scalar_factor = float(getattr(config, "scaling_factor", 1.0) or 1.0)

    fft_y_re = np.array(fft_data.y_re)
    fft_y_im = np.array(fft_data.y_im)

    # Also rescale observed FFT data
    if channel_stds is not None:
        observed_fft_re_rescaled = fft_y_re * channel_stds[None, :]
        observed_fft_im_rescaled = fft_y_im * channel_stds[None, :]
    else:
        observed_fft_re_rescaled = fft_y_re
        observed_fft_im_rescaled = fft_y_im

    # Compute and rescale observed cross-spectral density (periodogram)
    raw_psd = getattr(fft_data, "raw_psd", None)
    psd_has_global_scale = False
    if raw_psd is not None:
        observed_csd = np.asarray(raw_psd, dtype=np.complex128)
        psd_has_global_scale = True
        # raw_psd already has scaling_factor applied, so remove it before rescaling
        if channel_stds is not None:
            observed_csd = observed_csd / sf
    elif fft_data.u_re is not None and fft_data.u_im is not None:
        u_re = np.asarray(fft_data.u_re, dtype=np.float64)
        u_im = np.asarray(fft_data.u_im, dtype=np.float64)
        u_complex = u_re + 1j * u_im
        weights = (
            np.asarray(config.freq_weights, dtype=np.float64)
            if config.freq_weights is not None
            else None
        )
        if weights is not None and weights.shape != (fft_data.n_freq,):
            raise ValueError(
                "Frequency weights length must match number of frequencies."
            )
        observed_csd = wishart_u_to_psd(
            u_complex,
            nu=getattr(fft_data, "nu", 1),
            scaling_factor=float(getattr(fft_data, "scaling_factor", 1.0)),
            weights=weights,
        )
        psd_has_global_scale = True
        # wishart_u_to_psd already has scaling_factor applied, so remove it before rescaling
        if channel_stds is not None:
            observed_csd = observed_csd / sf
    else:
        y_re = observed_fft_re_rescaled
        y_im = observed_fft_im_rescaled
        n_freq, n_dim = y_re.shape
        observed_csd = np.zeros((n_freq, n_dim, n_dim), dtype=np.complex64)
        for i in range(n_dim):
            for j in range(n_dim):
                observed_csd[:, i, j] = (
                    y_re[:, i] + 1j * y_im[:, i]
                ) * np.conj(y_re[:, j] + 1j * y_im[:, j])
        # Keep complex form for consistency

    if channel_stds is not None and factor_matrix is not None:
        observed_csd = observed_csd * factor_matrix[None, :, :]
    else:
        if not psd_has_global_scale:
            observed_csd = observed_csd

    if config.verbose:
        logger.info(
            f"Rescaling multivariate posterior samples: max scaling ~{config.scaling_factor:.2e}"
        )

    observed_fft_data_rescaled = {
        "fft_re": observed_fft_re_rescaled,
        "fft_im": observed_fft_im_rescaled,
    }
    observed_psd_rescaled = {"periodogram": observed_csd}

    _handle_log_posterior(sample_stats)

    coords = {
        "percentile": percentiles,
        "freq": np.array(fft_data.freq),
        "channels": range(fft_data.n_dim),
        "channels2": range(fft_data.n_dim),
    }

    dims = {}

    for key, array in samples.items():
        array_shape = array.shape
        if key.startswith("weights_"):
            component = key[8:]  # Remove "weights_" prefix
            dims[key] = ["chain", "draw", f"{component}_basis_dim"]
            coords[f"{component}_basis_dim"] = range(array_shape[-1])
        elif key in ["phi", "delta"] or key.startswith(("phi_", "delta_")):
            dims[key] = ["chain", "draw"]

    for key, array in sample_stats.items():
        array_shape = array.shape
        if key == "log_delta_sq" and len(array_shape) == 4:
            dims[key] = ["chain", "draw", "freq", "channels"]
        elif key in ["theta_re", "theta_im"] and len(array_shape) == 4:
            dims[key] = ["chain", "draw", "freq", f"{key}_theta_dim"]
            coords[f"{key}_theta_dim"] = range(array_shape[-1])
        elif key == "log_likelihood":
            dims[key] = ["chain", "draw"]
        elif len(array_shape) == 2:
            dims[key] = ["chain", "draw"]
        else:
            dims[key] = ["chain", "draw"] + [
                f"{key}_dim_{i}" for i in range(len(array_shape) - 2)
            ]
            for i in range(2, len(array_shape)):
                coords[f"{key}_dim_{i-2}"] = range(array_shape[i])

    dims.update(
        {
            "fft_re": ["freq", "channels"],
            "fft_im": ["freq", "channels"],
            "periodogram": ["freq", "channels", "channels2"],
            "psd_matrix_real": ["percentile", "freq", "channels", "channels2"],
            "psd_matrix_imag": ["percentile", "freq", "channels", "channels2"],
        }
    )
    if coherence_q_rescaled is not None:
        dims["coherence"] = ["percentile", "freq", "channels", "channels2"]

    attributes.update(
        {
            "data_type": "multivariate",
            "n_channels": fft_data.n_dim,
            "n_freq": fft_data.n_freq,
            "n_theta": spline_model.n_theta,
            "frequencies": np.array(fft_data.freq),
        }
    )

    _prepare_attributes_and_dims(
        config,
        attributes,
        samples,
        coords,
        dims,
        sample_stats,
        fft_data,
        spline_model,
    )

    idata = az.from_dict(
        posterior=samples,
        sample_stats=sample_stats,
        observed_data={
            "fft_re": np.array(observed_fft_data_rescaled["fft_re"]),
            "fft_im": np.array(observed_fft_data_rescaled["fft_im"]),
            "periodogram": np.array(
                observed_psd_rescaled["periodogram"], dtype=np.complex128
            ),
        },
        dims={
            k: v
            for k, v in dims.items()
            if k
            not in [
                "psd_matrix_real",
                "psd_matrix_imag",
                "coherence",
                "psd",
                "lp",
            ]
        },
        coords=coords,
        attrs=attributes,
    )

    posterior_psd_vars = {
        "psd_matrix_real": DataArray(
            psd_real_q_rescaled,
            dims=["percentile", "freq", "channels", "channels2"],
            coords={
                "percentile": coords["percentile"],
                "freq": coords["freq"],
                "channels": coords["channels"],
                "channels2": coords["channels"],
            },
        ),
        "psd_matrix_imag": DataArray(
            psd_imag_q_rescaled,
            dims=["percentile", "freq", "channels", "channels2"],
            coords={
                "percentile": coords["percentile"],
                "freq": coords["freq"],
                "channels": coords["channels"],
                "channels2": coords["channels"],
            },
        ),
    }
    if coherence_q_rescaled is not None:
        posterior_psd_vars["coherence"] = DataArray(
            coherence_q_rescaled,
            dims=["percentile", "freq", "channels", "channels2"],
            coords={
                "percentile": coords["percentile"],
                "freq": coords["freq"],
                "channels": coords["channels"],
                "channels2": coords["channels"],
            },
        )

    idata.add_groups(posterior_psd=Dataset(posterior_psd_vars, coords=coords))

    idata.add_groups(spline_model=_pack_spline_model_multivar(spline_model))
    return idata


@jax.jit
def batch_spline_eval(
    basis: jnp.ndarray, weights_batch: jnp.ndarray
) -> jnp.ndarray:
    """JIT-compiled batch spline evaluation over multiple weight vectors."""
    return jnp.sum(basis[None, :, :] * weights_batch[:, None, :], axis=-1)


def _pack_model_component(
    model, prefix: str, data: Dict[str, Any], coords: Dict[str, Any]
) -> None:
    """Pack a single model component into data and coords dicts."""
    data.update(
        {
            f"{prefix}_knots": (
                [f"{prefix}_knots_dim"],
                np.array(model.knots),
            ),
            f"{prefix}_basis": (
                [f"{prefix}_freq", f"{prefix}_weights_dim"],
                np.array(model.basis),
            ),
            f"{prefix}_penalty_matrix": (
                [f"{prefix}_weights_dim_row", f"{prefix}_weights_dim_col"],
                np.array(model.penalty_matrix),
            ),
            f"{prefix}_parametric_model": (
                [f"{prefix}_freq"],
                np.array(model.parametric_model),
            ),
        }
    )
    coords.update(
        {
            f"{prefix}_knots_dim": np.arange(len(model.knots)),
            f"{prefix}_weights_dim": np.arange(model.basis.shape[1]),
            f"{prefix}_weights_dim_row": np.arange(
                model.penalty_matrix.shape[0]
            ),
            f"{prefix}_weights_dim_col": np.arange(
                model.penalty_matrix.shape[1]
            ),
            f"{prefix}_freq": np.arange(model.basis.shape[0]),
        }
    )


def _pack_spline_model_multivar(spline_model) -> Dataset:
    """Pack multivariate spline model parameters into xarray Dataset."""
    data = {
        "degree": spline_model.degree,
        "diffMatrixOrder": spline_model.diffMatrixOrder,
        "n_freq": spline_model.n_freq,
        "n_channels": spline_model.n_channels,
        "n_theta": spline_model.n_theta,
    }

    coords = {}

    for i, diag_model in enumerate(spline_model.diagonal_models):
        _pack_model_component(diag_model, f"diag_{i}", data, coords)

    if spline_model.offdiag_re_model is not None:
        _pack_model_component(
            spline_model.offdiag_re_model, "offdiag_re", data, coords
        )
    if spline_model.offdiag_im_model is not None:
        _pack_model_component(
            spline_model.offdiag_im_model, "offdiag_im", data, coords
        )

    return Dataset(
        {
            k: (
                DataArray(v[1], dims=v[0])
                if isinstance(v, tuple)
                else DataArray(v)
            )
            for k, v in data.items()
        },
        coords=coords,
    )


def _compute_posterior_predictive_multivar(
    samples: Dict[str, jnp.ndarray],
    sample_stats: Dict[str, jnp.ndarray],
    spline_model,
    fft_data: MultivarFFT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute PSD percentiles (and optional coherence percentiles) from samples."""
    # Keep concise logging
    logger.debug("_compute_posterior_predictive_multivar: entry")

    def _flatten_chain_dim(
        array: Optional[jnp.ndarray],
    ) -> Optional[jnp.ndarray]:
        """Flatten chain dimension (if present) into the draw dimension."""
        if array is None:
            return None
        arr = jnp.asarray(array)
        if arr.ndim == 0:
            return arr
        if arr.ndim >= 3:
            # Handle arrays with explicit chain dimension inserted by _prepare_samples_and_stats
            if arr.shape[1] == fft_data.n_freq:
                return arr
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.reshape((-1,) + tuple(arr.shape[2:]))
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    log_delta_sq = _flatten_chain_dim(sample_stats.get("log_delta_sq"))
    theta_re = _flatten_chain_dim(sample_stats.get("theta_re"))
    theta_im = _flatten_chain_dim(sample_stats.get("theta_im"))

    if log_delta_sq is None:
        log_delta_sq = _reconstruct_log_delta_sq(
            samples, spline_model, fft_data
        )

    if theta_re is None:
        theta_re = _reconstruct_theta_params(
            samples, spline_model, fft_data, "re"
        )

    if theta_im is None:
        theta_im = _reconstruct_theta_params(
            samples, spline_model, fft_data, "im"
        )

    percentiles = np.array([5.0, 50.0, 95.0], dtype=np.float32)
    psd_real_q, psd_imag_q, coh_q = spline_model.compute_psd_quantiles(
        log_delta_sq,
        theta_re,
        theta_im,
        percentiles=percentiles,
        n_samples_max=50,
        compute_coherence=fft_data.n_dim > 1,
    )
    return percentiles, psd_real_q, psd_imag_q, coh_q


def _reconstruct_log_delta_sq(
    samples: Dict[str, jnp.ndarray], spline_model, fft_data: MultivarFFT
) -> jnp.ndarray:
    """Reconstruct log_delta_sq from individual diagonal component samples."""
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    first_sample = next(iter(samples.values()))
    n_chains = first_sample.shape[0]
    n_samples = first_sample.shape[0] * first_sample.shape[1]
    log_delta_components = []

    for j in range(fft_data.n_dim):
        weights_key = f"weights_delta_{j}"
        if weights_key in samples:
            weights_full = samples[weights_key]
            weights = weights_full[0]
            log_delta_j = batch_spline_eval(all_bases[j], weights)
            log_delta_components.append(log_delta_j)

    if log_delta_components:
        return jnp.stack(log_delta_components, axis=2)
    else:
        return jnp.zeros((n_samples, fft_data.n_freq, fft_data.n_dim))


def _reconstruct_theta_params(
    samples: Dict[str, jnp.ndarray],
    spline_model,
    fft_data: MultivarFFT,
    param_type: str,
) -> jnp.ndarray:
    """Reconstruct theta parameters from samples."""
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    key = f"weights_theta_{param_type}"
    if key in samples and spline_model.n_theta > 0:
        weights_full = samples[key]
        weights = weights_full[0]
        basis_idx = fft_data.n_dim + (0 if param_type == "re" else 1)
        theta_base = batch_spline_eval(all_bases[basis_idx], weights)
        return jnp.tile(
            theta_base[:, :, None], (1, 1, max(1, spline_model.n_theta))
        )
    else:
        first_sample = next(iter(samples.values()))
        n_samples = first_sample.shape[1]
        return jnp.zeros(
            (n_samples, fft_data.n_freq, max(1, spline_model.n_theta))
        )


def _compute_psd_diagnostics(idata, config, data, model) -> Dict[str, Any]:
    """Compute PSD diagnostics when true_psd is provided using the posterior_psd from idata."""
    diagnostics = {}

    # Save true_psd in attributes
    diagnostics["true_psd"] = np.asarray(config.true_psd)
    coverage_recorded = False
    coverage_interval = (5.0, 95.0)
    coverage_level = 0.90

    # Compute relative integrated absolute error (RIAE) and CI coverage using stored posterior_psd
    if isinstance(data, Periodogram):
        # Univariate case - use idata.posterior_psd.psd
        if "psd" in idata.posterior_psd:
            psd_quant = idata.posterior_psd["psd"]
            perc = np.asarray(psd_quant.coords["percentile"].values)
            values = psd_quant.values

            def _grab(p: float) -> np.ndarray:
                idx = int(np.argmin(np.abs(perc - p)))
                return values[idx]

            q50 = _grab(50.0)
            q05 = _grab(5.0)
            q95 = _grab(95.0)

            riae = _compute_riae(q50, config.true_psd, data.freqs)
            riae_low = _compute_riae(q05, config.true_psd, data.freqs)
            riae_high = _compute_riae(q95, config.true_psd, data.freqs)
            ci_coverage = np.mean(
                (config.true_psd >= q05) & (config.true_psd <= q95)
            )

            diagnostics["riae"] = riae
            diagnostics["ci_coverage"] = ci_coverage
            diagnostics["coverage"] = ci_coverage
            coverage_recorded = True
            # Store errorbars as list/tuple instead of dict for netCDF serialization
            diagnostics["riae_errorbars"] = [
                float(riae_low),
                float(riae_low),
                float(riae),
                float(riae_high),
                float(riae_high),
            ]

    elif isinstance(data, MultivarFFT):
        # Multivariate case - use idata.posterior_psd.psd_matrix_real
        if "psd_matrix_real" in idata.posterior_psd:
            psd_real_quant = idata.posterior_psd["psd_matrix_real"]
            perc = np.asarray(psd_real_quant.coords["percentile"].values)

            def _grab_matrix(arr: np.ndarray, p: float) -> np.ndarray:
                idx = int(np.argmin(np.abs(perc - p)))
                return arr[idx]

            psd_real_vals = psd_real_quant.values
            psd_imag_vals = (
                idata.posterior_psd["psd_matrix_imag"].values
                if "psd_matrix_imag" in idata.posterior_psd
                else np.zeros_like(psd_real_vals)
            )

            q50_real = _grab_matrix(psd_real_vals, 50.0)
            q05_real = _grab_matrix(psd_real_vals, 5.0)
            q95_real = _grab_matrix(psd_real_vals, 95.0)

            median_psd_matrix = q50_real

            # Compute matrix RIAE using Frobenius norm
            if config.true_psd is not None and config.true_psd.ndim == 3:
                # Take real part of true_psd for numerical stability
                true_psd_real = np.real(config.true_psd)
                riae_matrix = _compute_matrix_riae(
                    median_psd_matrix, true_psd_real, np.array(data.freq)
                )

                riae_low = _compute_matrix_riae(
                    q05_real, true_psd_real, np.array(data.freq)
                )
                riae_high = _compute_matrix_riae(
                    q95_real, true_psd_real, np.array(data.freq)
                )
                riae_matrix_errorbars = {
                    "q05": float(riae_low),
                    "q25": float(riae_low),
                    "median": float(riae_matrix),
                    "q75": float(riae_high),
                    "q95": float(riae_high),
                }

                # Compute CI coverage for multivariate
                ci_coverage_matrix = model.get_psd_matrix_coverage(
                    psd_real_vals, true_psd_real
                )

                diagnostics["riae_matrix"] = riae_matrix
                diagnostics["ci_coverage"] = ci_coverage_matrix
                diagnostics["coverage"] = ci_coverage_matrix
                coverage_recorded = True
                diagnostics["riae_matrix_errorbars"] = [
                    riae_matrix_errorbars["q05"],
                    riae_matrix_errorbars["q25"],
                    riae_matrix_errorbars["median"],
                    riae_matrix_errorbars["q75"],
                    riae_matrix_errorbars["q95"],
                ]

        # Do not compute per-channel RIAE for multivariate - use matrix RIAE only

    if coverage_recorded:
        diagnostics["coverage_interval"] = list(coverage_interval)
        diagnostics["coverage_level"] = coverage_level

    return diagnostics


def _compute_riae(
    median_psd: np.ndarray, true_psd: np.ndarray, freqs: np.ndarray
) -> float:
    """Compute relative integrated absolute error (RIAE)."""
    numerator = float(simpson(np.abs(median_psd - true_psd), x=freqs))
    denominator = float(simpson(true_psd, x=freqs))
    return float(numerator / denominator) if denominator != 0 else float("nan")


def _compute_matrix_riae(
    median_psd_matrix: np.ndarray,
    true_psd_matrix: np.ndarray,
    freqs: np.ndarray,
) -> float:
    """Compute RIAE for multivariate PSD matrix using the Frobenius norm."""
    diff_frobenius = np.array(
        [
            np.linalg.norm(median_psd_matrix[k] - true_psd_matrix[k], "fro")
            for k in range(len(freqs))
        ]
    )
    true_frobenius = np.array(
        [np.linalg.norm(true_psd_matrix[k], "fro") for k in range(len(freqs))]
    )
    numerator = float(simpson(diff_frobenius, x=freqs))
    denominator = float(simpson(true_frobenius, x=freqs))
    return float(numerator / denominator) if denominator != 0 else float("nan")


def _compute_riae_errorbars(
    psd_samples: np.ndarray, true_psd: np.ndarray, freqs: np.ndarray
) -> Dict[str, float]:
    """Compute errorbars for RIAE based on quantiles from posterior predictive samples."""
    riae_samples = []
    for psd in psd_samples:
        riae = _compute_riae(psd, true_psd, freqs)
        riae_samples.append(riae)
    riae_samples = np.array(riae_samples)
    return {
        "q05": float(np.percentile(riae_samples, 5)),
        "q25": float(np.percentile(riae_samples, 25)),
        "median": float(np.median(riae_samples)),
        "q75": float(np.percentile(riae_samples, 75)),
        "q95": float(np.percentile(riae_samples, 95)),
    }


def _compute_ci_coverage_univar(
    psd_samples: np.ndarray, true_psd: np.ndarray
) -> float:
    """Compute 90% credible interval coverage for univariate PSD."""
    arr = np.asarray(psd_samples)
    if arr.ndim == 2 and arr.shape[0] == 3:
        posterior_lower = arr[0]
        posterior_upper = arr[-1]
    else:
        posterior_lower = np.percentile(arr, 5.0, axis=0)
        posterior_upper = np.percentile(arr, 95.0, axis=0)
    coverage = np.mean(
        (true_psd >= posterior_lower) & (true_psd <= posterior_upper)
    )
    return float(coverage)


def _compute_ci_coverage_multivar(
    psd_matrix_samples: np.ndarray, true_psd_real: np.ndarray
) -> float:
    """Compute 90% credible interval coverage for multivariate PSD matrix."""
    true_psd = np.zeros_like(true_psd_real)
    for i in range(true_psd_real.shape[0]):
        true_psd[i] = _complex_to_real(true_psd_real[i])

    arr = np.asarray(psd_matrix_samples)
    if arr.ndim == 4 and arr.shape[0] == 3:
        posterior_lower = arr[0]
        posterior_upper = arr[-1]
    else:
        psd_matrix_real = np.zeros_like(arr, dtype=np.float64)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                psd_matrix_real[i, j] = _complex_to_real(arr[i, j])
        posterior_lower = np.percentile(psd_matrix_real, 5.0, axis=0)
        posterior_upper = np.percentile(psd_matrix_real, 95.0, axis=0)

    coverage = np.mean(
        (true_psd >= posterior_lower) & (true_psd <= posterior_upper)
    )
    return float(coverage)


# Helper to transform complex matrices to a real-valued representation for CI checks
def _complex_to_real(mat: np.ndarray) -> np.ndarray:
    """Convert complex matrix into a real-valued stacked representation."""
    return np.concatenate([np.real(mat), np.imag(mat)], axis=-1)
