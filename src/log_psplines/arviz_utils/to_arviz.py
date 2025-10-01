"""Convert inference results to ArviZ InferenceData format."""

import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, Tuple, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson
from scipy.linalg import solve_triangular
from xarray import DataArray, Dataset

from log_psplines.datatypes import MultivarFFT, Periodogram

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

    if isinstance(data, Periodogram):
        idata = _create_univar_inference_data(
            samples, sample_stats, config, data, model, attributes
        )
    elif isinstance(data, MultivarFFT):
        idata = _create_multivar_inference_data(
            samples, sample_stats, config, data, model, attributes
        )
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Compute diagnostics if true_psd is provided (using posterior_psd from idata)
    if config.true_psd is not None:
        idata.attrs.update(_compute_psd_diagnostics(idata, config, data))

    return idata


def _add_chain_dim(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add chain dimension to sample arrays (shared utility)."""
    result = {}
    for k, v in data_dict.items():
        v_array = np.array(v)
        # Always add chain dimension as first dimension if not present
        # NumPyro with 1 chain returns samples without chain dimension
        if v_array.ndim == 1:  # Scalar parameters: (n_draws,) -> (1, n_draws)
            result[k] = v_array[None, :]
        elif (
            v_array.ndim == 2
        ):  # Vector parameters: (n_draws, dim) -> (1, n_draws, dim)
            result[k] = v_array[None, :, :]
        elif (
            v_array.ndim == 3
        ):  # Matrix parameters: (n_draws, dim1, dim2) -> (1, n_draws, dim1, dim2)
            result[k] = v_array[None, :, :, :]
        elif (
            v_array.ndim == 4
        ):  # Already has chain dim: (n_chains, n_draws, dim1, dim2)
            result[k] = v_array
        else:
            # Handle higher dimensional cases
            result[k] = v_array[None, ...]
    return result


def _prepare_samples_and_stats(
    samples: Dict[str, Any], sample_stats: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare samples and sample_stats by adding chain dimensions."""
    samples = _add_chain_dim(samples)
    sample_stats = _add_chain_dim(sample_stats)
    return samples, sample_stats


def _thin_draws(
    data_dict: Dict[str, np.ndarray], indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """Thin draw dimension (axis=1) for all arrays in the dictionary."""
    thinned = {}
    for key, value in data_dict.items():
        arr = np.asarray(value)
        if arr.ndim >= 2:
            thinned[key] = arr[:, indices, ...]
        else:
            thinned[key] = arr
    return thinned


def _to_numpy_float32(array: Any) -> np.ndarray:
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.bool_):
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int32, copy=False)
    return arr.astype(np.float32, copy=False)


def _handle_log_posterior(sample_stats: Dict[str, Any]) -> None:
    """Add log posterior to sample_stats if likelihood and prior exist."""
    if {"log_likelihood", "log_prior"}.issubset(sample_stats.keys()):
        sample_stats["lp"] = (
            sample_stats["log_likelihood"] + sample_stats["log_prior"]
        )


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
        k: int(v) if isinstance(v, bool) else v
        for k, v in asdict(config).items()
    }

    true_psd_attr = config_attrs.pop("true_psd", None)
    if true_psd_attr is not None:
        # Track that a reference PSD was supplied without serializing it twice
        config_attrs["true_psd_provided"] = 1

    attributes.update(config_attrs)

    # Add ESS calculation
    try:
        attributes.update(
            dict(ess=az.ess(samples).to_array().values.flatten())
        )
    except:
        attributes.update(dict(ess=[]))

    # Add base coordinates that both functions use
    coords.update(
        {
            "chain": range(samples[list(samples.keys())[0]].shape[0]),
            "draw": range(samples[list(samples.keys())[0]].shape[1]),
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
    samples, sample_stats = _prepare_samples_and_stats(samples, sample_stats)

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
    pp_samples = np.array(
        [spline_model(w) for w in weights_chain0[pp_idx]], dtype=np.float64
    )
    pp_samples = np.exp(pp_samples)

    # Ensure arrays are numpy before rescaling
    pp_samples = np.array(pp_samples, dtype=np.float64)
    observed_power = np.array(periodogram.power, dtype=np.float64)

    pp_samples_rescaled = pp_samples * config.scaling_factor
    observed_power_rescaled = observed_power * config.scaling_factor
    # Handle log posterior
    _handle_log_posterior(sample_stats)

    # Drop heavy deterministic arrays that have dedicated outputs
    for heavy_key in ["log_delta_sq", "theta_re", "theta_im"]:
        sample_stats.pop(heavy_key, None)

    # Setup coordinates and dimensions
    coords = {
        "pp_draw": range(n_pp),
        "weight_dim": range(n_weights),
        "freq": periodogram.freqs,
    }

    dims = {
        "phi": ["chain", "draw"],
        "delta": ["chain", "draw"],
        "weights": ["chain", "draw", "weight_dim"],
        **{k: ["chain", "draw"] for k in sample_stats.keys()},
        "periodogram": ["freq"],  # Observed data
        "psd": ["chain", "pp_draw", "freq"],  # Posterior predictive
    }

    # Prepare attributes and basic coords/dims using helper
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

    # Create base InferenceData
    idata = az.from_dict(
        posterior=samples,
        sample_stats=sample_stats,
        observed_data={"periodogram": observed_power_rescaled},
        dims={
            k: v for k, v in dims.items() if k not in ["psd", "lp"]
        },  # Exclude manually added groups
        coords=coords,
        attrs=attributes,
    )

    # Add posterior predictive samples
    idata.add_groups(
        posterior_psd=Dataset(
            {"psd": DataArray(pp_samples_rescaled, dims=["pp_draw", "freq"])},
            coords={"pp_draw": coords["pp_draw"], "freq": coords["freq"]},
        )
    )
    # Add spline model info
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
    samples, sample_stats = _prepare_samples_and_stats(samples, sample_stats)

    # Extract dimensions from a standard sample
    first_sample_key = next(
        (k for k, v in samples.items() if "weights_" in k),
        next(iter(samples.keys())),
    )
    sample_shape = samples[first_sample_key].shape
    n_chains, n_draws = sample_shape[:2]

    # Optionally thin draws before heavy processing
    n_total_draws = sample_shape[1]
    max_draws_retained = 400
    if n_total_draws > max_draws_retained:
        store_idx = np.linspace(
            0, n_total_draws - 1, max_draws_retained, dtype=int
        )
        if config.verbose:
            print(
                f"Thinning posterior draws from {n_total_draws} to {len(store_idx)} before ArviZ conversion"
            )
        samples = _thin_draws(samples, store_idx)
        sample_stats = _thin_draws(sample_stats, store_idx)
        first_sample_key = next(
            (k for k, v in samples.items() if "weights_" in k),
            next(iter(samples.keys())),
        )
        sample_shape = samples[first_sample_key].shape
        n_chains, n_draws = sample_shape[:2]
    else:
        store_idx = None

    # Convert to numpy float32/int32 to reduce memory footprint
    samples = {k: _to_numpy_float32(v) for k, v in samples.items()}
    sample_stats = {k: _to_numpy_float32(v) for k, v in sample_stats.items()}

    # Create posterior predictive samples
    if config.verbose:
        print("Reconstructing multivariate posterior PSD samples (subsampled)")
        t_reconstruct_start = time.perf_counter()

    psd_samples = _compute_posterior_predictive_multivar(
        samples, spline_model, fft_data, verbose=config.verbose
    )
    if config.verbose:
        print(
            f"  PSD reconstruction completed in {time.perf_counter() - t_reconstruct_start:.2f}s"
        )
    n_pp = psd_samples.shape[0]

    # Ensure arrays are numpy before rescaling
    psd_samples = np.array(psd_samples)
    fft_y_re = np.array(fft_data.y_re)
    fft_y_im = np.array(fft_data.y_im)

    # Rescale each cross-spectral component
    if config.verbose:
        t_rescale_start = time.perf_counter()
    psd_samples_rescaled = psd_samples * config.scaling_factor
    # Also rescale observed FFT data
    observed_fft_re_rescaled = fft_y_re * np.sqrt(config.scaling_factor)
    observed_fft_im_rescaled = fft_y_im * np.sqrt(config.scaling_factor)
    # Compute and rescale observed cross-spectral density (periodogram)
    y_re = observed_fft_re_rescaled
    y_im = observed_fft_im_rescaled
    n_freq, n_dim = y_re.shape
    observed_csd = np.zeros((n_freq, n_dim, n_dim), dtype=np.complex64)
    for i in range(n_dim):
        for j in range(n_dim):
            observed_csd[:, i, j] = (y_re[:, i] + 1j * y_im[:, i]) * np.conj(
                y_re[:, j] + 1j * y_im[:, j]
            )
            observed_csd[:, i, j] *= config.scaling_factor
    observed_csd = np.real(observed_csd)
    if config.verbose:
        print(
            f"  Rescaling completed in {time.perf_counter() - t_rescale_start:.2f}s"
        )
    if config.verbose:
        print(
            f"Rescaling multivariate posterior samples: max scaling ~{config.scaling_factor:.2e}"
        )
    psd_samples = psd_samples_rescaled
    observed_fft_data_rescaled = {
        "fft_re": observed_fft_re_rescaled,
        "fft_im": observed_fft_im_rescaled,
    }
    observed_psd_rescaled = {"periodogram": observed_csd}

    # Handle log posterior
    _handle_log_posterior(sample_stats)

    # Setup coordinates and dimensions
    coords = {
        "pp_draw": range(n_pp),
        "freq": np.array(fft_data.freq),
        "channels": range(fft_data.n_dim),
    }

    dims = {}

    if config.verbose:
        t_dims_start = time.perf_counter()

    # Posterior samples - handle weights with proper dimensions
    for key, array in samples.items():
        array_shape = array.shape
        if key.startswith("weights_"):
            component = key[8:]  # Remove "weights_" prefix
            dims[key] = ["chain", "draw", f"{component}_basis_dim"]
            coords[f"{component}_basis_dim"] = range(array_shape[-1])
        elif key in ["phi", "delta"] or key.startswith(("phi_", "delta_")):
            dims[key] = ["chain", "draw"]

    # Sample stats - handle multivariate-specific variables
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
        else:  # Generic fallback
            dims[key] = ["chain", "draw"] + [
                f"{key}_dim_{i}" for i in range(len(array_shape) - 2)
            ]
            for i in range(2, len(array_shape)):
                coords[f"{key}_dim_{i-2}"] = range(array_shape[i])

    if config.verbose:
        print(
            f"  Prepared dims/coords in {time.perf_counter() - t_dims_start:.2f}s"
        )

    # Observed data and posterior predictive
    dims.update(
        {
            "fft_re": ["freq", "channels"],
            "fft_im": ["freq", "channels"],
            "psd_matrix": ["pp_draw", "freq", "channels", "channels"],
        }
    )
    dims.update(
        {
            "fft_re": ["freq", "channels"],
            "fft_im": ["freq", "channels"],
            "psd_matrix": ["pp_draw", "freq", "channels", "channels2"],
        }
    )

    # Update attributes with multivar-specific values
    attributes.update(
        {
            "data_type": "multivariate",
            "n_channels": fft_data.n_dim,
            "n_freq": fft_data.n_freq,
            "n_theta": spline_model.n_theta,
            "frequencies": np.array(fft_data.freq),
        }
    )

    # Use shared attribute and dimension preparation
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

    # Create base InferenceData
    if config.verbose:
        t_from_dict_start = time.perf_counter()

    idata = az.from_dict(
        posterior=samples,
        sample_stats=sample_stats,
        observed_data={
            "fft_re": np.array(observed_fft_data_rescaled["fft_re"]),
            "fft_im": np.array(observed_fft_data_rescaled["fft_im"]),
        },
        dims={
            k: v for k, v in dims.items() if k not in ["psd_matrix", "lp"]
        },  # Exclude manually added groups
        coords=coords,
        attrs=attributes,
    )
    if config.verbose:
        print(
            f"  ArviZ from_dict completed in {time.perf_counter() - t_from_dict_start:.2f}s"
        )

    # Add posterior predictive samples
    if config.verbose:
        t_add_pp_start = time.perf_counter()
    idata.add_groups(
        posterior_psd=Dataset(
            {
                "psd_matrix": DataArray(
                    psd_samples,
                    dims=["pp_draw", "freq", "channels", "channels2"],
                )
            },
            coords={
                "pp_draw": coords["pp_draw"],
                "freq": coords["freq"],
                "channels": coords["channels"],
                "channels2": coords["channels"],
            },
        )
    )
    if config.verbose:
        print(
            f"  Added posterior_psd group in {time.perf_counter() - t_add_pp_start:.2f}s"
        )

    # Add spline model info
    if config.verbose:
        t_add_model_start = time.perf_counter()
    idata.add_groups(spline_model=_pack_spline_model_multivar(spline_model))
    if config.verbose:
        print(
            f"  Added spline_model group in {time.perf_counter() - t_add_model_start:.2f}s"
        )
    return idata


@jax.jit
def batch_spline_eval(
    basis: jnp.ndarray, weights_batch: jnp.ndarray
) -> jnp.ndarray:
    """JIT-compiled batch spline evaluation over multiple weight vectors.

    Args:
        basis: Basis matrix (n_freq, n_basis)
        weights_batch: Batch of weight vectors (n_samples, n_basis)

    Returns:
        Batch of spline evaluations (n_samples, n_freq)
    """
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

    # Pack diagonal models
    for i, diag_model in enumerate(spline_model.diagonal_models):
        _pack_model_component(diag_model, f"diag_{i}", data, coords)

    # Pack off-diagonal models if they exist
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
    spline_model,
    fft_data: MultivarFFT,
    verbose: bool = False,
) -> np.ndarray:
    """Compute posterior predictive PSD matrices from samples."""
    # Extract samples - handle different possible sample structures
    if "log_delta_sq" in samples:
        log_delta_sq = samples["log_delta_sq"]
        # Remove chain dimension if present (take first chain only for posterior predictive)
        if log_delta_sq.ndim == 4:  # (n_chains, n_draws, n_freq, n_channels)
            log_delta_sq = log_delta_sq[
                0
            ]  # Take first chain: (n_draws, n_freq, n_channels)
        theta_re = samples.get("theta_re")
        theta_im = samples.get("theta_im")
        # Default to zeros if not present
        if theta_re is None:
            theta_re = jnp.zeros((log_delta_sq.shape[0], fft_data.n_freq, 0))
        else:
            if theta_re.ndim == 4:
                theta_re = theta_re[0]
        if theta_im is None:
            theta_im = jnp.zeros((log_delta_sq.shape[0], fft_data.n_freq, 0))
        else:
            if theta_im.ndim == 4:
                theta_im = theta_im[0]
        n_draws = log_delta_sq.shape[0]
    else:
        # Reconstruct from individual component samples
        first_sample = next(iter(samples.values()))
        n_draws = first_sample.shape[1]
        log_delta_sq = None
        theta_re = None
        theta_im = None

    # Subsample draws for posterior predictive to limit cost
    n_pp = min(25, n_draws)
    if n_draws > n_pp:
        pp_idx = np.random.choice(n_draws, n_pp, replace=False)
    else:
        pp_idx = np.arange(n_draws)

    if verbose:
        print(
            f"  Subsampling {len(pp_idx)} draws (from {n_draws}) for PSD reconstruction"
        )

    if log_delta_sq is None:
        if verbose:
            print("  Reconstructing log_delta_sq from weights")
        log_delta_sq = _reconstruct_log_delta_sq(
            samples, spline_model, fft_data, draw_indices=pp_idx
        )
    else:
        log_delta_sq = log_delta_sq[pp_idx]

    if theta_re is None:
        if verbose:
            print("  Reconstructing theta_re from weights")
        theta_re = _reconstruct_theta_params(
            samples, spline_model, fft_data, "re", draw_indices=pp_idx
        )
    else:
        theta_re = theta_re[pp_idx]

    if theta_im is None:
        if verbose:
            print("  Reconstructing theta_im from weights")
        theta_im = _reconstruct_theta_params(
            samples, spline_model, fft_data, "im", draw_indices=pp_idx
        )
    else:
        theta_im = theta_im[pp_idx]

    log_delta_sq = np.asarray(log_delta_sq)
    theta_re = np.asarray(theta_re)
    theta_im = np.asarray(theta_im)

    if verbose:
        print("  Building PSD matrices via numpy")
    return _reconstruct_psd_numpy(log_delta_sq, theta_re, theta_im)


def _reconstruct_log_delta_sq(
    samples: Dict[str, jnp.ndarray],
    spline_model,
    fft_data: MultivarFFT,
    draw_indices: np.ndarray | None,
) -> jnp.ndarray:
    """Reconstruct log_delta_sq from individual diagonal component samples."""
    # Get all bases once
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    first_sample = next(iter(samples.values()))
    total_draws = first_sample.shape[1]
    if draw_indices is None:
        draw_indices = np.arange(total_draws)
    n_samples = len(draw_indices)
    log_delta_components = []

    for j in range(fft_data.n_dim):
        weights_key = f"weights_delta_{j}"
        if weights_key in samples:
            weights_full = samples[
                weights_key
            ]  # Shape: (n_chains, n_draws, n_weights)
            # For posterior predictive, use first chain only
            weights = weights_full[0]  # Shape: (n_draws, n_weights)
            weights = weights[draw_indices]
            # Vectorized spline evaluation using JAX
            log_delta_j = batch_spline_eval(all_bases[j], weights)
            log_delta_components.append(log_delta_j)

    if log_delta_components:
        return jnp.stack(
            log_delta_components, axis=2
        )  # (n_samples, n_freq, n_channels)
    else:
        # Fallback
        return jnp.zeros((n_samples, fft_data.n_freq, fft_data.n_dim))


def _reconstruct_theta_params(
    samples: Dict[str, jnp.ndarray],
    spline_model,
    fft_data: MultivarFFT,
    param_type: str,
    draw_indices: np.ndarray | None,
) -> jnp.ndarray:
    """Reconstruct theta parameters from samples."""
    # Get all bases once
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    key = f"weights_theta_{param_type}"
    if key in samples and spline_model.n_theta > 0:
        weights_full = samples[key]  # Shape: (n_chains, n_draws, n_weights)
        # For posterior predictive, use first chain only
        weights = weights_full[0]  # Shape: (n_draws, n_weights)
        if draw_indices is not None and len(draw_indices) < weights.shape[0]:
            weights = weights[draw_indices]
        basis_idx = fft_data.n_dim + (0 if param_type == "re" else 1)
        # Vectorized spline evaluation using JAX
        theta_base = batch_spline_eval(all_bases[basis_idx], weights)
        # Tile to match expected shape
        return jnp.tile(
            theta_base[:, :, None], (1, 1, max(1, spline_model.n_theta))
        )
    else:
        first_sample = next(iter(samples.values()))
        n_samples = (
            len(draw_indices)
            if draw_indices is not None
            else first_sample.shape[1]
        )
        return jnp.zeros(
            (n_samples, fft_data.n_freq, max(1, spline_model.n_theta))
        )


def _reconstruct_psd_numpy(
    log_delta_sq: np.ndarray,
    theta_re: np.ndarray,
    theta_im: np.ndarray,
) -> np.ndarray:
    n_samples, n_freq, n_channels = log_delta_sq.shape
    n_theta = theta_re.shape[2] if theta_re.ndim > 2 else 0

    psd = np.zeros(
        (n_samples, n_freq, n_channels, n_channels), dtype=np.float64
    )

    eye_c = np.eye(n_channels, dtype=np.complex128)
    tril_row, tril_col = np.tril_indices(n_channels, k=-1)
    n_lower = len(tril_row)

    for s in range(n_samples):
        for f in range(n_freq):
            diag_vals = np.exp(log_delta_sq[s, f])
            T = eye_c.copy()
            if n_theta > 0 and n_channels > 1:
                theta_complex = theta_re[s, f] + 1j * theta_im[s, f]
                n_use = min(theta_complex.shape[0], n_lower)
                if n_use > 0:
                    T[tril_row[:n_use], tril_col[:n_use]] -= theta_complex[
                        :n_use
                    ]

            temp = solve_triangular(T, np.diag(diag_vals), lower=True)
            T_inv = solve_triangular(T, eye_c, lower=True)
            psd_mat = temp @ T_inv.conj().T
            psd[s, f] = psd_mat.real / (2 * np.pi)

    return psd


def _compute_psd_diagnostics(idata, config, data) -> Dict[str, Any]:
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
            psd_samples = idata.posterior_psd[
                "psd"
            ].values  # Shape: (pp_draw, freq)
            median_psd = np.median(psd_samples, axis=0)

            riae = _compute_riae(median_psd, config.true_psd, data.freqs)
            riae_errorbars = _compute_riae_errorbars(
                psd_samples, config.true_psd, data.freqs
            )

            # Compute CI coverage
            ci_coverage = _compute_ci_coverage_univar(
                psd_samples, config.true_psd
            )

            diagnostics["riae"] = riae
            diagnostics["ci_coverage"] = ci_coverage
            diagnostics["coverage"] = ci_coverage
            coverage_recorded = True
            # Store errorbars as list/tuple instead of dict for netCDF serialization
            diagnostics["riae_errorbars"] = [
                riae_errorbars["q05"],
                riae_errorbars["q25"],
                riae_errorbars["median"],
                riae_errorbars["q75"],
                riae_errorbars["q95"],
            ]

    elif isinstance(data, MultivarFFT):
        # Multivariate case - use idata.posterior_psd.psd_matrix
        if "psd_matrix" in idata.posterior_psd:
            psd_matrix_samples = idata.posterior_psd[
                "psd_matrix"
            ].values  # Shape: (pp_draw, freq, channels, channels_out)
            median_psd_matrix = np.median(psd_matrix_samples, axis=0)

            # Compute matrix RIAE using Frobenius norm
            if config.true_psd is not None and config.true_psd.ndim == 3:
                # Take real part of true_psd for numerical stability
                true_psd_real = np.real(config.true_psd)
                riae_matrix = _compute_matrix_riae(
                    median_psd_matrix, true_psd_real, np.array(data.freq)
                )

                # Compute matrix RIAE errorbars from samples
                matrix_riae_samples = []
                for psd_matrix in psd_matrix_samples:
                    matrix_riae = _compute_matrix_riae(
                        psd_matrix, true_psd_real, np.array(data.freq)
                    )
                    matrix_riae_samples.append(matrix_riae)
                matrix_riae_samples = np.array(matrix_riae_samples)
                riae_matrix_errorbars = {
                    "q05": float(np.percentile(matrix_riae_samples, 5)),
                    "q25": float(np.percentile(matrix_riae_samples, 25)),
                    "median": float(np.median(matrix_riae_samples)),
                    "q75": float(np.percentile(matrix_riae_samples, 75)),
                    "q95": float(np.percentile(matrix_riae_samples, 95)),
                }

                # Compute CI coverage for multivariate
                ci_coverage_matrix = _compute_ci_coverage_multivar(
                    psd_matrix_samples, true_psd_real
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
    """Compute RIAE for multivariate PSD matrix using the Frobenius norm.

    The diagnostic integrates the relative deviation of the estimated PSD matrix
    from the reference matrix using

    .. math::

        \operatorname{RIAE} = \frac{\int \! \lVert \widehat{S}(f) - S(f) \rVert_F \, \mathrm{d}f}{\int \! \lVert S(f) \rVert_F \, \mathrm{d}f},

    where :math:`\lVert \cdot \rVert_F` is the matrix Frobenius norm and the
    integrals are approximated numerically via Simpson's rule over the frequency
    grid.
    """
    # Compute Frobenius norm for each frequency

    diff_frobenius = np.array(
        [
            np.linalg.norm(median_psd_matrix[k] - true_psd_matrix[k], "fro")
            for k in range(len(freqs))
        ]
    )
    true_frobenius = np.array(
        [np.linalg.norm(true_psd_matrix[k], "fro") for k in range(len(freqs))]
    )

    # Integrate using trapezoidal rule (or sum for uniform freq spacing)
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
    """Compute 95% credible interval coverage for univariate PSD."""
    # psd_samples: (pp_draw, freq)
    posterior_lower = np.percentile(psd_samples, 2.5, axis=0)  # (freq,)
    posterior_upper = np.percentile(psd_samples, 97.5, axis=0)  # (freq,)
    coverage = np.mean(
        (true_psd >= posterior_lower) & (true_psd <= posterior_upper)
    )
    return float(coverage)


def _complex_to_real(matrix):
    """
    Transform a complex valued Hermitian matrix to a real valued matrix.

    Upper triangular elements (including diagonal) contain the real parts,
    lower triangular elements (excluding diagonal) contain the imaginary parts.
    """
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(
        matrix[np.tril_indices(n, -1)]
    )
    return real_matrix


def _compute_ci_coverage_multivar(
    psd_matrix_samples: np.ndarray, true_psd_real: np.ndarray
) -> float:
    """Compute 95% credible interval coverage for multivariate PSD matrix."""
    # psd_matrix_samples: (pp_draw, freq, channels, channels_out)

    # Transform true_psd to real-valued representation
    true_psd = np.zeros_like(true_psd_real)
    for i in range(true_psd_real.shape[0]):
        true_psd[i] = _complex_to_real(true_psd_real[i])

    psd_matrix_real = np.zeros_like(psd_matrix_samples, dtype=np.float64)

    # transform each sample to real-valued representation
    for i in range(psd_matrix_samples.shape[0]):
        for j in range(psd_matrix_samples.shape[1]):
            psd_matrix_real[i, j] = _complex_to_real(psd_matrix_samples[i, j])

    # Compute 95% CI for each element across pp_draw
    posterior_lower = np.percentile(
        psd_matrix_real, 2.5, axis=0
    )  # (freq, channels, channels_out)
    posterior_upper = np.percentile(
        psd_matrix_real, 97.5, axis=0
    )  # (freq, channels, channels_out)
    # Coverage is fraction of matrix elements where true_psd is within CI
    coverage = np.mean(
        (true_psd_real >= posterior_lower) & (true_psd_real <= posterior_upper)
    )
    return float(coverage)
