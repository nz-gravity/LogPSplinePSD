"""Convert inference results to ArviZ InferenceData format."""

import warnings
from dataclasses import asdict
from typing import Any, Dict, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
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
        return _create_univar_inference_data(samples, sample_stats, config, data, model, attributes)
    elif isinstance(data, MultivarFFT):
        return _create_multivar_inference_data(samples, sample_stats, config, data, model, attributes)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def _pack_spline_model(spline_model) -> Dataset:
    """Pack spline model parameters into xarray Dataset."""
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
        config: "BaseSampler",
        periodogram: "Periodogram",
        spline_model: "LogPSplines",
        attributes: Dict[str, Any],
    ) -> az.InferenceData:
    """Create InferenceData for univariate case."""
    # Ensure all arrays have chain dimension
    def add_chain_dim(data_dict):
        return {
            k: np.array(v)[None, ...] if np.array(v).ndim <= 2 else np.array(v)
            for k, v in data_dict.items()
        }

    samples = add_chain_dim(samples)
    sample_stats = add_chain_dim(sample_stats)

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

    pp_samples = np.array([spline_model(w) for w in weights_chain0[pp_idx]])
    pp_samples = np.exp(pp_samples)

    # Coordinates
    coords = {
        "chain": range(n_chains),
        "draw": range(n_draws),
        "pp_draw": range(n_pp),
        "weight_dim": range(n_weights),
        "freq": periodogram.freqs,
    }

    # Dimensions for each data group
    dims = {
        # Posterior
        "phi": ["chain", "draw"],
        "delta": ["chain", "draw"],
        "weights": ["chain", "draw", "weight_dim"],
        # Sample stats
        **{k: ["chain", "draw"] for k in sample_stats.keys()},
        # Observed data
        "periodogram": ["freq"],
        # Posterior predictive
        "psd": ["chain", "pp_draw", "freq"],
    }

    # Add log posterior if both likelihood and prior exist
    if {"log_likelihood", "log_prior"}.issubset(sample_stats.keys()):
        sample_stats["lp"] = (
            sample_stats["log_likelihood"] + sample_stats["log_prior"]
        )

    # Convert config to attributes (handle booleans)
    config_attrs = {
        k: int(v) if isinstance(v, bool) else v
        for k, v in asdict(config).items()
    }
    attributes.update(config_attrs)
    attributes.update(dict(ess=az.ess(samples).to_array().values.flatten()))

    # Create InferenceData with custom posterior_psd group
    idata = az.from_dict(
        posterior=samples,
        sample_stats=sample_stats,
        observed_data={"periodogram": periodogram.power},
        dims={k: v for k, v in dims.items() if k != "psd"},
        coords={k: v for k, v in coords.items() if k != "pp_draw"},
        attrs=attributes,
    )

    # Add posterior predictive samples
    idata.add_groups(
        posterior_psd=Dataset(
            {
                "psd": DataArray(pp_samples, dims=["pp_draw", "freq"]),
            },
            coords={
                "pp_draw": coords["pp_draw"],
                "freq": coords["freq"],
            },
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

    # Ensure all arrays have chain dimension - FIXED VERSION
    def add_chain_dim(data_dict):
        result = {}
        for k, v in data_dict.items():
            v_array = np.array(v)
            # Always add chain dimension as first dimension if not present
            # NumPyro with 1 chain returns samples without chain dimension
            if v_array.ndim == 1:  # Scalar parameters: (n_draws,) -> (1, n_draws)
                result[k] = v_array[None, :]
            elif v_array.ndim == 2:  # Vector parameters: (n_draws, dim) -> (1, n_draws, dim)
                result[k] = v_array[None, :, :]
            elif v_array.ndim == 3:  # Matrix parameters: (n_draws, dim1, dim2) -> (1, n_draws, dim1, dim2)
                result[k] = v_array[None, :, :, :]
            elif v_array.ndim == 4:  # Already has chain dim: (n_chains, n_draws, dim1, dim2)
                result[k] = v_array
            else:
                # Handle higher dimensional cases
                result[k] = v_array[None, ...]
        return result

    samples = add_chain_dim(samples)
    sample_stats = add_chain_dim(sample_stats)

    # Extract dimensions from a standard sample
    first_sample_key = next((k for k, v in samples.items() if "weights_" in k), next(iter(samples.keys())))
    sample_shape = samples[first_sample_key].shape
    n_chains, n_draws = sample_shape[:2]

    # Create posterior predictive samples
    psd_samples = _compute_posterior_predictive_multivar(samples, spline_model, fft_data)
    n_pp = psd_samples.shape[0]

    # Coordinates
    coords = {
        "chain": range(n_chains),
        "draw": range(n_draws),
        "pp_draw": range(n_pp),
        "freq": np.array(fft_data.freq),
        "channels": range(fft_data.n_dim),
    }

    # Dimensions for each data group
    dims = {}

    # Posterior samples - handle weights with proper dimensions
    for key, array in samples.items():
        array_shape = array.shape
        if key.startswith("weights_"):
            # Extract the component type from the key (e.g., "delta_0", "theta_re")
            component = key[8:]  # Remove "weights_" prefix
            dims[key] = ["chain", "draw", f"{component}_basis_dim"]
            coords[f"{component}_basis_dim"] = range(array_shape[-1])
        elif key in ["phi", "delta"] or key.startswith(("phi_", "delta_")):
            # Scalar hyperparameters
            dims[key] = ["chain", "draw"]

    # Sample stats - handle multivariate-specific variables properly
    for key, array in sample_stats.items():
        array_shape = array.shape

        if key == "log_delta_sq":
            # Should now have shape: (1, n_draws, n_freq, n_channels)
            if len(array_shape) == 4:
                dims[key] = ["chain", "draw", "freq", "channels"]
            else:
                # Fallback for unexpected shapes
                dims[key] = ["chain", "draw"] + [f"{key}_dim_{i}" for i in range(len(array_shape)-2)]
                for i in range(2, len(array_shape)):
                    coords[f"{key}_dim_{i-2}"] = range(array_shape[i])

        elif key in ["theta_re", "theta_im"]:
            # Should have shape: (1, n_draws, n_freq, n_theta)
            if len(array_shape) == 4:
                dims[key] = ["chain", "draw", "freq", f"{key}_theta_dim"]
                coords[f"{key}_theta_dim"] = range(array_shape[-1])
            else:
                # Fallback
                dims[key] = ["chain", "draw"] + [f"{key}_dim_{i}" for i in range(len(array_shape)-2)]
                for i in range(2, len(array_shape)):
                    coords[f"{key}_dim_{i-2}"] = range(array_shape[i])

        elif key == "log_likelihood":
            # Scalar likelihood: should be (1, n_draws)
            dims[key] = ["chain", "draw"]
        else:
            # Generic handling for other sample stats
            nd = len(array_shape)
            if nd == 2:
                dims[key] = ["chain", "draw"]
            else:
                dims[key] = ["chain", "draw"] + [f"{key}_dim_{i}" for i in range(nd-2)]
                for i in range(2, nd):
                    coords[f"{key}_dim_{i-2}"] = range(array_shape[i])

    # Observed data
    dims.update({
        "fft_re": ["freq", "channels"],
        "fft_im": ["freq", "channels"],
        # Posterior predictive
        "psd_matrix": ["pp_draw", "freq", "channels", "channels"],
    })

    # Add log posterior if both likelihood and prior exist
    if {"log_likelihood", "log_prior"}.issubset(sample_stats.keys()):
        sample_stats["lp"] = (
                sample_stats["log_likelihood"] + sample_stats["log_prior"]
        )
        dims["lp"] = ["chain", "draw"]

    # Extract values from attributes dict and update with multivar-specific attributes
    attributes.update({
        "data_type": "multivariate",
        "n_channels": fft_data.n_dim,
        "n_freq": fft_data.n_freq,
        "n_theta": spline_model.n_theta,
        "frequencies": np.array(fft_data.freq),
    })

    # Convert config to attributes (handle booleans)
    config_attrs = {
        k: int(v) if isinstance(v, bool) else v
        for k, v in asdict(config).items()
    }
    attributes.update(config_attrs)

    # Add ESS calculation
    try:
        attributes.update(dict(ess=az.ess(samples).to_array().values.flatten()))
    except:
        attributes.update(dict(ess=[]))

    # Create InferenceData
    idata = az.from_dict(
        posterior=samples,
        sample_stats=sample_stats,
        observed_data={
            "fft_re": np.array(fft_data.y_re),
            "fft_im": np.array(fft_data.y_im)
        },
        dims=dims,
        coords=coords,
        attrs=attributes,
    )

    # Add posterior predictive samples
    idata.add_groups(
        posterior_psd=Dataset(
            {
                "psd_matrix": DataArray(psd_samples, dims=["pp_draw", "freq", "channels", "channels"]),
            },
            coords={
                "pp_draw": coords["pp_draw"],
                "freq": coords["freq"],
                "channels": coords["channels"],
            },
        )
    )

    # Add spline model info
    idata.add_groups(spline_model=_pack_spline_model_multivar(spline_model))
    return idata


@jax.jit
def batch_spline_eval(basis: jnp.ndarray, weights_batch: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled batch spline evaluation over multiple weight vectors.

    Args:
        basis: Basis matrix (n_freq, n_basis)
        weights_batch: Batch of weight vectors (n_samples, n_basis)

    Returns:
        Batch of spline evaluations (n_samples, n_freq)
    """
    return jnp.sum(basis[None, :, :] * weights_batch[:, None, :], axis=-1)


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

    # Diagonal models
    for i, diag_model in enumerate(spline_model.diagonal_models):
        prefix = f"diag_{i}"
        data.update({
            f"{prefix}_knots": ([f"{prefix}_knots_dim"], np.array(diag_model.knots)),
            f"{prefix}_basis": ([f"{prefix}_freq", f"{prefix}_weights_dim"], np.array(diag_model.basis)),
            f"{prefix}_penalty_matrix": ([f"{prefix}_weights_dim_row", f"{prefix}_weights_dim_col"],
                                         np.array(diag_model.penalty_matrix)),
            f"{prefix}_parametric_model": ([f"{prefix}_freq"], np.array(diag_model.parametric_model)),
        })
        coords.update({
            f"{prefix}_knots_dim": np.arange(len(diag_model.knots)),
            f"{prefix}_weights_dim": np.arange(diag_model.basis.shape[1]),
            f"{prefix}_weights_dim_row": np.arange(diag_model.penalty_matrix.shape[0]),
            f"{prefix}_weights_dim_col": np.arange(diag_model.penalty_matrix.shape[1]),
            f"{prefix}_freq": np.arange(diag_model.basis.shape[0]),
        })

    # Off-diagonal models
    if spline_model.offdiag_re_model is not None:
        data.update({
            "offdiag_re_knots": (["offdiag_knots_dim"], np.array(spline_model.offdiag_re_model.knots)),
            "offdiag_re_basis": (["offdiag_freq", "offdiag_weights_dim"],
                                 np.array(spline_model.offdiag_re_model.basis)),
            "offdiag_re_penalty_matrix": (["offdiag_weights_dim_row", "offdiag_weights_dim_col"],
                                          np.array(spline_model.offdiag_re_model.penalty_matrix)),
            "offdiag_re_parametric_model": (["offdiag_freq"],
                                            np.array(spline_model.offdiag_re_model.parametric_model)),
        })
        coords.update({
            "offdiag_knots_dim": np.arange(len(spline_model.offdiag_re_model.knots)),
            "offdiag_weights_dim": np.arange(spline_model.offdiag_re_model.basis.shape[1]),
            "offdiag_weights_dim_row": np.arange(spline_model.offdiag_re_model.penalty_matrix.shape[0]),
            "offdiag_weights_dim_col": np.arange(spline_model.offdiag_re_model.penalty_matrix.shape[1]),
            "offdiag_freq": np.arange(spline_model.offdiag_re_model.basis.shape[0]),
        })

    if spline_model.offdiag_im_model is not None:
        data.update({
            "offdiag_im_knots": (["offdiag_knots_dim"], np.array(spline_model.offdiag_im_model.knots)),
            "offdiag_im_basis": (["offdiag_freq", "offdiag_weights_dim"],
                                 np.array(spline_model.offdiag_im_model.basis)),
            "offdiag_im_penalty_matrix": (["offdiag_weights_dim_row", "offdiag_weights_dim_col"],
                                          np.array(spline_model.offdiag_im_model.penalty_matrix)),
            "offdiag_im_parametric_model": (["offdiag_freq"],
                                            np.array(spline_model.offdiag_im_model.parametric_model)),
        })
        coords.update({
            "offdiag_knots_dim": np.arange(len(spline_model.offdiag_im_model.knots)),
            "offdiag_weights_dim": np.arange(spline_model.offdiag_im_model.basis.shape[1]),
            "offdiag_weights_dim_row": np.arange(spline_model.offdiag_im_model.penalty_matrix.shape[0]),
            "offdiag_weights_dim_col": np.arange(spline_model.offdiag_im_model.penalty_matrix.shape[1]),
            "offdiag_freq": np.arange(spline_model.offdiag_im_model.basis.shape[0]),
        })

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


def _compute_posterior_predictive_multivar(samples: Dict[str, jnp.ndarray], spline_model, fft_data: MultivarFFT) -> jnp.ndarray:
    """Compute posterior predictive PSD matrices from samples."""
    # Extract samples - handle different possible sample structures
    if "log_delta_sq" in samples:
        log_delta_sq = samples["log_delta_sq"]
        # Remove chain dimension if present (take first chain only for posterior predictive)
        if log_delta_sq.ndim == 4:  # (n_chains, n_draws, n_freq, n_channels)
            log_delta_sq = log_delta_sq[0]  # Take first chain: (n_draws, n_freq, n_channels)
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
    else:
        # Reconstruct from individual component samples
        log_delta_sq = _reconstruct_log_delta_sq(samples, spline_model, fft_data)
        theta_re = _reconstruct_theta_params(samples, spline_model, fft_data, "re")
        theta_im = _reconstruct_theta_params(samples, spline_model, fft_data, "im")

    # Use spline model's reconstruction method
    return spline_model.reconstruct_psd_matrix(
        log_delta_sq, theta_re, theta_im, n_samples_max=50
    )


def _reconstruct_log_delta_sq(samples: Dict[str, jnp.ndarray], spline_model, fft_data: MultivarFFT) -> jnp.ndarray:
    """Reconstruct log_delta_sq from individual diagonal component samples."""
    # Get all bases once
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    first_sample = next(iter(samples.values()))
    n_chains = first_sample.shape[0]  # Assume chain dim added
    n_samples = first_sample.shape[0] * first_sample.shape[1]  # Total draws across chains
    log_delta_components = []

    for j in range(fft_data.n_dim):
        weights_key = f"weights_delta_{j}"
        if weights_key in samples:
            weights_full = samples[weights_key]  # Shape: (n_chains, n_draws, n_weights)
            # For posterior predictive, use first chain only
            weights = weights_full[0]  # Shape: (n_draws, n_weights)
            # Vectorized spline evaluation using JAX
            log_delta_j = batch_spline_eval(all_bases[j], weights)
            log_delta_components.append(log_delta_j)

    if log_delta_components:
        return jnp.stack(log_delta_components, axis=2)  # (n_samples, n_freq, n_channels)
    else:
        # Fallback
        return jnp.zeros((n_samples, fft_data.n_freq, fft_data.n_dim))


def _reconstruct_theta_params(samples: Dict[str, jnp.ndarray], spline_model, fft_data: MultivarFFT, param_type: str) -> jnp.ndarray:
    """Reconstruct theta parameters from samples."""
    # Get all bases once
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    key = f"weights_theta_{param_type}"
    if key in samples and spline_model.n_theta > 0:
        weights_full = samples[key]  # Shape: (n_chains, n_draws, n_weights)
        # For posterior predictive, use first chain only
        weights = weights_full[0]  # Shape: (n_draws, n_weights)
        basis_idx = fft_data.n_dim + (0 if param_type == "re" else 1)
        # Vectorized spline evaluation using JAX
        theta_base = batch_spline_eval(all_bases[basis_idx], weights)
        # Tile to match expected shape
        return jnp.tile(theta_base[:, :, None], (1, 1, max(1, spline_model.n_theta)))
    else:
        first_sample = next(iter(samples.values()))
        n_samples = first_sample.shape[1]  # n_draws
        return jnp.zeros((n_samples, fft_data.n_freq, max(1, spline_model.n_theta)))
