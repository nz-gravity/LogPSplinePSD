"""Convert inference results to ArviZ InferenceData format."""

import warnings
from dataclasses import asdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from xarray import DataArray, Dataset

from log_psplines.datatypes import MultivarFFT, Periodogram

from ..datatypes.multivar_utils import wishart_u_to_psd
from ..logger import logger

warnings.filterwarnings("ignore", module="arviz")

if TYPE_CHECKING:
    from ..psplines.multivar_psplines import MultivariateLogPSplines
    from ..psplines.psplines import LogPSplines
    from ..samplers.base_sampler import SamplerConfig


def results_to_arviz(
    samples: Mapping[str, Any],
    sample_stats: Dict[str, Any],
    config: "SamplerConfig",
    data: Union[Periodogram, MultivarFFT],
    model: Union["LogPSplines", "MultivariateLogPSplines"],
    attributes: Dict[str, Any],
) -> az.InferenceData:
    """Unified ArviZ conversion for both univar and multivar cases."""

    if isinstance(data, Periodogram):
        logger.debug("results_to_arviz: detected Periodogram")
        idata = _create_univar_inference_data(
            dict(samples),
            sample_stats,
            config,
            data,
            cast("LogPSplines", model),
            attributes,
        )
    elif isinstance(data, MultivarFFT):
        logger.debug("results_to_arviz: detected MultivarFFT")
        idata = _create_multivar_inference_data(
            dict(samples),
            sample_stats,
            config,
            data,
            cast("MultivariateLogPSplines", model),
            attributes,
        )
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Compute diagnostics if true_psd is provided (using posterior_psd from idata)
    if config.true_psd is not None:
        from ..diagnostics.psd_compare import _run as _run_psd_compare

        psd_metrics = _run_psd_compare(
            idata=idata, truth=np.asarray(config.true_psd)
        )
        idata.attrs.update(psd_metrics)
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

    def _sanitize_attr_value(value: Any) -> Any | None:
        """Return a NetCDF/HDF5-friendly attribute value or None if unsupported."""

        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float, str, np.number)):
            return value

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return None
            if value.dtype.kind == "U":
                # h5py cannot store numpy unicode arrays as attrs; encode to bytes.
                return value.astype("S")
            return value

        if isinstance(value, (list, tuple)):
            if not value:
                return np.asarray(value, dtype=float)
            if all(
                isinstance(v, (bool, int, float, np.number)) for v in value
            ):
                arr = np.asarray(
                    [int(v) if isinstance(v, bool) else v for v in value],
                    dtype=float,
                )
                return arr
            if all(isinstance(v, str) for v in value):
                # Keep as a plain Python list of strings (h5py can't store
                # numpy unicode arrays in attrs, but lists of strings work).
                return list(value)
            return None

        return None

    # Convert config to attributes (handle booleans) and drop non-serialisable
    # objects (e.g. plotting helpers) that would break netcdf saving.
    config_attrs: Dict[str, Any] = {}
    for key, val in asdict(config).items():
        if val is None:
            continue
        sanitized = _sanitize_attr_value(val)
        if sanitized is None:
            continue
        config_attrs[key] = sanitized

    true_psd_attr = config_attrs.pop("true_psd", None)
    if true_psd_attr is not None:
        config_attrs["true_psd_provided"] = 1

    attributes.update(config_attrs)

    # Add ESS calculation (best-effort; keep small set for speed)
    try:

        def _select_weight_indices(size: int, max_points: int) -> np.ndarray:
            if size <= 0 or max_points <= 0:
                return np.array([], dtype=int)
            if size <= max_points:
                return np.arange(size, dtype=int)
            idx = np.unique(
                np.linspace(
                    0, size - 1, num=max_points, dtype=int, endpoint=True
                )
            )
            return idx

        ess_samples: Dict[str, Any] = {}
        for name, arr in samples.items():
            if str(name).startswith("weights"):
                if getattr(arr, "ndim", 0) >= 3:
                    size = int(arr.shape[-1])
                    idx = _select_weight_indices(size, 6)
                    if idx.size == 0:
                        continue
                    ess_samples[name] = arr[..., idx]
                else:
                    ess_samples[name] = arr
                continue
            ess_samples[name] = arr

        if not ess_samples:
            ess_samples = dict(samples)

        ess_vars = list(ess_samples.keys())
        if len(ess_vars) > 10:
            ess_vars = ess_vars[:10]
        summary = az.summary(ess_samples, var_names=ess_vars, round_to=2)
        ess_vals = summary["ess_bulk"].values
        attributes.update(dict(ess=ess_vals))

        ess_sorted = summary["ess_bulk"].sort_values()
        n_low = min(5, len(ess_sorted))
        lowest_names = [str(name) for name in ess_sorted.index[:n_low]]
        lowest_vals = [float(val) for val in ess_sorted.iloc[:n_low]]
        attributes.update(
            dict(
                ess_lowest_names=lowest_names,
                ess_lowest_values=np.asarray(lowest_vals, dtype=float),
            )
        )
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
    pp_idx: np.ndarray | slice
    if n_draws > n_pp:
        pp_idx = np.unique(
            np.linspace(0, n_draws - 1, num=n_pp, dtype=int, endpoint=True)
        )
    else:
        pp_idx = slice(None)

    percentiles = np.array([5.0, 50.0, 95.0], dtype=np.float64)

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
            samples, sample_stats, spline_model, fft_data, config
        )
    )
    psd_real_q = np.asarray(psd_real_q, dtype=np.float64)
    psd_imag_q = np.asarray(psd_imag_q, dtype=np.float64)
    coh_q = np.asarray(coh_q, dtype=np.float64) if coh_q is not None else None

    channel_stds = getattr(config, "channel_stds", None)
    factor_matrix = None
    sf = float(getattr(fft_data, "scaling_factor", 1.0) or 1.0)
    if channel_stds is not None:
        channel_stds = np.asarray(channel_stds, dtype=np.float64)
        if channel_stds.shape[0] != fft_data.p:
            raise ValueError(
                "channel_stds length must match number of channels in FFT data."
            )
        scale_matrix = np.outer(channel_stds, channel_stds).astype(np.float64)
        factor_matrix = scale_matrix
        factor_4d = factor_matrix[None, None, :, :]
        psd_real_q_rescaled = psd_real_q * factor_4d
        psd_imag_q_rescaled = psd_imag_q * factor_4d
    else:
        psd_real_q_rescaled = psd_real_q
        psd_imag_q_rescaled = psd_imag_q
    coherence_q_rescaled = coh_q
    scalar_factor = float(getattr(config, "scaling_factor", 1.0) or 1.0)

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
        Nh = getattr(fft_data, "Nh", 1)
        if isinstance(Nh, bool) or not isinstance(Nh, (int, np.integer)):
            raise TypeError("fft_data.Nh must be a positive integer")
        Nh = int(Nh)
        if Nh <= 0:
            raise ValueError("fft_data.Nh must be positive")
        observed_csd = wishart_u_to_psd(
            fft_data.U,
            Nb=getattr(fft_data, "Nb", 1),
            scaling_factor=float(getattr(fft_data, "scaling_factor", 1.0)),
            Nh=Nh,
        )
        psd_has_global_scale = True
        # wishart_u_to_psd already has scaling_factor applied, so remove it before rescaling
        if channel_stds is not None:
            observed_csd = observed_csd / sf
    else:
        raise ValueError(
            "Multivariate observed data requires raw_psd or u_re/u_im."
        )

    if channel_stds is not None and factor_matrix is not None:
        observed_csd = observed_csd * factor_matrix[None, :, :]
    else:
        if not psd_has_global_scale:
            observed_csd = observed_csd

    if config.verbose:
        logger.info(
            f"Rescaling multivariate posterior samples: max scaling ~{config.scaling_factor:.2e}"
        )

    observed_psd_rescaled = {"periodogram": observed_csd}

    _handle_log_posterior(sample_stats)

    coords = {
        "percentile": percentiles,
        "freq": np.array(fft_data.freq),
        "channels": range(fft_data.p),
        "channels2": range(fft_data.p),
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
            "p": fft_data.p,
            "N": fft_data.N,
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
        "N": spline_model.N,
        "p": spline_model.p,
        "n_theta": spline_model.n_theta,
    }

    coords: Dict[str, np.ndarray] = {}

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
    config: Optional["SamplerConfig"] = None,
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
            if arr.shape[1] == fft_data.N:
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
    psd_real_q, psd_imag_q, coh_q = spline_model.compute_psd_quantiles(
        log_delta_sq,
        theta_re,
        theta_im,
        percentiles=percentiles,
        n_samples_max=n_draw_cap,
        compute_coherence=compute_coh,
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

    for j in range(fft_data.p):
        weights_key = f"weights_delta_{j}"
        if weights_key in samples:
            weights_full = samples[weights_key]
            weights = weights_full[0]
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
    all_bases, _ = spline_model.get_all_bases_and_penalties()

    key = f"weights_theta_{param_type}"
    if key in samples and spline_model.n_theta > 0:
        weights_full = samples[key]
        weights = weights_full[0]
        basis_idx = fft_data.p + (0 if param_type == "re" else 1)
        theta_base = batch_spline_eval(all_bases[basis_idx], weights)
        return jnp.tile(
            theta_base[:, :, None], (1, 1, max(1, spline_model.n_theta))
        )
    else:
        first_sample = next(iter(samples.values()))
        n_samples = first_sample.shape[1]
        return jnp.zeros((n_samples, fft_data.N, max(1, spline_model.n_theta)))
