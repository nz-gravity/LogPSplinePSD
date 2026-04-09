from __future__ import annotations

"""Extract data and derived summaries from ArviZ ``InferenceData`` objects."""

from types import SimpleNamespace
from typing import Literal

import arviz as az
import numpy as np
import xarray as xr

from ..datatypes import Periodogram
from ..psplines import LogPSplines, MultivariateLogPSplines
from .to_arviz import (
    _compute_prior_predictive_multivar,
    _flatten_posterior_draws,
    _reconstruct_log_delta_sq,
    _reconstruct_theta_params,
    _select_evenly_spaced_indices,
)


def _nearest_percentile_slice(
    values: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    """Return the percentile slice nearest to ``target``."""
    idx = int(np.argmin(np.abs(percentiles - target)))
    return np.asarray(values[idx])


def get_posterior_psd(idata: az.InferenceData):
    """Return (freqs, median_psd, lower, upper) from stored percentiles."""
    posterior_psd = getattr(idata, "posterior_psd", None)
    if posterior_psd is None or "psd" not in posterior_psd:
        raise KeyError("InferenceData missing posterior_psd 'psd' variable.")
    psd = posterior_psd["psd"]

    freqs = np.asarray(psd.coords["freq"].values)
    percentiles = np.asarray(psd.coords["percentile"].values)
    values = np.asarray(psd.values)

    def _grab(p: float) -> np.ndarray:
        return _nearest_percentile_slice(values, percentiles, p)

    median = _grab(50.0)
    lower = _grab(5.0)
    upper = _grab(95.0)
    return freqs, median, lower, upper


def get_multivar_ci_summary(
    idata: az.InferenceData,
    truth_group: str = "truth_psd",
) -> dict[str, np.ndarray]:
    """Extract multivariate PSD quantiles and truth arrays for plotting.

    Returns a dict with shape conventions:
    - ``freq``: ``(F,)``
    - ``psd_real_q05/q50/q95``: ``(F, C, C)``
    - ``psd_imag_q05/q50/q95``: ``(F, C, C)``
    - ``true_psd_real/true_psd_imag``: ``(F, C, C)``
    """
    truth = getattr(idata, truth_group)
    true_real = np.asarray(truth["psd_matrix_real"].values)
    true_imag = np.asarray(truth["psd_matrix_imag"].values)

    quantiles = get_multivar_posterior_psd_quantiles(idata)
    psd_real = np.asarray(quantiles["real"])
    psd_imag = np.asarray(quantiles["imag"])
    coherence = (
        np.asarray(quantiles["coherence"])
        if quantiles["coherence"] is not None
        else None
    )
    percentiles = np.asarray(quantiles["percentile"], dtype=float)
    freq = np.asarray(quantiles["freq"], dtype=float)

    result = {
        "freq": freq,
        "psd_real_q05": _nearest_percentile_slice(psd_real, percentiles, 5.0),
        "psd_real_q50": _nearest_percentile_slice(psd_real, percentiles, 50.0),
        "psd_real_q95": _nearest_percentile_slice(psd_real, percentiles, 95.0),
        "psd_imag_q05": _nearest_percentile_slice(psd_imag, percentiles, 5.0),
        "psd_imag_q50": _nearest_percentile_slice(psd_imag, percentiles, 50.0),
        "psd_imag_q95": _nearest_percentile_slice(psd_imag, percentiles, 95.0),
        "true_psd_real": np.asarray(true_real, dtype=np.float64),
        "true_psd_imag": np.asarray(true_imag, dtype=np.float64),
    }
    if coherence is not None:
        result["coh_q05"] = _nearest_percentile_slice(
            coherence, percentiles, 5.0
        )
        result["coh_q50"] = _nearest_percentile_slice(
            coherence, percentiles, 50.0
        )
        result["coh_q95"] = _nearest_percentile_slice(
            coherence, percentiles, 95.0
        )
    return result


def get_spline_model(idata: az.InferenceData) -> LogPSplines:
    """Extract spline model from inference data, handling different data structures."""
    dataset = idata["spline_model"]
    return LogPSplines.from_storage_dataset(dataset)


def get_weights(
    idata: az.InferenceData,
    thin: int = 1,
) -> np.ndarray:
    """
    Extract weight samples from arviz InferenceData.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing weight samples
    thin : int
        Thinning factor

    Returns
    -------
    jnp.ndarray
        Weight samples, shape (n_samples_thinned, n_weights)
    """
    weight_samples = idata["posterior"].weights.values
    weight_samples = weight_samples.reshape(-1, weight_samples.shape[-1])
    return weight_samples[::thin]


def get_periodogram(idata: az.InferenceData) -> Periodogram:
    """Extract periodogram from inference data, handling different data structures."""
    return Periodogram(
        power=np.array(idata["observed_data"]["periodogram"].values),
        freqs=np.array(
            idata["observed_data"]["periodogram"].coords["freq"].values
        ),
    )


def get_multivar_spline_model(
    idata: az.InferenceData,
) -> MultivariateLogPSplines:
    """Rehydrate a multivariate spline model from ``idata['spline_model']``."""
    dataset = idata["spline_model"]
    if hasattr(dataset, "ds"):
        dataset = dataset.ds

    degree = int(np.asarray(dataset["degree"]).item())
    diff_matrix_order = int(np.asarray(dataset["diffMatrixOrder"]).item())
    n_freq = int(np.asarray(dataset["N"]).item())
    n_channels = int(np.asarray(dataset["p"]).item())

    diagonal_models = [
        LogPSplines.from_storage_dataset(
            dataset,
            prefix=f"diag_{j}",
            degree=degree,
            diffMatrixOrder=diff_matrix_order,
            n=n_freq,
        )
        for j in range(n_channels)
    ]

    offdiag_re_models = {}
    offdiag_im_models = {}
    for j in range(1, n_channels):
        for l in range(j):
            offdiag_re_models[(j, l)] = LogPSplines.from_storage_dataset(
                dataset,
                prefix=f"theta_re_{j}_{l}",
                degree=degree,
                diffMatrixOrder=diff_matrix_order,
                n=n_freq,
            )
            offdiag_im_models[(j, l)] = LogPSplines.from_storage_dataset(
                dataset,
                prefix=f"theta_im_{j}_{l}",
                degree=degree,
                diffMatrixOrder=diff_matrix_order,
                n=n_freq,
            )

    return MultivariateLogPSplines(
        degree=degree,
        diffMatrixOrder=diff_matrix_order,
        N=n_freq,
        p=n_channels,
        diagonal_models=diagonal_models,
        offdiag_re_models=offdiag_re_models,
        offdiag_im_models=offdiag_im_models,
    )


def get_multivar_cholesky_params(
    idata: az.InferenceData,
    *,
    n_keep: int | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct ``log_delta_sq`` / ``theta_re`` / ``theta_im`` from posterior weights.

    Returns arrays with shapes:
    - ``log_delta_sq``: ``(S, F, p)``
    - ``theta_re``: ``(S, F, n_theta)``
    - ``theta_im``: ``(S, F, n_theta)``
    """
    posterior = idata.posterior
    weight_samples = {
        str(name): np.asarray(var.values)
        for name, var in posterior.data_vars.items()
        if str(name).startswith("weights_")
    }

    flat_samples = {
        key: _flatten_posterior_draws(value)
        for key, value in weight_samples.items()
    }
    if n_keep is not None and int(n_keep) > 0:
        first_key = next(iter(flat_samples))
        keep_idx = _select_evenly_spaced_indices(
            int(flat_samples[first_key].shape[0]),
            int(n_keep),
        )
        if keep_idx is not None:
            flat_samples = {
                key: value[keep_idx] for key, value in flat_samples.items()
            }

    spline_model = get_multivar_spline_model(idata)
    fft_stub = SimpleNamespace(N=spline_model.N, p=spline_model.p)

    return {
        "log_delta_sq": np.asarray(
            _reconstruct_log_delta_sq(flat_samples, spline_model, fft_stub)
        ),
        "theta_re": np.asarray(
            _reconstruct_theta_params(
                flat_samples, spline_model, fft_stub, "re"
            )
        ),
        "theta_im": np.asarray(
            _reconstruct_theta_params(
                flat_samples, spline_model, fft_stub, "im"
            )
        ),
    }


def _get_multivar_frequency_grid(idata: az.InferenceData) -> np.ndarray:
    """Return the multivariate retained frequency grid from ``idata``."""
    return np.asarray(
        idata["observed_data"]["periodogram"].coords["freq"].values,
        dtype=float,
    )


def _get_multivar_reconstruction_inputs_from_dataset(
    posterior: xr.Dataset,
    spline_model: MultivariateLogPSplines,
    *,
    n_keep: int | None,
) -> tuple[MultivariateLogPSplines, dict[str, np.ndarray]]:
    """Return the spline model and capped Cholesky parameters."""
    weight_samples = {
        str(name): np.asarray(var.values)
        for name, var in posterior.data_vars.items()
        if str(name).startswith("weights_")
    }

    flat_samples = {
        key: _flatten_posterior_draws(value)
        for key, value in weight_samples.items()
    }
    first_key = next(iter(flat_samples))
    keep_idx = None
    if n_keep is not None and int(n_keep) > 0:
        keep_idx = _select_evenly_spaced_indices(
            int(flat_samples[first_key].shape[0]),
            int(n_keep),
        )
        if keep_idx is not None:
            flat_samples = {
                key: value[keep_idx] for key, value in flat_samples.items()
            }

    fft_stub = SimpleNamespace(N=spline_model.N, p=spline_model.p)
    params = {
        "log_delta_sq": np.asarray(
            _reconstruct_log_delta_sq(flat_samples, spline_model, fft_stub)
        ),
        "theta_re": np.asarray(
            _reconstruct_theta_params(
                flat_samples, spline_model, fft_stub, "re"
            )
        ),
        "theta_im": np.asarray(
            _reconstruct_theta_params(
                flat_samples, spline_model, fft_stub, "im"
            )
        ),
    }
    return spline_model, params


def _quantiles_to_multivar_dataset(
    quantiles: dict[str, np.ndarray | None],
) -> xr.Dataset:
    """Materialize multivariate PSD quantiles into an ``xarray.Dataset``."""
    coords = {
        "percentile": np.asarray(quantiles["percentile"], dtype=float),
        "freq": np.asarray(quantiles["freq"], dtype=float),
    }
    n_channels = int(np.asarray(quantiles["real"]).shape[-1])
    coords["channels"] = np.arange(n_channels)
    coords["channels2"] = np.arange(n_channels)

    dataset = xr.Dataset(
        {
            "psd_matrix_real": xr.DataArray(
                np.asarray(quantiles["real"], dtype=np.float64),
                dims=("percentile", "freq", "channels", "channels2"),
                coords=coords,
            ),
            "psd_matrix_imag": xr.DataArray(
                np.asarray(quantiles["imag"], dtype=np.float64),
                dims=("percentile", "freq", "channels", "channels2"),
                coords=coords,
            ),
        },
        coords=coords,
    )
    if quantiles["coherence"] is not None:
        dataset["coherence"] = xr.DataArray(
            np.asarray(quantiles["coherence"], dtype=np.float64),
            dims=("percentile", "freq", "channels", "channels2"),
            coords=coords,
        )
    return dataset


def _compute_multivar_psd_quantiles(
    *,
    spline_model: MultivariateLogPSplines,
    params: dict[str, np.ndarray],
    freq: np.ndarray,
    percentiles: tuple[float, ...],
    n_keep: int | None,
    compute_coherence: bool,
    chunk_size: int,
    freq_idx: np.ndarray | list[int] | None,
) -> dict[str, np.ndarray | None]:
    """Compute multivariate PSD quantiles from reconstructed parameters."""
    percentiles_arr = np.asarray(percentiles, dtype=float)
    n_samples_max = (
        int(n_keep)
        if n_keep is not None and int(n_keep) > 0
        else int(params["log_delta_sq"].shape[0])
    )
    real_q, imag_q, coherence_q = spline_model.compute_psd_quantiles(
        params["log_delta_sq"],
        params["theta_re"],
        params["theta_im"],
        percentiles=percentiles_arr,
        n_samples_max=n_samples_max,
        chunk_size=chunk_size,
        compute_coherence=bool(compute_coherence),
    )
    if freq_idx is not None:
        idx = np.asarray(freq_idx, dtype=int)
        freq = np.asarray(freq)[idx]
        real_q = np.asarray(real_q)[:, idx, ...]
        imag_q = np.asarray(imag_q)[:, idx, ...]
        if coherence_q is not None:
            coherence_q = np.asarray(coherence_q)[:, idx, ...]
    return {
        "percentile": np.asarray(percentiles_arr, dtype=float),
        "freq": np.asarray(freq, dtype=float),
        "real": np.asarray(real_q, dtype=np.float64),
        "imag": np.asarray(imag_q, dtype=np.float64),
        "coherence": (
            np.asarray(coherence_q, dtype=np.float64)
            if coherence_q is not None
            else None
        ),
    }


def _rescale_multivar_psd_quantiles(
    idata: az.InferenceData,
    quantiles: dict[str, np.ndarray | None],
) -> dict[str, np.ndarray | None]:
    """Rescale multivariate PSD quantiles back to the physical data scale."""
    attrs = getattr(idata, "attrs", {}) or {}
    channel_stds = attrs.get("channel_stds")
    if channel_stds is None:
        return quantiles
    channel_stds = np.asarray(channel_stds, dtype=np.float64)
    factor_matrix = np.outer(channel_stds, channel_stds).astype(np.float64)
    factor_4d = factor_matrix[None, None, :, :]
    return {
        **quantiles,
        "real": np.asarray(quantiles["real"], dtype=np.float64) * factor_4d,
        "imag": np.asarray(quantiles["imag"], dtype=np.float64) * factor_4d,
    }


def _get_multivar_sample_dataset(
    idata: az.InferenceData,
    sample_source: Literal["posterior", "vi"],
) -> xr.Dataset:
    """Return the multivariate sample dataset for the requested source."""
    if sample_source == "posterior":
        return idata.posterior
    return get_multivar_vi_posterior(idata)


def _resolve_multivar_draw_cap(
    idata: az.InferenceData,
    sample_source: Literal["posterior", "vi"],
    n_keep: int | None,
) -> int | None:
    """Resolve the default posterior/VI reconstruction cap from ``idata.attrs``."""
    if n_keep is not None:
        return int(n_keep)
    attrs = getattr(idata, "attrs", {}) or {}
    attr_name = (
        "posterior_psd_max_draws"
        if sample_source == "posterior"
        else "vi_psd_max_draws"
    )
    value = attrs.get(attr_name, 50)
    if value is None:
        return None
    return int(value)


def _get_multivar_psd_quantiles_from_samples(
    idata: az.InferenceData,
    *,
    sample_source: Literal["posterior", "vi"],
    n_keep: int | None,
    percentiles: tuple[float, ...],
    compute_coherence: bool,
    chunk_size: int,
    freq_idx: np.ndarray | list[int] | None,
) -> dict[str, np.ndarray | None]:
    """Reconstruct multivariate PSD quantiles from posterior-like samples."""
    n_keep = _resolve_multivar_draw_cap(idata, sample_source, n_keep)
    spline_model = get_multivar_spline_model(idata)
    _, params = _get_multivar_reconstruction_inputs_from_dataset(
        _get_multivar_sample_dataset(idata, sample_source),
        spline_model,
        n_keep=n_keep,
    )
    return _rescale_multivar_psd_quantiles(
        idata,
        _compute_multivar_psd_quantiles(
            spline_model=spline_model,
            params=params,
            freq=_get_multivar_frequency_grid(idata),
            percentiles=percentiles,
            n_keep=n_keep,
            compute_coherence=compute_coherence,
            chunk_size=chunk_size,
            freq_idx=freq_idx,
        ),
    )


def get_multivar_posterior_psd_quantiles(
    idata: az.InferenceData,
    n_keep: int | None = None,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
    compute_coherence: bool = True,
    chunk_size: int = 2048,
    freq_idx: np.ndarray | list[int] | None = None,
) -> dict[str, np.ndarray | None]:
    """Return multivariate PSD/coherence quantiles reconstructed from posterior draws.

    Returned arrays follow the plotting/diagnostics layout:
    - ``percentile``: ``(Q,)``
    - ``freq``: ``(F,)``
    - ``real`` / ``imag``: ``(Q, F, p, p)``
    - ``coherence``: ``(Q, F, p, p)`` or ``None``
    """
    return _get_multivar_psd_quantiles_from_samples(
        idata,
        sample_source="posterior",
        n_keep=n_keep,
        percentiles=percentiles,
        compute_coherence=compute_coherence,
        chunk_size=chunk_size,
        freq_idx=freq_idx,
    )


def get_multivar_vi_posterior(idata: az.InferenceData) -> xr.Dataset:
    """Return the multivariate VI posterior sample dataset."""
    attrs = getattr(idata, "attrs", {}) or {}
    if bool(attrs.get("only_vi")):
        return idata.posterior
    vi_posterior = idata["vi_posterior"]
    if hasattr(vi_posterior, "ds"):
        return vi_posterior.ds
    return vi_posterior


def get_multivar_vi_psd_quantiles(
    idata: az.InferenceData,
    n_keep: int | None = None,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
    compute_coherence: bool = True,
    chunk_size: int = 2048,
    freq_idx: np.ndarray | list[int] | None = None,
) -> dict[str, np.ndarray | None]:
    """Return multivariate VI PSD/coherence quantiles reconstructed lazily."""
    return _get_multivar_psd_quantiles_from_samples(
        idata,
        sample_source="vi",
        n_keep=n_keep,
        percentiles=percentiles,
        compute_coherence=compute_coherence,
        chunk_size=chunk_size,
        freq_idx=freq_idx,
    )


def get_multivar_prior_psd_quantiles(
    idata: az.InferenceData,
    n_prior_draws: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray | None]:
    """Return multivariate prior PSD quantiles reconstructed lazily."""
    attrs = getattr(idata, "attrs", {}) or {}
    freq = _get_multivar_frequency_grid(idata)
    spline_model = get_multivar_spline_model(idata)
    fft_stub = SimpleNamespace(
        N=spline_model.N,
        p=spline_model.p,
        freq=freq,
        scaling_factor=float(attrs.get("scaling_factor", 1.0) or 1.0),
    )
    config_stub = SimpleNamespace(
        tau=attrs["tau"],
        design_psd=np.asarray(attrs["design_psd"]),
        channel_stds=attrs.get("channel_stds"),
        alpha_phi=float(attrs.get("alpha_phi", 1.0)),
        beta_phi=float(attrs.get("beta_phi", 1e-4)),
        alpha_delta=float(attrs.get("alpha_delta", 1.0)),
        beta_delta=float(attrs.get("beta_delta", 1.0)),
    )
    real_q, imag_q = _compute_prior_predictive_multivar(
        spline_model,
        fft_stub,
        config_stub,
        n_prior_draws=n_prior_draws,
        seed=seed,
    )
    channel_stds = attrs.get("channel_stds")
    if channel_stds is not None:
        channel_stds = np.asarray(channel_stds, dtype=np.float64)
        factor_matrix = np.outer(channel_stds, channel_stds).astype(np.float64)
        factor_4d = factor_matrix[None, None, :, :]
        real_q = np.asarray(real_q) * factor_4d
        imag_q = np.asarray(imag_q) * factor_4d
    return {
        "percentile": np.asarray([5.0, 50.0, 95.0], dtype=float),
        "freq": np.asarray(freq, dtype=float),
        "real": np.asarray(real_q, dtype=np.float64),
        "imag": np.asarray(imag_q, dtype=np.float64),
        "coherence": None,
    }


def get_multivar_psd_dataset(
    idata: az.InferenceData,
    source: Literal["posterior", "vi", "prior"] = "posterior",
) -> xr.Dataset:
    """Return lazily reconstructed multivariate PSD quantiles as a dataset."""
    quantiles_fn = {
        "posterior": get_multivar_posterior_psd_quantiles,
        "vi": get_multivar_vi_psd_quantiles,
        "prior": get_multivar_prior_psd_quantiles,
    }[source]
    return _quantiles_to_multivar_dataset(quantiles_fn(idata))


def get_posterior_ci(idata: az.InferenceData, n_max=500):
    spline_model = get_spline_model(idata)
    total_n = idata["posterior"].sizes["draw"]

    weights = get_weights(idata, thin=max(1, total_n // n_max))
    model = np.exp(
        np.array(
            [spline_model(w, use_parametric_model=True) for w in weights],
            dtype=np.float64,
        )
    )
    # get 1,2,3 sigma quantiles
    ci_3 = np.percentile(model, [16, 84], axis=0)
    ci_2 = np.percentile(model, [2.5, 97.5], axis=0)
    ci_1 = np.percentile(model, [0.15, 99.85], axis=0)
    return np.array([ci_1, ci_2, ci_3])
