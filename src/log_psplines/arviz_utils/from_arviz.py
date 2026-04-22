from __future__ import annotations

"""Extract data and derived summaries from canonical ``xarray.DataTree`` objects."""

from types import SimpleNamespace
from typing import Literal

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


def _require_dataset(idata: xr.DataTree, group: str) -> xr.Dataset:
    try:
        candidate = idata[group]
    except Exception:
        candidate = getattr(idata, group, None)
    if candidate is None:
        raise KeyError(f"DataTree missing required group '{group}'.")
    dataset = getattr(candidate, "dataset", None)
    if dataset is None and isinstance(candidate, xr.Dataset):
        dataset = candidate
    if dataset is None:
        raise TypeError(f"DataTree group '{group}' must contain a dataset.")
    return dataset


def _nearest_percentile_slice(
    values: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    """Return the percentile slice nearest to ``target``."""
    idx = int(np.argmin(np.abs(percentiles - target)))
    return np.asarray(values[idx])


SampleSource = Literal["best", "posterior", "vi", "prior", "primary"]
ResolvedSampleSource = Literal["posterior", "vi", "prior"]


def _canonical_sample_source(source: SampleSource) -> SampleSource:
    """Normalize legacy source aliases to the public source vocabulary."""
    return "posterior" if source == "primary" else source


def get_sample_dataset(
    idata: xr.DataTree,
    source: SampleSource = "best",
) -> xr.Dataset:
    """Return a posterior-like sample dataset for the requested source.

    Supported ``source`` values:
    - ``"best"``: first available among posterior, VI, then prior-like groups.
    - ``"posterior"`` / ``"primary"``: canonical MCMC posterior.
    - ``"vi"``: VI posterior samples.
    - ``"prior"``: prior samples if available.
    """
    source = _canonical_sample_source(source)

    if source == "posterior":
        return _require_dataset(idata, "posterior")
    if source == "vi":
        return _require_dataset(idata, "vi_posterior")
    if source == "prior":
        for group in ("prior", "prior_predictive"):
            try:
                return _require_dataset(idata, group)
            except (KeyError, TypeError):
                continue
        raise KeyError(
            "DataTree missing required prior group ('prior' or 'prior_predictive')."
        )

    for candidate in ("posterior", "vi", "prior"):
        try:
            return get_sample_dataset(idata, source=candidate)
        except (KeyError, TypeError):
            continue
    raise KeyError(
        "DataTree missing any supported sample group for source='best'."
    )


def _sample_dataset_for_source(
    idata: xr.DataTree, source: ResolvedSampleSource
) -> xr.Dataset:
    """Internal helper returning a dataset from a resolved sample source."""
    return get_sample_dataset(idata, source=source)


def _normalize_chain_draw_array(array: np.ndarray) -> np.ndarray:
    """Normalize posterior-like arrays to ``(chain, draw, ...)``."""
    arr = np.asarray(array)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, arr.shape[0])
    if arr.ndim == 2:
        return arr.reshape(1, arr.shape[0], arr.shape[1])
    return arr


def _rescale_multivar_psd(
    idata: xr.DataTree, spectral_density: np.ndarray
) -> np.ndarray:
    """Rescale multivariate spectra back to the physical data scale."""
    attrs = getattr(idata, "attrs", {}) or {}
    channel_stds = attrs.get("channel_stds")
    if channel_stds is None:
        return spectral_density
    channel_stds = np.asarray(channel_stds, dtype=np.float64)
    factor_matrix = np.outer(channel_stds, channel_stds).astype(np.float64)
    return spectral_density * factor_matrix[None, None, :, :, None]


def _compute_coherence_from_spectral_density(
    spectral_density: np.ndarray,
) -> np.ndarray:
    """Compute coherence from spectral matrices with shape ``(..., p, p, F)``."""
    diag = np.real(
        np.diagonal(spectral_density, axis1=2, axis2=3)
    )  # (..., F, p)
    diag = np.moveaxis(diag, -1, -2)  # (..., p, F)
    denom = diag[..., :, None, :] * diag[..., None, :, :]
    denom = np.where(denom > 0.0, denom, np.nan)
    coherence = (np.abs(spectral_density) ** 2) / denom
    coherence = np.nan_to_num(coherence, nan=0.0, posinf=0.0, neginf=0.0)
    p = spectral_density.shape[2]
    for channel in range(p):
        coherence[:, :, channel, channel, :] = 1.0
    return coherence.astype(np.float64, copy=False)


def _build_psd_dataset(
    *,
    spectral_density: np.ndarray,
    freq: np.ndarray,
    chain_count: int,
    draw_count: int,
) -> xr.Dataset:
    """Build a standardized PSD/CSD dataset."""
    p = int(spectral_density.shape[2])
    coherence = _compute_coherence_from_spectral_density(spectral_density)
    coords = {
        "chain": np.arange(chain_count),
        "draw": np.arange(draw_count),
        "channel": np.arange(p),
        "channel_aux": np.arange(p),
        "frequency": np.asarray(freq, dtype=float),
    }
    return xr.Dataset(
        {
            "spectral_density": xr.DataArray(
                np.asarray(spectral_density, dtype=np.complex128),
                dims=(
                    "chain",
                    "draw",
                    "channel",
                    "channel_aux",
                    "frequency",
                ),
                coords=coords,
            ),
            "coherence": xr.DataArray(
                np.asarray(coherence, dtype=np.float64),
                dims=(
                    "chain",
                    "draw",
                    "channel",
                    "channel_aux",
                    "frequency",
                ),
                coords=coords,
            ),
        },
        coords=coords,
    )


def _compute_univar_psd_dataset(
    idata: xr.DataTree, source: ResolvedSampleSource
) -> xr.Dataset:
    """Reconstruct univariate PSD draws as a 1x1 spectral matrix dataset."""
    posterior = _sample_dataset_for_source(idata, source)
    model = get_spline_model(idata)
    weights = _normalize_chain_draw_array(
        np.asarray(posterior["weights"].values)
    )
    basis = np.asarray(model.basis, dtype=np.float64)
    log_parametric = np.asarray(model.log_parametric_model, dtype=np.float64)
    log_psd = (
        np.einsum("fk,cdk->cdf", basis, weights)
        + log_parametric[None, None, :]
    )
    scaling_factor = float(
        (getattr(idata, "attrs", {}) or {}).get("scaling_factor", 1.0) or 1.0
    )
    psd = np.exp(log_psd) * scaling_factor
    spectral_density = psd[:, :, None, None, :].astype(np.complex128)
    freq = np.asarray(
        _require_dataset(idata, "observed_data")["periodogram"]
        .coords["freq"]
        .values,
        dtype=float,
    )
    return _build_psd_dataset(
        spectral_density=spectral_density,
        freq=freq,
        chain_count=int(spectral_density.shape[0]),
        draw_count=int(spectral_density.shape[1]),
    )


def _compute_multivar_psd_dataset(
    idata: xr.DataTree, source: ResolvedSampleSource
) -> xr.Dataset:
    """Reconstruct multivariate PSD/CSD draws from posterior-like weights."""
    posterior = _sample_dataset_for_source(idata, source)
    spline_model = get_multivar_spline_model(idata)
    sample_weight = next(
        np.asarray(var.values)
        for name, var in posterior.data_vars.items()
        if str(name).startswith("weights_")
    )
    sample_weight = _normalize_chain_draw_array(sample_weight)
    chain_count = int(sample_weight.shape[0])
    draw_count = int(sample_weight.shape[1])
    params = _get_multivar_reconstruction_inputs_from_dataset(
        posterior,
        spline_model,
        n_keep=None,
    )
    spectral_density = spline_model.reconstruct_psd_matrix(
        params["log_delta_sq"],
        params["theta_re"],
        params["theta_im"],
        n_samples_max=int(params["log_delta_sq"].shape[0]),
    )
    spectral_density = np.moveaxis(np.asarray(spectral_density), 1, -1)
    spectral_density = spectral_density.reshape(
        chain_count,
        draw_count,
        spectral_density.shape[1],
        spectral_density.shape[2],
        spectral_density.shape[3],
    )
    spectral_density = _rescale_multivar_psd(idata, spectral_density)
    return _build_psd_dataset(
        spectral_density=spectral_density,
        freq=_get_multivar_frequency_grid(idata),
        chain_count=chain_count,
        draw_count=draw_count,
    )


def _resolved_source_for_psd(source: SampleSource) -> ResolvedSampleSource:
    """Resolve and validate a PSD source for draw-level reconstruction."""
    source = _canonical_sample_source(source)
    if source not in {"posterior", "vi", "prior"}:
        raise ValueError(f"Unsupported PSD source '{source}'.")
    return source


def _get_psd_dataset_from_source(
    idata: xr.DataTree,
    source: ResolvedSampleSource,
) -> xr.Dataset:
    """Build standardized PSD/CSD draws from a resolved sample source."""
    attrs = getattr(idata, "attrs", {}) or {}
    is_multivar = str(attrs.get("data_type", "")).lower().startswith("multi")
    if is_multivar:
        return _compute_multivar_psd_dataset(idata, source)
    return _compute_univar_psd_dataset(idata, source)


def get_psd_dataset(
    idata: xr.DataTree,
    source: SampleSource = "best",
) -> xr.Dataset:
    """Return standardized PSD/CSD draws for the requested source.

    Supported ``source`` values:
    - ``"best"``: first available among posterior, VI, then prior.
    - ``"posterior"`` / ``"primary"``: MCMC posterior draws.
    - ``"vi"``: VI posterior draws.
    - ``"prior"``: prior-like draws if present.

    Returned datasets contain:
    - ``spectral_density``: complex, dims ``(chain, draw, channel, channel_aux, frequency)``
    - ``coherence``: real, dims ``(chain, draw, channel, channel_aux, frequency)``
    """
    source = _canonical_sample_source(source)
    if source == "best":
        for candidate in ("posterior", "vi", "prior"):
            try:
                return _get_psd_dataset_from_source(idata, candidate)
            except (KeyError, TypeError, ValueError, StopIteration):
                continue
        raise KeyError(
            "Unable to resolve PSD draws from source='best' for this DataTree."
        )
    resolved = _resolved_source_for_psd(source)
    return _get_psd_dataset_from_source(idata, resolved)


def _quantiles_from_psd_draws(
    dataset: xr.Dataset,
    *,
    n_keep: int | None,
    percentiles: tuple[float, ...],
    freq_idx: np.ndarray | list[int] | None,
) -> dict[str, np.ndarray | None]:
    """Compute PSD/CSD quantiles from standardized draw-level datasets."""
    spectral_density = np.asarray(dataset["spectral_density"].values)
    coherence = (
        np.asarray(dataset["coherence"].values)
        if "coherence" in dataset
        else _compute_coherence_from_spectral_density(spectral_density)
    )
    chain_count, draw_count = spectral_density.shape[:2]
    n_samples = chain_count * draw_count
    spectral_density = spectral_density.reshape(
        n_samples, *spectral_density.shape[2:]
    )
    coherence = coherence.reshape(n_samples, *coherence.shape[2:])

    if n_keep is not None and int(n_keep) > 0:
        keep_idx = _select_evenly_spaced_indices(n_samples, int(n_keep))
        if keep_idx is not None:
            spectral_density = spectral_density[keep_idx]
            coherence = coherence[keep_idx]

    freq = np.asarray(dataset.coords["frequency"].values, dtype=float)
    real = np.percentile(spectral_density.real, percentiles, axis=0)
    imag = np.percentile(spectral_density.imag, percentiles, axis=0)
    coherence_q = np.percentile(coherence, percentiles, axis=0)

    if freq_idx is not None:
        idx = np.asarray(freq_idx, dtype=int)
        freq = freq[idx]
        real = real[..., idx]
        imag = imag[..., idx]
        coherence_q = coherence_q[..., idx]

    real = np.moveaxis(real, -1, 1)
    imag = np.moveaxis(imag, -1, 1)
    coherence_q = np.moveaxis(coherence_q, -1, 1)
    return {
        "percentile": np.asarray(percentiles, dtype=float),
        "freq": freq,
        "real": np.asarray(real, dtype=np.float64),
        "imag": np.asarray(imag, dtype=np.float64),
        "coherence": np.asarray(coherence_q, dtype=np.float64),
    }


def get_posterior_psd(idata: xr.DataTree):
    """Return ``(freqs, median_psd, lower, upper)`` for the primary posterior."""
    quantiles = _quantiles_from_psd_draws(
        get_psd_dataset(idata, "posterior"),
        n_keep=None,
        percentiles=(5.0, 50.0, 95.0),
        freq_idx=None,
    )
    real = np.asarray(quantiles["real"], dtype=np.float64)[:, :, 0, 0]
    freqs = np.asarray(quantiles["freq"], dtype=float)
    percentiles = np.asarray(quantiles["percentile"], dtype=float)

    def _grab(target: float) -> np.ndarray:
        return _nearest_percentile_slice(real, percentiles, target)

    return freqs, _grab(50.0), _grab(5.0), _grab(95.0)


def get_multivar_ci_summary(
    idata: xr.DataTree,
    truth_group: str = "truth_psd",
) -> dict[str, np.ndarray]:
    """Extract multivariate PSD quantiles and truth arrays for plotting.

    Returns a dict with shape conventions:
    - ``freq``: ``(F,)``
    - ``psd_real_q05/q50/q95``: ``(F, C, C)``
    - ``psd_imag_q05/q50/q95``: ``(F, C, C)``
    - ``true_psd_real/true_psd_imag``: ``(F, C, C)``
    """
    truth = _require_dataset(idata, truth_group)
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


def get_spline_model(idata: xr.DataTree) -> LogPSplines:
    """Extract the stored univariate spline model."""
    dataset = _require_dataset(idata, "spline_model")
    return LogPSplines.from_storage_dataset(dataset)


def get_weights(
    idata: xr.DataTree,
    thin: int = 1,
) -> np.ndarray:
    """
    Extract weight samples from the canonical posterior group.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree containing posterior weight samples
    thin : int
        Thinning factor

    Returns
    -------
    jnp.ndarray
        Weight samples, shape (n_samples_thinned, n_weights)
    """
    weight_samples = _require_dataset(idata, "posterior")["weights"].values
    weight_samples = weight_samples.reshape(-1, weight_samples.shape[-1])
    return weight_samples[::thin]


def get_periodogram(idata: xr.DataTree) -> Periodogram:
    """Extract the observed periodogram."""
    observed_data = _require_dataset(idata, "observed_data")
    return Periodogram(
        power=np.array(observed_data["periodogram"].values),
        freqs=np.array(observed_data["periodogram"].coords["freq"].values),
    )


def get_multivar_spline_model(
    idata: xr.DataTree,
) -> MultivariateLogPSplines:
    """Rehydrate a multivariate spline model from ``idata['spline_model']``."""
    dataset = _require_dataset(idata, "spline_model")

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


def _get_multivar_frequency_grid(idata: xr.DataTree) -> np.ndarray:
    """Return the multivariate retained frequency grid from ``idata``."""
    observed_data = _require_dataset(idata, "observed_data")
    return np.asarray(
        observed_data["periodogram"].coords["freq"].values,
        dtype=float,
    )


def _get_multivar_reconstruction_inputs_from_dataset(
    posterior: xr.Dataset,
    spline_model: MultivariateLogPSplines,
    *,
    n_keep: int | None,
) -> dict[str, np.ndarray]:
    """Return capped multivariate Cholesky parameters."""
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


def _resolve_multivar_draw_cap(
    idata: xr.DataTree,
    sample_source: Literal["posterior", "vi"],
    n_keep: int | None,
) -> int | None:
    """Resolve the default posterior/VI reconstruction cap from ``idata.attrs``."""
    if n_keep is not None:
        return int(n_keep)
    attrs = idata.attrs or {}
    attr_name = (
        "posterior_psd_max_draws"
        if sample_source == "posterior"
        else "vi_psd_max_draws"
    )
    value = attrs.get(attr_name, 50)
    if value is None:
        return None
    return int(value)


def _get_multivar_psd_quantiles(
    idata: xr.DataTree,
    *,
    source: Literal["posterior", "vi"],
    n_keep: int | None,
    percentiles: tuple[float, ...],
    freq_idx: np.ndarray | list[int] | None,
) -> dict[str, np.ndarray | None]:
    return _quantiles_from_psd_draws(
        get_psd_dataset(idata, source),
        n_keep=_resolve_multivar_draw_cap(idata, source, n_keep),
        percentiles=percentiles,
        freq_idx=freq_idx,
    )


def get_multivar_posterior_psd_quantiles(
    idata: xr.DataTree,
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
    del compute_coherence, chunk_size
    return _get_multivar_psd_quantiles(
        idata,
        source="posterior",
        n_keep=n_keep,
        percentiles=percentiles,
        freq_idx=freq_idx,
    )


def get_multivar_vi_psd_quantiles(
    idata: xr.DataTree,
    n_keep: int | None = None,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
    compute_coherence: bool = True,
    chunk_size: int = 2048,
    freq_idx: np.ndarray | list[int] | None = None,
) -> dict[str, np.ndarray | None]:
    """Return multivariate VI PSD/coherence quantiles reconstructed lazily."""
    del compute_coherence, chunk_size
    return _get_multivar_psd_quantiles(
        idata,
        source="vi",
        n_keep=n_keep,
        percentiles=percentiles,
        freq_idx=freq_idx,
    )


def get_multivar_prior_psd_quantiles(
    idata: xr.DataTree,
    n_prior_draws: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray | None]:
    """Return multivariate prior PSD quantiles reconstructed lazily."""
    attrs = idata.attrs or {}
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


def get_posterior_ci(idata: xr.DataTree, n_max=500):
    spline_model = get_spline_model(idata)
    total_n = _require_dataset(idata, "posterior").sizes["draw"]

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
