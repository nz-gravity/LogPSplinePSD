from __future__ import annotations

"""Extract data and derived summaries from ArviZ ``InferenceData`` objects."""

from types import SimpleNamespace

import arviz as az
import numpy as np

from ..datatypes import Periodogram
from ..psplines import LogPSplines, MultivariateLogPSplines
from .to_arviz import (
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

    try:
        posterior_psd = getattr(idata, "posterior_psd")
        psd = posterior_psd["psd"]
    except (AttributeError, KeyError, TypeError):
        raise KeyError("InferenceData missing posterior_psd 'psd' variable.")

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
    posterior_group: str = "posterior_psd",
    truth_group: str = "truth_psd",
) -> dict[str, np.ndarray]:
    """Extract multivariate PSD quantiles and truth arrays for plotting.

    Returns a dict with shape conventions:
    - ``freq``: ``(F,)``
    - ``psd_real_q05/q50/q95``: ``(F, C, C)``
    - ``psd_imag_q05/q50/q95``: ``(F, C, C)``
    - ``true_psd_real/true_psd_imag``: ``(F, C, C)``
    """
    try:
        truth = getattr(idata, truth_group)
        true_real = np.asarray(truth["psd_matrix_real"].values)
        true_imag = np.asarray(truth["psd_matrix_imag"].values)
    except (AttributeError, KeyError, TypeError) as exc:
        raise KeyError(
            f"InferenceData missing '{truth_group}' group."
        ) from exc

    posterior = getattr(idata, posterior_group, None)
    if posterior is not None and "psd_matrix_real" in posterior:
        try:
            psd_real = np.asarray(posterior["psd_matrix_real"].values)
            psd_imag = np.asarray(posterior["psd_matrix_imag"].values)
            coherence = (
                np.asarray(posterior["coherence"].values)
                if "coherence" in posterior
                else None
            )
            percentiles = np.asarray(
                posterior["psd_matrix_real"].coords["percentile"].values,
                dtype=float,
            )
            freq = np.asarray(
                posterior["psd_matrix_real"].coords["freq"].values,
                dtype=float,
            )
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "InferenceData missing multivariate PSD variables required for plotting."
            ) from exc
    elif posterior_group == "posterior_psd":
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
    else:
        raise KeyError(f"InferenceData missing '{posterior_group}' group.")

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
    try:
        dataset = idata["spline_model"]
        return LogPSplines.from_storage_dataset(dataset)
    except KeyError:
        # For VI or other data structures where spline_model is not available
        # Return None or a default - let the calling function handle this
        raise KeyError(
            "No variable named 'knots'. Spline model data not available in idata."
        )


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
    try:
        # Get weight samples and flatten chains
        weight_samples = idata[
            "posterior"
        ].weights.values  # (chains, draws, n_weights)
        weight_samples = weight_samples.reshape(
            -1, weight_samples.shape[-1]
        )  # (chains*draws, n_weights)

        # Thin samples
        return weight_samples[::thin]
    except (KeyError, AttributeError):
        # For VI or other data structures where weights are not available
        raise KeyError("No weights data available in posterior")


def get_periodogram(idata: az.InferenceData) -> Periodogram:
    """Extract periodogram from inference data, handling different data structures."""
    try:
        return Periodogram(
            power=np.array(idata["observed_data"]["periodogram"].values),
            freqs=np.array(
                idata["observed_data"]["periodogram"].coords["freq"].values
            ),
        )
    except KeyError:
        # For VI or other data structures where periodogram is not available
        raise KeyError(
            "No variable named 'periodogram'. Observed data should include "
            "['freq', 'channels', 'periodogram']."
        )


def get_multivar_spline_model(
    idata: az.InferenceData,
) -> MultivariateLogPSplines:
    """Rehydrate a multivariate spline model from ``idata['spline_model']``."""
    try:
        dataset = idata["spline_model"]
    except KeyError as exc:
        raise KeyError("InferenceData is missing 'spline_model'.") from exc

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
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise KeyError("InferenceData is missing posterior samples.")

    weight_samples = {
        str(name): np.asarray(var.values)
        for name, var in posterior.data_vars.items()
        if str(name).startswith("weights_")
    }
    if not weight_samples:
        raise KeyError(
            "InferenceData posterior does not contain weight samples."
        )

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


def _get_multivar_frequency_grid(
    idata: az.InferenceData,
    n_freq: int,
) -> np.ndarray:
    """Return the multivariate retained frequency grid from ``idata``."""
    observed = getattr(idata, "observed_data", None)
    if observed is not None and "periodogram" in observed:
        try:
            return np.asarray(
                observed["periodogram"].coords["freq"].values, dtype=float
            )
        except Exception:
            pass

    attrs = getattr(idata, "attrs", {}) or {}
    if hasattr(attrs, "get"):
        try:
            freq = attrs.get("frequencies")
            if freq is not None:
                arr = np.asarray(freq, dtype=float)
                if arr.ndim == 1 and arr.size == n_freq:
                    return arr
        except Exception:
            pass

    return np.arange(n_freq, dtype=float)


def get_multivar_posterior_psd_quantiles(
    idata: az.InferenceData,
    n_keep: int = 50,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
    compute_coherence: bool = True,
    chunk_size: int = 2048,
    freq_idx: np.ndarray | list[int] | None = None,
) -> dict[str, np.ndarray | None]:
    """Return multivariate PSD/coherence quantiles from stored or reconstructed draws.

    Returned arrays follow the plotting/diagnostics layout:
    - ``percentile``: ``(Q,)``
    - ``freq``: ``(F,)``
    - ``real`` / ``imag``: ``(Q, F, p, p)``
    - ``coherence``: ``(Q, F, p, p)`` or ``None``
    """
    posterior_psd = getattr(idata, "posterior_psd", None)
    percentiles_arr = np.asarray(percentiles, dtype=float)

    if posterior_psd is not None and "psd_matrix_real" in posterior_psd:
        real_q = np.asarray(posterior_psd["psd_matrix_real"].values)
        imag_q = np.asarray(posterior_psd["psd_matrix_imag"].values)
        coherence_q = (
            np.asarray(posterior_psd["coherence"].values)
            if "coherence" in posterior_psd
            else None
        )
        freq = np.asarray(
            posterior_psd["psd_matrix_real"].coords["freq"].values,
            dtype=float,
        )
        stored_percentiles = np.asarray(
            posterior_psd["psd_matrix_real"].coords["percentile"].values,
            dtype=float,
        )
        if stored_percentiles.shape == percentiles_arr.shape and np.allclose(
            stored_percentiles, percentiles_arr
        ):
            percentiles_arr = stored_percentiles
        else:
            percentiles_arr = stored_percentiles
    else:
        params = get_multivar_cholesky_params(idata, n_keep=n_keep)
        spline_model = get_multivar_spline_model(idata)
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
        freq = _get_multivar_frequency_grid(idata, spline_model.N)

    if freq_idx is not None:
        idx = np.asarray(freq_idx, dtype=int)
        freq = freq[idx]
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
