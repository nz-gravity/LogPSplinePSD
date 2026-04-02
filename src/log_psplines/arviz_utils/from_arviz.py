from __future__ import annotations

"""Extracts data from arviz InferenceData objects"""

import arviz as az
import numpy as np

from ..datatypes import Periodogram
from ..psplines import LogPSplines


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
        posterior = getattr(idata, posterior_group)
        truth = getattr(idata, truth_group)
    except AttributeError as exc:
        raise KeyError(
            f"InferenceData missing '{posterior_group}' or '{truth_group}' group."
        ) from exc

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
            posterior["psd_matrix_real"].coords["freq"].values, dtype=float
        )
        true_real = np.asarray(truth["psd_matrix_real"].values)
        true_imag = np.asarray(truth["psd_matrix_imag"].values)
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "InferenceData missing multivariate PSD variables required for plotting."
        ) from exc

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
