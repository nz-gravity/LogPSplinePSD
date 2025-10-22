"""Extracts data from arviz InferenceData objects"""

import arviz as az
import jax.numpy as jnp
import numpy as np

from ..datatypes import Periodogram
from ..psplines import LogPSplines


def get_posterior_psd(idata: az.InferenceData):
    """Return (freqs, median_psd, lower, upper) from stored percentiles."""

    try:
        psd = idata.posterior_psd["psd"]
    except (AttributeError, KeyError):
        raise KeyError("InferenceData missing posterior_psd 'psd' variable.")

    freqs = np.asarray(psd.coords["freq"].values)
    percentiles = np.asarray(psd.coords["percentile"].values)
    values = np.asarray(psd.values)

    def _grab(p: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(percentiles - p)))
        return values[idx]

    median = _grab(50.0)
    lower = _grab(5.0)
    upper = _grab(95.0)
    return freqs, median, lower, upper


def get_spline_model(idata: az.InferenceData) -> LogPSplines:
    """Extract spline model from inference data, handling different data structures."""
    try:
        dataset = idata["spline_model"]
        knots = dataset["knots"].values
        degree = dataset["degree"].item()
        diffMatrixOrder = dataset["diffMatrixOrder"].item()
        n = dataset["n"].item()
        basis = jnp.array(dataset["basis"].values)
        penalty_matrix = jnp.array(dataset["penalty_matrix"].values)
        parametric_model = jnp.array(
            dataset.get("parametric_model", jnp.ones(n)).values  # type: ignore
        )

        return LogPSplines(
            knots=knots,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            n=n,
            basis=basis,
            penalty_matrix=penalty_matrix,
            parametric_model=parametric_model,
        )
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
            "No variable named 'periodogram'. Variables on the dataset include ['freq', 'channels', 'fft_im', 'fft_re']"
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
