import dataclasses

import arviz as az
import jax.numpy as jnp
import numpy as np

from ..datatypes import Periodogram
from ..psplines import LogPSplines

__all__ = ["unpack_data"]


@dataclasses.dataclass
class PlottingData:
    freqs: jnp.ndarray = None
    pdgrm: jnp.ndarray = None
    model: jnp.ndarray = None
    ci: jnp.ndarray = None

    @property
    def n(self):
        if self.freqs is not None:
            return len(self.freqs)
        elif self.pdgrm is not None:
            return len(self.pdgrm)
        elif self.model is not None:
            return len(self.model)
        else:
            raise ValueError("No data to get length from.")


def unpack_data(
    pdgrm: Periodogram = None,
    spline_model: LogPSplines = None,
    weights=None,
    yscalar=1.0,
    use_uniform_ci=True,
    use_parametric_model=True,
    freqs=None,
):
    plt_dat = PlottingData()
    if pdgrm is not None:
        plt_dat.pdgrm = np.array(pdgrm.power, dtype=np.float64) * yscalar
        plt_dat.freqs = pdgrm.freqs

    if spline_model is not None:
        ln_param = spline_model.log_parametric_model
        if not use_parametric_model:
            ln_param = jnp.zeros_like(ln_param)

        if weights is None:
            # just use the initial weights/0 weights
            ln_spline = spline_model()

        elif weights.ndim == 1:
            # only one set of weights -- no CI possible
            ln_spline = spline_model(weights)

        else:  # weights.ndim == 2
            # multiple sets of weights -- CI possible
            ln_splines = jnp.array([spline_model(w) for w in weights])

            if use_uniform_ci:
                ln_ci = _get_uni_ci(ln_splines)
            else:  # percentile
                ln_ci = jnp.percentile(
                    ln_splines, q=jnp.array([16, 50, 84]), axis=0
                )
            ln_ci = jnp.array(ln_ci)
            plt_dat.ci = np.exp(ln_ci + ln_param, dtype=np.float64) * yscalar
            ln_spline = ln_ci[1]
        plt_dat.model = (
            np.exp(ln_spline + ln_param, dtype=np.float64) * yscalar
        )

    if plt_dat.freqs is None and freqs is None:
        plt_dat.freqs = np.linspace(0, 1, plt_dat.n)
    elif freqs is not None:
        plt_dat.freqs = freqs

    return plt_dat


def _get_uni_ci(samples, alpha=0.1):
    """
    Compute a uniform (simultaneous) confidence band for a set of function samples.

    Args:
        samples (jnp.ndarray): Shape (num_samples, num_points) array of function samples.
        alpha (float): Significance level (default 0.1 for 90% CI).

    Returns:
        tuple: (lower_bound, median, upper_bound) arrays.
    """
    num_samples, num_points = samples.shape

    # Compute pointwise median and standard deviation
    median = jnp.median(samples, axis=0)
    std = jnp.std(samples, axis=0)

    # Compute the max deviation over all samples
    deviations = (samples - median[None, :]) / std[
        None, :
    ]  # Normalize deviations
    max_deviation = jnp.max(
        jnp.abs(deviations), axis=1
    )  # Max deviation per sample

    # Compute the scaling factor using the distribution of max deviations
    k_alpha = jnp.percentile(
        max_deviation, 100 * (1 - alpha)
    )  # Critical value

    # Compute uniform confidence bands
    lower_bound = median - k_alpha * std
    upper_bound = median + k_alpha * std

    return lower_bound, median, upper_bound


def get_weights(
    idata: az.InferenceData,
    thin: int = 10,
) -> jnp.ndarray:
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
    # Get weight samples and flatten chains
    weight_samples = (
        idata.posterior.weights.values
    )  # (chains, draws, n_weights)
    weight_samples = weight_samples.reshape(
        -1, weight_samples.shape[-1]
    )  # (chains*draws, n_weights)

    # Thin samples
    return weight_samples[::thin]


def get_psd_samples_arviz(
    idata: az.InferenceData, spline_model: LogPSplines, thin: int = 10
) -> jnp.ndarray:
    """
    Extract PSD samples from arviz InferenceData.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing weight samples
    spline_model : LogPSplines
        Spline model for reconstruction
    thin : int
        Thinning factor

    Returns
    -------
    jnp.ndarray
        PSD samples, shape (n_samples_thinned, n_frequencies)
    """
    # Get weight samples and flatten chains
    weight_samples = get_weights(idata, thin=thin)

    # Compute PSD samples
    psd_samples = []
    for weights in weight_samples:
        ln_spline = spline_model.basis.T @ weights
        ln_psd = ln_spline + spline_model.log_parametric_model
        psd_samples.append(jnp.exp(ln_psd))

    return jnp.array(psd_samples)
