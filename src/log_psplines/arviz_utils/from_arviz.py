"""Extracts data from arviz InferenceData objects"""
import arviz as az
import numpy as np
import jax.numpy as jnp

from ..psplines import LogPSplines
from ..datatypes import Periodogram


def get_spline_model(idata: az.InferenceData) -> LogPSplines:
    dataset = idata["spline_model"]
    knots = dataset["knots"].values
    degree = dataset["degree"].item()
    diffMatrixOrder = dataset["diffMatrixOrder"].item()
    n = dataset["n"].item()
    basis = jnp.array(dataset["basis"].values)
    penalty_matrix = jnp.array(dataset["penalty_matrix"].values)
    parametric_model = jnp.array(dataset.get("parametric_model", jnp.ones(n)).values)

    return LogPSplines(
        knots=knots,
        degree=degree,
        diffMatrixOrder=diffMatrixOrder,
        n=n,
        basis=basis,
        penalty_matrix=penalty_matrix,
        parametric_model=parametric_model,
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
    # Get weight samples and flatten chains
    weight_samples = (
        idata["posterior"].weights.values
    )  # (chains, draws, n_weights)
    weight_samples = weight_samples.reshape(
        -1, weight_samples.shape[-1]
    )  # (chains*draws, n_weights)

    # Thin samples
    return weight_samples[::thin]


def get_periodogram(idata: az.InferenceData) -> Periodogram:
    return Periodogram(
        power=jnp.array(idata["observed_data"]["periodogram"].values),
        freqs=jnp.array(idata["observed_data"]["periodogram"].coords["freq"].values)
    )
