import os
from typing import List

import arviz as az
import numpy as np
from xarray import DataArray, Dataset

from .psplines import LogPSplines


def get_weights(
    idata: az.InferenceData,
    thin: int = 10,
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
        idata.posterior.weights.values
    )  # (chains, draws, n_weights)
    weight_samples = weight_samples.reshape(
        -1, weight_samples.shape[-1]
    )  # (chains*draws, n_weights)

    # Thin samples
    return weight_samples[::thin]


def get_psd_samples_arviz(
    idata: az.InferenceData, spline_model: LogPSplines, thin: int = 10
) -> np.ndarray:
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
        psd_samples.append(np.exp(ln_psd))

    return np.array(psd_samples)


def _make_dataset_from_dict(data_dict, coords=None):
    dataset_vars = {}
    for k, v in data_dict.items():
        if (
            isinstance(v, tuple)
            and len(v) == 2
            and isinstance(v[0], (list, str))
        ):
            dims, data = v
            dataset_vars[k] = DataArray(data, dims=dims)
        else:
            dataset_vars[k] = DataArray(v)
    return Dataset(dataset_vars, coords=coords)


def compare_runs(
    run1: az.InferenceData,
    run2: az.InferenceData,
    labels: List[str],
    outdir: str,
) -> Dataset:
    """
    Compare two InferenceData runs and return a Dataset with differences.

    Parameters
    ----------
    run1 : az.InferenceData
        First run to compare
    run2 : az.InferenceData
        Second run to compare

    Returns
    -------
    Dataset
        Dataset containing the differences between the two runs
    """
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # Ensure both runs have the same variables
    common_vars = set(run1.posterior.data_vars) & set(run2.posterior.data_vars)
    if not common_vars:
        raise ValueError("No common variables found in the two runs.")

    # Plot density
    fig = az.plot_density(
        [run1.posterior, run2.posterior],
        data_labels=labels,
        shade=0.2,
        hdi_prob=0.94,
    )
    plt.suptitle("Density Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{outdir}/density_comparison.png")
    plt.close()

    # Get summaries
    summary1 = az.summary(run1)
    summary2 = az.summary(run2)

    # Compute difference in summaries
    common_vars = summary1.index.intersection(summary2.index)
    diff = summary1.loc[common_vars] - summary2.loc[common_vars]
    diff.to_csv(f"{outdir}/summary_diff.csv")

    print("Summary Differences:")
    print(diff)
