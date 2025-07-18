import os
from typing import List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def compare_results(
    run1: az.InferenceData,
    run2: az.InferenceData,
    labels: List[str],
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)

    # Ensure both runs have the same variables
    common_vars = set(run1["posterior"].data_vars) & set(
        run2["posterior"].data_vars
    )
    if not common_vars:
        raise ValueError("No common variables found in the two runs.")

    ### 1) Plot density
    az.plot_density(
        [run1["posterior"], run2["posterior"]],
        data_labels=labels,
        shade=0.2,
        hdi_prob=0.94,
    )
    plt.suptitle("Density Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{outdir}/density_comparison.png")
    plt.close()

    ### 2) Plot ESS
    ess1 = _get_ess(run1)
    ess2 = _get_ess(run2)
    plt.figure(figsize=(8, 5))
    plt.boxplot([ess1, ess2], labels=labels, showfliers=False)
    plt.ylabel("Effective Sample Size (ESS)")
    plt.title("Comparison of ESS Distributions")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{outdir}/ess_comparison.png")
    plt.close()

    ### 3) Get summaries
    summary1 = az.summary(run1)
    summary2 = az.summary(run2)

    # Compute difference in summaries
    common_vars = summary1.index.intersection(summary2.index)
    diff = summary1.loc[common_vars] - summary2.loc[common_vars]
    diff.to_csv(f"{outdir}/summary_diff.csv")

    print("Summary Differences:")
    print(diff)


def _get_ess(run: az.InferenceData) -> np.array:
    """
    Get the effective sample size (ESS) for each variable in the run.
    """
    ess = az.ess(run)
    values = ess.to_array().values.flatten()
    return values[~np.isnan(values)]  # remove NaNs
