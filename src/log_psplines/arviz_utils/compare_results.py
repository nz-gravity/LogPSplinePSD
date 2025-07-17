import os
from typing import List
import matplotlib.pyplot as plt
import arviz as az


def compare_results(
        run1: az.InferenceData,
        run2: az.InferenceData,
        labels: List[str],
        outdir: str,
):
    os.makedirs(outdir, exist_ok=True)

    # Ensure both runs have the same variables
    common_vars = set(run1["posterior"].data_vars) & set(run2["posterior"].data_vars)
    if not common_vars:
        raise ValueError("No common variables found in the two runs.")

    # Plot density
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

    # Get summaries
    summary1 = az.summary(run1)
    summary2 = az.summary(run2)

    # Compute difference in summaries
    common_vars = summary1.index.intersection(summary2.index)
    diff = summary1.loc[common_vars] - summary2.loc[common_vars]
    diff.to_csv(f"{outdir}/summary_diff.csv")

    print("Summary Differences:")
    print(diff)
