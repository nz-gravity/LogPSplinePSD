import os
from typing import List, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from ..logger import logger
from ..plotting.plot_ess_evolution import plot_ess_evolution


def compare_results(
    run1: Union[az.InferenceData, str],
    run2: Union[az.InferenceData, str],
    labels: List[str],
    outdir: str,
    colors: List[str] = None,
):
    if colors is None:
        colors = ["tab:blue", "tab:orange"]
    os.makedirs(outdir, exist_ok=True)

    if isinstance(run1, str):
        run1 = az.from_netcdf(run1)
    if isinstance(run2, str):
        run2 = az.from_netcdf(run2)

    # Ensure both runs have the same variables
    common_vars = set(run1["posterior"].data_vars) & set(
        run2["posterior"].data_vars
    )
    if not common_vars:
        raise ValueError("No common variables found in the two runs.")

    ### 1) Plot density
    try:
        az.plot_density(
            [run1["posterior"], run2["posterior"]],
            data_labels=labels,
            shade=0.2,
            hdi_prob=0.94,
            colors=colors,
        )
        plt.suptitle("Density Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{outdir}/density_comparison.png")
    except Exception as exc:  # pragma: no cover - plotting safeguard
        logger.warning(f"Density comparison failed: {exc}")
    finally:
        plt.close()

    ### 2) Plot ESS
    ess1 = _get_ess(run1)
    ess2 = _get_ess(run2)
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [ess1, ess2],
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=colors[0]),
        medianprops=dict(color="black"),
    )
    for patch, color in zip(plt.gca().artists, colors):
        patch.set_facecolor(color)
    plt.ylabel("Effective Sample Size (ESS)")
    plt.title("Comparison of ESS Distributions")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{outdir}/ess_comparison.png")
    plt.close()

    ### 3) Plot ESS evolution
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    plot_ess_evolution(
        run1, ax=ax, n_points=50, ess_threshold=400, color=colors[0]
    )
    plot_ess_evolution(
        run2, ax=ax, n_points=50, ess_threshold=400, color=colors[1]
    )
    # ax legend 2 columns, run1/run2, plus bulk/tail ESS markers
    ax.legend(
        loc="upper left",
        handles=[
            plt.Line2D([0], [0], color=colors[0], lw=1, label=labels[0]),
            plt.Line2D([0], [0], color=colors[1], lw=1, label=labels[1]),
            plt.Line2D([0], [0], color="black", lw=1, label="Bulk ESS"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                lw=1,
                linestyle="dotted",
                label="Tail ESS",
            ),
        ],
        labels=[labels[0], labels[1], "Bulk ESS", "Tail ESS"],
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.5,
        handletextpad=0.5,
    )

    plt.tight_layout()
    plt.savefig(f"{outdir}/ess_evolution.png", dpi=300, bbox_inches="tight")

    ### 3) Get summaries
    summary1 = az.summary(run1)
    summary2 = az.summary(run2)

    # Compute difference in summaries
    common_vars = summary1.index.intersection(summary2.index)
    diff = summary1.loc[common_vars] - summary2.loc[common_vars]
    diff.to_csv(f"{outdir}/summary_diff.csv")

    logger.info("Summary Differences:")
    logger.info(f"\n{diff}")


def _get_ess(run: az.InferenceData) -> np.array:
    """
    Get the effective sample size (ESS) for each variable in the run.
    """
    ess = az.ess(run)
    values = ess.to_array().values.flatten()
    return values[~np.isnan(values)]  # remove NaNs
