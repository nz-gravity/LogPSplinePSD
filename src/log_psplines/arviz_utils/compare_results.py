import os
from typing import List, cast

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..logger import logger
from ..plotting.plot_ess_evolution import plot_ess_evolution


def _load_inference_data(run: az.InferenceData | str) -> az.InferenceData:
    if isinstance(run, str):
        loaded = az.from_netcdf(run)
        return cast(az.InferenceData, loaded)
    return run


def compare_results(
    run1: az.InferenceData | str,
    run2: az.InferenceData | str,
    labels: List[str],
    outdir: str,
    colors: List[str] | None = None,
):
    if colors is None:
        colors = ["tab:blue", "tab:orange"]
    os.makedirs(outdir, exist_ok=True)

    run1_idata = _load_inference_data(run1)
    run2_idata = _load_inference_data(run2)

    # Ensure both runs have the same variables
    common_vars = set(run1_idata["posterior"].data_vars) & set(
        run2_idata["posterior"].data_vars
    )
    if not common_vars:
        raise ValueError("No common variables found in the two runs.")

    ### 1) Plot density
    try:
        az.plot_density(
            [run1_idata["posterior"], run2_idata["posterior"]],
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
    ess1 = _get_ess(run1_idata)
    ess2 = _get_ess(run2_idata)
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
        if hasattr(patch, "set_facecolor"):
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
        run1_idata, ax=ax, n_points=50, ess_threshold=400, color=colors[0]
    )
    plot_ess_evolution(
        run2_idata, ax=ax, n_points=50, ess_threshold=400, color=colors[1]
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
    summary1 = az.summary(run1_idata)
    summary2 = az.summary(run2_idata)

    # Compute difference in summaries
    common_vars = summary1.index.intersection(summary2.index)
    diff = summary1.loc[common_vars] - summary2.loc[common_vars]
    diff.to_csv(f"{outdir}/summary_diff.csv")

    logger.info("Summary Differences:")
    logger.info(f"\n{diff}")


def _get_ess(run: az.InferenceData) -> NDArray[np.float64]:
    """
    Get the effective sample size (ESS) for each variable in the run.
    """
    ess = az.ess(run)
    values = ess.to_array().values.flatten()
    return values[~np.isnan(values)]  # remove NaNs
