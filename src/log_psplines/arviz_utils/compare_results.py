from __future__ import annotations

import os
from typing import List, cast

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arviz import ess

try:
    from arviz_base import convert_to_datatree as convert_to_dataset
except ImportError:
    try:
        from arviz import convert_to_dataset  # type: ignore[no-redef]
    except ImportError:
        convert_to_dataset = None  # type: ignore[assignment]

try:
    from arviz.utils import _var_names, get_coords
except ImportError:
    # arviz >= 1.0.0 removed these helpers; provide simple replacements
    def get_coords(data, coords):  # type: ignore[misc]
        if coords:
            return data.sel(coords)
        return data

    def _var_names(var_names, data, filter_vars):  # type: ignore[misc]
        if var_names is not None:
            return list(var_names)
        return list(data.data_vars)


from numpy.typing import NDArray

from ..logger import logger


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
    ess_result = az.ess(run)
    # arviz >= 1.0.0 returns a DataTree; iterate over its data_vars directly
    all_vals: list[np.ndarray] = []
    for v in ess_result.data_vars.values():
        arr = np.asarray(v).flatten()
        all_vals.append(arr)
    if not all_vals:
        return np.array([], dtype=np.float64)
    values = np.concatenate(all_vals)
    return values[~np.isnan(values)]  # remove NaNs


def plot_ess_evolution(
    idata, n_points=50, ess_threshold=400, ax=None, color="tab:blue"
):
    coords: dict[str, object] = {}
    # Extract posterior group — works for both old Dataset and new DataTree
    if hasattr(idata, "posterior"):
        _posterior = idata.posterior
    elif hasattr(idata, "data_vars"):
        _posterior = idata
    elif hasattr(idata, "children") and "posterior" in idata.children:
        _posterior = idata["posterior"]
    else:
        _posterior = idata
    data = get_coords(_posterior, coords)
    var_names = _var_names(None, data, None)
    n_draws = data.sizes["draw"]
    n_samples = n_draws * data.sizes["chain"]

    # Setup draw slicing
    first_draw = data.draw.values[0]
    xdata = np.linspace(n_samples / n_points, n_samples, n_points)
    draw_divisions = np.linspace(
        n_draws // n_points, n_draws, n_points, dtype=int
    )

    # Compute ESS for each draw slice
    ess_dataset = xr.concat(
        [
            ess(
                data.sel(draw=slice(first_draw, first_draw + draw_div)),
                var_names=var_names,
                relative=False,
                method="bulk",
            )
            for draw_div in draw_divisions
        ],
        dim="ess_dim",
    )

    ess_tail_dataset = xr.concat(
        [
            ess(
                data.sel(draw=slice(first_draw, first_draw + draw_div)),
                var_names=var_names,
                relative=False,
                method="tail",
            )
            for draw_div in draw_divisions
        ],
        dim="ess_dim",
    )

    # Convert datasets to (n_vars, n_points) arrays
    def _dataset_to_ndarray(dataset):
        return np.concatenate(
            [
                v.values.reshape(v.shape[0], -1).T
                for v in dataset.data_vars.values()
            ],
            axis=0,
        )

    x = _dataset_to_ndarray(ess_dataset)
    xtail = _dataset_to_ndarray(ess_tail_dataset)

    for xi, xtaili in zip(x, xtail):
        ax.plot(xdata, xi, alpha=0.5, color=color)
        ax.plot(xdata, xtaili, alpha=0.5, color=color, linestyle="dotted")
    ax.axhline(
        ess_threshold,
        linestyle="--",
        color="gray",
        label=f"ESS = {ess_threshold}",
    )
    ax.set_xlabel("Total number of draws")
    ax.set_ylabel("ESS")
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(xdata), max(xdata))
    ax.set_title("ESS Evolution (Bulk & Tail)")
