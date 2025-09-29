"""Compare the impact of different numbers of knots on IAE using MH sampler (with repeats)"""

import os.path
from pathlib import Path

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import UnivariateSpline
from spectrum import pyule

from log_psplines.arviz_utils.from_arviz import get_posterior_psd
from log_psplines.example_datasets import ARData
from log_psplines.mcmc import run_mcmc

np.random.seed(0)

order, fs = 4, 512.0


jax.config.update("jax_enable_x64", True)


def yule_walker_psd(time_series: np.ndarray, order: int, fs: float = 1.0):
    p = pyule(time_series, order, sampling=fs, scale_by_freq=False)
    yule_psd = np.array(p.psd)
    freqs = np.array(p.frequencies())
    return yule_psd[1:], freqs[1:]


def compute_iae(
    prediction: np.ndarray, truth: np.ndarray, freqs: np.ndarray
) -> float:
    """Compute Integrated Average Error using trapezoidal integration"""
    abs_error = np.abs(prediction - truth)
    iae = scipy.integrate.trapezoid(abs_error, freqs) / (freqs[-1] - freqs[0])
    return iae


def run_analysis_for_knots(
    data: ARData,
    n_knots: int,
    use_parametric_model: bool = True,
    outdir: str = "output",
    n_repeats: int = 5,
) -> list[dict]:
    """Run analysis for a specific number of knots multiple times and return IAE results"""

    parametric_model = None
    if use_parametric_model:
        parametric_model = yule_walker_psd(
            data.ts.y, order=data.order, fs=data.fs
        )[0]

    results = []

    for rep in range(n_repeats):
        kawrgs = dict(
            pdgrm=data.periodogram,
            parametric_model=parametric_model,
            n_knots=n_knots,
            n_samples=3000,  # Reduced for faster computation
            n_warmup=2000,
            rng_key=np.random.randint(
                0, 1_000_000
            ),  # randomize seed for repeats
            knot_kwargs=dict(method="uniform"),
        )

        run_outdir = f"{outdir}/mh_out/rep_{rep}"
        Path(run_outdir).mkdir(parents=True, exist_ok=True)
        inference_path = f"{run_outdir}/inference_data.nc"

        if os.path.exists(inference_path):
            print(f"Loading existing results from {inference_path}")
            inf_obj = az.from_netcdf(inference_path)
        else:
            print(f"{inference_path} not found. Running MCMC...")
            inf_obj = run_mcmc(**kawrgs, sampler="mh", outdir=run_outdir)

        median_mh = get_posterior_psd(inf_obj)[1]

        # Compute IAE
        iae_mh = compute_iae(median_mh, data.psd_theoretical, data.freqs)

        result = {
            "n_knots": n_knots,
            "rep": rep,
            "iae": iae_mh,
        }
        results.append(result)
        print(f"Knots: {n_knots}, Rep: {rep}, IAE: {iae_mh:.6f}")

    return results


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def darken(color, amount=0.6):
    """Darken a matplotlib color"""
    r, g, b = mcolors.to_rgb(color)
    return (r * amount, g * amount, b * amount)


def plot_comparison_for_knots(results_dict: dict, outdir: str = "output"):
    """Create box plots for different numbers of knots comparing with/without parametric."""

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    colors = {
        "with_parametric": "#2ca02c",  # green
        "without_parametric": "#ff7f0e",  # orange
    }
    dark_colors = {k: darken(v, 0.5) for k, v in colors.items()}

    # Collect all knot values (union across groups)
    all_knots = sorted(
        set(
            k
            for results in results_dict.values()
            for k in pd.DataFrame(results)["n_knots"].unique()
        )
    )

    # Positioning for side-by-side boxes
    base_positions = np.arange(len(all_knots)) + 1
    offset = 0.05
    width = 0.35
    alpha_box = 0.6

    # Plot each group
    for i, (label, results_list) in enumerate(results_dict.items()):
        df = pd.DataFrame(results_list)

        positions = base_positions + (i - 0.5) * 2 * offset

        data = [df.loc[df["n_knots"] == k, "iae"].values for k in all_knots]

        ax.boxplot(
            data,
            positions=positions,
            widths=width,
            notch=False,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor=colors[label], color=colors[label], alpha=alpha_box
            ),
            whiskerprops=dict(color=colors[label], alpha=alpha_box),
            capprops=dict(color=colors[label], alpha=alpha_box),
            medianprops=dict(color=colors[label], alpha=alpha_box),
        )

    # Formatting
    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(k) for k in all_knots])
    ax.set_xlabel("Number of Knots")
    ax.set_ylabel("Integrated Average Error")
    # ax.set_title("IAE vs Number of Knots (MH Sampler)")

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors["with_parametric"],
            lw=6,
            alpha=alpha_box,
            label="With Parametric",
        ),
        Line2D(
            [0],
            [0],
            color=colors["without_parametric"],
            lw=6,
            alpha=alpha_box,
            label="Without Parametric",
        ),
        # Line2D([0], [0], color="gray", lw=2, label="Mean (dashed)", ls="--"),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(
        f"{outdir}/iae_vs_knots_comparison_box.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig


def plot_comparison_for_knots_bands(
    results_dict: dict, outdir: str = "output", smoothing_factor: float = 0.1
):
    """Create band plots for different numbers of knots comparing with/without parametric."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    colors = {
        "with_parametric": "#2ca02c",  # green
        "without_parametric": "#ff7f0e",  # orange
    }

    # Collect all knot values (union across groups)
    all_knots = sorted(
        set(
            k
            for results in results_dict.values()
            for k in pd.DataFrame(results)["n_knots"].unique()
        )
    )

    # Create smooth x values for interpolation
    x_smooth = np.linspace(min(all_knots), max(all_knots), 200)

    # Plot each group
    for label, results_list in results_dict.items():
        df = pd.DataFrame(results_list)

        # Calculate statistics for each knot value
        medians = []
        q1_lower, q1_upper = [], []  # 1σ (68%): 16th-84th percentiles
        q2_lower, q2_upper = [], []  # 2σ (95%): 2.5th-97.5th percentiles
        q3_lower, q3_upper = [], []  # 3σ (99.7%): 0.15th-99.85th percentiles

        for k in all_knots:
            data = df.loc[df["n_knots"] == k, "iae"].values

            medians.append(np.median(data))

            # 1σ (68%)
            q1_lower.append(np.percentile(data, 16))
            q1_upper.append(np.percentile(data, 84))

            # 2σ (95%)
            q2_lower.append(np.percentile(data, 2.5))
            q2_upper.append(np.percentile(data, 97.5))

            # 3σ (99.7%)
            q3_lower.append(np.percentile(data, 0.15))
            q3_upper.append(np.percentile(data, 99.85))

        # Create smooth interpolations
        # Note: s parameter controls smoothing (0 = exact interpolation, higher = more smoothing)
        median_smooth = UnivariateSpline(
            all_knots, medians, s=smoothing_factor
        )(x_smooth)

        q1_lower_smooth = UnivariateSpline(
            all_knots, q1_lower, s=smoothing_factor
        )(x_smooth)
        q1_upper_smooth = UnivariateSpline(
            all_knots, q1_upper, s=smoothing_factor
        )(x_smooth)

        q2_lower_smooth = UnivariateSpline(
            all_knots, q2_lower, s=smoothing_factor
        )(x_smooth)
        q2_upper_smooth = UnivariateSpline(
            all_knots, q2_upper, s=smoothing_factor
        )(x_smooth)

        q3_lower_smooth = UnivariateSpline(
            all_knots, q3_lower, s=smoothing_factor
        )(x_smooth)
        q3_upper_smooth = UnivariateSpline(
            all_knots, q3_upper, s=smoothing_factor
        )(x_smooth)

        # Plot smooth bands (from outermost to innermost)
        color = colors[label]

        # 3σ band (lightest)
        ax.fill_between(
            x_smooth,
            q3_lower_smooth,
            q3_upper_smooth,
            color=color,
            alpha=0.15,
            label=f'{label.replace("_", " ").title()} 3σ',
        )

        # # 2σ band (medium)
        # ax.fill_between(x_smooth, q2_lower_smooth, q2_upper_smooth,
        #                 color=color, alpha=0.25)
        #
        # # 1σ band (darkest)
        # ax.fill_between(x_smooth, q1_lower_smooth, q1_upper_smooth,
        #                 color=color, alpha=0.4)

        # Median line
        ax.plot(
            x_smooth,
            median_smooth,
            color=color,
            linewidth=2,
            label=f'{label.replace("_", " ").title()} Median',
        )

        # # Optional: add points at actual knot locations
        # ax.scatter(all_knots, medians, color=color, s=20, zorder=10, alpha=0.8)

    # Formatting
    ax.set_xlabel("Number of Knots")
    ax.set_ylabel("Integrated Average Error")
    ax.set_xticks(all_knots)

    # Legend
    ax.legend(frameon=False, loc="best")

    # Optional: add grid for easier reading

    plt.tight_layout()
    fig.savefig(
        f"{outdir}/iae_vs_knots_comparison_bands_smooth.png",
        dpi=300,
        bbox_inches="tight",
    )
    return fig


def rerun(knot_range, data, n_repeats, all_results):

    for use_parametric_model in [False, True]:
        label = (
            "without_parametric"
            if not use_parametric_model
            else "with_parametric"
        )
        print(f"\n{'=' * 50}")
        print(f"Running analysis {label}")
        print(f"{'=' * 50}")

        results_list = []
        for n_knots in knot_range:
            print(f"\nProcessing {n_knots} knots...")
            outdir = f"output/{label}/knots_{n_knots}"
            data = ARData(
                order=order,
                duration=2.0,
                fs=fs,
                sigma=1.0,
                seed=np.random.randint(0, 1_000_000),
            )
            knot_results = run_analysis_for_knots(
                data,
                n_knots,
                use_parametric_model=use_parametric_model,
                outdir=outdir,
                n_repeats=n_repeats,
            )
            results_list.extend(knot_results)

        all_results[label] = results_list

        # Save results
        df = pd.DataFrame(results_list)
        Path(f"output/{label}").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"output/{label}/iae_results.csv", index=False)
        print(f"\nResults saved to output/{label}/iae_results.csv")


def main():
    """Main analysis function"""

    knot_range = list(range(5, 21, 1))  # Knots from 5 to 20 in steps of 1
    n_repeats = 3

    # Store results for both cases
    all_results = {
        "with_parametric": [],
        "without_parametric": [],
    }

    # if not (os.path.exists("output/with_parametric/iae_results.csv") and os.path.exists("output/without_parametric/iae_results.csv")):

    # rerun(knot_range, None, n_repeats, all_results)

    # read in all results from csv to ensure consistency
    for label in all_results.keys():
        df = pd.read_csv(f"output/{label}/iae_results.csv")
        all_results[label] = df.to_dict(orient="records")

    # Now plot both curves together with error bars
    plot_comparison_for_knots(all_results, outdir="output")
    plot_comparison_for_knots_bands(all_results, outdir="output")


if __name__ == "__main__":
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    main()
