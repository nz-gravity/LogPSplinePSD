"""Compare the impact of different numbers of knots on IAE using MH sampler"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from log_psplines.example_datasets import ARData
from log_psplines.mcmc import run_mcmc
from spectrum import pyule
import jax
import arviz as az
from log_psplines.arviz_utils.from_arviz import  get_posterior_psd


jax.config.update("jax_enable_x64", True)


def yule_walker_psd(time_series: np.ndarray, order: int, fs: float = 1.0):
    p = pyule(time_series, order, sampling=fs, scale_by_freq=False)
    yule_psd = np.array(p.psd)
    freqs = np.array(p.frequencies())
    return yule_psd[1:], freqs[1:]


def compute_iae(prediction: np.ndarray, truth: np.ndarray, freqs: np.ndarray) -> float:
    """Compute Integrated Average Error using trapezoidal integration"""
    abs_error = np.abs(prediction - truth)
    # Use trapezoidal rule for integration over frequency
    iae = np.trapz(abs_error, freqs) / (freqs[-1] - freqs[0])
    return iae



def run_analysis_for_knots(data: ARData, n_knots: int, use_parametric_model: bool = True,
                          outdir: str = "output") -> dict:
    """Run analysis for a specific number of knots and return IAE results"""

    parametric_model = None
    if use_parametric_model:
        parametric_model = yule_walker_psd(data.ts.y, order=data.order, fs=data.fs)[0]

    kawrgs = dict(
        pdgrm=data.periodogram,
        parametric_model=parametric_model,
        n_knots=n_knots,
        n_samples=3000,  # Reduced for faster computation
        n_warmup=2000,
        rng_key=0,
        knot_kwargs=dict(method="uniform"),
    )

    # Create output directory
    run_outdir = f"{outdir}/mh_out"
    Path(f"{outdir}/mh_out").mkdir(parents=True, exist_ok=True)
    inference_fn = f"{run_outdir}/inference_data.nc"
    if os.path.exists(inference_fn):
        inf_obj = az.from_netcdf(inference_fn)
    else:
        inf_obj = run_mcmc(**kawrgs, sampler="mh", outdir=f"{outdir}/mh_out")

    median_mh = get_posterior_psd(inf_obj)[1]

    # Compute IAE
    iae_mh = compute_iae(median_mh, data.psd_theoretical, data.freqs)

    results = {
        'n_knots': n_knots,
        'iae': iae_mh
    }

    print(f"Knots: {n_knots}, IAE: {iae_mh:.6f}")
    return results


def plot_comparison_for_knots(results_dict: dict, outdir: str = "output"):
    """Create comparison plots for different numbers of knots, overlaying multiple cases"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {"with_parametric": "tab:blue", "without_parametric": "tab:orange"}

    for label, results_list in results_dict.items():
        valid_results = [r for r in results_list if not np.isnan(r['iae'])]
        knots = [r['n_knots'] for r in valid_results]
        iae_values = [r['iae'] for r in valid_results]

        ax.plot(
            knots,
            iae_values,
            'o-',
            color=colors[label],
            label=label.replace("_", " ").title(),
            linewidth=2,
            markersize=6
        )

    ax.set_xlabel('Number of Knots')
    ax.set_ylabel('Integrated Average Error')
    ax.set_title('IAE vs Number of Knots (MH Sampler)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{outdir}/iae_vs_knots_comparison.png", dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main analysis function"""
    order, fs = 4, 512.0
    data = ARData(order=order, duration=2.0, fs=fs, sigma=1.0, seed=42)

    knot_range = [5, 10, 15, 20]  # Reduced for faster computation

    # Store results for both cases
    all_results = {}

    for use_parametric_model in [False, True]:
        label = "without_parametric" if not use_parametric_model else "with_parametric"
        print(f"\n{'='*50}")
        print(f"Running analysis {label}")
        print(f"{'='*50}")

        results_list = []
        for n_knots in knot_range:
            print(f"\nProcessing {n_knots} knots...")
            outdir = f"output/{label}/knots_{n_knots}"

            result = run_analysis_for_knots(
                data,
                n_knots,
                use_parametric_model=use_parametric_model,
                outdir=outdir
            )
            results_list.append(result)

        all_results[label] = results_list

        # Save results
        df = pd.DataFrame(results_list)
        Path(f"output/{label}").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"output/{label}/iae_results.csv", index=False)
        print(f"\nResults saved to output/{label}/iae_results.csv")

    # Now plot both curves together
    plot_comparison_for_knots(all_results, outdir="output")

if __name__ == "__main__":
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    main()