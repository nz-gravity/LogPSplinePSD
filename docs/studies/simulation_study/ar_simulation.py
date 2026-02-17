"""Compare the impact of a parametric model vs no parametric model"""

import jax
import matplotlib.pyplot as plt
import morphz
import numpy as np
from spectrum import pyule

from log_psplines.arviz_utils.compare_results import compare_results
from log_psplines.example_datasets import ARData
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm

jax.config.update("jax_enable_x64", True)


def yule_walker_psd(time_series: np.ndarray, order: int, fs: float = 1.0):
    p = pyule(time_series, order, sampling=fs, scale_by_freq=False)
    yule_psd = np.array(p.psd)
    freqs = np.array(p.frequencies())
    return yule_psd[1:], freqs[1:]


def run_analysis(
    data: ARData, use_parametric_model: bool = True, outdir: str = "output"
):
    parametric_model = None
    if use_parametric_model:
        parametric_model = yule_walker_psd(
            data.ts.y, order=data.order, fs=data.fs
        )[0]

    kawrgs = dict(
        pdgrm=data.periodogram,
        parametric_model=parametric_model,
        n_knots=15,
        n_samples=5000,
        n_warmup=3000,
        rng_key=0,
        knot_kwargs=dict(method="uniform"),
    )

    inference_nuts = run_mcmc(**kawrgs, outdir=f"{outdir}/nuts_out")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(
        data.freqs,
        data.psd_theoretical,
        color="k",
        linestyle="--",
        label="True PSD",
        zorder=10,
    )
    plot_pdgrm(
        idata=inference_nuts,
        ax=ax,
        model_color="tab:orange",
        model_label="NUTS",
        data_label="_",
        show_knots=True,
    )

    ax.set_xscale("linear")
    fig.savefig(
        f"{outdir}/compare.png",
        transparent=False,
        bbox_inches="tight",
        dpi=300,
    )


# with and without parametric model
order, fs = 4, 512.0
data = ARData(order=order, duration=2.0, fs=fs, sigma=1.0, seed=42)
true_psd = data.psd_theoretical

results = {}
for use_parametric_model in [False, True]:
    label = "without_parametric"
    if use_parametric_model:
        label = "with_parametric"
    outdir = f"output/{label}"
    run_analysis(
        data, use_parametric_model=use_parametric_model, outdir=outdir
    )
    results[label] = f"{outdir}/nuts_out/inference_data.nc"

compare_results(
    results["without_parametric"],
    results["with_parametric"],
    labels=["No parametric", "Parametric"],
    outdir="output/parametric_compare",
)
