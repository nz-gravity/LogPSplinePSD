import matplotlib.pyplot as plt

from log_psplines.arviz_utils.from_arviz import get_spline_model
from log_psplines.example_datasets import ARData
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm

ar4 = ARData(order=4, duration=2.0, fs=512.0, sigma=1.0, seed=42)
kawrgs = dict(
    pdgrm=ar4.periodogram,
    n_knots=15,
    n_samples=2500,
    n_warmup=1000,
    rng_key=0,
    knot_kwargs=dict(method="uniform"),
)
inference_mh = run_mcmc(**kawrgs, sampler="mh", outdir="mh_out")
inference_nuts = run_mcmc(**kawrgs, sampler="nuts", outdir="nuts_out")

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(
    ar4.freqs,
    ar4.psd_theoretical,
    color="k",
    linestyle="--",
    label="True PSD",
    zorder=10,
)
plot_pdgrm(
    idata=inference_mh,
    ax=ax,
    model_label="MH",
    model_color="tab:blue",
    show_knots=False,
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
fig.savefig("demo.png", transparent=True, bbox_inches="tight", dpi=300)

get_spline_model(inference_nuts).plot_basis()
