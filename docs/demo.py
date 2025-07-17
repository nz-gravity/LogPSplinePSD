from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
from log_psplines.example_datasets import ARData
import matplotlib.pyplot as plt

ar4 = ARData(order=4, duration=2.0, fs=512.0, sigma=1.0, seed=42)
inference_data = run_mcmc(
    ar4.periodogram, n_knots=15, n_samples=250, n_warmup=1000, rng_key=0, sampler='mh',
    knot_kwargs=dict(frac_uniform=1.0)
)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(ar4.freqs, ar4.psd_theoretical, color="k", linestyle="--", label="True PSD", zorder=10)
plot_pdgrm(idata=inference_data, ax=ax)
ax.set_xscale('linear')
ax.set_axis_off()
fig.savefig("demo.png", transparent=True, bbox_inches="tight", dpi=300)
