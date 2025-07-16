import numpy as np
import scipy

from log_psplines.datatypes import Timeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
from log_psplines.example_datasets import ARData

np.random.seed(0)


ar4 = ARData(order=4, duration=8.0, fs=1024.0, sigma=1.0, seed=42)
mcmc, spline_model = run_mcmc(
    ar4.periodogram, n_knots=30, num_samples=250, num_warmup=1000, rng_key=0
)

fig, ax = plot_pdgrm(ar4.periodogram, spline_model, samples["weights"])
ax.set_axis_off()
fig.savefig("demo.png", transparent=True, bbox_inches="tight", dpi=300)
