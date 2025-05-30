import numpy as np
import scipy

from log_psplines.datasets import Timeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm

np.random.seed(0)

n_samples, fs = 1024, 100
t = np.linspace(0, (n_samples - 1) * 1 / fs, n_samples)
noise = scipy.signal.lfilter(
    b=[1],
    a=[1, -2.2137, 2.9403, -2.1697, 0.9606],
    x=np.random.randn(n_samples),
)
mock_pdgrm = Timeseries(t, noise).standardise().to_periodogram().highpass(5)

mcmc, spline_model = run_mcmc(
    mock_pdgrm, n_knots=30, num_samples=250, num_warmup=1000, rng_key=0
)
samples = mcmc.get_samples()

fig, ax = plot_pdgrm(mock_pdgrm, spline_model, samples["weights"])
ax.set_axis_off()
fig.savefig("demo.png", transparent=True, bbox_inches="tight", dpi=300)
