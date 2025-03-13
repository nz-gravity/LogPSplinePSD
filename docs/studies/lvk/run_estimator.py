import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from lvk_data import LVKData

from log_psplines.datasets import Periodogram
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm, plot_trace

fs = 4096
fmin, fmax = 10, 2048

lvk_data = LVKData.load()
pdgrm_pwr = lvk_data.psds[0]
welch_psd = lvk_data.welch_psd
freqs = jnp.array(lvk_data._freq)


# cut all data outside frange
mask = (freqs > fmin) & (freqs < fmax)
freqs = freqs[mask]
pdgrm_pwr = pdgrm_pwr[mask]
welch_psd = welch_psd[mask]

# for stability we move these power values higher (so not e-20, but more like e+1)
min_power = min(pdgrm_pwr)
pdgrm_pwr = jnp.array(pdgrm_pwr / min_power)
welch_psd = jnp.array(welch_psd / min_power)


analysis_pdgrm = Periodogram(freqs, pdgrm_pwr)
mcmc, spline_model = run_mcmc(
    analysis_pdgrm,
    num_warmup=500,
    num_samples=1000,
    n_knots=30,
    parametric_model=welch_psd,
    frac_log=0.3,
)
samples = mcmc.get_samples()
fig, ax = plot_pdgrm(
    analysis_pdgrm,
    spline_model,
    samples["weights"],
    show_knots=True,
    yscalar=min_power,
)
ax.loglog(
    freqs,
    np.array(welch_psd, dtype=np.float64) * min_power,
    color="k",
    label="Welch PSD",
    alpha=0.3,
)
fig.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False)
plt.savefig("lvk_noise_and_splines.pdf", bbox_inches="tight", dpi=300)
plot_trace(mcmc, "traceplot.png")

fig, ax = plot_pdgrm(
    spline_model=spline_model,
    weights=samples["weights"],
    show_knots=True,
    yscalar=min_power,
    use_parametric_model=False,
    freqs=freqs,
)
plt.savefig("just_splines.pdf", bbox_inches="tight", dpi=300)
