import data
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
import matplotlib.pyplot as plt

true_psd = data.load_lvk_psd().highpass(10.0)
pdgrm = data.get_lvk_noise_realisation()
pdgrm = pdgrm.highpass(10.0)
fig, ax = plot_pdgrm(pdgrm)
ax.loglog(true_psd.freqs, true_psd.power, color="k", label="True PSD")
plt.savefig("lvk_noise.png", bbox_inches="tight")

samples, spline_model = run_mcmc(pdgrm, num_warmup=150, num_samples=100)
fig, ax = plot_pdgrm(pdgrm, spline_model, samples['weights'],show_knots=False)
ax.loglog(true_psd.freqs, true_psd.power, color="k", label="True PSD")
plt.savefig("lvk_noise_and_splines.png", bbox_inches="tight")
