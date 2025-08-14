import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from log_psplines.datatypes import Periodogram
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import plot_pdgrm
from log_psplines.preprocessing.line_locator import estimate_psd_with_lines, plot_line_summary, plot_psd_analysis, analyze_psd_lines

from data import load_lvk_psd, get_lvk_noise_realisation

lvk_psd = load_lvk_psd()
lvk_noise = get_lvk_noise_realisation(sampling_frequency=4096.0, duration=32.0)
freqs = lvk_noise.freqs

fmin, fmax = 10, 2048

running_median, is_line_bin, psd_model, line_details = estimate_psd_with_lines(
    lvk_noise.freqs, lvk_noise.power, fmin=fmin, fmax=fmax
)

plt.figure(figsize=(10, 6))
plt.loglog(lvk_psd.freqs, lvk_psd.power, label='LIGO PSD', color='k')
plt.loglog(lvk_noise.freqs, lvk_noise.power, label='Observed PSD', color='blue')
plt.plot(lvk_noise.freqs, running_median, label='Running Median', color='orange')
for f_start, f_end, f0, bandwidth, max_ratio in line_details:
    plt.axvspan(f_start, f_end, color='red', alpha=0.3, label='Line Interval')

plt.xlabel('Frequency  (Hz)')
plt.ylabel('Power Spectral Density')
plt.xlim(fmin, fmax)
plt.savefig('LVK_noise.png')
