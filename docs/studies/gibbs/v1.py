import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from spectrum import aryule, arma2psd
from tqdm.auto import trange


# ------------------------------
# Helper functions for periodogram and PSD
# ------------------------------
def compute_periodogram(signal, dt):
    """Compute periodogram using FFT."""
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=dt)[: n // 2]
    periodogram = (np.abs(fft_vals[: n // 2]) ** 2) * dt / n
    return freqs, periodogram


def compute_estimated_psd(AR, P, T, NFFT):
    """
    Compute one-sided PSD from an AR model using arma2psd.
    """
    psd_full = arma2psd(AR, [1], P, T, NFFT)
    one_sided = psd_full[: NFFT // 2 + 1]
    return one_sided


# ------------------------------
# Plotting function for posterior-predictive PSD + signal
# ------------------------------
def plot_progress(iteration, t, data, current_signal, dt, current_AR, current_P, fs,
                  true_signal_pdgrm, true_psd):
    """
    Plot the data periodogram, estimated signal periodogram, true signal periodogram,
    true noise PSD, and estimated noise PSD.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    # Periodogram of data and current estimated signal
    freqs, data_periodogram = compute_periodogram(data, dt)
    _, signal_periodogram = compute_periodogram(current_signal, dt)

    # NFFT for PSD computation
    NFFT = len(freqs) * 2
    psd_freqs = np.linspace(0, fs / 2, NFFT // 2 + 1)

    est_noise_psd = compute_estimated_psd(current_AR, current_P, T=fs, NFFT=NFFT)

    # Plotting on log-log axes
    ax.loglog(freqs, data_periodogram, label='Data', color='gray', alpha=0.5)
    ax.loglog(freqs, signal_periodogram, label='Estimated Signal', color='green', linestyle='-.', alpha=0.25)
    ax.loglog(freqs, true_signal_pdgrm, label='True Signal', color='green', alpha=0.25, lw=2)
    ax.loglog(psd_freqs, true_psd, label='True Noise', color='red', alpha=0.25, lw=2)
    ax.loglog(psd_freqs, est_noise_psd, label='Estimated Noise', color='red', linestyle='--', alpha=0.25)

    ax.set_xlim(left=1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Iteration: {iteration}")
    ax.legend()
    ax.set_ylim(bottom=1e-6, top=1e4)

    plt.tight_layout()
    plt.savefig(f"psd_signal_itr{iteration:02d}.png")
    plt.close()


# ------------------------------
# NumPyro model with Whittle likelihood
# ------------------------------
def sine_whittle_model(data, t, dt, fs, noise_psd, a_bounds, f_bounds):
    """
    A NumPyro model for a sine wave with a Whittle likelihood.

    Parameters:
      data (array): Observed data.
      t (array): Time grid.
      dt (float): Sampling interval.
      fs (float): Sampling frequency.
      noise_psd (tuple): (psd_freqs, psd_values) for the noise.
      a_bounds (tuple): Lower and upper bound for amplitude.
      f_bounds (tuple): Lower and upper bound for frequency.
    """
    a = numpyro.sample("a", dist.Uniform(a_bounds[0], a_bounds[1]))
    f = numpyro.sample("f", dist.Uniform(f_bounds[0], f_bounds[1]))

    model_signal = a * jnp.sin(2 * jnp.pi * f * t)
    residuals = data - model_signal

    # Compute periodogram of residuals.
    n = residuals.shape[0]
    fft_vals = jnp.fft.fft(residuals)
    freqs = jnp.fft.fftfreq(n, d=dt)[: n // 2]
    periodogram = (jnp.abs(fft_vals[: n // 2]) ** 2) * dt / n

    # Unpack noise PSD and interpolate.
    psd_freqs, psd_values = noise_psd
    interp_psd = jnp.interp(freqs, psd_freqs, psd_values)

    logl = -0.5 * jnp.sum(periodogram / interp_psd + jnp.log(interp_psd))
    numpyro.factor("whittle", logl)


def run_numpyro_mcmc(data, t, dt, fs, noise_psd, a_bounds, f_bounds,
                     num_warmup=500, num_samples=1000, init_params=None,
                     rng_key=random.PRNGKey(0)):
    """
    Runs NumPyro MCMC using NUTS.
    Returns samples grouped by chain.
    """
    kernel = NUTS(sine_whittle_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, jit_model_args=True)
    mcmc.run(rng_key, data=data, t=t, dt=dt, fs=fs, noise_psd=noise_psd,
             a_bounds=a_bounds, f_bounds=f_bounds, init_params=init_params)
    # Return chain samples (shape: [n_chains, n_samples])
    return mcmc.get_samples(group_by_chain=True)


# ------------------------------
# Plotting trace and histogram (rotated) for NumPyro samples
# ------------------------------
def plot_trace_with_hist(samples_dict, true_values, bounds, labels, iteration):
    """
    Plot trace plots (one line per chain) with an adjacent histogram (rotated 90°)
    for each parameter.

    Parameters:
      samples_dict (dict): Dictionary with keys (e.g., 'a', 'f') and arrays of shape (n_chains, n_samples).
      true_values (list): True values for parameters.
      bounds (list of tuples): [(min, max), ...] for each parameter.
      labels (list of str): Parameter names.
      iteration (int): Current Gibbs iteration (for file naming).
    """
    ndim = len(labels)
    fig, axes = plt.subplots(ndim, 2, figsize=(8, 2.5 * ndim),
                             gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})

    for i in range(ndim):
        # Left: Trace plot (each chain)
        ax_trace = axes[i, 0]
        # samples_dict[label] is of shape (n_chains, n_samples)
        chains = samples_dict[labels[i]]
        n_chains, n_samples = chains.shape
        for chain in range(n_chains):
            ax_trace.plot(np.arange(n_samples), chains[chain], color="k", alpha=0.3)
        ax_trace.axhline(true_values[i], color="red", linestyle="--", label="True")
        ax_trace.set_ylabel(labels[i])
        ax_trace.set_ylim(bounds[i])
        if i == ndim - 1:
            ax_trace.set_xlabel("MCMC Iteration")
        else:
            ax_trace.set_xticklabels([])

        # Right: Histogram (rotated 90°)
        ax_hist = axes[i, 1]
        all_samples = chains.flatten()
        ax_hist.hist(all_samples, bins=30, orientation="horizontal",
                     color="gray", alpha=0.7)
        ax_hist.axhline(true_values[i], color="red", linestyle="--")
        ax_hist.set_ylim(bounds[i])
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])

    plt.suptitle(f"Trace and Posterior Histogram (Gibbs Iteration {iteration})", y=1.02)
    plt.tight_layout()
    plt.savefig(f"trace_hist_itr{iteration:02d}.png")
    plt.close()


# ------------------------------
# Functions for JSD computation and plotting
# ------------------------------
def compute_jsd(file1, file2, bin_ranges, n_bins=30):
    """
    Compute the Jensen–Shannon Divergence (JSD) between two posterior sample files.

    Parameters:
      file1 (str): Path to the first sample file.
      file2 (str): Path to the baseline sample file.
      bin_ranges (tuple): ((a_min, a_max), (f_min, f_max)) for histogram binning.
      n_bins (int): Number of bins.

    Returns:
      float: Sum of JSD for a and f.
    """
    samples1 = np.loadtxt(file1)
    samples2 = np.loadtxt(file2)

    def jsd_param(p, q, rng):
        p_hist, _ = np.histogram(p, bins=n_bins, range=rng, density=True)
        q_hist, _ = np.histogram(q, bins=n_bins, range=rng, density=True)
        p_hist += 1e-10
        q_hist += 1e-10
        m = 0.5 * (p_hist + q_hist)
        return 0.5 * (scipy.stats.entropy(p_hist, m) + scipy.stats.entropy(q_hist, m))

    jsd_a = jsd_param(samples1[:, 0], samples2[:, 0], bin_ranges[0])
    jsd_f = jsd_param(samples1[:, 1], samples2[:, 1], bin_ranges[1])
    return jsd_a + jsd_f


def plot_jsd(n_bins=30):
    """
    Loads saved sample files, computes the JSD (against the final iteration),
    and plots JSD vs. Gibbs iteration.
    """
    sample_files = sorted(glob.glob("samples_*.txt"), key=lambda x: int(x.split("_")[1].split(".")[0]))
    if not sample_files:
        raise FileNotFoundError("No sample files found.")

    baseline_file = sample_files[-1]
    # Determine bin ranges from all samples
    all_samples = []
    for fpath in sample_files:
        all_samples.append(np.loadtxt(fpath))
    all_samples = np.vstack(all_samples)
    a_range = (all_samples[:, 0].min(), all_samples[:, 0].max())
    f_range = (all_samples[:, 1].min(), all_samples[:, 1].max())
    bin_ranges = (a_range, f_range)

    jsd_values = []
    iterations = []
    for fpath in sample_files[:-1]:
        iteration = int(fpath.split("_")[1].split(".")[0])
        jsd_val = compute_jsd(fpath, baseline_file, bin_ranges, n_bins=n_bins)
        jsd_values.append(jsd_val)
        iterations.append(iteration)

    plt.figure(figsize=(5, 3))
    plt.plot(iterations, jsd_values, marker="o", linestyle="-")
    plt.xlabel("Gibbs Iteration")
    plt.ylabel("JSD")
    plt.title("JSD of [a, f] posterior vs. final posterior")
    plt.savefig("jsd_vs_iterations.png")
    plt.close()


# ------------------------------
# Main Gibbs Sampling Workflow
# ------------------------------
# --- Synthetic Data Generation ---
np.random.seed(42)

# True sine-wave parameters
true_a = 1.0  # Amplitude
true_f = 4.0  # Frequency (Hz)

# AR filter coefficients for noise generation
a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
true_rho = 0.1

# Data length and time grid
n_samples = 1024
fs = 10  # Hz
dt = 1.0 / fs
t = np.linspace(0, (n_samples - 1) * dt, n_samples)

# Generate true signal and noise
true_signal = true_a * np.sin(2 * np.pi * true_f * t)
noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
data = true_signal + noise

# Estimate true noise PSD (using aryule on noise)
order = 4
AR_est, P_est, _ = aryule(noise, order)
NFFT = len(data)
true_psd = compute_estimated_psd(AR_est, P_est, T=fs, NFFT=NFFT)
psd_freqs = np.linspace(0, fs / 2, len(true_psd))
noise_psd_default = (psd_freqs, true_psd)

# Compute true signal periodogram (for plotting)
_, true_signal_pdgrm = compute_periodogram(true_signal, dt)

# Parameter bounds for the sine model
a_bounds = (true_a - 0.5, true_a + 0.5)
f_bounds = (true_f - 1, true_f + 1)

# ------------------------------
# Gibbs Sampler Setup
# ------------------------------
n_gibbs = 30
n_mcmc = 100
n_warmup = 50

# Initialize current signal estimate (starting with zeros)
current_signal = np.zeros_like(data)
current_noise_psd = noise_psd_default

# For saving posterior samples and tracking median estimates
posterior_samples_files = []
# Starting initial guesses at the center of the bounds
last_median_a = (a_bounds[0] + a_bounds[1]) / 2
last_median_f = (f_bounds[0] + f_bounds[1]) / 2

# Define labels and bounds for trace plots (order: 'a', 'f')
param_labels = ['a', 'f']
param_bounds = [a_bounds, f_bounds]
# We'll use the true values for overlay in trace plots
true_params = [true_a, true_f]

for gibbs_iter in trange(n_gibbs, desc='Gibbs Sampling'):
    # 1. Update noise model using current residuals
    residuals = data - current_signal
    AR_est, P_est, _ = aryule(residuals, order)
    current_psd = compute_estimated_psd(AR_est, P_est, T=fs, NFFT=NFFT)
    psd_freqs = np.linspace(0, fs / 2, len(current_psd))
    current_noise_psd = (psd_freqs, current_psd)

    # 2. Sample [a, f] using NumPyro with the Whittle likelihood.
    init_params = {"a": last_median_a, "f": last_median_f}
    rng_key = random.PRNGKey(gibbs_iter)
    samples_chain = run_numpyro_mcmc(data, t, dt, fs, current_noise_psd,
                                     a_bounds, f_bounds,
                                     num_warmup=n_warmup, num_samples=n_mcmc,
                                     init_params=init_params,
                                     rng_key=rng_key)
    # samples_chain is a dict with keys 'a' and 'f', each of shape (n_chains, n_samples)
    # Also get aggregated samples for updating the median.
    a_samples = samples_chain["a"].flatten()
    f_samples = samples_chain["f"].flatten()

    # Update median estimates (could also pick a random sample)
    last_median_a = np.median(a_samples)
    last_median_f = np.median(f_samples)

    # Update current signal estimate using the median parameters.
    current_signal = last_median_a * np.sin(2 * np.pi * last_median_f * t)

    # 3. Save posterior samples into a text file: two columns [a, f] (aggregated over chains)
    current_samples = np.column_stack([a_samples, f_samples])
    filename = f"samples_{gibbs_iter + 1:02d}.txt"
    np.savetxt(filename, current_samples)
    posterior_samples_files.append(filename)

    # 4. Plot trace and histogram for the current MCMC run.
    # For trace plots, we use the chain samples from NumPyro.
    # We need to form a dictionary with keys 'a' and 'f' where each is shape (n_chains, n_samples).
    samples_for_plot = {"a": samples_chain["a"], "f": samples_chain["f"]}
    plot_trace_with_hist(samples_for_plot, true_params, param_bounds, param_labels, gibbs_iter + 1)

    # 5. Plot posterior-predictive PSD and signal vs. true PSD and signal.
    # Here, we plot using the current signal estimate and current noise PSD.
    # (true_signal_pdgrm and true_psd were computed before.)
    plot_progress(gibbs_iter + 1, t, data, current_signal, dt,
                  current_AR=AR_est, current_P=P_est, fs=fs,
                  true_signal_pdgrm=true_signal_pdgrm, true_psd=true_psd)

# ------------------------------
# Final JSD Computation and Plotting
# ------------------------------
plot_jsd(n_bins=30)

print("Gibbs sampling complete!")
print("JSD plot saved as 'jsd_vs_iterations.png'.")
