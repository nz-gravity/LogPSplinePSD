import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy.signal
from spectrum import aryule, arma2psd
from tqdm.auto import trange
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import glob
import imageio
import os


OUT = 'outdir'

os.makedirs(OUT, exist_ok=True)

# ------------------------------
# Set random seed for reproducibility
# ------------------------------
np.random.seed(42)


# ======================
# Helper functions
# ======================
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

    Parameters:
      AR (array): AR coefficients (including the leading 1)
      P (float): Estimated white noise variance (rho)
      T (float): Sampling frequency in Hz
      NFFT (int): Number of FFT points for arma2psd
      sides (str): 'twosided' (default) or 'centerdc'

    Returns:
      one_sided (array): One-sided PSD estimate.
    """
    # Here we use B = [1] since we have no MA part.
    psd_full = arma2psd(AR, [1], P, T, NFFT)
    # For even NFFT, one-sided PSD is the first NFFT//2+1 points.
    one_sided = psd_full[: NFFT // 2 + 1]
    return one_sided


def plot_progress(iteration, t, data, current_signal, dt, current_AR, current_P, fs):
    """
    Plot time series and PSD comparisons for the current Gibbs iteration.

    The periodogram subplot now uses log–log axes and includes:
      - The data periodogram.
      - The periodogram of the estimated signal.
      - The true noise PSD (from the AR filter coefficients).
      - The estimated noise PSD from the current AR estimate.
    """
    fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))

    # --- PSD Plot (log-log) ---
    # Compute periodograms
    freqs, data_periodogram = compute_periodogram(data, dt)
    _, signal_periodogram = compute_periodogram(current_signal, dt)

    # Choose NFFT for PSD computation (e.g., twice the number of periodogram points)
    NFFT = len(freqs) * 2
    # Frequency vector for the PSD (Hz)
    psd_freqs = np.linspace(0, fs / 2, NFFT // 2 + 1)

    # Compute PSDs using arma2psd via our helper function
    # true_noise_psd = compute_estimated_psd(true_AR, true_P, T=fs, NFFT=NFFT)
    est_noise_psd = compute_estimated_psd(current_AR, current_P, T=fs, NFFT=NFFT)

    # Plot on a log-log scale
    axs.loglog(freqs, data_periodogram, label='Data', color='gray', alpha=0.5)
    axs.loglog(freqs, signal_periodogram, label='Estimated Signal', color='green', linestyle='-.')
    axs.loglog(freqs, true_signal_pdgrm, label='True Signal', color='green')
    axs.loglog(psd_freqs, true_psd, label='True Noise', color='red')
    axs.loglog(psd_freqs, est_noise_psd, label='Estimated Noise', color='red', linestyle='--')
    axs.set_xlim(left=1)

    axs.set_xlabel("Frequency [Hz]")
    axs.set_ylabel("PSD")
    axs.set_title(f"Iteration: {iteration}")
    axs.legend()
    axs.set_ylim(bottom=10 ** -6, top=10 ** 4)

    plt.tight_layout()
    plt.savefig(f"{OUT}/itr{iteration:02d}.png")


def plot_trace_with_hist(samples, true_values, bounds, labels, itr):
    """
    Plots MCMC trace plots with adjacent rotated histograms.

    Parameters:
        samples (np.ndarray): MCMC samples of shape (n_samples, n_walkers, ndim).
        true_values (list): True parameter values for reference.
        bounds (list of tuples): [(ymin, ymax), ...] bounds for each parameter.
        labels (list of str): Labels for each parameter.
    """
    ndim = samples.shape[2]
    fig, axes = plt.subplots(ndim, 2, figsize=(6, 2 * ndim), gridspec_kw={"width_ratios": [3, 1], "wspace": 0})

    for i in range(ndim):
        trace_ax = axes[i, 0]
        hist_ax = axes[i, 1]

        # Trace plot
        trace_ax.plot(samples[:, :, i], "k", alpha=0.3)
        trace_ax.axhline(true_values[i], color="r", linestyle="--", label="True value")
        trace_ax.set_xlim(0, samples.shape[0])
        trace_ax.set_ylabel(labels[i])
        trace_ax.yaxis.set_label_coords(-0.15, 0.5)
        trace_ax.set_ylim(bounds[i])  # Set shared y-limits

        # Histogram (rotated 90 degrees)
        hist_ax.hist(samples[:, :, i].flatten(), bins=30, orientation="horizontal", color="gray", alpha=0.7)
        hist_ax.axhline(true_values[i], color="r", linestyle="--")
        hist_ax.set_ylim(bounds[i])  # Match y-limits
        hist_ax.set_xticks([])  # Remove x-ticks

    axes[-1, 0].set_xlabel("Iteration")  # Label only the last trace plot
    fig.suptitle(f"Gibbs {itr+1}")

    plt.tight_layout()
    plt.savefig(f"{OUT}/trace_{itr+1}.png")
    plt.close()


def create_gif(image_pattern, output_filename, duration=0.5, repeat_last=3, cleanup=False):
    """
    Creates a GIF from a sequence of images.

    Parameters:
        image_pattern (str): Pattern for image filenames, e.g., "itr{:02d}.png".
        output_filename (str): Name of the output GIF file.
        duration (float): Duration per frame in seconds.
        repeat_last (int): Number of times to repeat the last frame for emphasis.
        cleanup (bool): Whether to delete individual image files after GIF creation.
    """
    # Collect image filenames
    image_files = sorted([f for f in os.listdir() if f.startswith(image_pattern.split("{")[0])])

    if not image_files:
        raise FileNotFoundError(f"No images found matching pattern '{image_pattern}'.")

    # Load images
    images = [imageio.imread(f) for f in image_files]

    # Add the last image multiple times for emphasis
    images.extend([images[-1]] * repeat_last)

    # Save as GIF
    imageio.mimsave(output_filename, images, duration=duration)

    # Clean up individual image files if needed
    if cleanup:
        for filename in image_files:
            os.remove(filename)

    print(f"GIF created: {output_filename}")


def compute_jsd(file1, file2, bin_ranges, n_bins=30):
    """
    Computes the Jensen-Shannon Divergence (JSD) between two posterior samples.

    Parameters:
        file1 (str): Path to the first sample file.
        file2 (str): Path to the second (baseline) sample file.
        bin_ranges (tuple): Min and max range for binning.
        n_bins (int): Number of histogram bins.

    Returns:
        float: The sum of JSDs for both parameters.
    """
    samples1 = np.loadtxt(file1)
    samples2 = np.loadtxt(file2)

    def jsd(p, q):
        p_hist, _ = np.histogram(p, bins=n_bins, range=bin_ranges, density=True)
        q_hist, _ = np.histogram(q, bins=n_bins, range=bin_ranges, density=True)

        p_hist += 1e-10  # Avoid log(0)
        q_hist += 1e-10
        m = 0.5 * (p_hist + q_hist)

        return 0.5 * (scipy.stats.entropy(p_hist, m) + scipy.stats.entropy(q_hist, m))

    jsd_a = jsd(samples1[:, 0], samples2[:, 0])
    jsd_f = jsd(samples1[:, 1], samples2[:, 1])

    return jsd_a + jsd_f


def plot_jsd(n_bins=30):
    """
    Computes and plots the JSD evolution across Gibbs iterations.

    Parameters:
        n_bins (int): Number of bins for histogram binning.
    """
    sample_files = sorted(glob.glob("samples_*.txt"), key=lambda x: int(x.split("_")[1].split(".")[0]))

    if not sample_files:
        raise ValueError("No sample files found.")

    baseline_file = sample_files[-1]  # Last file is the final reference posterior

    # Determine bin ranges from all samples
    all_samples = np.vstack([np.loadtxt(f) for f in sample_files])
    bin_ranges = (all_samples.min(axis=0), all_samples.max(axis=0))

    jsd_values = []
    iteration_numbers = []

    for sample_file in sample_files[:-1]:  # Exclude the last iteration
        iteration = int(sample_file.split("_")[1].split(".")[0])
        jsd_value = compute_jsd(sample_file, baseline_file, bin_ranges, n_bins)

        jsd_values.append(jsd_value)
        iteration_numbers.append(iteration)

    # Plot JSD vs. iteration number
    plt.figure(figsize=(5, 3))
    plt.plot(iteration_numbers, jsd_values, marker="o", linestyle="-")
    plt.xlabel("Gibbs Iteration")
    plt.ylabel("JSD")
    plt.title("JSD of [a, f] posterior vs. final posterior")
    plt.savefig("jsd_vs_iterations.png")
    plt.close()



# ======================
# Generate synthetic data
# ======================
# True sine-wave parameters
true_a = 20.0  # Amplitude
true_f = 25.0  # Frequency in Hz

# AR filter coefficients for noise generation (for lfilter, filter numerator = [1])
a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
# True noise white variance (rho) used for scaling the PSD
true_rho = 0.1

# Data length and time grid
n_samples = 1024  # number of samples
fs = 100  # sampling frequency in Hz
dt = 1.0 / fs
t = np.linspace(0, (n_samples - 1) * dt, n_samples)

# Generate the true sine-wave signal
true_signal = true_a * np.sin(2 * np.pi * true_f * t)

# Generate AR noise using scipy.signal.lfilter and white noise
noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
# Create observed data as the sum of signal and noise
data = true_signal + noise

freqs, data_periodogram = compute_periodogram(data, dt)
NFFT = len(freqs) * 2
order = 4
AR_est, P_est, k_est = aryule(noise, order)
true_psd = compute_estimated_psd(AR=AR_est, P=P_est, T=fs, NFFT=NFFT)
_, true_signal_pdgrm = compute_periodogram(true_signal, dt)

posterior_samples = []

# ======================
# Gibbs sampler setup
# ======================
n_gibbs = 1  # Number of Gibbs iterations
n_mcmc = 5000  # MCMC steps per Gibbs iteration
burnin = 200  # MCMC burn-in steps
n_walkers = 32

# Initialize the current signal estimate (starting with zero)
current_signal = np.zeros_like(data)
# In our Gibbs loop we estimate the AR model from the residuals.
# Here we assume an AR model of order 4 (matching len(a_coeff)-1).
order = 4
# Initialize current noise PSD estimate as a dictionary (AR and P).
current_psd = {'AR': np.array([1.0] + [0.0] * order), 'P': 1.0}

# Parameter bounds for the sine-wave parameters (amplitude, frequency)
a_bounds = (true_a - 5, true_a + 5)
f_bounds = (true_f - 5, true_f + 5)

# To save the final MCMC samples from the last Gibbs iteration for the credible interval
samples = None

# ======================
# Gibbs sampling loop
# ======================
for gibbs_iter in trange(n_gibbs, desc='Gibbs'):

    # 1. Estimate the noise PSD using aryule (Yule–Walker) on the residuals.
    residuals = data - current_signal
    # aryule returns (AR, P, order); AR includes the leading 1.
    AR_est, P_est, k_est = aryule(residuals, order)
    current_psd = {'AR': AR_est, 'P': P_est}


    # 2. Sample sine-wave parameters (amplitude and frequency) using MCMC.
    def log_prob(params):
        a, f = params
        # Enforce parameter bounds.
        if not (a_bounds[0] <= a <= a_bounds[1] and f_bounds[0] <= f <= f_bounds[1]):
            return -np.inf

        # Generate sine-wave model.
        model_signal = a * np.sin(2 * np.pi * f * t)
        res_model = data - model_signal

        # Compute periodogram of the residuals.
        freqs, periodogram = compute_periodogram(res_model, dt)
        NFFT = len(freqs) * 2
        # Use the current AR estimate to compute the noise PSD.
        model_psd = compute_estimated_psd(current_psd['AR'], current_psd['P'], T=fs, NFFT=NFFT)
        psd_freqs = np.linspace(0, fs / 2, len(model_psd))
        # Interpolate the model PSD to match the periodogram frequencies.
        interp_psd = np.interp(freqs, psd_freqs, model_psd)

        # Compute the Whittle likelihood (up to an additive constant)
        logl = -0.5 * np.sum(periodogram / interp_psd + np.log(interp_psd))
        return logl

    # Initialize walkers near the true parameters with small random offsets.
    initial = np.array([true_a, true_f]) + 0.1 * np.random.randn(n_walkers, 2)
    sampler = emcee.EnsembleSampler(n_walkers, 2, log_prob)

    # Run the MCMC sampler.
    sampler.run_mcmc(initial, n_mcmc, progress=False)

    # Extract posterior samples after discarding burn-in.
    samples = sampler.get_chain(discard=burnin, flat=True)
    a_post = samples[:, 0]
    f_post = samples[:, 1]

    # save posterior samples
    np.savetxt(f'{OUT}/samples_{gibbs_iter}.txt', samples)

    a_random = np.random.choice(a_post)
    f_random = np.random.choice(f_post)

    # Update current signal estimate using the median MCMC parameters.
    current_signal = a_random * np.sin(2 * np.pi * f_random * t)

    # Save the final MCMC samples from the last Gibbs iteration.
    final_mcmc_samples = samples.copy()

    # Plot progress at this Gibbs iteration.
    plot_progress(gibbs_iter + 1, t, data, current_signal, dt,
                  current_AR=current_psd['AR'], current_P=current_psd['P'],
                  fs=fs)
    plot_trace_with_hist(sampler.get_chain(discard=burnin), [true_a, true_f], [a_bounds, f_bounds],
                         ['a', 'f'], gibbs_iter)

print("\nGibbs sampling complete!")
