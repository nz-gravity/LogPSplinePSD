import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.interpolate import BSpline
from scipy.signal.windows import tukey

CLEAN = True


# comment this out for faster but less accurate float32 computations
# We might be able to do the analysis in float32 if we rescale all the data
# jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore")

import os
import pickle

np.random.seed(1234)
numpyro.set_platform("cpu")

results_dir = "simple_2d_results"
os.makedirs(results_dir, exist_ok=True)


def noise_PSD_AE(f: np.ndarray, TDI="TDI2"):
    """
    Takes in frequency, spits out TDI2 A channel, same as E channel
    is equal and constant arm length approx.
    """
    f = np.asarray(f)
    f = np.where(f == 0, 1e-10, f)
    L = 2.5e9
    c = 299758492
    x = 2 * np.pi * (L / c) * f

    Spm = (
        (3e-15) ** 2
        * (1 + ((4e-4) / f) ** 2)
        * (1 + (f / (8e-3)) ** 4)
        * (1 / (2 * np.pi * f)) ** 4
    )

    Sop = (15e-12) ** 2 * (1 + ((2e-3) / f) ** 4) * (2 * np.pi * f / c) ** 2

    S_val = 2 * Spm * (3 + 2 * np.cos(x) + np.cos(2 * x)) + Sop * (
        2 + np.cos(x)
    )

    if TDI == "TDI1":
        S = 8 * (np.sin(x) ** 2) * S_val
    elif TDI == "TDI2":
        S = 32 * np.sin(x) ** 2 * np.sin(2 * x) ** 2 * S_val
    else:
        raise ValueError("TDI must be either TDI1 or TDI2")

    return S


tmax = 365 * 24 * 60 * 60  # 1 year

# Orbital modulation parameters

A_ORBIT = 0.8  # Amplitude of the modulation
F_ORBIT = 4.0 / tmax  # Frequency of the modulation

# The modulation function now includes only the orbital cosine term
modulation = lambda t: 1 + A_ORBIT * np.cos(2 * np.pi * F_ORBIT * t)

# Simulation parameters

fmin = 1e-4
fmax = 1e-3
fs = 2 * fmax
delta_t = 1 / fs
n_data = 2 ** int(np.log(tmax / delta_t) / np.log(2))

print(f"Number of data points (n_data): {n_data}")
print(f"Sampling frequency (fs): {fs} Hz")
print(f"Delta T: {delta_t}")
print(f"Maximum frequency (f_nyquist): {fs / 2} Hz")

# Generate frequency array and PSD
freq = np.fft.rfftfreq(n_data, delta_t)[1:]
PSD = noise_PSD_AE(freq)


# Generate noise in frequency domain
T = n_data * delta_t
std_f = np.sqrt(PSD * T / 2)
noise_fourier_components = std_f * (
    np.random.randn(len(freq)) + 1j * np.random.randn(len(freq))
)

# Convert to time domain
ts = np.fft.irfft(np.concatenate(([0], noise_fourier_components)))

scale = np.std(ts)
ts = (ts - np.mean(ts)) / scale


# Generate modulated timeseries
t = np.arange(0, n_data * delta_t, delta_t)
modulated_ts = ts * modulation(t)

# Create time-varying PSD for comparison
Nt = 10
t_indices = np.linspace(0, n_data - 1, Nt, dtype=int)
t_downsampled = t[t_indices]
time_grid, freq_grid = np.meshgrid(t_downsampled, freq)
time_varying_PSD = modulation(time_grid) ** 2 * noise_PSD_AE(freq_grid)


class Periodogram:
    def __init__(self, freqs, power):
        self.freqs = np.array(freqs)
        self.power = np.array(power)
        self.n = len(freqs)


def segment_timeseries(
    timeseries, time_array, n_segments=52, freq_min=1e-5, freq_max=None
):
    """
    Segment timeseries and compute periodograms with corrected normalization
    """
    n_total = len(timeseries)
    segment_length = n_total // n_segments
    segment_periodograms, segment_midpoints = [], []

    for j in range(n_segments):
        start_idx = j * segment_length
        end_idx = min(start_idx + segment_length, n_total)
        segment_data = timeseries[start_idx:end_idx]
        segment_time = time_array[start_idx:end_idx]

        t_mid = (segment_time[0] + segment_time[-1]) / 2.0
        segment_midpoints.append(t_mid)

        n_seg = len(segment_data)
        window = tukey(n_seg, alpha=0.25)
        windowed_data = segment_data * window

        # Corrected window normalization
        window_norm = np.sum(window**2) / n_seg

        fft_data = np.fft.rfft(windowed_data)
        freqs = np.fft.rfftfreq(n_seg, delta_t)[1:]

        # Corrected periodogram normalization
        power = (2 * delta_t / (n_seg * window_norm)) * (
            np.abs(fft_data[1:]) ** 2
        )

        freq_mask = freqs >= freq_min
        if freq_max is not None:
            freq_mask &= freqs <= freq_max
        segment_periodograms.append(
            Periodogram(freqs[freq_mask], power[freq_mask])
        )

    return segment_periodograms, np.array(segment_midpoints)


def create_bspline_basis(knots, degree, x_eval):
    """Create B-spline basis matrix"""
    knots_extended = np.concatenate(
        [np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)]
    )

    n_basis = len(knots) + degree - 1
    basis_matrix = np.zeros((len(x_eval), n_basis))

    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        bspline = BSpline(knots_extended, coeffs, degree)
        basis_matrix[:, i] = bspline(x_eval)

    return basis_matrix


def create_penalty_matrix(n_basis, order=2):
    """Create penalty matrix for smoothness"""
    if order == 0:
        return np.eye(n_basis)

    D = np.eye(n_basis)
    for _ in range(order):
        D_new = np.zeros((D.shape[0] - 1, D.shape[1]))
        for i in range(D.shape[0] - 1):
            D_new[i] = D[i + 1] - D[i]
        D = D_new

    return D.T @ D


def create_log_spaced_knots(n_knots, freq_min, freq_max):
    """Create logarithmically spaced knots"""
    if n_knots < 2:
        raise ValueError("Need at least 2 knots")

    freq_min_log = max(freq_min, 1e-12)
    knot_freqs = np.logspace(
        np.log10(freq_min_log), np.log10(freq_max), n_knots
    )
    knots_normalized = (knot_freqs - freq_min) / (freq_max - freq_min)

    return np.clip(knots_normalized, 0.0, 1.0), knot_freqs


# Segment the modulated timeseries
n_segments = 52
segment_periodograms, segment_midpoints = segment_timeseries(
    modulated_ts, t, n_segments=n_segments, freq_min=1e-5
)

samples_file = os.path.join(results_dir, "orbital_samples.pkl")
posts_file = os.path.join(results_dir, "posterior_data.pkl")

# Set up B-splines with more knots for better fit
ref_periodogram = segment_periodograms[0]
seg_freq = ref_periodogram.freqs
n_knots, degree, penalty_order = 20, 3, 2  # Increased knots for better fit

knots_normalized, knot_freqs = create_log_spaced_knots(
    n_knots, ref_periodogram.freqs[0], ref_periodogram.freqs[-1]
)

basis_matrix = create_bspline_basis(
    knots_normalized,
    degree,
    (ref_periodogram.freqs - ref_periodogram.freqs[0])
    / (ref_periodogram.freqs[-1] - ref_periodogram.freqs[0]),
)

n_basis = basis_matrix.shape[1]
penalty_matrix = create_penalty_matrix(n_basis, penalty_order)


@jax.jit
def orbital_modulated_likelihood_exponential(
    weights,
    orbital_params,
    segment_log_periodograms,
    segment_midpoints,
    basis_matrix,
):
    a, phi, omega = orbital_params
    log_base_psd = basis_matrix @ weights
    modulation_factors = 1.0 + a * jnp.cos(
        2 * jnp.pi * omega * segment_midpoints + phi
    )
    modulation_factors = jnp.clip(modulation_factors, 1e-8, None)

    log_modulated_psd = (
        log_base_psd[None, :] + 2.0 * jnp.log(modulation_factors)[:, None]
    )

    # likelihood: -log(S) - P/S
    modulated_psd = jnp.exp(log_modulated_psd)
    segment_powers = jnp.exp(segment_log_periodograms)
    log_terms = -log_modulated_psd - segment_powers / modulated_psd
    return jnp.sum(log_terms)


def bayesian_orbital_model_corrected(
    segment_log_periodograms,
    segment_midpoints,
    basis_matrix,
    penalty_matrix,
    alpha_phi=1.0,
    beta_phi=1.0,
    alpha_delta=1e-4,
    beta_delta=1e-4,
    expected_omega=None,
):
    delta = numpyro.sample("delta", dist.Gamma(alpha_delta, beta_delta))
    phi = numpyro.sample("phi", dist.Gamma(alpha_phi, delta * beta_phi))

    n_weights = basis_matrix.shape[1]
    weights = numpyro.sample(
        "weights", dist.Normal(0, 1).expand([n_weights]).to_event(1)
    )

    a = numpyro.sample("orbital_amplitude", dist.Beta(3, 2))
    phi_orbital = numpyro.sample("orbital_phase", dist.Uniform(0, 2 * jnp.pi))

    if expected_omega is None:
        expected_omega = F_ORBIT
    omega_std = expected_omega * 0.05  # Tighter constraint
    omega = numpyro.sample(
        "orbital_frequency", dist.Normal(expected_omega, omega_std)
    )

    orbital_params = jnp.array([a, phi_orbital, omega])

    quad_form = jnp.dot(weights, jnp.dot(penalty_matrix, weights))
    log_prior_weights = 0.5 * n_weights * jnp.log(phi) - 0.5 * phi * quad_form

    # Use corrected exponential likelihood
    log_likelihood = orbital_modulated_likelihood_exponential(
        weights,
        orbital_params,
        segment_log_periodograms,
        segment_midpoints,
        basis_matrix,
    )

    numpyro.factor("log_prior_weights", log_prior_weights)
    numpyro.factor("orbital_likelihood", log_likelihood)
    numpyro.deterministic("log_posterior", log_prior_weights + log_likelihood)


# Prepare data for MCMC
segment_powers = np.array([seg.power for seg in segment_periodograms])
segment_log_periodograms_2d = jnp.log(segment_powers)
segment_midpoints_jax = jnp.array(segment_midpoints)
basis_matrix_jax = jnp.array(basis_matrix)
penalty_matrix_jax = jnp.array(penalty_matrix)


if os.path.exists(samples_file) and os.path.exists(posts_file) and not CLEAN:
    print("Loading existing results...")
    with open(samples_file, "rb") as f:
        orbital_samples = pickle.load(f)
    with open(posts_file, "rb") as f:
        post_data = pickle.load(f)
    times = post_data["times"]
    S_mod_median = post_data["S_mod_median"]
    time_varying_PSD_at_seg = post_data["time_varying_PSD_at_seg"]
    seg_freq = post_data["seg_freq"]
    t_downsampled = post_data["t_downsampled"]
    segment_midpoints = post_data["segment_midpoints"]
else:
    nuts_kernel = NUTS(bayesian_orbital_model_corrected)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=1)
    print("Running MCMC...")
    mcmc.run(
        jax.random.PRNGKey(123),
        segment_log_periodograms=segment_log_periodograms_2d,
        segment_midpoints=segment_midpoints_jax,
        basis_matrix=basis_matrix_jax,
        penalty_matrix=penalty_matrix_jax,
        expected_omega=F_ORBIT,
    )
    orbital_samples = mcmc.get_samples()


# Compute base PSD median from samples (even if loaded)
weights_samples = np.array(orbital_samples["weights"])
basis_np = np.array(basis_matrix_jax)

log_base_samples = weights_samples @ basis_np.T
base_psd_samples = np.exp(log_base_samples)
S_med_base = np.median(base_psd_samples, axis=0)

# Compute posterior predictions
a_samples = np.array(orbital_samples["orbital_amplitude"])
phi_samples = np.array(orbital_samples["orbital_phase"])
omega_samples = np.array(orbital_samples["orbital_frequency"])
# Choose segment midpoints as time grid
times = np.array(segment_midpoints)
# Compute modulation factor per sample, per time
phase_terms = (
    2 * np.pi * omega_samples[:, None] * times[None, :] + phi_samples[:, None]
)
mods = 1.0 + a_samples[:, None] * np.cos(phase_terms)
mods2 = mods**2
# Broadcast into (n_samps, n_segments, n_freq)
S_mod_samples = base_psd_samples[:, None, :] * mods2[:, :, None]
# Posterior median time-varying PSD
S_mod_median = np.median(S_mod_samples, axis=0)

# Compute time-varying PSD at segment frequencies
seg_freq_indices = np.searchsorted(freq, ref_periodogram.freqs)
time_varying_PSD_at_seg = time_varying_PSD[seg_freq_indices, :].T


if not (os.path.exists(samples_file) and os.path.exists(posts_file)):
    # Compute posterior predictions if not loaded
    a_samples = np.array(orbital_samples["orbital_amplitude"])
    phi_samples = np.array(orbital_samples["orbital_phase"])
    omega_samples = np.array(orbital_samples["orbital_frequency"])
    # Choose segment midpoints as time grid
    times = np.array(segment_midpoints)
    # Compute modulation factor per sample, per time
    phase_terms = (
        2 * np.pi * omega_samples[:, None] * times[None, :]
        + phi_samples[:, None]
    )
    mods = 1.0 + a_samples[:, None] * np.cos(phase_terms)
    mods2 = mods**2
    # Broadcast into (n_samps, n_segments, n_freq)
    S_mod_samples = base_psd_samples[:, None, :] * mods2[:, :, None]
    # Posterior median time-varying PSD
    S_mod_median = np.median(S_mod_samples, axis=0)
    # Ensure consistent frequency grids for comparison
    seg_freq = ref_periodogram.freqs
    seg_freq_indices = np.searchsorted(freq, seg_freq)
    time_varying_PSD_at_seg = time_varying_PSD[seg_freq_indices, :].T

    # Save samples and posterior data
    with open(samples_file, "wb") as f:
        pickle.dump(orbital_samples, f)
    post_data = {
        "times": times,
        "S_mod_median": S_mod_median,
        "time_varying_PSD_at_seg": time_varying_PSD_at_seg,
        "seg_freq": seg_freq,
        "t_downsampled": t_downsampled,
        "segment_midpoints": segment_midpoints,
    }
    with open(posts_file, "wb") as f:
        pickle.dump(post_data, f)


# Print parameter estimates
print(f"True orbital amplitude: {A_ORBIT:.3f}")
print(
    f"Estimated orbital amplitude: {np.median(orbital_samples['orbital_amplitude']):.3f} ± {np.std(orbital_samples['orbital_amplitude']):.3f}"
)
print(f"True orbital frequency: {F_ORBIT:.2e}")
print(
    f"Estimated orbital frequency: {np.median(orbital_samples['orbital_frequency']):.2e} ± {np.std(orbital_samples['orbital_frequency']):.2e}"
)


# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Clip to prevent log errors
S_mod_median = np.clip(S_mod_median, a_min=1e-15, a_max=None)
time_varying_PSD_at_seg = np.clip(
    time_varying_PSD_at_seg, a_min=1e-15, a_max=None
)

# Expected PSD
im0 = axes[0].imshow(
    np.log10(time_varying_PSD_at_seg.T),
    aspect="auto",
    origin="lower",
    extent=[
        t_downsampled.min(),
        t_downsampled.max(),
        seg_freq.min(),
        seg_freq.max(),
    ],
)
axes[0].set_title("Expected time-varying PSD (theory)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_yscale("log")
fig.colorbar(im0, ax=axes[0], label="log10(PSD)")

# Estimated PSD
im1 = axes[1].imshow(
    np.log10(S_mod_median.T),
    aspect="auto",
    origin="lower",
    extent=[times.min(), times.max(), seg_freq.min(), seg_freq.max()],
)
axes[1].set_title("Posterior median time-varying PSD")
axes[1].set_xlabel("Time (s)")
axes[1].set_yscale("log")
fig.colorbar(im1, ax=axes[1], label="log10(PSD)")

plt.tight_layout()
plt.savefig("corrected_2d_orbital_fit.png", dpi=150)
plt.show()

# Compute and print error
# Interpolate time_varying_PSD_at_seg to match S_mod_median time grid
time_varying_PSD_at_seg_interp = np.zeros_like(S_mod_median)
for i_f in range(len(seg_freq)):
    time_varying_PSD_at_seg_interp[:, i_f] = np.interp(
        times, t_downsampled, time_varying_PSD_at_seg[:, i_f]
    )
error = (
    np.abs(time_varying_PSD_at_seg_interp - S_mod_median)
    / time_varying_PSD_at_seg_interp
)
mean_error = np.mean(error)
print(
    f"Mean relative error between expected and posterior median PSD: {mean_error:.3f}"
)

# Plot parameter traces for diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

params = ["orbital_amplitude", "orbital_frequency", "orbital_phase"]
for i, param in enumerate(params):
    axes[i].plot(orbital_samples[param])
    axes[i].set_title(f"{param} trace")
    axes[i].set_xlabel("Sample")

plt.tight_layout()
plt.savefig("parameter_traces.png", dpi=150)
plt.show()

# Plot histograms of posterior samples
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].hist(
    orbital_samples["orbital_amplitude"], bins=30, alpha=0.7, label="Posterior"
)
axes[0].axvline(
    A_ORBIT, color="red", linestyle="--", linewidth=2, label=f"True: {A_ORBIT}"
)
axes[0].set_xlabel("Orbital Amplitude")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Posterior Distribution: Orbital Amplitude")
axes[0].legend()

axes[1].hist(
    orbital_samples["orbital_frequency"], bins=30, alpha=0.7, label="Posterior"
)
axes[1].axvline(
    F_ORBIT,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"True: {F_ORBIT:.2e}",
)
axes[1].set_xlabel("Orbital Frequency (Hz)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Posterior Distribution: Orbital Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("parameter_histograms.png", dpi=150)
plt.show()

# Plot unmodulated PSD S(f)
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(freq, PSD, label="True S(f)", linewidth=2)
ax.loglog(
    ref_periodogram.freqs,
    S_med_base,
    label="Median Estimated S(f)",
    linewidth=2,
    linestyle="--",
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD S(f) (1/Hz)")
ax.legend()
ax.set_title("Unmodulated PSD S(f) Comparison")
plt.tight_layout()
plt.savefig("unmodulated_psd_comparison.png", dpi=150)
plt.show()

# Plot the periodogram of the unmodulated timeseries
unmod_segment_periodograms, _ = segment_timeseries(
    ts, t, n_segments=52, freq_min=1e-5
)
unmod_periodogram = unmod_segment_periodograms[
    0
]  # Take the first segment to match ref_periodogram

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(
    ref_periodogram.freqs,
    unmod_periodogram.power**scale,
    label="Periodogram of unmodulated ts",
    linewidth=2,
)
ax.loglog(
    ref_periodogram.freqs,
    noise_PSD_AE(ref_periodogram.freqs),
    label="True PSD",
    linewidth=2,
    linestyle="--",
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD S(f) (1/Hz)")
ax.legend()
ax.set_title("Non-modulated PSD Comparison")
plt.tight_layout()
plt.savefig("non_modulated_psd.png", dpi=150)
plt.show()

print("Analysis complete!")
