# All necessary imports
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


# A simple class to replace the pywavelet dependency
class TimeSeries:
    def __init__(self, data, time):
        """Initializes the TimeSeries object."""
        self.data = np.asarray(data)
        self.time = np.asarray(time)

    def to_frequencyseries(self):
        """Calculates the FFT of the time series."""
        # Normalize the FFT to match standard periodogram definitions
        fft_data = np.fft.fft(self.data)
        freqs = np.fft.fftfreq(len(self.data), d=self.time[1] - self.time[0])
        return FrequencySeries(data=fft_data, freq=freqs)


class FrequencySeries:
    def __init__(self, data, freq):
        """Initializes the FrequencySeries object."""
        self.data = np.asarray(data)
        self.freq = np.asarray(freq)

    def to_timeseries(self):
        """Calculates the inverse FFT to get the time series."""
        time_series = np.fft.ifft(self.data)
        dt = 1 / (self.freq.max() * 2)  # Crude approximation, assuming Nyquist
        # More robust: dt = 1/(2*self.freq[-1]) for positive frequencies
        # The freq array from fftfreq is already correct
        dt = 1 / (len(self.freq) * (self.freq[1] - self.freq[0]))
        time_array = np.arange(len(self.freq)) * dt
        return TimeSeries(data=time_series.real, time=time_array)


np.random.seed(1234)


def noise_PSD_AE(f: np.ndarray, TDI="TDI1"):
    """
    Takes in frequency, spits out TDI1 or TDI2 A channel, same as E channel is equal and constant arm length approx.
    """
    # Ensure f is a numpy array and handle zero frequencies
    f = np.asarray(f)
    f = np.where(f == 0, 1e-10, f)

    L = 2.5e9
    c = 299758492
    x = 2 * np.pi * (L / c) * f

    # Spm is acceleration noise, converted to strain noise by dividing by (2pi*f)^2
    Spm = (
        (3e-15) ** 2
        * (1 + ((4e-4) / f) ** 2)
        * (1 + (f / (8e-3)) ** 4)
        * (1 / (2 * np.pi * f)) ** 4
    )
    # Sop is optical metrology sensor noise, converted to strain noise
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

    # Handle the DC component which can become NaN or inf
    S[0] = S[1]
    return FrequencySeries(data=S, freq=f)


def generate_stationary_noise(
    ND: int,
    dt: float,
    psd: FrequencySeries,
    time_domain: bool = False,
    seed=None,
) -> Union[TimeSeries, FrequencySeries]:
    """
    Generates stationary noise from a given PSD.

    Parameters:
    -----------
    ND : int
        Number of data points.
    dt : float
        Time step (1/sampling frequency).
    psd : FrequencySeries
        FrequencySeries object containing the PSD values.
    time_domain : bool, optional
        If True, returns a TimeSeries object. If False, returns a FrequencySeries.
    seed : int, optional
        Seed for the random number generator.

    Returns:
    --------
    Union[TimeSeries, FrequencySeries]
        The generated noise.
    """

    if seed is not None:
        np.random.seed(seed)

    print(f"Generating stationary noise... [Seed:{seed}]")

    # The relationship between a single-sided PSD S(f) and the FFT of the signal H(f)
    # is E[|H(f)|^2] = S(f) * T / 2, where T is the total duration ND*dt.
    # The standard deviation for the real and imaginary parts of the FFT components
    # is therefore sqrt(S(f) * T / 4).
    T = ND * dt
    std_f = np.sqrt(psd.data * T / 4)

    # Generate noise in frequency domain
    real_noise_f = np.random.normal(0, std_f)
    imag_noise_f = np.random.normal(0, std_f)
    noise_f = real_noise_f + 1j * imag_noise_f

    # The DC (f=0) and Nyquist (f=fs/2) components must be real for the
    # inverse FFT to be real. Their imaginary parts should be zero.
    noise_f[0] = noise_f[0].real
    if ND % 2 == 0:
        noise_f[ND // 2] = noise_f[ND // 2].real
        # Ensure Hermitian symmetry for the negative frequencies
        noise_f[ND // 2 + 1 :] = np.conj(noise_f[1 : ND // 2][::-1])
    else:
        # For odd number of points, the last positive freq is the highest,
        # and the negative freqs are its conjugates.
        noise_f[-1 : ND // 2 : -1] = np.conj(noise_f[1 : ND // 2][::-1])
        noise_f[ND // 2 + 1 :] = np.conj(noise_f[1 : ND // 2][::-1])
        # This seems to have issues with odd numbers of points, let's keep it even for now
        # The original code used even, let's stick to that.

    noise_series = FrequencySeries(noise_f, psd.freq)

    # Convert to time-series
    if time_domain:
        return noise_series.to_timeseries()
    else:
        return noise_series


# Simulation parameters
tmax = 6 * 24 * 60 * 60  # 1 day in seconds
fs = 1.0  # Sampling rate in Hz
dt = 1.0 / fs
n_data = 2 ** int(np.log2(tmax * fs))  # Power of 2 for FFT efficiency

print(f"Generating {n_data} data points over {tmax / 3600:.1f} hours")
print(f"Frequency resolution: {fs / n_data:.2e} Hz")
print(f"Nyquist frequency: {fs / 2:.2e} Hz")

# --- Non-stationary analysis parameters ---
# Modulation parameters
A_TRUE = 1.0
ALPHA_TRUE = 0.5  # Modulation depth
F_TRUE = 1e-5
t_true = 1.0 / F_TRUE

# Non-stationary analysis parameters
q = int(np.log(n_data) / np.log(2))
qf = int(q / 2) + 1
Nf = 2**qf
Nt = 2 ** (q - qf)
mult = 32
K = mult * 2 * Nf

tobs = tmax
wdm_dt = tobs / Nt
wdm_df = Nt / (2 * tobs)

print(
    f"Duration which the process is approximately stationary is {t_true / 10:,.2f}"
)
print(f"ND: {n_data:,}, Nf: {Nf}, Nt: {Nt}, Nf * Nt: {Nf * Nt:,}")
print(f"dt={wdm_dt}, df={wdm_df}")
# --- End of Non-stationary analysis parameters ---


# Generate unmodulated LISA noise for verification
print("\nGenerating unmodulated colored noise from LISA PSD...")
freqs = np.fft.fftfreq(n_data, d=dt)
psd_obj = noise_PSD_AE(freqs, TDI="TDI1")
data_unmodulated = generate_stationary_noise(
    n_data, dt, psd_obj, time_domain=True
)

# Convert to frequency domain for verification
print("Converting unmodulated noise to frequency domain...")
prdgm_unmodulated = data_unmodulated.to_frequencyseries()

# Calculate true PSD at the same frequencies
true_psd_data = psd_obj.data
true_psd_freqs = psd_obj.freq

# Modulation function as requested
modulation_func = lambda t: A_TRUE * (
    1 + ALPHA_TRUE * np.cos(2 * np.pi * F_TRUE * t)
)
plt.plot(
    data_unmodulated.time / (24 * 60 * 60),
    modulation_func(data_unmodulated.time),
    label="Modulation Function",
    color="tab:orange",
)
plt.xlabel("Time (days)")
plt.ylabel("Modulation Amplitude")
plt.savefig("modulation_function.png")

# Create a new time series object for the modulated data
data_modulated = TimeSeries(
    data=data_unmodulated.data * modulation_func(data_unmodulated.time),
    time=data_unmodulated.time,
)
print(
    f"Modulation applied with A={A_TRUE}, alpha={ALPHA_TRUE}, and period of {t_true / (24 * 60 * 60):.0f} days."
)

# Create plots to verify the noise follows the PSD
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Time series (Unmodulated and Modulated)
t_hours = data_unmodulated.time / 3600
axes[0, 0].plot(
    t_hours,
    data_unmodulated.data,
    color="tab:blue",
    alpha=0.5,
    label="Unmodulated Noise",
)
axes[0, 0].plot(
    t_hours,
    data_modulated.data,
    color="tab:green",
    alpha=0.7,
    label="Modulated Noise",
)
axes[0, 0].set_xlabel("Time (hours)")
axes[0, 0].set_ylabel("Strain")
axes[0, 0].set_title("Generated LISA Noise Time Series")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: PSD comparison (Unmodulated, Modulated, and True)
freq_mask = prdgm_unmodulated.freq > 0  # Only positive frequencies
valid_freqs = prdgm_unmodulated.freq[freq_mask]

# Calculate the periodogram from the unmodulated FFT data
periodogram_unmodulated = (2 / (n_data * fs)) * np.abs(
    prdgm_unmodulated.data[freq_mask]
) ** 2
axes[0, 1].loglog(
    valid_freqs,
    periodogram_unmodulated,
    color="tab:blue",
    alpha=0.5,
    label="Raw Periodogram (Unmodulated)",
)

# Calculate the periodogram from the modulated FFT data
prdgm_modulated = data_modulated.to_frequencyseries()
periodogram_modulated = (2 / (n_data * fs)) * np.abs(
    prdgm_modulated.data[freq_mask]
) ** 2
axes[0, 1].loglog(
    valid_freqs,
    periodogram_modulated,
    color="tab:green",
    alpha=0.7,
    label="Raw Periodogram (Modulated)",
)

# Smoothed periodogram of unmodulated noise for comparison
freq_bins = np.logspace(
    np.log10(valid_freqs[0]), np.log10(valid_freqs[-1]), 50
)
binned_psd_unmodulated = []
binned_freq = []
for i in range(len(freq_bins) - 1):
    mask = (valid_freqs >= freq_bins[i]) & (valid_freqs < freq_bins[i + 1])
    if np.any(mask):
        binned_psd_unmodulated.append(np.mean(periodogram_unmodulated[mask]))
        binned_freq.append(np.sqrt(freq_bins[i] * freq_bins[i + 1]))

axes[0, 1].loglog(
    binned_freq,
    binned_psd_unmodulated,
    "o-",
    color="tab:blue",
    linewidth=2,
    markersize=4,
    label="Binned Periodogram (Unmodulated)",
)

# True PSD
true_psd_positive = true_psd_data[freq_mask]
axes[0, 1].loglog(
    valid_freqs,
    true_psd_positive,
    color="tab:red",
    linewidth=3,
    label="True LISA PSD",
)
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("PSD (strain²/Hz)")
axes[0, 1].set_title("PSD Comparison")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Ratio of generated to true PSD (using binned unmodulated data)
true_psd_binned = noise_PSD_AE(np.array(binned_freq), TDI="TDI1").data
ratio_binned = np.array(binned_psd_unmodulated) / true_psd_binned
axes[1, 0].semilogx(
    binned_freq,
    ratio_binned,
    "o-",
    color="tab:blue",
    markersize=4,
    label="Binned ratio (Unmodulated)",
)
axes[1, 0].axhline(
    y=1, color="red", linestyle="--", linewidth=2, label="Perfect match"
)
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Ratio (Generated/True)")
axes[1, 0].set_title("Unmodulated PSD Ratio (Binned)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0.1, 10)

# Plot 4: Histogram of unmodulated time series values
axes[1, 1].hist(
    data_unmodulated.data, bins=50, density=True, alpha=0.7, color="tab:purple"
)
axes[1, 1].set_xlabel("Strain amplitude")
axes[1, 1].set_ylabel("Probability density")
axes[1, 1].set_title("Distribution of Unmodulated noise values")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lisa_noise_generation.png")

# Print diagnostics for the modulated signal
print(f"\nDiagnostics for Modulated Signal:")
print(f"Time series stats:")
print(f"  Mean: {np.mean(data_modulated.data):.2e}")
print(f"  Std:  {np.std(data_modulated.data):.2e}")
print(f"  Min:  {np.min(data_modulated.data):.2e}")
print(f"  Max:  {np.max(data_modulated.data):.2e}")

print(f"\nPSD verification (unmodulated noise):")
if len(binned_freq) > 0:
    true_psd_binned = noise_PSD_AE(np.array(binned_freq), TDI="TDI1").data
    ratio_binned = np.array(binned_psd_unmodulated) / true_psd_binned
    print(f"  Mean ratio (generated/true): {np.mean(ratio_binned):.3f}")
    print(f"  Std of ratio: {np.std(ratio_binned):.3f}")
    print(
        f"  Ratio range: {np.min(ratio_binned):.3f} to {np.max(ratio_binned):.3f}"
    )

    if 0.5 < np.mean(ratio_binned) < 2.0:
        print("  ✓ PSD scaling looks reasonable")
    else:
        print("  ✗ PSD scaling issue detected")
else:
    print("  No frequency bins available for comparison")


from pywavelet.transforms import from_time_to_wavelet

### PART 2 Wavelet
from pywavelet.types import FrequencySeries, TimeSeries, Wavelet

data_modulated = TimeSeries(data=data_modulated.data, time=data_modulated.time)
data_wavelet = from_time_to_wavelet(
    data_modulated, Nf=Nf, Nt=Nt, nx=4.0, mult=mult
)


# analytical PSD
tn = data_wavelet.time
fm = data_wavelet.freq
lisa_psd = noise_PSD_AE(fm, TDI="TDI1").data
stf = np.sqrt(
    np.dot(np.asarray([modulation_func(tn) ** 2]).T, np.asarray([lisa_psd]))
).T
analytical_wavelet = Wavelet(data=stf, time=tn, freq=fm)
# look at ratios
ratios = np.sqrt(
    np.abs(data_wavelet.data) ** 2 / np.abs(analytical_wavelet.data) ** 2
)
ratios = Wavelet(data=ratios, time=tn, freq=fm)

frange = [0.01, 0.1]  # Frequency range for the analysis
trange = [tn[16], tn[-16]]  # Time range for the analysis

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
kwgs = dict(
    absolute=True, zscale="log", freq_scale="log", freq_range=(10e-3, 10e-1)
)
_, ax = data_wavelet.plot(ax=axes[0], **kwgs)
_ = ax.set_title("Wavelet transform of the modulated data", y=1.01)
_, ax = analytical_wavelet.plot(ax=axes[1], **kwgs)
_ = ax.set_title("Analytical PSD", y=1.01)
_, ax = ratios.plot(ax=axes[2], **kwgs)
_ = ax.set_title("Ratio of modulated data to analytical wavelet", y=1.01)
ax.set_ylim(*frange)
ax.set_xlim(*trange)
plt.savefig("wavelet_analysis.png")


import scipy.stats as stats

d = data_wavelet.data  # Wavelet data
a = analytical_wavelet.data  # Analytical wavelet data

# filter out freq and time ranges
freq_mask = (fm > frange[0]) & (fm < frange[1])
time_mask = (tn > trange[0]) & (tn < trange[1])

d = d[freq_mask, :][:, time_mask].flatten()
a = a[freq_mask, :][:, time_mask].flatten()

# The new formula for amplitude ratios
amp_ratios = d / np.sqrt(a * fs / 2)

# plot the ratios

plt.figure(0)
plt.hist(
    amp_ratios.flatten(), density=True, bins=100, histtype="step"
)  # , range=[0, 0.5])
# x = np.linspace(-3, 3, 100)
# plt.plot(x, stats.norm.pdf(x), ls="--")
# plt.xlim(-3, 3)
plt.xlabel(r"$\frac{\sqrt{|w_{nm}| \tau_s}}{S_{x}(f_m, t_n)}$")
plt.tight_layout()
plt.savefig("amplitude_ratios_histogram.png")
