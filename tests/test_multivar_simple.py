import jax
import jax.numpy as jnp
import numpy as np

from log_psplines.psplines.initialisation import init_basis_and_penalty

def generate_simple_varma():
    np.random.seed(42)

    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    n_samples = 1024

    # Generate correlated noise
    L = np.array([[1.0, 0.5], [0.0, 0.866]])
    noise = np.random.normal(0, 1, (n_samples, 2))
    x = noise @ L.T  # Cholesky to generate correlated noise

    return x

# Generate data
x = generate_simple_varma()
n_dim = 2

print(f"Data shape: {x.shape}, dim={n_dim}")

# Compute simple FFT
x_fft = np.fft.fft(x, axis=0)
n_time = len(x)
freqs = np.fft.fftfreq(n_time, 1.0)[:n_time//2]
n_freq = len(freqs)

y_re = np.real(x_fft[:n_freq])
y_im = np.imag(x_fft[:n_freq])

# Empirical PSD
empirical_psd = (y_re**2 + y_im**2) / n_time
print(f"Empirical PSD mean: {empirical_psd.mean(axis=0)}")
print(f"Empirical PSD std: {empirical_psd.std(axis=0)}")

# Setup P-splines
from log_psplines.psplines.initialisation import init_basis_and_penalty

knots = jnp.linspace(0, 1, 10)
degree = 3
n_nonzero = 5  # simple for test

basis, penalty = init_basis_and_penalty(knots, degree, n_freq, 2)

print("Basis shape:", basis.shape)
print("Basis max after normalization:", basis.max())

# Simulate weights for ONE component
weights = np.random.normal(0, 1, basis.shape[1])
log_delta_sq = basis @ weights
delta_sq = np.exp(log_delta_sq)

print(f"log_delta_sq typical: {log_delta_sq[::50]}")
print(f"delta_sq typical: {delta_sq[::50]}")

# Compare to empirical
print("Comparison:")
print(f"Empirical first few: {empirical_psd[:10, 0]}")
print(f"Model first few: {delta_sq[:10]}")

# To fix scaling, perhaps we need to adjust the knot placement or something.

# Perhaps the issue is the frequency normalization.
freq_norm = (freqs - freqs.min()) / (freqs.max() - freqs.min())

print("All good?")
