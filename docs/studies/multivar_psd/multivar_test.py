import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from dataclasses import dataclass
from typing import Tuple, List
from log_psplines.psplines.initialisation import init_basis_and_penalty
from numpyro.infer import MCMC, NUTS


@dataclass
class DiscreteFFT:
    """Discrete FFTs for each timeseries, with real and imaginary parts separated.
    fft: shape (n_freq, n_dim, 2), where [:,:,0] is real and [:,:,1] is imaginary.
    Z_matrix: shape (n_freq, n_dim, n_theta_total) - design matrix for multivariate components.
    """
    fft: jnp.ndarray  # Shape: (n_freq, n_dim, 2) - real and imag parts for each timeseries
    freq: jnp.ndarray  # Shape: (n_freq,) - frequency grid
    n_freq: int  # Number of frequencies
    n_dim: int  # Number of timeseries
    Z_matrix: jnp.ndarray  # Shape: (n_freq, n_dim, n_theta) - design matrix for theta parameters


def compute_discrete_fft(x: np.ndarray, fs: float = 1.0, compute_z_matrix: bool = True) -> DiscreteFFT:
    """
    Compute discrete FFTs for each timeseries, storing real and imaginary parts separately.

    Parameters
    ----------
    x : np.ndarray, shape (n_time, n_dim)
        Multivariate time series
    fs : float
        Sampling frequency
    compute_z_matrix : bool, optional
        Whether to compute the Z matrix (design matrix for multivariate components)

    Returns
    -------
    DiscreteFFT
    """
    n_time, n_dim = x.shape
    # Compute FFT for each timeseries (column)
    x_fft = np.fft.fft(x, axis=0)  # shape: (n_time, n_dim)
    freqs = np.fft.fftfreq(n_time, 1 / fs)
    # Keep only positive frequencies
    pos_freq_idx = freqs > 0
    freqs = freqs[pos_freq_idx]
    x_fft = x_fft[pos_freq_idx, :]  # shape: (n_freq, n_dim)
    n_freq = len(freqs)
    # Split into real and imaginary parts
    x_fft_real = np.real(x_fft)
    x_fft_imag = np.imag(x_fft)
    x_fft_split = np.stack([x_fft_real, x_fft_imag], axis=-1)  # shape: (n_freq, n_dim, 2)

    # Compute Z matrix if requested
    if compute_z_matrix:
        Z_matrix = compute_Zmatrix(x_fft)
    else:
        Z_matrix = jnp.zeros((n_freq, n_dim, 0), dtype=jnp.complex64)

    return DiscreteFFT(
        fft=jnp.array(x_fft_split),
        freq=jnp.array(freqs),
        n_freq=n_freq,
        n_dim=n_dim,
        Z_matrix=jnp.array(Z_matrix)
    )


def compute_Zmatrix(y_k: np.ndarray) -> np.ndarray:
    """
    Compute the design matrix Z_k for each frequency k.
    Parameters:
    y_k (np.ndarray): Fourier transformed time series data of shape (n_freq, p).
    Returns:
    np.ndarray: Design matrix Z_k of shape (n_freq, p, p*(p-1)/2).
    """
    n, p = y_k.shape
    if p == 1:
        return np.zeros((n, p, 0), dtype=np.complex64)

    Z_k = np.zeros((n, p, int(p * (p - 1) / 2)), dtype=np.complex64)
    for j in range(n):  # for each frequency
        count = 0
        for i in range(1, p):  # for each component starting from 1
            Z_k[j, i, count: count + i] = y_k[j, :i]  # fill with previous components
            count += i
    return Z_k


def compute_discrete_fft_with_Z(x: np.ndarray, fs: float = 1.0) -> Tuple[DiscreteFFT, jnp.ndarray]:
    """
    Compute discrete FFTs and Z matrix.
    """
    fft_data = compute_discrete_fft(x, fs)

    # Reconstruct complex FFT for Z matrix computation
    fft_complex = fft_data.fft[:, :, 0] + 1j * fft_data.fft[:, :, 1]

    # Compute Z matrix
    Z = compute_Zmatrix(fft_complex)

    return fft_data, jnp.array(Z)


def setup_psplines_for_sequential_multivariate_fixed(
        freq: jnp.ndarray,
        n_dim: int,
        n_knots: int = 10,
        degree: int = 3,
        diff_order: int = 2
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Set up P-splines for sequential multivariate model.
    """
    freq_norm = (freq - freq.min()) / (freq.max() - freq.min())
    n_freq = len(freq_norm)

    knots = jnp.linspace(0, 1, n_knots)
    basis, penalty = init_basis_and_penalty(
        knots, degree, n_freq, diff_order
    )

    # Components: n_dim diagonal + 2 off-diagonal (real/imag theta)
    total_components = n_dim + 2

    print(f"Total components needed: {total_components}")
    print(f"  - Diagonal (delta): {n_dim}")
    print(f"  - Off-diagonal (theta): 2 (real + imag)")

    all_bases = [basis for _ in range(total_components)]
    all_penalties = [penalty for _ in range(total_components)]

    return all_bases, all_penalties


def multivariate_whittle_likelihood_Z_matrix(
        fft: jnp.ndarray,  # shape (n_freq, n_dim, 2)
        Z_matrix: jnp.ndarray,  # shape (n_freq, n_dim, n_theta_total)
        log_deltas: List[jnp.ndarray],  # List of n_dim arrays, each shape (n_freq,)
        theta_re_all: jnp.ndarray,  # shape (n_freq, n_theta_total)
        theta_im_all: jnp.ndarray,  # shape (n_freq, n_theta_total)
) -> jnp.ndarray:
    """
    Sequential multivariate Whittle likelihood using Z matrix approach (like SGVB).
    """
    n_freq, n_dim, _ = fft.shape
    y_re = fft[:, :, 0]  # (n_freq, n_dim)
    y_im = fft[:, :, 1]  # (n_freq, n_dim)

    # Extract real and imaginary parts of Z matrix
    Z_re = Z_matrix.real  # (n_freq, n_dim, n_theta_total)
    Z_im = Z_matrix.imag  # (n_freq, n_dim, n_theta_total)

    # Compute exp(-log_deltas) for all components
    exp_neg_log_deltas = [jnp.exp(-log_deltas[j]) for j in range(n_dim)]

    # Compute residuals for all components
    if Z_matrix.shape[2] > 0:  # If we have theta parameters
        # Vectorized computation like in SGVB
        # Z_theta_re = Z_re @ theta_re - Z_im @ theta_im
        # Z_theta_im = Z_re @ theta_im + Z_im @ theta_re
        Z_theta_re = jnp.einsum('fij,fj->fi', Z_re, theta_re_all) - jnp.einsum('fij,fj->fi', Z_im, theta_im_all)
        Z_theta_im = jnp.einsum('fij,fj->fi', Z_re, theta_im_all) + jnp.einsum('fij,fj->fi', Z_im, theta_re_all)

        u_re = y_re - Z_theta_re
        u_im = y_im - Z_theta_im
    else:
        u_re = y_re
        u_im = y_im

    # Compute likelihood
    numerator = u_re ** 2 + u_im ** 2  # (n_freq, n_dim)

    total_log_lik = 0.0
    for j in range(n_dim):
        # Sum of log(deltas) for this component
        sum_log_delta = jnp.sum(log_deltas[j])

        # Weighted residuals for this component
        internal_j = numerator[:, j] * exp_neg_log_deltas[j]
        tmp2_j = -jnp.sum(internal_j)

        component_log_lik = sum_log_delta + tmp2_j
        total_log_lik += component_log_lik

    return total_log_lik


def multivariate_psplines_model_sequential_fixed(
        fft_data: DiscreteFFT,
        all_bases: List[jnp.ndarray],
        all_penalties: List[jnp.ndarray],
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
):
    """
    NumPyro model using sequential Whittle likelihood with Z matrix approach.
    """
    n_dim = fft_data.n_dim
    n_freq = fft_data.n_freq
    Z_matrix = fft_data.Z_matrix
    n_theta_total = Z_matrix.shape[2]  # Total number of theta parameters

    # Sample P-spline components for each diagonal element (log delta^2)
    log_deltas = []
    component_idx = 0

    for j in range(n_dim):
        delta = numpyro.sample(f"delta_{j}", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample(f"phi_delta_{j}", dist.Gamma(alpha_phi, delta * beta_phi))

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample(f"weights_delta_{j}",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        # Prior on weights
        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor(f"weights_prior_delta_{j}", log_prior_w)

        # Compute log(delta^2) for this component
        log_delta_sq = all_bases[component_idx] @ weights
        log_deltas.append(log_delta_sq)
        component_idx += 1

    # Sample theta parameters (vectorized like in SGVB)
    theta_re_all = jnp.zeros((n_freq, n_theta_total))
    theta_im_all = jnp.zeros((n_freq, n_theta_total))

    if n_theta_total > 0:
        # Real parts of theta
        delta = numpyro.sample("delta_theta_re", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_re", dist.Gamma(alpha_phi, delta * beta_phi))

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_re",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_re", log_prior_w)

        theta_re_values = all_bases[component_idx] @ weights
        if n_theta_total == 1:
            theta_re_all = theta_re_values[:, None]
        else:
            # Broadcast or tile to match n_theta_total
            theta_re_all = jnp.tile(theta_re_values[:, None], (1, n_theta_total))
        component_idx += 1

        # Imaginary parts of theta
        delta = numpyro.sample("delta_theta_im", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_im", dist.Gamma(alpha_phi, delta * beta_phi))

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_im",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_im", log_prior_w)

        theta_im_values = all_bases[component_idx] @ weights
        if n_theta_total == 1:
            theta_im_all = theta_im_values[:, None]
        else:
            # Broadcast or tile to match n_theta_total
            theta_im_all = jnp.tile(theta_im_values[:, None], (1, n_theta_total))

    # Sequential Whittle likelihood using Z matrix
    log_likelihood = multivariate_whittle_likelihood_Z_matrix(
        fft_data.fft, Z_matrix, log_deltas, theta_re_all, theta_im_all
    )
    numpyro.factor("likelihood", log_likelihood)

    # Store for diagnostics
    numpyro.deterministic("log_deltas", log_deltas)
    numpyro.deterministic("theta_re_all", theta_re_all)
    numpyro.deterministic("theta_im_all", theta_im_all)
    numpyro.deterministic("log_likelihood", log_likelihood)


def reconstruct_psd_matrices_from_samples(log_deltas_samples, theta_re_samples, theta_im_samples, Z_matrix):
    """
    Robust numpy version: reconstruct PSD matrices from posterior samples using Cholesky parameterization.
    Follows TensorFlow logic: PSD = inv(T^H D^{-1} T)
    All inputs should be numpy arrays.
    """
    log_deltas_samples = np.array(log_deltas_samples)
    theta_re_samples = np.array(theta_re_samples)
    theta_im_samples = np.array(theta_im_samples)
    Z_matrix = np.array(Z_matrix)

    n_samples = log_deltas_samples.shape[0]
    n_dim = log_deltas_samples.shape[1]
    n_freq = log_deltas_samples.shape[2]
    n_theta = theta_re_samples.shape[2] if theta_re_samples.ndim == 3 else 0

    psd_samples = np.zeros((n_samples, n_freq, n_dim, n_dim), dtype=np.complex64)

    # Get lower triangle indices for theta placement
    row_indices, col_indices = np.tril_indices(n_dim, k=-1)

    for i in range(min(100, n_samples)):
        for k in range(n_freq):
            # D: diagonal matrix of exp(log_delta)
            delta_vec = np.exp(log_deltas_samples[i, :, k])
            D = np.diag(delta_vec)
            D_inv = np.linalg.inv(D + 1e-8 * np.eye(n_dim))  # regularize for safety

            # T: lower-triangular matrix, fill lower triangle with -theta
            T = np.eye(n_dim, dtype=np.complex64)
            if n_theta > 0:
                theta_vec = - (theta_re_samples[i, k, :] + 1j * theta_im_samples[i, k, :])
                T[row_indices, col_indices] = theta_vec

            # S_inv = T^H D^{-1} T
            T_conj_trans = np.conj(T.T)
            S_inv = T_conj_trans @ D_inv @ T
            # PSD = inv(S_inv)
            PSD = np.linalg.inv(S_inv + 1e-8 * np.eye(n_dim))  # regularize for safety
            psd_samples[i, k] = PSD
    return psd_samples


if __name__ == "__main__":
    # Import the simulation
    from sgvb_psd.utils.sim_varma import SimVARMA

    # Generate VARMA data
    np.random.seed(42)

    SIM_KWGS = dict(
        sigma=np.array([[1.0, 0.9], [0.9, 1.0]]),
        var_coeffs=np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]]),
        vma_coeffs=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        n_samples=1024,
    )
    varma = SimVARMA(**SIM_KWGS)
    x = varma.data
    n_dim = varma.dim

    print(f"Simulated VARMA data shape: {x.shape}, dim={n_dim}")

    # Compute FFT and Z matrix
    fft_data, Z_matrix = compute_discrete_fft_with_Z(x, fs=1.0)
    print(f"FFT data shape: {fft_data.fft.shape}")
    print(f"Z matrix shape: {Z_matrix.shape}")
    print(f"Frequency range: {fft_data.freq.min():.3f} to {fft_data.freq.max():.3f}")

    # Set up P-splines
    all_bases, all_penalties = setup_psplines_for_sequential_multivariate_fixed(
        fft_data.freq, n_dim=n_dim, n_knots=15
    )

    # Run MCMC with the fixed model
    nuts_kernel = NUTS(multivariate_psplines_model_sequential_fixed)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500, num_chains=1)

    # Now just use fft_data directly, which contains the Z matrix
    print("Starting MCMC sampling...")
    mcmc.run(
        jax.random.PRNGKey(0),
        fft_data=fft_data,
        all_bases=all_bases,
        all_penalties=all_penalties
    )

    # Get samples
    samples = mcmc.get_samples()
    print(f"Sampling completed. Keys: {list(samples.keys())}")

    # Print summary statistics
    mcmc.print_summary()

    # Reconstruct PSD matrices from samples
    print("\nReconstructing PSD matrices...")
    log_deltas_samples = samples['log_deltas']
    theta_re_samples = samples['theta_re_all']
    theta_im_samples = samples['theta_im_all']

    # Now use Z matrix from fft_data object
    psd_samples = reconstruct_psd_matrices_from_samples(
        log_deltas_samples, theta_re_samples, theta_im_samples, fft_data.Z_matrix
    )
    print(f"Reconstructed PSD samples shape: {psd_samples.shape}")

    # Compute posterior statistics
    psd_mean = jnp.mean(psd_samples, axis=0)
    psd_std = jnp.std(psd_samples, axis=0)

    print(f"PSD mean shape: {psd_mean.shape}")
    print(f"PSD std shape: {psd_std.shape}")

    # Print some statistics
    print(f"\nDiagonal PSD values at first frequency:")
    for i in range(n_dim):
        print(f"  Component {i}: {psd_mean[0, i, i].real:.6f} ± {psd_std[0, i, i].real:.6f}")

    print(f"\nOff-diagonal PSD values at first frequency:")
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            val = psd_mean[0, i, j]
            std = psd_std[0, i, j]
            print(f"  ({i},{j}): {val.real:.6f}+{val.imag:.6f}i ± {std.real:.6f}+{std.imag:.6f}i")

    # Plotting code
    import matplotlib.pyplot as plt

    freq = np.array(fft_data.freq)
    n_freq = len(freq)

    # Compute empirical periodogram and CSD
    fft_complex = fft_data.fft[:, :, 0] + 1j * fft_data.fft[:, :, 1]  # (n_freq, n_dim)
    periodogram = np.abs(fft_complex) ** 2  # (n_freq, n_dim)
    csd_real = np.zeros((n_freq, n_dim, n_dim))
    csd_imag = np.zeros((n_freq, n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            csd = fft_complex[:, i] * np.conj(fft_complex[:, j])
            csd_real[:, i, j] = csd.real
            csd_imag[:, i, j] = csd.imag


    # Compute posterior quantiles
    def get_quantiles(arr, axis=0):
        q05 = np.percentile(arr, 5, axis=axis)
        q50 = np.percentile(arr, 50, axis=axis)
        q95 = np.percentile(arr, 95, axis=axis)
        return q05, q50, q95


    fig, axes = plt.subplots(n_dim, n_dim, figsize=(3 * n_dim, 3 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [[axes]]
    elif n_dim == 2:
        axes = axes.reshape(n_dim, n_dim)

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i][j]
            # Diagonal: PSD
            if i == j:
                # Posterior quantiles
                q05, q50, q95 = get_quantiles(psd_samples[:, :, i, i].real, axis=0)
                ax.fill_between(freq, q05, q95, color='blue', alpha=0.2, label='Posterior 90%')
                ax.plot(freq, q50, color='blue', label='Posterior median')
                # Empirical periodogram
                ax.plot(freq, periodogram[:, i], color='black', alpha=0.3, lw=1, label='Empirical', zorder=-1)
                ax.set_title(f"PSD {i}")
                ax.set_yscale('log')
            # Below diagonal: real part of CSD
            elif i > j:
                q05, q50, q95 = get_quantiles(psd_samples[:, :, i, j].real, axis=0)
                ax.fill_between(freq, q05, q95, color='green', alpha=0.2, label='Posterior 90%')
                ax.plot(freq, q50, color='green', label='Posterior median')
                ax.plot(freq, csd_real[:, i, j], color='black', alpha=0.3, lw=1, label='Empirical', zorder=-1)
                ax.set_title(f"CSD real ({i},{j})")
            # Above diagonal: imag part of CSD
            else:
                q05, q50, q95 = get_quantiles(psd_samples[:, :, i, j].imag, axis=0)
                ax.fill_between(freq, q05, q95, color='red', alpha=0.2, label='Posterior 90%')
                ax.plot(freq, q50, color='red', label='Posterior median')
                ax.plot(freq, csd_imag[:, i, j], color='black', alpha=0.3, lw=1, label='Empirical', zorder=-1)
                ax.set_title(f"CSD imag ({i},{j})")
            ax.set_xlabel("Frequency")
            if j == 0:
                ax.set_ylabel(f"Ch {i}")
            if i == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('multivar_psd_csd_matrix_fixed.png')
    plt.show()
