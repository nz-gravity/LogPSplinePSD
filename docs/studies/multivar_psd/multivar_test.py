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
    """Discrete FFTs for each timeseries, with real and imaginary parts separated."""
    fft: jnp.ndarray  # Shape: (n_freq, n_dim, 2) - real and imag parts
    freq: jnp.ndarray  # Shape: (n_freq,) - frequency grid
    n_freq: int  # Number of frequencies
    n_dim: int  # Number of timeseries
    Z_matrix: jnp.ndarray  # Shape: (n_freq, n_dim, n_theta) - design matrix


def compute_discrete_fft(x: np.ndarray, fs: float = 1.0) -> DiscreteFFT:
    """Compute discrete FFTs and Z matrix."""
    n_time, n_dim = x.shape
    x_fft = np.fft.fft(x, axis=0)
    freqs = np.fft.fftfreq(n_time, 1 / fs)
    pos_freq_idx = freqs > 0
    freqs = freqs[pos_freq_idx]
    x_fft = x_fft[pos_freq_idx, :]
    n_freq = len(freqs)

    # Split into real and imaginary parts
    x_fft_real = np.real(x_fft)
    x_fft_imag = np.imag(x_fft)
    x_fft_split = np.stack([x_fft_real, x_fft_imag], axis=-1)

    # Compute Z matrix
    Z_matrix = compute_Zmatrix(x_fft)

    return DiscreteFFT(
        fft=jnp.array(x_fft_split),
        freq=jnp.array(freqs),
        n_freq=n_freq,
        n_dim=n_dim,
        Z_matrix=jnp.array(Z_matrix)
    )


def compute_Zmatrix(y_k: np.ndarray) -> np.ndarray:
    """Compute the design matrix Z_k exactly as in SGVB."""
    n, p = y_k.shape
    if p <= 1:
        return np.zeros((n, p, 0), dtype=np.complex64)

    n_theta = int(p * (p - 1) / 2)
    Z_k = np.zeros((n, p, n_theta), dtype=np.complex64)

    for j in range(n):  # for each frequency
        count = 0
        for i in range(1, p):  # for each component starting from 1
            Z_k[j, i, count:count + i] = y_k[j, :i]  # fill with previous components
            count += i
    return Z_k


def setup_psplines_for_multivariate(
        freq: jnp.ndarray,
        n_dim: int,
        n_knots: int = 10,
        degree: int = 3,
        diff_order: int = 2
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """Set up P-splines - one for each diagonal + one for each theta component."""
    freq_norm = (freq - freq.min()) / (freq.max() - freq.min())
    n_freq = len(freq_norm)

    knots = jnp.linspace(0, 1, n_knots)
    basis, penalty = init_basis_and_penalty(knots, degree, n_freq, diff_order)

    # Components: n_dim diagonal + n_theta_real + n_theta_imag
    n_theta = int(n_dim * (n_dim - 1) / 2)
    total_components = n_dim + 2 * n_theta  # delta + theta_real + theta_imag

    print(f"Total components needed: {total_components}")
    print(f"  - Diagonal (delta): {n_dim}")
    print(f"  - Theta components: {2 * n_theta} (real + imag)")

    all_bases = [basis for _ in range(total_components)]
    all_penalties = [penalty for _ in range(total_components)]

    return all_bases, all_penalties


def multivariate_whittle_likelihood_corrected(
        fft: jnp.ndarray,  # shape (n_freq, n_dim, 2)
        Z_matrix: jnp.ndarray,  # shape (n_freq, n_dim, n_theta)
        log_deltas: List[jnp.ndarray],  # List of n_dim arrays, each shape (n_freq,)
        theta_re: jnp.ndarray,  # shape (n_freq, n_theta)
        theta_im: jnp.ndarray,  # shape (n_freq, n_theta)
) -> jnp.ndarray:
    """Corrected likelihood following SGVB exactly."""
    n_freq, n_dim, _ = fft.shape
    y_re = fft[:, :, 0]  # (n_freq, n_dim)
    y_im = fft[:, :, 1]  # (n_freq, n_dim)

    Z_re = Z_matrix.real  # (n_freq, n_dim, n_theta)
    Z_im = Z_matrix.imag  # (n_freq, n_dim, n_theta)

    # Compute sum of log deltas (equivalent to sum_xγ in SGVB)
    sum_log_deltas = sum(jnp.sum(log_delta) for log_delta in log_deltas)

    # Compute exp(-log_deltas) for each component (equivalent to exp_xγ_inv in SGVB)
    exp_neg_log_deltas = jnp.array([jnp.exp(-log_delta) for log_delta in log_deltas])
    exp_neg_log_deltas = exp_neg_log_deltas.T  # Shape: (n_freq, n_dim)

    if Z_matrix.shape[2] > 0:  # If we have theta parameters
        # SGVB-style computation: Z_theta = Z @ theta (complex multiplication)
        Z_theta_re = jnp.einsum('fij,fj->fi', Z_re, theta_re) - jnp.einsum('fij,fj->fi', Z_im, theta_im)
        Z_theta_im = jnp.einsum('fij,fj->fi', Z_re, theta_im) + jnp.einsum('fij,fj->fi', Z_im, theta_re)

        # Residuals
        u_re = y_re - Z_theta_re
        u_im = y_im - Z_theta_im
    else:
        u_re = y_re
        u_im = y_im

    # SGVB likelihood computation
    numerator = u_re ** 2 + u_im ** 2  # (n_freq, n_dim)
    internal = numerator * exp_neg_log_deltas  # (n_freq, n_dim)
    tmp2_ = -jnp.sum(internal)  # sum over all freq and components

    log_lik = sum_log_deltas + tmp2_
    return log_lik


def multivariate_psplines_model_corrected(
        fft_data: DiscreteFFT,
        all_bases: List[jnp.ndarray],
        all_penalties: List[jnp.ndarray],
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
):
    """Corrected NumPyro model."""
    n_dim = fft_data.n_dim
    n_freq = fft_data.n_freq
    Z_matrix = fft_data.Z_matrix
    n_theta = Z_matrix.shape[2]  # Number of theta parameters

    # Sample P-splines for diagonal components (log delta^2)
    log_deltas = []
    component_idx = 0

    for j in range(n_dim):
        delta = numpyro.sample(f"delta_{j}", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample(f"phi_delta_{j}", dist.Gamma(alpha_phi, delta * beta_phi))

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample(f"weights_delta_{j}",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor(f"weights_prior_delta_{j}", log_prior_w)

        log_delta_sq = all_bases[component_idx] @ weights
        log_deltas.append(log_delta_sq)
        component_idx += 1

    # Sample P-splines for theta components
    theta_re_components = []
    theta_im_components = []

    # Real parts of theta
    for t in range(n_theta):
        delta = numpyro.sample(f"delta_theta_re_{t}", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample(f"phi_theta_re_{t}", dist.Gamma(alpha_phi, delta * beta_phi))

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample(f"weights_theta_re_{t}",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor(f"weights_prior_theta_re_{t}", log_prior_w)

        theta_re_t = all_bases[component_idx] @ weights
        theta_re_components.append(theta_re_t)
        component_idx += 1

    # Imaginary parts of theta
    for t in range(n_theta):
        delta = numpyro.sample(f"delta_theta_im_{t}", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample(f"phi_theta_im_{t}", dist.Gamma(alpha_phi, delta * beta_phi))

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample(f"weights_theta_im_{t}",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor(f"weights_prior_theta_im_{t}", log_prior_w)

        theta_im_t = all_bases[component_idx] @ weights
        theta_im_components.append(theta_im_t)
        component_idx += 1

    # Stack theta components
    if n_theta > 0:
        theta_re_all = jnp.stack(theta_re_components, axis=1)  # (n_freq, n_theta)
        theta_im_all = jnp.stack(theta_im_components, axis=1)  # (n_freq, n_theta)
    else:
        theta_re_all = jnp.zeros((n_freq, 0))
        theta_im_all = jnp.zeros((n_freq, 0))

    # Likelihood
    log_likelihood = multivariate_whittle_likelihood_corrected(
        fft_data.fft, Z_matrix, log_deltas, theta_re_all, theta_im_all
    )
    numpyro.factor("likelihood", log_likelihood)

    # Store for diagnostics
    numpyro.deterministic("log_deltas", log_deltas)
    numpyro.deterministic("theta_re_all", theta_re_all)
    numpyro.deterministic("theta_im_all", theta_im_all)
    numpyro.deterministic("log_likelihood", log_likelihood)


def reconstruct_psd_matrices_corrected(log_deltas_samples, theta_re_samples, theta_im_samples, n_dim):
    """Reconstruct PSD matrices using Cholesky decomposition."""
    log_deltas_samples = np.array(log_deltas_samples)
    theta_re_samples = np.array(theta_re_samples)
    theta_im_samples = np.array(theta_im_samples)

    n_samples = log_deltas_samples.shape[0]
    n_freq = log_deltas_samples.shape[2]
    n_theta = theta_re_samples.shape[2] if theta_re_samples.ndim == 3 else 0

    psd_samples = np.zeros((n_samples, n_freq, n_dim, n_dim), dtype=np.complex128)

    for i in range(min(100, n_samples)):
        for k in range(n_freq):
            # Build diagonal matrix D
            D = np.diag([np.exp(log_deltas_samples[i, j, k]) for j in range(n_dim)])

            # Build lower triangular matrix T
            T = np.eye(n_dim, dtype=np.complex128)
            if n_theta > 0:
                theta_idx = 0
                for row in range(1, n_dim):
                    for col in range(row):
                        if theta_idx < n_theta:
                            theta_val = theta_re_samples[i, k, theta_idx] + 1j * theta_im_samples[i, k, theta_idx]
                            T[row, col] = -theta_val
                            theta_idx += 1

            # Compute S = T^{-H} D T^{-1}
            try:
                T_inv = np.linalg.inv(T)
                S = T_inv.conj().T @ D @ T_inv
                psd_samples[i, k] = S
            except np.linalg.LinAlgError:
                # If singular, use identity
                psd_samples[i, k] = np.eye(n_dim)

    return psd_samples


if __name__ == "__main__":
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

    # Compute FFT data
    fft_data = compute_discrete_fft(x, fs=1.0)
    print(f"FFT data shape: {fft_data.fft.shape}")
    print(f"Z matrix shape: {fft_data.Z_matrix.shape}")

    # Set up P-splines
    all_bases, all_penalties = setup_psplines_for_multivariate(
        fft_data.freq, n_dim=n_dim, n_knots=15
    )

    # Run MCMC
    nuts_kernel = NUTS(multivariate_psplines_model_corrected)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500, num_chains=1)

    print("Starting MCMC sampling...")
    mcmc.run(
        jax.random.PRNGKey(0),
        fft_data=fft_data,
        all_bases=all_bases,
        all_penalties=all_penalties
    )

    # Get samples and reconstruct
    samples = mcmc.get_samples()
    print("Sampling completed!")

    mcmc.print_summary()

    # Reconstruct PSD matrices
    psd_samples = reconstruct_psd_matrices_corrected(
        samples['log_deltas'],
        samples['theta_re_all'],
        samples['theta_im_all'],
        n_dim
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

    # --- Plotting code ---
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
    plt.savefig('multivar_psd_csd_matrix_corrected.png', dpi=150)
    plt.show()

    print(f"Plot saved as 'multivar_psd_csd_matrix_corrected.png'")