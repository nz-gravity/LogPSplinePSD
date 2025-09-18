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
    """
    fft: jnp.ndarray  # Shape: (n_freq, n_dim, 2) - real and imag parts for each timeseries
    freq: jnp.ndarray  # Shape: (n_freq,) - frequency grid
    n_freq: int  # Number of frequencies
    n_dim: int  # Number of timeseries


def compute_discrete_fft(x: np.ndarray, fs: float = 1.0) -> DiscreteFFT:
    """
    Compute discrete FFTs for each timeseries, storing real and imaginary parts separately.

    Parameters
    ----------
    x : np.ndarray, shape (n_time, n_dim)
        Multivariate time series
    fs : float
        Sampling frequency

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
    return DiscreteFFT(
        fft=jnp.array(x_fft_split),
        freq=jnp.array(freqs),
        n_freq=n_freq,
        n_dim=n_dim
    )


def setup_psplines_for_sequential_multivariate(
        freq: jnp.ndarray,
        n_dim: int,
        n_knots: int = 10,
        degree: int = 3,
        diff_order: int = 2
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Set up P-splines for sequential multivariate model.

    Total components needed:
    - n_dim diagonal components (log delta^2)
    - Sum_{j=1}^{n_dim-1} 2*j off-diagonal components (real/imag theta)
    """
    freq_norm = (freq - freq.min()) / (freq.max() - freq.min())
    n_freq = len(freq_norm)

    knots = jnp.linspace(0, 1, n_knots)
    basis, penalty = init_basis_and_penalty(
        knots, degree, n_freq, diff_order
    )

    # Calculate total number of components
    n_diagonal = n_dim
    n_theta_pairs = sum(j for j in range(n_dim))  # 0 + 1 + 2 + ... + (n_dim-1)
    n_theta_components = 2 * n_theta_pairs  # real + imaginary
    total_components = n_diagonal + n_theta_components

    print(f"Total components needed: {total_components}")
    print(f"  - Diagonal (delta): {n_diagonal}")
    print(f"  - Off-diagonal (theta): {n_theta_components}")

    # For now, we have the same basis for each component 
    all_bases = [basis for _ in range(total_components)]
    all_penalties = [penalty for _ in range(total_components)]

    return all_bases, all_penalties


def multivariate_whittle_likelihood_sequential(
        fft: jnp.ndarray,  # shape (n_freq, n_dim, 2)
        log_deltas: List[jnp.ndarray],  # List of n_dim arrays, each shape (n_freq,)
        theta_res: List[jnp.ndarray],  # List of n_dim arrays, each shape (n_freq, j)
        theta_ims: List[jnp.ndarray],  # List of n_dim arrays, each shape (n_freq, j)
) -> jnp.ndarray:
    """
    Sequential multivariate Whittle likelihood following the Cholesky decomposition.

    For each component j, compute:
    L_j ∝ ∏_k δ_{jk}^{-2} exp(-|d_j(f_k) - ∑_{l=1}^{j-1} θ_{jl}^{(k)} d_l(f_k)|^2 / δ_{jk}^2)
    """
    n_freq, n_dim, _ = fft.shape
    y_re = fft[:, :, 0]  # (n_freq, n_dim)
    y_im = fft[:, :, 1]  # (n_freq, n_dim)

    total_log_lik = 0.0

    # For each component j = 0, 1, 2, ... (0-indexed)
    for j in range(n_dim):

        # Get log(delta_j^2) for this component
        log_delta_j_sq = log_deltas[j]  # shape (n_freq,)

        # Start with the raw data for component j
        residual_re = y_re[:, j]  # shape (n_freq,)
        residual_im = y_im[:, j]  # shape (n_freq,)

        # Subtract contributions from previous components l = 0, 1, ..., j-1
        for l in range(j):
            # theta_{jl} for this component pair (j,l)
            theta_jl_re = theta_res[j][:, l]  # shape (n_freq,)
            theta_jl_im = theta_ims[j][:, l]  # shape (n_freq,)

            # Subtract θ_{jl}^{(k)} * d_l(f_k) from residual
            # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            residual_re = residual_re - (theta_jl_re * y_re[:, l] - theta_jl_im * y_im[:, l])
            residual_im = residual_im - (theta_jl_re * y_im[:, l] + theta_jl_im * y_re[:, l])

        # Compute likelihood for component j
        # |residual|^2 = residual_re^2 + residual_im^2
        residual_sq = residual_re ** 2 + residual_im ** 2

        # δ_{jk}^2 = exp(log_delta_j_sq)
        delta_j_sq = jnp.exp(log_delta_j_sq)

        # Log-likelihood: ∑_k [-log(δ_{jk}^2) - |residual|^2 / δ_{jk}^2]
        component_log_lik = jnp.sum(-log_delta_j_sq - residual_sq / delta_j_sq)
        total_log_lik += component_log_lik

    return total_log_lik


def multivariate_psplines_model_sequential(
        fft_data: DiscreteFFT,
        all_bases: List[jnp.ndarray],
        all_penalties: List[jnp.ndarray],
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
):
    """
    NumPyro model using sequential Whittle likelihood structure.
    """
    n_dim = fft_data.n_dim
    n_freq = fft_data.n_freq

    # Sample P-spline components for each diagonal element (log delta^2)
    log_deltas = []
    component_idx = 0

    for j in range(n_dim):
        with numpyro.plate(f"delta_component_{j}", 1):
            delta = numpyro.sample(f"delta_{j}", dist.Gamma(alpha_delta, beta_delta))
            phi = numpyro.sample(f"phi_delta_{j}", dist.Gamma(alpha_phi, delta * beta_phi))

            k = all_penalties[component_idx].shape[0]
            weights = numpyro.sample(f"weights_delta_{j}",
                                     dist.Normal(0, 1).expand([k]).to_event(1))
            weights = weights.reshape(-1)  # Ensure 1D shape

            # Prior on weights
            wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
            log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
            numpyro.factor(f"weights_prior_delta_{j}", log_prior_w)

            # Compute log(delta^2) for this component
            log_delta_sq = all_bases[component_idx] @ weights
            log_deltas.append(log_delta_sq)
            component_idx += 1

    # Sample P-spline components for theta parameters (off-diagonal)
    theta_res = []  # Real parts
    theta_ims = []  # Imaginary parts

    for j in range(n_dim):
        # Component j has theta parameters for l = 0, 1, ..., j-1
        theta_re_j = jnp.zeros((n_freq, j))
        theta_im_j = jnp.zeros((n_freq, j))

        for l in range(j):
            # Real part of theta_{jl}
            with numpyro.plate(f"theta_re_component_{j}_{l}", 1):
                delta = numpyro.sample(f"delta_theta_re_{j}_{l}",
                                       dist.Gamma(alpha_delta, beta_delta))
                phi = numpyro.sample(f"phi_theta_re_{j}_{l}",
                                     dist.Gamma(alpha_phi, delta * beta_phi))

                k = all_penalties[component_idx].shape[0]
                weights = numpyro.sample(f"weights_theta_re_{j}_{l}",
                                         dist.Normal(0, 1).expand([k]).to_event(1))

                weights = weights.reshape(-1)  # Ensure 1D shape
                wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
                log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
                numpyro.factor(f"weights_prior_theta_re_{j}_{l}", log_prior_w)

                theta_re_jl = all_bases[component_idx] @ weights
                theta_re_j = theta_re_j.at[:, l].set(theta_re_jl)
                component_idx += 1

            # Imaginary part of theta_{jl}
            with numpyro.plate(f"theta_im_component_{j}_{l}", 1):
                delta = numpyro.sample(f"delta_theta_im_{j}_{l}",
                                       dist.Gamma(alpha_delta, beta_delta))
                phi = numpyro.sample(f"phi_theta_im_{j}_{l}",
                                     dist.Gamma(alpha_phi, delta * beta_phi))

                k = all_penalties[component_idx].shape[0]
                weights = numpyro.sample(f"weights_theta_im_{j}_{l}",
                                         dist.Normal(0, 1).expand([k]).to_event(1))
                weights = weights.reshape(-1)  # Ensure 1D shape
                wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
                log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
                numpyro.factor(f"weights_prior_theta_im_{j}_{l}", log_prior_w)

                theta_im_jl = all_bases[component_idx] @ weights
                theta_im_j = theta_im_j.at[:, l].set(theta_im_jl)
                component_idx += 1

        theta_res.append(theta_re_j)
        theta_ims.append(theta_im_j)

    # Sequential Whittle likelihood
    log_likelihood = multivariate_whittle_likelihood_sequential(
        fft_data.fft, log_deltas, theta_res, theta_ims
    )
    numpyro.factor("likelihood", log_likelihood)

    # Store for diagnostics
    numpyro.deterministic("log_deltas", log_deltas)
    numpyro.deterministic("theta_res", theta_res)
    numpyro.deterministic("theta_ims", theta_ims)
    numpyro.deterministic("log_likelihood", log_likelihood)


def reconstruct_psd_matrices(log_deltas, theta_res, theta_ims):
    """
    Reconstruct the spectral density matrices from the Cholesky components.
    S = T^{-H} D T^{-1} where T is lower triangular and D is diagonal.
    """
    n_dim = len(log_deltas)
    n_freq = log_deltas[0].shape[0]

    psd_matrices = jnp.zeros((n_freq, n_dim, n_dim), dtype=complex)

    for k in range(n_freq):
        # Build T matrix (lower triangular)
        T = jnp.eye(n_dim, dtype=complex)
        for j in range(1, n_dim):
            # Use the actual number of columns in theta_res[j] (should be j)
            n_theta = theta_res[j].shape[1]
            for l in range(n_theta):
                theta_jl = theta_res[j][k, l] + 1j * theta_ims[j][k, l]
                T = T.at[j, l].set(-theta_jl)

        # Build D matrix (diagonal)
        D = jnp.diag(jnp.array([jnp.exp(log_deltas[j][k]) for j in range(n_dim)]))

        # S = T^{-H} D T^{-1}
        T_inv = jnp.linalg.inv(T)
        S = T_inv.conj().T @ D @ T_inv
        psd_matrices = psd_matrices.at[k].set(S)

    return psd_matrices




if __name__ == "__main__":
    # Generate or load your 3D time series data
    np.random.seed(42)
    n_time, n_dim = 1000, 3
    x = np.random.randn(n_time, n_dim)

    # Add some correlation structure
    x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]
    x[:, 2] = 0.5 * x[:, 0] + 0.4 * x[:, 1] + 0.3 * x[:, 2]

    # Compute discrete FFTs for each timeseries
    fft_data = compute_discrete_fft(x, fs=1.0)
    print(f"FFT data shape: {fft_data.fft.shape}")
    print(f"Frequency range: {fft_data.freq.min():.3f} to {fft_data.freq.max():.3f}")

    # Set up P-splines for sequential model
    all_bases, all_penalties = setup_psplines_for_sequential_multivariate(
        fft_data.freq, n_dim=3, n_knots=15
    )

    # Run MCMC
    nuts_kernel = NUTS(multivariate_psplines_model_sequential)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=2)

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
    theta_res_samples = samples['theta_res']
    theta_ims_samples = samples['theta_ims']

    # Reconstruct for a few samples
    n_samples_to_reconstruct = min(100, len(log_deltas_samples))
    psd_samples = []

    for i in range(n_samples_to_reconstruct):
        log_deltas_i = [log_deltas_samples[i][j] for j in range(n_dim)]
        theta_res_i = [theta_res_samples[i][j] for j in range(n_dim)]
        theta_ims_i = [theta_ims_samples[i][j] for j in range(n_dim)]

        psd_i = reconstruct_psd_matrices(log_deltas_i, theta_res_i, theta_ims_i)
        psd_samples.append(psd_i)

    psd_samples = jnp.array(psd_samples)
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

    # --- Plot matrix of PSD and CSD with posterior quantiles and data ---
    import matplotlib.pyplot as plt
    import numpy as np
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

    fig, axes = plt.subplots(n_dim, n_dim, figsize=(3*n_dim, 3*n_dim), sharex=True)
    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            # Diagonal: PSD
            if i == j:
                # Posterior quantiles
                q05, q50, q95 = get_quantiles(psd_samples[:, :, i, i].real, axis=0)
                ax.fill_between(freq, q05, q95, color='blue', alpha=0.2, label='Posterior 90%')
                ax.plot(freq, q50, color='blue', label='Posterior median')
                # Empirical periodogram
                ax.plot(freq, periodogram[:, i], color='black', alpha=0.3, lw=1, label='Empirical', zorder=-1)
                ax.set_title(f"PSD {i}")
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
    plt.savefig('multivar_psd_csd_matrix_with_quantiles.png')
    plt.show()
