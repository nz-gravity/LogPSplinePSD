import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from dataclasses import dataclass
from typing import Tuple, List
from log_psplines.psplines.initialisation import init_basis_and_penalty
from numpyro.infer import MCMC, NUTS
from tqdm.auto import trange


@dataclass
class DiscreteFFT:
    """Discrete FFTs matching SGVB structure exactly."""
    y_re: jnp.ndarray  # (n_freq, n_dim)
    y_im: jnp.ndarray  # (n_freq, n_dim)
    Z_re: jnp.ndarray  # (n_freq, n_dim, n_theta)
    Z_im: jnp.ndarray  # (n_freq, n_dim, n_theta)
    freq: jnp.ndarray  # (n_freq,)
    n_freq: int
    n_dim: int

    @classmethod
    def compute_discrete_fft(cls, x: np.ndarray, fs: float = 1.0) -> 'DiscreteFFT':
        n_time, n_dim = x.shape
        # Standard FFT without normalization (SGVB style)
        x_fft = np.fft.fft(x, axis=0) / np.sqrt(n_time)
        freqs = np.fft.fftfreq(n_time, 1 / fs)
        pos_freq_idx = freqs > 0
        freqs = freqs[pos_freq_idx]
        x_fft = x_fft[pos_freq_idx, :]

        y_re = np.real(x_fft)
        y_im = np.imag(x_fft)
        Z_re, Z_im = compute_z_matrix(x_fft)

        return cls(
            y_re=jnp.array(y_re),
            y_im=jnp.array(y_im),
            Z_re=jnp.array(Z_re),
            Z_im=jnp.array(Z_im),
            freq=jnp.array(freqs),
            n_freq=len(freqs),
            n_dim=n_dim
        )


def compute_z_matrix(x_fft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Z matrix exactly like SGVB."""
    n, p = x_fft.shape
    if p <= 1:
        return np.zeros((n, p, 0)), np.zeros((n, p, 0))

    n_theta = int(p * (p - 1) / 2)
    Z_k = np.zeros((n, p, n_theta), dtype=np.complex64)

    for j in range(n):
        count = 0
        for i in range(1, p):
            Z_k[j, i, count:count + i] = x_fft[j, :i]
            count += i

    return np.real(Z_k), np.imag(Z_k)


def setup_psplines_corrected(
        freq: jnp.ndarray,
        n_dim: int,
        n_knots: int = 15,
        degree: int = 3,
        diff_order: int = 2
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """Setup P-splines with proper scaling."""
    freq_norm = (freq - freq.min()) / (freq.max() - freq.min())
    n_freq = len(freq_norm)

    knots = jnp.linspace(0, 1, n_knots)
    basis, penalty = init_basis_and_penalty(knots, degree, n_freq, diff_order)

    n_theta = int(n_dim * (n_dim - 1) / 2)
    total_components = n_dim + (2 if n_theta > 0 else 0)

    print(f"Components: {n_dim} delta + {2 if n_theta > 0 else 0} theta")

    all_bases = [basis for _ in range(total_components)]
    all_penalties = [penalty for _ in range(total_components)]

    return all_bases, all_penalties


def corrected_whittle_likelihood(
        data: DiscreteFFT,
        log_delta_sq: jnp.ndarray,  # Shape: (n_freq, n_dim) - log(delta^2)
        theta_re: jnp.ndarray,  # Shape: (n_freq, n_theta)
        theta_im: jnp.ndarray,  # Shape: (n_freq, n_theta)
) -> jnp.ndarray:
    """Corrected Whittle likelihood matching SGVB exactly."""

    # SGVB Step 1: sum_xγ = -sum(log(delta^2)) (negative log-determinant)
    sum_xγ = -jnp.sum(log_delta_sq)

    # SGVB Step 2: exp_xγ_inv = 1/delta^2 (precision)
    exp_xγ_inv = jnp.exp(-log_delta_sq)  # (n_freq, n_dim)

    # SGVB Step 3: Compute residuals
    if data.Z_re.shape[2] > 0:
        # Z_theta computation (complex multiplication)
        Z_theta_re = jnp.einsum('fij,fj->fi', data.Z_re, theta_re) - jnp.einsum('fij,fj->fi', data.Z_im, theta_im)
        Z_theta_im = jnp.einsum('fij,fj->fi', data.Z_re, theta_im) + jnp.einsum('fij,fj->fi', data.Z_im, theta_re)

        u_re = data.y_re - Z_theta_re
        u_im = data.y_im - Z_theta_im
    else:
        u_re = data.y_re
        u_im = data.y_im

    # SGVB Step 4: numerator = |residual|^2
    numerator = u_re ** 2 + u_im ** 2  # (n_freq, n_dim)

    # SGVB Step 5: internal = |residual|^2 / delta^2
    internal = numerator * exp_xγ_inv  # (n_freq, n_dim)

    # SGVB Step 6: tmp2_ = -sum(|residual|^2 / delta^2)
    tmp2_ = -jnp.sum(internal)

    # SGVB Step 7: Final likelihood
    log_lik = sum_xγ + tmp2_

    return log_lik


def corrected_multivariate_psplines_model(
        data: DiscreteFFT,
        all_bases: List[jnp.ndarray],
        all_penalties: List[jnp.ndarray],
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
):
    """Corrected model with proper scaling and signs."""
    n_dim = data.n_dim
    n_freq = data.n_freq
    n_theta = data.Z_re.shape[2]

    component_idx = 0

    # Sample delta components (modeling log(delta^2), i.e., log variance)
    log_delta_components = []
    for j in range(n_dim):
        delta = numpyro.sample(f"delta_{j}", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample(f"phi_delta_{j}", dist.Gamma(alpha_phi, beta_phi))  # Fixed: remove delta *

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample(f"weights_delta_{j}",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor(f"weights_prior_delta_{j}", log_prior_w)

        # This gives log(delta^2) directly
        log_delta_sq_j = all_bases[component_idx] @ weights
        log_delta_components.append(log_delta_sq_j)
        component_idx += 1

    # Stack all log(delta^2) components
    log_delta_sq = jnp.stack(log_delta_components, axis=1)  # (n_freq, n_dim)

    # Sample theta components
    if n_theta > 0:
        # Real theta - single P-spline broadcast to all theta components
        delta = numpyro.sample("delta_theta_re", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_re", dist.Gamma(alpha_phi, beta_phi))  # Fixed: remove delta *

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_re",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_re", log_prior_w)

        theta_re_base = all_bases[component_idx] @ weights
        theta_re = jnp.tile(theta_re_base[:, None], (1, max(1, n_theta)))
        component_idx += 1

        # Imaginary theta
        delta = numpyro.sample("delta_theta_im", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_im", dist.Gamma(alpha_phi, beta_phi))  # Fixed: remove delta *

        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_im",
                                 dist.Normal(0, 1).expand([k]).to_event(1))

        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_im", log_prior_w)

        theta_im_base = all_bases[component_idx] @ weights
        theta_im = jnp.tile(theta_im_base[:, None], (1, max(1, n_theta)))
    else:
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))

    # Corrected likelihood
    log_likelihood = corrected_whittle_likelihood(data, log_delta_sq, theta_re, theta_im)
    numpyro.factor("likelihood", log_likelihood)

    # Store diagnostics
    numpyro.deterministic("log_delta_sq", log_delta_sq)
    numpyro.deterministic("theta_re", theta_re)
    numpyro.deterministic("theta_im", theta_im)
    numpyro.deterministic("log_likelihood", log_likelihood)


def reconstruct_psd_from_cholesky(log_delta_sq_samples, theta_re_samples, theta_im_samples):
    """Properly reconstruct PSD matrix from Cholesky components."""
    n_samples, n_freq, n_dim = log_delta_sq_samples.shape
    n_theta = theta_re_samples.shape[2] if theta_re_samples.ndim > 2 else 0

    n_samps = min(50, n_samples)
    psd_samples = np.zeros((n_samps, n_freq, n_dim, n_dim), dtype=complex)

    for i in trange(n_samps):
        for k in range(n_freq):
            # Build D matrix (diagonal with delta^2 values)
            D = np.diag(np.exp(log_delta_sq_samples[i, k, :]))

            # Build T matrix (lower triangular)
            T = np.eye(n_dim, dtype=complex)

            if n_theta > 0 and n_dim > 1:
                # Fill lower triangular part with -theta values
                theta_idx = 0
                for row in range(1, n_dim):
                    for col in range(row):
                        if theta_idx < n_theta:
                            theta_val = (theta_re_samples[i, k, theta_idx] +
                                         1j * theta_im_samples[i, k, theta_idx])
                            T[row, col] = -theta_val  # Note the negative sign
                            theta_idx += 1

            # Reconstruct PSD: S = T^{-1} D T^{-H}
            try:
                T_inv = np.linalg.inv(T)
                S = T_inv @ D @ T_inv.conj().T
                psd_samples[i, k] = S
            except np.linalg.LinAlgError:
                # Fallback to diagonal if singular
                psd_samples[i, k] = D

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

    print(f"VARMA data shape: {x.shape}, dim={n_dim}")

    # Compute FFT data (corrected)
    data = DiscreteFFT.compute_discrete_fft(x, fs=1.0)
    print(f"FFT shapes: y_re={data.y_re.shape}, Z_re={data.Z_re.shape}")

    # Setup P-splines with better scaling
    all_bases, all_penalties = setup_psplines_corrected(
        data.freq, n_dim=n_dim, n_knots=10, degree=3
    )

    # Run MCMC with more conservative settings
    nuts_kernel = NUTS(corrected_multivariate_psplines_model)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=500, num_chains=1)

    print("Starting corrected MCMC...")
    mcmc.run(
        jax.random.PRNGKey(0),
        data=data,
        all_bases=all_bases,
        all_penalties=all_penalties,
        alpha_phi=1.0,  # More reasonable priors
        beta_phi=1.0,
        alpha_delta=1.0,
        beta_delta=1.0
    )

    samples = mcmc.get_samples()
    mcmc.print_summary()

    # Check results
    ll_samples = samples['log_likelihood']
    print(f"Log likelihood range: {ll_samples.min():.2f} to {ll_samples.max():.2f}")

    # Properly reconstruct PSD matrices
    print("Reconstructing PSD matrices from Cholesky components...")
    psd_reconstructed = reconstruct_psd_from_cholesky(
        samples['log_delta_sq'],
        samples['theta_re'],
        samples['theta_im']
    )

    print(f"Reconstructed PSD shape: {psd_reconstructed.shape}")

    # Empirical periodogram (with proper normalization)
    fft_complex = data.y_re + 1j * data.y_im
    n_time = len(x)

    # Since we normalized FFT by sqrt(n_time), periodogram = |FFT|^2 * 2 (factor 2 for one-sided)
    empirical_psd_matrix = np.zeros((data.n_freq, n_dim, n_dim), dtype=complex)
    for i in range(n_dim):
        for j in range(n_dim):
            # Cross-spectrum with factor of 2 for one-sided spectrum
            empirical_psd_matrix[:, i, j] = 2 * (fft_complex[:, i] *
                                                 np.conj(fft_complex[:, j]))

    # Compare scales
    emp_diag_range = ([empirical_psd_matrix[:, i, i].real.min() for i in range(n_dim)],
                      [empirical_psd_matrix[:, i, i].real.max() for i in range(n_dim)])
    model_diag_range = ([psd_reconstructed[:, :, i, i].real.min() for i in range(n_dim)],
                        [psd_reconstructed[:, :, i, i].real.max() for i in range(n_dim)])

    print(f"Empirical PSD diagonal ranges: {emp_diag_range}")
    print(f"Model PSD diagonal ranges: {model_diag_range}")

    # Plot with properly reconstructed PSD
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_dim, n_dim, figsize=(4 * n_dim, 4 * n_dim))
    if n_dim == 1:
        axes = [[axes]]
    elif n_dim == 2:
        axes = axes.reshape(n_dim, n_dim)

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i][j]

            if i == j:  # Diagonal: PSD
                # Model quantiles
                q05 = np.percentile(psd_reconstructed[:, :, i, i].real, 5, axis=0)
                q50 = np.percentile(psd_reconstructed[:, :, i, i].real, 50, axis=0)
                q95 = np.percentile(psd_reconstructed[:, :, i, i].real, 95, axis=0)

                ax.fill_between(data.freq, q05, q95, alpha=0.3, color='blue', label='Model 90% CI')
                ax.plot(data.freq, q50, color='blue', label='Model Median')
                ax.plot(data.freq, empirical_psd_matrix[:, i, i].real, 'k--',
                        alpha=0.2, zorder=-10,  label='Empirical')
                ax.set_title(f'PSD Component {i}')
                ax.set_yscale('log')

            elif i > j:  # Lower triangle: Real part of CSD
                q05 = np.percentile(psd_reconstructed[:, :, i, j].real, 5, axis=0)
                q50 = np.percentile(psd_reconstructed[:, :, i, j].real, 50, axis=0)
                q95 = np.percentile(psd_reconstructed[:, :, i, j].real, 95, axis=0)

                ax.fill_between(data.freq, q05, q95, alpha=0.3, color='green', label='Model 90% CI')
                ax.plot(data.freq, q50, color='green', label='Model Median')
                ax.plot(data.freq, empirical_psd_matrix[:, i, j].real, 'k--',
                        alpha=0.2, zorder=-10,  label='Empirical')
                ax.set_title(f'CSD Real ({i},{j})')

            else:  # Upper triangle: Imaginary part of CSD
                q05 = np.percentile(psd_reconstructed[:, :, i, j].imag, 5, axis=0)
                q50 = np.percentile(psd_reconstructed[:, :, i, j].imag, 50, axis=0)
                q95 = np.percentile(psd_reconstructed[:, :, i, j].imag, 95, axis=0)

                ax.fill_between(data.freq, q05, q95, alpha=0.3, color='red', label='Model 90% CI')
                ax.plot(data.freq, q50, color='red', label='Model Median')
                ax.plot(data.freq, empirical_psd_matrix[:, i, j].imag, 'k--',
                        alpha=0.2, zorder=-10,  label='Empirical')
                ax.set_title(f'CSD Imag ({i},{j})')

            ax.set_xlabel('Frequency')
            ax.legend()

    plt.tight_layout()
    plt.savefig('properly_reconstructed_psd.png', dpi=150)

    print("Now the PSD should match the empirical scale!")

    # Additional diagnostic: Print some scale comparisons
    print("\nDiagonal PSD scale comparison at first frequency:")
    for i in range(n_dim):
        emp_val = empirical_psd_matrix[0, i, i].real
        model_median = np.percentile(psd_reconstructed[:, 0, i, i].real, 50)
        print(f"  Component {i}: Empirical={emp_val:.6f}, Model median={model_median:.6f}")

    print("\nOff-diagonal CSD scale comparison at first frequency:")
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            emp_real = empirical_psd_matrix[0, i, j].real
            emp_imag = empirical_psd_matrix[0, i, j].imag
            model_real = np.percentile(psd_reconstructed[:, 0, i, j].real, 50)
            model_imag = np.percentile(psd_reconstructed[:, 0, i, j].imag, 50)
            print(f"  ({i},{j}): Emp={emp_real:.6f}+{emp_imag:.6f}i, Model={model_real:.6f}+{model_imag:.6f}i")
