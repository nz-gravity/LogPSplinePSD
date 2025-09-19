import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from typing import List
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.datatypes import MultivarFFT
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines


def whittle_likelihood(
        data: MultivarFFT,
        log_delta_sq: jnp.ndarray,  # (n_freq, n_dim)
        theta_re: jnp.ndarray,      # (n_freq, n_theta)
        theta_im: jnp.ndarray       # (n_freq, n_theta)
) -> jnp.ndarray:
    """
    Multivariate Whittle likelihood for Cholesky PSD parameterization.

    .. math::
        \log L = -\sum_{j,k} \log \delta_{jk}^2 - \sum_{j,k} \frac{|y_{jk} - \sum_l \theta_{jl} y_{lk}|^2}{\delta_{jk}^2}
    """
    sum_log_det = -jnp.sum(log_delta_sq)
    exp_neg_log_delta = jnp.exp(-log_delta_sq)
    if data.Z_re.shape[2] > 0:
        Z_theta_re = jnp.einsum('fij,fj->fi', data.Z_re, theta_re) - jnp.einsum('fij,fj->fi', data.Z_im, theta_im)
        Z_theta_im = jnp.einsum('fij,fj->fi', data.Z_re, theta_im) + jnp.einsum('fij,fj->fi', data.Z_im, theta_re)
        u_re = data.y_re - Z_theta_re
        u_im = data.y_im - Z_theta_im
    else:
        u_re = data.y_re
        u_im = data.y_im
    numerator = u_re ** 2 + u_im ** 2
    internal = numerator * exp_neg_log_delta
    tmp2 = -jnp.sum(internal)
    return sum_log_det + tmp2

def multivariate_psplines_model(
        data: MultivarFFT,
        all_bases: List[jnp.ndarray],
        all_penalties: List[jnp.ndarray],
        alpha_phi: float = 1.0,
        beta_phi: float = 1.0,
        alpha_delta: float = 1e-4,
        beta_delta: float = 1e-4,
):
    """
    NumPyro model for multivariate PSD estimation using P-splines and Cholesky parameterization.
    """
    n_dim = data.n_dim
    n_freq = data.n_freq
    n_theta = data.Z_re.shape[2]
    component_idx = 0
    log_delta_components = []
    for j in range(n_dim):
        delta = numpyro.sample(f"delta_{j}", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample(f"phi_delta_{j}", dist.Gamma(alpha_phi, delta*beta_phi))
        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample(f"weights_delta_{j}", dist.Normal(0, 1).expand((k,)).to_event(1))
        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor(f"weights_prior_delta_{j}", log_prior_w)
        log_delta_sq_j = all_bases[component_idx] @ weights
        log_delta_components.append(log_delta_sq_j)
        component_idx += 1
    log_delta_sq = jnp.stack(log_delta_components, axis=1)
    if n_theta > 0:
        delta = numpyro.sample("delta_theta_re", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_re", dist.Gamma(alpha_phi, delta*beta_phi))
        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_re", dist.Normal(0, 1).expand((k,)).to_event(1))
        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_re", log_prior_w)
        theta_re_base = all_bases[component_idx] @ weights
        theta_re = jnp.tile(theta_re_base[:, None], (1, max(1, n_theta)))
        component_idx += 1
        delta = numpyro.sample("delta_theta_im", dist.Gamma(alpha_delta, beta_delta))
        phi = numpyro.sample("phi_theta_im", dist.Gamma(alpha_phi, delta*beta_phi))
        k = all_penalties[component_idx].shape[0]
        weights = numpyro.sample("weights_theta_im", dist.Normal(0, 1).expand((k,)).to_event(1))
        wPw = jnp.dot(weights, jnp.dot(all_penalties[component_idx], weights))
        log_prior_w = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
        numpyro.factor("weights_prior_theta_im", log_prior_w)
        theta_im_base = all_bases[component_idx] @ weights
        theta_im = jnp.tile(theta_im_base[:, None], (1, max(1, n_theta)))
    else:
        theta_re = jnp.zeros((n_freq, 0))
        theta_im = jnp.zeros((n_freq, 0))
    log_likelihood = whittle_likelihood(data, log_delta_sq, theta_re, theta_im)
    numpyro.factor("likelihood", log_likelihood)
    numpyro.deterministic("log_delta_sq", log_delta_sq)
    numpyro.deterministic("theta_re", theta_re)
    numpyro.deterministic("theta_im", theta_im)
    numpyro.deterministic("log_likelihood", log_likelihood)

if __name__ == "__main__":
    np.random.seed(42)
    varma = VARMAData()
    x = varma.data
    n_dim = varma.dim
    print(f"VARMA data shape: {x.shape}, dim={n_dim}")
    data = MultivarFFT.compute_fft(x, fs=1.0)
    print(f"FFT shapes: y_re={data.y_re.shape}, Z_re={data.Z_re.shape}")

    # Use MultivariateLogPSplines to set up basis/penalty matrices
    psplines_model = MultivariateLogPSplines.from_multivar_fft(
        data,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2
    )
    all_bases, all_penalties = psplines_model.get_all_bases_and_penalties()

    nuts_kernel = NUTS(multivariate_psplines_model)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=500, num_chains=1)
    print("Starting MCMC...")
    mcmc.run(
        jax.random.PRNGKey(0),
        data=data,
        all_bases=all_bases,
        all_penalties=all_penalties,
        alpha_phi=1.0,
        beta_phi=1.0,
        alpha_delta=1.0,
        beta_delta=1.0
    )
    samples = mcmc.get_samples()
    mcmc.print_summary()
    ll_samples = samples['log_likelihood']
    print(f"Log likelihood range: {ll_samples.min():.2f} to {ll_samples.max():.2f}")
    print("Reconstructing PSD matrices from Cholesky components...")
    psd_reconstructed = psplines_model.reconstruct_psd_matrix(
        samples['log_delta_sq'],
        samples['theta_re'],
        samples['theta_im']
    )
    print(f"Reconstructed PSD shape: {psd_reconstructed.shape}")
    fft_complex = data.y_re + 1j * data.y_im
    n_time = len(x)
    empirical_psd_matrix = np.zeros((data.n_freq, n_dim, n_dim), dtype=complex)
    for i in range(n_dim):
        for j in range(n_dim):
            empirical_psd_matrix[:, i, j] = 2 * (fft_complex[:, i] * np.conj(fft_complex[:, j]))
    emp_diag_range = ([empirical_psd_matrix[:, i, i].real.min() for i in range(n_dim)],
                      [empirical_psd_matrix[:, i, i].real.max() for i in range(n_dim)])
    model_diag_range = ([psd_reconstructed[:, :, i, i].real.min() for i in range(n_dim)],
                        [psd_reconstructed[:, :, i, i].real.max() for i in range(n_dim)])
    print(f"Empirical PSD diagonal ranges: {emp_diag_range}")
    print(f"Model PSD diagonal ranges: {model_diag_range}")
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(4 * n_dim, 4 * n_dim))
    if n_dim == 1:
        axes = [[axes]]
    elif n_dim == 2:
        axes = axes.reshape(n_dim, n_dim)
    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i][j]
            if i == j:
                q05 = np.percentile(psd_reconstructed[:, :, i, i].real, 5, axis=0)
                q50 = np.percentile(psd_reconstructed[:, :, i, i].real, 50, axis=0)
                q95 = np.percentile(psd_reconstructed[:, :, i, i].real, 95, axis=0)
                ax.fill_between(data.freq, q05, q95, alpha=0.3, color='blue', label='Model 90% CI')
                ax.plot(data.freq, q50, color='blue', label='Model Median')
                ax.plot(data.freq, empirical_psd_matrix[:, i, i].real, 'k--',
                        alpha=0.2, zorder=-10,  label='Empirical')
                ax.set_title(f'PSD Component {i}')
                ax.set_yscale('log')
            elif i > j:
                q05 = np.percentile(psd_reconstructed[:, :, i, j].real, 5, axis=0)
                q50 = np.percentile(psd_reconstructed[:, :, i, j].real, 50, axis=0)
                q95 = np.percentile(psd_reconstructed[:, :, i, j].real, 95, axis=0)
                ax.fill_between(data.freq, q05, q95, alpha=0.3, color='green', label='Model 90% CI')
                ax.plot(data.freq, q50, color='green', label='Model Median')
                ax.plot(data.freq, empirical_psd_matrix[:, i, j].real, 'k--',
                        alpha=0.2, zorder=-10,  label='Empirical')
                ax.set_title(f'CSD Real ({i},{j})')
            else:
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
    plt.savefig('psd_matrix_plot.png', dpi=150)
    print("Now the PSD should match the empirical scale!")
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
