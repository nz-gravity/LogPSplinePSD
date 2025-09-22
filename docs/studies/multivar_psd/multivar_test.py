import jax
import numpy as np
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.datatypes import MultivarFFT
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines
from log_psplines.samplers.multivar.multivar_nuts import multivariate_psplines_model



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
