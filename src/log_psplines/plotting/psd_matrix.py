"""Plotting utilities for multivariate PSD matrices."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional




def plot_psd_matrix(
    idata,
    n_channels: int,
    freq: np.ndarray,
    empirical_psd: np.ndarray,
    outdir: str,
    filename: str = "psd_matrix_posterior.png",
    dpi: int = 150,
    xscale='linear',
    diag_yscale='linear',
    show_coherence: bool = True,
) -> None:
    """Plot the reconstructed PSD matrix components.

    Args:
        idata: ArviZ InferenceData object containing posterior_predictive with "psd_matrix"
        n_channels: Number of channels in the multivariate PSD
        freq: Frequency array (n_freq,)
        empirical_psd: Empirical PSD matrix for comparison (n_freq, n_channels, n_channels)
        outdir: Output directory path
        filename: Output filename
        dpi: Resolution for saved figure
        show_coherence: If True (default), show coherence in the lower triangle and hide upper triangle.
                        If False, show real and imaginary parts in lower and upper triangles respectively.
    """
    if "psd_matrix" not in idata.posterior_psd:
        return

    psd_samples = idata.posterior_psd["psd_matrix"].values

    fig, axes = plt.subplots(n_channels, n_channels, figsize=(4 * n_channels, 4 * n_channels))
    if n_channels == 1:
        axes = [[axes]]
    elif n_channels == 2:
        axes = axes.reshape(n_channels, n_channels)

    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i][j]

            if i == j:  # Diagonal elements (auto-spectra)
                q05 = np.percentile(psd_samples[:, :, i, i].real, 5, axis=0)
                q50 = np.percentile(psd_samples[:, :, i, i].real, 50, axis=0)
                q95 = np.percentile(psd_samples[:, :, i, i].real, 95, axis=0)

                ax.fill_between(freq, q05, q95, alpha=0.3, color='blue', label='Model 90% CI')
                ax.plot(freq, q50, color='blue', label='Model Median')
                ax.plot(freq, empirical_psd[:, i, i].real, 'k--', alpha=0.3, label='Empirical')

                # Plot true PSD if available
                if 'true_psd' in idata.attrs and idata.attrs['true_psd'] is not None:
                    true_psd_matrix = idata.attrs['true_psd']
                    if true_psd_matrix.ndim == 3 and true_psd_matrix.shape[0] == len(freq):
                        ax.plot(freq, np.real(true_psd_matrix[:, i, i]), 'r-', alpha=0.7, label='True PSD')
                ax.set_title(f'Auto-spectrum Channel {i}')
                ax.set_yscale('log')

            elif show_coherence:
                if i > j:
                    # Lower triangle: coherence
                    # Compute coherence samples: |S_xy|^2 / (S_xx * S_yy)
                    coh_samples = np.abs(psd_samples[:, :, i, j]) ** 2 / (np.abs(psd_samples[:, :, i, i]) * np.abs(psd_samples[:, :, j, j]))

                    q05 = np.percentile(coh_samples, 5, axis=0)
                    q50 = np.percentile(coh_samples, 50, axis=0)
                    q95 = np.percentile(coh_samples, 95, axis=0)

                    ax.fill_between(freq, q05, q95, alpha=0.3, color='purple', label='Model 90% CI')
                    ax.plot(freq, q50, color='purple', label='Model Median')

                    # Empirical coherence
                    emp_coh = np.abs(empirical_psd[:, i, j]) ** 2 / (np.abs(empirical_psd[:, i, i]) * np.abs(empirical_psd[:, j, j]))
                    ax.plot(freq, emp_coh, 'k--', alpha=0.3, label='Empirical')

                    # Plot true PSD if available
                    if 'true_psd' in idata.attrs and idata.attrs['true_psd'] is not None:
                        true_psd_matrix = idata.attrs['true_psd']
                        if true_psd_matrix.ndim == 3 and true_psd_matrix.shape[0] == len(freq):
                            true_coh = np.abs(true_psd_matrix[:, i, j]) ** 2 / (np.abs(true_psd_matrix[:, i, i]) * np.abs(true_psd_matrix[:, j, j]))
                            ax.plot(freq, true_coh, 'r-', alpha=0.7, label='True Coherence')
                    ax.set_title(f'Coherence ({i},{j})')
                    ax.set_ylim(0, 1)  # Coherence ranges from 0 to 1
                else:
                    # Upper triangle: hide
                    ax.axis('off')
            else:
                if i > j:
                    # Lower triangle: real parts
                    q05 = np.percentile(psd_samples[:, :, i, j].real, 5, axis=0)
                    q50 = np.percentile(psd_samples[:, :, i, j].real, 50, axis=0)
                    q95 = np.percentile(psd_samples[:, :, i, j].real, 95, axis=0)

                    ax.fill_between(freq, q05, q95, alpha=0.3, color='green', label='Model 90% CI')
                    ax.plot(freq, q50, color='green', label='Model Median')
                    ax.plot(freq, empirical_psd[:, i, j].real, 'k--', alpha=0.3, label='Empirical')

                    # Plot true PSD if available
                    if 'true_psd' in idata.attrs and idata.attrs['true_psd'] is not None:
                        true_psd_matrix = idata.attrs['true_psd']
                        if true_psd_matrix.ndim == 3 and true_psd_matrix.shape[0] == len(freq):
                            ax.plot(freq, np.real(true_psd_matrix[:, i, j]), 'r-', alpha=0.7, label='True PSD')
                    ax.set_title(f'Cross-spectrum Real ({i},{j})')
                else:
                    # Upper triangle: imaginary parts of cross-spectra
                    q05 = np.percentile(psd_samples[:, :, i, j].imag, 5, axis=0)
                    q50 = np.percentile(psd_samples[:, :, i, j].imag, 50, axis=0)
                    q95 = np.percentile(psd_samples[:, :, i, j].imag, 95, axis=0)

                    ax.fill_between(freq, q05, q95, alpha=0.3, color='red', label='Model 90% CI')
                    ax.plot(freq, q50, color='red', label='Model Median')
                    ax.plot(freq, empirical_psd[:, i, j].imag, 'k--', alpha=0.3, label='Empirical')

                    # Plot true PSD if available
                    if 'true_psd' in idata.attrs and idata.attrs['true_psd'] is not None:
                        true_psd_matrix = idata.attrs['true_psd']
                        if true_psd_matrix.ndim == 3 and true_psd_matrix.shape[0] == len(freq):
                            ax.plot(freq, np.imag(true_psd_matrix[:, i, j]), 'r-', alpha=0.7, label='True PSD')
                    ax.set_title(f'Cross-spectrum Imag ({i},{j})')


            ax.set_xscale(xscale)
            if i == j:
                ax.set_yscale(diag_yscale)

            ax.set_xlabel('Frequency [Hz]')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/{filename}", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
