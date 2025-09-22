"""Plotting utilities for multivariate PSD matrices."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def compute_empirical_psd(fft_data_re: np.ndarray, fft_data_im: np.ndarray, n_channels: int) -> np.ndarray:
    """Compute empirical PSD matrix for comparison from FFT data.

    Args:
        fft_data_re: Real part of FFT data (n_freq, n_channels)
        fft_data_im: Imaginary part of FFT data (n_freq, n_channels)
        n_channels: Number of channels

    Returns:
        Empirical PSD matrix (n_freq, n_channels, n_channels) as complex array
    """
    fft_complex = fft_data_re + 1j * fft_data_im
    empirical_psd = np.zeros((fft_data_re.shape[0], n_channels, n_channels), dtype=complex)

    for i in range(n_channels):
        for j in range(n_channels):
            empirical_psd[:, i, j] = 2 * (fft_complex[:, i] * np.conj(fft_complex[:, j]))

    return empirical_psd


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
    """
    if "psd_matrix" not in idata.posterior_predictive:
        return

    psd_samples = idata.posterior_predictive["psd_matrix"].values

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
                ax.set_title(f'Auto-spectrum Channel {i}')
                ax.set_yscale('log')

            elif i > j:  # Lower triangle (real parts of cross-spectra)
                q05 = np.percentile(psd_samples[:, :, i, j].real, 5, axis=0)
                q50 = np.percentile(psd_samples[:, :, i, j].real, 50, axis=0)
                q95 = np.percentile(psd_samples[:, :, i, j].real, 95, axis=0)

                ax.fill_between(freq, q05, q95, alpha=0.3, color='green', label='Model 90% CI')
                ax.plot(freq, q50, color='green', label='Model Median')
                ax.plot(freq, empirical_psd[:, i, j].real, 'k--', alpha=0.3, label='Empirical')
                ax.set_title(f'Cross-spectrum Real ({i},{j})')

            else:  # Upper triangle (imaginary parts of cross-spectra)
                q05 = np.percentile(psd_samples[:, :, i, j].imag, 5, axis=0)
                q50 = np.percentile(psd_samples[:, :, i, j].imag, 50, axis=0)
                q95 = np.percentile(psd_samples[:, :, i, j].imag, 95, axis=0)

                ax.fill_between(freq, q05, q95, alpha=0.3, color='red', label='Model 90% CI')
                ax.plot(freq, q50, color='red', label='Model Median')
                ax.plot(freq, empirical_psd[:, i, j].imag, 'k--', alpha=0.3, label='Empirical')
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
