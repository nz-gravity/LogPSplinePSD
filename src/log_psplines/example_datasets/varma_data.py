from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft


class VARMAData:
    """
    Simulate Vector Autoregressive Moving Average (VARMA) processes and compute related spectral properties.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        sigma: np.ndarray = np.array([[1.0, 0.9], [0.9, 1.0]]),
        var_coeffs: np.ndarray = np.array(
            [[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]]
        ),
        vma_coeffs: np.ndarray = np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        seed: int = None,
        fs: float = 1.0,
    ):
        """
        Initialize the SimVARMA class.

        Args:
            n_samples (int): Number of samples to generate.
            var_coeffs (np.ndarray): VAR coefficient array.
            vma_coeffs (np.ndarray): VMA coefficient array.
            sigma (np.ndarray): Covariance matrix or scalar variance.
        """
        self.n_samples = n_samples
        self.var_coeffs = var_coeffs
        self.vma_coeffs = vma_coeffs
        self.sigma = sigma
        self.dim = vma_coeffs.shape[1]
        self.psd_scaling = 1.0
        self.n_freq_samples = n_samples // 2

        self.fs = float(fs)
        # Frequency grid in Hz (drop the DC bin to match PSD plotting expectations)
        self.freq = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)[1:]
        self.time = np.arange(n_samples) / self.fs
        self.data = None  # set in "resimulate"
        self.periodogram = None  # set in "resimulate"
        self.welch_psd = None  # set in "resimulate"
        self.welch_f = None  # set in "resimulate"
        self.resimulate(seed=seed)

        self.channel_stds = np.std(self.data, axis=0)
        self.psd_scaling = float(np.std(self.data) ** 2)
        self.psd = _calculate_true_varma_psd(
            self.freq,
            self.dim,
            self.var_coeffs,
            self.vma_coeffs,
            self.sigma,
            self.fs,
            self.channel_stds,
            self.psd_scaling,
        )

    def resimulate(self, seed=None):
        """
        Simulate VARMA process.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: Simulated VARMA process data.
        """
        if seed is not None:
            np.random.seed(seed)

        lag_ma = self.vma_coeffs.shape[0]
        lag_ar = self.var_coeffs.shape[0]

        if self.sigma.shape[0] == 1:
            cov_matrix = np.identity(self.dim) * self.sigma
        else:
            cov_matrix = self.sigma

        x_init = np.zeros((lag_ar + 1, self.dim))
        x = np.empty((self.n_samples + 101, self.dim))
        x[:] = np.nan
        x[: lag_ar + 1] = x_init
        epsilon = np.random.multivariate_normal(
            np.zeros(self.dim), cov_matrix, size=[lag_ma]
        )

        for i in range(lag_ar + 1, x.shape[0]):
            epsilon = np.concatenate(
                [
                    np.random.multivariate_normal(
                        np.zeros(self.dim), cov_matrix, size=[1]
                    ),
                    epsilon[:-1],
                ]
            )
            x[i] = np.sum(
                np.matmul(
                    self.var_coeffs,
                    x[i - 1 : i - lag_ar - 1 : -1][..., np.newaxis],
                ),
                axis=(0, -1),
            ) + np.sum(
                np.matmul(self.vma_coeffs, epsilon[..., np.newaxis]),
                axis=(0, -1),
            )

        self.data = x[101:]

    def get_periodogram(self):
        """
        Return a one-sided PSD matrix estimate for the simulated data.
        """
        n_freq = self.n_freq_samples
        dim = self.dim
        data = self.data
        N = data.shape[0]
        data = data - np.mean(data, axis=0)
        fft_vals = rfft(data, axis=0)[1 : n_freq + 1]
        if fft_vals.shape[0] != self.freq.shape[0]:
            raise ValueError("FFT output and frequency grid mismatch.")
        scale = np.full(self.freq.shape, 2.0 / (N * self.fs), dtype=np.float64)
        if N % 2 == 0 and scale.size > 0:
            scale[-1] = 1.0 / (N * self.fs)
        periodogram = scale[:, None, None] * (
            fft_vals[:, :, None] * np.conj(fft_vals[:, None, :])
        )
        # Add epsilon to avoid log(0) and extremely small values
        eps = 1e-12
        periodogram = np.where(np.abs(periodogram) < eps, eps, periodogram)
        return periodogram

    def get_true_psd(self):
        """Return the theoretical one-sided PSD matrix."""
        eps = 1e-12
        true_psd = np.where(np.abs(self.psd) < eps, eps, self.psd)
        return true_psd

    def plot(self, axs=None, fname: Optional[str] = None):
        """
        Matrix plot: diagonal is the PSD, below diagonal is real CSD, above diagonal is imag CSD.
        Plots both the true PSD and the periodogram using consistent scaling.
        """
        if self.data is None:
            raise ValueError("No data to plot. Run resimulate first.")
        dim = self.dim
        periodogram = self.get_periodogram()
        true_psd = self.get_true_psd()
        freq_hz = self.freq
        # Setup axes
        if axs is None:
            fig, axs = plt.subplots(
                dim, dim, figsize=(4 * dim, 4 * dim), sharex=True
            )
        else:
            fig = axs[0, 0].figure
        data_kwgs = dict(alpha=0.3, lw=2, zorder=-10, color="k")
        true_kwgs = dict(lw=1, zorder=10, color="k")
        for i in range(dim):
            for j in range(dim):
                ax = axs[i, j]
                if i == j:
                    ax.plot(
                        freq_hz,
                        true_psd[:, i, i].real,
                        label="True PSD",
                        **true_kwgs,
                    )
                    ax.plot(
                        freq_hz,
                        periodogram[:, i, i].real,
                        label="Periodogram",
                        **data_kwgs,
                    )
                    ax.set_title(f"PSD: channel {i + 1}")
                    ax.set_yscale("log")
                elif i > j:
                    ax.plot(
                        freq_hz,
                        true_psd[:, i, j].real,
                        label="True Re(CSD)",
                        **true_kwgs,
                    )
                    ax.plot(
                        freq_hz,
                        periodogram[:, i, j].real,
                        label="Periodogram Re(CSD)",
                        **data_kwgs,
                    )
                    ax.set_title(f"Re(CSD): {i + 1},{j + 1}")
                else:
                    ax.plot(
                        freq_hz,
                        true_psd[:, i, j].imag,
                        label="True Im(CSD)",
                        **true_kwgs,
                    )
                    ax.plot(
                        freq_hz,
                        periodogram[:, i, j].imag,
                        label="Periodogram Im(CSD)",
                        **data_kwgs,
                    )
                    ax.set_title(f"Im(CSD): {i + 1},{j + 1}")
                if i == dim - 1:
                    ax.set_xlabel("Frequency (Hz)")
                if j == 0:
                    ax.set_ylabel("Power / CSD")
                ax.legend(fontsize=8)
        fig.tight_layout()
        if fname:
            fig.savefig(fname, bbox_inches="tight")
        return axs


def _calculate_true_varma_psd(
    freqs_hz: np.ndarray,
    dim: int,
    var_coeffs: np.ndarray,
    vma_coeffs: np.ndarray,
    sigma: np.ndarray,
    fs: float,
    channel_stds: np.ndarray,
    scaling_factor: float,
) -> np.ndarray:
    """
    Calculate the spectral matrix for given frequencies.

    Args:
        freqs_hz (np.ndarray): Positive frequency grid in Hz.
        dim (int): Process dimension.
        var_coeffs/vma_coeffs: VAR/MA coefficient arrays.
        sigma (np.ndarray): Innovation covariance matrix.
        fs (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: One-sided PSD matrix evaluated at ``freqs_hz``.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    if freqs_hz.size:
        nyquist = fs / 2.0
        if freqs_hz.max() > nyquist + 1e-12:
            raise ValueError(
                "Frequency grid should be in Hz (0 .. fs/2). "
                "Angular-frequency input detected."
            )

    omega = 2.0 * np.pi * freqs_hz / fs  # radians per sample
    spec_matrix = np.empty((freqs_hz.size, dim, dim), dtype=np.complex128)
    for idx, w in enumerate(omega):
        spec_matrix[idx] = _calculate_spec_matrix_helper(
            float(w), dim, var_coeffs, vma_coeffs, sigma
        )

    # Convert to one-sided PSD: double positive frequencies except Nyquist
    psd = (2.0 * spec_matrix) / fs
    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] *= 0.5
    if channel_stds is not None and scaling_factor not in (None, 0):
        scale_matrix = np.outer(channel_stds, channel_stds) / float(
            scaling_factor
        )
        psd = psd * scale_matrix[None, :, :]
    return psd


def _calculate_spec_matrix_helper(omega, dim, var_coeffs, vma_coeffs, sigma):
    """
    Helper function to calculate spectral matrix for a single frequency.

    Args:
        omega (float): Angular frequency (rad/sample).
    """
    if sigma.shape[0] == 1:
        cov_matrix = np.identity(dim) * sigma
    else:
        cov_matrix = sigma

    k_ar = np.arange(1, var_coeffs.shape[0] + 1)
    angles_ar = k_ar[:, np.newaxis, np.newaxis] * omega
    A_f_re_ar = np.sum(var_coeffs * np.cos(angles_ar), axis=0)
    A_f_im_ar = -np.sum(var_coeffs * np.sin(angles_ar), axis=0)
    A_f_ar = A_f_re_ar + 1j * A_f_im_ar
    A_bar_f_ar = np.identity(dim) - A_f_ar
    H_f_ar = np.linalg.inv(A_bar_f_ar)

    k_ma = np.arange(vma_coeffs.shape[0])
    angles_ma = k_ma[:, np.newaxis, np.newaxis] * omega
    A_f_re_ma = np.sum(vma_coeffs * np.cos(angles_ma), axis=0)
    A_f_im_ma = -np.sum(vma_coeffs * np.sin(angles_ma), axis=0)
    A_f_ma = A_f_re_ma + 1j * A_f_im_ma
    A_bar_f_ma = A_f_ma
    H_f_ma = A_bar_f_ma

    spec_mat = H_f_ar @ H_f_ma @ cov_matrix @ H_f_ma.conj().T @ H_f_ar.conj().T
    return spec_mat
