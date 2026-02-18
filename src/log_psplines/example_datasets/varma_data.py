from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft

from ..logger import logger


class VARMAData:
    """Simulate a multivariate VARMA process and related spectral quantities.

    The simulated process uses the recursion
    ``x_t = sum_{k=1}^p A_k x_{t-k} + sum_{j=0}^q B_j eps_{t-j}``,
    where ``eps_t`` is Gaussian white noise with covariance ``sigma``.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        sigma: np.ndarray = np.array([[1.0, 0.9], [0.9, 1.0]]),
        var_coeffs: np.ndarray = np.array(
            [[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]]
        ),
        vma_coeffs: np.ndarray = np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        seed: int | None = None,
        fs: float = 1.0,
    ):
        """Initialise and simulate a VARMA dataset.

        Parameters
        ----------
        n_samples : int, default=1024
            Number of time samples to generate.
        sigma : np.ndarray, shape (C, C) or (1, 1), default=[[1.0, 0.9], [0.9, 1.0]]
            Innovation covariance matrix. A `(1, 1)` input is treated as an isotropic
            variance and expanded to ``variance * I_C``.
        var_coeffs : np.ndarray, shape (P, C, C), optional
            VAR coefficient matrices ``A_1, ..., A_P``.
        vma_coeffs : np.ndarray, shape (Q + 1, C, C), optional
            VMA coefficient matrices ``B_0, ..., B_Q``.
        seed : int | None, default=None
            Seed for NumPy RNG used during simulation.
        fs : float, default=1.0
            Sampling frequency in Hz.
        """
        _validate_varma_inputs(
            var_coeffs=var_coeffs, vma_coeffs=vma_coeffs, sigma=sigma
        )
        self.n_samples = n_samples
        self.var_coeffs = var_coeffs
        self.vma_coeffs = vma_coeffs
        self.sigma = sigma
        self.p = vma_coeffs.shape[1]
        self.psd_scaling = 1.0
        self.n_freq_samples = n_samples // 2

        self.fs = float(fs)
        # Frequency grid in Hz (drop the DC bin to match PSD plotting expectations)
        self.freq = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)[1:]
        self.time = np.arange(n_samples) / self.fs
        self.data: np.ndarray | None = None  # set in "resimulate"
        self.periodogram: np.ndarray | None = None  # set in "resimulate"
        self.welch_psd: np.ndarray | None = None  # set in "resimulate"
        self.welch_f: np.ndarray | None = None  # set in "resimulate"
        self.is_var_stationary: bool | None = None
        self.is_valid_var_dataset: bool | None = None
        self.var_companion_spectral_radius: float | None = None
        self.psd: np.ndarray = np.zeros(
            (self.freq.size, self.p, self.p), dtype=np.complex128
        )
        self.resimulate(seed=seed)
        if self.data is None:
            raise ValueError("VARMA simulation produced no data.")

        self.channel_stds = np.std(self.data, axis=0)
        self.psd_scaling = float(np.std(self.data) ** 2)
        self.psd = _calculate_true_varma_psd(
            self.freq,
            self.p,
            self.var_coeffs,
            self.vma_coeffs,
            self.sigma,
            self.fs,
            channel_stds=None,
            scaling_factor=1.0,
        )

    def resimulate(self, seed: int | None = None) -> np.ndarray:
        """Simulate or re-simulate the VARMA process.

        Parameters
        ----------
        seed : int | None, default=None
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray, shape (N, C)
            Simulated multivariate time series.
        """
        if seed is not None:
            np.random.seed(seed)

        lag_ma = self.vma_coeffs.shape[0]
        lag_ar = self.var_coeffs.shape[0]

        if self.sigma.shape[0] == 1:
            cov_matrix = np.identity(self.p) * self.sigma
        else:
            cov_matrix = self.sigma

        x_init = np.zeros((lag_ar + 1, self.p))
        x = np.empty((self.n_samples + 101, self.p))
        x[:] = np.nan
        x[: lag_ar + 1] = x_init
        epsilon = np.random.multivariate_normal(
            np.zeros(self.p), cov_matrix, size=[lag_ma]
        )

        for i in range(lag_ar + 1, x.shape[0]):
            epsilon = np.concatenate(
                [
                    np.random.multivariate_normal(
                        np.zeros(self.p), cov_matrix, size=[1]
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
        self._run_var_validity_checks()
        return self.data

    def _run_var_validity_checks(self) -> None:
        """Evaluate and log basic VAR validity checks for generated samples.

        The check is based on the VAR companion matrix implied by ``var_coeffs``:
        stationarity requires all companion eigenvalues to lie strictly inside the unit
        circle. In addition, all generated samples must be finite.
        """
        if self.data is None:
            raise ValueError(
                "No simulated data available for validity checks."
            )
        is_stationary, spectral_radius = _is_var_stationary(self.var_coeffs)
        all_finite = bool(np.all(np.isfinite(self.data)))
        self.is_var_stationary = is_stationary
        self.var_companion_spectral_radius = spectral_radius
        self.is_valid_var_dataset = bool(is_stationary and all_finite)

        if self.is_valid_var_dataset:
            logger.info(
                f"VAR sanity check passed: stationary AR dynamics "
                f"(companion spectral radius={spectral_radius:.6f}) with finite samples."
            )
            return

        issues: list[str] = []
        if not is_stationary:
            issues.append(
                "AR dynamics are non-stationary "
                f"(companion spectral radius={spectral_radius:.6f} >= 1)."
            )
        if not all_finite:
            issues.append("simulated samples include NaN/Inf.")
        logger.warning(f"VAR sanity check failed: {' '.join(issues)}")

    def get_periodogram(self) -> np.ndarray:
        """Compute a one-sided matrix periodogram from the simulated data.

        Returns
        -------
        np.ndarray, shape (F, C, C)
            One-sided PSD/CSD matrix estimate evaluated on ``self.freq``.
        """
        _ = self.n_freq_samples
        _ = self.p
        data = self.data
        if data is None:
            raise ValueError("No simulated data available.")
        N = data.shape[0]
        data = data - np.mean(data, axis=0)
        fft_vals = rfft(data, axis=0)[1 : N + 1]
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

    def get_true_psd(self) -> np.ndarray:
        """Return the theoretical one-sided PSD matrix in original data units."""
        eps = 1e-12
        true_psd = np.where(np.abs(self.psd) < eps, eps, self.psd)
        return true_psd

    def plot(
        self, axs: Optional[np.ndarray] = None, fname: Optional[str] = None
    ) -> np.ndarray:
        """Plot PSD/CSD panels for true PSD and empirical periodogram.

        Parameters
        ----------
        axs : Optional[np.ndarray], default=None
            Optional pre-created axes array with shape ``(C, C)``.
        fname : Optional[str], default=None
            If provided, save the figure to this path.

        Returns
        -------
        np.ndarray
            Axes array with shape ``(C, C)``.
        """
        if self.data is None:
            raise ValueError("No data to plot. Run resimulate first.")
        p = self.p
        periodogram = self.get_periodogram()
        true_psd = self.get_true_psd()
        freq_hz = self.freq
        # Setup axes
        if axs is None:
            fig, axs = plt.subplots(p, p, figsize=(4 * p, 4 * p), sharex=True)
        else:
            fig = axs[0, 0].figure
        data_kwgs = dict(alpha=0.3, lw=2, zorder=-10, color="k")
        true_kwgs = dict(lw=1, zorder=10, color="k")
        for i in range(p):
            for j in range(p):
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
                if i == p - 1:
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
    p: int,
    var_coeffs: np.ndarray,
    vma_coeffs: np.ndarray,
    sigma: np.ndarray,
    fs: float,
    channel_stds: Optional[np.ndarray],
    scaling_factor: Optional[float],
) -> np.ndarray:
    """Calculate the one-sided theoretical VARMA spectral matrix.

    Parameters
    ----------
    freqs_hz : np.ndarray, shape (F,)
        Positive frequencies in Hz.
    p : int
        Number of channels.
    var_coeffs : np.ndarray, shape (P, C, C)
        VAR coefficient matrices.
    vma_coeffs : np.ndarray, shape (Q + 1, C, C)
        VMA coefficient matrices.
    sigma : np.ndarray, shape (C, C) or (1, 1)
        Innovation covariance matrix.
    fs : float
        Sampling frequency in Hz.
    channel_stds : Optional[np.ndarray], shape (C,)
        Optional channel-wise standard deviations for PSD re-scaling.
    scaling_factor : Optional[float]
        Optional global multiplicative scaling.

    Returns
    -------
    np.ndarray, shape (F, C, C)
        One-sided PSD matrix evaluated at ``freqs_hz``.
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
    spec_matrix = np.empty((freqs_hz.size, p, p), dtype=np.complex128)
    for idx, w in enumerate(omega):
        spec_matrix[idx] = _calculate_spec_matrix_helper(
            float(w), p, var_coeffs, vma_coeffs, sigma
        )

    # Convert to one-sided PSD: double positive frequencies except Nyquist
    psd = (2.0 * spec_matrix) / fs
    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] *= 0.5
    # Optional rescaling helpers:
    # - If channel_stds are provided, return a channel-standardised PSD where the
    #   corresponding time series has been divided by channel_stds.
    # - If scaling_factor is provided, apply it as a global multiplicative PSD
    #   scaling (useful for matching pipelines that attach an overall scaling
    #   factor during FFT/Wishart conversion).
    if channel_stds is not None:
        channel_stds = np.asarray(channel_stds, dtype=np.float64)
        if channel_stds.shape != (p,):
            raise ValueError(
                f"channel_stds must have shape ({p},), got {channel_stds.shape}."
            )
        denom = np.outer(channel_stds, channel_stds).astype(np.float64)
        denom = np.where(denom == 0.0, np.nan, denom)
        psd = psd / denom[None, :, :]
    if scaling_factor is not None and scaling_factor != 0:
        psd = psd * float(scaling_factor)
    return psd


def _calculate_spec_matrix_helper(omega, p, var_coeffs, vma_coeffs, sigma):
    """Calculate the VARMA spectral matrix at one angular frequency.

    Parameters
    ----------
    omega : float
        Angular frequency in rad/sample.
    p : int
        Number of channels.
    var_coeffs : np.ndarray, shape (P, C, C)
        VAR coefficient matrices.
    vma_coeffs : np.ndarray, shape (Q + 1, C, C)
        VMA coefficient matrices.
    sigma : np.ndarray, shape (C, C) or (1, 1)
        Innovation covariance matrix.

    Returns
    -------
    np.ndarray, shape (C, C)
        Complex spectral matrix value at ``omega``.
    """
    if sigma.shape[0] == 1:
        cov_matrix = np.identity(p) * sigma
    else:
        cov_matrix = sigma

    k_ar = np.arange(1, var_coeffs.shape[0] + 1)
    angles_ar = k_ar[:, np.newaxis, np.newaxis] * omega
    A_f_re_ar = np.sum(var_coeffs * np.cos(angles_ar), axis=0)
    A_f_im_ar = -np.sum(var_coeffs * np.sin(angles_ar), axis=0)
    A_f_ar = A_f_re_ar + 1j * A_f_im_ar
    A_bar_f_ar = np.identity(p) - A_f_ar
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


def _validate_varma_inputs(
    var_coeffs: np.ndarray, vma_coeffs: np.ndarray, sigma: np.ndarray
) -> None:
    """Validate core VARMA parameter shapes and matrix consistency.

    Parameters
    ----------
    var_coeffs : np.ndarray, shape (P, C, C)
        VAR coefficient matrices.
    vma_coeffs : np.ndarray, shape (Q + 1, C, C)
        VMA coefficient matrices.
    sigma : np.ndarray, shape (C, C) or (1, 1)
        Innovation covariance matrix.
    """
    if var_coeffs.ndim != 3:
        raise ValueError(
            f"var_coeffs must have shape (P, C, C), got ndim={var_coeffs.ndim}."
        )
    if vma_coeffs.ndim != 3:
        raise ValueError(
            f"vma_coeffs must have shape (Q + 1, C, C), got ndim={vma_coeffs.ndim}."
        )
    if var_coeffs.shape[1] != var_coeffs.shape[2]:
        raise ValueError(
            "var_coeffs channel matrices must be square with shape (C, C)."
        )
    if vma_coeffs.shape[1] != vma_coeffs.shape[2]:
        raise ValueError(
            "vma_coeffs channel matrices must be square with shape (C, C)."
        )
    if var_coeffs.shape[1] != vma_coeffs.shape[1]:
        raise ValueError(
            "var_coeffs and vma_coeffs must use the same channel dimension."
        )
    if sigma.ndim != 2:
        raise ValueError(
            f"sigma must be a 2D array with shape (C, C) or (1, 1), got ndim={sigma.ndim}."
        )
    if sigma.shape != (1, 1) and sigma.shape != (
        vma_coeffs.shape[1],
        vma_coeffs.shape[1],
    ):
        raise ValueError(
            "sigma must have shape (C, C) matching channels, or shape (1, 1)."
        )


def _is_var_stationary(
    var_coeffs: np.ndarray, tol: float = 1e-10
) -> tuple[bool, float]:
    """Check VAR stationarity from the companion matrix eigenvalues.

    For ``x_t = A_1 x_{t-1} + ... + A_P x_{t-P} + e_t``, covariance stationarity
    holds when all eigenvalues of the companion matrix satisfy ``|lambda| < 1``.

    Parameters
    ----------
    var_coeffs : np.ndarray, shape (P, C, C)
        VAR coefficient matrices.
    tol : float, default=1e-10
        Tolerance margin used in ``|lambda| < 1 - tol``.

    Returns
    -------
    tuple[bool, float]
        `(is_stationary, spectral_radius)`.
    """
    ar_order = int(var_coeffs.shape[0])
    n_channels = int(var_coeffs.shape[1])
    if ar_order == 0:
        return True, 0.0

    companion = np.zeros(
        (n_channels * ar_order, n_channels * ar_order), dtype=float
    )
    companion[:n_channels, : n_channels * ar_order] = np.hstack(var_coeffs)
    if ar_order > 1:
        companion[n_channels:, :-n_channels] = np.eye(
            n_channels * (ar_order - 1), dtype=float
        )

    eigvals = np.linalg.eigvals(companion)
    spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    return bool(spectral_radius < (1.0 - tol)), spectral_radius
