"""Locally stationary multivariate time-varying VAR(2) example.

The process is

    X_t = A1 X_{t-1} + A2(u_t) X_{t-2} + eps_t,

where

    u_t = t / N,

    A2(u) = q(u) A2_base,

    q(u) = 1 + alpha * b(u),

and the default modulation is the same smooth modulation used by the
LS2 benchmark,

    b(u) = 1.1 cos(1.5 - cos(4 pi u)).

The default A2_base couples all three channels so that every channel pair
has time-varying cross-spectral structure.

For each rescaled time u, the frozen-time process has transfer function

    H(u, f)
        = [I
           - A1 exp(-i omega)
           - A2(u) exp(-2 i omega)]^{-1},

with omega = 2 pi f / fs.

The corresponding one-sided local spectral density matrix is

    S(u, f)
        = (2 / fs) H(u, f) Sigma H(u, f)^*,

with the usual factor-of-two correction removed at Nyquist.

The returned truth therefore has shape

    (n_time, n_frequency, n_channels, n_channels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from log_psplines.data import TimeSeries

_PLOT_GRID_SIZE = 256


def _default_a1() -> np.ndarray:
    """Default first-lag VAR coefficient matrix."""
    return np.diag([0.4, 0.3, 0.2])


def _default_a2() -> np.ndarray:
    """Default second-lag VAR coefficient matrix."""
    return np.array(
        [
            [-0.20, 0.35, 0.25],
            [0.30, -0.10, -0.22],
            [0.20, 0.18, -0.35],
        ],
        dtype=float,
    )


def _default_sigma() -> np.ndarray:
    """Default innovation covariance matrix."""
    return np.array(
        [
            [0.25, 0.00, 0.08],
            [0.00, 0.25, 0.08],
            [0.08, 0.08, 0.25],
        ],
        dtype=float,
    )


def _companion_spectral_radius(
    a1: np.ndarray,
    a2: np.ndarray,
) -> float:
    """Return the companion spectral radius for two ``(p, p)`` lag matrices."""
    p = a1.shape[0]

    companion = np.block(
        [
            [a1, a2],
            [np.eye(p), np.zeros((p, p))],
        ]
    )

    eigvals = np.linalg.eigvals(companion)

    return float(np.max(np.abs(eigvals)))


@dataclass
class TVVARData:
    """Three-channel locally stationary time-varying VAR(2) process.

    Parameters
    ----------
    n_samples:
        Number of returned time-domain samples.

    fs:
        Sampling frequency in Hz.

    seed:
        Random seed.

    alpha:
        Strength of the time variation. ``alpha=0`` gives a stationary VAR(2)
        with coefficients ``a1`` and ``a2_base``.

    a1:
        First-lag VAR coefficient matrix.

    a2_base:
        Baseline second-lag VAR coefficient matrix.

    sigma:
        Innovation covariance matrix.

    modulation_amplitude:
        Amplitude of the LS2-style modulation b(u).

    modulation_offset:
        Offset inside the outer cosine.

    modulation_frequency:
        Frequency multiplying pi * u inside the inner cosine.

    burn_in:
        Number of stationary burn-in samples generated using the coefficients
        at u=0.

    stability_grid_size:
        Number of points on u in [0, 1] used to check local VAR stability.
    """

    n_samples: int = 4096
    fs: float = 64.0
    seed: int | None = None

    alpha: float | int = 0.25

    a1: np.ndarray = field(default_factory=_default_a1)
    a2_base: np.ndarray = field(default_factory=_default_a2)
    sigma: np.ndarray = field(default_factory=_default_sigma)

    modulation_amplitude: float = 1.1
    modulation_offset: float = 1.5
    modulation_frequency: float = 4.0

    burn_in: int = 512
    stability_grid_size: int = 1001

    data: np.ndarray = field(init=False)
    time: np.ndarray = field(init=False)
    freq: np.ndarray = field(init=False)

    max_local_spectral_radius: float = field(init=False)
    is_locally_stable: bool = field(init=False)

    def __post_init__(self) -> None:
        self.a1 = np.asarray(self.a1, dtype=float)
        self.a2_base = np.asarray(self.a2_base, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)

        self._validate_inputs()

        self.time = np.arange(self.n_samples) / self.fs

        # Positive Fourier frequencies:
        # DC excluded, Nyquist included.
        self.freq = np.fft.rfftfreq(
            self.n_samples,
            d=self.dt,
        )[1:]

        self._check_local_stability()

        self.data = self.simulate(seed=self.seed)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Sampling interval in seconds."""
        return 1.0 / self.fs

    @property
    def p(self) -> int:
        """Number of channels."""
        return int(self.a1.shape[0])

    @property
    def duration(self) -> float:
        """Nominal observation duration in seconds."""
        return self.n_samples / self.fs

    @property
    def rescaled_time(self) -> np.ndarray:
        """Locally stationary time coordinate u=t/N in [0, 1)."""
        return np.arange(self.n_samples) / self.n_samples

    @property
    def ts(self) -> TimeSeries:
        """Return the realization as the canonical TimeSeries object."""
        return TimeSeries(
            data=self.data,
            t=self.time,
        )

    # ------------------------------------------------------------------
    # Time-varying coefficients
    # ------------------------------------------------------------------

    def modulation(
        self,
        u: np.ndarray | float,
    ) -> np.ndarray | float:
        """Return the LS2-style modulation b(u).

        The default is

            b(u) = 1.1 cos(1.5 - cos(4 pi u)).
        """
        u = np.asarray(u, dtype=float)

        return self.modulation_amplitude * np.cos(
            self.modulation_offset
            - np.cos(self.modulation_frequency * np.pi * u)
        )

    def scale(
        self,
        u: np.ndarray | float,
    ) -> np.ndarray | float:
        """Return the multiplicative modulation q(u)."""
        return 1.0 + self.alpha * self.modulation(u)

    def a2_at(
        self,
        u: np.ndarray | float,
    ) -> np.ndarray:
        """Return the second-lag coefficient matrix A2(u).

        Scalar input returns shape ``(p, p)``.

        Array input with shape ``(...)`` returns shape ``(..., p, p)``.
        """
        q = self.scale(u)

        return q[..., None, None] * self.a2_base

    def var_coeffs_at(
        self,
        u: float,
    ) -> np.ndarray:
        """Return [A1, A2(u)] with shape (2, p, p)."""
        return np.stack(
            [
                self.a1,
                self.a2_at(u),
            ],
            axis=0,
        )

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def local_spectral_radius(
        self,
        u: float,
    ) -> float:
        """Return the frozen-time VAR companion spectral radius."""
        return _companion_spectral_radius(
            self.a1,
            self.a2_at(u),
        )

    def _check_local_stability(self) -> None:
        """Check stability of the frozen-time VAR over u in [0, 1]."""
        u_grid = np.linspace(
            0.0,
            1.0,
            self.stability_grid_size,
        )

        radii = np.asarray(
            [self.local_spectral_radius(float(u)) for u in u_grid]
        )

        self.max_local_spectral_radius = float(np.max(radii))
        self.is_locally_stable = bool(self.max_local_spectral_radius < 1.0)

        if not self.is_locally_stable:
            raise ValueError(
                "Time-varying VAR is not locally stable: "
                "maximum companion spectral radius is "
                f"{self.max_local_spectral_radius:.6f}."
            )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate one realization of the time-varying VAR(2).

        Returns
        -------
        ndarray
            Array with shape ``(n_samples, p)``.
        """
        rng = np.random.default_rng(seed)

        total = self.burn_in + self.n_samples + 2

        x = np.zeros(
            (total, self.p),
            dtype=float,
        )

        # --------------------------------------------------------------
        # Burn in using the frozen coefficients at u=0.
        # --------------------------------------------------------------
        a2_burn = self.a2_at(0.0)

        for t in range(2, self.burn_in + 2):
            eps = rng.multivariate_normal(
                mean=np.zeros(self.p),
                cov=self.sigma,
            )

            x[t] = self.a1 @ x[t - 1] + a2_burn @ x[t - 2] + eps

        # --------------------------------------------------------------
        # Time-varying record.
        # --------------------------------------------------------------
        start = self.burn_in + 2

        for k in range(self.n_samples):
            t = start + k
            u = k / self.n_samples

            eps = rng.multivariate_normal(
                mean=np.zeros(self.p),
                cov=self.sigma,
            )

            x[t] = self.a1 @ x[t - 1] + self.a2_at(u) @ x[t - 2] + eps

        return x[start : start + self.n_samples]

    def resimulate(
        self,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate and store a new realization."""
        self.data = self.simulate(seed=seed)
        return self.data

    # ------------------------------------------------------------------
    # Analytic local spectrum
    # ------------------------------------------------------------------

    def get_true_psd(
        self,
        *,
        time_grid: np.ndarray | None = None,
        freq_grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the analytic local spectral density matrix S(u, f).

        Parameters
        ----------
        time_grid:
            Rescaled time coordinates u in [0, 1]. If omitted,
            ``self.rescaled_time`` is used.

        freq_grid:
            Frequency coordinates in Hz. If omitted, ``self.freq`` is used.

        Returns
        -------
        ndarray
            Complex Hermitian spectral matrix with shape

                (n_time, n_freq, p, p).

        Notes
        -----
        For each frozen time u,

            H(u, f)
                = [
                    I
                    - A1 exp(-i omega)
                    - A2(u) exp(-2 i omega)
                  ]^{-1},

        where

            omega = 2 pi f / fs.

        The one-sided spectrum is

            S(u, f)
                = (2 / fs)
                  H(u, f)
                  Sigma
                  H(u, f)^*.

        DC and Nyquist are not doubled when requested. Truth is evaluated
        lazily on the supplied grids; no spectrum is cached at construction.
        """
        if time_grid is None:
            time_grid = self.rescaled_time

        if freq_grid is None:
            freq_grid = self.freq

        u = np.asarray(
            time_grid,
            dtype=float,
        )

        f = np.asarray(
            freq_grid,
            dtype=float,
        )

        if u.ndim != 1:
            raise ValueError("time_grid must be a one-dimensional array.")

        if f.ndim != 1:
            raise ValueError("freq_grid must be a one-dimensional array.")

        if not np.isfinite(u).all() or np.any((u < 0.0) | (u > 1.0)):
            raise ValueError("time_grid must contain finite values in [0, 1].")

        # Allow only rounding error at the Nyquist endpoint.
        frequency_tolerance = 8.0 * np.finfo(float).eps * self.fs
        if (
            not np.isfinite(f).all()
            or np.any(f < 0.0)
            or np.any(f > self.fs / 2.0 + frequency_tolerance)
        ):
            raise ValueError("freq_grid must lie between 0 and fs/2.")

        # --------------------------------------------------------------
        # Build H(u, f) on the complete time-frequency grid.
        #
        # Shapes:
        #   A2       -> (T, p, p)
        #   z        -> (F,)
        #   transfer -> (T, F, p, p)
        # --------------------------------------------------------------
        a2 = self.a2_at(u)

        omega = 2.0 * np.pi * f / self.fs

        z1 = np.exp(-1j * omega)
        z2 = np.exp(-2j * omega)

        identity = np.eye(
            self.p,
            dtype=np.complex128,
        )

        transfer_inverse = (
            identity[None, None, :, :]
            - self.a1[None, None, :, :] * z1[None, :, None, None]
            - a2[:, None, :, :] * z2[None, :, None, None]
        )

        h = np.linalg.inv(transfer_inverse)

        # H Sigma H^*
        spectrum = (
            h @ self.sigma[None, None, :, :] @ np.swapaxes(h.conj(), -1, -2)
        )

        # Convert digital spectral density to one-sided PSD / Hz.
        spectrum *= 2.0 / self.fs

        # DC and Nyquist should not receive the factor of two.
        if f.size:
            endpoint_mask = (f == 0.0) | np.isclose(
                f,
                self.fs / 2.0,
                rtol=0.0,
                atol=frequency_tolerance,
            )
            spectrum[:, endpoint_mask] *= 0.5

        # Remove tiny numerical violations of Hermiticity.
        spectrum = 0.5 * (spectrum + np.swapaxes(spectrum.conj(), -1, -2))

        return spectrum

    def plot(
        self,
        *,
        fname: str | Path | None = None,
        show: bool = False,
        cmap: str = "viridis",
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Plot the true local spectrum in a ``(p, p)`` channel grid.

        Diagonal panels share a log10 auto PSD scale per Hz. Upper/lower panels
        show real/imaginary coherence with separate shared symmetric scales,
        centred at zero and bounded by each group's maximum absolute value.
        Truth is sampled at up to 256 points per axis for bounded plot memory.
        """
        u = np.linspace(
            0.0,
            (self.n_samples - 1) / self.n_samples,
            min(self.n_samples, _PLOT_GRID_SIZE),
        )
        freq = np.linspace(
            self.freq[0], self.freq[-1], min(self.freq.size, _PLOT_GRID_SIZE)
        )
        spectrum = self.get_true_psd(time_grid=u, freq_grid=freq)
        auto_psd = np.diagonal(spectrum, axis1=-2, axis2=-1).real
        coherence = spectrum / np.sqrt(
            auto_psd[..., :, None] * auto_psd[..., None, :]
        )
        upper = np.triu_indices(self.p, k=1)
        lower = np.tril_indices(self.p, k=-1)
        real_max = float(
            np.max(
                np.abs(coherence[..., upper[0], upper[1]].real), initial=0.0
            )
        )
        imag_max = float(
            np.max(
                np.abs(coherence[..., lower[0], lower[1]].imag), initial=0.0
            )
        )
        # Unit bounds keep the normalization defined for an identically zero group.
        real_max = real_max or 1.0
        imag_max = imag_max or 1.0
        real_norm = TwoSlopeNorm(vmin=-real_max, vcenter=0.0, vmax=real_max)
        imag_norm = TwoSlopeNorm(vmin=-imag_max, vcenter=0.0, vmax=imag_max)
        if figsize is None:
            figsize = (3.2 * self.p + 2.0, 2.7 * self.p + 1.0)

        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            self.p,
            self.p + 1,
            width_ratios=[1] * self.p + [0.06],
            left=0.08,
            right=0.88,
            bottom=0.09,
            top=0.94,
            wspace=0.10,
            hspace=0.18,
        )
        axes = np.empty((self.p, self.p), dtype=object)
        for i in range(self.p):
            for j in range(self.p):
                axes[i, j] = fig.add_subplot(
                    grid[i, j],
                    sharex=axes[0, 0] if i or j else None,
                    sharey=axes[0, 0] if i or j else None,
                )
        diagonal = np.log10(auto_psd)
        diagonal_min = float(np.min(diagonal))
        diagonal_max = float(np.max(diagonal))
        colorbar_grid = grid[:, -1].subgridspec(3 if self.p > 1 else 1, 1)
        meshes = {}
        for i in range(self.p):
            for j in range(self.p):
                ax = axes[i, j]
                if i == j:
                    values = diagonal[..., i]
                    group = "diagonal"
                    options = {
                        "cmap": cmap,
                        "vmin": diagonal_min,
                        "vmax": diagonal_max,
                    }
                    title = rf"$\log_{{10}} S_{{{i + 1}{j + 1}}}$"
                elif i < j:
                    values = coherence[..., i, j].real
                    group = "real"
                    options = {
                        "cmap": "RdBu_r",
                        "norm": real_norm,
                    }
                    title = rf"$\operatorname{{Re}}\,C_{{{i + 1}{j + 1}}}$"
                else:
                    values = coherence[..., i, j].imag
                    group = "imaginary"
                    options = {
                        "cmap": "RdBu_r",
                        "norm": imag_norm,
                    }
                    title = rf"$\operatorname{{Im}}\,C_{{{i + 1}{j + 1}}}$"

                meshes[group] = ax.pcolormesh(
                    u * self.duration,
                    freq,
                    values.T,
                    shading="auto",
                    **options,
                )
                ax.set_title(title, fontsize=11, pad=5)
                ax.tick_params(
                    which="both",
                    bottom=i == self.p - 1,
                    left=j == 0,
                    top=False,
                    right=False,
                    labelbottom=i == self.p - 1,
                    labelleft=j == 0,
                )
                if i == self.p - 1:
                    ax.set_xlabel("Time [s]")
                if j == 0:
                    ax.set_ylabel("Frequency [Hz]")

        for index, (group, label) in enumerate(
            (
                ("diagonal", "log10 PSD [1/Hz]"),
                ("real", "Re coherence"),
                ("imaginary", "Im coherence"),
            )
        ):
            if group in meshes:
                colorbar_ax = fig.add_subplot(colorbar_grid[index, 0])
                fig.colorbar(meshes[group], cax=colorbar_ax, label=label)

        if fname is not None:
            path = Path(fname)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200)
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Validate dimensions and innovation covariance."""
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2.")

        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("fs must be positive.")

        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")

        if self.stability_grid_size < 2:
            raise ValueError("stability_grid_size must be at least 2.")

        if self.a1.ndim != 2:
            raise ValueError("a1 must be a two-dimensional square matrix.")

        if self.a1.shape[0] == 0 or self.a1.shape[0] != self.a1.shape[1]:
            raise ValueError("a1 must be non-empty and square.")

        if self.a2_base.shape != self.a1.shape:
            raise ValueError("a2_base must have the same shape as a1.")

        if self.sigma.shape != self.a1.shape:
            raise ValueError("sigma must have shape (p, p).")

        for name in ("a1", "a2_base", "sigma"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must contain only finite values.")

        if not np.allclose(
            self.sigma,
            self.sigma.T,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("sigma must be symmetric.")

        eigvals = np.linalg.eigvalsh(self.sigma)

        if np.min(eigvals) <= 0.0:
            raise ValueError("sigma must be positive definite.")


@dataclass
class TVData:
    """Univariate locally stationary time-varying AR(2) process.

    A single-channel special case of :class:`TVVARData`, exposing the same
    ``data``/``freq``/``get_true_psd`` conventions as :class:`LS2Data` for
    convenience in univariate examples and tests.
    """

    n_samples: int = 512
    fs: float = 64.0
    seed: int | None = None

    alpha: float = 0.25
    a1: float = 0.5
    a2_base: float = -0.3
    sigma: float = 0.25

    modulation_amplitude: float = 1.1
    modulation_offset: float = 1.5
    modulation_frequency: float = 4.0

    burn_in: int = 512
    stability_grid_size: int = 1001

    data: np.ndarray = field(init=False)
    time: np.ndarray = field(init=False)
    freq: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._tv = TVVARData(
            n_samples=self.n_samples,
            fs=self.fs,
            seed=self.seed,
            alpha=self.alpha,
            a1=np.array([[self.a1]]),
            a2_base=np.array([[self.a2_base]]),
            sigma=np.array([[self.sigma]]),
            modulation_amplitude=self.modulation_amplitude,
            modulation_offset=self.modulation_offset,
            modulation_frequency=self.modulation_frequency,
            burn_in=self.burn_in,
            stability_grid_size=self.stability_grid_size,
        )

        self.data = self._tv.data
        self.time = self._tv.time
        self.freq = self._tv.freq

    @property
    def dt(self) -> float:
        """Sampling interval in seconds."""
        return self._tv.dt

    @property
    def duration(self) -> float:
        """Total nominal duration in seconds."""
        return self._tv.duration

    @property
    def rescaled_time(self) -> np.ndarray:
        """Sample positions on the locally stationary time coordinate."""
        return self._tv.rescaled_time

    def get_true_psd(
        self,
        *,
        time_grid: np.ndarray | None = None,
        freq_grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the analytic pointwise one-sided PSD, shape ``(n_time, n_freq)``."""
        psd = self._tv.get_true_psd(time_grid=time_grid, freq_grid=freq_grid)
        return psd[..., 0, 0].real

    def plot(
        self,
        *,
        fname: str | Path | None = None,
        show: bool = False,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (8.0, 5.0),
    ) -> None:
        """Plot the analytic one-sided local PSD in a single panel."""
        self._tv.plot(fname=fname, show=show, cmap=cmap, figsize=figsize)
