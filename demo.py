import dataclasses
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from jax import numpy as jnp
import jax
import optax
from scipy.interpolate import interp1d
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis


class LogPSplines:
    """Model for log power splines using B-spline basis and a penalty matrix."""

    def __init__(
        self,
        knots: np.ndarray,
        degree: int,
        diffMatrixOrder: int = 2,
        n: int = None,
    ):
        assert degree > diffMatrixOrder, "Degree must be larger than diffMatrixOrder."
        assert degree in [0, 1, 2, 3, 4, 5], "Degree must be between 0 and 5."
        assert diffMatrixOrder in [0, 1, 2], "diffMatrixOrder must be 0, 1, or 2."
        assert len(knots) >= degree, f"#knots: {len(knots)}, degree: {degree}"

        self.knots = knots
        self.degree = degree
        self.order = self.degree + 1
        self.diffMatrixOrder = diffMatrixOrder
        self.n = max(501, 10 * self.n_basis) if n is None else n

        self.basis, self.penalty_matrix = generate_basis_and_penalty_matrix(
            self.knots, self.degree, self.n, self.diffMatrixOrder
        )

    def __repr__(self):
        return f"LogPSplines(knots={self.knots}, degree={self.degree}, n={self.n})"

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    @property
    def n_basis(self) -> int:
        return self.n_knots + self.degree - 1

    def __call__(self, weights: jnp.ndarray = None) -> jnp.ndarray:
        """Return the weighted sum of B-spline basis functions minus a constant."""
        weighted_splines = jnp.sum(weights[:, None] * self.basis.T, axis=0)
        return weighted_splines - jnp.log(2 * jnp.pi)


def lnlikelihood(lndata, log_psplines: LogPSplines, weights: jnp.ndarray) -> float:
    """Compute the Whittle log likelihood.

    Args:
        lndata: Log power spectral density data.
        log_psplines: Instance of LogPSplines.
        weights: Spline weights.

    Returns:
        The computed log likelihood.
    """
    lnmodel = log_psplines(weights)
    integrand = lnmodel + jnp.exp(lndata - lnmodel - jnp.log(2 * np.pi))
    lnlike = -jnp.sum(integrand) / 2

    # Return a large negative value if lnlike is not finite
    lnlike = jnp.where(jnp.isfinite(lnlike), lnlike, -1e300)
    return lnlike


def data_peak_knots(data: np.ndarray, n_knots: int) -> np.ndarray:
    """Select knots at the peaks of the data.

    Args:
        data: The input data array.
        n_knots: Number of knots to select.

    Returns:
        An array of knots.
    """
    aux = np.sqrt(data)
    dens = np.abs(aux - np.mean(aux)) / np.std(aux)
    n = len(data)

    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)

    # Create forward and inverse interpolators
    df = interp1d(
        np.linspace(0, 1, num=n), cumf, kind="linear", fill_value=(0, 1)
    )
    invDf = interp1d(
        df(np.linspace(0, 1, num=n)),
        np.linspace(0, 1, num=n),
        kind="linear",
        fill_value=(0, 1),
        bounds_error=False,
    )
    knots = invDf(np.linspace(0, 1, num=n_knots))
    assert len(np.unique(knots)) == len(knots), f"Knots are not unique: {knots}"

    return knots


def generate_basis_and_penalty_matrix(
    knots: np.ndarray,
    degree: int,
    n_grid_points: int,
    diffMatrixOrder: int,
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a B-spline basis matrix and penalty matrix.

    Args:
        knots: Array of knots (between 0 and 1).
        degree: Degree of the B-spline.
        n_grid_points: Number of grid points.
        diffMatrixOrder: Order of the differential operator for regularization.
        epsilon: Small constant to ensure numerical stability.

    Returns:
        A tuple (basis_matrix, penalty_matrix) as JAX arrays.
    """
    order = degree + 1
    basis = BSplineBasis(order=order, knots=knots)
    grid_points = np.linspace(0, 1, n_grid_points)
    basis_matrix = basis.to_basis().to_grid(grid_points).data_matrix.squeeze().T

    # Augment knots with boundary knots for normalization
    knots_with_boundary = np.concatenate(
        [np.repeat(0, degree), knots, np.repeat(1, degree)]
    )

    n_knots_total = len(knots_with_boundary)
    mid_to_end = knots_with_boundary[degree + 1:]
    start_to_mid = knots_with_boundary[: (n_knots_total - degree - 1)]
    norm_factor = (mid_to_end - start_to_mid) / (degree + 1)
    norm_factor[norm_factor == 0] = np.inf
    basis_matrix = basis_matrix / norm_factor

    # Compute the penalty matrix using L2 regularization
    regularization = L2Regularization(LinearDifferentialOperator(diffMatrixOrder))
    p = regularization.penalty_matrix(basis)
    p = p / np.max(p)
    p = p + epsilon * np.eye(p.shape[1])

    return jnp.array(basis_matrix), jnp.array(p)


@dataclasses.dataclass
class Periodogram:
    freqs: jnp.ndarray
    power: jnp.ndarray

    @property
    def fs(self) -> float:
        """Sampling frequency computed from the frequency array."""
        return float(2 * self.freqs[-1])

    def highpass(self, min_freq: float) -> "Periodogram":
        """Return a new Periodogram with frequencies above a given threshold."""
        mask = self.freqs > min_freq
        return Periodogram(self.freqs[mask], self.power[mask])


@dataclasses.dataclass
class Timeseries:
    t: jnp.ndarray
    y: jnp.ndarray

    @property
    def fs(self) -> float:
        """Sampling frequency computed from time differences."""
        return float(1 / (self.t[1] - self.t[0]))

    def to_periodogram(self) -> "Periodogram":
        """Compute the one-sided periodogram of the timeseries."""
        freq = jnp.fft.rfftfreq(len(self.y), d=1 / self.fs)
        power = jnp.abs(jnp.fft.rfft(self.y)) ** 2 / len(self.y)
        return Periodogram(freq[1:], power[1:])


def generate_data() -> Timeseries:
    """Generate synthetic AR noise data."""
    a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
    n_samples = 1024
    fs = 100  # Sampling frequency in Hz
    dt = 1.0 / fs
    t = np.linspace(0, (n_samples - 1) * dt, n_samples)
    noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
    return Timeseries(t, noise)


def optimize_logpsplines_weights(
    noise_f: Periodogram, log_psplines: LogPSplines, weights: jnp.ndarray
) -> jnp.ndarray:
    """Optimize spline weights to match the periodogram data."""

    @jax.jit
    def compute_loss(params: jnp.ndarray) -> float:
        lnspline = log_psplines(params)
        return jnp.mean((jnp.log(noise_f.power) - lnspline) ** 2)

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(weights)

    for _ in range(1000):
        grads = jax.grad(compute_loss)(weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)

    return weights


def main():
    # Generate synthetic timeseries data and standardize it
    noise = generate_data()
    scale = jnp.std(noise.y)
    noise.y = (noise.y - jnp.mean(noise.y)) / scale

    # Compute the periodogram and apply a high-pass filter (above 5 Hz)
    noise_f = noise.to_periodogram().highpass(5)

    # Determine knots based on the periodogram frequencies and initialize the spline model
    knots = data_peak_knots(noise_f.freqs, n_knots=20)
    spline_model = LogPSplines(
        knots=knots, degree=3, diffMatrixOrder=2, n=len(noise_f.freqs)
    )
    weights = jnp.zeros(spline_model.n_basis)

    # Compute initial likelihood and optimize spline weights
    lnl_initial = lnlikelihood(noise_f.power, spline_model, weights)
    print("Initial log likelihood:", lnl_initial)

    weights = optimize_logpsplines_weights(noise_f, spline_model, weights)
    spline = jnp.exp(spline_model(weights)) * scale**2

    lnl_final = lnlikelihood(noise_f.power, spline_model, weights)
    print("Final log likelihood:", lnl_final)

    # Plot the timeseries and periodogram with the fitted spline model
    fig, ax = plt.subplots(2, 1, figsize=(4, 6))
    ax[0].plot(noise.t, noise.y, color="lightgray")
    ax[1].loglog(noise_f.freqs, noise_f.power * scale**2, color="lightgray")
    ax[1].loglog(noise_f.freqs, spline, label="Spline", color="tab:orange")

    # Mark the knots on the plot
    knots_freq = spline_model.knots * noise_f.fs
    idx = jnp.argmin(jnp.abs(noise_f.freqs[:, None] - knots_freq[None, :]), axis=0)
    ax[1].loglog(noise_f.freqs[idx], spline[idx], "o", label="Knots", color="tab:orange")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
