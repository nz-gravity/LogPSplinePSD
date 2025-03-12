import dataclasses
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from jax import numpy as jnp
import jax
import optax
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis

from log_psplines.datasets import Periodogram, Timeseries


class LogPSplines:
    """Model for log power splines using a B-spline basis and a penalty matrix."""

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
        """Compute the weighted sum of the B-spline basis functions minus a constant."""
        weighted_splines = jnp.sum(weights[:, None] * self.basis.T, axis=0)
        return weighted_splines - jnp.log(2 * jnp.pi)


def lnlikelihood(lndata_log: jnp.ndarray, log_psplines: LogPSplines, weights: jnp.ndarray) -> float:
    """Compute the Whittle log likelihood.

    Args:
        lndata_log: Log power spectral density data.
        log_psplines: Instance of LogPSplines.
        weights: Spline weights.

    Returns:
        The computed log likelihood.
    """
    lnmodel = log_psplines(weights)
    # Compute the likelihood term on the log-scale.
    integrand = lnmodel + jnp.exp(lndata_log - lnmodel - jnp.log(2 * np.pi))
    lnlike = -jnp.sum(integrand) / 2

    # If lnlike is not finite, return a very large negative value.
    lnlike = jnp.where(jnp.isfinite(lnlike), lnlike, -1e300)
    return lnlike




def generate_basis_and_penalty_matrix(
        knots: np.ndarray,
        degree: int,
        n_grid_points: int,
        diffMatrixOrder: int,
        epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a B-spline basis matrix and penalty matrix.

    Args:
        knots: Array of knots (values between 0 and 1).
        degree: Degree of the B-spline.
        n_grid_points: Number of grid points.
        diffMatrixOrder: Order of the differential operator for regularization.
        epsilon: Small constant for numerical stability.

    Returns:
        A tuple (basis_matrix, penalty_matrix) as JAX arrays.
    """
    order = degree + 1
    basis = BSplineBasis(domain_range=[0,1], order=order, knots=knots)
    grid_points = np.linspace(0, 1, n_grid_points)
    basis_matrix = basis.to_basis().to_grid(grid_points).data_matrix.squeeze().T

    # Augment knots with boundary values for proper normalization.
    knots_with_boundary = np.concatenate(
        [np.repeat(0, degree), knots, np.repeat(1, degree)]
    )
    n_knots_total = len(knots_with_boundary)
    mid_to_end = knots_with_boundary[degree + 1:]
    start_to_mid = knots_with_boundary[: (n_knots_total - degree - 1)]
    norm_factor = (mid_to_end - start_to_mid) / (degree + 1)
    norm_factor[norm_factor == 0] = np.inf  # Prevent division by zero.
    basis_matrix = basis_matrix / norm_factor

    # Compute the penalty matrix using L2 regularization.
    regularization = L2Regularization(LinearDifferentialOperator(diffMatrixOrder))
    p = regularization.penalty_matrix(basis)
    p = p / np.max(p)
    p = p + epsilon * np.eye(p.shape[1])
    return jnp.array(basis_matrix), jnp.array(p)





def data_peak_knots(
    periodogram: Periodogram, n_knots: int,
        frac_uniform: float = 0.0,
        frac_log: float = 0.8
) -> np.ndarray:
    """Select knots with a mix of uniform, log-spaced, and density-based placement.

         Instead of using a fixed grid (via log‐ or geomspace) to “force” a knot allocation,
     you can let the periodogram’s power distribution guide you. For example,
     you can interpret the (normalized) power as a probability density over frequency,
     compute its cumulative distribution function (CDF), and then choose knots at equally
     spaced quantiles of that CDF. In regions where the power (and hence “spikiness”) is higher,
      the CDF rises faster, so more knots will be allocated there.

    Ensures the first and last knots are at the min and max frequency.
    The remaining knots are allocated:
    - `frac_uniform`: Using uniform spacing.
    - `frac_log`: Using logarithmic spacing.
    - The rest: Using power-based density sampling.

Args:
        periodogram: Periodogram object with freqs and power.
        n_knots: Total number of knots to select.
        frac_uniform: Fraction of knots to place uniformly (can be 0).
        frac_log: Fraction of knots to place logarithmically (can be 0).

    Returns:
        An array of knot locations (frequencies).
    """
    if n_knots < 2:
        raise ValueError("At least two knots are required (min and max frequencies).")

    min_freq, max_freq = periodogram.freqs[0], periodogram.freqs[-1]

    if n_knots == 2:
        return np.array([min_freq, max_freq])

    # Ensure fractions sum to at most 1
    frac_uniform = max(0.0, min(frac_uniform, 1.0))
    frac_log = max(0.0, min(frac_log, 1.0))
    frac_density = 1.0 - (frac_uniform + frac_log)

    # Compute number of knots in each category
    n_uniform = int(frac_uniform * (n_knots - 2)) if frac_uniform > 0 else 0
    n_log = int(frac_log * (n_knots - 2)) if frac_log > 0 else 0
    n_density = max(0, (n_knots - 2) - (n_uniform + n_log))  # Remaining knots

    # Uniformly spaced knots (excluding min/max)
    uniform_knots = (
        np.linspace(min_freq, max_freq, n_uniform + 2)[1:-1] if n_uniform > 0 else np.array([])
    )

    # Log-spaced knots (excluding min/max)
    log_knots = (
        np.logspace(np.log10(min_freq), np.log10(max_freq), n_log + 2)[1:-1] if n_log > 0 else np.array([])
    )

    # Power-based density sampling
    density_knots = np.array([])
    if n_density > 0:
        power = periodogram.power.copy()
        density = power / np.sum(power)
        cdf = np.cumsum(density)

        # Compute quantiles for density-based knots
        quantiles = np.linspace(0, 1, n_density + 2)[1:-1]
        density_knots = np.interp(quantiles, cdf, periodogram.freqs)

    # Combine and sort
    knots = np.concatenate(([min_freq], uniform_knots, log_knots, density_knots, [max_freq]))
    knots = np.sort(knots)  # Ensure order

    # normalize to [0, 1]
    knots = (knots - min_freq) / (max_freq - min_freq)
    print(f"Selected knots: {knots}")
    return knots




def generate_data() -> Timeseries:
    """Generate synthetic AR noise data."""
    a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
    n_samples = 1024
    fs = 100  # Sampling frequency in Hz.
    dt = 1.0 / fs
    t = np.linspace(0, (n_samples - 1) * dt, n_samples)
    noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
    return Timeseries(t, noise)


def optimize_logpsplines_weights(
        noise_f: Periodogram, log_psplines: LogPSplines, init_weights: jnp.ndarray, num_steps: int = 1000
) -> jnp.ndarray:
    """
    Optimize spline weights by directly minimizing the negative Whittle log likelihood.

    This function wraps the optimization loop in a JAX-compiled loop using jax.lax.fori_loop.
    """

    # Now we assume that the likelihood function expects log power,
    # so compute the log of the power spectrum.
    noise_f_log = jnp.log(noise_f.power)

    # Define the loss as the negative log likelihood.
    def compute_loss(weights: jnp.ndarray) -> float:
        return -lnlikelihood(noise_f_log, log_psplines, weights)

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(init_weights)

    def step(i, state):
        weights, opt_state = state
        loss, grads = jax.value_and_grad(compute_loss)(weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (weights, opt_state)

    init_state = (init_weights, opt_state)
    final_state = jax.lax.fori_loop(0, num_steps, step, init_state)
    final_weights, _ = final_state
    return final_weights


def main():
    # Generate synthetic timeseries data and standardize it.
    noise = generate_data()
    scale = jnp.std(noise.y)
    noise.y = (noise.y - jnp.mean(noise.y)) / scale

    # Compute the periodogram and apply a high-pass filter (above 5 Hz).
    noise_f = noise.to_periodogram().highpass(5)

    # Determine knots based on the periodogram frequencies and initialize the spline model.
    knots = data_peak_knots(noise_f, n_knots=20)
    spline_model = LogPSplines(knots=knots, degree=3, diffMatrixOrder=2, n=len(noise_f.freqs))
    init_weights = jnp.zeros(spline_model.n_basis)

    # Compute the initial log likelihood.
    lnl_initial = lnlikelihood(jnp.log(noise_f.power), spline_model, init_weights)
    print("Initial log likelihood:", lnl_initial)

    # Optimize the spline weights by directly minimizing the negative log likelihood.
    optimized_weights = optimize_logpsplines_weights(noise_f, spline_model, init_weights)
    spline = jnp.exp(spline_model(optimized_weights)) * scale ** 2

    lnl_final = lnlikelihood(jnp.log(noise_f.power), spline_model, optimized_weights)
    print("Final log likelihood:", lnl_final)

    # Plot the timeseries and periodogram with the fitted spline model.
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.loglog(noise_f.freqs, noise_f.power * scale ** 2, color="lightgray", label="Data")
    ax.loglog(noise_f.freqs, spline, label="Spline", color="tab:orange")

    # get freq of knots (knots are at % of the freqs)
    idx = (knots * len(noise_f.freqs)).astype(int)
    ax.loglog(noise_f.freqs[idx], spline[idx], "o", label="Knots", color="tab:orange", ms=4)
    ax.legend(frameon=False)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
