from typing import Tuple

import numpy as np
from jax import numpy as jnp
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis
from scipy.interpolate import interp1d
import scipy
import matplotlib.pyplot as plt

class LogPSplines:

    def __init__(
            self,
            knots: np.array,
            degree: int,
            diffMatrixOrder: int = 2,
            n_grid_points: int = None,
    ):
        """Initialise the PSplines class

        Parameters:
        ----------
        knots : np.array
            The knots of the spline basis
        degree : int
            The degree of the spline basis
        diffMatrixOrder : int
            The order of the difference matrix used to calculate the penalty matrix
        n_grid_points : int
            The number of points to evaluate the basis functions at
            If None, then the number of grid points is set to the maximum
            between 501 and 10 times the number of basis elements.
        logged : bool
            If True, the penalty matrix is calculated using all the knots
            If False, the penalty matrix is calculated using all the knots except the last one
        """
        assert degree > diffMatrixOrder
        assert degree in [0, 1, 2, 3, 4, 5]
        assert diffMatrixOrder in [0, 1, 2]
        assert len(knots) >= degree, f"#knots: {len(knots)}, degree: {degree}"

        self.knots = knots
        self.degree = degree
        self.order = self.degree + 1
        self.diffMatrixOrder = diffMatrixOrder
        self.n_grid_points = max(501, 10 * self.n_basis) if n_grid_points is None else n_grid_points

        self.basis, self.penalty_matrix = generate_basis_and_penalty_matrix(
            self.knots,
            self.degree,
            self.n_grid_points,
            self.diffMatrixOrder,
        )

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    @property
    def n_basis(self) -> int:
        return self.n_knots + self.degree - 1

    def __call__(self, weights: jnp.ndarray = None) -> jnp.ndarray:
        weighted_splines = jnp.sum(weights[:, None] * self.basis.T, axis=0)
        return weighted_splines - jnp.log(2 * jnp.pi)


def lnlikelihood(lndata, log_psplines: LogPSplines, weights: jnp.ndarray, alph: float = None) -> float:
    """Whittle log likelihood"""
    lnmodel = log_psplines(weights)
    integrand = lnmodel + np.exp(lndata - lnmodel - jnp.log(2 * np.pi))
    lnlike = -jnp.sum(integrand) / 2

    # if inf make the lnlikelihood the smallest possible value
    if not jnp.isfinite(lnlike):
        return -1e300
    return lnlike


def data_peak_knots(data: np.ndarray, n_knots: int) -> np.ndarray:
    """Returns knots at the peaks of the data"""
    aux = np.sqrt(data)
    dens = np.abs(aux - np.mean(aux)) / np.std(aux)
    n = len(data)

    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)

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
    return invDf(np.linspace(0, 1, num=n_knots))


def generate_basis_and_penalty_matrix(knots: np.ndarray, degree: int, n_grid_points: int, diffMatrixOrder: int,
                                      epsilon: float = 1e-6, ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a B-spline basis matrix of any degree given a set of knots

    Uses:
    grid_points : np.ndarray of shape (n,)
    knots : np.ndarray of shape (k,) [Starting at 0, ending at 1]
    degree : int

    Returns:
    --------
    basis_matrix : np.ndarray of shape (n_grid_points, n_basis_elements)
    """
    order = degree + 1
    basis = BSplineBasis(order=order, knots=knots)
    grid_points = np.linspace(0, 1, n_grid_points)
    basis_matrix = basis.to_basis().to_grid(grid_points).data_matrix.squeeze().T

    knots_with_boundary = np.concatenate(
        [
            np.repeat(0, degree),
            knots,
            np.repeat(1, degree),
        ]
    )

    # normalize basis
    n_knots = len(knots_with_boundary)
    mid_to_end_knots = knots_with_boundary[degree + 1:]
    start_to_mid_knots = knots_with_boundary[
                         : (n_knots - degree - 1)
                         ]
    norm_factor = (mid_to_end_knots - start_to_mid_knots) / (
            degree + 1
    )
    norm_factor[norm_factor == 0] = np.inf
    basis_matrix = basis_matrix / norm_factor

    # penalty matrix
    regularization = L2Regularization(
        LinearDifferentialOperator(diffMatrixOrder)
    )
    p = regularization.penalty_matrix(basis)
    p = p / np.max(p)
    p = p + epsilon * np.eye(p.shape[1])

    return jnp.array(basis_matrix), jnp.array(p)



def generate_data():
    # AR filter coefficients for noise generation (for lfilter, filter numerator = [1])
    a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
    n_samples = 1024  # number of samples
    fs = 100  # sampling frequency in Hz
    dt = 1.0 / fs
    t = np.linspace(0, (n_samples - 1) * dt, n_samples)
    noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
    return t, noise





def main():

    t, noise = generate_data()
    fs = 1/ (t[1] - t[0])

    # plot the data + periodogram
    fig, ax = plt.subplots(2, 1, figsize=(4, 6))
    ax[0].plot(t, noise)
    ax[1].magnitude_spectrum(noise, Fs=fs, scale='dB', color='C1')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()