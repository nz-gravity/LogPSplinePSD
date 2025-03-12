

from log_psplines.mcmc import bayesian_model
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

from log_psplines.psplines import LogPSplines, data_peak_knots
from log_psplines.datasets import Periodogram, Timeseries
from log_psplines.initialisation import optimize_logpsplines_weights
import scipy


def generate_data():
    a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
    n_samples = 1024
    fs = 100  # Sampling frequency in Hz.
    dt = 1.0 / fs
    t = np.linspace(0, (n_samples - 1) * dt, n_samples)
    noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
    noise = (noise - np.mean(noise)) / np.std(noise)
    noise_t = Timeseries(t, noise)
    noise_f = noise_t.to_periodogram().highpass(5)
    return noise_f




if __name__ == '__main__':
    pdgrm = generate_data()
    knots = data_peak_knots(pdgrm, n_knots=20)
    spline_model = LogPSplines(
        knots=knots, degree=3, diffMatrixOrder=2, n=len(pdgrm.freqs)
    )
    samples = run_mcmc(pdgrm, knots, spline_model)

    posterior_predictive = jnp.array([spline_model(samples['weights'][i]) for i in range(1000)])
    qtls = jnp.percentile(posterior_predictive, q=jnp.array([16, 50, 84]), axis=0)
    qtls = jnp.exp(qtls)

    # make plots
    plt.figure()
    plt.loglog(pdgrm.freqs, pdgrm.power, color='gray', alpha=0.3)
    plt.plot(pdgrm.freqs, qtls[1], color='tab:orange')
    plt.fill_between(pdgrm.freqs, qtls[0], qtls[2], color='tab:orange', alpha=0.5)
    plt.show()


