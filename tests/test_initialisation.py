import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pytest
import scipy

from log_psplines.datasets import Periodogram, Timeseries
from log_psplines.initialisation import optimize_logpsplines_weights
from log_psplines.psplines import LogPSplines, data_peak_knots
from log_psplines.sampling import lnlikelihood


@pytest.fixture
def outdir():
    outdir = "test_output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


@pytest.fixture
def mock_timeseries() -> Timeseries:
    """Generate synthetic AR noise data."""
    a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
    n_samples = 1024
    fs = 100  # Sampling frequency in Hz.
    dt = 1.0 / fs
    t = np.linspace(0, (n_samples - 1) * dt, n_samples)
    noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
    return Timeseries(t, noise)


def test_spline_init(mock_timeseries: Timeseries, outdir):
    t0 = time.time()

    scale = jnp.std(mock_timeseries.y)
    mock_timeseries.y = (
        mock_timeseries.y - jnp.mean(mock_timeseries.y)
    ) / scale

    # Compute the periodogram and apply a high-pass filter (above 5 Hz).
    noise_f = mock_timeseries.to_periodogram().highpass(5)

    # Determine knots based on the periodogram frequencies and initialize the spline model.
    knots = data_peak_knots(noise_f, n_knots=20)
    spline_model = LogPSplines(
        knots=knots, degree=3, diffMatrixOrder=2, n=len(noise_f.freqs)
    )
    init_weights = jnp.zeros(spline_model.n_basis)

    # Compute the initial log likelihood.
    lnl_initial = lnlikelihood(
        jnp.log(noise_f.power), spline_model(init_weights)
    )
    print("Initial log likelihood:", lnl_initial)
    # check lnl is finite
    assert jnp.isfinite(lnl_initial)

    # Optimize the spline weights by directly minimizing the negative log likelihood.
    optimized_weights = optimize_logpsplines_weights(
        noise_f, spline_model, init_weights
    )
    spline = jnp.exp(spline_model(optimized_weights)) * scale**2

    lnl_final = lnlikelihood(
        jnp.log(noise_f.power), spline_model(optimized_weights)
    )
    print("Final log likelihood:", lnl_final)

    # Plot the timeseries and periodogram with the fitted spline model.
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.loglog(
        noise_f.freqs,
        noise_f.power * scale**2,
        color="lightgray",
        label="Data",
    )
    ax.loglog(noise_f.freqs, spline, label="Spline", color="tab:orange")

    # get freq of knots (knots are at % of the freqs)
    idx = (knots * len(noise_f.freqs)).astype(int)
    ax.loglog(
        noise_f.freqs[idx],
        spline[idx],
        "o",
        label="Knots",
        color="tab:orange",
        ms=4,
    )
    ax.legend(frameon=False)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "test_spline_init.png"))

    assert lnl_final > lnl_initial
    assert float(time.time() - t0) < 2
