import os

import numpy as np
import pytest
import scipy

from log_psplines.datatypes import Periodogram, Timeseries


@pytest.fixture
def outdir():
    outdir = "test_output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


@pytest.fixture
def mock_pdgrm() -> Periodogram:
    """Generate synthetic AR noise data."""
    a_coeff = [1, -2.2137, 2.9403, -2.1697, 0.9606]
    n_samples = 1024
    fs = 100  # Sampling frequency in Hz.
    dt = 1.0 / fs
    t = np.linspace(0, (n_samples - 1) * dt, n_samples)
    noise = scipy.signal.lfilter([1], a_coeff, np.random.randn(n_samples))
    noise = (noise - np.mean(noise)) / np.std(noise)
    return Timeseries(t, noise).to_periodogram().highpass(5)
