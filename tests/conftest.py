import os

import numpy as np
import pytest
import scipy

from log_psplines.datatypes import Periodogram, Timeseries
from log_psplines.example_datasets.ar_data import ARData


@pytest.fixture
def outdir():
    outdir = "test_output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


@pytest.fixture
def mock_pdgrm() -> Periodogram:
    """Generate synthetic AR noise data."""
    return ARData(order=4, duration=4.0, fs=512, seed=42).periodogram
