"""Lightweight checks for example datasets used by researchers."""

import numpy as np

from log_psplines.example_datasets.ls2_data import LS2Data
from log_psplines.example_datasets.varma_data import VARMAData


def test_varma_example_has_consistent_data_and_spectrum():
    data = VARMAData.ar(order=2, n_samples=128, fs=32.0, seed=42)
    assert data.ts.data.shape == (128, 1)
    assert data.get_true_psd().shape == (64, 1, 1)
    assert np.isfinite(data.ts.data).all()
    assert np.isfinite(data.get_true_psd()).all()
    assert np.all(data.get_true_psd()[:, 0, 0].real > 0)


def test_ls2_example_exposes_finite_series_and_frequency_grid():
    data = LS2Data(n_samples=256, fs=32.0, seed=42)
    assert data.data.shape == (256, 1)
    assert data.get_true_psd().shape == (256, len(data.freq))
    assert np.isfinite(data.data).all()
    assert np.isfinite(data.get_true_psd()).all()
    assert np.all(data.freq > 0)
    assert np.isclose(data.freq.max(), data.fs / 2)
