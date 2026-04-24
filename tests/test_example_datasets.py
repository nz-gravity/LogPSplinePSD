import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.example_datasets.ar_data import ARData
from log_psplines.example_datasets.varma_data import VARMAData


def test_ar(outdir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ar_data = ARData(
            order=i + 1, duration=8.0, fs=1024.0, sigma=1e-21, seed=42
        )
        ax = ar_data.plot(ax=ax)
        ax.set_title(f"AR({i + 1}) Process")
        ax.grid(True)
        # turn off axes spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/ar_processes.png", bbox_inches="tight", dpi=300)


def test_varma_data(outdir):
    varma_data = VARMAData(n_samples=512, fs=64.0, seed=0)
    varma_data.plot(fname=f"{outdir}/varma_data_analysis.png")
    freq_hz = np.asarray(varma_data.freq)

    # Frequency axis should live in Hz up to the Nyquist frequency
    assert np.all(freq_hz > 0)
    assert np.isclose(freq_hz.max(), varma_data.fs / 2.0)
    np.testing.assert_allclose(np.diff(freq_hz), freq_hz[0])

    # True PSD should integrate to the channel variances (within sampling noise)
    psd = varma_data.get_true_psd()
    psd_vars = np.array(
        [np.trapezoid(psd[:, i, i].real, freq_hz) for i in range(varma_data.p)]
    )
    empirical_vars = np.var(varma_data.data, axis=0)
    np.testing.assert_allclose(psd_vars, empirical_vars, rtol=0.1)


def test_varma_validity_flags():
    stable = VARMAData(
        n_samples=128,
        var_coeffs=np.array([[[0.35, 0.0], [0.0, 0.25]]], dtype=float),
        vma_coeffs=np.eye(2)[None, ...],
        sigma=np.eye(2),
        seed=1,
    )
    assert stable.is_var_stationary is True
    assert stable.is_valid_var_dataset is True
    assert stable.is_empirically_stationary is True
    assert stable.empirical_stationarity_metrics is not None
    assert stable.var_companion_spectral_radius is not None
    assert stable.var_companion_spectral_radius < 1.0

    non_stationary = VARMAData(
        n_samples=128,
        var_coeffs=np.array([[[1.05, 0.0], [0.0, 0.95]]], dtype=float),
        vma_coeffs=np.eye(2)[None, ...],
        sigma=np.eye(2),
        seed=2,
    )
    assert non_stationary.is_var_stationary is False
    assert non_stationary.is_valid_var_dataset is False
    assert non_stationary.is_empirically_stationary is False
    assert non_stationary.empirical_stationarity_metrics is not None
    assert non_stationary.var_companion_spectral_radius is not None
    assert non_stationary.var_companion_spectral_radius >= 1.0
