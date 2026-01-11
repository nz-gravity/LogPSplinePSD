import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.diagnostics.psd_posterior_diagnostics import (
    compute_psd_credible_bands,
    compute_riae,
    compute_welch_coverage,
    plot_psd_with_bands_and_welch,
)


def test_compute_psd_credible_bands_shapes():
    rng = np.random.default_rng(0)
    psd_samples = rng.normal(size=(5, 2, 4))
    median, lower, upper = compute_psd_credible_bands(psd_samples, 0.8)
    assert median.shape == (2, 4)
    assert lower.shape == (2, 4)
    assert upper.shape == (2, 4)


def test_compute_psd_credible_bands_rejects_invalid_shape():
    with pytest.raises(ValueError):
        compute_psd_credible_bands(np.ones((2, 3)), 0.9)


def test_compute_welch_coverage_identity():
    welch = np.array([[1.0, 2.0], [3.0, 4.0]])
    bands = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[0.5, 1.5], [2.5, 3.5]]),
        np.array([[1.5, 2.5], [3.5, 4.5]]),
    )
    coverage = compute_welch_coverage(welch, bands)
    np.testing.assert_allclose(coverage, np.ones(2))


def test_compute_welch_coverage_mismatched_shapes():
    welch = np.ones((2, 3))
    bands = (np.ones((2, 3)), np.ones((1, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        compute_welch_coverage(welch, bands)


def test_compute_riae_identity_and_bad_freqs():
    freqs = np.array([1.0, 2.0, 3.0])
    psd = np.array([1.0, 1.0, 1.0])
    result = compute_riae(psd, psd, freqs)
    assert float(result.squeeze()) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        compute_riae(psd, psd, np.array([1.0, 2.0]))


def test_plot_psd_with_bands_and_welch_smoke():
    freqs = np.array([1.0, 2.0, 3.0])
    median = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
    lower = median * 0.9
    upper = median * 1.1
    welch = median * 1.05
    fig = plot_psd_with_bands_and_welch(freqs, median, lower, upper, welch)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_psd_with_bands_and_welch_rejects_channel_names():
    freqs = np.array([1.0, 2.0, 3.0])
    median = np.array([[1.0, 1.5, 2.0]])
    lower = median * 0.9
    upper = median * 1.1
    welch = median * 1.05
    with pytest.raises(ValueError):
        plot_psd_with_bands_and_welch(
            freqs, median, lower, upper, welch, channel_names=["a", "b"]
        )
