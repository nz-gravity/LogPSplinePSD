import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.plotting.base import (
    PlotConfig,
    compute_coherence_ci,
    compute_confidence_intervals,
    compute_cross_spectra_ci,
    safe_plot,
    setup_plot_style,
    subsample_weights,
    validate_plotting_data,
)


def test_safe_plot_writes_file_and_returns_true(tmp_path):
    out = tmp_path / "safe_plot.png"

    @safe_plot(str(out), dpi=50)
    def _plot():
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 2])
        return fig

    result = _plot()
    assert result is True
    assert out.exists()


def test_safe_plot_handles_exception(tmp_path):
    out = tmp_path / "bad_plot.png"

    @safe_plot(str(out), dpi=50)
    def _plot():
        raise RuntimeError("boom")

    result = _plot()
    assert result is False
    assert not out.exists()


def test_compute_confidence_intervals_percentile():
    samples = np.arange(12.0).reshape(3, 4)
    lower, median, upper = compute_confidence_intervals(
        samples, quantiles=(0, 50, 100), method="percentile"
    )
    np.testing.assert_allclose(lower, samples.min(axis=0))
    np.testing.assert_allclose(upper, samples.max(axis=0))
    np.testing.assert_allclose(median, np.median(samples, axis=0))


def test_compute_confidence_intervals_uniform():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(10, 5))
    lower, median, upper = compute_confidence_intervals(
        samples, method="uniform", alpha=0.2
    )
    assert lower.shape == (5,)
    assert median.shape == (5,)
    assert upper.shape == (5,)


def test_compute_confidence_intervals_invalid_method():
    with pytest.raises(ValueError):
        compute_confidence_intervals(np.ones((2, 3)), method="invalid")


def test_compute_coherence_ci_upper_triangle():
    psd_samples = np.zeros((4, 3, 2, 2), dtype=np.complex128)
    psd_samples[:, :, 0, 0] = 2.0
    psd_samples[:, :, 1, 1] = 3.0
    psd_samples[:, :, 1, 0] = 1.0 + 1.0j
    ci = compute_coherence_ci(psd_samples)
    assert set(ci.keys()) == {(1, 0)}
    q05, q50, q95 = ci[(1, 0)]
    assert q05.shape == (3,)
    assert q50.shape == (3,)
    assert q95.shape == (3,)


def test_compute_cross_spectra_ci_off_diagonal():
    psd_samples = np.zeros((3, 4, 2, 2), dtype=np.complex128)
    psd_samples[:, :, 0, 1] = 1.0 + 2.0j
    psd_samples[:, :, 1, 0] = 2.0 + 1.0j
    real_ci, imag_ci = compute_cross_spectra_ci(psd_samples)
    assert set(real_ci.keys()) == {(0, 1), (1, 0)}
    assert set(imag_ci.keys()) == {(0, 1), (1, 0)}
    assert real_ci[(0, 1)][0].shape == (4,)


def test_setup_plot_style_returns_config():
    config = setup_plot_style(PlotConfig(fontsize=9))
    assert isinstance(config, PlotConfig)


def test_validate_plotting_data_missing_keys():
    assert validate_plotting_data({"a": 1}, ["a", "b"]) is False
    assert validate_plotting_data({"a": 1, "b": 2}, ["a", "b"]) is True


def test_subsample_weights_limits_samples():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(10, 2))
    np.random.seed(0)
    sub = subsample_weights(weights, max_samples=5)
    assert sub.shape == (5, 2)
    assert {tuple(row) for row in sub}.issubset(
        {tuple(row) for row in weights}
    )


def test_subsample_weights_returns_same_when_small():
    weights = np.arange(6.0).reshape(3, 2)
    sub = subsample_weights(weights, max_samples=5)
    np.testing.assert_allclose(sub, weights)
