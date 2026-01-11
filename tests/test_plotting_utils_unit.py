import numpy as np
import pytest

from log_psplines.datatypes import Periodogram
from log_psplines.plotting.utils import PlottingData, unpack_data


class DummySpline:
    def __init__(self, n):
        self.n = n

    def __call__(self, weights=None, use_parametric_model=True):
        base = np.linspace(1.0, 2.0, self.n)
        offset = 0.0 if weights is None else 0.01 * np.sum(weights)
        return np.log(base + offset)


def test_plotting_data_n_property():
    data = PlottingData(freqs=np.arange(4))
    assert data.n == 4
    data = PlottingData(pdgrm=np.arange(3))
    assert data.n == 3
    data = PlottingData(model=np.arange(2))
    assert data.n == 2
    with pytest.raises(ValueError):
        PlottingData().n


def test_unpack_data_with_periodogram_and_scaling():
    pdgrm = Periodogram(freqs=np.array([1.0, 2.0]), power=np.array([2.0, 4.0]))
    out = unpack_data(pdgrm=pdgrm, yscalar=0.5)
    np.testing.assert_allclose(out.pdgrm, np.array([1.0, 2.0]))
    np.testing.assert_allclose(out.freqs, pdgrm.freqs)


def test_unpack_data_with_posterior_quantiles():
    quantiles = {
        "percentile": np.array([5.0, 50.0, 95.0]),
        "values": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    }
    out = unpack_data(
        posterior_psd_quantiles=quantiles, freqs=np.array([0.1, 0.2])
    )
    np.testing.assert_allclose(out.model, np.array([3.0, 4.0]))
    assert out.ci.shape == (3, 2)


def test_unpack_data_with_model_ci():
    model_ci = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
    out = unpack_data(model_ci=model_ci, freqs=np.array([0.1, 0.2]))
    np.testing.assert_allclose(out.model, model_ci[1])
    np.testing.assert_allclose(out.ci, model_ci)
    np.testing.assert_allclose(out.freqs, np.array([0.1, 0.2]))


def test_unpack_data_with_spline_and_weights():
    spline = DummySpline(n=3)
    weights = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    out = unpack_data(
        spline_model=spline, weights=weights, freqs=np.array([0.1, 0.2, 0.3])
    )
    assert out.model.shape == (3,)
    assert out.ci.shape == (3, 3)
