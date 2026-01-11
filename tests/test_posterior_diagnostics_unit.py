import numpy as np
import pytest
import xarray as xr

from log_psplines.diagnostics.posterior_diagnostics import (
    _collect_functional_idata,
    _prepare_psd_matrix,
    _select_weight_vars,
    compute_psd_functionals,
)


def test_prepare_psd_matrix_diagonal_input():
    psd_samples = np.zeros((2, 3, 4))
    psd_samples[:, 0, :] = 1.0
    psd_samples[:, 1, :] = 2.0
    psd_samples[:, 2, :] = 3.0
    psd_matrix = _prepare_psd_matrix(psd_samples, n_channels=3)
    assert psd_matrix.shape == (2, 3, 3, 4)
    assert np.all(psd_matrix[:, 0, 0, :] == 1.0)
    assert np.all(psd_matrix[:, 1, 1, :] == 2.0)
    assert np.all(psd_matrix[:, 2, 2, :] == 3.0)
    assert np.all(psd_matrix[:, 0, 1, :] == 0.0)


def test_prepare_psd_matrix_transposes_full_matrix():
    psd_matrix = np.zeros((2, 2, 2, 4))
    psd_matrix[:, 0, 0, :] = 1.0
    psd_matrix[:, 1, 1, :] = 2.0
    psd_samples = np.transpose(psd_matrix, (0, 3, 1, 2))
    result = _prepare_psd_matrix(psd_samples, n_channels=2)
    np.testing.assert_allclose(result, psd_matrix)


def test_prepare_psd_matrix_rejects_invalid_shape():
    with pytest.raises(ValueError):
        _prepare_psd_matrix(np.zeros((2, 2)), n_channels=2)


def test_compute_psd_functionals_constant_psd():
    freqs = np.linspace(1.0, 3.0, 5)
    psd_samples = np.ones((3, 2, 5))
    psd_samples[:, 0, :] *= 2.0
    psd_samples[:, 1, :] *= 3.0
    bands = [(1.0, 3.0)]
    channel_pairs = [(0, 1)]

    results = compute_psd_functionals(psd_samples, freqs, bands, channel_pairs)
    variance = results["variance"]
    band_powers = results["band_powers"]
    coherence = results["coherence"]

    expected_var = np.array([4.0, 6.0])
    expected_matrix = np.tile(expected_var, (variance.shape[0], 1))
    np.testing.assert_allclose(variance, expected_matrix)
    np.testing.assert_allclose(band_powers[:, :, 0], expected_matrix)
    np.testing.assert_allclose(coherence[:, 0, 0], 0.0)


def test_collect_functional_idata_builds_posterior():
    freqs = np.linspace(1.0, 2.0, 4)
    psd_samples = np.ones((2, 2, 4))
    results = compute_psd_functionals(
        psd_samples, freqs, bands=[(1.0, 2.0)], channel_pairs=[(0, 1)]
    )
    idata = _collect_functional_idata(results)
    assert idata is not None
    assert "band_power" in idata.posterior
    assert "variance" in idata.posterior


def test_select_weight_vars_filters_subset():
    posterior = xr.Dataset(
        {
            "weight_0": ("draw", np.zeros(3)),
            "spline_param": ("draw", np.ones(3)),
            "alpha": ("draw", np.ones(3)),
        }
    )
    subset = _select_weight_vars(posterior, ["weight_0", "beta"])
    assert subset == ["weight_0"]

    auto = _select_weight_vars(posterior, None)
    assert set(auto).issubset({"weight_0", "spline_param"})
