import arviz as az
import numpy as np
import pytest
import xarray as xr

from log_psplines.arviz_utils.from_arviz import (
    get_periodogram,
    get_posterior_psd,
    get_weights,
)


def test_get_posterior_psd_extracts_percentiles():
    freqs = np.array([0.1, 0.2])
    percentiles = np.array([5.0, 50.0, 95.0])
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    psd_da = xr.DataArray(
        values,
        coords={"percentile": percentiles, "freq": freqs},
        dims=("percentile", "freq"),
    )
    idata = az.InferenceData(posterior_psd=xr.Dataset({"psd": psd_da}))
    out_freqs, median, lower, upper = get_posterior_psd(idata)
    np.testing.assert_allclose(out_freqs, freqs)
    np.testing.assert_allclose(median, values[1])
    np.testing.assert_allclose(lower, values[0])
    np.testing.assert_allclose(upper, values[2])


def test_get_posterior_psd_missing_group():
    idata = az.InferenceData()
    with pytest.raises(KeyError):
        get_posterior_psd(idata)


def test_get_weights_thins_samples():
    weights = np.arange(12.0).reshape(1, 6, 2)
    idata = az.from_dict(posterior={"weights": weights})
    thinned = get_weights(idata, thin=2)
    assert thinned.shape == (3, 2)
    np.testing.assert_allclose(thinned[0], weights.reshape(-1, 2)[0])


def test_get_weights_missing_group():
    idata = az.InferenceData()
    with pytest.raises(KeyError):
        get_weights(idata, thin=1)


def test_get_periodogram_extracts_data():
    freqs = np.array([0.1, 0.2, 0.3])
    periodogram = np.array([1.0, 2.0, 3.0])
    periodogram_da = xr.DataArray(
        periodogram, coords={"freq": freqs}, dims=("freq",)
    )
    idata = az.InferenceData(
        observed_data=xr.Dataset({"periodogram": periodogram_da})
    )
    out = get_periodogram(idata)
    np.testing.assert_allclose(out.freqs, freqs)
    np.testing.assert_allclose(out.power, periodogram)


def test_get_periodogram_missing_group():
    idata = az.InferenceData()
    with pytest.raises(KeyError):
        get_periodogram(idata)
