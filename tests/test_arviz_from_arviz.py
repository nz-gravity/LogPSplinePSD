from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import xarray as xr

import log_psplines.arviz_utils.from_arviz as from_arviz_mod
from log_psplines.arviz_utils.from_arviz import (
    get_multivar_ci_summary,
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
    idata = az.from_dict({})
    idata["posterior_psd"] = xr.DataTree(dataset=xr.Dataset({"psd": psd_da}))
    out_freqs, median, lower, upper = get_posterior_psd(idata)
    np.testing.assert_allclose(out_freqs, freqs)
    np.testing.assert_allclose(median, values[1])
    np.testing.assert_allclose(lower, values[0])
    np.testing.assert_allclose(upper, values[2])


def test_get_posterior_psd_missing_group():
    idata = az.from_dict({})
    with pytest.raises(KeyError):
        get_posterior_psd(idata)


def test_get_weights_thins_samples():
    weights = np.arange(12.0).reshape(1, 6, 2)
    idata = az.from_dict({"posterior": {"weights": weights}})
    thinned = get_weights(idata, thin=2)
    assert thinned.shape == (3, 2)
    np.testing.assert_allclose(thinned[0], weights.reshape(-1, 2)[0])


def test_get_weights_missing_group():
    idata = az.from_dict({})
    with pytest.raises(KeyError):
        get_weights(idata, thin=1)


def test_get_periodogram_extracts_data():
    freqs = np.array([0.1, 0.2, 0.3])
    periodogram = np.array([1.0, 2.0, 3.0])
    periodogram_da = xr.DataArray(
        periodogram, coords={"freq": freqs}, dims=("freq",)
    )
    idata = az.from_dict({})
    idata["observed_data"] = xr.DataTree(
        dataset=xr.Dataset({"periodogram": periodogram_da})
    )
    out = get_periodogram(idata)
    np.testing.assert_allclose(out.freqs, freqs)
    np.testing.assert_allclose(out.power, periodogram)


def test_get_periodogram_missing_group():
    idata = az.from_dict({})
    with pytest.raises(KeyError):
        get_periodogram(idata)


def test_get_multivar_ci_summary_extracts_quantiles_and_truth():
    freqs = np.array([0.1, 0.2])
    percentiles = np.array([5.0, 50.0, 95.0])
    psd_real = np.arange(3 * 2 * 2 * 2, dtype=float).reshape(3, 2, 2, 2)
    psd_imag = -psd_real
    truth_real = np.ones((2, 2, 2), dtype=float) * 7.0
    truth_imag = np.ones((2, 2, 2), dtype=float) * -3.0

    truth_ds = xr.Dataset(
        {
            "psd_matrix_real": xr.DataArray(
                truth_real,
                coords={
                    "freq": freqs,
                    "channels": [0, 1],
                    "channels2": [0, 1],
                },
                dims=("freq", "channels", "channels2"),
            ),
            "psd_matrix_imag": xr.DataArray(
                truth_imag,
                coords={
                    "freq": freqs,
                    "channels": [0, 1],
                    "channels2": [0, 1],
                },
                dims=("freq", "channels", "channels2"),
            ),
        }
    )

    idata = az.from_dict({})
    idata["truth_psd"] = xr.DataTree(dataset=truth_ds)

    def _fake_quantiles(_idata, **kwargs):
        return {
            "freq": freqs,
            "percentile": percentiles,
            "real": psd_real,
            "imag": psd_imag,
            "coherence": None,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        from_arviz_mod, "get_multivar_posterior_psd_quantiles", _fake_quantiles
    )

    summary = get_multivar_ci_summary(idata)
    monkeypatch.undo()
    np.testing.assert_allclose(summary["freq"], freqs)
    np.testing.assert_allclose(summary["psd_real_q05"], psd_real[0])
    np.testing.assert_allclose(summary["psd_real_q50"], psd_real[1])
    np.testing.assert_allclose(summary["psd_real_q95"], psd_real[2])
    np.testing.assert_allclose(summary["psd_imag_q05"], psd_imag[0])
    np.testing.assert_allclose(summary["true_psd_real"], truth_real)
    np.testing.assert_allclose(summary["true_psd_imag"], truth_imag)
