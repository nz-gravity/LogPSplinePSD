import arviz as az
import numpy as np
import pytest
import xarray as xr
from scipy.integrate import simpson

from log_psplines.diagnostics import psd_bands


def test_run_returns_empty_without_psd():
    idata = az.InferenceData()
    assert psd_bands._run(idata=idata) == {}


def test_run_univariate_with_percentiles():
    freqs = np.array([0.1, 0.2, 0.3])
    percentiles = np.array([5.0, 50.0, 95.0])
    values = np.array(
        [
            [0.8, 0.8, 0.8],
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
        ]
    )
    psd_da = xr.DataArray(
        values,
        coords={"percentile": percentiles, "freq": freqs},
        dims=("percentile", "freq"),
    )
    idata = az.InferenceData(posterior_psd=xr.Dataset({"psd": psd_da}))
    metrics = psd_bands._run(idata=idata)
    expected_median = float(simpson(values[1], x=freqs))
    expected_width = float(
        simpson(values[2], x=freqs) - simpson(values[0], x=freqs)
    )
    assert metrics["variance_median"] == pytest.approx(expected_median)
    assert metrics["variance_ci_width"] == pytest.approx(expected_width)


def test_run_univariate_without_percentile_coord():
    freqs = np.array([0.1, 0.2, 0.3])
    values = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )
    psd_da = xr.DataArray(
        values, coords={"freq": freqs}, dims=("sample", "freq")
    )
    idata = az.InferenceData(posterior_psd=xr.Dataset({"psd": psd_da}))
    metrics = psd_bands._run(idata=idata)
    assert "variance_median" in metrics
    assert "variance_ci_width" in metrics


def test_run_multivariate_with_coherence():
    freqs = np.array([0.1, 0.2, 0.3])
    percentiles = np.array([5.0, 50.0, 95.0])
    base = np.zeros((3, 2, 2))
    base[:, 0, 0] = 1.0
    base[:, 1, 1] = 2.0
    psd_real = np.stack([base * 0.8, base, base * 1.2], axis=0)
    psd_da = xr.DataArray(
        psd_real,
        coords={"percentile": percentiles, "freq": freqs},
        dims=("percentile", "freq", "channel", "channel2"),
    )
    coherence = np.stack(
        [np.zeros((3, 2, 2)), np.full((3, 2, 2), 0.5), np.ones((3, 2, 2))],
        axis=0,
    )
    coh_da = xr.DataArray(
        coherence,
        coords={"percentile": percentiles, "freq": freqs},
        dims=("percentile", "freq", "channel", "channel2"),
    )
    idata = az.InferenceData(
        posterior_psd=xr.Dataset(
            {"psd_matrix_real": psd_da, "coherence": coh_da}
        )
    )
    metrics = psd_bands._run(idata=idata)
    expected_var0 = float(simpson(base[:, 0, 0], x=freqs))
    expected_var1 = float(simpson(base[:, 1, 1], x=freqs))
    assert metrics["variance_median_mean"] == pytest.approx(
        (expected_var0 + expected_var1) / 2.0
    )
    assert metrics["variance_ci_width_mean"] > 0.0
    assert metrics["coherence_median_max"] == pytest.approx(0.5)
