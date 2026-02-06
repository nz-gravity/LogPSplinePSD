import arviz as az
import numpy as np
import pytest
import xarray as xr

from log_psplines.diagnostics import psd_compare


def test_run_univariate_identity_metrics():
    freqs = np.array([0.1, 0.2, 0.3])
    truth = np.array([1.0, 2.0, 3.0])
    percentiles = np.array([5.0, 50.0, 95.0])
    values = np.stack([truth * 0.8, truth, truth * 1.2], axis=0)
    psd_da = xr.DataArray(
        values,
        coords={"percentile": percentiles, "freq": freqs},
        dims=("percentile", "freq"),
    )
    idata = az.InferenceData(posterior_psd=xr.Dataset({"psd": psd_da}))

    metrics = psd_compare._run(idata=idata, truth=truth)
    assert metrics["riae"] == pytest.approx(0.0)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["riae_p05"] > 0.0
    assert metrics["riae_p95"] > 0.0


def test_run_univariate_scaled_metrics():
    freqs = np.array([0.1, 0.2, 0.3])
    truth = np.ones_like(freqs)
    percentiles = np.array([50.0])
    values = np.array([truth * 2.0])
    psd_da = xr.DataArray(
        values,
        coords={"percentile": percentiles, "freq": freqs},
        dims=("percentile", "freq"),
    )
    idata = az.InferenceData(posterior_psd=xr.Dataset({"psd": psd_da}))

    metrics = psd_compare._run(idata=idata, truth=truth)
    assert metrics["riae"] == pytest.approx(1.0)
    assert "coverage" not in metrics


def test_run_multivariate_identity_metrics():
    freqs = np.array([0.1, 0.2, 0.3])
    true_psd = np.zeros((3, 2, 2))
    true_psd[:, 0, 0] = 1.0
    true_psd[:, 1, 1] = 2.0
    percentiles = np.array([5.0, 50.0, 95.0])
    psd_real = np.stack([true_psd * 0.9, true_psd, true_psd * 1.1], axis=0)
    psd_imag = np.zeros_like(psd_real)
    psd_ds = xr.Dataset(
        {
            "psd_matrix_real": xr.DataArray(
                psd_real,
                coords={"percentile": percentiles, "freq": freqs},
                dims=("percentile", "freq", "channel", "channel2"),
            ),
            "psd_matrix_imag": xr.DataArray(
                psd_imag,
                coords={"percentile": percentiles, "freq": freqs},
                dims=("percentile", "freq", "channel", "channel2"),
            ),
        }
    )
    idata = az.InferenceData(posterior_psd=psd_ds)

    metrics = psd_compare._run(idata=idata, truth=true_psd)
    assert metrics["riae_matrix"] == pytest.approx(0.0)
    assert metrics["riae_diag_mean"] == pytest.approx(0.0)
    assert metrics["riae_diag_max"] == pytest.approx(0.0)
    assert metrics["coverage"] == pytest.approx(1.0)


def test_run_returns_empty_when_missing_inputs():
    freqs = np.array([0.1, 0.2])
    truth = np.ones_like(freqs)
    idata = az.InferenceData()
    assert psd_compare._run(idata=idata, truth=truth) == {}
    assert psd_compare._run(idata=None, truth=None) == {}
