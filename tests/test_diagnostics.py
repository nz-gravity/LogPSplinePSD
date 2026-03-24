import arviz as az
import matplotlib
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from log_psplines.diagnostics import psd_compare, run_all_diagnostics
from log_psplines.diagnostics._utils import (
    compute_riae,
    interior_frequency_slice,
)
from log_psplines.diagnostics.derived_weights import (
    HDI_PROB,
    compute_weight_summaries,
)
from log_psplines.diagnostics.plotting import DiagnosticsConfig


def _build_idata_with_psd(truth: np.ndarray, q50_scale: float = 1.1):
    freqs = np.linspace(0.0, 1.0, truth.size)
    percentiles = np.array([5.0, 50.0, 95.0], dtype=float)
    q05 = truth * 0.9
    q50 = truth * q50_scale
    q95 = truth * 1.2
    psd_values = np.stack([q05, q50, q95], axis=0)

    posterior_psd = Dataset(
        {
            "psd": DataArray(
                psd_values,
                dims=["percentile", "freq"],
                coords={"percentile": percentiles, "freq": freqs},
            )
        }
    )

    idata = az.from_dict(
        {
            "posterior": {"x": np.random.randn(2, 20)},
            "sample_stats": {
                "diverging": np.zeros((2, 20)),
                "acceptance_rate": np.full((2, 20), 0.8),
            },
        }
    )
    idata["posterior_psd"] = xr.DataTree(dataset=posterior_psd)
    return idata, freqs, psd_values


def test_psd_compare_matches_manual_riae():
    truth = np.linspace(1.0, 2.0, 5)
    idata, freqs, psd_values = _build_idata_with_psd(truth)
    result = psd_compare._run(idata=idata, truth=truth)

    idx = interior_frequency_slice(freqs.size)
    expected_riae = compute_riae(psd_values[1][idx], truth[idx], freqs[idx])
    assert np.isclose(result["riae"], expected_riae)
    assert 0.0 < result["coverage"] <= 1.0


def test_run_all_orchestrates_modules():
    rng = np.random.default_rng(0)
    truth = np.linspace(1.0, 2.0, 5)
    idata, _, _ = _build_idata_with_psd(truth, q50_scale=1.05)

    signals = rng.normal(size=100)
    vi_diag = {
        "losses": np.array([0.0, -0.5, -0.7]),
        "psis_khat_max": 0.6,
        "psis_moment_summary": {
            "hyperparameters": [{"bias_pct": 12.0, "var_ratio": 0.9}],
            "weights": {"var_ratio_median": 1.1},
        },
        "vi_posterior_psd": xr.DataTree(dataset=idata.posterior_psd.dataset),
    }

    result = run_all_diagnostics(
        idata=idata,
        truth=truth,
        signals=signals,
        idata_vi=vi_diag,
    )

    expected_modules = {
        "mcmc",
        "psd_compare",
        "psd_bands",
        "vi",
    }
    assert expected_modules.issubset(set(result.keys()))

    for module_metrics in result.values():
        assert isinstance(module_metrics, dict)
        assert all(isinstance(v, float) for v in module_metrics.values())
    assert result["vi"]["ci_width_vs_truth"] > 0.0
    assert result["vi"]["riae_vs_truth"] >= 0.0


def test_run_all_multivar_vi_exports_l2_metric():
    percentiles = np.array([5.0, 50.0, 95.0], dtype=float)
    freqs = np.array([0.1, 0.2, 0.3], dtype=float)
    true_psd = np.zeros((3, 2, 2), dtype=float)
    true_psd[:, 0, 0] = 1.0
    true_psd[:, 1, 1] = 2.0

    q05 = true_psd * 0.9
    q50 = true_psd * 1.05
    q95 = true_psd * 1.2
    psd_real = np.stack([q05, q50, q95], axis=0)
    psd_imag = np.zeros_like(psd_real)
    vi_psd = xr.Dataset(
        {
            "psd_matrix_real": DataArray(
                psd_real,
                dims=["percentile", "freq", "channel", "channel2"],
                coords={"percentile": percentiles, "freq": freqs},
            ),
            "psd_matrix_imag": DataArray(
                psd_imag,
                dims=["percentile", "freq", "channel", "channel2"],
                coords={"percentile": percentiles, "freq": freqs},
            ),
        }
    )

    idata = az.from_dict({"posterior": {"x": np.random.randn(1, 4)}})
    vi_diag = {
        "losses": np.array([0.0, -0.5, -0.7]),
        "vi_posterior_psd": xr.DataTree(dataset=vi_psd),
    }

    result = run_all_diagnostics(
        idata=idata,
        truth=true_psd,
        idata_vi=vi_diag,
    )

    assert result["vi"]["l2_matrix_vs_truth"] >= 0.0
    assert result["vi"]["riae_matrix_vs_truth"] >= 0.0


def test_run_all_ignores_energy_channel_metrics():
    rng = np.random.default_rng(123)
    truth = np.linspace(1.0, 2.0, 5)
    idata, _, _ = _build_idata_with_psd(truth, q50_scale=1.0)

    energy = rng.normal(size=(2, 20))
    idata.sample_stats["energy_channel_0"] = ("chain", "draw"), energy

    result = run_all_diagnostics(
        idata=idata,
        truth=truth,
        signals=rng.normal(size=100),
    )

    assert "energy" not in result


def test_derived_weight_summaries_penalty():
    weights = np.array([[[1.0, -2.0, 3.0], [0.5, -1.5, 2.0]]])
    idata = az.from_dict({"posterior": {"weights": weights}})
    penalty = np.diag([1.0, 2.0, 3.0])
    spline_model = Dataset(
        {
            "penalty_matrix": DataArray(
                penalty,
                dims=["weights_dim_row", "weights_dim_col"],
            )
        }
    )
    idata["spline_model"] = xr.DataTree(dataset=spline_model)

    scalar, vector = compute_weight_summaries(idata, hdi_prob=HDI_PROB)
    assert "w_rms__weights" in scalar
    assert "w_maxabs__weights" in scalar
    assert "penalty__weights" in scalar
    assert "w_hdi_width__weights" in vector

    w = weights.reshape((weights.shape[0], weights.shape[1], -1))
    expected_rms = np.sqrt(np.mean(w * w, axis=-1))
    expected_penalty = np.einsum("...i,ij,...j->...", w, penalty, w)
    np.testing.assert_allclose(
        scalar["w_rms__weights"].values, expected_rms, rtol=1e-6
    )
    np.testing.assert_allclose(
        scalar["penalty__weights"].values, expected_penalty, rtol=1e-6
    )
