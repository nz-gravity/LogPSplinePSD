import arviz as az
import numpy as np
from xarray import DataArray, Dataset

from log_psplines.diagnostics import psd_compare, run_all_diagnostics
from log_psplines.diagnostics._utils import compute_riae


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
        posterior={"x": np.random.randn(2, 20)},
        sample_stats={
            "diverging": np.zeros((2, 20)),
            "acceptance_rate": np.full((2, 20), 0.8),
        },
    )
    idata.add_groups(posterior_psd=posterior_psd)
    return idata, freqs, psd_values


def test_psd_compare_matches_manual_riae():
    truth = np.linspace(1.0, 2.0, 5)
    idata, freqs, psd_values = _build_idata_with_psd(truth)
    result = psd_compare.run(idata=idata, truth=truth)

    expected_riae = compute_riae(psd_values[1], truth, freqs)
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
        "time_domain",
        "whitening",
        "vi",
    }
    assert expected_modules.issubset(set(result.keys()))

    for module_metrics in result.values():
        assert isinstance(module_metrics, dict)
        assert all(isinstance(v, float) for v in module_metrics.values())
