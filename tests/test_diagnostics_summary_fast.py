from __future__ import annotations

import arviz as az
import numpy as np


def _make_min_idata() -> az.InferenceData:
    rng = np.random.default_rng(0)
    posterior = {
        "phi": rng.normal(size=(1, 8)),
        "delta": rng.normal(size=(1, 8)),
        "weights": rng.normal(size=(1, 8, 3)),
    }
    sample_stats = {
        "accept_prob": rng.uniform(size=(1, 8)),
        "diverging": np.zeros((1, 8), dtype=int),
        "num_steps": np.ones((1, 8), dtype=int),
        "tree_depth": np.ones((1, 8), dtype=int),
        "step_size": np.ones((1, 8), dtype=float) * 0.1,
    }
    idata = az.from_dict(
        {"posterior": posterior, "sample_stats": sample_stats}
    )
    idata.attrs["sampler_type"] = "nuts"
    idata.attrs["max_tree_depth"] = 10
    idata.attrs["ess"] = np.asarray([250.0, 500.0])
    idata.attrs["rhat"] = np.asarray([1.001, 1.005, 1.010])
    return idata


def _attach_vi_psd(idata: az.InferenceData) -> az.InferenceData:
    freq = np.array([0.1, 0.2, 0.3], dtype=float)
    percentiles = np.array([5.0, 50.0, 95.0], dtype=float)
    values = np.array(
        [
            [0.8, 0.9, 1.0],
            [1.0, 1.1, 1.2],
            [1.4, 1.5, 1.6],
        ]
    )
    import xarray as xr

    dataset = xr.Dataset(
        {
            "psd": xr.DataArray(
                values,
                dims=("percentile", "freq"),
                coords={"percentile": percentiles, "freq": freq},
            )
        }
    )
    idata["vi_posterior_psd"] = xr.DataTree(dataset=dataset)
    return idata


def test_generate_diagnostics_summary_light_skips_expensive(
    monkeypatch, tmp_path
):
    import log_psplines.diagnostics.plotting as diag_mod

    idata = _make_min_idata()

    def boom(*args, **kwargs):  # pragma: no cover
        raise AssertionError("This should not run in light mode")

    monkeypatch.setattr(diag_mod, "run_all_diagnostics", boom)
    monkeypatch.setattr(diag_mod.az, "loo", boom)
    monkeypatch.setattr(diag_mod.az, "ess", boom)
    monkeypatch.setattr(diag_mod.az, "rhat", boom)

    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="light"
    )
    assert "Sampler: nuts" in text
    assert (tmp_path / "diagnostics_summary.txt").exists()


def test_generate_diagnostics_summary_off_writes_nothing(tmp_path):
    import log_psplines.diagnostics.plotting as diag_mod

    idata = _make_min_idata()
    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="off"
    )
    assert text == ""
    assert not (tmp_path / "diagnostics_summary.txt").exists()


def test_generate_diagnostics_summary_full_uses_run_all(monkeypatch, tmp_path):
    import log_psplines.diagnostics.plotting as diag_mod

    idata = _make_min_idata()
    calls = {"n": 0}

    def fake_run_all_diagnostics(
        *, idata=None, truth=None, psd_ref=None, **kwargs
    ):
        calls["n"] += 1
        return {
            "mcmc": {
                "ess_bulk_min": 123.0,
                "ess_bulk_median": 456.0,
                "rhat_max": 1.02,
                "rhat_mean": 1.01,
                "psis_khat_max": 0.55,
            }
        }

    def boom(*args, **kwargs):  # pragma: no cover
        raise AssertionError("Fallback scans should not run in this test")

    monkeypatch.setattr(
        diag_mod, "run_all_diagnostics", fake_run_all_diagnostics
    )
    monkeypatch.setattr(diag_mod.az, "ess", boom)
    monkeypatch.setattr(diag_mod.az, "rhat", boom)
    monkeypatch.setattr(diag_mod.az, "loo", boom)

    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="full"
    )
    assert calls["n"] == 1
    assert "ESS bulk: min=123" in text
    assert "PSIS k-hat" in text
    assert (tmp_path / "diagnostics_summary.txt").exists()


def test_generate_diagnostics_summary_full_uses_cached_attrs(
    monkeypatch, tmp_path
):
    import log_psplines.diagnostics.plotting as diag_mod

    idata = _make_min_idata()
    idata.attrs["full_diagnostics_computed"] = 1
    idata.attrs["mcmc_ess_bulk_min"] = 123.0
    idata.attrs["mcmc_ess_bulk_median"] = 456.0
    idata.attrs["mcmc_ess_tail_min"] = 222.0
    idata.attrs["mcmc_ess_tail_median"] = 333.0
    idata.attrs["mcmc_rhat_max"] = 1.02
    idata.attrs["mcmc_rhat_mean"] = 1.01
    idata.attrs["mcmc_psis_khat_max"] = 0.55
    idata.attrs["energy_ebfmi_overall"] = 0.25
    idata.attrs["psd_bands_variance_median"] = 1.2

    def boom(*args, **kwargs):  # pragma: no cover
        raise AssertionError("This should not run when cached attrs exist")

    monkeypatch.setattr(diag_mod, "run_all_diagnostics", boom)

    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="full"
    )
    assert "ESS bulk: min=123" in text
    assert "ESS tail: min=222" in text
    assert "PSIS k-hat" in text
    assert "E-BFMI" in text
    assert (tmp_path / "diagnostics_summary.txt").exists()


def test_generate_diagnostics_summary_includes_vi_vs_nuts_block(tmp_path):
    import log_psplines.diagnostics.plotting as diag_mod

    idata = _attach_vi_psd(_make_min_idata())
    idata.attrs["riae"] = 0.11
    idata.attrs["l2_matrix"] = 0.22
    idata.attrs["coverage"] = 0.92
    idata.attrs["ci_width"] = 0.4
    idata.attrs["vi_riae_vs_truth"] = 0.18
    idata.attrs["vi_coverage_vs_truth"] = 0.87
    idata.attrs["vi_ci_width_vs_truth"] = 0.6

    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="light"
    )
    assert "VI vs NUTS PSD accuracy:" in text
    assert "L2 (matrix): 0.220" in text
    assert "RIAE: VI=0.180 | NUTS=0.110" in text
    assert "Coverage: VI=87.0% | NUTS=92.0%" in text
    assert "CI width: VI=0.6 | NUTS=0.4" in text


def test_generate_diagnostics_summary_reads_vi_metrics_from_group_attrs(
    tmp_path,
):
    import log_psplines.diagnostics.plotting as diag_mod

    idata = _attach_vi_psd(_make_min_idata())
    idata.attrs["riae_matrix"] = 0.11
    idata.attrs["coverage"] = 0.92
    idata.attrs["ci_width"] = 0.4
    idata.vi_posterior_psd.attrs["riae_matrix"] = 0.18
    idata.vi_posterior_psd.attrs["coverage"] = 0.87

    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="light"
    )
    assert "VI vs NUTS PSD accuracy:" in text
    assert "RIAE: VI=0.180 | NUTS=0.110" in text
    assert "Coverage: VI=87.0% | NUTS=92.0%" in text
    assert "CI width: VI=unavailable | NUTS=0.4" in text


def test_generate_diagnostics_summary_ci_width_survives_missing_truth(
    tmp_path,
):
    import log_psplines.diagnostics.plotting as diag_mod

    idata = _attach_vi_psd(_make_min_idata())
    idata.attrs["ci_width"] = 0.25

    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="light"
    )
    assert "VI vs NUTS PSD accuracy:" in text
    assert "CI width:" in text
    assert "RIAE: unavailable" not in text
    assert "Coverage: unavailable" not in text
