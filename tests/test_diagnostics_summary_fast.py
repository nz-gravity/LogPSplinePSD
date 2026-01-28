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
    idata = az.from_dict(posterior=posterior, sample_stats=sample_stats)
    idata.attrs["sampler_type"] = "nuts"
    idata.attrs["max_tree_depth"] = 10
    idata.attrs["ess"] = np.asarray([250.0, 500.0])
    idata.attrs["rhat"] = np.asarray([1.001, 1.005, 1.010])
    return idata


def test_generate_diagnostics_summary_light_skips_expensive(
    monkeypatch, tmp_path
):
    import log_psplines.plotting.diagnostics as diag_mod

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
    import log_psplines.plotting.diagnostics as diag_mod

    idata = _make_min_idata()
    text = diag_mod.generate_diagnostics_summary(
        idata, str(tmp_path), mode="off"
    )
    assert text == ""
    assert not (tmp_path / "diagnostics_summary.txt").exists()


def test_generate_diagnostics_summary_full_uses_run_all(monkeypatch, tmp_path):
    import log_psplines.plotting.diagnostics as diag_mod

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
