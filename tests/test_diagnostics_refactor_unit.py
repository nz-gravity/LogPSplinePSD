import numpy as np

from log_psplines.diagnostics import (
    mcmc,
    psd_bands,
    psd_compare,
    run_all,
    vi,
)
from log_psplines.diagnostics.mcmc import _metric_from_attr_or_compute


class _DummyIData:
    def __init__(self, attrs=None):
        self.attrs = attrs or {}


def test_metric_from_attr_or_compute_prefers_attrs():
    idata = _DummyIData(attrs={"rhat": np.array([1.01, 1.02, np.nan])})
    out = _metric_from_attr_or_compute(
        idata,
        attr_key="rhat",
        attr_metrics=lambda vals: {"max": float(np.max(vals))},
        compute_fn=lambda: np.array([99.0]),
        compute_metrics=lambda vals: {"max": float(np.max(vals))},
    )
    assert out["max"] == 1.02


def test_run_all_registry_dispatches_private_entries(monkeypatch):
    calls = []

    def _mk(name):
        def _fn(**kwargs):
            calls.append(name)
            return {"x": 1.0}

        return _fn

    monkeypatch.setattr(run_all.mcmc, "_run", _mk("mcmc"))
    monkeypatch.setattr(run_all.psd_compare, "_run", _mk("psd_compare"))
    monkeypatch.setattr(run_all.psd_bands, "_run", _mk("psd_bands"))
    monkeypatch.setattr(run_all.vi, "_run", _mk("vi"))

    idata = _DummyIData(attrs={})
    idata.posterior_psd = object()

    res = run_all.run_all_diagnostics(
        idata=idata,
        truth=np.array([1.0]),
        signals=np.array([0.0, 1.0]),
        idata_vi={},
    )

    assert set(calls) == {
        "mcmc",
        "psd_compare",
        "psd_bands",
        "vi",
    }
    assert set(res.keys()) == {
        "mcmc",
        "psd_compare",
        "psd_bands",
        "vi",
    }


def test_internal_run_entrypoints_are_private_only():
    assert not hasattr(mcmc, "run")
    assert not hasattr(psd_compare, "run")
    assert not hasattr(psd_bands, "run")
    assert not hasattr(vi, "run")
