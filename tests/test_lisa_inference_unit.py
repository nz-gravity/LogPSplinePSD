from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig


def _load_lisa_inference_module():
    lisa_root = (
        Path(__file__).resolve().parents[1] / "docs" / "studies" / "lisa"
    )
    if str(lisa_root) not in sys.path:
        sys.path.insert(0, str(lisa_root))
    import utils.inference as module

    return module


def test_run_lisa_mcmc_defaults_enable_coarse_vi_and_denoised_init(
    monkeypatch,
):
    module = _load_lisa_inference_module()
    captured: dict[str, object] = {}

    def _fake_run_mcmc(*, data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module, "run_mcmc", _fake_run_mcmc)

    n = 32
    ts = MultivariateTimeseries(
        y=np.zeros((n, 3), dtype=np.float64),
        t=np.arange(n, dtype=np.float64),
    )
    freq_true = np.linspace(1e-4, 1e-2, 8, dtype=np.float64)
    s_true = np.repeat(np.eye(3, dtype=np.complex128)[None, :, :], 8, axis=0)

    module.run_lisa_mcmc(
        ts,
        Nb=2,
        coarse_cfg=CoarseGrainConfig(enabled=False),
        freq_true=freq_true,
        S_true=s_true,
        vi=True,
    )

    assert captured["init_from_vi"] is True
    assert captured["auto_coarse_vi"] is True


def test_run_lisa_mcmc_forwards_eta_controls(monkeypatch):
    module = _load_lisa_inference_module()
    captured: dict[str, object] = {}

    def _fake_run_mcmc(*, data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module, "run_mcmc", _fake_run_mcmc)
    monkeypatch.setattr(
        module, "attach_truth_psd_group", lambda idata, **_: idata
    )

    n = 32
    ts = MultivariateTimeseries(
        y=np.zeros((n, 3), dtype=np.float64),
        t=np.arange(n, dtype=np.float64),
    )
    freq_true = np.linspace(1e-4, 1e-2, 8, dtype=np.float64)
    s_true = np.repeat(np.eye(3, dtype=np.complex128)[None, :, :], 8, axis=0)

    module.run_lisa_mcmc(
        ts,
        Nb=7,
        coarse_cfg=CoarseGrainConfig(enabled=False),
        freq_true=freq_true,
        S_true=s_true,
        vi=True,
        eta="auto",
        eta_c=2.5,
    )

    assert captured["eta"] == "auto"
    assert captured["eta_c"] == 2.5
