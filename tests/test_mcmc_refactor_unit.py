import numpy as np
import pytest

from log_psplines.datatypes.univar import Periodogram
from log_psplines.mcmc import (
    RunMCMCConfig,
    _maybe_build_welch_overlay,
    _normalize_run_config,
    run_mcmc,
)


class _DummySampler:
    def sample(self, n_samples: int, n_warmup: int, only_vi: bool = False):
        return {
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "only_vi": only_vi,
        }


def test_normalize_run_config_from_legacy_kwargs():
    cfg = _normalize_run_config(
        None,
        {
            "sampler": "nuts",
            "n_samples": 12,
            "n_warmup": 6,
            "n_knots": 7,
            "degree": 4,
            "outdir": "tmp",
        },
    )
    assert isinstance(cfg, RunMCMCConfig)
    assert cfg.n_samples == 12
    assert cfg.n_warmup == 6
    assert cfg.model.n_knots == 7
    assert cfg.model.degree == 4
    assert cfg.diagnostics.outdir == "tmp"


def test_normalize_run_config_rejects_mixed_inputs():
    cfg = RunMCMCConfig()
    with pytest.raises(ValueError):
        _normalize_run_config(cfg, {"n_samples": 10})


def test_run_mcmc_config_path_uses_factory(monkeypatch):
    freq = np.linspace(0.1, 1.0, 8)
    power = np.linspace(1.0, 2.0, 8)
    pdgrm = Periodogram(freqs=freq, power=power)

    captured = {}

    def _fake_build_model(processed_data, model_config):
        captured["model_n_knots"] = model_config.n_knots
        return object()

    def _fake_create_sampler(data, model, config):
        captured["sampler_type"] = config.sampler_type
        captured["scaling_factor"] = config.scaling_factor
        return _DummySampler()

    monkeypatch.setattr(
        "log_psplines.mcmc._build_model_from_data", _fake_build_model
    )
    monkeypatch.setattr(
        "log_psplines.mcmc._create_sampler", _fake_create_sampler
    )

    cfg = RunMCMCConfig(n_samples=5, n_warmup=3)
    result = run_mcmc(pdgrm, config=cfg)

    assert result == {"n_samples": 5, "n_warmup": 3, "only_vi": False}
    assert captured["sampler_type"] == "nuts"
    assert captured["model_n_knots"] == cfg.model.n_knots
    assert captured["scaling_factor"] == 1.0


def test_welch_overlay_guard_returns_none_without_outdir():
    cfg = RunMCMCConfig()
    overlays = _maybe_build_welch_overlay(None, None, cfg)
    assert overlays == (None, None, None)
