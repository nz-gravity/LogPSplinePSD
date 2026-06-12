from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import log_psplines.mcmc as mcmc_module
import log_psplines.pipeline.vi as vi_module
from log_psplines.pipeline.config import PipelineConfig


def _model():
    return None


class _DummyGuide:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("specifier", "expected_name"),
    [
        ("mvn", "mvn"),
        ("lowrank", "lowrank:10"),
        ("lowrank:2", "lowrank:2"),
        ("flow", "flow:1"),
        ("flow:3", "flow:3"),
        ("flowbnaf:2", "flowbnaf:2"),
    ],
)
def test_resolve_guide_string_variants(monkeypatch, specifier, expected_name):
    for name in (
        "AutoMultivariateNormal",
        "AutoLowRankMultivariateNormal",
        "AutoIAFNormal",
        "AutoBNAFNormal",
    ):
        monkeypatch.setattr(vi_module, name, _DummyGuide)

    guide, guide_name = vi_module.resolve_guide(
        specifier,
        _model,
        init_values={"x": np.asarray([1.0])},
    )

    assert isinstance(guide, _DummyGuide)
    assert guide.model is _model
    assert "init_loc_fn" in guide.kwargs
    assert guide_name == expected_name
    if specifier.startswith("lowrank"):
        expected_rank = 2 if specifier.endswith(":2") else 10
        assert guide.kwargs["rank"] == expected_rank
    if specifier.startswith("flow"):
        expected_flows = (
            int(specifier.rsplit(":", 1)[1]) if ":" in specifier else 1
        )
        assert guide.kwargs["num_flows"] == expected_flows


def test_resolve_guide_custom_and_invalid_variants():
    class CustomGuide:
        def __init__(self, model):
            self.model = model

    def guide_factory(model):
        return {"model": model}

    class CallableGuide:
        def __call__(self, model):
            return (model,)

    guide, name = vi_module.resolve_guide(CustomGuide, _model)
    assert isinstance(guide, CustomGuide)
    assert name == "CustomGuide"

    guide, name = vi_module.resolve_guide(guide_factory, _model)
    assert guide == {"model": _model}
    assert name == "guide_factory"

    guide, name = vi_module.resolve_guide(CallableGuide(), _model)
    assert guide == (_model,)
    assert name == "custom_guide"

    with pytest.raises(ValueError, match="Unknown VI guide"):
        vi_module.resolve_guide("unknown", _model)
    with pytest.raises(TypeError, match="Guide must be"):
        vi_module.resolve_guide(123, _model)


def test_run_mcmc_wrapper_kwargs_config_save_and_validation(
    tmp_path, monkeypatch
):
    saved = {}
    captured = {}

    class DummyResult:
        idata = xr.DataTree()

        def save(self, outdir, *, true_psd=None):
            saved["outdir"] = outdir
            saved["true_psd"] = true_psd

    class DummyPipeline:
        data = object()

        def run(self):
            return DummyResult()

    def fake_make_pipeline(data, config):
        captured["data"] = data
        captured["config"] = config
        return DummyPipeline()

    monkeypatch.setattr(mcmc_module, "make_pipeline", fake_make_pipeline)
    monkeypatch.setattr(
        mcmc_module,
        "align_true_psd_to_freq",
        lambda true_psd, data: ("aligned", true_psd, data),
    )

    idata = mcmc_module.run_mcmc(
        "series",
        only_vi=True,
        n_knots=4,
        vi_steps=2,
        outdir=str(tmp_path),
        true_psd=np.ones(3),
    )

    assert isinstance(idata, xr.DataTree)
    assert captured["data"] == "series"
    assert isinstance(captured["config"], PipelineConfig)
    assert saved["outdir"] == str(tmp_path)
    assert saved["true_psd"][0] == "aligned"

    with pytest.raises(ValueError, match="Cannot use both config and kwargs"):
        mcmc_module.run_mcmc("series", PipelineConfig(), only_vi=True)
