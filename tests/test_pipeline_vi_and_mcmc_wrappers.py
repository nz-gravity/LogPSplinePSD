from __future__ import annotations

import numpy as np
import pytest

import log_psplines.inference.vi as vi_module


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
