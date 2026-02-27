from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "studies"
        / "lisa"
        / "evidence_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lisa_evidence_utils",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_hypothesis_mode_accepts_expected_values():
    mod = _load_module()
    assert mod.parse_hypothesis_mode("full") == "full"
    assert mod.parse_hypothesis_mode("diag") == "diag"
    assert mod.parse_hypothesis_mode("both") == "both"
    assert mod.parse_hypothesis_mode(" BOTH ") == "both"


def test_parse_hypothesis_mode_rejects_invalid_values():
    mod = _load_module()
    with pytest.raises(ValueError, match="Unsupported LISA_HYPOTHESIS_MODE"):
        mod.parse_hypothesis_mode("invalid")


def test_parse_channel_names_validates_shape_and_uniqueness():
    mod = _load_module()
    assert mod.parse_channel_names("X,Y,Z") == ["X", "Y", "Z"]
    assert mod.parse_channel_names("X, Y, Z") == ["X", "Y", "Z"]
    assert mod.parse_channel_names("X,Y") == ["X", "Y", "Z"]
    assert mod.parse_channel_names("X,X,Z") == ["X", "Y", "Z"]


def test_extract_lnz_handles_valid_missing_and_nan():
    mod = _load_module()

    class Dummy:
        def __init__(self, attrs):
            self.attrs = attrs

    lnz, err, valid = mod.extract_lnz(Dummy({"lnz": 1.5, "lnz_err": 0.2}))
    assert (lnz, err, valid) == (1.5, 0.2, True)

    lnz, err, valid = mod.extract_lnz(Dummy({}))
    assert not valid
    assert np.isnan(lnz)
    assert np.isnan(err)

    lnz, err, valid = mod.extract_lnz(Dummy({"lnz": np.nan, "lnz_err": 0.2}))
    assert not valid
    assert np.isnan(lnz)
    assert err == 0.2


def test_combine_diag_lnz_uses_sum_and_quadrature_error():
    mod = _load_module()
    channels = [
        {"name": "X", "lnz": 1.0, "lnz_err": 0.2, "valid": True},
        {"name": "Y", "lnz": 2.0, "lnz_err": 0.3, "valid": True},
        {"name": "Z", "lnz": 3.0, "lnz_err": 0.4, "valid": True},
    ]
    lnz, err, valid = mod.combine_diag_lnz(channels)
    assert valid
    assert lnz == pytest.approx(6.0)
    assert err == pytest.approx(np.sqrt(0.2**2 + 0.3**2 + 0.4**2))


def test_compare_full_vs_diag_propagates_bayes_factor_uncertainty():
    mod = _load_module()
    log_bf, log_bf_err, valid = mod.compare_full_vs_diag(
        5.0,
        0.5,
        2.0,
        0.4,
        full_valid=True,
        diag_valid=True,
    )
    assert valid
    assert log_bf == pytest.approx(3.0)
    assert log_bf_err == pytest.approx(np.sqrt(0.5**2 + 0.4**2))


def test_compare_full_vs_diag_invalid_inputs():
    mod = _load_module()
    log_bf, log_bf_err, valid = mod.compare_full_vs_diag(
        5.0,
        0.5,
        np.nan,
        0.4,
        full_valid=True,
        diag_valid=False,
    )
    assert not valid
    assert np.isnan(log_bf)
    assert np.isnan(log_bf_err)
