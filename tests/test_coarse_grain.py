import numpy as np
import pytest

from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)


def test_coarse_grain_config_accepts_even_nh() -> None:
    cfg = CoarseGrainConfig(enabled=True, Nc=None, Nh=4)
    assert cfg.Nh == 4


def test_compute_binning_structure_even_nh_uses_lower_middle() -> None:
    freqs = np.linspace(0.1, 0.8, 8)
    spec = compute_binning_structure(freqs, Nh=4)

    assert spec.Nc == 2
    assert spec.Nh == 4
    assert np.array_equal(spec.J_start, np.array([0, 4], dtype=np.int32))
    assert np.array_equal(spec.J_mid, np.array([1, 5], dtype=np.int32))
    assert np.allclose(spec.f_coarse, freqs[[1, 5]])


def test_compute_binning_structure_invalid_nh_reports_valid_suggestion() -> (
    None
):
    freqs = np.linspace(0.1, 1.0, 628)
    with pytest.raises(ValueError, match=r"Closest valid Nh=4 \(Nc=157\)"):
        compute_binning_structure(freqs, Nh=7)
